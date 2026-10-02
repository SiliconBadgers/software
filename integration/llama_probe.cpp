#include "sim_host.hpp"
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>

// Metadata goes through the simulated controller. Tensor computation stays on the host.
struct Probe {
    SimHost &sim;
    uint32_t sequence=0;
    uint64_t command_cycles=0;
    std::map<std::string, unsigned> operations;
    std::string error;
    bool enabled=false;
    void submit(ggml_tensor *t) {
        uint16_t op=0x8002;
        if(t->op==GGML_OP_MUL_MAT) op=1;
        else if(t->op==GGML_OP_GATED_DELTA_NET || t->op==GGML_OP_SSM_SCAN ||
                t->op==GGML_OP_GATED_LINEAR_ATTN) op=0x8003;
        auto result=sim.submit(op,++sequence, uint32_t(t->ne[0]),uint32_t(t->ne[1]),
                              t->src[0]?uint32_t(t->src[0]->ne[0]):1);
        if(result.status!=0x80 || result.sequence!=sequence)
            throw std::runtime_error("stub must return UNIMPLEMENTED with matching sequence");
        command_cycles+=result.cycles;
        ++operations[ggml_op_name(t->op)];
    }
    static bool observe(ggml_tensor *t, bool ask, void *data) {
        auto &p=*static_cast<Probe *>(data);
        if(ask) {
            // Observe compute operations only. Views/copies are not vector work.
            switch(t->op) {
                case GGML_OP_MUL_MAT: case GGML_OP_UNARY: case GGML_OP_MUL:
                case GGML_OP_ADD: case GGML_OP_RMS_NORM: case GGML_OP_GLU:
                case GGML_OP_SCALE: case GGML_OP_ROPE: case GGML_OP_SSM_CONV:
                case GGML_OP_GATED_DELTA_NET: return true;
                default: return false;
            }
        }
        if(!p.enabled || !p.error.empty()) return p.error.empty();
        try { p.submit(t); return true; }
        catch(const std::exception &e) { p.error=e.what();return false; }
    }
    void json() const {
        std::cout << "\"commands\":"<<sequence<<",\"control_cycles\":"<<command_cycles<<",\"operations\":{";
        bool first=true;
        for(auto &entry:operations) {
            if(!first) std::cout<<',';
            std::cout<<'"'<<entry.first<<"\":"<<entry.second;first=false;
        }
        std::cout<<'}';
    }
};

void graph_probe(SimHost &sim) {
    constexpr int K=16,H=24,N=3;
    ggml_context *ctx=ggml_init({2*1024*1024,nullptr,true});
    if(!ctx) throw std::runtime_error("ggml context allocation failed");
    auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N);
    auto *up=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,H);
    auto *gate=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,H);
    auto *down=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,H,K);
    auto *u=ggml_mul_mat(ctx,up,x);
    auto *g=ggml_mul_mat(ctx,gate,x);
    auto *s=ggml_silu(ctx,g);
    auto *h=ggml_mul(ctx,u,s);
    auto *y=ggml_mul_mat(ctx,down,h);
    auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    auto backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,1);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer) throw std::runtime_error("ggml buffer allocation failed");
    auto values=[](int count,int seed) {
        std::vector<float> a(count);
        for(int i=0;i<count;++i) a[i]=float((i*7+seed)%23-11)/32.0f;
        return a;
    };
    auto xv=values(K*N,1),uv=values(K*H,3),gv=values(K*H,5),dv=values(H*K,9);
    ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);
    ggml_backend_tensor_set(up,uv.data(),0,uv.size()*4);
    ggml_backend_tensor_set(gate,gv.data(),0,gv.size()*4);
    ggml_backend_tensor_set(down,dv.data(),0,dv.size()*4);
    Probe p{sim};
    // This whole graph falls back to CPU after every selected operation rejects offload.
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) p.submit(ggml_graph_node(graph,i));
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("ggml CPU graph failed");
    std::vector<float> actual(K*N);ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
    double worst=0;
    for(int n=0;n<N;++n) {
        double hidden[H]{};
        for(int j=0;j<H;++j) {
            double a=0,b=0;
            for(int k=0;k<K;++k) { a+=uv[j*K+k]*xv[n*K+k];b+=gv[j*K+k]*xv[n*K+k]; }
            hidden[j]=a*b/(1+std::exp(-b));
        }
        for(int k=0;k<K;++k) {
            double ref=0;for(int j=0;j<H;++j) ref+=dv[k*H+j]*hidden[j];
            if(!std::isfinite(actual[n*K+k])) throw std::runtime_error("nonfinite graph result");
            worst=std::max(worst,std::abs(ref-actual[n*K+k]));
        }
    }
    if(worst>1e-5 || p.sequence!=5) throw std::runtime_error("graph reference mismatch");
    std::cout<<"{\"test\":\"ggml_mlp_cpu_fallback\",\"max_abs_error\":"<<worst<<',';
    p.json();std::cout<<",\"pass\":true}\n";
    ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);
}

void model_probe(SimHost &sim,const char *path) {
    llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path,mp),llama_model_free);
    if(!model) throw std::runtime_error("model load failed");
    auto vocab=llama_model_get_vocab(model.get());
    const std::string prompt="The purpose of a hardware accelerator is";
    int count=-llama_tokenize(vocab,prompt.c_str(),int(prompt.size()),nullptr,0,true,true);
    if(count<=0 || count>128) throw std::runtime_error("unexpected token count");
    std::vector<llama_token> tokens(count);
    if(llama_tokenize(vocab,prompt.c_str(),int(prompt.size()),tokens.data(),count,true,true)!=count)
        throw std::runtime_error("tokenization failed");
    Probe p{sim};
    auto run=[&](bool instrument) {
        p.enabled=instrument;
        auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=128;cp.n_ubatch=128;
        cp.n_threads=1;cp.n_threads_batch=1;cp.cb_eval=Probe::observe;cp.cb_eval_user_data=&p;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if(!ctx) throw std::runtime_error("llama context creation failed");
        std::vector<std::vector<float>> output;
        auto batch=llama_batch_get_one(tokens.data(),count);
        llama_token next=0;
        for(int stage=0;stage<2;++stage) {
            if(llama_decode(ctx.get(),batch)!=0 || !p.error.empty())
                throw std::runtime_error("llama evaluation failed: "+p.error);
            auto *logits=llama_get_logits_ith(ctx.get(),-1);
            output.emplace_back(logits,logits+llama_vocab_n_tokens(vocab));
            next=llama_token(std::max_element(output.back().begin(),output.back().end())-output.back().begin());
            batch=llama_batch_get_one(&next,1);
        }
        return output;
    };
    auto ref=run(false);
    auto actual=run(true);
    double worst=0;
    for(size_t s=0;s<ref.size();++s) for(size_t i=0;i<ref[s].size();++i) {
        if(!std::isfinite(ref[s][i]) || !std::isfinite(actual[s][i]))
            throw std::runtime_error("nonfinite llama logits");
        worst=std::max(worst,double(std::abs(ref[s][i]-actual[s][i])));
    }
    if(worst>1e-5 || p.sequence==0 || p.operations["MUL_MAT"]==0)
        throw std::runtime_error("llama logit comparison or command coverage failed");
    std::cout<<"{\"test\":\"llama_cpu_with_control_probe\",\"prompt_tokens\":"<<count
             <<",\"decode_tokens\":1,\"max_abs_logit_error\":"<<worst<<',';
    p.json();std::cout<<",\"pass\":true}\n";
    model.reset();llama_backend_free();
}
int main(int argc,char **argv) {
    try {
        if(argc<2 || argc>3) throw std::runtime_error("usage: llama_probe firmware.bin [model.gguf]");
        SimHost sim(argv[1]);graph_probe(sim);
        if(argc==3) model_probe(sim,argv[2]);
        return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
