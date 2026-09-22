#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
static double us(Clock::time_point a){return std::chrono::duration<double,std::micro>(Clock::now()-a).count();}
static void check(int ret){if(ret){std::cerr<<"decode failed "<<ret<<"\n";std::exit(3);}}
static void phase(const std::string &s){setenv("SB_CPU_PHASE",s.c_str(),1);}
static void logits(llama_context*c,int nv,const std::string &path){float*p=llama_get_logits_ith(c,-1);for(int i=0;i<nv;i++)if(!std::isfinite(p[i])){std::cerr<<"Nonfinite logits\n";std::exit(4);}if(!path.empty()){std::ofstream f(path,std::ios::binary);f.write((char*)p,nv*sizeof(float));}}
int main(int argc,char**argv){
 if(argc<8){std::cerr<<"profile MODEL N_GPU_LAYERS THREADS REPETITIONS DECODE_STEPS LENGTHS_CSV OUTPUT_PREFIX\n";return 1;}
 const std::string modelpath=argv[1],out=argv[7];int ngl=atoi(argv[2]),threads=atoi(argv[3]),reps=atoi(argv[4]),steps=atoi(argv[5]);std::vector<int>lengths;std::stringstream ss(argv[6]);std::string seg;while(std::getline(ss,seg,','))lengths.push_back(std::stoi(seg));
 ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=ngl;
 auto start=Clock::now();llama_model*m=llama_model_load_from_file(modelpath.c_str(),mp);if(!m)return 2;double load_us=us(start);auto*v=llama_model_get_vocab(m);int nv=llama_vocab_n_tokens(v);
 std::string paragraph="SiliconBadgers is studying how language models execute on a computer. The team compares matrix multiplication, attention, recurrent state updates, normalization, and data movement. Each experiment records the model, numerical format, input length, and software version. A useful accelerator must preserve the model's results while moving data efficiently between memory and arithmetic units. Prefill processes a prompt, whereas decode adds one token to the existing sequence. Engineers use measurements and reference tests to understand these differences.\n";std::string prompt;for(int i=0;i<100;i++)prompt+=paragraph;
 int nt=-llama_tokenize(v,prompt.c_str(),prompt.size(),nullptr,0,true,true);std::vector<llama_token>tokens(nt);check(llama_tokenize(v,prompt.c_str(),prompt.size(),tokens.data(),nt,true,true)<0?-1:0);
 if(nt<*std::max_element(lengths.begin(),lengths.end())+steps)return 5;
 std::ofstream pf(out+"-prompt.txt");pf<<prompt;pf.close();std::ofstream tf(out+"-tokens.json");tf<<"[";for(int i=0;i<nt;i++){if(i)tf<<",";tf<<tokens[i];}tf<<"]\n";
 std::cout<<std::setprecision(10);std::cout<<"{\"kind\":\"metadata\",\"ngl\":"<<ngl<<",\"threads\":"<<threads<<",\"reps\":"<<reps<<",\"decode_steps\":"<<steps<<",\"layers\":"<<llama_model_n_layer(m)<<",\"vocabulary\":"<<nv<<",\"load_us\":"<<load_us<<",\"mode\":\"teacher_forced_text_continuation\",\"n_ubatch\":512,\"flash_attention\":\"on\"}\n"<<std::flush;
 for(int n:lengths){auto cp=llama_context_default_params();cp.n_ctx=n+steps+256;cp.n_batch=n;cp.n_ubatch=512;cp.n_threads=threads;cp.n_threads_batch=threads;cp.n_seq_max=1;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.no_perf=false;cp.op_offload=ngl>0;cp.offload_kqv=ngl>0;llama_context*c=llama_init_from_model(m,cp);if(!c)return 2;
  phase("warmup");check(llama_decode(c,llama_batch_get_one(tokens.data(),n)));llama_synchronize(c);for(int j=0;j<8;j++){check(llama_decode(c,llama_batch_get_one(&tokens[n+j],1)));llama_synchronize(c);}llama_memory_clear(llama_get_memory(c),true);
  for(int rep=0;rep<reps;rep++){
   llama_memory_clear(llama_get_memory(c),true);phase("p"+std::to_string(n)+"_r"+std::to_string(rep)+"_prefill");auto t=Clock::now();check(llama_decode(c,llama_batch_get_one(tokens.data(),n)));llama_synchronize(c);double pre=us(t);phase("idle");logits(c,nv,rep==0?out+"-p"+std::to_string(n)+"-prefill.f32":"");
   std::vector<double>times;for(int j=0;j<steps;j++){phase("p"+std::to_string(n)+"_r"+std::to_string(rep)+"_decode"+std::to_string(j));t=Clock::now();check(llama_decode(c,llama_batch_get_one(&tokens[n+j],1)));llama_synchronize(c);times.push_back(us(t));}phase("idle");logits(c,nv,rep==0?out+"-p"+std::to_string(n)+"-decode.f32":"");
   std::cout<<"{\"kind\":\"measurement\",\"prompt_tokens\":"<<n<<",\"rep\":"<<rep<<",\"prefill_us\":"<<pre<<",\"decode_us\":[";for(size_t j=0;j<times.size();j++){if(j)std::cout<<",";std::cout<<times[j];}std::cout<<"]}\n"<<std::flush;
  }llama_free(c);
 }llama_model_free(m);llama_backend_free();return 0;
}
