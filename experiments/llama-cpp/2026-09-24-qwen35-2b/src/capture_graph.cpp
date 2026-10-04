#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
using json = nlohmann::ordered_json;

struct Capture {
    std::unordered_map<const ggml_tensor *, int> ids;
    json tensors = json::array();
    std::vector<int> scheduled;
    bool enabled = false;

    int visit(const ggml_tensor * t) {
        if (!t) return -1;
        auto it = ids.find(t);
        if (it != ids.end()) return it->second;
        int id = int(tensors.size());
        ids[t] = id;
        tensors.push_back(nullptr);
        json sources = json::array();
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (t->src[i]) sources.push_back({{"slot", i}, {"tensor", visit(t->src[i])}});
        }
        int view = visit(t->view_src);
        json shape = json::array(), strides = json::array(), params = json::array();
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            shape.push_back(t->ne[i]); strides.push_back(t->nb[i]);
        }
        for (size_t i = 0; i < sizeof(t->op_params)/sizeof(int32_t); ++i) params.push_back(t->op_params[i]);
        auto * buffer = t->buffer;
        const ggml_tensor * base = t;
        while (base->view_src) base = base->view_src;
        if (!buffer) buffer = base->buffer;
        bool weight = buffer && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        tensors[id] = {
            {"id", id}, {"name", t->name}, {"op", ggml_op_name(t->op)},
            {"op_desc", ggml_op_desc(t)}, {"dtype", ggml_type_name(t->type)},
            {"shape", shape}, {"strides_bytes", strides}, {"elements", ggml_nelements(t)},
            {"span_bytes", ggml_nbytes(t)},
            {"logical_bytes", ggml_row_size(t->type,t->ne[0])*t->ne[1]*t->ne[2]*t->ne[3]},
            {"contiguous", ggml_is_contiguous(t)}, {"flags", t->flags},
            {"weight_buffer", weight}, {"buffer", buffer ? ggml_backend_buffer_name(buffer) : ""},
            {"sources", sources}, {"view_source", view}, {"view_offset_bytes", t->view_offs},
            {"op_params_i32", params}
        };
        return id;
    }
    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & c = *static_cast<Capture *>(data);
        if (ask && c.enabled) c.scheduled.push_back(c.visit(t));
        return false;
    }
    void begin() { ids.clear(); tensors = json::array(); scheduled.clear(); enabled = true; }
    void save(const std::filesystem::path & path, json metadata) {
        enabled = false;
        if (scheduled.empty()) throw std::runtime_error("No scheduled nodes captured");
        std::ofstream out(path);
        if (!out) throw std::runtime_error("Cannot open graph output");
        out << json({{"schema_version",1}, {"metadata",metadata}, {"scheduled_nodes",scheduled}, {"tensors",tensors}}).dump(2) << '\n';
        if (!out) throw std::runtime_error("Graph write failed");
        std::cout << path << ": " << scheduled.size() << " scheduled nodes, " << tensors.size() << " tensors\n";
    }
};

static llama_token check_logits(llama_context * ctx, const llama_vocab * vocab) {
    float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) throw std::runtime_error("Missing logits");
    int count = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < count; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("Non-finite logits");
    return llama_token(std::max_element(logits, logits + count) - logits);
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: capture-graph MODEL.gguf OUTPUT_DIR [prompt_tokens=128] [context=512]\n";
        return 1;
    }
    try {
        int prompt_tokens = argc >= 4 ? std::stoi(argv[3]) : 128;
        int context = argc >= 5 ? std::stoi(argv[4]) : 512;
        if (prompt_tokens < 2 || context < prompt_tokens+1) throw std::runtime_error("Require prompt >= 2 and context > prompt");
        std::filesystem::path outdir(argv[2]);
        std::filesystem::create_directories(outdir);
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Model load failed");
        Capture capture;
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = prompt_tokens;
        cp.n_ubatch = prompt_tokens;
        cp.n_seq_max = 1;
        cp.n_threads = 6;
        cp.n_threads_batch = 6;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.cb_eval = Capture::callback;
        cp.cb_eval_user_data = &capture;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("Context initialization failed");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string phrase = "The purpose of this workload is to capture the computational graph for accelerator performance analysis. ";
        std::string prompt;
        for (int i=0; i<prompt_tokens; ++i) prompt += phrase;
        int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, false);
        if (n < prompt_tokens) throw std::runtime_error("Not enough prompt tokens");
        std::vector<llama_token> tokens(n);
        if (llama_tokenize(vocab,prompt.c_str(),prompt.size(),tokens.data(),n,true,false) != n) throw std::runtime_error("Tokenization failed");
        tokens.resize(prompt_tokens);
        auto batch = llama_batch_init(prompt_tokens, 0, 1);
        batch.n_tokens = prompt_tokens;
        for (int i=0;i<prompt_tokens;++i) {
            batch.token[i]=tokens[i]; batch.pos[i]=i; batch.n_seq_id[i]=1; batch.seq_id[i][0]=0;
            batch.logits[i]=(i==prompt_tokens-1);
        }
        json meta = {{"model_file",std::filesystem::absolute(argv[1]).string()},
            {"backend","CPU"}, {"capture_method","ggml backend eval callback, ask=true, no tensor data readback"},
            {"tensor_dimension_order","ggml ne[0] is innermost"}, {"flash_attention",false},
            {"sequence_batch",1}, {"context_capacity",llama_n_ctx(ctx)},
            {"prompt_tokens",prompt_tokens}, {"logit_output_tokens",1},
            {"layers",llama_model_n_layer(model)}, {"model_parameters",llama_model_n_params(model)},
            {"kv_dtype","f16"}, {"scope","text trunk only; no vision, MTP, sampling, or tokenizer cost"}};
        capture.begin();
        if (llama_decode(ctx,batch)!=0) throw std::runtime_error("Prefill decode failed");
        llama_synchronize(ctx);
        llama_token next = check_logits(ctx,vocab);
        meta["phase"]="prefill"; meta["input_tokens"]=prompt_tokens; meta["past_tokens"]=0; meta["finite_logits"]=true;
        capture.save(outdir/"prefill.json",meta);
        batch.n_tokens=1; batch.token[0]=next; batch.pos[0]=prompt_tokens; batch.logits[0]=true;
        capture.begin();
        if (llama_decode(ctx,batch)!=0) throw std::runtime_error("Generation decode failed");
        llama_synchronize(ctx);
        check_logits(ctx,vocab);
        meta["phase"]="decode"; meta["input_tokens"]=1; meta["past_tokens"]=prompt_tokens;
        capture.save(outdir/"decode.json",meta);
        json input = {{"prompt_token_ids",tokens},{"decode_token_id",next},{"prompt_seed_text",phrase}};
        std::ofstream(outdir/"workload.json") << input.dump(2) << '\n';
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
