// Per-prompt timing harness. It follows profile.cpp's measurement loop step for step (same warm-up, same
// phase names, same output records, same context settings) but takes its prompt from a text file, so the
// prompt length is that text's natural token count.
//
//   sb-profile-prompt MODEL N_GPU_LAYERS THREADS REPETITIONS DECODE_STEPS PROMPT_FILE CONTINUATION_FILE OUTPUT_PREFIX
//
// Prefill processes the whole prompt file. The continuation is teacher-forced: DECODE_STEPS tokens taken from
// CONTINUATION_FILE (repeated if it is shorter) are fed one at a time, whatever the model would have chosen. Every
// prompt therefore gets the same continuation, and CPU and Metal see identical inputs.
//
// Output records match profile.cpp: one {"kind":"metadata"} line, then one {"kind":"measurement"} line per
// repetition, on stdout. Logits of repetition 0 are saved after prefill and after the last decode step.
#include "llama.h"
#include "ggml-backend.h"
#include "backend_metadata.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double elapsed_us(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

static void check(int ret) {
    if (ret) {
        std::cerr << "decode failed " << ret << "\n";
        std::exit(3);
    }
}

// The CPU trace patch (when present) reads this to label each graph it records.
static void set_phase(const std::string & name) {
    setenv("SB_CPU_PHASE", name.c_str(), 1);
}

static std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(6);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, false);
    std::vector<llama_token> tokens(count);
    if (count <= 0 || llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), count, add_special, false) != count) {
        std::cerr << "tokenization failed\n";
        std::exit(7);
    }
    return tokens;
}

// Verifies the logits are finite, then (if a path is given) writes them once.
static void save_logits(llama_context * ctx, int vocab_size, const std::string & path) {
    float * logits = llama_get_logits_ith(ctx, -1);
    for (int i = 0; i < vocab_size; i++) {
        if (!std::isfinite(logits[i])) {
            std::cerr << "Nonfinite logits\n";
            std::exit(4);
        }
    }
    if (!path.empty()) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(logits), vocab_size * sizeof(float));
    }
}

int main(int argc, char ** argv) {
    if (argc < 9) {
        std::cerr << "profile-prompt MODEL N_GPU_LAYERS THREADS REPETITIONS DECODE_STEPS PROMPT_FILE CONTINUATION_FILE OUTPUT_PREFIX [NO_HOST]\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const int n_gpu_layers = std::atoi(argv[2]);
    const int threads = std::atoi(argv[3]);
    const int repetitions = std::atoi(argv[4]);
    const int steps = std::atoi(argv[5]);
    const std::string prompt_file = argv[6];
    const std::string continuation_file = argv[7];
    const std::string out = argv[8];
    const bool no_host = argc > 9 && std::atoi(argv[9]) != 0;
    const int warmup_steps = 8;

    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.no_host = no_host;
    auto load_start = Clock::now();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        return 2;
    }
    const double load_us = elapsed_us(load_start);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int vocab_size = llama_vocab_n_tokens(vocab);

    const std::vector<llama_token> prompt = tokenize(vocab, read_file(prompt_file), true);
    const std::vector<llama_token> source = tokenize(vocab, read_file(continuation_file), false);
    std::vector<llama_token> continuation;
    for (int i = 0; i < steps + warmup_steps; i++) {
        continuation.push_back(source[i % source.size()]);
    }
    const int n = static_cast<int>(prompt.size());

    std::ofstream tokens_out(out + "-tokens.json");
    tokens_out << "{\"prompt\":[";
    for (int i = 0; i < n; i++) {
        tokens_out << (i ? "," : "") << prompt[i];
    }
    tokens_out << "],\"continuation\":[";
    for (int i = 0; i < steps; i++) {
        tokens_out << (i ? "," : "") << continuation[i];
    }
    tokens_out << "]}\n";
    tokens_out.close();

    auto context_params = llama_context_default_params();
    context_params.n_ctx = n + steps + 256;
    context_params.n_batch = n;
    context_params.n_ubatch = 512;
    context_params.n_threads = threads;
    context_params.n_threads_batch = threads;
    context_params.n_seq_max = 1;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.no_perf = false;
    context_params.op_offload = n_gpu_layers > 0;
    context_params.offload_kqv = n_gpu_layers > 0;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!ctx) {
        return 2;
    }

    std::cout << std::setprecision(10);
    std::cout << "{\"kind\":\"metadata\",\"ngl\":" << n_gpu_layers << ",\"threads\":" << threads
              << ",\"reps\":" << repetitions << ",\"decode_steps\":" << steps
              << ",\"layers\":" << llama_model_n_layer(model) << ",\"vocabulary\":" << vocab_size
              << ",\"load_us\":" << load_us << ",\"mode\":\"teacher_forced_shared_continuation\""
              << ",\"n_ubatch\":512,\"flash_attention\":\"on\",\"prompt_tokens\":" << n
              << ",\"continuation_tokens\":" << steps;
    sb_write_backend_metadata(std::cout, ctx, n_gpu_layers, no_host);
    std::cout << "}\n" << std::flush;

    // Warm-up: one full prefill and a few decode steps, then start each repetition from an empty cache.
    set_phase("warmup");
    check(llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), n)));
    llama_synchronize(ctx);
    for (int j = 0; j < warmup_steps; j++) {
        check(llama_decode(ctx, llama_batch_get_one(&continuation[j], 1)));
        llama_synchronize(ctx);
    }
    llama_memory_clear(llama_get_memory(ctx), true);

    const std::string tag = "p" + std::to_string(n);
    for (int rep = 0; rep < repetitions; rep++) {
        const std::string run = tag + "_r" + std::to_string(rep);
        llama_memory_clear(llama_get_memory(ctx), true);

        set_phase(run + "_prefill");
        auto start = Clock::now();
        check(llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), n)));
        llama_synchronize(ctx);
        const double prefill_us = elapsed_us(start);
        set_phase("idle");
        save_logits(ctx, vocab_size, rep == 0 ? out + "-" + tag + "-prefill.f32" : "");

        std::vector<double> decode_us;
        for (int j = 0; j < steps; j++) {
            set_phase(run + "_decode" + std::to_string(j));
            start = Clock::now();
            check(llama_decode(ctx, llama_batch_get_one(&continuation[j], 1)));
            llama_synchronize(ctx);
            decode_us.push_back(elapsed_us(start));
        }
        set_phase("idle");
        save_logits(ctx, vocab_size, rep == 0 ? out + "-" + tag + "-decode.f32" : "");

        std::cout << "{\"kind\":\"measurement\",\"prompt_tokens\":" << n << ",\"rep\":" << rep
                  << ",\"prefill_us\":" << prefill_us << ",\"decode_us\":[";
        for (size_t j = 0; j < decode_us.size(); j++) {
            std::cout << (j ? "," : "") << decode_us[j];
        }
        std::cout << "]}\n" << std::flush;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
