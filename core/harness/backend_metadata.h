#pragma once

#include "llama-ext.h"

#include <cstdint>
#include <ostream>
#include <string>

static const char * sb_device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "meta";
    }
    return "unknown";
}

static void sb_json_string(std::ostream & out, const char * value) {
    out << '"';
    for (const unsigned char ch : std::string(value ? value : "")) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    out << "\\u00" << hex[ch >> 4] << hex[ch & 0x0f];
                } else {
                    out << ch;
                }
        }
    }
    out << '"';
}

// Emit measured model placement after a context exists. requested_ngl is kept separately because it is intent,
// while model_buffers and accelerator_model_bytes describe what llama.cpp actually loaded.
static void sb_write_backend_metadata(std::ostream & out, const llama_context * ctx, int requested_ngl, bool no_host) {
    const llama_memory_breakdown memory = llama_get_memory_breakdown(ctx);
    uint64_t accelerator_model_bytes = 0;
    for (const auto & entry : memory) {
        ggml_backend_buffer_type_t buft = entry.first;
        ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
        // Metal's shared/mapped device buffers are host-accessible too. Exclude only a device's
        // staging host buffer (e.g. CUDA_Host), rather than all host-accessible allocations.
        if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU
                && buft != ggml_backend_dev_host_buffer_type(device)) {
            accelerator_model_bytes += entry.second.model;
        }
    }

    out << ",\"backend\":{\"requested_ngl\":" << requested_ngl
        << ",\"no_host\":" << (no_host ? "true" : "false")
        << ",\"accelerator_model_bytes\":" << accelerator_model_bytes
        << ",\"actual_accelerator\":" << (accelerator_model_bytes > 0 ? "true" : "false")
        << ",\"cpu_fallback\":" << (requested_ngl > 0 && accelerator_model_bytes == 0 ? "true" : "false")
        << ",\"model_buffers\":[";
    bool first = true;
    for (const auto & entry : memory) {
        ggml_backend_buffer_type_t buft = entry.first;
        ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
        if (entry.second.model == 0) {
            continue;
        }
        if (!first) {
            out << ',';
        }
        first = false;
        out << "{\"buffer_type\":";
        sb_json_string(out, ggml_backend_buft_name(buft));
        out << ",\"device\":";
        sb_json_string(out, device ? ggml_backend_dev_name(device) : "unknown");
        out << ",\"device_description\":";
        sb_json_string(out, device ? ggml_backend_dev_description(device) : "unknown");
        out << ",\"backend_name\":";
        sb_json_string(out, device ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)) : "unknown");
        out << ",\"device_type\":";
        sb_json_string(out, device ? sb_device_type_name(ggml_backend_dev_type(device)) : "unknown");
        out << ",\"host\":" << (ggml_backend_buft_is_host(buft) ? "true" : "false")
            << ",\"model_bytes\":" << entry.second.model << '}';
    }
    out << "]}";
}
