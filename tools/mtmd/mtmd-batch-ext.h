#pragma once

// This base keeps the classic llama_batch. Upstream's mtmd writes every decode through llama_batch_ext,
// so this renders that interface over llama_decode and the rest of tools/mtmd can stay identical to upstream.

#include "ggml.h"
#include "llama.h"

#include <array>
#include <memory>
#include <vector>

enum llama_process_type {
    LLAMA_PROCESS_TYPE_ENCODE,
    LLAMA_PROCESS_TYPE_DECODE,
};

struct llama_embd {
    const float * data;
    size_t n_rows;
    size_t n_embd;
};

struct llama_batch_ext {
    int n_pos_per_embd = 1;
    size_t n_embd = 0;
    std::vector<llama_token>  tokens;
    std::vector<float>        embd;
    std::vector<std::array<llama_pos, 4>> positions;
    std::vector<llama_seq_id> seq_ids;
    std::vector<int8_t>       logits;
};

inline llama_batch_ext * llama_batch_ext_init(llama_context * ctx) {
    GGML_UNUSED(ctx);
    return new llama_batch_ext();
}

inline void llama_batch_ext_free(llama_batch_ext * batch) {
    delete batch;
}

inline void llama_batch_ext_clear(llama_batch_ext * batch) {
    batch->tokens.clear();
    batch->embd.clear();
    batch->positions.clear();
    batch->seq_ids.clear();
    batch->logits.clear();
}

inline int32_t llama_batch_ext_add_token(llama_batch_ext * batch, llama_seq_id seq_id, llama_token id) {
    const int32_t index = (int32_t) batch->seq_ids.size();
    batch->tokens.push_back(id);
    batch->positions.push_back({0, 0, 0, 0});
    batch->seq_ids.push_back(seq_id);
    batch->logits.push_back(0);
    return index;
}

inline int32_t llama_batch_ext_add_embd(llama_batch_ext * batch, llama_seq_id seq_id, llama_embd row) {
    const int32_t index = (int32_t) batch->seq_ids.size();
    batch->n_embd = row.n_embd;
    batch->embd.insert(batch->embd.end(), row.data, row.data + row.n_rows * row.n_embd);
    batch->positions.push_back({0, 0, 0, 0});
    batch->seq_ids.push_back(seq_id);
    batch->logits.push_back(0);
    return index;
}

inline bool llama_batch_ext_set_pos(llama_batch_ext * batch, int32_t index, const llama_pos * pos) {
    const int n_pos = batch->tokens.empty() ? batch->n_pos_per_embd : 1;
    for (int section = 0; section < n_pos; section++) {
        batch->positions[index][section] = pos[section];
    }
    return true;
}

inline bool llama_batch_ext_set_output_logits(llama_batch_ext * batch, int32_t index, bool output) {
    batch->logits[index] = output ? 1 : 0;
    return true;
}

inline int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * ext) {
    const int32_t n_tokens = (int32_t) ext->seq_ids.size();
    const bool has_embd = ext->tokens.empty();
    const int n_pos = has_embd ? ext->n_pos_per_embd : 1;

    std::vector<llama_pos> pos((size_t) n_tokens * (size_t) n_pos);
    for (int section = 0; section < n_pos; section++) {
        for (int32_t i = 0; i < n_tokens; i++) {
            pos[(size_t) section * (size_t) n_tokens + (size_t) i] = ext->positions[i][section];
        }
    }

    std::vector<int32_t> n_seq_id((size_t) n_tokens, 1);
    std::vector<llama_seq_id *> seq_ids((size_t) n_tokens + 1, nullptr);
    for (int32_t i = 0; i < n_tokens; i++) {
        seq_ids[i] = &ext->seq_ids[i];
    }

    const llama_batch batch = {
        /*n_tokens =*/ n_tokens,
        /*token    =*/ has_embd ? nullptr : ext->tokens.data(),
        /*embd     =*/ has_embd ? ext->embd.data() : nullptr,
        /*pos      =*/ pos.data(),
        /*n_seq_id =*/ n_seq_id.data(),
        /*seq_id   =*/ seq_ids.data(),
        /*logits   =*/ ext->logits.data(),
    };
    return type == LLAMA_PROCESS_TYPE_ENCODE ? llama_encode(ctx, batch) : llama_decode(ctx, batch);
}

struct llama_batch_ext_deleter {
    void operator()(llama_batch_ext * batch) { llama_batch_ext_free(batch); }
};
using llama_batch_ext_ptr = std::unique_ptr<llama_batch_ext, llama_batch_ext_deleter>;
