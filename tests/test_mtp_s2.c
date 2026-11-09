/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Model-free contracts for the private, exact MTP helpers. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "llm/model_impl.h"
#include "llm/vmath.h"

static void candidates(void)
{
    struct janas_llm_model m = {0};
    float logits[] = {3, 3, 1, -INFINITY, 0};
    int32_t map[] = {70000, 15, 140001, 9, 31999};
    m.dh_logits = logits; m.dh_ids = map; m.dh_n = 5;
    m.n_vocab = 151936; m.mtp_head_calls = 1;
    m.dh_scored_sum = 2.0 + janas_exp_det(-2) + janas_exp_det(-3);
    m.dh_scored_rows = 5; m.dh_scored_best = 0;
    int32_t ids[18]; float probs[18];
    for (unsigned i = 0; i < 18; i++) { ids[i] = -333; probs[i] = -999; }
    assert(janas_llm_mtp_candidates(&m, ids + 1, probs + 1, 2) == 2);
    assert(ids[0] == -333 && ids[3] == -333 && probs[0] == -999 && probs[3] == -999);
    assert(ids[1] == 70000 && ids[2] == 15); /* row-order ties, global IDs */
    float conf = (float)(1.0 / m.dh_scored_sum);
    assert(memcmp(&conf, probs + 1, sizeof(conf)) == 0);
    assert(memcmp(probs + 1, probs + 2, sizeof(conf)) == 0);
    assert(janas_llm_mtp_candidates(&m, ids, probs, 16) == 5);
    assert(ids[0] == 70000 && ids[1] == 15 && ids[2] == 140001 &&
           ids[3] == 31999 && ids[4] == 9);
    float p = (float)((double)janas_exp_det(-2) / m.dh_scored_sum);
    assert(memcmp(&p, probs + 2, sizeof(p)) == 0);
    assert(janas_llm_mtp_candidates(NULL, NULL, NULL, 0) == 0);
    assert(janas_llm_mtp_candidates(&m, ids, probs, 17) == -1);
    assert(janas_llm_mtp_candidates(&m, NULL, probs, 2) == -1);
    assert(janas_llm_mtp_candidates(&m, ids, NULL, 2) == -1);
    int32_t saved_ids[18]; float saved_probs[18];
    memcpy(saved_ids, ids, sizeof(ids)); memcpy(saved_probs, probs, sizeof(probs));
#define REFUSES() do { \
    assert(janas_llm_mtp_candidates(&m, ids, probs, 2) == -1); \
    assert(!memcmp(ids, saved_ids, sizeof(ids))); \
    assert(!memcmp(probs, saved_probs, sizeof(probs))); \
} while (0)
    m.dh_scored_rows = 4; REFUSES(); m.dh_scored_rows = 5;
    m.dh_scored_policy = 1; REFUSES(); m.dh_scored_policy = 0;
    m.dh_scored_epoch = 1; REFUSES(); m.dh_scored_epoch = 0;
    m.dh_scored_best = 1; REFUSES(); m.dh_scored_best = 0;
    m.dh_scored_sum = NAN; REFUSES(); m.dh_scored_sum = INFINITY; REFUSES();
    m.dh_scored_sum = 0; REFUSES(); m.dh_scored_sum = 2;
    logits[2] = NAN; REFUSES(); logits[2] = INFINITY; REFUSES(); logits[2] = 1;
    logits[0] = -INFINITY; REFUSES(); logits[0] = 3;
    map[1] = -1; REFUSES(); map[1] = 151936; REFUSES(); map[1] = 15;
    m.mtp_head_calls = 0; REFUSES(); m.mtp_head_calls = 1;
    m.dh_n = 0; REFUSES();
#undef REFUSES
}

static void capability_and_guards(void)
{
    struct janas_llm_model m = {0};
    struct janas_arch arch = {.name = "qwen3next"};
    uint32_t capacity[] = {32, 32};
    uint8_t ring[] = {0, 0}, swa[] = {0, 0};
    m.a = &arch; m.mtp = 1;
    m.n_attn = 1; m.mtp_layer.slot = 1; m.n_ctx = 32; m.blk = 32;
    m.kv_cap = capacity; m.kv_ring = ring; m.kv_swa = swa;
    assert(janas_llm_mtp_branch_capable(&m));
#define REJECT(field, value) do { \
    __typeof__(m.field) saved = m.field; m.field = (value); \
    assert(!janas_llm_mtp_branch_capable(&m)); m.field = saved; \
} while (0)
    REJECT(mtp, 0); REJECT(gpu, (void *)1); REJECT(npu, (void *)1);
    REJECT(asst, (void *)1); REJECT(hc_n, 1); REJECT(miss_skip, 0.01f);
    REJECT(ana_mode, 1); REJECT(ana_frame, (void *)1); REJECT(ana_tok, 1);
    REJECT(trace, 1); REJECT(n_media, 1); REJECT(mtp_layer.rec, 1);
    REJECT(mtp_layer.swa, 1); REJECT(mtp_layer.lat_down, (void *)1);
    REJECT(mtp_layer.slot, 0); REJECT(kv_cap, NULL);
    ring[1] = 1; assert(!janas_llm_mtp_branch_capable(&m)); ring[1] = 0;
    swa[1] = 1; assert(!janas_llm_mtp_branch_capable(&m)); swa[1] = 0;
    capacity[1] = 16; assert(!janas_llm_mtp_branch_capable(&m)); capacity[1] = 32;
    arch.name = "qwen35moe"; assert(!janas_llm_mtp_branch_capable(&m));
    arch.name = NULL; assert(!janas_llm_mtp_branch_capable(&m));
    assert(!janas_llm_mtp_branch_capable(NULL));
#undef REJECT
    int32_t tok = 0, next = -1; float hidden = 0, conf = -1;
    assert(janas_llm_mtp_maintain(NULL, &tok, &hidden, 1, 0) == -1);
    assert(janas_llm_mtp_draft_tail(NULL, &tok, &hidden, 1, 0, &next, &conf) == -1);
    assert(janas_llm_mtp_maintain(&m, &tok, &hidden, 0, 0) == -1);
    assert(janas_llm_mtp_maintain(&m, &tok, &hidden, 33, 0) == -1);
    assert(janas_llm_mtp_maintain(&m, &tok, &hidden, 1, UINT32_MAX) == -1);
    assert(janas_llm_mtp_maintain(&m, &tok, &hidden, 2, 31) == -1);
    assert(janas_llm_mtp_draft_tail(&m, &tok, &hidden, 1, 0, NULL, &conf) == -1);
    assert(janas_llm_mtp_draft_tail(&m, &tok, &hidden, 1, 0, &next, NULL) == -1);
    /* Unsupported empty models take the established failure path before any
       tensor or allocation is touched. Outputs remain the caller's. */
    memset(&m, 0, sizeof(m)); m.blk = 32; m.n_ctx = 32;
    assert(janas_llm_mtp_maintain(&m, &tok, &hidden, 1, 0) == -1);
    assert(janas_llm_mtp_draft_tail(&m, &tok, &hidden, 1, 0, &next, &conf) == -1);
    assert(next == -1 && conf == -1);
}

int main(void)
{
    candidates(); capability_and_guards();
    puts("test_mtp_s2: ok");
    return 0;
}
