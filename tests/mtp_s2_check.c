/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Single-instance native-model proof for the opt-in MTP helpers.
   Usage: mtp_s2_check model.jns mtp.jns tokens.int32 [cache=MiB]
   No target model, weights, sampler, or recurrent state is duplicated. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "janas/llm.h"
#include "llm/model_impl.h"

static unsigned width;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "MTP exact check failed: width=%u line=%d: %s\n", \
            width, __LINE__, #c); exit(1); \
} } while (0)

struct kv_copy {
    unsigned char *data;
    size_t ko, so, kb, sb;
    uint32_t ring_hi;
};

static struct kv_copy kv_alloc(const struct janas_llm_model *m)
{
    uint32_t s = m->mtp_layer.slot;
    struct kv_copy out = {.ko = m->kv_off[s], .so = m->ks_off[s],
        .kb = m->kv_off[s + 1] - m->kv_off[s],
        .sb = (m->ks_off[s + 1] - m->ks_off[s]) * sizeof(float)};
    CHECK(out.kb && out.sb && out.kb <= SIZE_MAX / 2 &&
          out.sb <= SIZE_MAX / 2 - out.kb);
    out.data = malloc(2 * (out.kb + out.sb));
    CHECK(out.data);
    return out;
}
static void kv_save(struct kv_copy *s, const struct janas_llm_model *m)
{
    memcpy(s->data, m->kcache + s->ko, s->kb);
    memcpy(s->data + s->kb, m->vcache + s->ko, s->kb);
    memcpy(s->data + 2 * s->kb, m->kscale + s->so, s->sb);
    memcpy(s->data + 2 * s->kb + s->sb, m->vscale + s->so, s->sb);
    s->ring_hi = m->ring_hi;
}
static void kv_load(const struct kv_copy *s, struct janas_llm_model *m)
{
    memcpy(m->kcache + s->ko, s->data, s->kb);
    memcpy(m->vcache + s->ko, s->data + s->kb, s->kb);
    memcpy(m->kscale + s->so, s->data + 2 * s->kb, s->sb);
    memcpy(m->vscale + s->so, s->data + 2 * s->kb + s->sb, s->sb);
    m->ring_hi = s->ring_hi;
}
static void kv_equal(const struct kv_copy *s, const struct janas_llm_model *m)
{
    CHECK(!memcmp(s->data, m->kcache + s->ko, s->kb));
    CHECK(!memcmp(s->data + s->kb, m->vcache + s->ko, s->kb));
    CHECK(!memcmp(s->data + 2 * s->kb, m->kscale + s->so, s->sb));
    CHECK(!memcmp(s->data + 2 * s->kb + s->sb, m->vscale + s->so, s->sb));
    CHECK(s->ring_hi == m->ring_hi);
}
static uint64_t hash(uint64_t h, const void *p, size_t n)
{
    const unsigned char *v = p;
    for (size_t i = 0; i < n; i++) { h ^= v[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static uint64_t main_kv_hash(const struct janas_llm_model *m)
{
    size_t kb = m->kv_off[m->n_attn], sb = m->ks_off[m->n_attn] * sizeof(float);
    uint64_t h = hash(UINT64_C(14695981039346656037), m->kcache, kb);
    h = hash(h, m->vcache, kb); h = hash(h, m->kscale, sb);
    return hash(h, m->vscale, sb);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: mtp_s2_check model.jns mtp.jns tokens.int32 [cache=MiB]\n");
        return 2;
    }
    uint64_t cache_mib = 32768;
    if (argc > 4 && !strncmp(argv[4], "cache=", 6))
        cache_mib = strtoull(argv[4] + 6, NULL, 10);
    CHECK(cache_mib <= (UINT64_MAX >> 20));
    int32_t tokens[65];
    FILE *f = fopen(argv[3], "rb"); CHECK(f);
    size_t nt = fread(tokens, sizeof(*tokens), 65, f); fclose(f);
    CHECK(nt >= 33);
    int cpus[] = {0, 1, 2, 3, 4, 5, 6, 7};
    struct janas_llm_options o = {.cache_bytes = cache_mib << 20,
        .n_ctx = 128, .compute_cpus = cpus, .n_compute = 8, .n_io = 4,
        .mtp_path = argv[2], .expert_bits = 6,
        .min_quality = JANAS_LLM_QUALITY_HIGH, .no_npu = 1};
    char err[512];
    struct janas_llm_model *m = janas_llm_model_load(argv[1], &o, err, sizeof(err));
    if (!m) { fprintf(stderr, "%s\n", err); return 1; }
    CHECK(janas_llm_mtp_branch_capable(m));
    CHECK(m->blk >= 32 && m->d_model == janas_llm_model_hidden_width(m));
    size_t dm = m->d_model, nbd = dm / JANAS_QK;
    size_t hb = 17 * dm * sizeof(float), qb = nbd * sizeof(*m->xq);
    uint32_t rows = janas_llm_mtp_vocab_rows(m);
    float *target_logits = malloc((size_t)32 * m->n_vocab * sizeof(float));
    float *hidden = malloc((size_t)32 * dm * sizeof(float));
    float *expected_hidden = malloc(hb), *expected_logits = malloc((size_t)rows * sizeof(float));
    float *guard = malloc(hb);
    void *expected_quant = malloc(qb);
    CHECK(target_logits && hidden && expected_hidden && expected_logits && guard && expected_quant);
    CHECK(!janas_llm_model_forward(m, tokens, 32, 0, target_logits, 1));
    memcpy(hidden, janas_llm_model_hidden(m), (size_t)32 * dm * sizeof(float));
    uint64_t untouched = main_kv_hash(m);
    /* Build a shared valid prefix. Every comparison uses subviews with a
       nonzero position; only MTP's small attention slot is restored. */
    const uint32_t prefix = 4;
    CHECK(!janas_llm_mtp_forward(m, tokens + 1, hidden, prefix, 0, NULL, 0));
    struct kv_copy base = kv_alloc(m), reference = kv_alloc(m), tip = kv_alloc(m);
    kv_save(&base, m);
    unsigned prefix_checks = 0;
    uint64_t proof = UINT64_C(14695981039346656037);
    for (width = 1; width <= 17; width++) {
        const int32_t *input = tokens + prefix + 1;
        const float *view = hidden + (size_t)prefix * dm;
        int32_t ref_id, new_id, ref_ids[2], new_ids[2];
        float ref_conf, new_conf, ref_probs[2], new_probs[2];
        kv_load(&base, m);
        CHECK(!janas_llm_mtp_draft(m, input, view, width, prefix, &ref_id, &ref_conf));
        kv_save(&reference, m);
        memcpy(expected_hidden, m->mtp_hid, (size_t)width * dm * sizeof(float));
        memcpy(expected_quant, m->xq + (size_t)(width - 1) * nbd, qb);
        memcpy(expected_logits, m->dh_logits, (size_t)rows * sizeof(float));
        CHECK(janas_llm_mtp_candidates(m, ref_ids, ref_probs, 2) == 2);
        CHECK(ref_ids[0] == ref_id && !memcmp(ref_probs, &ref_conf, sizeof(float)));
        kv_load(&base, m);
        memset(m->mtp_hid, 0xa5, hb); memcpy(guard, m->mtp_hid, hb);
        struct janas_mtp_costs before, after;
        struct janas_llm_runtime_stats rb, ra;
        janas_llm_mtp_costs(m, &before); janas_llm_model_runtime_stats(m, &rb);
        CHECK(!janas_llm_mtp_maintain(m, input, view, width, prefix));
        janas_llm_mtp_costs(m, &after); janas_llm_model_runtime_stats(m, &ra);
        kv_equal(&reference, m);
        CHECK(!memcmp(guard, m->mtp_hid, hb));
        CHECK(before.head_calls == after.head_calls && rb.expert_groups == ra.expert_groups);
        CHECK(janas_llm_mtp_candidates(m, new_ids, new_probs, 2) == -1);
        kv_load(&base, m);
        CHECK(!janas_llm_mtp_draft_tail(m, input, view, width, prefix, &new_id, &new_conf));
        kv_equal(&reference, m);
        CHECK(new_id == ref_id && !memcmp(&new_conf, &ref_conf, sizeof(float)));
        CHECK(!memcmp(expected_hidden + (size_t)(width - 1) * dm,
                      m->mtp_hid + (size_t)(width - 1) * dm, dm * sizeof(float)));
        CHECK(!memcmp(expected_quant, m->xq + (size_t)(width - 1) * nbd, qb));
        CHECK(!memcmp(expected_logits, m->dh_logits, (size_t)rows * sizeof(float)));
        CHECK(!memcmp(guard, m->mtp_hid, (size_t)(width - 1) * dm * sizeof(float)));
        CHECK(janas_llm_mtp_candidates(m, new_ids, new_probs, 2) == 2);
        CHECK(!memcmp(ref_ids, new_ids, sizeof(ref_ids)));
        CHECK(!memcmp(ref_probs, new_probs, sizeof(ref_probs)));
        proof = hash(proof, expected_logits, (size_t)rows * sizeof(float));
        /* A rejected suffix leaves bytes beyond the committed prefix in the
           slot. Replay each possible processed-input prefix from that tip,
           including accepted-drafts+1 == 1. Both paths must leave identical
           stale tails and write identical keys/scales for the live prefix. */
        kv_save(&tip, m);
        for (uint32_t accepted = 0; accepted < width; accepted++) {
            kv_load(&tip, m);
            CHECK(!janas_llm_mtp_forward(m, input, view, accepted + 1, prefix, NULL, 0));
            kv_save(&reference, m);
            kv_load(&tip, m);
            CHECK(!janas_llm_mtp_maintain(m, input, view, accepted + 1, prefix));
            kv_equal(&reference, m); prefix_checks++;
        }
        CHECK(main_kv_hash(m) == untouched);
        CHECK(!memcmp(hidden, janas_llm_model_hidden(m), (size_t)32 * dm * sizeof(float)));
    }
    /* Reject capability without changing any architectural arithmetic. The
       wrappers must retain the original all-row hidden-state semantics. */
    struct janas_arch unsupported = *m->a;
    const struct janas_arch *original = m->a;
    unsupported.name = "qwen3next-helper-fallback"; m->a = &unsupported;
    width = 3; CHECK(!janas_llm_mtp_branch_capable(m));
    kv_load(&base, m);
    int32_t rid, nid; float rc, nc;
    CHECK(!janas_llm_mtp_draft(m, tokens + prefix + 1, hidden + prefix * dm,
                              width, prefix, &rid, &rc));
    kv_save(&reference, m);
    memcpy(expected_hidden, m->mtp_hid, width * dm * sizeof(float));
    memcpy(expected_logits, m->dh_logits, (size_t)rows * sizeof(float));
    kv_load(&base, m); memset(m->mtp_hid, 0xa5, hb);
    CHECK(!janas_llm_mtp_draft_tail(m, tokens + prefix + 1, hidden + prefix * dm,
                                   width, prefix, &nid, &nc));
    kv_equal(&reference, m);
    CHECK(rid == nid && !memcmp(&rc, &nc, sizeof(float)));
    CHECK(!memcmp(expected_hidden, m->mtp_hid, width * dm * sizeof(float)));
    CHECK(!memcmp(expected_logits, m->dh_logits, (size_t)rows * sizeof(float)));
    kv_load(&base, m); memset(m->mtp_hid, 0xa5, hb);
    CHECK(!janas_llm_mtp_maintain(m, tokens + prefix + 1, hidden + prefix * dm,
                                 width, prefix));
    kv_equal(&reference, m);
    CHECK(!memcmp(expected_hidden, m->mtp_hid, width * dm * sizeof(float)));
    m->a = original;
    printf("{\"passed\":true,\"widths\":17,\"prefix_checks\":%u,"
           "\"kv_bytes_per_snapshot\":%zu,\"single_model_instance\":true,"
           "\"draft_logit_hash\":\"%016" PRIx64 "\"}\n",
           prefix_checks, 2 * (base.kb + base.sb), proof);
    free(base.data); free(reference.data); free(tip.data);
    free(target_logits); free(hidden); free(expected_hidden); free(expected_logits);
    free(guard); free(expected_quant); janas_llm_model_free(m);
    return 0;
}
