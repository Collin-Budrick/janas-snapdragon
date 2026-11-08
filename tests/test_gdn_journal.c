/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* Derived GDN journal lifecycle, budget and unchanged snapshot payloads.
   No model file, accelerator or inference is needed. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "llm/deltanet.h"
#include "llm/model_impl.h"

static int failures;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#if defined(__aarch64__) || defined(_M_ARM64)
struct mode_job { uint64_t saved; int restore; };
static void worker_mode(void *arg, int tid, int n_threads)
{
    (void)n_threads;
    struct mode_job *a = arg;
    if (tid != 1) return;
    uint64_t mode;
    if (!a->restore) {
        __asm__ volatile("mrs %0, fpcr" : "=r"(mode));
        a->saved = mode;
        mode |= UINT64_C(1) << 22;
    } else mode = a->saved;
    __asm__ volatile("msr fpcr, %0" : : "r"(mode));
}
#endif

static void publish(struct janas_llm_model *m, uint32_t pos, uint32_t n)
{
    m->base_pos = pos;
    m->log_n = n;
    m->gdn_written_mask = m->n_rec == 64 ? UINT64_MAX
                                        : (UINT64_C(1) << m->n_rec) - 1;
    janas_m_gdn_finish(m);
}

static int same_serialized(struct janas_llm_model *m,
                            const struct janas_llm_state *a,
                            const struct janas_llm_state *b)
{
    FILE *fa = tmpfile(), *fb = tmpfile();
    int same = fa && fb && !janas_llm_state_write(m, a, fa) &&
               !janas_llm_state_write(m, b, fb);
    if (same) {
        rewind(fa), rewind(fb);
        int ca, cb;
        do {
            ca = fgetc(fa), cb = fgetc(fb);
            if (ca != cb) { same = 0; break; }
        } while (ca != EOF);
    }
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    return same;
}

static void snapshots(struct janas_pool *pool)
{
    const struct janas_arch arch = {.rec = JANAS_REC_NEXT};
    struct janas_llm_model m = {.a = &arch, .compute = pool, .n_rec = 1,
        .n_vh = 1, .ds = 32, .ssm_per = 1024, .conv_ch = 96, .d_conv = 2,
        .n_ctx = 64, .base_pos = 1, .log_n = 3};
    m.ssm_buf = calloc(2048, sizeof(uint16_t));
    m.conv_buf = calloc(192, sizeof(float));
    m.log_mixed = calloc(JANAS_LLM_MAX_BLOCK * 96, sizeof(float));
    m.log_beta = calloc(JANAS_LLM_MAX_BLOCK, sizeof(float));
    m.log_decay = calloc(JANAS_LLM_MAX_BLOCK, sizeof(float));
    CHECK(m.ssm_buf && m.conv_buf && m.log_mixed && m.log_beta && m.log_decay);
    for (size_t i = 0; i < 2048; i++) m.ssm_buf[i] = (uint16_t)i;
    for (size_t i = 0; i < 192; i++) m.conv_buf[i] = (float)i;
    struct janas_llm_state *a = janas_llm_model_state_save(&m, 2);
    CHECK(a);
    janas_m_gdn_prepare(&m, 2, 9);
    struct janas_llm_state *b = janas_llm_model_state_save(&m, 2);
    CHECK(b && a && janas_llm_state_bytes(a) == janas_llm_state_bytes(b));
    if (a && b) CHECK(same_serialized(&m, a, b));
    m.gdn_journal_valid = 1;
    if (a) CHECK(janas_llm_model_state_load(&m, a) == 0);
    CHECK(!m.gdn_journal_valid && !m.gdn_recover_active && !m.gdn_chain_active);
    janas_m_gdn_prepare(&m, 2, 5);
    CHECK(!m.gdn_recover_active); /* restore uses the original input log */
    janas_llm_state_free(a), janas_llm_state_free(b);
    janas_m_gdn_free(&m);
    free(m.ssm_buf), free(m.conv_buf), free(m.log_mixed);
    free(m.log_beta), free(m.log_decay);
}

struct fixture {
    struct janas_llm_model m;
    struct layer ly;
    struct janas_jns_tensor tensor[4];
    float wt[4 * 128];
    uint32_t corrupt_from;
};

static void fixture_init(struct fixture *f, struct janas_pool *pool)
{
    static const struct janas_arch arch = {.rec = JANAS_REC_NEXT};
    memset(f, 0, sizeof(*f));
    f->corrupt_from = UINT32_MAX;
    struct janas_llm_model *m = &f->m;
    m->a = &arch, m->compute = pool;
    m->n_rec = 1, m->n_vh = 2, m->n_kh = 1, m->ds = 32;
    m->ssm_per = 2 * 32 * 32, m->conv_ch = 128, m->d_conv = 4;
    m->d_inner = 64, m->qkvz_dim = 192, m->eps = 1e-5f;
    m->n_ctx = 128;
    m->ssm_buf = calloc(2 * m->ssm_per, sizeof(uint16_t));
    m->conv_buf = calloc(2 * 3 * 128, sizeof(float));
    m->log_mixed = calloc(JANAS_LLM_MAX_BLOCK * 128, sizeof(float));
    m->log_beta = calloc(JANAS_LLM_MAX_BLOCK * 2, sizeof(float));
    m->log_decay = calloc(JANAS_LLM_MAX_BLOCK * 2, sizeof(float));
    m->mixed = calloc(2 * JANAS_LLM_MAX_BLOCK * 128, sizeof(float));
    m->beta = calloc(2 * JANAS_LLM_MAX_BLOCK * 2, sizeof(float));
    m->decay = calloc(2 * JANAS_LLM_MAX_BLOCK * 2, sizeof(float));
    m->conv_out = calloc(2 * JANAS_LLM_MAX_BLOCK * 128, sizeof(float));
    m->att = calloc(JANAS_LLM_MAX_BLOCK * 64, sizeof(float));
    m->big = calloc(JANAS_LLM_MAX_BLOCK * 192, sizeof(float));
    m->ba = calloc(JANAS_LLM_MAX_BLOCK * 4, sizeof(float));
    m->resident = calloc(36, sizeof(float));
    CHECK(m->ssm_buf && m->conv_buf && m->log_mixed && m->log_beta &&
          m->log_decay && m->mixed && m->beta && m->decay && m->conv_out &&
          m->att && m->big && m->ba && m->resident);
    float *res = (float *)m->resident;
    res[0] = 0.1f, res[1] = 0.2f, res[2] = -0.1f, res[3] = -0.2f;
    for (size_t i = 4; i < 36; i++) res[i] = 0.9f + (float)i * 0.001f;
    f->tensor[0].offset = 0;
    f->tensor[1].offset = 2 * sizeof(float);
    f->tensor[2].offset = 4 * sizeof(float);
    f->ly.dt_bias = &f->tensor[0], f->ly.ssm_a = &f->tensor[1];
    f->ly.ssm_norm = &f->tensor[2], f->ly.ssm_qkv = &f->tensor[3];
    f->ly.conv_t = f->wt;
    for (size_t i = 0; i < 4 * 128; i++)
        f->wt[i] = 0.05f + (float)(i % 11) * 0.003f;
}

static void fixture_free(struct fixture *f)
{
    struct janas_llm_model *m = &f->m;
    janas_m_gdn_free(m);
    free(m->ssm_buf), free(m->conv_buf), free(m->log_mixed);
    free(m->log_beta), free(m->log_decay), free(m->mixed), free(m->beta);
    free(m->decay), free(m->conv_out), free(m->att), free(m->big);
    free(m->ba), free(m->resident);
}

static void fixture_run(struct fixture *f, uint32_t pos, uint32_t n,
                         int chain, int journal, int key_only)
{
    struct janas_llm_model *m = &f->m;
    setenv("JANAS_GDN_CHAIN", chain ? "1" : "0", 1);
    setenv("JANAS_GDN_JOURNAL", journal ? "1" : "0", 1);
    setenv("JANAS_GDN_K_ONLY", key_only ? "1" : "0", 1);
    if (!pos) janas_m_gdn_invalidate(m);
    janas_m_gdn_prepare(m, pos, n);
    if (!pos) {
        m->src = 0, m->replay = 0;
        memset(m->ssm_buf, 0, m->ssm_per * sizeof(uint16_t));
        memset(m->conv_buf, 0, 3 * 128 * sizeof(float));
    } else if (pos == m->base_pos + m->log_n && m->log_n) {
        m->src = 1 - m->base, m->replay = 0;
    } else {
        m->src = m->base, m->replay = pos - m->base_pos;
    }
    m->base_pos = UINT32_MAX;
    for (uint32_t t = 0; t < n; t++) {
        float corrupt = pos + t >= f->corrupt_from ? 0.375f : 0.0f;
        for (size_t i = 0; i < 192; i++)
            m->big[(size_t)t * 192 + i] =
                (float)((int)((pos + t) * 3 + i * 7) % 31 - 15) * 0.03125f + corrupt;
        for (size_t i = 0; i < 4; i++)
            m->ba[(size_t)t * 4 + i] = (float)((pos + t + i) % 13) * 0.0625f + corrupt;
    }
    if (key_only && m->gdn_konly_active)
        for (uint32_t t = 0; t < m->replay; t++)
            for (size_t i = 0; i < 128; i++)
                if (i < 32 || i >= 64)
                    m->conv_out[(size_t)t * 128 + i] = NAN;
    janas_m_rec_core(m, &f->ly, n, now());
    m->base = m->src, m->base_pos = pos, m->log_n = n;
    janas_m_gdn_finish(m);
}

static void same_fixture(const struct fixture *a, const struct fixture *b,
                          uint32_t n)
{
    CHECK(!memcmp(a->m.ssm_buf, b->m.ssm_buf, 2 * 2048 * sizeof(uint16_t)));
    CHECK(!memcmp(a->m.conv_buf, b->m.conv_buf, 2 * 3 * 128 * sizeof(float)));
    CHECK(!memcmp(a->m.att, b->m.att, (size_t)n * 64 * sizeof(float)));
    CHECK(!memcmp(a->m.log_mixed, b->m.log_mixed, (size_t)n * 128 * sizeof(float)));
    CHECK(!memcmp(a->m.log_beta, b->m.log_beta, (size_t)n * 2 * sizeof(float)));
    CHECK(!memcmp(a->m.log_decay, b->m.log_decay, (size_t)n * 2 * sizeof(float)));
    CHECK(a->m.base == b->m.base && a->m.base_pos == b->m.base_pos &&
          a->m.log_n == b->m.log_n);
}

static void integrated_chains(struct janas_pool *pool)
{
    struct fixture a, b;
    fixture_init(&a, pool), fixture_init(&b, pool);
    for (int journal = 0; journal <= 1; journal++)
        for (int key_only = 0; key_only <= journal; key_only++)
            for (uint32_t prefix = 1; prefix <= 17; prefix++) {
                fixture_run(&a, 0, 17, 0, 0, 0);
                fixture_run(&b, 0, 17, 1, journal, key_only);
                same_fixture(&a, &b, 17);
                float *buffer = b.m.gdn_delta;
                fixture_run(&a, prefix, 9, 0, 0, 0);
                fixture_run(&b, prefix, 9, 1, journal, key_only);
                same_fixture(&a, &b, 9);
                fixture_run(&a, prefix + 1, 5, 0, 0, 0);
                fixture_run(&b, prefix + 1, 5, 1, journal, key_only);
                same_fixture(&a, &b, 5);
                if (journal && janas_gdn_chain_eligible()) {
                    CHECK(b.m.gdn_journal_valid && b.m.gdn_delta == buffer);
                    CHECK(b.m.gdn_delta_bytes <= (size_t)16 * 1024 * 1024);
                }
            }
    fixture_run(&b, 0, 17, 1, 1, 1);
    b.m.rec_cont = 1;
    fixture_run(&b, 17, 5, 1, 1, 1);
    CHECK(!b.m.gdn_journal_valid); /* split device continuation cannot publish */
    b.m.rec_cont = 0;
    fixture_free(&a), fixture_free(&b);
    puts("journal integration: chain/full-conv/K-only, every prefix, repeated rejection exact");
}

/* Keep a genuinely uninterrupted scalar recurrence alongside the production
   causal convolution. This checks each half-state word and gated output at
   every token, independently of the optimized chain and its journal. */
static void scalar_step(const struct fixture *f, uint16_t state[2048],
                         float out[64])
{
    const struct janas_llm_model *m = &f->m;
    const float *co = m->conv_out, *weights = (const float *)m->resident + 4;
    for (size_t h = 0; h < 2; h++) {
        janas_gdn_step_h_ref(state + h * 32 * 32, co, co + 32,
                             co + 64 + h * 32, m->decay[h], m->beta[h],
                             1.0f / sqrtf(32.0f), out + h * 32, 32, 32);
        rms_norm(out + h * 32, out + h * 32, weights, 32, m->eps);
        const float *z = m->big + 128 + h * 32;
        for (size_t i = 0; i < 32; i++)
            out[h * 32 + i] *= z[i] / (1.0f + janas_expf(-z[i]));
    }
    CHECK(!memcmp(state, m->ssm_buf + (size_t)(1 - m->src) * 2048,
                  2048 * sizeof(uint16_t)));
    CHECK(!memcmp(out, m->att, 64 * sizeof(float)));
}

static void restored_prefixes(struct janas_pool *pool)
{
    enum { NEW = 9 };
    struct fixture reference, restored;
    fixture_init(&reference, pool), fixture_init(&restored, pool);
    for (uint32_t prefix = 1; prefix < 17; prefix++) {
        uint16_t scalar[2048] = {0}, prefix_state[2048];
        float prefix_conv[3 * 128], expected[NEW * 64];
        float mixed[NEW * 128], beta[NEW * 2], decay[NEW * 2];
        for (uint32_t pos = 0; pos < prefix + NEW; pos++) {
            float output[64];
            fixture_run(&reference, pos, 1, 0, 0, 0);
            scalar_step(&reference, scalar, output);
            if (pos + 1 == prefix) {
                memcpy(prefix_state, scalar, sizeof scalar);
                memcpy(prefix_conv,
                       reference.m.conv_buf + (size_t)(1 - reference.m.src) * 3 * 128,
                       sizeof prefix_conv);
            }
            if (pos >= prefix) {
                size_t t = pos - prefix;
                memcpy(expected + t * 64, output, sizeof output);
                memcpy(mixed + t * 128, reference.m.log_mixed, 128 * sizeof(float));
                memcpy(beta + t * 2, reference.m.log_beta, 2 * sizeof(float));
                memcpy(decay + t * 2, reference.m.log_decay, 2 * sizeof(float));
            }
        }
        /* The unaccepted suffix deliberately differs from the canonical
           continuation, so retaining a rejected tip cannot pass this test. */
        restored.corrupt_from = prefix;
        fixture_run(&restored, 0, 17, 1, 1, 1);
        restored.corrupt_from = UINT32_MAX;
        if (janas_gdn_chain_eligible()) CHECK(restored.m.gdn_journal_valid);
        struct janas_llm_state *saved =
            janas_llm_model_state_save(&restored.m, prefix);
        CHECK(saved);
        if (!saved) continue;
        FILE *file = tmpfile();
        CHECK(file);
        if (!file) { janas_llm_state_free(saved); continue; }
        CHECK(!janas_llm_state_write(&restored.m, saved, file));
        rewind(file);
        struct janas_llm_state *loaded = janas_llm_state_read(&restored.m, file);
        fclose(file);
        CHECK(loaded);
        janas_llm_state_free(saved);
        if (!loaded) continue;
        float *buffer = restored.m.gdn_delta;
        size_t bytes = restored.m.gdn_delta_bytes;
        uint64_t recovered = restored.m.gdn_journal_replay_tokens;
        uint64_t recomputed = restored.m.gdn_scalar_replay_tokens;
        /* Force the load to restore both buffers and every projected-input
           log. Its derived journal deliberately remains allocated/stale. */
        memset(restored.m.ssm_buf, 0xa5, 2 * 2048 * sizeof(uint16_t));
        memset(restored.m.conv_buf, 0x5a, 2 * 3 * 128 * sizeof(float));
        memset(restored.m.log_mixed, 0x5a, JANAS_LLM_MAX_BLOCK * 128 * sizeof(float));
        memset(restored.m.log_beta, 0x5a, JANAS_LLM_MAX_BLOCK * 2 * sizeof(float));
        memset(restored.m.log_decay, 0x5a, JANAS_LLM_MAX_BLOCK * 2 * sizeof(float));
        restored.m.base_pos = UINT32_MAX, restored.m.log_n = 0;
        CHECK(!janas_llm_model_state_load(&restored.m, loaded));
        janas_llm_state_free(loaded);
        CHECK(!restored.m.gdn_journal_valid && !restored.m.gdn_recover_active &&
              !restored.m.gdn_chain_active && !restored.m.gdn_konly_active);
        fixture_run(&restored, prefix, NEW, 1, 1, 1);
        CHECK(restored.m.gdn_journal_replay_tokens == recovered);
        CHECK(restored.m.gdn_scalar_replay_tokens == recomputed + prefix);
        CHECK(!memcmp(restored.m.ssm_buf + (size_t)restored.m.base * 2048,
                      prefix_state, sizeof prefix_state));
        CHECK(!memcmp(restored.m.ssm_buf + (size_t)(1 - restored.m.base) * 2048,
                      scalar, sizeof scalar));
        CHECK(!memcmp(restored.m.conv_buf + (size_t)restored.m.base * 3 * 128,
                      prefix_conv, sizeof prefix_conv));
        CHECK(!memcmp(restored.m.conv_buf + (size_t)(1 - restored.m.base) * 3 * 128,
                      reference.m.conv_buf + (size_t)(1 - reference.m.src) * 3 * 128,
                      sizeof prefix_conv));
        CHECK(!memcmp(restored.m.att, expected, sizeof expected));
        CHECK(!memcmp(restored.m.log_mixed, mixed, sizeof mixed));
        CHECK(!memcmp(restored.m.log_beta, beta, sizeof beta));
        CHECK(!memcmp(restored.m.log_decay, decay, sizeof decay));
        if (janas_gdn_chain_eligible()) {
            CHECK(restored.m.gdn_journal_valid); /* only the fresh pass publishes */
            CHECK(restored.m.gdn_delta == buffer && restored.m.gdn_delta_bytes == bytes);
        }
    }
    fixture_free(&reference), fixture_free(&restored);
    puts("journal snapshot: all partial prefixes, serialized load, scalar continuation/state/output exact");
}

int main(void)
{
    alarm(120);
    setenv("JANAS_GDN_CHAIN", "1", 1);
    setenv("JANAS_GDN_JOURNAL", "1", 1);
    setenv("JANAS_GDN_K_ONLY", "1", 1);
    struct janas_pool *pool = janas_pool_create(2, NULL);
    CHECK(pool);
    if (!pool) return 1;
    const struct janas_arch arch = {.rec = JANAS_REC_NEXT};
    struct janas_llm_model m = {.a = &arch, .compute = pool, .n_rec = 36,
        .n_vh = 32, .ds = 128, .base_pos = 10, .log_n = 5};
    janas_m_gdn_prepare(&m, 13, 17);
    if (janas_gdn_chain_eligible()) {
        CHECK(m.gdn_chain_active && m.gdn_journal_active);
        CHECK(!m.gdn_recover_active && !m.gdn_journal_valid);
        CHECK(m.gdn_delta_bytes == (size_t)17 * 36 * 32 * 128 * sizeof(float));
        CHECK(m.gdn_delta_bytes <= (size_t)16 * 1024 * 1024);
        publish(&m, 13, 17);
        CHECK(m.gdn_journal_valid);
        float *same = m.gdn_delta;
        janas_m_gdn_prepare(&m, 18, 9);
        CHECK(m.gdn_recover_active && m.gdn_konly_active);
        CHECK(m.gdn_delta == same); /* one allocation, old/new alias */
        CHECK(m.gdn_delta_capacity == 17);
        CHECK(!m.gdn_journal_valid); /* unpublished until all layers finish */
        m.base_pos = 18, m.log_n = 9;
        m.gdn_written_mask = 1;
        janas_m_gdn_finish(&m);
        CHECK(!m.gdn_journal_valid); /* partial/failed forward cannot publish */
        janas_m_gdn_prepare(&m, 18, 9);
        publish(&m, 18, 9);
        CHECK(m.gdn_journal_valid);
        m.base ^= 1;
        janas_m_gdn_prepare(&m, 20, 5);
        CHECK(!m.gdn_recover_active); /* buffer identity is part of the key */
        publish(&m, 20, 5);
        janas_m_gdn_prepare(&m, 25, 5);
        CHECK(!m.gdn_recover_active); /* full acceptance starts from the tip */
        publish(&m, 25, 5);
#if defined(__aarch64__) || defined(_M_ARM64)
        struct mode_job mode = {0};
        janas_pool_run(pool, worker_mode, &mode);
        janas_m_gdn_prepare(&m, 27, 5);
        CHECK(!m.gdn_chain_active && !m.gdn_journal_valid);
        mode.restore = 1;
        janas_pool_run(pool, worker_mode, &mode);
        janas_m_gdn_prepare(&m, 27, 5);
        CHECK(m.gdn_chain_active && !m.gdn_recover_active);
        publish(&m, 27, 5);
        uint64_t saved = janas_gdn_fp_mode();
        uint64_t altered = saved | (UINT64_C(1) << 19);
        __asm__ volatile("msr fpcr, %0" : : "r"(altered));
        janas_m_gdn_prepare(&m, 29, 5);
        CHECK(!m.gdn_chain_active && !m.gdn_journal_valid);
        __asm__ volatile("msr fpcr, %0" : : "r"(saved));
#endif
        m.gtok_where = JANAS_GT_DEV;
        janas_m_gdn_prepare(&m, 27, 5);
        CHECK(!m.gdn_chain_active && !m.gdn_journal_valid);
        m.gtok_where = JANAS_GT_SAME;
        m.n_rec = 64, m.n_vh = 32, m.ds = 256;
        janas_m_gdn_prepare(&m, 27, 17);
        CHECK(m.gdn_chain_active && !m.gdn_journal_active);
        CHECK(m.gdn_delta_bytes <= (size_t)16 * 1024 * 1024);
        m.n_rec = UINT32_MAX, m.n_vh = UINT32_MAX;
        janas_m_gdn_prepare(&m, 27, 17);
        CHECK(!m.gdn_journal_active); /* overflow-sized geometry is rejected */
        unsetenv("JANAS_GDN_CHAIN");
        janas_m_gdn_prepare(&m, 27, 5);
        CHECK(!m.gdn_chain_active && !m.gdn_journal_valid);
        setenv("JANAS_GDN_CHAIN", "1", 1);
    } else {
        CHECK(!m.gdn_chain_active && !m.gdn_delta);
        puts("journal production path unavailable on this architecture/FPCR");
    }
    janas_m_gdn_free(&m);
    CHECK(!m.gdn_delta && !m.gdn_delta_bytes && !m.gdn_delta_capacity);
    snapshots(pool);
    integrated_chains(pool);
    restored_prefixes(pool);
    janas_pool_destroy(pool);
    unsetenv("JANAS_GDN_CHAIN"), unsetenv("JANAS_GDN_JOURNAL");
    unsetenv("JANAS_GDN_K_ONLY");
    if (failures) return 1;
    puts("test_gdn_journal: budget/lifecycle/restore/payload ok");
    return 0;
}
