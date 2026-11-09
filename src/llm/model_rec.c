/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * model_rec.c - the Gated DeltaNet layers of the forward pass: causal
 * convolution, the recurrence, the gated norm (see deltanet.h).
 */
#define _GNU_SOURCE
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "attention.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "deltanet.h"
#include "tuner.h"
#include "gpu.h"
#include "matvec.h"
#include "quant.h"
#include "vmath.h"

#include "model_impl.h"

#define GDN_JOURNAL_BUDGET ((size_t)16 * 1024 * 1024)

static int gdn_opt(const char *name)
{
    const char *s = getenv(name);
    return s && !strcmp(s, "1");
}

struct gdn_fp_job {
    uint64_t mode;
    atomic_int okay;
};

static void gdn_fp_worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct gdn_fp_job *a = arg;
    if (!janas_gdn_chain_eligible() || janas_gdn_fp_mode() != a->mode)
        atomic_store_explicit(&a->okay, 0, memory_order_relaxed);
}

void janas_m_gdn_invalidate(struct janas_llm_model *m)
{
    m->gdn_journal_valid = 0;
    m->gdn_chain_active = m->gdn_journal_active = m->gdn_recover_active = 0;
    m->gdn_konly_active = 0;
    m->gdn_written_mask = 0;
}

void janas_m_gdn_free(struct janas_llm_model *m)
{
    janas_m_gdn_invalidate(m);
    free(m->gdn_delta);
    m->gdn_delta = NULL;
    m->gdn_delta_bytes = m->gdn_delta_stride = 0;
    m->gdn_delta_capacity = 0;
}

void janas_m_gdn_prepare(struct janas_llm_model *m, uint32_t pos, uint32_t n)
{
    uint64_t fpcr = janas_gdn_fp_mode();
    int old = m->gdn_journal_valid && m->gdn_journal_pos == m->base_pos &&
              m->gdn_journal_n == m->log_n && m->gdn_journal_base == m->base &&
              m->gdn_journal_fpcr == fpcr;
    janas_m_gdn_invalidate(m);
    if ((m->a->rec != JANAS_REC_NEXT && m->a->rec != JANAS_REC_Q35) ||
        !gdn_opt("JANAS_GDN_CHAIN") ||
        !janas_gdn_chain_eligible() || !n || n > JANAS_GDN_CHAIN_MAX ||
        !m->ds || m->ds > MAX_SSM_DIM || m->ds % SSM_ROWS ||
        m->gtok_where == JANAS_GT_DEV)
        return;
    uint32_t replay = 0;
    if (pos && m->base_pos != UINT32_MAX && pos >= m->base_pos &&
        (uint64_t)pos < (uint64_t)m->base_pos + m->log_n)
        replay = pos - m->base_pos;
    if (replay > JANAS_GDN_CHAIN_MAX)
        return;
    /* Check every active pool thread BEFORE omitting any convolution output.
       The pool's workers keep their own FPCR; checking only the caller would
       permit a different worker mode to reuse deltas from the wrong mode. */
    struct gdn_fp_job check = {.mode = fpcr};
    atomic_init(&check.okay, 1);
    janas_pool_run(m->compute, gdn_fp_worker, &check);
    if (!atomic_load_explicit(&check.okay, memory_order_relaxed))
        return;
    m->gdn_chain_active = 1;
    m->gdn_pending_pos = pos;
    m->gdn_pending_n = n;
    m->gdn_journal_fpcr = fpcr;
    if (!gdn_opt("JANAS_GDN_JOURNAL") || !m->n_rec || m->n_rec > 64 ||
        !m->n_vh)
        return;
    /* Multiplication is bounded by the byte budget before each step. */
    size_t limit = GDN_JOURNAL_BUDGET / sizeof(float);
    size_t stride = m->n_rec;
    if (stride > limit / m->n_vh)
        return;
    stride *= m->n_vh;
    if (stride > limit / m->ds)
        return;
    stride *= m->ds;
    uint32_t capacity = n;
    if (old && replay && m->gdn_journal_n > capacity)
        capacity = m->gdn_journal_n;
    if (stride > limit / capacity)
        return;
    size_t bytes = stride * capacity * sizeof(float);
    if (m->gdn_delta_stride != stride) {
        free(m->gdn_delta);
        m->gdn_delta = NULL;
        m->gdn_delta_bytes = m->gdn_delta_capacity = 0;
        old = 0;
    }
    if (bytes > m->gdn_delta_bytes) {
        float *p = realloc(m->gdn_delta, bytes);
        if (!p)
            return; /* full projected-input replay remains available */
        m->gdn_delta = p;
        m->gdn_delta_bytes = bytes;
        m->gdn_delta_capacity = capacity;
    }
    m->gdn_delta_stride = stride;
    m->gdn_journal_active = 1;
    m->gdn_recover_active = old && replay != 0;
    m->gdn_konly_active = m->gdn_recover_active && gdn_opt("JANAS_GDN_K_ONLY");
}

void janas_m_gdn_finish(struct janas_llm_model *m)
{
    uint64_t all = m->n_rec >= 64 ? UINT64_MAX
                                : (m->n_rec ? (UINT64_C(1) << m->n_rec) - 1 : 0);
    int valid = m->gdn_journal_active && m->n_rec <= 64 &&
                m->gdn_written_mask == all &&
                m->base_pos == m->gdn_pending_pos &&
                m->log_n == m->gdn_pending_n &&
                m->gdn_journal_fpcr == janas_gdn_fp_mode() &&
                m->gtok_where != JANAS_GT_DEV;
    janas_m_gdn_invalidate(m);
    if (valid) {
        m->gdn_journal_pos = m->base_pos;
        m->gdn_journal_n = m->log_n;
        m->gdn_journal_base = m->base;
        m->gdn_journal_valid = 1;
    }
}

/*
 * Gated DeltaNet in four steps per layer, over the rows of a call: rows
 * 0 .. replay - 1 are logged tokens of the previous call replayed on the base
 * (see the recurrent state in struct janas_llm_model), the n rows after them
 * are the tokens of the call.
 *
 * 1. Causal convolution of the q|k|v channels, then SiLU; the q and k heads
 *    are then L2-normalized. Item = one head's worth of channels (ds), done
 *    8 channels at a time across all tokens.
 */
struct ssm_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    uint32_t n;
    int chain, recover, journal, k_only;
};

static void conv_head_ref(struct janas_llm_model *m, const float *wt,
                          float *src, float *dst, uint32_t c0, uint32_t n,
                          int k_only)
{
    uint32_t cc = m->conv_ch, dc = m->d_conv, a = m->replay;
    for (uint32_t c = c0; c < c0 + m->ds; c++) {
        float win[MAX_CONV];
        for (uint32_t i = 0; i + 1 < dc; i++)
            win[i] = src[(size_t)i * cc + c];
        for (uint32_t t = 0; t < a + n; t++) {
            if (t == a && a > 0) /* the base moves past the replayed tokens */
                for (uint32_t i = 0; i + 1 < dc; i++)
                    src[(size_t)i * cc + c] = win[i];
            win[dc - 1] = m->mixed[(size_t)t * cc + c];
            if (!k_only || t >= a ||
                (c >= m->n_kh * m->ds && c < 2 * m->n_kh * m->ds)) {
                float s = 0.0f;
                for (uint32_t i = 0; i < dc; i++)
                    s += win[i] * wt[(size_t)i * cc + c];
                m->conv_out[(size_t)t * cc + c] = s / (1.0f + janas_expf(-s));
            }
            for (uint32_t i = 0; i + 1 < dc; i++)
                win[i] = win[i + 1];
        }
        for (uint32_t i = 0; i + 1 < dc; i++)
            dst[(size_t)i * cc + c] = win[i];
    }
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static void
conv_head_avx2(struct janas_llm_model *m, const float *wt, float *src,
               float *dst, uint32_t c0, uint32_t n)
{
    uint32_t cc = m->conv_ch, dc = m->d_conv, a = m->replay;
    for (uint32_t c = c0; c < c0 + m->ds; c += 8) {
        __m256 win[MAX_CONV], w[MAX_CONV];
        for (uint32_t i = 0; i < dc; i++)
            w[i] = _mm256_loadu_ps(wt + (size_t)i * cc + c);
        for (uint32_t i = 0; i + 1 < dc; i++)
            win[i] = _mm256_loadu_ps(src + (size_t)i * cc + c);
        for (uint32_t t = 0; t < a + n; t++) {
            if (t == a && a > 0)
                for (uint32_t i = 0; i + 1 < dc; i++)
                    _mm256_storeu_ps(src + (size_t)i * cc + c, win[i]);
            win[dc - 1] = _mm256_loadu_ps(m->mixed + (size_t)t * cc + c);
            __m256 s = _mm256_mul_ps(win[0], w[0]);
            for (uint32_t i = 1; i < dc; i++)
                s = _mm256_fmadd_ps(win[i], w[i], s);
            _mm256_storeu_ps(m->conv_out + (size_t)t * cc + c, janas_silu8(s));
            for (uint32_t i = 0; i + 1 < dc; i++)
                win[i] = win[i + 1];
        }
        for (uint32_t i = 0; i + 1 < dc; i++)
            _mm256_storeu_ps(dst + (size_t)i * cc + c, win[i]);
    }
}

/* x / sqrt(sum x^2 + eps), in place, n a multiple of 8. */
__attribute__((target("avx2,fma"))) static void
l2_norm_avx2(float *x, uint32_t n, float eps)
{
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m256 inv = _mm256_set1_ps(1.0f / sqrtf(janas_hsum8(acc) + eps));
    for (uint32_t i = 0; i < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), inv));
}
#endif

static void l2_norm_ref(float *x, uint32_t n, float eps)
{
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++)
        ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss + eps);
    for (uint32_t i = 0; i < n; i++)
        x[i] *= inv;
}

static void conv_worker(void *arg, int tid, int n_threads)
{
    struct ssm_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t cc = m->conv_ch, ds = m->ds, kd = m->n_kh * ds;
    size_t sz = (size_t)m->n_rec * (m->d_conv - 1) * cc;
    float *src = m->conv_buf +
                 (size_t)(m->rec_cont ? 1 - m->src : m->src) * sz +
                 (size_t)a->ly->slot * (m->d_conv - 1) * cc;
    float *dst = m->conv_buf + (size_t)(1 - m->src) * sz +
                 (size_t)a->ly->slot * (m->d_conv - 1) * cc;
    for (uint32_t c0 = (uint32_t)tid * ds; c0 < cc;
         c0 += (uint32_t)n_threads * ds) {
#if defined(__x86_64__)
        if (m->avx2)
            conv_head_avx2(m, a->ly->conv_t, src, dst, c0, a->n);
        else
#endif
            conv_head_ref(m, a->ly->conv_t, src, dst, c0, a->n, a->k_only);
        if (c0 >= 2 * kd) /* a value head: not normalized */
            continue;
        for (uint32_t t = a->k_only && c0 < kd ? m->replay : 0;
             t < m->replay + a->n; t++) {
            float *x = m->conv_out + (size_t)t * cc + c0;
#if defined(__x86_64__)
            if (m->avx2)
                l2_norm_avx2(x, ds, m->eps);
            else
#endif
                l2_norm_ref(x, ds, m->eps);
        }
    }
}

/*
 * 2. The recurrence, item = (value head, block of SSM_ROWS rows of its
 *    state): rows are independent, so a head is split among threads and each
 *    slice stays in one core's cache for all the tokens. Replayed tokens
 *    update the base in place; the call's tokens start from the base (or the
 *    tip) and leave their state in the other buffer. Raw outputs go to att.
 */
static void rec_worker(void *arg, int tid, int n_threads)
{
    struct ssm_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t ds = m->ds, nv = m->n_vh, rf = nv / m->n_kh, kd = m->n_kh * ds;
    uint32_t cc = m->conv_ch, blocks = ds / SSM_ROWS;
    size_t sz = (size_t)m->n_rec * nv * ds * ds;
    float scale = 1.0f / sqrtf((float)ds), junk[SSM_ROWS];
    for (uint32_t it = (uint32_t)tid; it < nv * blocks;
         it += (uint32_t)n_threads) {
        uint32_t h = it / blocks, r0 = it % blocks * SSM_ROWS;
        uint32_t g = m->vmap_mod ? h % m->n_kh : h / rf;
        size_t off = (((size_t)a->ly->slot * nv + h) * ds + r0) * ds;
        uint16_t *M =
            m->ssm_buf + (size_t)(m->rec_cont ? 1 - m->src : m->src) * sz + off;
        uint16_t *D = m->ssm_buf + (size_t)(1 - m->src) * sz + off;
        if (a->chain) {
            size_t jo = ((size_t)a->ly->slot * nv + h) * ds + r0;
            struct janas_gdn_chain chain = {
                .q = m->conv_out + (size_t)g * ds,
                .k = m->conv_out + kd + (size_t)g * ds,
                .v = m->conv_out + 2 * kd + (size_t)h * ds + r0,
                .decay = m->decay + h,
                .beta = m->beta + h,
                .x_stride = cc,
                .gate_stride = nv,
                .out = m->att + (size_t)h * ds + r0,
                .out_stride = m->d_inner,
                .replay_delta = a->recover ? m->gdn_delta + jo : NULL,
                .delta = a->journal ? m->gdn_delta + jo : NULL,
                .delta_stride = m->gdn_delta_stride};
            /* Geometry/pointers were checked before convolution. In
               particular, a K-only replay must never fall into the old
               path that consumes its intentionally absent Q/V outputs. */
            (void)janas_gdn_chain_h(M, D, &chain, scale, ds, SSM_ROWS,
                                    m->replay, a->n);
            continue;
        }
        for (uint32_t t = 0; t < m->replay + a->n; t++) {
            if (t == m->replay) {
                if (M != D) /* not a call going on from the one before */
                    memcpy(D, M, (size_t)SSM_ROWS * ds * sizeof(uint16_t));
                M = D;
            }
            const float *co = m->conv_out + (size_t)t * cc;
            float *o = t < m->replay
                           ? junk
                           : m->att + (size_t)(t - m->replay) * m->d_inner +
                                 (size_t)h * ds + r0;
            janas_gdn_step_h(
                M, co + (size_t)g * ds, co + kd + (size_t)g * ds,
                co + 2 * kd + (size_t)h * ds + r0, m->decay[(size_t)t * nv + h],
                m->beta[(size_t)t * nv + h], scale, o, ds, SSM_ROWS);
        }
    }
}

#if defined(__x86_64__)
/* rms_norm(o) * w * SiLU(z), in place, n a multiple of 8; with sig, times
   sigmoid(z) instead (qwen4exp) */
__attribute__((target("avx2,fma"))) static void
gate_avx2(float *o, const float *z, const float *w, uint32_t n, float eps,
          int sig)
{
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(o + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m256 sc = _mm256_set1_ps(1.0f / sqrtf(janas_hsum8(acc) / n + eps));
    for (uint32_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(o + i), sc),
                                 _mm256_loadu_ps(w + i));
        __m256 zv = _mm256_loadu_ps(z + i);
        /* sigmoid(z) as silu's: 1 / (1 + e^-z) */
        __m256 g =
            sig ? _mm256_div_ps(_mm256_set1_ps(1.0f),
                                _mm256_add_ps(_mm256_set1_ps(1.0f),
                                              janas_exp8(_mm256_sub_ps(
                                                  _mm256_setzero_ps(), zv))))
                : janas_silu8(zv);
        _mm256_storeu_ps(o + i, _mm256_mul_ps(v, g));
    }
}
#endif

/* 3. RMS norm (ssm_norm) of each head's output, gated by SiLU(z) - by
      sigmoid(z) in qwen4exp, the one difference from Qwen3.5's - item =
      (token, value head). */
static void gate_worker(void *arg, int tid, int n_threads)
{
    struct ssm_job *a = arg;
    struct janas_llm_model *m = a->m;
    uint32_t ds = m->ds, nv = m->n_vh, rf = nv / m->n_kh;
    uint32_t gsz = 2 * ds + 2 * ds * rf;
    const float *w = f32(m, a->ly->ssm_norm);
    for (uint32_t it = (uint32_t)tid; it < a->n * nv;
         it += (uint32_t)n_threads) {
        uint32_t j = it / nv, h = it % nv, g = h / rf, r = h % rf;
        float *o = m->att + (size_t)j * m->d_inner + (size_t)h * ds;
        const float *z =
            a->ly->ssm_qkv
                ? m->big + (size_t)j * m->qkvz_dim + m->conv_ch + (size_t)h * ds
                : m->big + (size_t)j * m->qkvz_dim + (size_t)g * gsz + 2 * ds +
                      ds * rf + r * ds;
#if defined(__x86_64__)
        if (m->avx2) {
            gate_avx2(o, z, w, ds, m->eps, m->a->hc);
            continue;
        }
#endif
        rms_norm(o, o, w, ds, m->eps);
        for (uint32_t i = 0; i < ds; i++)
            o[i] *= m->a->hc ? 1.0f / (1.0f + janas_expf(-z[i]))
                             : z[i] / (1.0f + janas_expf(-z[i]));
    }
}

/* Qwen3.5's betas then alphas: item = token, each dot as one thread alone
   would make it (the same bits whatever the threads) */
struct ba_job {
    struct janas_llm_model *m;
    const struct layer *ly;
    const float *xn;
    uint32_t n;
};

static void ba_worker(void *arg, int tid, int n_threads)
{
    struct ba_job *b = arg;
    struct janas_llm_model *m = b->m;
    uint32_t dm = m->d_model, nv = m->n_vh;
    const float *wb = b->ly->ssm_beta, *wa = b->ly->ssm_alpha;
    for (uint32_t j = (uint32_t)tid; j < b->n; j += (uint32_t)n_threads)
        for (uint32_t h = 0; h < nv; h++) {
            const float *xj = b->xn + (size_t)j * dm;
            m->ba[(size_t)j * 2 * nv + h] =
                janas_dot_f32(wb + (size_t)h * dm, xj, dm);
            m->ba[(size_t)j * 2 * nv + nv + h] =
                janas_dot_f32(wa + (size_t)h * dm, xj, dm);
        }
}

void janas_m_rec_ba(struct janas_llm_model *m, const struct layer *ly,
                    const float *xn, uint32_t n)
{
    struct ba_job b = {m, ly, xn, n};
    if (n < 2)
        ba_worker(&b, 0, 1);
    else
        janas_pool_run(m->compute, ba_worker, &b);
}

/* Gated DeltaNet layer on the normalized block in xn / xq; added to x. */
void janas_m_rec_layer(struct janas_llm_model *m, const struct layer *ly,
                       uint32_t n)
{
    uint32_t dm = m->d_model, nv = m->n_vh, cc = m->conv_ch;
    size_t nbi = m->d_inner / JANAS_QK;
    double ta = now();
    if (m->a->rec == JANAS_REC_Q35) {
        /* q|k|v and z side by side in big (row stride qkvz_dim) */
        struct janas_matvec_task t[2] = {{.type = (int)ly->ssm_qkv->type,
                                          .w = data(m, ly->ssm_qkv),
                                          .x = m->xq,
                                          .y = m->big,
                                          .rows = cc,
                                          .cols = dm,
                                          .n_vec = n,
                                          .y_stride = m->qkvz_dim},
                                         {.type = (int)ly->ssm_z->type,
                                          .w = data(m, ly->ssm_z),
                                          .x = m->xq,
                                          .y = m->big + cc,
                                          .rows = m->d_inner,
                                          .cols = dm,
                                          .n_vec = n,
                                          .y_stride = m->qkvz_dim}};
        janas_m_products(m, janas_m_gpu(m, 0), t, 2);
        /* beta then alpha per head, as in the grouped layout's order */
        janas_m_rec_ba(m, ly, m->xn, n);
    } else if (ly->ssm_qkv) {
        /* qwen3next as llama.cpp writes it since 2026: q|k|v and z as for
           qwen35, the betas and alphas fused and grouped as before */
        struct janas_matvec_task t[3] = {{.type = (int)ly->ssm_qkv->type,
                                          .w = data(m, ly->ssm_qkv),
                                          .x = m->xq,
                                          .y = m->big,
                                          .rows = cc,
                                          .cols = dm,
                                          .n_vec = n,
                                          .y_stride = m->qkvz_dim},
                                         {.type = (int)ly->ssm_z->type,
                                          .w = data(m, ly->ssm_z),
                                          .x = m->xq,
                                          .y = m->big + cc,
                                          .rows = m->d_inner,
                                          .cols = dm,
                                          .n_vec = n,
                                          .y_stride = m->qkvz_dim},
                                         {.type = (int)ly->ssm_ba->type,
                                          .w = data(m, ly->ssm_ba),
                                          .x = m->xq,
                                          .y = m->ba,
                                          .rows = 2 * nv,
                                          .cols = dm,
                                          .n_vec = n}};
        janas_m_products(m, janas_m_gpu(m, 0), t, 3);
    } else {
        struct janas_matvec_task t[2] = {{.type = (int)ly->ssm_in->type,
                                          .w = data(m, ly->ssm_in),
                                          .x = m->xq,
                                          .y = m->big,
                                          .rows = m->qkvz_dim,
                                          .cols = dm,
                                          .n_vec = n},
                                         {.type = (int)ly->ssm_ba->type,
                                          .w = data(m, ly->ssm_ba),
                                          .x = m->xq,
                                          .y = m->ba,
                                          .rows = 2 * nv,
                                          .cols = dm,
                                          .n_vec = n}};
        janas_m_products(m, janas_m_gpu(m, 0), t, 2);
    }
    janas_m_rec_core(m, ly, n, ta);
    double tc = now();
    for (uint32_t j = 0; j < n; j++)
        janas_q8k_quantize(m->att + (size_t)j * m->d_inner, m->attq + j * nbi,
                           m->d_inner);
    struct janas_matvec_task to = {.type = (int)ly->ssm_out->type,
                                   .w = data(m, ly->ssm_out),
                                   .x = m->attq,
                                   .y = m->xn,
                                   .rows = dm,
                                   .cols = m->d_inner,
                                   .n_vec = n};
    janas_m_products(m, janas_m_gpu(m, 0), &to, 1);
    trace_row(m, "ssm_out", m->xn + (size_t)(n - 1) * dm, dm);
    for (uint32_t i = 0; i < n * dm; i++)
        m->x[i] += m->xn[i];
    trace_row(m, "residual", m->x + (size_t)(n - 1) * dm, dm);
    m->phase[JANAS_PH_WO] += now() - tc;
}

void janas_m_rec_core(struct janas_llm_model *m, const struct layer *ly,
                      uint32_t n, double ta)
{
    uint32_t ds = m->ds, nv = m->n_vh, rf = nv / m->n_kh;
    uint32_t kd = m->n_kh * ds, cc = m->conv_ch, gsz = 2 * ds + 2 * ds * rf;
    /* per key head g the projection holds q, k, the values and the z gates
       of its rf value heads, then per g: rf betas and rf alphas */
    const float *dt = f32(m, ly->dt_bias), *A = f32(m, ly->ssm_a);
    uint32_t a = m->replay;
    size_t B = JANAS_LLM_MAX_BLOCK;
    float *lm = m->log_mixed + (size_t)ly->slot * B * cc;
    float *lb = m->log_beta + (size_t)ly->slot * B * nv;
    float *ld = m->log_decay + (size_t)ly->slot * B * nv;
    /* rows 0 .. a - 1: the logged tokens to replay, then this call's */
    memcpy(m->mixed, lm, (size_t)a * cc * sizeof(float));
    memcpy(m->beta, lb, (size_t)a * nv * sizeof(float));
    memcpy(m->decay, ld, (size_t)a * nv * sizeof(float));
    for (uint32_t j = 0; j < n; j++) {
        const float *src = m->big + (size_t)j * m->qkvz_dim;
        float *mx = m->mixed + (size_t)(a + j) * cc;
        const float *b = m->ba + (size_t)j * 2 * nv;
        if (m->a->rec == JANAS_REC_Q35) {
            /* already in the convolution's channel order */
            memcpy(mx, src, (size_t)cc * sizeof(float));
            for (uint32_t h = 0; h < nv; h++) {
                float al = b[nv + h] + dt[h];
                float sp = al > 20.0f ? al : janas_softplusf(al);
                m->beta[(size_t)(a + j) * nv + h] = sigmoid(b[h]);
                m->decay[(size_t)(a + j) * nv + h] = janas_expf(sp * A[h]);
            }
            continue;
        }
        if (ly->ssm_qkv) /* q|k|v in the convolution's order already */
            memcpy(mx, src, (size_t)cc * sizeof(float));
        for (uint32_t g = 0; g < m->n_kh; g++) {
            const float *s = src + (size_t)g * gsz;
            if (!ly->ssm_qkv) {
                memcpy(mx + (size_t)g * ds, s, ds * sizeof(float));
                memcpy(mx + kd + (size_t)g * ds, s + ds, ds * sizeof(float));
                memcpy(mx + 2 * kd + (size_t)g * ds * rf, s + 2 * ds,
                       ds * rf * sizeof(float));
            }
            for (uint32_t r = 0; r < rf; r++) {
                uint32_t h = g * rf + r;
                float al = b[2 * g * rf + rf + r] + dt[h];
                float sp = al > 20.0f ? al : janas_softplusf(al);
                m->beta[(size_t)(a + j) * nv + h] = sigmoid(b[2 * g * rf + r]);
                m->decay[(size_t)(a + j) * nv + h] = janas_expf(sp * A[h]);
            }
        }
    }
    double tb = now();
    m->phase[JANAS_PH_QKV] += tb - ta;
    trace_row(m, "qkv_mixed", m->mixed + (size_t)(a + n - 1) * cc, cc);
    trace_row(m, "z",
              m->big + (size_t)(n - 1) * m->qkvz_dim +
                  (ly->ssm_qkv ? m->conv_ch : 0),
              m->d_inner);
    int chain = m->gdn_chain_active && !m->rec_cont &&
                n == m->gdn_pending_n && a <= JANAS_GDN_CHAIN_MAX;
    if (!chain) {
        /* A device-split continuation can visit a layer more than once.
           Its first fragment must not publish a complete-pass journal. */
        m->gdn_journal_active = m->gdn_recover_active = m->gdn_konly_active = 0;
    }
    struct ssm_job sj = {.m = m,
                         .ly = ly,
                         .n = n,
                         .chain = chain,
                         .recover = chain && m->gdn_recover_active,
                         .journal = chain && m->gdn_journal_active,
                         .k_only = chain && m->gdn_konly_active};
    double tv = now();
    janas_pool_run(m->compute, conv_worker, &sj);
    double conv_seconds = now() - tv;
    m->gdn_conv_seconds += conv_seconds;
    trace_row(m, "conv_out", m->conv_out + (size_t)(m->replay + n - 1) * cc,
              cc);
    if (m->trace >= 3) {
        const float *co = m->conv_out + (size_t)(m->replay + n - 1) * cc;
        trace_row(m, "conv_q_h15", co + (size_t)15 * ds, ds);
        trace_row(m, "conv_k_h15", co + kd + (size_t)15 * ds, ds);
        trace_row(m, "conv_v_h31", co + 2 * kd + (size_t)31 * ds, ds);
        trace_row(m, "conv_v_h0", co + 2 * kd, ds);
    }
    tv = now();
    janas_pool_run(m->compute, rec_worker, &sj);
    double core_seconds = now() - tv;
    m->gdn_core_seconds += core_seconds;
    /* Recovery and new rows deliberately share traversal and state scratch.
       Allocate measured pool time by row count; this is an estimate, not an
       independently timed phase and never extra generation wall time. */
    if (a && a + n)
        m->gdn_recovery_seconds += (conv_seconds + core_seconds) * a / (a + n);
    if (chain)
        m->gdn_chain_tokens += n;
    if (sj.recover)
        m->gdn_journal_replay_tokens += a;
    else
        m->gdn_scalar_replay_tokens += a;
    if (sj.k_only)
        m->gdn_konly_replay_tokens += a;
    if (sj.journal && ly->slot < 64)
        m->gdn_written_mask |= UINT64_C(1) << ly->slot;
    trace_row(m, "ssm_raw", m->att + (size_t)(n - 1) * m->d_inner, m->d_inner);
    if (m->trace >= 3) {
        const float *o = m->att + (size_t)(n - 1) * m->d_inner;
        trace_row(m, "ssm_raw_h31", o + (size_t)31 * ds, ds);
        trace_row(m, "ssm_raw_h0", o, ds);
        fprintf(stderr, "  beta h0 %.6f h31 %.6f | decay h0 %.6g h31 %.6g\n",
                m->beta[(size_t)(a + n - 1) * nv],
                m->beta[(size_t)(a + n - 1) * nv + 31],
                m->decay[(size_t)(a + n - 1) * nv],
                m->decay[(size_t)(a + n - 1) * nv + 31]);
    }
    janas_pool_run(m->compute, gate_worker, &sj);
    trace_row(m, "ssm_gated", m->att + (size_t)(n - 1) * m->d_inner,
              m->d_inner);
    if (m->trace >= 3) {
        trace_row(m, "gated_h31",
                  m->att + (size_t)(n - 1) * m->d_inner + (size_t)31 * ds, ds);
        trace_row(m, "z_h31",
                  m->big + (size_t)(n - 1) * m->qkvz_dim +
                      (ly->ssm_qkv ? m->conv_ch : 0) + (size_t)31 * ds,
                  ds);
    }
    /* logged for a call that may be taken back; a longer one (a prefill
       block) is not, and forward() leaves the state at its end */
    if (n <= JANAS_LLM_MAX_BLOCK) {
        memcpy(lm, m->mixed + (size_t)a * cc, (size_t)n * cc * sizeof(float));
        memcpy(lb, m->beta + (size_t)a * nv, (size_t)n * nv * sizeof(float));
        memcpy(ld, m->decay + (size_t)a * nv, (size_t)n * nv * sizeof(float));
    }
    m->phase[JANAS_PH_SSM] += now() - tb;
}
