/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * deltanet.c - Gated DeltaNet step (see deltanet.h).
 */
#include "deltanet.h"

#include <math.h>
#include <string.h>

#include "quant.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

void janas_gdn_step_ref(float *M, const float *q, const float *k,
                        const float *v, float decay, float beta, float scale,
                        float *o, size_t dim, size_t rows)
{
    for (size_t j = 0; j < rows; j++) {
        float *r = M + j * dim, sk = 0.0f, sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            r[i] *= decay;
            sk += r[i] * k[i];
        }
        float d = (v[j] - sk) * beta;
        for (size_t i = 0; i < dim; i++) {
            r[i] += d * k[i];
            sq += r[i] * q[i];
        }
        o[j] = sq * scale;
    }
}

#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static inline float hsum(__m256 a)
{
    __m128 s =
        _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/*
 * Row by row: the row is decayed and dotted with k in the same loop, then
 * updated and dotted with q; it is read and written once per step.
 */
__attribute__((target("avx2,fma"))) static void
gdn_step_avx2(float *M, const float *q, const float *k, const float *v,
              float decay, float beta, float scale, float *o, size_t dim,
              size_t rows)
{
    __m256 vd = _mm256_set1_ps(decay);
    for (size_t j = 0; j < rows; j++) {
        float *r = M + j * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_mul_ps(_mm256_loadu_ps(r + i), vd);
            __m256 x1 = _mm256_mul_ps(_mm256_loadu_ps(r + i + 8), vd);
            _mm256_storeu_ps(r + i, x0);
            _mm256_storeu_ps(r + i + 8, x1);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(k + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_mul_ps(_mm256_loadu_ps(r + i), vd);
            _mm256_storeu_ps(r + i, x0);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
        }
        float d = (v[j] - hsum(_mm256_add_ps(a0, a1))) * beta;
        __m256 vdl = _mm256_set1_ps(d);
        a0 = _mm256_setzero_ps();
        a1 = _mm256_setzero_ps();
        for (i = 0; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            __m256 x1 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i + 8),
                                        _mm256_loadu_ps(r + i + 8));
            _mm256_storeu_ps(r + i, x0);
            _mm256_storeu_ps(r + i + 8, x1);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(q + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            _mm256_storeu_ps(r + i, x0);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
        }
        o[j] = hsum(_mm256_add_ps(a0, a1)) * scale;
    }
}
#endif

__attribute__((noinline)) void
janas_gdn_step_h_ref(uint16_t *M, const float *q, const float *k,
                          const float *v, float decay, float beta, float scale,
                          float *o, size_t dim, size_t rows)
{
    float r[256];
    for (size_t j = 0; j < rows; j++) {
        uint16_t *m = M + j * dim;
        float sk = 0.0f, sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            r[i] = janas_fp16_to_fp32(m[i]) * decay;
            sk += r[i] * k[i];
        }
        float d = (v[j] - sk) * beta;
        for (size_t i = 0; i < dim; i++) {
            r[i] += d * k[i];
            sq += r[i] * q[i];
            m[i] = janas_fp32_to_fp16(r[i]);
        }
        o[j] = sq * scale;
    }
}

/* Exceptional inputs use the actual scalar step for output, retaining its
   NaN operand/payload behavior as well as finite arithmetic order. Recovery
   with recorded d still needs no discarded Q/V. */
static void gdn_chain_h_legacy(uint16_t *M, uint16_t *tip,
                               const struct janas_gdn_chain *a, float scale,
                               size_t dim, size_t rows, size_t replay,
                               size_t n)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (M != tip)
        memcpy(tip, M, dim * rows * sizeof(*M));
    for (size_t t = 0; t < replay + n; t++) {
        const float *k = a->k + t * a->x_stride;
        float decay = a->decay[t * a->gate_stride];
        if (t < replay && a->replay_delta) {
            for (size_t j = 0; j < rows; j++) {
                float d = a->replay_delta[t * a->delta_stride + j];
                for (size_t i = 0; i < dim; i++) {
                    float x = janas_fp16_to_fp32(tip[j * dim + i]) * decay;
                    x += d * k[i];
                    tip[j * dim + i] = janas_fp32_to_fp16(x);
                }
            }
        } else {
            const float *v = a->v + t * a->x_stride;
            float beta = a->beta[t * a->gate_stride];
            if (t >= replay && a->delta)
                for (size_t j = 0; j < rows; j++) {
                    float sk = 0.0f;
                    for (size_t i = 0; i < dim; i++) {
                        float x = janas_fp16_to_fp32(tip[j * dim + i]) * decay;
                        sk += x * k[i];
                    }
                    a->delta[(t - replay) * a->delta_stride + j] =
                        (v[j] - sk) * beta;
                }
            if (t >= replay)
                janas_gdn_step_h_ref(tip, a->q + t * a->x_stride, k, v,
                                     decay, beta, scale,
                                     a->out + (t - replay) * a->out_stride,
                                     dim, rows);
            else
                for (size_t j = 0; j < rows; j++) {
                    float junk;
                    janas_gdn_step_h_ref(tip + j * dim, k, k, v + j, decay,
                                         beta, scale, &junk, dim, 1);
                }
        }
        if (t + 1 == replay && M != tip)
            memcpy(M, tip, dim * rows * sizeof(*M));
    }
}

static struct janas_gdn_chain gdn_chain_rows(const struct janas_gdn_chain *a,
                                             size_t j)
{
    struct janas_gdn_chain b = *a;
    if (b.v) b.v += j;
    if (b.out) b.out += j;
    if (b.replay_delta) b.replay_delta += j;
    if (b.delta) b.delta += j;
    return b;
}

#if defined(__aarch64__) || defined(_M_ARM64)
/* Use the hardware conversion only in the IEEE, round-to-nearest FPCR
   configuration tested against the software conversion. Do not change the
   caller's FPCR: other rounding/half modes retain the portable path. */
static int gdn_neon_half_ok(void)
{
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return !(fpcr & ((UINT64_C(1) << 26) | (UINT64_C(3) << 22) |
                     (UINT64_C(1) << 19)));
}

static float32x4_t gdn_load_half4_flag(const uint16_t *p, int *special)
{
    uint16x4_t h = vld1_u16(p);
    /* The software conversion preserves a signalling NaN's payload. FCVTL
       quiets it, so retain the software behavior for nonfinite inputs. */
    if (vmaxv_u16(vceq_u16(vand_u16(h, vdup_n_u16(0x7c00)),
                            vdup_n_u16(0x7c00)))) {
        if (special)
            *special = 1;
        float f[4];
        for (int i = 0; i < 4; i++)
            f[i] = janas_fp16_to_fp32(p[i]);
        return vld1q_f32(f);
    }
    return vcvt_f32_f16(vreinterpret_f16_u16(h));
}

static float32x4_t gdn_load_half4(const uint16_t *p)
{
    return gdn_load_half4_flag(p, NULL);
}

static void gdn_store_half4(uint16_t *p, float32x4_t f)
{
    uint32x4_t bits = vreinterpretq_u32_f32(f);
    if (vmaxvq_u32(vceqq_u32(vandq_u32(bits, vdupq_n_u32(0x7f800000)),
                              vdupq_n_u32(0x7f800000)))) {
        float a[4];
        vst1q_f32(a, f);
        for (int i = 0; i < 4; i++)
            p[i] = janas_fp32_to_fp16(a[i]);
        return;
    }
    /* FCVTN is round-to-nearest, ties-to-even, including half subnormals. */
    vst1_u16(p, vreinterpret_u16_f16(vcvt_f16_f32(f)));
}

/* Four rows become the lanes of a vector. The temporary is transposed, but
   persistent state stays row-major. Every lane adds i=0,1,... in the scalar
   order; neither reduction is reassociated and no multiply-add is fused. */
static void gdn_transpose4(float32x4_t *a, float32x4_t *b, float32x4_t *c,
                            float32x4_t *d)
{
    float32x4x2_t ab = vtrnq_f32(*a, *b), cd = vtrnq_f32(*c, *d);
    *a = vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0]));
    *b = vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1]));
    *c = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    *d = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
}

static void gdn_step_h_neon(uint16_t *M, const float *q, const float *k,
                            const float *v, float decay, float beta, float scale,
                            float *o, size_t dim, size_t rows)
{
    float32x4_t r[256];
    float32x4_t vd = vdupq_n_f32(decay), vb = vdupq_n_f32(beta);
    size_t j = 0;
    for (; j + 4 <= rows; j += 4) {
        uint16_t *m = M + j * dim;
        float32x4_t sk = vdupq_n_f32(0), sq = vdupq_n_f32(0);
        for (size_t i = 0; i < dim; i += 4) {
            float32x4_t a = gdn_load_half4(m + i);
            float32x4_t b = gdn_load_half4(m + dim + i);
            float32x4_t c = gdn_load_half4(m + 2 * dim + i);
            float32x4_t d = gdn_load_half4(m + 3 * dim + i);
            gdn_transpose4(&a, &b, &c, &d);
            r[i] = vmulq_f32(a, vd);
            r[i + 1] = vmulq_f32(b, vd);
            r[i + 2] = vmulq_f32(c, vd);
            r[i + 3] = vmulq_f32(d, vd);
            sk = vaddq_f32(sk, vmulq_n_f32(r[i], k[i]));
            sk = vaddq_f32(sk, vmulq_n_f32(r[i + 1], k[i + 1]));
            sk = vaddq_f32(sk, vmulq_n_f32(r[i + 2], k[i + 2]));
            sk = vaddq_f32(sk, vmulq_n_f32(r[i + 3], k[i + 3]));
        }
        float32x4_t delta = vmulq_f32(vsubq_f32(vld1q_f32(v + j), sk), vb);
        for (size_t i = 0; i < dim; i += 4) {
            float32x4_t a = vaddq_f32(r[i], vmulq_n_f32(delta, k[i]));
            float32x4_t b = vaddq_f32(r[i + 1], vmulq_n_f32(delta, k[i + 1]));
            float32x4_t c = vaddq_f32(r[i + 2], vmulq_n_f32(delta, k[i + 2]));
            float32x4_t d = vaddq_f32(r[i + 3], vmulq_n_f32(delta, k[i + 3]));
            sq = vaddq_f32(sq, vmulq_n_f32(a, q[i]));
            sq = vaddq_f32(sq, vmulq_n_f32(b, q[i + 1]));
            sq = vaddq_f32(sq, vmulq_n_f32(c, q[i + 2]));
            sq = vaddq_f32(sq, vmulq_n_f32(d, q[i + 3]));
            gdn_transpose4(&a, &b, &c, &d);
            gdn_store_half4(m + i, a);
            gdn_store_half4(m + dim + i, b);
            gdn_store_half4(m + 2 * dim + i, c);
            gdn_store_half4(m + 3 * dim + i, d);
        }
        vst1q_f32(o + j, vmulq_n_f32(sq, scale));
    }
    if (j < rows)
        janas_gdn_step_h_ref(M + j * dim, q, k, v + j, decay, beta, scale,
                             o + j, dim, rows - j);
}

static float32x4_t gdn_round_half4(float32x4_t f, int *special)
{
    uint32x4_t bits = vreinterpretq_u32_f32(f);
    if (vmaxvq_u32(vceqq_u32(vandq_u32(bits, vdupq_n_u32(0x7f800000)),
                              vdupq_n_u32(0x7f800000)))) {
        *special = 1;
        uint16_t h[4];
        gdn_store_half4(h, f);
        return gdn_load_half4(h);
    }
    return vcvt_f32_f16(vcvt_f16_f32(f));
}

/* The scratch holds four rows, transposed, throughout the chain. Persisting
   only at the replay/new boundary and tip removes repeated row transposes
   and half-state loads/stores, but NEVER removes a token's half rounding. */
static void gdn_chain_store4(uint16_t *m, const float32x4_t *r, size_t dim)
{
    for (size_t i = 0; i < dim; i += 4) {
        float32x4_t a = r[i], b = r[i + 1], c = r[i + 2], d = r[i + 3];
        gdn_transpose4(&a, &b, &c, &d);
        gdn_store_half4(m + i, a);
        gdn_store_half4(m + dim + i, b);
        gdn_store_half4(m + 2 * dim + i, c);
        gdn_store_half4(m + 3 * dim + i, d);
    }
}

static size_t gdn_chain_h_neon(uint16_t *M, uint16_t *tip,
                               const struct janas_gdn_chain *a, float scale,
                               size_t dim, size_t rows, size_t replay,
                               size_t n)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    float32x4_t r[256];
    size_t j = 0;
    for (; j + 4 <= rows; j += 4) {
        int special = 0;
        for (size_t i = 0; i < dim; i += 4) {
            float32x4_t x = gdn_load_half4_flag(M + j * dim + i, &special);
            float32x4_t y = gdn_load_half4_flag(M + (j + 1) * dim + i, &special);
            float32x4_t z = gdn_load_half4_flag(M + (j + 2) * dim + i, &special);
            float32x4_t w = gdn_load_half4_flag(M + (j + 3) * dim + i, &special);
            gdn_transpose4(&x, &y, &z, &w);
            r[i] = x, r[i + 1] = y, r[i + 2] = z, r[i + 3] = w;
        }
        if (special) {
            struct janas_gdn_chain b = gdn_chain_rows(a, j);
            gdn_chain_h_legacy(M + j * dim, tip + j * dim, &b, scale,
                               dim, 4, replay, n);
            continue;
        }
        for (size_t t = 0; t < replay + n; t++) {
            const float *k = a->k + t * a->x_stride;
            float32x4_t decay = vdupq_n_f32(a->decay[t * a->gate_stride]);
            float32x4_t sk = vdupq_n_f32(0), delta;
            int recover = t < replay && a->replay_delta;
            for (size_t i = 0; i < dim; i++) {
                r[i] = vmulq_f32(r[i], decay);
                if (!recover)
                    sk = vaddq_f32(sk, vmulq_n_f32(r[i], k[i]));
            }
            if (recover)
                delta = vld1q_f32(a->replay_delta + t * a->delta_stride + j);
            else {
                const float *v = a->v + t * a->x_stride + j;
                delta = vmulq_n_f32(vsubq_f32(vld1q_f32(v), sk),
                                     a->beta[t * a->gate_stride]);
            }
            int output = t >= replay;
            if (output && a->delta)
                vst1q_f32(a->delta + (t - replay) * a->delta_stride + j,
                           delta);
            float32x4_t sq = vdupq_n_f32(0);
            const float *q = output ? a->q + t * a->x_stride : NULL;
            for (size_t i = 0; i < dim; i++) {
                float32x4_t x = vaddq_f32(r[i], vmulq_n_f32(delta, k[i]));
                if (output)
                    sq = vaddq_f32(sq, vmulq_n_f32(x, q[i]));
                r[i] = gdn_round_half4(x, &special);
            }
            if (special) {
                struct janas_gdn_chain b = gdn_chain_rows(a, j);
                size_t redo_replay = replay;
                if (t >= replay && replay && M != tip) {
                    /* The accepted prefix is already saved in M. New
                       deltas may have replaced old ones; redo new work only. */
                    b.q += replay * b.x_stride;
                    b.k += replay * b.x_stride;
                    b.v += replay * b.x_stride;
                    b.decay += replay * b.gate_stride;
                    b.beta += replay * b.gate_stride;
                    b.replay_delta = NULL;
                    redo_replay = 0;
                }
                gdn_chain_h_legacy(M + j * dim, tip + j * dim, &b, scale,
                                   dim, 4, redo_replay, n);
                break;
            }
            if (output)
                vst1q_f32(a->out + (t - replay) * a->out_stride + j,
                           vmulq_n_f32(sq, scale));
            if (t + 1 == replay && M != tip)
                gdn_chain_store4(M + j * dim, r, dim);
        }
        if (!special)
            gdn_chain_store4(tip + j * dim, r, dim);
    }
    return j;
}
#endif

uint64_t janas_gdn_fp_mode(void)
{
#if defined(__aarch64__) || defined(_M_ARM64)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
#else
    return 0;
#endif
}

int janas_gdn_chain_eligible(void)
{
#if defined(__aarch64__) || defined(_M_ARM64)
    /* Omitting unused replay arithmetic also requires FP exceptions to be
       masked. Do not change the caller's floating-point environment. */
    return gdn_neon_half_ok() && !(janas_gdn_fp_mode() & UINT64_C(0x9f00));
#else
    return 0;
#endif
}

int janas_gdn_chain_h(uint16_t *M, uint16_t *tip,
                      const struct janas_gdn_chain *a, float scale,
                      size_t dim, size_t rows, size_t replay, size_t n)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (!a || dim > 256 || replay > JANAS_GDN_CHAIN_MAX ||
        n > JANAS_GDN_CHAIN_MAX || (!M && rows && dim) ||
        (!tip && rows && dim) ||
        ((replay + n) && (!a->k || !a->decay)) ||
        (n && (!a->q || !a->v || !a->beta || !a->out)) ||
        (replay && !a->replay_delta && (!a->v || !a->beta)))
        return -1;
    if (!rows || (replay == 0 && n == 0))
        return 0;
    size_t j = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (dim % 4 == 0 && gdn_neon_half_ok() && !(M == tip && replay))
        j = gdn_chain_h_neon(M, tip, a, scale, dim, rows, replay, n);
#endif
    if (j < rows) {
        struct janas_gdn_chain b = gdn_chain_rows(a, j);
        gdn_chain_h_legacy(M + j * dim, tip + j * dim, &b, scale, dim,
                           rows - j, replay, n);
    }
    return 0;
}

#ifdef JANAS_DELTANET_TEST
int janas_gdn_test_neon_half_enabled(void)
{
#if defined(__aarch64__) || defined(_M_ARM64)
    return gdn_neon_half_ok();
#else
    return 0;
#endif
}

void janas_gdn_test_half_to_float(const uint16_t *h, float *f, size_t n)
{
    size_t i = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (gdn_neon_half_ok())
        for (; i + 4 <= n; i += 4)
            vst1q_f32(f + i, gdn_load_half4(h + i));
#endif
    for (; i < n; i++)
        f[i] = janas_fp16_to_fp32(h[i]);
}

void janas_gdn_test_float_to_half(const float *f, uint16_t *h, size_t n)
{
    size_t i = 0;
#if defined(__aarch64__) || defined(_M_ARM64)
    if (gdn_neon_half_ok())
        for (; i + 4 <= n; i += 4)
            gdn_store_half4(h + i, vld1q_f32(f + i));
#endif
    for (; i < n; i++)
        h[i] = janas_fp32_to_fp16(f[i]);
}
#endif

#if defined(__x86_64__)
__attribute__((target("avx2,fma,f16c"))) static void
gdn_step_h_avx2(uint16_t *M, const float *q, const float *k, const float *v,
                float decay, float beta, float scale, float *o, size_t dim,
                size_t rows)
{
    __m256 vd = _mm256_set1_ps(decay);
    float r[256];
    for (size_t j = 0; j < rows; j++) {
        uint16_t *m = M + j * dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m + i))), vd);
            __m256 x1 = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m + i + 8))),
                vd);
            _mm256_storeu_ps(r + i, x0);
            _mm256_storeu_ps(r + i + 8, x1);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(k + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(m + i))), vd);
            _mm256_storeu_ps(r + i, x0);
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(k + i), a0);
        }
        float d = (v[j] - hsum(_mm256_add_ps(a0, a1))) * beta;
        __m256 vdl = _mm256_set1_ps(d);
        a0 = _mm256_setzero_ps();
        a1 = _mm256_setzero_ps();
        for (i = 0; i + 16 <= dim; i += 16) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            __m256 x1 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i + 8),
                                        _mm256_loadu_ps(r + i + 8));
            _mm_storeu_si128((__m128i *)(m + i), _mm256_cvtps_ph(x0, 0));
            _mm_storeu_si128((__m128i *)(m + i + 8), _mm256_cvtps_ph(x1, 0));
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
            a1 = _mm256_fmadd_ps(x1, _mm256_loadu_ps(q + i + 8), a1);
        }
        for (; i < dim; i += 8) {
            __m256 x0 = _mm256_fmadd_ps(vdl, _mm256_loadu_ps(k + i),
                                        _mm256_loadu_ps(r + i));
            _mm_storeu_si128((__m128i *)(m + i), _mm256_cvtps_ph(x0, 0));
            a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(q + i), a0);
        }
        o[j] = hsum(_mm256_add_ps(a0, a1)) * scale;
    }
}
#endif

static int use_avx2; /* read by janas_gdn_kernel_name on every architecture */
#if defined(__x86_64__)
static int use_f16c;
#endif

__attribute__((constructor)) static void gdn_init(void)
{
#if defined(__x86_64__)
    __builtin_cpu_init();
    use_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    use_f16c = __builtin_cpu_supports("f16c");
#endif
}

void janas_gdn_step(float *M, const float *q, const float *k, const float *v,
                    float decay, float beta, float scale, float *o, size_t dim,
                    size_t rows)
{
#if defined(__x86_64__)
    if (use_avx2) {
        gdn_step_avx2(M, q, k, v, decay, beta, scale, o, dim, rows);
        return;
    }
#endif
    janas_gdn_step_ref(M, q, k, v, decay, beta, scale, o, dim, rows);
}

void janas_gdn_step_h(uint16_t *M, const float *q, const float *k,
                      const float *v, float decay, float beta, float scale,
                      float *o, size_t dim, size_t rows)
{
#if defined(__x86_64__)
    if (use_avx2 && use_f16c) {
        gdn_step_h_avx2(M, q, k, v, decay, beta, scale, o, dim, rows);
        return;
    }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    if (dim <= 256 && dim % 4 == 0 && gdn_neon_half_ok()) {
        gdn_step_h_neon(M, q, k, v, decay, beta, scale, o, dim, rows);
        return;
    }
#endif
    janas_gdn_step_h_ref(M, q, k, v, decay, beta, scale, o, dim, rows);
}

const char *janas_gdn_kernel_name(void)
{
    return use_avx2 ? "avx2" : "ref";
}
