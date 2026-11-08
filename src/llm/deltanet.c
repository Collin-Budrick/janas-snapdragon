/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * deltanet.c - Gated DeltaNet step (see deltanet.h).
 */
#include "deltanet.h"

#include <math.h>

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

void janas_gdn_step_h_ref(uint16_t *M, const float *q, const float *k,
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

static float32x4_t gdn_load_half4(const uint16_t *p)
{
    uint16x4_t h = vld1_u16(p);
    /* The software conversion preserves a signalling NaN's payload. FCVTL
       quiets it, so retain the software behavior for nonfinite inputs. */
    if (vmaxv_u16(vceq_u16(vand_u16(h, vdup_n_u16(0x7c00)),
                            vdup_n_u16(0x7c00)))) {
        float f[4];
        for (int i = 0; i < 4; i++)
            f[i] = janas_fp16_to_fp32(p[i]);
        return vld1q_f32(f);
    }
    return vcvt_f32_f16(vreinterpret_f16_u16(h));
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
#endif

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
