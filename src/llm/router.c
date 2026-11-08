/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "router.h"
#include "quant.h"

#include <stdint.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

size_t janas_router4_bytes(size_t rows, size_t cols)
{
    if (!rows || !cols || rows > SIZE_MAX - 3)
        return 0;
    size_t padded = (rows + 3) / 4 * 4;
    if (cols > SIZE_MAX / sizeof(uint16_t) / padded)
        return 0;
    return padded * cols * sizeof(uint16_t);
}

int janas_router4_pack(uint16_t *dst, size_t bytes, const uint16_t *src,
                       size_t rows, size_t cols)
{
    size_t need = janas_router4_bytes(rows, cols);
    if (!src || !dst || !need || bytes < need)
        return -1;
    size_t used = rows * cols * sizeof(uint16_t);
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (a > UINTPTR_MAX - used || b > UINTPTR_MAX - need ||
        (a < b + need && b < a + used))
        return -1;
    for (size_t i = 0; i < rows * cols; i++)
        if ((src[i] & 0x7c00) == 0x7c00)
            return -1;
    for (size_t e = 0; e < rows; e += 4)
        for (size_t i = 0; i < cols; i++)
            for (size_t lane = 0; lane < 4; lane++)
                dst[e * cols + 4 * i + lane] =
                    e + lane < rows ? src[(e + lane) * cols + i] : 0;
    return 0;
}

static void dot4_ref(const uint16_t *packed, const float *x, size_t cols,
                     float out[4])
{
    float acc[4] = {0};
    for (size_t i = 0; i < cols; i++)
        for (size_t lane = 0; lane < 4; lane++)
            acc[lane] += janas_fp16_to_fp32(packed[4 * i + lane]) * x[i];
    for (size_t lane = 0; lane < 4; lane++)
        out[lane] = acc[lane];
}

void janas_router4_dot(const uint16_t *packed, const float *x, size_t cols,
                       float out[4])
{
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    /* The scalar converter ignores alternate-half/flush modes. Retain it
       when the caller uses a different FP environment from our exact tests. */
    if (fpcr & ((UINT64_C(1) << 19) | (UINT64_C(3) << 22) |
                (UINT64_C(1) << 26))) {
        dot4_ref(packed, x, cols, out);
        return;
    }
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < cols; i++) {
        float32x4_t w = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(packed + 4 * i)));
        /* Build flags prohibit FP contraction: each lane must retain the
           scalar router's rounding and sequential accumulation. */
        acc = vaddq_f32(acc, vmulq_n_f32(w, x[i]));
    }
    vst1q_f32(out, acc);
#else
    dot4_ref(packed, x, cols, out);
#endif
}
