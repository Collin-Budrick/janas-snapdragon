/* SPDX-License-Identifier: GPL-3.0-or-later */
#undef NDEBUG /* Exactness checks must also run in release builds. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/quant.h"
#include "llm/router.h"

static uint32_t bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static float reference(const uint16_t *w, const float *x, size_t n)
{
    float sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += janas_fp16_to_fp32(w[i]) * x[i];
    return sum;
}

static void geometry(size_t rows, size_t cols)
{
    size_t bytes = janas_router4_bytes(rows, cols);
    uint16_t *w = malloc(rows * cols * sizeof(*w));
    uint16_t *p = malloc(bytes + 16);
    float *x = malloc((cols + 3) * 3 * sizeof(*x));
    assert(w && p && x && bytes);
    memset(p, 0xa5, bytes + 16);
    for (size_t i = 0; i < rows * cols; i++)
        w[i] = janas_fp32_to_fp16((float)((int)(i * 37 % 101) - 50) / 31.0f);
    for (size_t i = 0; i < (cols + 3) * 3; i++)
        x[i] = (float)((int)(i * 23 % 113) - 56) / 41.0f;
    assert(janas_router4_pack(p, bytes - 1, w, rows, cols) == -1);
    assert(janas_router4_pack(p, bytes, w, rows, cols) == 0);
    for (size_t i = bytes; i < bytes + 16; i++)
        assert(((const uint8_t *)p)[i] == 0xa5);
    for (size_t e = 0; e < rows; e += 4)
        for (size_t t = 0; t < 3; t++) {
            const float *v = x + t * (cols + 3);
            float got[4];
            janas_router4_dot(p + e * cols, v, cols, got);
            for (size_t lane = 0; lane < 4; lane++) {
                float want = e + lane < rows
                                 ? reference(w + (e + lane) * cols, v, cols)
                                 : 0.0f;
                assert(bits(got[lane]) == bits(want));
            }
        }
    /* The resident F32 router's two halves are exactly adjacent. */
    size_t used = rows * cols * sizeof(*w);
    uint8_t *resident = malloc(used + bytes);
    assert(resident);
    memcpy(resident, w, used);
    assert(janas_router4_pack((uint16_t *)(resident + used), bytes,
                               (const uint16_t *)resident, rows, cols) == 0);
    assert(memcmp(resident, w, used) == 0);
    assert(memcmp(resident + used, p, bytes) == 0);
    assert(janas_router4_pack((uint16_t *)resident, used + bytes,
                               (const uint16_t *)resident, rows, cols) == -1);
    free(resident);
    free(w);
    free(p);
    free(x);
}

static void all_half_patterns(void)
{
    const float x[] = {1.0f, -1.0f, 0.0f, -0.0f, 0x1p-126f, 0x1p100f};
    for (uint32_t h = 0; h < 65536; h += 4) {
        uint16_t w[4] = {(uint16_t)h, (uint16_t)(h + 1),
                          (uint16_t)(h + 2), (uint16_t)(h + 3)};
        for (size_t t = 0; t < sizeof(x) / sizeof(x[0]); t++) {
            float got[4];
            janas_router4_dot(w, x + t, 1, got);
            for (size_t lane = 0; lane < 4; lane++)
                assert(bits(got[lane]) == bits(reference(w + lane, x + t, 1)));
        }
    }
}

int main(void)
{
    assert(!janas_router4_bytes(0, 2));
    assert(!janas_router4_bytes(4, 0));
    assert(!janas_router4_bytes(SIZE_MAX, 1));
    assert(!janas_router4_bytes(4, SIZE_MAX));
    uint16_t dummy[4] = {0};
    assert(janas_router4_pack(NULL, 8, dummy, 4, 1) == -1);
    assert(janas_router4_pack(dummy, 8, NULL, 4, 1) == -1);
    uint16_t nonfinite[4] = {0, 0, 0, 0x7c01};
    assert(janas_router4_pack(dummy, 8, nonfinite, 4, 1) == -1);
    for (size_t i = 0; i < 4; i++)
        assert(dummy[i] == 0);
    const size_t rows[] = {1, 3, 4, 5, 7, 8, 17, 512};
    const size_t cols[] = {1, 3, 7, 8, 15, 128, 256, 2048};
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++)
        for (size_t j = 0; j < sizeof(cols) / sizeof(cols[0]); j++)
            geometry(rows[i], cols[j]);
    all_half_patterns();
#if defined(__aarch64__)
    uint64_t saved;
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved));
    const uint64_t modes[] = {UINT64_C(1) << 19, UINT64_C(1) << 22,
                               UINT64_C(2) << 22, UINT64_C(3) << 22,
                               UINT64_C(1) << 26};
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        uint64_t mode = saved | modes[i];
        __asm__ volatile("msr fpcr, %0" : : "r"(mode) : "memory");
        all_half_patterns();
    }
    __asm__ volatile("msr fpcr, %0" : : "r"(saved) : "memory");
#endif
    puts("test_router: exact four-row products, packing, tails and all half patterns passed");
    return 0;
}
