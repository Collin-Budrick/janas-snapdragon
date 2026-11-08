/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Explicit hardware check, separate from the optional-backend unit tests. */
#include "common/npu.h"
#include "llm/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int packed_check(struct janas_npu *u, int type, int npu_type,
                         const char *name)
{
    enum { K = 256, N = 32, MAX_M = 16 };
    static const int batches[] = {1, 2, 4, 16};
    float original[N * K], reference[N * K];
    struct janas_block_q4k q4[N];
    struct janas_block_q6k q6[N];
    void *weights = type == JANAS_Q4_K ? (void *)q4 : (void *)q6;
    for (int i = 0; i < N * K; i++)
        original[i] = (float)((i * 13 + i / K * 7) % 61 - 30) / 17.0f;
    if (type == JANAS_Q4_K)
        janas_q4k_quantize(original, q4, N * K);
    else
        janas_q6k_quantize(original, q6, N * K);
    janas_dequantize(type, weights, reference, N * K);
    float *x = janas_npu_alloc(u, MAX_M * K * sizeof(float));
    float *y = janas_npu_alloc(u, MAX_M * N * sizeof(float));
    if (!x || !y)
        return 1;
    int failed = 0;
    char error[512];
    for (unsigned b = 0; b < sizeof batches / sizeof *batches; b++) {
        int m = batches[b];
        struct janas_npu_mm *p = janas_npu_mm_create(
            u, m, N, K, npu_type, weights, NULL, x, y, error, sizeof error);
        if (!p) {
            fprintf(stderr, "%s batch %d unsupported: %s\n", name, m, error);
            failed = 1;
            break;
        }
        double worst = 0, squared_error = 0, squared_reference = 0;
        for (int round = 0; round < 3; round++) {
            for (int v = 0; v < m; v++)
                for (int col = 0; col < K; col++)
                    x[v * K + col] =
                        (float)((col + 3 * v + round) % 17 - 8) / 16.0f;
            if (janas_npu_mm_run(p)) {
                failed = 1;
                break;
            }
            for (int v = 0; v < m; v++)
                for (int row = 0; row < N; row++) {
                    double ref = 0;
                    for (int col = 0; col < K; col++)
                        ref += (double)x[v * K + col] * reference[row * K + col];
                    double diff = fabs((double)y[v * N + row] - ref);
                    squared_error += diff * diff;
                    squared_reference += ref * ref;
                    if (diff > worst)
                        worst = diff;
                    if (!isfinite(y[v * N + row]))
                        failed = 1;
                }
        }
        /* The matching GGML MUL_MAT correctness suite uses NMSE <= 5e-4
           for quantized weights and F32 inputs (1ec81880, test_mul_mat). */
        double nmse = squared_error / (squared_reference ? squared_reference : 1);
        if (!isfinite(nmse) || nmse > 5e-4)
            failed = 1;
        printf("%s MatMul batch=%d, max_abs_error=%.8g, nmse=%.8g: %s\n",
               name, m, worst, nmse,
               failed ? "FAILED" : "OK");
        janas_npu_mm_destroy(p);
        if (failed)
            break;
    }
    janas_npu_free(u, x);
    janas_npu_free(u, y);
    return failed;
}

int main(int argc, char **argv)
{
    enum { K = 256, N = 16, MAX_M = 16 };
    static const uint16_t half[] = {0, 0x3000, 0x3400, 0x3800,
                                    0x3c00, 0xb800, 0xbc00};
    static const float value[] = {0, 0.125f, 0.25f, 0.5f, 1, -0.5f, -1};
    static const int batches[] = {1, 2, 4, 16};
    uint16_t w[N * K];
    char error[512];
    struct janas_npu *u = janas_npu_create(error, sizeof error);
    if (!u) {
        fprintf(stderr, "NPU unavailable: %s\n", error);
        return 77;
    }
    if (!janas_npu_is_hexagon(u)) {
        fprintf(stderr, "Expected Hexagon, got %s\n", janas_npu_name(u));
        janas_npu_destroy(u);
        return 1;
    }
    printf("Janas provider: %s\n", janas_npu_name(u));
    float *x = janas_npu_alloc(u, MAX_M * K * sizeof(float));
    float *y = janas_npu_alloc(u, MAX_M * N * sizeof(float));
    if (!x || !y) {
        fprintf(stderr, "Cannot allocate NPU staging buffers\n");
        janas_npu_destroy(u);
        return 1;
    }
    for (int row = 0; row < N; row++)
        for (int col = 0; col < K; col++)
            w[row * K + col] = half[(row + 3 * col) % 7];
    int failed = 0;
    for (unsigned b = 0; b < sizeof batches / sizeof *batches; b++) {
        int m = batches[b];
        struct janas_npu_mm *p = janas_npu_mm_create(
            u, m, N, K, JANAS_NPU_F16, w, NULL, x, y, error, sizeof error);
        if (!p) {
            fprintf(stderr, "Cannot create batch %d: %s\n", m, error);
            failed = 1;
            break;
        }
        double worst = 0;
        for (int round = 0; round < 3; round++) {
            for (int v = 0; v < m; v++)
                for (int col = 0; col < K; col++)
                    x[v * K + col] =
                        (float)((col + 3 * v + round) % 17 - 8) / 16.0f;
            if (janas_npu_mm_run(p)) {
                fprintf(stderr, "Cannot execute batch %d\n", m);
                failed = 1;
                break;
            }
            for (int v = 0; v < m; v++)
                for (int row = 0; row < N; row++) {
                    double ref = 0;
                    for (int col = 0; col < K; col++)
                        ref += (double)x[v * K + col] *
                               value[(row + 3 * col) % 7];
                    double diff = fabs((double)y[v * N + row] - ref);
                    if (diff > worst)
                        worst = diff;
                    if (!isfinite(y[v * N + row]) || diff > 0.002 * (1 + fabs(ref)))
                        failed = 1;
                }
        }
        printf("F16 MatMul batch=%d, max_abs_error=%.8g: %s\n", m, worst,
               failed ? "FAILED" : "OK");
        janas_npu_mm_destroy(p);
        if (failed)
            break;
    }
    if (!failed && argc > 1 && !strcmp(argv[1], "--packed")) {
        failed |= packed_check(u, JANAS_Q4_K, JANAS_NPU_Q4_K, "Q4_K");
        if (!failed)
            failed |= packed_check(u, JANAS_Q6_K, JANAS_NPU_Q6_K, "Q6_K");
    }
    if (!failed) {
        size_t before = janas_npu_weight_bytes(u);
        struct janas_npu_mm *base = janas_npu_mm_create(
            u, 1, N, K, JANAS_NPU_F16, w, NULL, x, y, error, sizeof error);
        size_t charged = janas_npu_weight_bytes(u);
        /* Creation owns W; rebatching must share that owned copy. */
        memset(w, 0, sizeof w);
        struct janas_npu_mm *clone = base
            ? janas_npu_mm_rebatch(base, 4, x, y, error, sizeof error) : NULL;
        if (!base || !clone || charged <= before ||
            janas_npu_weight_bytes(u) != charged)
            failed = 1;
        janas_npu_mm_destroy(base);
        if (clone && (janas_npu_weight_bytes(u) != charged ||
                      janas_npu_mm_run(clone)))
            failed = 1;
        if (clone && !failed) {
            for (int v = 0; v < 4; v++)
                for (int row = 0; row < N; row++) {
                    double ref = 0;
                    for (int col = 0; col < K; col++)
                        ref += (double)x[v * K + col] *
                               value[(row + 3 * col) % 7];
                    if (!isfinite(y[v * N + row]) ||
                        fabs(y[v * N + row] - ref) > 0.002 * (1 + fabs(ref)))
                        failed = 1;
                }
        }
        janas_npu_mm_destroy(clone);
        if (janas_npu_weight_bytes(u) != before)
            failed = 1;
        printf("Shared weights, rebatch, original-plan destruction: %s\n",
               failed ? "FAILED" : "OK");
    }
    int products, cached;
    double setup;
    janas_npu_stats(u, &products, &cached, &setup);
    printf("Products=%d, cached=%d, setup_seconds=%.6f\n", products, cached, setup);
    janas_npu_destroy(u);
    /* Model capability discovery creates and releases a provider before the
       model opens its own. Runtime DLL state must survive that lifecycle. */
    for (int attempt = 0; !failed && attempt < 3; attempt++) {
        u = janas_npu_create(error, sizeof error);
        if (!u) {
            fprintf(stderr, "Cannot reopen Hexagon provider: %s\n", error);
            failed = 1;
            break;
        }
        x = janas_npu_alloc(u, K * sizeof(float));
        y = janas_npu_alloc(u, N * sizeof(float));
        for (int row = 0; row < N; row++)
            for (int col = 0; col < K; col++)
                w[row * K + col] = half[(row + 3 * col) % 7];
        struct janas_npu_mm *p = x && y ? janas_npu_mm_create(
            u, 1, N, K, JANAS_NPU_F16, w, NULL, x, y, error, sizeof error) : NULL;
        if (x && y) {
            for (int col = 0; col < K; col++)
                x[col] = (float)((col + attempt) % 17 - 8) / 16.0f;
            for (int row = 0; row < N; row++)
                y[row] = NAN;
        }
        if (!p || janas_npu_mm_run(p)) {
            fprintf(stderr, "Reopen attempt %d matrix failed: %s\n", attempt,
                    error);
            failed = 1;
        }
        for (int row = 0; !failed && row < N; row++) {
            double ref = 0;
            for (int col = 0; col < K; col++)
                ref += (double)x[col] * value[(row + 3 * col) % 7];
            if (!isfinite(y[row]) || fabs(y[row] - ref) > 0.002 * (1 + fabs(ref))) {
                fprintf(stderr, "Reopen attempt %d row %d: expected %.9g, got %.9g\n",
                        attempt, row, ref, y[row]);
                failed = 1;
            }
        }
        janas_npu_mm_destroy(p);
        janas_npu_destroy(u);
    }
    printf("Repeated provider creation and execution: %s\n",
           failed ? "FAILED" : "OK");
    return failed ? 1 : 0;
}
