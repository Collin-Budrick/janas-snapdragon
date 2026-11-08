/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * test_deltanet.c - the Gated DeltaNet step against the recurrence written
 * out in double precision on the untransposed state S[i][j].
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "llm/deltanet.h"
#include "llm/quant.h"

static int failures;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fputc('\n', stderr);                                               \
        }                                                                      \
    } while (0)

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static float rndf(float lo, float hi)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return lo + (hi - lo) * (float)(rng_state >> 40) / (float)(1u << 24);
}

static void unit(float *x, size_t n)
{
    double ss = 0;
    for (size_t i = 0; i < n; i++)
        ss += (double)x[i] * x[i];
    for (size_t i = 0; i < n; i++)
        x[i] = (float)(x[i] / sqrt(ss));
}

typedef void (*step_fn)(float *, const float *, const float *, const float *,
                        float, float, float, float *, size_t, size_t);

/* Runs `steps` tokens through fn and through the double reference. */
static void run(const char *name, step_fn fn, size_t dim, int steps)
{
    float *M = calloc(dim * dim, sizeof(float));
    double *S = calloc(dim * dim, sizeof(double));
    float *q = malloc(dim * sizeof(float)), *k = malloc(dim * sizeof(float));
    float *v = malloc(dim * sizeof(float)), *o = malloc(dim * sizeof(float));
    double worst = 0, scale = 1.0 / sqrt((double)dim);
    for (int t = 0; t < steps; t++) {
        for (size_t i = 0; i < dim; i++) {
            q[i] = rndf(-1, 1);
            k[i] = rndf(-1, 1);
            v[i] = rndf(-2, 2);
        }
        unit(q, dim);
        unit(k, dim);
        float decay = expf(rndf(-0.5f, 0.0f)), beta = rndf(0.05f, 0.95f);
        /* the rows in two ranges, as threads would split a head */
        size_t half = dim / 2;
        fn(M, q, k, v, decay, beta, (float)scale, o, dim, half);
        fn(M + half * dim, q, k, v + half, decay, beta, (float)scale, o + half,
           dim, dim - half);
        for (size_t i = 0; i < dim * dim; i++)
            S[i] *= decay;
        for (size_t j = 0; j < dim; j++) {
            double sk = 0;
            for (size_t i = 0; i < dim; i++)
                sk += S[i * dim + j] * k[i];
            double d = (v[j] - sk) * beta;
            for (size_t i = 0; i < dim; i++)
                S[i * dim + j] += k[i] * d;
        }
        for (size_t j = 0; j < dim; j++) {
            double oj = 0;
            for (size_t i = 0; i < dim; i++)
                oj += S[i * dim + j] * q[i];
            oj *= scale;
            double e = fabs(oj - o[j]) / (1.0 + fabs(oj));
            if (e > worst)
                worst = e;
        }
    }
    double sworst = 0;
    for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++) {
            double e = fabs(S[i * dim + j] - M[j * dim + i]);
            if (e > sworst)
                sworst = e;
        }
    CHECK(worst < 1e-4 && sworst < 1e-4,
          "%s dim %zu: output error %g, state error %g", name, dim, worst,
          sworst);
    printf("%s dim %3zu, %d steps: output error %.2e, state error %.2e\n", name,
           dim, steps, worst, sworst);
    free(M);
    free(S);
    free(q);
    free(k);
    free(v);
    free(o);
}

/* The half-precision state: the same recurrence, the state rounded to f16
   after every step, against the double reference. */
static void run_h(size_t dim, int steps)
{
    uint16_t *M = calloc(dim * dim, sizeof(uint16_t));
    double *S = calloc(dim * dim, sizeof(double));
    float *q = malloc(dim * sizeof(float)), *k = malloc(dim * sizeof(float));
    float *v = malloc(dim * sizeof(float)), *o = malloc(dim * sizeof(float));
    double worst = 0, scale = 1.0 / sqrt((double)dim), omax = 0;
    for (int t = 0; t < steps; t++) {
        for (size_t i = 0; i < dim; i++) {
            q[i] = rndf(-1, 1);
            k[i] = rndf(-1, 1);
            v[i] = rndf(-2, 2);
        }
        unit(q, dim);
        unit(k, dim);
        float decay = expf(rndf(-0.5f, 0.0f)), beta = rndf(0.05f, 0.95f);
        janas_gdn_step_h(M, q, k, v, decay, beta, (float)scale, o, dim, dim);
        for (size_t i = 0; i < dim * dim; i++)
            S[i] *= decay;
        for (size_t j = 0; j < dim; j++) {
            double sk = 0;
            for (size_t i = 0; i < dim; i++)
                sk += S[i * dim + j] * k[i];
            double d = (v[j] - sk) * beta;
            for (size_t i = 0; i < dim; i++)
                S[i * dim + j] += k[i] * d;
        }
        for (size_t j = 0; j < dim; j++) {
            double oj = 0;
            for (size_t i = 0; i < dim; i++)
                oj += S[i * dim + j] * q[i];
            oj *= scale;
            if (fabs(oj) > omax)
                omax = fabs(oj);
            double e = fabs(oj - o[j]);
            if (e > worst)
                worst = e;
        }
    }
    /* relative to the output scale: f16 keeps ~3 decimal digits */
    CHECK(worst / omax < 5e-3, "half state dim %zu: error %g of %g", dim, worst,
          omax);
    printf("half state dim %3zu, %d steps: output error %.2e (of %.2f)\n", dim,
           steps, worst, omax);
    free(M);
    free(S);
    free(q);
    free(k);
    free(v);
    free(o);
}

/* ARM lanes are independent rows, so every state word and output must match
   the scalar float recurrence, not merely a tolerance against double. x86's
   existing FMA/reassociated kernel has a different numerical contract. */
#if !defined(__x86_64__)
static void run_h_exact(size_t dim, size_t rows, int steps)
{
    enum { GUARD = 8 };
    size_t count = dim * rows;
    uint16_t *a = malloc((count + 2 * GUARD) * sizeof(*a));
    uint16_t *b = malloc((count + 2 * GUARD) * sizeof(*b));
    float *q = malloc((dim + 1) * sizeof(*q));
    float *k = malloc((dim + 1) * sizeof(*k));
    float *v = malloc((rows + 1) * sizeof(*v));
    float *oa = malloc((rows + 2 * GUARD) * sizeof(*oa));
    float *ob = malloc((rows + 2 * GUARD) * sizeof(*ob));
    CHECK(a && b && q && k && v && oa && ob, "exact test allocation");
    if (!a || !b || !q || !k || !v || !oa || !ob)
        goto done;
    for (size_t i = 0; i < count + 2 * GUARD; i++)
        a[i] = b[i] = 0x3555;
    for (size_t i = 0; i < rows + 2 * GUARD; i++)
        oa[i] = ob[i] = 123.25f;
    static const uint16_t edge[] = {0, 0x8000, 1, 0x8001, 0x03ff,
                                    0x83ff, 0x0400, 0x8400, 0x3555, 0xb555};
    for (size_t i = 0; i < count; i++)
        a[GUARD + i] = b[GUARD + i] =
            i % 3 ? janas_fp32_to_fp16(rndf(-4, 4))
                  : edge[i % (sizeof(edge) / sizeof(edge[0]))];
    for (int step = 0; step < steps; step++) {
        for (size_t i = 0; i < dim; i++) {
            q[i] = rndf(-0.125f, 0.125f);
            k[i] = rndf(-0.125f, 0.125f);
        }
        for (size_t j = 0; j < rows; j++)
            v[j] = rndf(-2, 2);
        float decay = step % 5 ? rndf(0.7f, 1.0f) : 1.0f;
        float beta = step % 7 ? rndf(0.05f, 0.95f) : 0.0f;
        float scale = step % 11 ? 0.125f : -0.0f;
        janas_gdn_step_h_ref(a + GUARD, q, k, v, decay, beta, scale,
                             oa + GUARD, dim, rows);
        /* Unaligned starting rows and tails model the pool's head slices. */
        size_t split = rows ? (size_t)step % rows : 0;
        janas_gdn_step_h(b + GUARD, q, k, v, decay, beta, scale, ob + GUARD,
                         dim, split);
        janas_gdn_step_h(b + GUARD + split * dim, q, k, v + split, decay,
                         beta, scale, ob + GUARD + split, dim, rows - split);
        if (memcmp(a, b, (count + 2 * GUARD) * sizeof(*a)) ||
            memcmp(oa, ob, (rows + 2 * GUARD) * sizeof(*oa))) {
            CHECK(0, "half exact dim %zu rows %zu step %d", dim, rows, step);
            break;
        }
    }
    for (size_t i = 0; i < GUARD; i++) {
        CHECK(a[i] == 0x3555 && b[i] == 0x3555 &&
                  a[GUARD + count + i] == 0x3555 &&
                  b[GUARD + count + i] == 0x3555,
              "half state guard dim %zu rows %zu", dim, rows);
        CHECK(oa[i] == 123.25f && ob[i] == 123.25f &&
                  oa[GUARD + rows + i] == 123.25f &&
                  ob[GUARD + rows + i] == 123.25f,
              "half output guard dim %zu rows %zu", dim, rows);
    }
done:
    free(a), free(b), free(q), free(k), free(v), free(oa), free(ob);
}

static void half_special_exact(void)
{
    uint16_t a[8 * 8], b[8 * 8];
    float q[8], k[8], v[8], oa[8], ob[8];
    static const uint16_t edge[] = {0x7c00, 0xfc00, 0x7c01, 0xfc01,
                                    0x7fff, 0xffff, 0x7bff, 0xfbff};
    for (size_t i = 0; i < 8; i++)
        q[i] = k[i] = 0.125f, v[i] = (float)i;
    for (int c = 0; c < 8; c++) {
        for (size_t i = 0; i < 8 * 8; i++)
            a[i] = b[i] = i % 9 == 0 ? edge[c] : 0x3555;
        janas_gdn_step_h_ref(a, q, k, v, 0.9375f, 0.5f, 0.125f, oa, 8, 8);
        janas_gdn_step_h(b, q, k, v, 0.9375f, 0.5f, 0.125f, ob, 8, 8);
        CHECK(!memcmp(a, b, sizeof a) && !memcmp(oa, ob, sizeof oa),
              "half special pattern %04x", edge[c]);
    }
    static const uint32_t fedge[] = {0, 0x80000000, 0x7f800000, 0xff800000,
                                     0x7f800123, 0xffc12345, 0x7f7fffff, 1};
    for (int role = 0; role < 6; role++)
        for (size_t c = 0; c < sizeof(fedge) / sizeof(fedge[0]); c++) {
            float special;
            memcpy(&special, &fedge[c], sizeof special);
            for (size_t i = 0; i < 8 * 8; i++)
                a[i] = b[i] = 0x3555;
            for (size_t i = 0; i < 8; i++)
                q[i] = k[i] = 0.125f, v[i] = (float)i;
            float decay = 0.9375f, beta = 0.5f, scale = 0.125f;
            if (role == 0)
                q[1] = special;
            else if (role == 1)
                k[2] = special;
            else if (role == 2)
                v[3] = special;
            else if (role == 3)
                decay = special;
            else if (role == 4)
                beta = special;
            else
                scale = special;
            janas_gdn_step_h_ref(a, q, k, v, decay, beta, scale, oa, 8, 8);
            janas_gdn_step_h(b, q, k, v, decay, beta, scale, ob, 8, 8);
            CHECK(!memcmp(a, b, sizeof a) && !memcmp(oa, ob, sizeof oa),
                  "half float special role %d bits %08x", role, fedge[c]);
        }
}
#endif

#ifdef JANAS_DELTANET_TEST
static void half_conversions(void)
{
    enum { N = 65536 };
    uint16_t *h = malloc(N * sizeof(*h)), *back = malloc(N * sizeof(*back));
    float *f = malloc(N * sizeof(*f));
    CHECK(h && back && f, "conversion test allocation");
    if (!h || !back || !f)
        goto done;
    for (size_t i = 0; i < N; i++)
        h[i] = (uint16_t)i;
    janas_gdn_test_half_to_float(h, f, N);
    janas_gdn_test_float_to_half(f, back, N);
    size_t wrong_in = 0, wrong_out = 0;
    for (size_t i = 0; i < N; i++) {
        float ref = janas_fp16_to_fp32(h[i]);
        wrong_in += memcmp(&f[i], &ref, sizeof ref) != 0;
        wrong_out += back[i] != janas_fp32_to_fp16(ref);
    }
    CHECK(!wrong_in && !wrong_out,
          "all half patterns: %zu input and %zu output mismatches", wrong_in,
          wrong_out);
    /* Both signs, immediately below/at/above every finite half midpoint.
       These cover even/odd ties, exponent carries and half subnormal edges. */
    size_t wrong_ties = 0;
    for (uint16_t i = 0; i < 0x7bff; i++) {
        float lo = janas_fp16_to_fp32(i);
        float hi = janas_fp16_to_fp32((uint16_t)(i + 1));
        float mid = (lo + hi) * 0.5f;
        float a[8] = {nextafterf(mid, -INFINITY), mid,
                       nextafterf(mid, INFINITY), -nextafterf(mid, -INFINITY),
                       -mid, -nextafterf(mid, INFINITY), 65520.0f, -65520.0f};
        uint16_t out[8];
        janas_gdn_test_float_to_half(a, out, 8);
        for (int j = 0; j < 8; j++)
            wrong_ties += out[j] != janas_fp32_to_fp16(a[j]);
    }
    CHECK(!wrong_ties, "half midpoint conversions: %zu mismatches", wrong_ties);
    /* Non-multiple vector lengths and the float exponent/payload space. */
    size_t wrong_bits = 0;
    for (uint32_t i = 0; i < 65536; i++) {
        uint32_t bits[7] = {i << 16, (i << 16) | 0x7fff, (i << 16) | 0xffff,
                            i, i | 0x80000000u, 0x7f800001u, 0xffc12345u};
        float a[7];
        uint16_t out[7];
        memcpy(a, bits, sizeof a);
        janas_gdn_test_float_to_half(a, out, 7);
        for (int j = 0; j < 7; j++)
            wrong_bits += out[j] != janas_fp32_to_fp16(a[j]);
    }
    CHECK(!wrong_bits, "float bit-pattern conversions: %zu mismatches", wrong_bits);
    printf("half conversions: all 65536 patterns, midpoints and float edges exact\n");
done:
    free(h), free(back), free(f);
}
#endif

/* Match the scalar step's exact d, before it mutates any half-state word. */
static void exact_delta(const uint16_t *state, const float *k, const float *v,
                        float decay, float beta, float *d, size_t dim,
                        size_t rows)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (size_t j = 0; j < rows; j++) {
        float sk = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float x = janas_fp16_to_fp32(state[j * dim + i]) * decay;
            sk += x * k[i];
        }
        d[j] = (v[j] - sk) * beta;
    }
}

static void chain_exact(size_t dim, size_t rows, size_t replay, size_t n,
                         int factors, int alias)
{
    enum { GUARD = 8 };
    size_t count = dim * rows, total = replay + n;
    size_t xs = (dim > rows ? dim : rows) + 3, os = rows + 5;
    uint16_t *initial = malloc((count + 2 * GUARD) * sizeof(*initial));
    uint16_t *base = malloc((count + 2 * GUARD) * sizeof(*base));
    uint16_t *tip = malloc((count + 2 * GUARD) * sizeof(*tip));
    uint16_t *ref = malloc(count * sizeof(*ref));
    uint16_t *boundary = malloc(count * sizeof(*boundary));
    float *q = calloc(total * xs + 1, sizeof(*q));
    float *k = calloc(total * xs + 1, sizeof(*k));
    float *v = calloc(total * xs + 1, sizeof(*v));
    float *decay = calloc(total + 1, sizeof(*decay));
    float *beta = calloc(total + 1, sizeof(*beta));
    float *out = malloc((n * os + 2 * GUARD) * sizeof(*out));
    float *expected = malloc((total * os + 1) * sizeof(*expected));
    float *journal = malloc((total * os + 2 * GUARD) * sizeof(*journal));
    float *new_delta = malloc((n * os + 2 * GUARD) * sizeof(*new_delta));
    float *rd = malloc((total * os + 1) * sizeof(*rd));
    CHECK(initial && base && tip && ref && boundary && q && k && v &&
              decay && beta && out && expected && journal && new_delta && rd,
          "chain allocation");
    if (!initial || !base || !tip || !ref || !boundary || !q || !k || !v ||
        !decay || !beta || !out || !expected || !journal || !new_delta || !rd)
        goto done;
    for (size_t i = 0; i < count + 2 * GUARD; i++)
        initial[i] = 0x3555;
    for (size_t i = 0; i < count; i++)
        initial[GUARD + i] = i % 7 == 0 ? (uint16_t)(i & 0x83ff)
                                       : janas_fp32_to_fp16(rndf(-2, 2));
    memcpy(base, initial, (count + 2 * GUARD) * sizeof(*base));
    memcpy(tip, initial, (count + 2 * GUARD) * sizeof(*tip));
    memcpy(ref, initial + GUARD, count * sizeof(*ref));
    memcpy(boundary, ref, count * sizeof(*ref));
    for (size_t i = 0; i < n * os + 2 * GUARD; i++)
        out[i] = new_delta[i] = 123.25f;
    for (size_t i = 0; i < total * os + 2 * GUARD; i++)
        journal[i] = 123.25f;
    float scale = 0.125f;
    for (size_t t = 0; t < total; t++) {
        for (size_t i = 0; i < dim; i++) {
            q[t * xs + i] = rndf(-0.125f, 0.125f);
            k[t * xs + i] = rndf(-0.125f, 0.125f);
        }
        for (size_t j = 0; j < rows; j++)
            v[t * xs + j] = rndf(-2, 2);
        decay[t] = t % 5 ? rndf(0.7f, 1.0f) : 1.0f;
        beta[t] = t % 7 ? rndf(0.05f, 0.95f) : 0.0f;
        exact_delta(ref, k + t * xs, v + t * xs, decay[t], beta[t],
                    rd + t * os, dim, rows);
        janas_gdn_step_h_ref(ref, q + t * xs, k + t * xs, v + t * xs,
                             decay[t], beta[t], scale, expected + t * os,
                             dim, rows);
        if (t + 1 == replay)
            memcpy(boundary, ref, count * sizeof(*ref));
        if (t < replay)
            memcpy(journal + GUARD + t * os, rd + t * os,
                   rows * sizeof(float));
    }
    struct janas_gdn_chain chain = {.q = q,
                                    .k = k,
                                    .v = v,
                                    .decay = decay,
                                    .beta = beta,
                                    .x_stride = xs,
                                    .gate_stride = 1,
                                    .out = out + GUARD,
                                    .out_stride = os,
                                    .replay_delta = factors ? journal + GUARD : NULL,
                                    .delta = alias ? journal + GUARD
                                                   : new_delta + GUARD,
                                    .delta_stride = os};
    /* Poison unused recovery Q/V: factors must make both independent of
       discarded outputs. The next convolution optimization relies on it. */
    if (factors)
        for (size_t t = 0; t < replay; t++)
            for (size_t i = 0; i < xs; i++)
                q[t * xs + i] = v[t * xs + i] = NAN;
    int rc = janas_gdn_chain_h(base + GUARD, alias == 2 ? base + GUARD
                                                       : tip + GUARD,
                               &chain, scale, dim, rows, replay, n);
    CHECK(rc == 0, "chain return dim %zu rows %zu R%zu N%zu", dim, rows,
          replay, n);
    CHECK(!memcmp(alias == 2 ? base + GUARD : tip + GUARD, ref,
                  count * sizeof(*ref)),
          "chain final state dim %zu rows %zu R%zu N%zu factors%d alias%d",
          dim, rows, replay, n, factors, alias);
    if (alias != 2)
        CHECK(!memcmp(base + GUARD, boundary, count * sizeof(*base)),
              "chain replay boundary dim %zu rows %zu R%zu N%zu", dim,
              rows, replay, n);
    for (size_t t = 0; t < n; t++) {
        CHECK(!memcmp(out + GUARD + t * os, expected + (replay + t) * os,
                      rows * sizeof(float)),
              "chain output dim %zu rows %zu R%zu N%zu t%zu", dim, rows,
              replay, n, t);
        CHECK(!memcmp(chain.delta + t * os, rd + (replay + t) * os,
                      rows * sizeof(float)),
              "chain delta dim %zu rows %zu R%zu N%zu t%zu", dim, rows,
              replay, n, t);
    }
    for (size_t i = 0; i < GUARD; i++) {
        CHECK(base[i] == 0x3555 && tip[i] == 0x3555 &&
                  base[GUARD + count + i] == 0x3555 &&
                  tip[GUARD + count + i] == 0x3555,
              "chain state guards");
        CHECK(out[i] == 123.25f && out[GUARD + n * os + i] == 123.25f,
              "chain output guards");
    }
done:
    free(initial), free(base), free(tip), free(ref), free(boundary);
    free(q), free(k), free(v), free(decay), free(beta), free(out);
    free(expected), free(journal), free(new_delta), free(rd);
}

static void chains(void)
{
    /* Every draft width and every possible accepted-input prefix. The
       prefix includes the known root input: A accepted drafts means A+1. */
    for (size_t n = 1; n <= JANAS_GDN_CHAIN_MAX; n++)
        for (size_t r = 0; r <= JANAS_GDN_CHAIN_MAX; r++)
            for (int factors = 0; factors <= 1; factors++)
                chain_exact(128, 7, r, n, factors, factors);
    static const size_t dims[] = {0, 1, 3, 4, 7, 8, 127, 255, 256};
    static const size_t rows[] = {0, 1, 3, 4, 5, 11};
    for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++)
        for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); r++)
            for (int alias = 0; alias <= 2; alias++)
                chain_exact(dims[d], rows[r], 5, 9, 1, alias);
    uint16_t m[4] = {1, 2, 3, 4}, before[4];
    memcpy(before, m, sizeof m);
    struct janas_gdn_chain bad = {0};
    CHECK(janas_gdn_chain_h(m, m, &bad, 1, 257, 1, 0, 1) == -1 &&
              !memcmp(m, before, sizeof m),
          "invalid chain leaves state untouched");
    /* Two ties may not be combined into one larger update: the half-state
       rounds back to 1 after EACH token, while each raw output exceeds 1. */
    uint16_t tie[4] = {0x3c00, 0x3c00, 0x3c00, 0x3c00};
    float tq[2] = {1, 1}, tk[2] = {1, 1}, tv[8], td[2] = {1, 1};
    float tb[2] = {0.5f, 0.5f}, to[8];
    for (size_t i = 0; i < 8; i++) tv[i] = 1.0009765625f;
    struct janas_gdn_chain ties = {.q = tq, .k = tk, .v = tv,
        .decay = td, .beta = tb, .x_stride = 1, .gate_stride = 1,
        .out = to, .out_stride = 4};
    /* A common stride needs separate padded Q/K and V storage. */
    float tqp[8], tkp[8];
    for (size_t i = 0; i < 8; i++) tqp[i] = tkp[i] = 1;
    ties.q = tqp, ties.k = tkp, ties.x_stride = 4;
    CHECK(!janas_gdn_chain_h(tie, tie, &ties, 1, 1, 4, 0, 2), "tie chain");
    for (size_t i = 0; i < 4; i++)
        CHECK(tie[i] == 0x3c00, "per-token half tie rounding");
    for (size_t i = 0; i < 8; i++)
        CHECK(to[i] == 1.00048828125f, "raw output before half rounding");
    printf("half chain: widths 1..17, all accepted prefixes, journal alias, tails exact\n");
}

static void chain_specials(void)
{
    static const uint16_t hedge[] = {0x7c00, 0xfc00, 0x7c01, 0xfc01,
                                     0x7fff, 0xffff, 0x7bff, 0xfbff};
    static const uint32_t fedge[] = {0, 0x80000000, 0x7f800000, 0xff800000,
                                      0x7f800123, 0xffc12345, 0x7f7fffff, 1};
    enum { D = 8, R = 7, N = 3, X = 8 };
    for (int role = -1; role < 6; role++)
        for (size_t c = 0; c < 8; c++) {
            uint16_t a[D * R], b[D * R], boundary[D * R], tip[D * R];
            float q[N * X], k[N * X], v[N * X], decay[N], beta[N];
            float out[N * R], expected[N * R], delta[N * R];
            float special, scale = 0.125f;
            memcpy(&special, &fedge[c], sizeof special);
            for (size_t i = 0; i < D * R; i++)
                a[i] = b[i] = role == -1 && i % 9 == 0 ? hedge[c] : 0x3555;
            for (size_t t = 0; t < N; t++) {
                decay[t] = 0.9375f, beta[t] = 0.5f;
                for (size_t i = 0; i < X; i++)
                    q[t * X + i] = k[t * X + i] = 0.125f,
                    v[t * X + i] = (float)i;
                if (role == 0) q[t * X + 1] = special;
                if (role == 1) k[t * X + 2] = special;
                if (role == 2) v[t * X + 3] = special;
                if (role == 3) decay[t] = special;
                if (role == 4) beta[t] = special;
                if (role == 5) scale = special;
                exact_delta(a, k + t * X, v + t * X, decay[t], beta[t],
                            delta + t * R, D, R);
                janas_gdn_step_h_ref(a, q + t * X, k + t * X, v + t * X,
                                     decay[t], beta[t], scale, expected + t * R,
                                     D, R);
                if (t == 0) memcpy(boundary, a, sizeof a);
            }
            struct janas_gdn_chain chain = {.q = q, .k = k, .v = v,
                .decay = decay, .beta = beta, .x_stride = X, .gate_stride = 1,
                .out = out, .out_stride = R, .replay_delta = delta,
                .delta_stride = R};
            CHECK(!janas_gdn_chain_h(b, tip, &chain, scale, D, R, 1, 2),
                  "special chain");
            CHECK(!memcmp(b, boundary, sizeof b) && !memcmp(tip, a, sizeof a),
                  "chain special state role%d bits%08x", role, fedge[c]);
            CHECK(!memcmp(out, expected + R, 2 * R * sizeof(float)),
                  "chain special output role%d bits%08x", role, fedge[c]);
            if (memcmp(out, expected + R, 2 * R * sizeof(float)))
                for (size_t i = 0; i < 2 * R; i++) {
                    uint32_t actual, want;
                    memcpy(&actual, out + i, sizeof actual);
                    memcpy(&want, expected + R + i, sizeof want);
                    if (actual != want)
                        fprintf(stderr, "special output %zu: %08x != %08x\n",
                                i, actual, want);
                }
        }
    puts("half chain: nonfinite state/activation payloads exact");
}

int main(void)
{
    /* a test that deadlocks must fail, not hold the suite for ever */
    alarm(600);
    size_t dims[] = {8, 24, 128};
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        run("ref ", janas_gdn_step_ref, dims[i], 64);
        run("fast", janas_gdn_step, dims[i], 64);
    }
    run_h(128, 256);
    chains();
    chain_specials();
#if !defined(__x86_64__)
    size_t hdims[] = {0, 1, 3, 4, 7, 8, 12, 24, 127, 128, 255, 256};
    size_t hrows[] = {0, 1, 3, 4, 5, 7, 8, 11};
    for (size_t i = 0; i < sizeof(hdims) / sizeof(hdims[0]); i++)
        for (size_t j = 0; j < sizeof(hrows) / sizeof(hrows[0]); j++)
            run_h_exact(hdims[i], hrows[j], 64);
    run_h_exact(128, 128, 256);
    half_special_exact();
    printf("half recurrence: multi-step state/output bit-exact, including tails\n");
#endif
#ifdef JANAS_DELTANET_TEST
    printf("half NEON enabled in initial FPCR: %s\n",
           janas_gdn_test_neon_half_enabled() ? "yes" : "no");
    half_conversions();
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    uint64_t saved_fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved_fpcr));
    const uint64_t fpcr_modes[] = {UINT64_C(1) << 19, UINT64_C(1) << 26,
                                   UINT64_C(1) << 22, UINT64_C(2) << 22,
                                   UINT64_C(3) << 22};
    for (size_t i = 0; i < sizeof(fpcr_modes) / sizeof(fpcr_modes[0]); i++) {
        uint64_t mode = (saved_fpcr & ~(UINT64_C(3) << 22)) | fpcr_modes[i];
        __asm__ volatile("msr fpcr, %0" : : "r"(mode));
        run_h_exact(8, 7, 8);
        chain_exact(8, 7, 5, 9, 1, 1);
#ifdef JANAS_DELTANET_TEST
        half_conversions();
#endif
    }
    __asm__ volatile("msr fpcr, %0" : : "r"(saved_fpcr));
    printf("half recurrence/conversions: FPCR fallback and rounding modes exact\n");
#endif
    printf("kernel: %s\n", janas_gdn_kernel_name());
    if (failures) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    printf("test_deltanet: ok\n");
    return 0;
}
