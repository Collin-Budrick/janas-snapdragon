/* SPDX-License-Identifier: GPL-3.0-or-later */
/* AUTO precision must be independent of verification width. Access the
 * calibrated thresholds directly: never benchmark or write a user's cache. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/llm/attention.c"

static int failures;
static uint32_t random_word = 12345;
#define CHECK(c, msg) do { if (!(c)) { \
    fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); failures++; \
} } while (0)

static uint32_t next_word(void)
{
    random_word = random_word * UINT32_C(1664525) + UINT32_C(1013904223);
    return random_word;
}
static float next_float(void)
{ return ((int32_t)(next_word() >> 16) - 32768) / 16384.0f; }

static void check_case(struct janas_pool *pool, unsigned heads, unsigned kvheads,
                       unsigned hd, unsigned n, unsigned pos,
                       uint32_t token_threshold, uint32_t block_threshold,
                       int custom_last, unsigned window, int fixed)
{
    const unsigned ctx = 1088, guard = 11;
    size_t row = (size_t)heads * hd, count = (size_t)n * row;
    size_t kvcount = (size_t)kvheads * ctx * hd;
    size_t scales = (size_t)kvheads * ctx;
    struct janas_attn *a = janas_attn_create(heads, kvheads, hd, ctx,
                                           janas_pool_size(pool), JANAS_ATTN_AUTO);
    struct janas_attn *ref = janas_attn_create(heads, kvheads, hd, ctx,
                                             janas_pool_size(pool), JANAS_ATTN_AUTO);
    int8_t *k = malloc(kvcount), *v = malloc(kvcount);
    float *ks = malloc(scales * sizeof(*ks)), *vs = malloc(scales * sizeof(*vs));
    float *queries = malloc((count + 2 * guard) * sizeof(float));
    float *blocked = malloc((count + 2 * guard) * sizeof(float));
    float *single = malloc(row * sizeof(float));
    uint32_t last[JANAS_ATTN_MAX_BLOCK];
    CHECK(a && ref && k && v && ks && vs && queries && blocked && single,
          "AUTO fixture allocates small independent attention buffers");
    if (!a || !ref || !k || !v || !ks || !vs || !queries || !blocked || !single)
        goto done;
    a->calibrated = ref->calibrated = 1;
    a->from16[0] = ref->from16[0] = token_threshold;
    a->from16[1] = ref->from16[1] = block_threshold;
    ref->scores_auto = 0;
    if (fixed) { a->scores_auto = 0; a->scores = fixed; }
    janas_attn_set_window(a, window); janas_attn_set_window(ref, window);
    for (size_t i = 0; i < kvcount; i++) {
        k[i] = (int8_t)((int)(next_word() % 255) - 127);
        v[i] = (int8_t)((int)(next_word() % 255) - 127);
    }
    for (size_t i = 0; i < scales; i++) {
        ks[i] = 0.003f + (next_word() % 300) * 0.00001f;
        vs[i] = 0.002f + (next_word() % 300) * 0.00001f;
    }
    for (size_t i = 0; i < count + 2 * guard; i++) {
        queries[i] = -9876.5f; blocked[i] = 8765.25f;
    }
    float *q = queries + guard, *out = blocked + guard;
    for (size_t i = 0; i < count; i++) q[i] = next_float();
    /* A noncausal span straddles the precision threshold. Every row sees
       the same image end, so dropping or misoffsetting last[] is visible. */
    for (unsigned j = 0; j < n; j++) last[j] = pos + n + 3;
    if (custom_last) janas_attn_set_last(a, last);
    janas_attn_run(a, pool, q, k, ks, v, vs, n, pos, out);
    CHECK(a->last == NULL, "set_last is consumed once, including a split AUTO call");
    int equal = 1;
    for (unsigned j = 0; j < n; j++) {
        unsigned absolute = pos + j;
        ref->scores = fixed ? fixed : token_threshold != UINT32_MAX && absolute >= token_threshold
                                          ? JANAS_ATTN_INT16 : JANAS_ATTN_FLOAT;
        if (custom_last) janas_attn_set_last(ref, last + j);
        janas_attn_run(ref, pool, q + (size_t)j * row, k, ks, v, vs, 1,
                       absolute, single);
        if (memcmp(out + (size_t)j * row, single, row * sizeof(float))) equal = 0;
    }
    if (!equal) {
        fprintf(stderr, "AUTO geometry=%u/%u/%u pos=%u n=%u thresholds=%u/%u last=%d window=%u fixed=%d\n",
                heads, kvheads, hd, pos, n, token_threshold, block_threshold,
                custom_last, window, fixed);
        CHECK(equal, "every output word equals the unchanged single-token precision policy");
    }
    for (unsigned j = 0; j < guard; j++) {
        CHECK(queries[j] == -9876.5f && queries[guard + count + j] == -9876.5f,
              "query subview guards are unchanged");
        CHECK(blocked[j] == 8765.25f && blocked[guard + count + j] == 8765.25f,
              "split output row offsets preserve both guards");
    }
    /* The next call must be causal again after consuming a noncausal span. */
    unsigned next = pos + n;
    janas_attn_run(a, pool, q, k, ks, v, vs, 1, next, out);
    ref->scores = fixed ? fixed : token_threshold != UINT32_MAX && next >= token_threshold
                                      ? JANAS_ATTN_INT16 : JANAS_ATTN_FLOAT;
    janas_attn_run(ref, pool, q, k, ks, v, vs, 1, next, single);
    CHECK(memcmp(out, single, row * sizeof(float)) == 0,
          "subsequent causal call never inherits an image span pointer");
done:
    janas_attn_destroy(a); janas_attn_destroy(ref);
    free(k); free(v); free(ks); free(vs);
    free(queries); free(blocked); free(single);
}

int main(void)
{
    /* The fixture sets calibrated state directly; no timing/cache heuristic
       or environment override may select different math for this test. */
    if (setenv("JANAS_ATTN", "auto", 1) != 0) return 1;
    struct janas_pool *pool = janas_pool_create(2, NULL);
    CHECK(pool != NULL, "attention pool created");
    if (!pool) return 1;
    const uint32_t thresholds[][2] = {
        {UINT32_MAX, 1024}, {0, UINT32_MAX}, {1024, UINT32_MAX}, {1024, 1024}
    };
    const unsigned positions[] = {1008, 1016, 1023, 1024, 1025};
    for (unsigned n = 1; n <= 17; n++)
        for (unsigned t = 0; t < sizeof(thresholds) / sizeof(thresholds[0]); t++)
            for (unsigned p = 0; p < sizeof(positions) / sizeof(positions[0]); p++)
                for (int span = 0; span <= 1; span++)
                    check_case(pool, 4, 2, 32, n, positions[p],
                               thresholds[t][0], thresholds[t][1], span, 0, 0);
    /* Production geometry, GQA subgroup tails, window clipping and explicit
       FLOAT/INT16 modes retain their existing arithmetic and row ordering. */
    const unsigned widths[] = {1, 2, 3, 5, 7, 8, 9, 16, 17};
    for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        check_case(pool, 16, 2, 256, widths[i], 1023, 1024, UINT32_MAX, 1, 0, 0);
        check_case(pool, 3, 1, 48, widths[i], 1018, 1024, 0, 1, 257, 0);
        check_case(pool, 4, 2, 32, widths[i], 1023, 0, UINT32_MAX, 1, 0, JANAS_ATTN_FLOAT);
        check_case(pool, 4, 2, 32, widths[i], 1023, UINT32_MAX, 0, 1, 0, JANAS_ATTN_INT16);
    }
    janas_pool_destroy(pool);
    printf("test_attention_auto: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
