/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * A small teacher-forced numerical quality probe, not a language benchmark.
 * All profiles see the same tokens, original expert/head bits and all routed
 * experts. No sampling, MTP or generated-token feedback is involved.
 *
 * quality_check model.jns [--profile cpu|gpu|npu|gpu+npu] [--steps 64]
 *   [--batch 1 | --batches 1,3,5,8,9,17]
 *   [--cache-mib 16384] [--context 1024] [--threads N]
 *   [--tokens prompt.int32] [--write-reference cpu.qref | --reference cpu.qref]
 *
 * Device-specific Hexagon and GPU settings remain available. The tool overrides
 * JANAS_GPU, JANAS_NPU, device masks, expert/head bits, attention scores,
 * warmup, tuning, block threads and missing-expert skipping. --batch 1
 * exercises single-token passes; small/larger batches exercise the main model's
 * draft/prompt pass shapes. Actual completed accelerator operations are
 * recorded for the measured passes. Exit 3 retains metrics but means the
 * requested hardware profile was not verified (for example CPU fallback). A CPU
 * reference is a numerical baseline, not external model truth.
 *
 * References stream float32 logits with model header fingerprints and exact
 * tokens. They are bounded at 256 MiB and checked for truncation/corruption.
 * Fingerprints/digests are FNV-1a, not full-model cryptographic verification;
 * verify the model payload SHA-256 separately. No reference means NOT COMPARED,
 * never a quality pass. Metrics have no arbitrary pass/fail quality threshold.
 * --batches compares independent sequences after one CPU model load. Each
 * width restarts at position zero and validates the complete reference again.
 * Expert-cache warmth is shared: this is functional validation, not timing.
 *
 * --self-test needs no model. A dependency-free fixture build is also possible:
 * cc -std=c11 -O2 -DQUALITY_CHECK_SELFTEST_ONLY tests/quality_check.c -lm -o qc
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fenv.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#ifndef QUALITY_CHECK_SELFTEST_ONLY
#include "common/devices.h"
#include "common/sysinfo.h"
#include "janas/llm.h"
#include "llm/gpu.h"
#include "llm/jns.h"
#include "llm/model.h"
#include "llm/tokenizer.h"
#endif

#define MAX_STEPS 512u
#define MAX_REFERENCE_BYTES (256ull << 20)
#define FNV_INIT UINT64_C(0xcbf29ce484222325)
#define MAX_BATCHES 16u

static int parse_batches(const char *text, uint32_t widths[MAX_BATCHES],
                           uint32_t *count)
{
    uint32_t n = 0;
    if (!text || !*text) return -1;
    for (;;) {
        uint32_t value = 0;
        if (*text < '0' || *text > '9') return -1;
        do {
            unsigned digit = (unsigned)(*text++ - '0');
            if (value > (MAX_STEPS - digit) / 10) return -1;
            value = value * 10 + digit;
        } while (*text >= '0' && *text <= '9');
        if (!value || value > MAX_STEPS || n == MAX_BATCHES) return -1;
        for (uint32_t i = 0; i < n; i++)
            if (widths[i] == value) return -1;
        widths[n++] = value;
        if (!*text) break;
        if (*text++ != ',' || !*text) return -1;
    }
    *count = n;
    return 0;
}

struct identity {
    uint64_t file_bytes, header, tables, metadata;
};

struct metrics {
    uint64_t rows, agree, values;
    double nll, ref_nll, abs_error, sq_error, max_error, kl;
};

/* Matrix-only, bounded by the existing 256 MiB reference limit. Keep exact
   source rows as the cache key: a digest alone cannot authorize statistic
   reuse. The scalar NLL is accumulated in the original token order. */
struct reference_stats_cache {
    float *rows;
    uint32_t nv, steps;
    struct { double nll, zero_kl; uint32_t target; int valid; } row[MAX_STEPS];
    uint64_t hits, prepared, unequal, nonfinite, control_key, control_changes;
};

static int reference_control_key(uint64_t *key)
{
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ volatile("mrs %0, fpcr" : "=r"(*key));
    return 0;
#elif defined(__x86_64__) || defined(_M_X64)
    int rounding = fegetround();
    if (rounding < 0) return -1;
    /* Exclude sticky exception STATUS flags, include DAZ, masks, rounding and
       flush-to-zero, plus the C/x87 rounding mode used by the math library. */
    *key = (uint64_t)(_mm_getcsr() & 0xffc0u) |
           ((uint64_t)(unsigned)rounding << 32);
    return 0;
#elif defined(FE_TONEAREST)
    int rounding = fegetround();
    if (rounding < 0) return -1;
    *key = (uint64_t)(unsigned)rounding;
    return 0;
#else
    (void)key;
    return -1;
#endif
}

static void reference_cache_invalidate(struct reference_stats_cache *c)
{
    if (c) memset(c->row, 0, sizeof(c->row));
}

static int reference_cache_init(struct reference_stats_cache *c,
                                  uint32_t nv, uint32_t steps)
{
    memset(c, 0, sizeof(*c));
    if (reference_control_key(&c->control_key)) return -1;
    if (!nv || !steps || steps > MAX_STEPS) return -1;
    uint64_t bytes = (uint64_t)nv * steps * sizeof(float);
    if (bytes > MAX_REFERENCE_BYTES || bytes > SIZE_MAX) return -1;
    c->rows = malloc((size_t)bytes);
    if (!c->rows) return -1;
    c->nv = nv; c->steps = steps;
    return 0;
}

static void reference_cache_free(struct reference_stats_cache *c)
{
    free(c->rows);
    memset(c, 0, sizeof(*c));
}

static uint64_t digest(uint64_t h, const void *data, size_t n)
{
    const unsigned char *p = data;
    for (size_t i = 0; i < n; i++)
        h = (h ^ p[i]) * UINT64_C(0x100000001b3);
    return h;
}

/* Stable logsumexp, rejecting every nonfinite value, including -infinity. */
static int row_stats(const float *x, uint32_t nv, uint32_t target, double *nll,
                     uint32_t *top, double *lse)
{
    if (!nv || target >= nv)
        return -1;
    *top = 0;
    for (uint32_t i = 0; i < nv; i++) {
        if (!isfinite(x[i]))
            return -1;
        if (x[i] > x[*top])
            *top = i;
    }
    double sum = 0, mx = x[*top];
    for (uint32_t i = 0; i < nv; i++)
        sum += exp((double)x[i] - mx);
    *lse = mx + log(sum);
    *nll = *lse - x[target];
    return isfinite(*nll) ? 0 : -1;
}

static int measure(struct metrics *s, const float *x, const float *ref,
                   uint32_t nv, uint32_t target)
{
    double nll, lse, rnll = 0, rlse = 0;
    uint32_t top, rtop = 0;
    if (row_stats(x, nv, target, &nll, &top, &lse) ||
        (ref && row_stats(ref, nv, target, &rnll, &rtop, &rlse)))
        return -1;
    s->nll += nll;
    s->rows++;
    if (ref) {
        s->ref_nll += rnll;
        s->agree += top == rtop;
        s->values += nv;
        for (uint32_t i = 0; i < nv; i++) {
            double d = (double)x[i] - ref[i], a = fabs(d);
            s->abs_error += a;
            s->sq_error += d * d;
            if (a > s->max_error)
                s->max_error = a;
            /* KL(reference || candidate), insensitive to a logit offset. */
            s->kl += exp((double)ref[i] - rlse) *
                     (((double)ref[i] - rlse) - ((double)x[i] - lse));
        }
    }
    return 0;
}

static int measure_cached(struct metrics *s, const float *x, const float *ref,
                             uint32_t nv, uint32_t target,
                             struct reference_stats_cache *cache, uint32_t row)
{
    if (!cache || !cache->rows || !ref || nv != cache->nv ||
        row >= cache->steps || target >= nv)
        return measure(s, x, ref, nv, target);
    uint64_t control;
    if (reference_control_key(&control)) {
        reference_cache_invalidate(cache);
        return measure(s, x, ref, nv, target);
    }
    if (control != cache->control_key) {
        reference_cache_invalidate(cache);
        cache->control_key = control;
        cache->control_changes++;
    }
    size_t bytes = (size_t)nv * sizeof(float);
    /* Compare every candidate value before replacing ANY computation. Drift
       still takes the original numerical comparison, including finite checks. */
    if (memcmp(x, ref, bytes)) {
        cache->unequal++;
        return measure(s, x, ref, nv, target);
    }
    for (uint32_t i = 0; i < nv; i++)
        if (!isfinite(ref[i])) {
            cache->nonfinite++;
            reference_cache_invalidate(cache);
            return measure(s, x, ref, nv, target);
        }
    float *stored = cache->rows + (size_t)row * nv;
    if (!cache->row[row].valid || cache->row[row].target != target ||
        memcmp(stored, ref, bytes)) {
        double nll, lse;
        uint32_t top;
        if (row_stats(ref, nv, target, &nll, &top, &lse))
            return measure(s, x, ref, nv, target);
        memcpy(stored, ref, bytes);
        cache->row[row].nll = nll;
        /* Preserve the original zero's sign under directed rounding. The
           separately loaded operands prevent folding x-x to positive zero. */
        volatile double equal = (double)ref[0] - lse;
        cache->row[row].zero_kl = equal - equal;
        cache->row[row].target = target;
        cache->row[row].valid = 1;
        cache->prepared++;
    } else {
        cache->hits++;
    }
    /* Bit-identical finite inputs have the same separately ordered softmax
       reductions and exact zero error/KL. Retain the original sum order. */
    s->nll += cache->row[row].nll;
    s->rows++;
    s->ref_nll += cache->row[row].nll;
    s->agree++;
    s->values += nv;
    /* One zero addition has the same value/sign and FTZ effect as the
       original nv additions; retain it for directed/denormal control modes. */
    volatile double positive_zero = 0;
    s->abs_error += positive_zero;
    s->sq_error += positive_zero;
    s->kl += cache->row[row].zero_kl;
    return 0;
}

/* Version 1 is little-endian IEEE-754 float32, as on Windows/Linux ARM64. */
static int header(FILE *f, int writing, uint32_t nv, uint32_t steps,
                  uint32_t context, const struct identity *id,
                  const int32_t *tokens)
{
    const uint32_t endian = 1;
    const float one = 1.0f;
    uint32_t one_bits;
    memcpy(&one_bits, &one, sizeof one_bits);
    if (sizeof(float) != 4 || *(const unsigned char *)&endian != 1 ||
        one_bits != UINT32_C(0x3f800000) || !nv || !steps ||
        steps > MAX_STEPS || (uint64_t)nv * steps * 4 > MAX_REFERENCE_BYTES)
        return -1;
    unsigned char want[64] = "JANASQ01", got[64];
    uint32_t dims[4] = {one_bits, nv, steps, context};
    uint64_t ids[5] = {id->file_bytes, id->header, id->tables, id->metadata,
                       digest(FNV_INIT, tokens, (steps + 1) * sizeof(*tokens))};
    memcpy(want + 8, dims, sizeof dims);
    memcpy(want + 24, ids, sizeof ids);
    if (writing)
        return fwrite(want, 1, sizeof want, f) != sizeof want ||
                       fwrite(tokens, sizeof(*tokens), steps + 1, f) !=
                           steps + 1
                   ? -1
                   : 0;
    int32_t saved[MAX_STEPS + 1];
    return fread(got, 1, sizeof got, f) != sizeof got ||
                   memcmp(want, got, sizeof got) ||
                   fread(saved, sizeof(*tokens), steps + 1, f) != steps + 1 ||
                   memcmp(tokens, saved, (steps + 1) * sizeof(*tokens))
               ? -1
               : 0;
}

static int footer(FILE *f, int writing, uint64_t h)
{
    uint64_t saved = 0;
    char magic[8];
    if (writing)
        return fwrite("QREFDONE", 1, 8, f) != 8 ||
                       fwrite(&h, sizeof h, 1, f) != 1
                   ? -1
                   : 0;
    return fread(magic, 1, 8, f) != 8 || memcmp(magic, "QREFDONE", 8) ||
                   fread(&saved, sizeof saved, 1, f) != 1 || saved != h ||
                   fgetc(f) != EOF || ferror(f)
               ? -1
               : 0;
}

static int cached_reference_header(FILE *f, uint32_t nv, uint32_t steps,
                                     uint32_t context, const struct identity *id,
                                     const int32_t *tokens,
                                     struct reference_stats_cache *cache)
{
    if (header(f, 0, nv, steps, context, id, tokens)) {
        reference_cache_invalidate(cache);
        return -1;
    }
    return 0;
}

static int cached_reference_row(FILE *f, float *want, uint32_t nv,
                                  struct reference_stats_cache *cache)
{
    if (fread(want, sizeof(float), nv, f) != nv) {
        reference_cache_invalidate(cache);
        return -1;
    }
    return 0;
}

static int cached_reference_footer(FILE *f, uint64_t h,
                                     struct reference_stats_cache *cache)
{
    if (footer(f, 0, h)) {
        reference_cache_invalidate(cache);
        return -1;
    }
    return 0;
}

static int cache_self_test(void)
{
    struct reference_stats_cache cache = {0}, refused = {0};
    if (reference_cache_init(&cache, 3, 2) ||
        reference_cache_init(&refused, 0, 2) == 0 ||
        reference_cache_init(&refused, 3, 0) == 0 ||
        reference_cache_init(&refused, 1000000, MAX_STEPS) == 0 || refused.rows)
        return 1;
    float r[] = {998, 999, 997}, changed[] = {1, 0, 2};
    int bad = 0;
    for (unsigned pass = 0; pass < 3; pass++) {
        struct metrics full = {0}, cached = {0};
        for (unsigned row = 0; row < 2; row++)
            bad |= measure(&full, r, r, 3, row) ||
                   measure_cached(&cached, r, r, 3, row, &cache, row);
        bad |= memcmp(&full, &cached, sizeof(full)) != 0;
    }
    bad |= cache.prepared != 2 || cache.hits != 4;
    struct metrics full = {0}, cached = {0};
    bad |= measure(&full, changed, changed, 3, 1) ||
           measure_cached(&cached, changed, changed, 3, 1, &cache, 0) ||
           memcmp(&full, &cached, sizeof(full)) != 0 || cache.prepared != 3;
    /* Same bytes with a different target cannot reuse that row's old NLL. */
    memset(&full, 0, sizeof(full)); memset(&cached, 0, sizeof(cached));
    bad |= measure(&full, changed, changed, 3, 2) ||
           measure_cached(&cached, changed, changed, 3, 2, &cache, 0) ||
           memcmp(&full, &cached, sizeof(full)) != 0 || cache.prepared != 4;
    float drift[3]; memcpy(drift, r, sizeof(drift)); drift[0] = nextafterf(r[0], INFINITY);
    memset(&full, 0, sizeof(full)); memset(&cached, 0, sizeof(cached));
    bad |= measure(&full, drift, r, 3, 1) ||
           measure_cached(&cached, drift, r, 3, 1, &cache, 0) ||
           memcmp(&full, &cached, sizeof(full)) != 0 ||
           cached.max_error == 0 || cache.unequal != 1;
    for (unsigned i = 0; i < 2; i++) {
        float invalid[] = {0, 1, i ? INFINITY : NAN};
        memset(&cached, 0, sizeof(cached));
        bad |= measure_cached(&cached, invalid, invalid, 3, 1, &cache, 0) == 0 || cached.rows;
        bad |= measure_cached(&cached, invalid, r, 3, 1, &cache, 0) == 0 || cached.rows;
    }
    bad |= cache.nonfinite != 2;
    /* Refused/absent cache storage retains the original full comparison. */
    memset(&full, 0, sizeof(full)); memset(&cached, 0, sizeof(cached));
    bad |= measure(&full, r, r, 3, 1) ||
           measure_cached(&cached, r, r, 3, 1, &refused, 0) ||
           memcmp(&full, &cached, sizeof(full)) != 0;
    /* A real rounding/control-mode change must invalidate the old scalar
       statistics, even while every candidate/reference float stays identical. */
#if defined(FE_DOWNWARD) && defined(FE_UPWARD)
    int original_round = fegetround();
    int alternate_round = original_round == FE_DOWNWARD ? FE_UPWARD : FE_DOWNWARD;
    if (original_round >= 0 && fesetround(alternate_round) == 0) {
        uint64_t prepared = cache.prepared, changes = cache.control_changes;
        uint64_t control;
        bad |= reference_control_key(&control) || control == cache.control_key;
        memset(&full, 0, sizeof(full)); memset(&cached, 0, sizeof(cached));
        bad |= measure(&full, r, r, 3, 1) ||
               measure_cached(&cached, r, r, 3, 1, &cache, 0) ||
               memcmp(&full, &cached, sizeof(full)) != 0 ||
               cache.prepared != prepared + 1 || cache.control_changes != changes + 1;
        bad |= fesetround(original_round) != 0;
        memset(&full, 0, sizeof(full)); memset(&cached, 0, sizeof(cached));
        bad |= measure(&full, r, r, 3, 1) ||
               measure_cached(&cached, r, r, 3, 1, &cache, 0) ||
               memcmp(&full, &cached, sizeof(full)) != 0 ||
               cache.prepared != prepared + 2 || cache.control_changes != changes + 2;
    }
#endif
    /* Flush-to-zero is not described by fegetround(). Prove the complete
       native control key notices it, with subnormal prior error accumulators. */
#if (defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))) || defined(__x86_64__) || defined(_M_X64)
    uint64_t saved_control;
#if defined(__aarch64__)
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved_control));
    uint64_t toggled_control = saved_control ^ (UINT64_C(1) << 24);
    __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(toggled_control) : "memory");
#else
    saved_control = _mm_getcsr();
    _mm_setcsr((unsigned)saved_control ^ 0x8000u);
#endif
    uint64_t prepared = cache.prepared, changes = cache.control_changes, control;
    bad |= reference_control_key(&control) || control == cache.control_key;
    memset(&full, 0, sizeof(full)); memset(&cached, 0, sizeof(cached));
    uint64_t tiny_bits = 1;
    memcpy(&full.abs_error, &tiny_bits, sizeof(tiny_bits));
    memcpy(&full.sq_error, &tiny_bits, sizeof(tiny_bits));
    memcpy(&full.kl, &tiny_bits, sizeof(tiny_bits));
    cached = full;
    bad |= measure(&full, r, r, 3, 1) ||
           measure_cached(&cached, r, r, 3, 1, &cache, 0) ||
           memcmp(&full, &cached, sizeof(full)) != 0 ||
           cache.prepared != prepared + 1 || cache.control_changes != changes + 1;
#if defined(__aarch64__)
    __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(saved_control) : "memory");
#else
    _mm_setcsr((unsigned)saved_control);
#endif
    memset(&full, 0, sizeof(full)); memset(&cached, 0, sizeof(cached));
    bad |= measure(&full, r, r, 3, 1) ||
           measure_cached(&cached, r, r, 3, 1, &cache, 0) ||
           memcmp(&full, &cached, sizeof(full)) != 0 ||
           cache.prepared != prepared + 2 || cache.control_changes != changes + 2;
#endif
    FILE *f = tmpfile();
    if (!f) { reference_cache_free(&cache); return 1; }
    struct identity id = {100, 1, 2, 3};
    int32_t tokens[] = {1, 2, 1};
    /* Empty input and malformed headers cannot retain prepared statistics. */
    bad |= cached_reference_header(f, 3, 2, 16, &id, tokens, &cache) == 0 ||
           cache.row[0].valid || cache.row[1].valid;
    rewind(f);
    uint64_t h = digest(digest(FNV_INIT, r, sizeof(r)), r, sizeof(r));
    bad |= header(f, 1, 3, 2, 16, &id, tokens) ||
           fwrite(r, sizeof(float), 3, f) != 3 ||
           fwrite(r, sizeof(float), 3, f) != 3 || footer(f, 1, h);
    rewind(f);
    memset(&cached, 0, sizeof(cached));
    bad |= measure_cached(&cached, r, r, 3, 1, &cache, 0);
    bad |= cached_reference_header(f, 3, 2, 17, &id, tokens, &cache) == 0 || cache.row[0].valid;
    rewind(f);
    float want[3];
    bad |= cached_reference_header(f, 3, 2, 16, &id, tokens, &cache) ||
           cached_reference_row(f, want, 3, &cache) ||
           cached_reference_row(f, want, 3, &cache) ||
           cached_reference_footer(f, h, &cache);
    memset(&cached, 0, sizeof(cached));
    bad |= measure_cached(&cached, r, r, 3, 1, &cache, 0);
    bad |= cached_reference_row(f, want, 3, &cache) == 0 || cache.row[0].valid;
    rewind(f);
    bad |= cached_reference_header(f, 3, 2, 16, &id, tokens, &cache) ||
           cached_reference_row(f, want, 3, &cache) ||
           cached_reference_row(f, want, 3, &cache);
    memset(&cached, 0, sizeof(cached));
    bad |= measure_cached(&cached, r, r, 3, 1, &cache, 0);
    bad |= cached_reference_footer(f, h + 1, &cache) == 0 || cache.row[0].valid;
    fclose(f);
    reference_cache_free(&cache);
    return bad != 0;
}

static int self_test(void)
{
    if (cache_self_test()) return 1;
    uint32_t widths[MAX_BATCHES], count = 0;
    if (parse_batches("1,3,5,8,9,17", widths, &count) || count != 6 ||
        widths[0] != 1 || widths[5] != 17 ||
        parse_batches("", widths, &count) == 0 ||
        parse_batches("0,1", widths, &count) == 0 ||
        parse_batches("1,1", widths, &count) == 0 ||
        parse_batches("1,", widths, &count) == 0 ||
        parse_batches("1,,3", widths, &count) == 0 ||
        parse_batches("513", widths, &count) == 0 ||
        parse_batches("4294967297", widths, &count) == 0)
        return 1;
    float x[] = {1000, 1001, 999}, r[] = {998, 999, 997};
    struct metrics m = {0};
    double expected = log(exp(-1.0) + 1 + exp(-2.0));
    if (measure(&m, x, r, 3, 1) || fabs(m.nll - expected) > 1e-12 ||
        m.agree != 1 || fabs(m.sq_error - 12) > 1e-12 || fabs(m.kl) > 1e-12)
        return 1;
    x[0] = NAN;
    if (measure(&m, x, NULL, 3, 1) == 0)
        return 1;
    x[0] = INFINITY;
    if (measure(&m, x, NULL, 3, 1) == 0 || measure(&m, r, NULL, 3, 3) == 0)
        return 1;
    FILE *f = tmpfile();
    if (!f)
        return 1;
    struct identity id = {100, 1, 2, 3};
    int32_t tokens[] = {1, 2, 1};
    uint64_t h = digest(digest(FNV_INIT, r, sizeof r), r, sizeof r);
    int bad = header(f, 1, 3, 2, 16, &id, tokens) ||
              fwrite(r, sizeof(float), 3, f) != 3 ||
              fwrite(r, sizeof(float), 3, f) != 3 || footer(f, 1, h);
    rewind(f);
    float readback[3] = {0};
    bad |= header(f, 0, 3, 2, 16, &id, tokens) ||
           fread(readback, sizeof(float), 3, f) != 3;
    uint64_t rh = digest(FNV_INIT, readback, sizeof readback);
    bad |= fread(readback, sizeof(float), 3, f) != 3 ||
           footer(f, 0, digest(rh, readback, sizeof readback));
    /* Every matrix width must independently restart and authenticate the same
       immutable reference, including its complete logit digest and EOF. */
    for (unsigned width = 0; width < 6; width++) {
        rewind(f);
        bad |= header(f, 0, 3, 2, 16, &id, tokens) ||
               fread(readback, sizeof(float), 3, f) != 3;
        rh = digest(FNV_INIT, readback, sizeof readback);
        bad |= fread(readback, sizeof(float), 3, f) != 3 ||
               footer(f, 0, digest(rh, readback, sizeof readback));
    }
    bad |= fseek(f, 64 + 3 * 4 + 2 * 3 * 4, SEEK_SET) != 0 ||
           footer(f, 0, h + 1) == 0; /* corrupted logit digest */
    rewind(f);
    tokens[0] = 2;
    bad |= header(f, 0, 3, 2, 16, &id, tokens) == 0;
    tokens[0] = 1;
    rewind(f);
    bad |= header(f, 0, 3, 2, 17, &id, tokens) == 0;
    rewind(f);
    id.header++;
    bad |= header(f, 0, 3, 2, 16, &id, tokens) == 0;
    rewind(f);
    bad |= footer(f, 0, h) == 0;
    fclose(f);
    f = tmpfile();
    if (!f)
        return 1;
    bad |= footer(f, 0, h) == 0; /* truncated footer */
    bad |= header(f, 1, 1000000, MAX_STEPS, 1024, &id, tokens) == 0;
    fclose(f);
    if (!bad)
        puts("quality_check self-test: ok");
    return bad != 0;
}

#ifndef QUALITY_CHECK_SELFTEST_ONLY
/* Authored for this public probe; plain text, no model-specific chat template.
 */
static const char corpus[] =
    "The Earth orbits the Sun. Seven plus five is twelve. In Python, "
    "sum([1, 2, 3]) returns six. A library lends books to readers.\n"
    "A small library opens at nine each morning. The librarian places new books "
    "on a wooden table near the window. Readers return borrowed books before "
    "choosing another story.\n"
    "Water freezes at zero degrees Celsius at standard atmospheric pressure. "
    "The Earth orbits the Sun. A triangle has three sides.\n"
    "If a train travels 60 kilometers in one hour, its average speed is "
    "60 kilometers per hour. Seven plus five is twelve. Three times four "
    "is also twelve.\n"
    "The following Python function adds the integers from one through n:\n"
    "def triangular(n):\n    return n * (n + 1) // 2\n"
    "For n equal to four, the function returns ten.\n"
    "A careful experiment changes one condition at a time. The observer writes "
    "down the method, the measurements, and the uncertainty. Another person "
    "can repeat the experiment and compare the results.\n"
    "The garden contains red flowers, green leaves, and a narrow stone path. "
    "After the rain, drops of water remain on the leaves. A bird lands on "
    "the fence, pauses, and flies toward the trees.\n";

static void usage(void)
{
    fputs(
        "usage: quality_check model.jns [--profile cpu|gpu|npu|gpu+npu]\n"
        "  [--steps 64] [--batch 1 | --batches 1,3,5,8,9,17]\n"
        "  [--cache-mib 16384] [--context 1024]\n"
        "  [--threads N] [--tokens prompt.int32]\n"
        "  [--write-reference cpu.qref | --reference cpu.qref]\n"
        "  quality_check --self-test\n"
        "JSON metrics have no quality pass threshold. No reference = not_compared.\n"
        "Exit 3: metrics recorded, requested hardware execution unverified.\n",
        stderr);
}

static int number(const char *s, uint32_t min, uint32_t max, uint32_t *v)
{
    char *end;
    errno = 0;
    unsigned long n = strtoul(s, &end, 10);
    if (!*s || *s == '-' || *end || errno || n < min || n > max)
        return -1;
    *v = (uint32_t)n;
    return 0;
}

static void print_perplexity(double mean_nll)
{
    double p = exp(mean_nll);
    if (isfinite(p))
        printf("%.12g", p);
    else
        fputs("null", stdout); /* JSON has no infinity literal. */
}

static double probe_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_reference_cache(const struct reference_stats_cache *cache,
                                     const uint64_t start[5])
{
    uint64_t bytes = cache->rows ? (uint64_t)cache->nv * cache->steps * sizeof(float) +
                                  sizeof(*cache) : 0;
    printf(",\"reference_stats_cache_bytes\":%" PRIu64
           ",\"reference_stats_cache_hits\":%" PRIu64
           ",\"reference_stats_cache_prepared\":%" PRIu64
           ",\"reference_stats_cache_unequal_rows\":%" PRIu64
           ",\"reference_stats_cache_nonfinite_rows\":%" PRIu64
           ",\"reference_stats_cache_control_changes\":%" PRIu64,
           bytes, cache->hits - start[0], cache->prepared - start[1],
           cache->unequal - start[2], cache->nonfinite - start[3],
           cache->control_changes - start[4]);
}

static struct janas_llm_accel_counters
accel_delta(const struct janas_llm_accel_counters *end,
            const struct janas_llm_accel_counters *start)
{
    return (struct janas_llm_accel_counters){
        .npu_products = end->npu_products - start->npu_products,
        .npu_ffns = end->npu_ffns - start->npu_ffns,
        .npu_layer_passes = end->npu_layer_passes - start->npu_layer_passes,
        .npu_prepared_plans =
            end->npu_prepared_plans - start->npu_prepared_plans,
        .npu_weight_bytes = end->npu_weight_bytes,
        .gpu_product_submissions =
            end->gpu_product_submissions - start->gpu_product_submissions,
        .gpu_products = end->gpu_products - start->gpu_products,
        .gpu_whole_passes = end->gpu_whole_passes - start->gpu_whole_passes};
}

static void print_accel_counters(const struct janas_llm_accel_counters *s)
{
    printf(
        ",\"accelerator_counters\":{\"count_scope\":\"probe_delta\","
        "\"npu_products\":%" PRIu64 ",\"npu_ffns\":%" PRIu64 ","
        "\"npu_layer_passes\":%" PRIu64 ",\"npu_prepared_plans\":%" PRIu64 ","
        "\"npu_weight_bytes\":%" PRIu64 ","
        "\"npu_weight_bytes_scope\":\"current_unique_resident\","
        "\"gpu_product_submissions\":%" PRIu64 ",\"gpu_products\":%" PRIu64 ","
        "\"gpu_whole_passes\":%" PRIu64 "}",
        s->npu_products, s->npu_ffns, s->npu_layer_passes,
        s->npu_prepared_plans, s->npu_weight_bytes, s->gpu_product_submissions,
        s->gpu_products, s->gpu_whole_passes);
}

static int completed_devices(const struct janas_llm_accel_counters *s)
{
    int devices = 0;
    if (s->gpu_products || s->gpu_whole_passes)
        devices |= JANAS_DEV_GPU;
    if (s->npu_products || s->npu_ffns || s->npu_layer_passes)
        devices |= JANAS_DEV_NPU;
    return devices; /* preparation and submission alone are not execution proof
                     */
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--self-test"))
        return self_test();
    if (argc < 2 || !strcmp(argv[1], "--help")) {
        usage();
        return argc < 2 ? 2 : 0;
    }
    const char *profile = "cpu", *input = NULL, *ref_path = NULL, *out = NULL;
    uint32_t steps = 64, batch = 1, cache_mib = 16384, context = 1024,
             threads = 0;
    uint32_t batches[MAX_BATCHES] = {1}, n_batches = 1;
    int matrix = 0, single_batch_seen = 0;
    double process_started = probe_now(), load_seconds = 0;
    for (int i = 2; i < argc; i++) {
        if (i + 1 == argc) {
            usage();
            return 2;
        }
        const char *arg = argv[i], *v = argv[++i];
        int bad = 0;
        if (!strcmp(arg, "--profile"))
            profile = v;
        else if (!strcmp(arg, "--tokens"))
            input = v;
        else if (!strcmp(arg, "--reference"))
            ref_path = v;
        else if (!strcmp(arg, "--write-reference"))
            out = v;
        else if (!strcmp(arg, "--steps"))
            bad = number(v, 1, MAX_STEPS, &steps);
        else if (!strcmp(arg, "--batch")) {
            single_batch_seen = 1;
            bad = number(v, 1, MAX_STEPS, &batch);
        } else if (!strcmp(arg, "--batches")) {
            bad = matrix || parse_batches(v, batches, &n_batches);
            matrix = 1;
        } else if (!strcmp(arg, "--cache-mib"))
            bad = number(v, 1, 1048576, &cache_mib);
        else if (!strcmp(arg, "--context"))
            bad = number(v, 2, 65536, &context);
        else if (!strcmp(arg, "--threads"))
            bad = number(v, 1, 255, &threads);
        else
            bad = 1;
        if (bad) {
            fprintf(stderr, "invalid option/value: %s %s\n", arg, v);
            return 2;
        }
    }
    int gpu = !strcmp(profile, "gpu") || !strcmp(profile, "gpu+npu");
    int npu = !strcmp(profile, "npu") || !strcmp(profile, "gpu+npu");
    if (matrix && (single_batch_seen || gpu || npu || !ref_path || out)) {
        fputs("--batches requires a CPU reference comparison and excludes --batch/--write-reference\n",
              stderr);
        return 2;
    }
    if (!matrix) batches[0] = batch;
    uint32_t max_batch = 0;
    for (uint32_t i = 0; i < n_batches; i++)
        if (batches[i] > max_batch) max_batch = batches[i];
    if ((!gpu && !npu && strcmp(profile, "cpu")) || context <= steps ||
        (out && (ref_path || strcmp(profile, "cpu")))) {
        fputs("invalid profile/context; only a CPU run may write a reference\n",
              stderr);
        return 2;
    }
    char err[256] = {0};
    const char *vars[] = {"JANAS_EXPERT_BITS",  "JANAS_HEAD_BITS",
                          "JANAS_MISS_SKIP",    "JANAS_ATTN",
                          "JANAS_WARM",         "JANAS_GPU",
                          "JANAS_NPU",          "JANAS_TUNE",
                          "JANAS_TUNE_THREADS", "JANAS_GPU_MIN_N"};
    const char *vals[] = {
        "6", "6", "0", "float", "0", gpu ? "1" : "0", npu ? "1" : "0",
        "0", "0", "1"};
    for (size_t i = 0; i < sizeof vars / sizeof *vars; i++)
        if (setenv(vars[i], vals[i], 1)) {
            perror(vars[i]);
            return 1;
        }
    for (int i = 0; i < JANAS_DEVICES_OPTIONS; i++)
        if (janas_devices_option(janas_devices_options[i], profile, err,
                                 sizeof err) < 0) {
            fprintf(stderr, "%s\n", err);
            return 2;
        }
    struct janas_jns j;
    if (janas_jns_open(argv[1], &j, err, sizeof err)) {
        fprintf(stderr, "quality_check: model metadata: %s\n",
                err[0] ? err : "open failed without an engine error message");
        return 1;
    }
    struct identity id = {j.h.file_bytes, j.h.header_checksum,
                          j.h.tables_checksum, j.h.metadata_checksum};
    struct janas_tokenizer *tok = janas_tokenizer_create(&j, err, sizeof err);
    janas_jns_close(&j);
    if (!tok) {
        fprintf(stderr, "quality_check: tokenizer: %s\n",
                err[0] ? err
                       : "creation failed without an engine error message");
        return 1;
    }
    uint32_t nv = janas_tokenizer_n_vocab(tok);
    int32_t tokens[MAX_STEPS + 1];
    int bad = 0;
    if (input) {
        FILE *f = fopen(input, "rb");
        if (!f) {
            fprintf(stderr, "quality_check: tokens %s: %s\n", input,
                    strerror(errno));
            bad = 1;
        } else {
            bad = fread(tokens, sizeof(*tokens), steps + 1, f) != steps + 1;
            fclose(f);
        }
    } else {
        long n = janas_tokenizer_encode(tok, corpus, sizeof corpus - 1, 0,
                                        tokens, steps + 1);
        bad = n < (long)steps + 1;
    }
    janas_tokenizer_destroy(tok);
    for (uint32_t i = 0; i <= steps && !bad; i++)
        bad = tokens[i] < 0 || (uint32_t)tokens[i] >= nv;
    if (bad || (uint64_t)steps * nv * 4 > MAX_REFERENCE_BYTES) {
        fputs("invalid/short token input or reference would exceed 256 MiB\n",
              stderr);
        return 2;
    }
    FILE *ref = NULL, *dump = NULL;
    if (ref_path && (!(ref = fopen(ref_path, "rb")) ||
                     header(ref, 0, nv, steps, context, &id, tokens))) {
        fputs(
            "reference missing/incompatible (model, tokens, context or shape)\n",
            stderr);
        if (ref)
            fclose(ref);
        return 1;
    }
    if (ref) {
        long start = ftell(ref);
        uint64_t bytes =
            64 + (uint64_t)(steps + 1) * 4 + (uint64_t)steps * nv * 4 + 16;
        if (start < 0 || fseek(ref, 0, SEEK_END) || ftell(ref) != (long)bytes ||
            fseek(ref, start, SEEK_SET)) {
            fputs("reference has an invalid length\n", stderr);
            fclose(ref);
            return 1;
        }
    }
    if (out) {
        /* A completed reference is never overwritten by a failed rerun. */
        FILE *existing = fopen(out, "rb");
        if (existing) {
            fclose(existing);
            fputs("reference output already exists\n", stderr);
            return 1;
        }
        dump = fopen(out, "wb");
        if (!dump || header(dump, 1, nv, steps, context, &id, tokens)) {
            if (dump)
                fclose(dump);
            remove(out);
            fputs("cannot write reference header\n", stderr);
            return 1;
        }
    }
    struct janas_cpu_layout lay;
    int split_compute[JANAS_MAX_CPUS], io_cpus[JANAS_MAX_CPUS];
    int split_n_compute = 0, n_io = 0;
    struct janas_llm_model *m = NULL;
    float *logits = NULL, *want = NULL;
    struct reference_stats_cache reference_cache = {0};
    int rc = 1;
    if (janas_cpu_layout(NULL, 0, &lay) ||
        (threads && threads > (uint32_t)lay.n)) {
        fputs("invalid/unavailable CPU layout\n", stderr);
        goto done;
    }
    /* Match janas_llm_open: an I/O pool is required even on a CPU-only run.
       A zero-initialized n_io makes model_load fail without an error string. */
    int split = janas_cpu_split(split_compute, &split_n_compute, io_cpus, &n_io,
                                JANAS_MAX_CPUS);
    const int *io_pin = split > 0 && n_io > 0 ? io_cpus : NULL;
    if (split < 0 || n_io < 1)
        n_io = 4;
    if (n_io > 255)
        n_io = 255; /* janas_pool_create's upper bound */
    int n_compute = threads ? (int)threads : lay.n;
    if (n_compute > 255)
        n_compute = 255;
    /* A saved tuner choice or inherited block-thread override must not
       change the thread count while comparing numerical profiles. */
    char block_threads[16];
    snprintf(block_threads, sizeof block_threads, "%d", n_compute);
    if (setenv("JANAS_BLOCK_THREADS", block_threads, 1)) {
        perror("JANAS_BLOCK_THREADS");
        goto done;
    }
    struct janas_llm_options o = {.cache_bytes = (uint64_t)cache_mib << 20,
                                  .n_ctx = context,
                                  .compute_cpus = lay.cpus,
                                  .n_compute = n_compute,
                                  .io_cpus = io_pin,
                                  .n_io = n_io,
                                  .use_gpu = gpu,
                                  .expert_bits = 6,
                                  .head_bits = 6,
                                  .min_quality = JANAS_LLM_QUALITY_HIGH,
                                  .attn_scores = JANAS_LLM_ATTN_EXACT,
                                  .no_npu = !npu,
                                  .min_block = max_batch};
    fprintf(stderr,
            "quality_check: loading model; compute threads %d, I/O threads %d, "
            "cache %u MiB, context %u, profile %s\n",
            o.n_compute, o.n_io, cache_mib, context, profile);
    fflush(stderr);
    err[0] = 0;
    double load_started = probe_now();
    m = janas_llm_model_load(argv[1], &o, err, sizeof err);
    load_seconds = probe_now() - load_started;
    if (!m) {
        fprintf(stderr,
                "quality_check: model load failed: %s "
                "(compute threads %d, I/O threads %d)\n",
                err[0] ? err : "engine returned NULL without an error message",
                o.n_compute, o.n_io);
        goto done;
    }
    fputs("quality_check: model loaded; checking original-weight settings\n",
          stderr);
    janas_llm_model_set_experts(m, 0);
    uint32_t full_k, k = janas_llm_model_experts(m, &full_k);
    uint64_t before, after;
    janas_llm_model_head_requant(m, &before, &after);
    int bits = janas_llm_model_expert_bits(m);
    if (k != full_k || (full_k && bits != 6) || before || after ||
        janas_llm_model_miss_skip(m) != 0 ||
        janas_llm_model_head16(m, &before, &after) ||
        janas_llm_model_hc_quant(m, &before, &after) ||
        janas_llm_model_n_vocab(m) != nv ||
        max_batch > janas_llm_model_max_block(m)) {
        fputs("model settings do not preserve the original weights/experts\n",
              stderr);
        goto done;
    }
    logits = malloc((size_t)max_batch * nv * sizeof(float));
    want = ref ? malloc((size_t)nv * sizeof(float)) : NULL;
    if (!logits || (ref && !want)) {
        fputs("logit buffer allocation failed\n", stderr);
        goto done;
    }
    if (matrix && reference_cache_init(&reference_cache, nv, steps))
        fputs("quality_check: bounded reference-stat cache unavailable; using full comparisons\n", stderr);
    double setup_seconds = probe_now() - process_started;
    rc = 0;
    for (uint32_t matrix_index = 0; matrix_index < n_batches; matrix_index++) {
        batch = batches[matrix_index];
        int width_rc = 1;
        const char *width_error = "incomplete_width";
        double width_started = probe_now();
        uint64_t cache_start[] = {reference_cache.hits, reference_cache.prepared,
                                   reference_cache.unequal, reference_cache.nonfinite,
                                   reference_cache.control_changes};
        struct metrics s = {0};
        uint32_t actual_batches[MAX_STEPS], actual_count = 0;
        if (matrix) {
            /* Position zero resets recurrent state and overwrites the live causal
               KV prefix. Only immutable weights and expert-cache warmth are shared. */
            fprintf(stderr, "quality_check: matrix width %u (%u/%u), fresh sequence\n",
                    batch, matrix_index + 1, n_batches);
            clearerr(ref);
            if (fseek(ref, 0, SEEK_SET) ||
                cached_reference_header(ref, nv, steps, context, &id, tokens,
                                         &reference_cache)) {
                width_error = "reference_restart_failed";
                fputs("reference restart failed\n", stderr);
                goto width_done;
            }
        }
        struct janas_llm_accel_counters accel0, accel1;
        janas_llm_model_accel_counters(m, &accel0);
        uint64_t h = FNV_INIT, rh = FNV_INIT;
        for (uint32_t p = 0; p < steps;) {
            uint32_t n = steps - p < batch ? steps - p : batch;
            if (janas_llm_model_forward(m, tokens + p, n, p, logits, 1)) {
                fprintf(stderr, "forward failed at %u\n", p);
                width_error = "forward_failed";
                goto width_done;
            }
            actual_batches[actual_count++] = n;
            for (uint32_t i = 0; i < n; i++) {
                float *row = logits + (size_t)i * nv;
                if (ref && cached_reference_row(ref, want, nv, &reference_cache)) {
                    fputs("truncated reference logits\n", stderr);
                    width_error = "truncated_reference_logits";
                    goto width_done;
                }
                if (measure_cached(&s, row, want, nv, (uint32_t)tokens[p + i + 1],
                                     &reference_cache, p + i)) {
                    fprintf(stderr, "nonfinite logits at %u\n", p + i);
                    width_error = "nonfinite_logits";
                    goto width_done;
                }
                h = digest(h, row, (size_t)nv * sizeof(float));
                if (ref)
                    rh = digest(rh, want, (size_t)nv * sizeof(float));
                if (dump && fwrite(row, sizeof(float), nv, dump) != nv) {
                    fputs("cannot write reference logits\n", stderr);
                    width_error = "reference_write_failed";
                    goto width_done;
                }
            }
            p += n;
            fprintf(stderr, "quality_check: %u/%u teacher-forced positions\n", p,
                    steps);
        }
        janas_llm_model_accel_counters(m, &accel1);
        struct janas_llm_accel_counters executed = accel_delta(&accel1, &accel0);
        if ((ref && cached_reference_footer(ref, rh, &reference_cache)) ||
            (dump && footer(dump, 1, h))) {
            fputs("reference incomplete/corrupt or write failed\n", stderr);
            width_error = "reference_footer_failed";
            goto width_done;
        }
        if (dump) {
            bad = fclose(dump) != 0;
            dump = NULL;
            if (bad) {
                fputs("cannot finish reference\n", stderr);
                width_error = "reference_close_failed";
                goto width_done;
            }
        }
        int used_gpu = janas_llm_model_gpu_used(m),
            used_npu = janas_llm_model_npu_used(m);
        const int actual_devices = completed_devices(&executed);
        const int actual_gpu = (actual_devices & JANAS_DEV_GPU) != 0;
        const int actual_npu = (actual_devices & JANAS_DEV_NPU) != 0;
        const int verified = gpu == actual_gpu && npu == actual_npu;
        const char *effective = actual_gpu   ? (actual_npu ? "gpu+npu" : "gpu")
                                : actual_npu ? "npu"
                                             : "cpu";
        size_t gpu_min_tokens =
            janas_llm_model_gpu(m) ? janas_gpu_min_n(janas_llm_model_gpu(m)) : 0;
        if (!verified) {
            fprintf(
                stderr,
                "quality_check: hardware execution UNVERIFIED for requested %s; "
                "observed %s%s. GPU completed products/whole passes %" PRIu64
                "/%" PRIu64 ", NPU products/FFNs/layer passes %" PRIu64 "/%" PRIu64
                "/%" PRIu64 ". Metrics describe observed execution only.\n",
                profile, effective,
                !actual_gpu && !actual_npu ? " (CPU fallback)" : "",
                executed.gpu_products, executed.gpu_whole_passes,
                executed.npu_products, executed.npu_ffns,
                executed.npu_layer_passes);
        }
        printf(
            "{\"profile\":\"%s\",\"profile_effective\":\"%s\","
            "\"hardware_execution_verified\":%s,\"comparison\":\"%s\",\"positions\":%u,"
            "\"batch\":%u,\"context\":%u,\"cache_mib\":%u,\"threads\":%d,\"io_threads\":%d,"
            "\"vocab\":%u,\"finite_logits\":true,\"expert_bits\":%d,"
            "\"experts\":%u,\"gpu_selected\":%d,\"npu_selected\":%d,\"gpu_min_tokens\":%zu,"
            "\"device_masks\":[%d,%d,%d,%d],\"min_quality\":2,\"tuning\":false,"
            "\"expert_skip\":false,\"head_requantized\":false,"
            "\"reference_written\":%s,\"model_header_fnv1a64\":\"%016" PRIx64 "\","
            "\"token_fnv1a64\":\"%016" PRIx64 "\",\"logit_fnv1a64\":\"%016" PRIx64
            "\",\"nll\":%.12g,\"perplexity\":",
            profile, effective, verified ? "true" : "false",
            ref ? (verified ? "compared_to_cpu"
                            : "unverified_profile_compared_to_cpu")
                : "not_compared",
            steps, batch, context, cache_mib, o.n_compute, o.n_io, nv, bits, k,
            used_gpu, used_npu, gpu_min_tokens, janas_llm_model_devices(m, 0),
            janas_llm_model_devices(m, 1), janas_llm_model_devices(m, 2),
            janas_llm_model_devices(m, 3), out ? "true" : "false", id.header,
            digest(FNV_INIT, tokens, (steps + 1) * sizeof(*tokens)), h,
            s.nll / s.rows);
        print_perplexity(s.nll / s.rows);
        print_accel_counters(&executed);
        if (ref) {
            printf(",\"reference_nll\":%.12g,\"reference_perplexity\":",
                   s.ref_nll / s.rows);
            print_perplexity(s.ref_nll / s.rows);
            printf(",\"nll_delta\":%.12g,\"top1_agreement\":%.12g,"
                   "\"logit_mae\":%.12g,\"logit_rmse\":%.12g,"
                   "\"logit_max_abs\":%.12g,\"mean_kl_ref_to_candidate\":%.12g",
                   (s.nll - s.ref_nll) / s.rows, (double)s.agree / s.rows,
                   s.abs_error / s.values, sqrt(s.sq_error / s.values), s.max_error,
                   fmax(0, s.kl / s.rows));
        }
        width_rc = verified ? 0 : 3;
        if (matrix) {
            printf(",\"completed\":true,\"exit_code\":%d,\"matrix_index\":%u,"
                   "\"matrix_count\":%u,\"max_batch_capacity\":%u,"
                   "\"model_load_count\":1,\"functional_only\":true,"
                   "\"sequence_restart\":\"forward_position_zero\","
                   "\"model_load_seconds\":%.9g,\"setup_seconds\":%.9g,"
                   "\"width_seconds\":%.9g,\"actual_batches\":[",
                   width_rc, matrix_index, n_batches, max_batch,
                   load_seconds, setup_seconds, probe_now() - width_started);
            for (uint32_t i = 0; i < actual_count; i++)
                printf("%s%u", i ? "," : "", actual_batches[i]);
            putchar(']');
            print_reference_cache(&reference_cache, cache_start);
        }
        puts("}");
    width_done:
        if (width_rc == 1) reference_cache_invalidate(&reference_cache);
        if (matrix && width_rc == 1) {
            printf("{\"batch\":%u,\"positions\":%" PRIu64 ",\"expected_positions\":%u,"
                   "\"completed\":false,\"exit_code\":1,\"error\":\"%s\","
                   "\"matrix_index\":%u,\"matrix_count\":%u,"
                   "\"max_batch_capacity\":%u,\"model_load_count\":1,"
                   "\"functional_only\":true,\"sequence_restart\":\"forward_position_zero\","
                   "\"model_load_seconds\":%.9g,\"setup_seconds\":%.9g,"
                   "\"width_seconds\":%.9g,\"actual_batches\":[",
                   batch, s.rows, steps, width_error, matrix_index, n_batches, max_batch,
                   load_seconds, setup_seconds, probe_now() - width_started);
            for (uint32_t i = 0; i < actual_count; i++)
                printf("%s%u", i ? "," : "", actual_batches[i]);
            putchar(']');
            print_reference_cache(&reference_cache, cache_start);
            puts("}");
        }
        fflush(stdout);
        if (width_rc == 1 || (width_rc == 3 && !rc)) rc = width_rc;
    }
done:
    reference_cache_free(&reference_cache);
    free(logits);
    free(want);
    if (m)
        janas_llm_model_free(m);
    if (ref)
        fclose(ref);
    if (dump)
        fclose(dump);
    if (rc && out)
        remove(out); /* never leave an incomplete reference looking complete */
    return rc;
}
#else
int main(void)
{
    return self_test();
}
#endif
