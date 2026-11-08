/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * llm_eval.c - run token ids through the Janas-LLM forward pass and compare
 * the logits with a reference (llama.cpp dump: float32 x n_tokens x n_vocab).
 *
 * Usage: llm_eval <model.jns> <tokens.int32> [reference.logits]
 *                 [cache_MiB] [compute_cpus] [io_cpus] [block=B] [reject=R]
 *                 [dump=F] [bdump=F] [strict=1]
 * With block=B the tokens are also run in blocks of B (the prefill path) and
 * the logits compared with the token-by-token ones: they must be identical.
 * With reject=R (R < B) each block but the last keeps only its first B - R
 * tokens and the next starts after them, as when the last R tokens of a
 * chain of drafts are rejected: a recurrent model then replays the kept
 * tokens of its log on the state before the block.
 * With dump=F the logits are written to F in the reference format; with
 * bdump=F (and block=B) those of the blocks. Block/rejection checks cover
 * the first 256 input tokens and fail on any unequal logit bit pattern.
 * External references are compared numerically by default; strict=1 also
 * requires their bit patterns to match. Nonfinite logits always fail.
 * Defaults: 20480 MiB cache, compute CPUs 0..7, I/O CPUs 8..11. Override
 * both lists on machines with another topology; neither list may be empty.
 * Reports, per position, whether the most likely token agrees, the largest and
 * mean absolute logit difference, and the time per token.
 */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "janas/llm.h"
#include "llm/model.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int number(const char *s, uint64_t max, uint64_t *v)
{
    char *end;
    if (!s[0] || s[0] < '0' || s[0] > '9')
        return -1;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || *end || n > max)
        return -1;
    *v = (uint64_t)n;
    return 0;
}

static int parse_cpus(const char *s, int *cpus)
{
    int n = 0;
    while (*s) {
        char *end;
        if (*s < '0' || *s > '9' || n == 256)
            return -1;
        errno = 0;
        long v = strtol(s, &end, 10);
        if (errno || v > INT_MAX || (*end && *end != ','))
            return -1;
        for (int i = 0; i < n; i++)
            if (cpus[i] == v)
                return -1;
        cpus[n++] = (int)v;
        if (!*end)
            break;
        s = end + 1;
        if (!*s)
            return -1;
    }
    return n > 0 ? n : -1;
}

static int finite_row(const float *row, uint32_t nv, const char *what,
                      size_t pos)
{
    for (uint32_t i = 0; i < nv; i++)
        if (!isfinite(row[i])) {
            fprintf(stderr, "%s has nonfinite logit at position %zu, token %u\n",
                    what, pos, i);
            return -1;
        }
    return 0;
}

static int read_reference(FILE *f, float *row, uint32_t nv, size_t pos)
{
    if (fread(row, sizeof(float), nv, f) != nv) {
        fprintf(stderr, "cannot read complete reference row at position %zu%s\n",
                pos, ferror(f) ? " (I/O error)" : " (file is too short)");
        return -1;
    }
    return finite_row(row, nv, "reference", pos);
}

static int write_row(FILE *f, const float *row, uint32_t nv,
                     const char *what, size_t pos)
{
    if (f && fwrite(row, sizeof(float), nv, f) != nv) {
        fprintf(stderr, "cannot write %s row at position %zu\n", what, pos);
        return -1;
    }
    return 0;
}

static float *rows_alloc(size_t rows, uint32_t nv)
{
    if (!nv || rows > SIZE_MAX / sizeof(float) / nv)
        return NULL;
    return malloc(rows * nv * sizeof(float));
}

static void print_phases(const char *what, const double *a, const double *b,
                         size_t n)
{
    static const char *names[JANAS_PH_COUNT] = {
        "qkv", "attention", "wo", "router", "experts", "output", "ssm", "mix"};
    double sum = 0;
    printf("%s, ms/token:", what);
    for (int i = 0; i < JANAS_PH_COUNT; i++) {
        printf(" %s %.2f", names[i], (b[i] - a[i]) / n * 1e3);
        sum += b[i] - a[i];
    }
    printf(" (sum %.2f)\n", sum / n * 1e3);
}

static uint32_t argmax(const float *v, uint32_t n)
{
    uint32_t b = 0;
    for (uint32_t i = 1; i < n; i++)
        if (v[i] > v[b])
            b = i;
    return b;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: llm_eval <model.jns> <tokens> [reference] "
                        "[cache_MiB] [compute_cpus] [io_cpus] "
                        "[block=B] [reject=R] [dump=F] [bdump=F] [strict=1]\n");
        return 2;
    }
    const char *ref_path = NULL, *dump_path = NULL, *bdump_path = NULL;
    const char *compute = "0,1,2,3,4,5,6,7", *io = "8,9,10,11";
    uint64_t cache = 20480, block = 0, reject = 0, strict = 0;
    int positional = 0, block_set = 0, reject_set = 0;
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i];
        if (!strncmp(a, "block=", 6)) {
            if (number(a + 6, 4096, &block) || block == 0)
                goto bad_args;
            block_set = 1;
        } else if (!strncmp(a, "reject=", 7)) {
            if (number(a + 7, 4096, &reject))
                goto bad_args;
            reject_set = 1;
        } else if (!strncmp(a, "strict=", 7)) {
            if (number(a + 7, 1, &strict))
                goto bad_args;
        } else if (!strncmp(a, "dump=", 5) && a[5]) {
            dump_path = a + 5;
        } else if (!strncmp(a, "bdump=", 6) && a[6]) {
            bdump_path = a + 6;
        } else if (strchr(a, '=') || positional >= 4) {
            goto bad_args;
        } else {
            switch (positional++) {
            case 0: ref_path = strcmp(a, "-") ? a : NULL; break;
            case 1:
                if (number(a, UINT64_MAX >> 20, &cache))
                    goto bad_args;
                break;
            case 2: compute = a; break;
            case 3: io = a; break;
            }
        }
    }
    if ((reject_set && !block_set) || (bdump_path && !block_set) ||
        (strict && !ref_path) || (block_set && reject >= block))
        goto bad_args;
    if (reject && block > JANAS_LLM_MAX_BLOCK) {
        fprintf(stderr, "rejection replay supports block <= %d\n",
                JANAS_LLM_MAX_BLOCK);
        return 2;
    }
    int ccpus[256], icpus[256];
    int nc = parse_cpus(compute, ccpus), ni = parse_cpus(io, icpus);
    if (nc < 1 || ni < 1) {
        fprintf(stderr, "compute_cpus and io_cpus need nonempty, distinct "
                        "nonnegative CPU IDs within each list\n");
        return 2;
    }
    if ((dump_path && (!strcmp(dump_path, argv[2]) ||
                       (ref_path && !strcmp(dump_path, ref_path)))) ||
        (bdump_path && (!strcmp(bdump_path, argv[2]) ||
                        (ref_path && !strcmp(bdump_path, ref_path)) ||
                        (dump_path && !strcmp(bdump_path, dump_path))))) {
        fprintf(stderr, "dump files must differ from inputs and each other\n");
        return 2;
    }
    int rc = 1;
    FILE *f = NULL, *ref = NULL, *dump = NULL, *bdump = NULL;
    struct janas_llm_model *m = NULL;
    float *logits = NULL, *want = NULL, *seq = NULL, *blk = NULL;
    f = fopen(argv[2], "rb");
    if (!f) {
        perror(argv[2]);
        goto done;
    }
    int32_t tokens[4096];
    size_t bytes = fread(tokens, 1, sizeof(tokens), f);
    int extra = fgetc(f);
    if (ferror(f) || extra != EOF || !bytes || bytes % sizeof(int32_t)) {
        fprintf(stderr, "%s: expected 1..4096 complete int32 tokens\n", argv[2]);
        goto done;
    }
    size_t n = bytes / sizeof(int32_t);
    if (fclose(f)) {
        f = NULL;
        perror(argv[2]);
        goto done;
    }
    f = NULL;
    if (ref_path && !(ref = fopen(ref_path, "rb"))) {
        perror(ref_path);
        goto done;
    }
    struct janas_llm_options o = {
        .cache_bytes = cache << 20,
        .n_ctx = 4096,
        .compute_cpus = ccpus,
        .n_compute = nc,
        .io_cpus = icpus,
        .n_io = ni,
        .expert_bits = 6,
        .head_bits = 0,
        .min_quality = JANAS_LLM_QUALITY_HIGH};
    char err[256] = {0};
    m = janas_llm_model_load(argv[1], &o, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s (compute CPUs %s; I/O CPUs %s)\n",
                argv[1], err[0] ? err : "model load failed", compute, io);
        goto done;
    }
    if (block > janas_llm_model_max_block(m)) {
        fprintf(stderr, "block=%llu exceeds model maximum %u\n",
                (unsigned long long)block, janas_llm_model_max_block(m));
        rc = 2;
        goto done;
    }
    if (dump_path && !(dump = fopen(dump_path, "wb"))) {
        perror(dump_path);
        goto done;
    }
    if (bdump_path && !(bdump = fopen(bdump_path, "wb"))) {
        perror(bdump_path);
        goto done;
    }
    uint32_t nv = janas_llm_model_n_vocab(m);
    logits = rows_alloc(1, nv);
    want = rows_alloc(1, nv);
    if (!logits || !want) {
        fprintf(stderr, "cannot allocate logits for vocabulary %u\n", nv);
        goto done;
    }
    size_t row_bytes = (size_t)nv * sizeof(float);
    size_t agree = 0, compared = 0;
    double worst = 0, mean_sum = 0, t0 = now(), t_half = t0;
    double ph0[JANAS_PH_COUNT] = {0};
    for (size_t p = 0; p < n; p++) {
        if (p == n / 2) {
            t_half = now();
            janas_llm_model_phases(m, ph0);
        }
        if (janas_llm_model_decode(m, tokens[p], (uint32_t)p, logits) != 0) {
            fprintf(stderr, "decode failed at %zu\n", p);
            goto done;
        }
        if (finite_row(logits, nv, "decode", p) ||
            write_row(dump, logits, nv, "decode dump", p))
            goto done;
        if (ref) {
            if (read_reference(ref, want, nv, p))
                goto done;
            if (strict && memcmp(logits, want, row_bytes)) {
                fprintf(stderr, "strict reference mismatch at position %zu\n", p);
                goto done;
            }
            double mx = 0, sum = 0;
            for (uint32_t i = 0; i < nv; i++) {
                double d = fabs((double)logits[i] - want[i]);
                sum += d;
                if (d > mx)
                    mx = d;
            }
            uint32_t a = argmax(logits, nv), b = argmax(want, nv);
            agree += a == b;
            compared++;
            mean_sum += sum / nv;
            if (mx > worst)
                worst = mx;
            if (p < 8 || a != b)
                printf("pos %3zu: top-1 %6u vs %6u %s, max |diff| %.3f, mean "
                       "%.4f\n",
                       p, a, b, a == b ? "ok  " : "DIFF", mx, sum / nv);
        }
    }
    double dt = now() - t0;
    struct janas_expert_cache_stats st;
    janas_expert_cache_stats(janas_llm_model_cache(m), &st);
    double t2 = now() - t_half, ph_end[JANAS_PH_COUNT];
    janas_llm_model_phases(m, ph_end);
    printf("%zu tokens in %.1f s (%.1f ms/token; second half %.1f ms/token = "
           "%.1f token/s), cold cache: %.1f%% of experts from RAM\n",
           n, dt, dt / n * 1e3, t2 / (n - n / 2) * 1e3, (n - n / 2) / t2,
           100.0 * st.hits / (st.requests ? st.requests : 1));
    if (block > 0) {
        /* token by token, keeping every logit row, then the same in blocks */
        size_t nn = n < 256 ? n : 256;
        seq = rows_alloc(nn, nv);
        blk = rows_alloc((size_t)block, nv);
        if (!seq || !blk) {
            fprintf(stderr, "cannot allocate block comparison logits\n");
            goto done;
        }
        for (size_t p = 0; p < nn; p++) {
            if (janas_llm_model_decode(m, tokens[p], (uint32_t)p, seq + p * nv)) {
                fprintf(stderr, "self-reference decode failed at %zu\n", p);
                goto done;
            }
            if (finite_row(seq + p * nv, nv, "self-reference decode", p))
                goto done;
        }
        /* Retained rows are consumed in order, even after rejection. Rewind
           once instead of using long-sized byte offsets on Windows. */
        if (ref && fseek(ref, 0, SEEK_SET)) {
            fprintf(stderr, "cannot rewind reference for block comparison\n");
            goto done;
        }
        double pb0[JANAS_PH_COUNT], pb1[JANAS_PH_COUNT];
        janas_llm_model_phases(m, pb0);
        double tb = now(), maxd = 0, sumd = 0, rsum = 0;
        size_t same = 0, top = 0, rtop = 0, passes = 0, all_rows = 0;
        for (size_t p0 = 0, kept; p0 < nn; p0 += kept) {
            uint32_t b =
                (uint32_t)(nn - p0 < (size_t)block ? nn - p0 : (size_t)block);
            if (janas_llm_model_forward(m, tokens + p0, b, (uint32_t)p0, blk,
                                        1) != 0) {
                fprintf(stderr, "block forward failed at %zu\n", p0);
                goto done;
            }
            passes++;
            /* Check every computed row, including drafts later rejected. */
            for (uint32_t j = 0; j < b; j++) {
                const float *a = seq + (p0 + j) * nv;
                const float *c = blk + (size_t)j * nv;
                if (finite_row(c, nv, "block", p0 + j))
                    goto done;
                if (memcmp(a, c, row_bytes)) {
                    fprintf(stderr, "block/replay mismatch at position %zu "
                                    "(pass %zu, row %u)\n", p0 + j, passes, j);
                    goto done;
                }
                all_rows++;
            }
            /* the last block keeps all its tokens, the others b - reject:
               the positions after them come again in the next block */
            kept = p0 + b >= nn ? b : b - (uint32_t)reject;
            for (uint32_t j = 0; j < kept; j++) {
                const float *a = seq + (p0 + j) * nv, *c = blk + (size_t)j * nv;
                if (write_row(bdump, c, nv, "block dump", p0 + j))
                    goto done;
                int eq = 1;
                for (uint32_t i = 0; i < nv; i++) {
                    double d = fabs((double)a[i] - c[i]);
                    if (d > maxd)
                        maxd = d;
                    sumd += d;
                    eq &= a[i] == c[i];
                }
                same += eq;
                top += argmax(a, nv) == argmax(c, nv);
                /* and against the reference, as the main loop compares */
                if (ref) {
                    if (read_reference(ref, want, nv, p0 + j))
                        goto done;
                    if (strict && memcmp(c, want, row_bytes)) {
                        fprintf(stderr, "strict block reference mismatch at "
                                        "position %zu\n", p0 + j);
                        goto done;
                    }
                    for (uint32_t i = 0; i < nv; i++)
                        rsum += fabs((double)c[i] - want[i]);
                    rtop += argmax(c, nv) == argmax(want, nv);
                }
            }
        }
        double tbs = now() - tb;
        janas_llm_model_phases(m, pb1);
        print_phases("blocks", pb0, pb1, nn);
        if (reject)
            printf("blocks of %llu keeping %llu: %zu passes, the kept tokens "
                   "replayed by the next\n",
                   (unsigned long long)block,
                   (unsigned long long)(block - reject), passes);
        printf(
            "blocks of %llu: %zu/%zu positions bit-identical to token by token, "
            "max |diff| %g; %.1f ms/token in blocks (%.1f token/s)\n",
            (unsigned long long)block, same, nn, maxd, tbs / nn * 1e3, nn / tbs);
        printf("checked %zu block rows including rejected drafts; %zu/%zu "
               "input positions covered\n", all_rows, nn, n);
        printf("blocks of %llu: most likely token as token by token at %zu/%zu, "
               "mean |diff| %.4f\n",
               (unsigned long long)block, top, nn, sumd / ((double)nn * nv));
        if (ref)
            printf("blocks of %llu against the reference: top-1 %zu/%zu, mean "
                   "|diff| %.4f\n",
                   (unsigned long long)block, rtop, nn, rsum / ((double)nn * nv));
    }
    print_phases("second half", ph0, ph_end, n - n / 2);
    if (compared)
        printf("top-1 agreement %zu/%zu, max |diff| %.3f, mean |diff| %.4f\n",
               agree, compared, worst, mean_sum / compared);
    rc = 0;
done:
    free(logits);
    free(want);
    free(seq);
    free(blk);
    if (m)
        janas_llm_model_free(m);
    FILE *files[] = {f, ref, dump, bdump};
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++)
        if (files[i] && fclose(files[i])) {
            fprintf(stderr, "cannot close %s file\n",
                    i == 0 ? "token" : i == 1 ? "reference" : "dump");
            rc = 1;
        }
    if (fflush(stdout) || ferror(stdout))
        rc = 1;
    return rc;
bad_args:
    fprintf(stderr, "invalid arguments: block must be 1..4096, "
                    "0 <= reject < block, strict=1 needs a reference, "
                    "and cache_MiB must be nonnegative; use '-' for no reference\n");
    return 2;
}
