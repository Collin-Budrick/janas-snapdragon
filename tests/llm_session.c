/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * llm_session.c - checks sessions on a real model:
 *   1. turns: A, 12 generated, B, 24 generated, token by token, gives the
 *      same 24 as a fresh session given A + the 12 + B at once;
 *   2. the same with speculation on, which leaves generated tokens unread
 *      when B is appended (they must be dropped);
 *   3. sampling (temperature, top-p, seed): the same tokens with and
 *      without speculation.
 *
 * Usage: llm_session <model.jns> <tokens.int32> [mtp=<mtp.jns>]
 *   [planner=legacy|cycle|fixed] [speculation=mtp|hybrid] [drafts=1..16]
 *   [draft_vocab=legacy|adaptive] [rows=0|8192|16384|32768]
 *   [hybrid_policy=legacy|cost] [lookup_drafts=1..16] [mtp_drafts=1..16]
 *   [hybrid_probe_percent=0..100]
 */
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/sysinfo.h"
#include "janas/llm.h"
#include "../src/llm/generate.h"
#include "../src/llm/model.h"

#define NA 120
#define NB 40
#define G1 12
#define G2 24

static int unsigned_value(const char *text, unsigned *value)
{
    unsigned n = 0;
    if (!*text) return 0;
    for (; *text; text++) {
        if (*text < '0' || *text > '9' ||
            n > (UINT_MAX - (unsigned)(*text - '0')) / 10)
            return 0;
        n = n * 10 + (unsigned)(*text - '0');
    }
    *value = n;
    return 1;
}

static int run_turns(struct janas_llm_model *m,
                     const struct janas_gen_options *o, const int32_t *text,
                     int32_t *g1, int32_t *g2)
{
    struct janas_mtp_costs before, after;
    janas_llm_mtp_costs(m, &before);
    struct janas_llm_session *s = janas_llm_session_create(m, o);
    if (!s || janas_llm_session_append(s, text, NA) != 0)
        return -1;
    for (int i = 0; i < G1; i++)
        if (janas_llm_session_next(s, &g1[i]) != 0)
            return -1;
    if (janas_llm_session_append(s, text + NA, NB) != 0)
        return -1;
    for (int i = 0; i < G2; i++)
        if (janas_llm_session_next(s, &g2[i]) != 0)
            return -1;
    int ok = janas_llm_session_length(s) == NA + G1 + NB + G2;
    struct janas_gen_metrics metrics;
    janas_llm_session_metrics(s, &metrics);
    janas_llm_mtp_costs(m, &after);
    struct janas_gen_hybrid_metrics hybrid;
    janas_llm_session_hybrid_metrics(s, &hybrid);
    ok &= metrics.cycles > 0 && metrics.delivered_tokens == G1 + G2;
    ok &= metrics.last.resolved_planner == (uint32_t)o->spec_planner;
    if (o->spec_mode == JANAS_SPEC_HYBRID)
        ok &= metrics.lookup_hits + metrics.lookup_misses > 0;
    if (o->draft_vocab_policy && metrics.source_drafted[JANAS_GEN_SOURCE_MTP])
        ok &= after.repacks > before.repacks && after.head_calls > before.head_calls;
    unsigned max_width = 0;
    for (unsigned width = 1; width <= JANAS_GEN_MAX_K + 1; width++)
        if (metrics.pass_width_count[width]) max_width = width;
    printf("runtime: mode=%d planner=%u cycles=%" PRIu64
           " max_width=%u lookup_attempts=%" PRIu64 " mtp_drafts=%" PRIu64
           " head_calls=%" PRIu64 " repacks=%" PRIu64
           " head_policy=%u head_rows=%u head_epoch=%" PRIu64 " %s\n",
           o->spec_mode, metrics.last.resolved_planner, metrics.cycles, max_width,
           metrics.lookup_hits + metrics.lookup_misses,
           metrics.source_drafted[JANAS_GEN_SOURCE_MTP],
           after.head_calls - before.head_calls, after.repacks - before.repacks,
           after.vocab_policy, after.vocab_rows, after.vocab_epoch,
           ok ? "verified" : "FAILED");
    if (o->spec_mode == JANAS_SPEC_HYBRID && o->hybrid_policy == 1) {
        ok &= hybrid.decisions > 0 && hybrid.mtp_selected > 0;
        ok &= max_width <= (unsigned)o->spec_k + 1;
        for (unsigned width = (unsigned)o->hybrid_lookup_max_drafts + 1;
             width <= JANAS_GEN_MAX_K; width++)
            ok &= hybrid.lookup_depth_count[width] == 0;
        printf("hybrid: policy=cost decisions=%" PRIu64 " offered=%" PRIu64
               " lookup=%" PRIu64 " mtp=%" PRIu64 " probes=%" PRIu64
               " cooldown=%" PRIu64 " accepted=%" PRIu64 " %s\n",
               hybrid.decisions, hybrid.offered_matches, hybrid.lookup_selected,
               hybrid.mtp_selected, hybrid.probes, hybrid.cooldown_skips,
               hybrid.lookup_accepted, ok ? "verified" : "FAILED");
    }
    if (o->draft_vocab_policy && !metrics.source_drafted[JANAS_GEN_SOURCE_MTP])
        printf("adaptive head not exercised: no MTP proposal executed\n");
    janas_llm_session_destroy(s);
    return ok ? 0 : -1;
}

static int run_once(struct janas_llm_model *m,
                    const struct janas_gen_options *o, const int32_t *seq,
                    uint32_t n, int32_t *g, int count)
{
    struct janas_llm_session *s = janas_llm_session_create(m, o);
    if (!s || janas_llm_session_append(s, seq, n) != 0)
        return -1;
    for (int i = 0; i < count; i++)
        if (janas_llm_session_next(s, &g[i]) != 0)
            return -1;
    janas_llm_session_destroy(s);
    return 0;
}

static int same(const char *what, const int32_t *a, const int32_t *b, int n)
{
    int ok = memcmp(a, b, (size_t)n * sizeof(int32_t)) == 0;
    printf("%-44s %s\n", what, ok ? "identical" : "DIFFERENT");
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: llm_session <model.jns> <tokens.int32> "
                        "[mtp=<mtp.jns>] [planner=legacy|cycle|fixed] "
                        "[speculation=mtp|hybrid] [drafts=1..16] "
                        "[draft_vocab=legacy|adaptive] "
                        "[rows=0|8192|16384|32768] "
                        "[hybrid_policy=legacy|cost] [lookup_drafts=1..16] "
                        "[mtp_drafts=1..16] [hybrid_probe_percent=0..100]\n");
        return 2;
    }
    const char *mtp_path = NULL;
    int planner = 0, speculation = -1, vocab_policy = 0;
    unsigned drafts = 4, rows = 0, lookup_drafts = 8, mtp_drafts = 4, hybrid_probe_percent = 2;
    int hybrid_policy = 0;
    for (int i = 3; i < argc; i++) {
        const char *a = argv[i];
        if (!strncmp(a, "mtp=", 4) && a[4]) mtp_path = a + 4;
        else if (!strcmp(a, "planner=legacy")) planner = 0;
        else if (!strcmp(a, "planner=cycle")) planner = 1;
        else if (!strcmp(a, "planner=fixed")) planner = 2;
        else if (!strcmp(a, "speculation=mtp")) speculation = JANAS_SPEC_MTP;
        else if (!strcmp(a, "speculation=hybrid")) speculation = JANAS_SPEC_HYBRID;
        else if (!strcmp(a, "hybrid_policy=legacy")) hybrid_policy = 0;
        else if (!strcmp(a, "hybrid_policy=cost")) hybrid_policy = 1;
        else if (!strncmp(a, "lookup_drafts=", 14) && unsigned_value(a + 14, &lookup_drafts) &&
                 lookup_drafts >= 1 && lookup_drafts <= JANAS_GEN_MAX_K) { }
        else if (!strncmp(a, "mtp_drafts=", 11) && unsigned_value(a + 11, &mtp_drafts) &&
                 mtp_drafts >= 1 && mtp_drafts <= JANAS_GEN_MAX_K) { }
        else if (!strncmp(a, "hybrid_probe_percent=", 21) && unsigned_value(a + 21, &hybrid_probe_percent) &&
                 hybrid_probe_percent <= 100) { }
        else if (!strcmp(a, "draft_vocab=legacy")) vocab_policy = 0;
        else if (!strcmp(a, "draft_vocab=adaptive")) vocab_policy = 1;
        else if (!strncmp(a, "drafts=", 7) && unsigned_value(a + 7, &drafts) &&
                 drafts >= 1 && drafts <= JANAS_GEN_MAX_K) { }
        else if (!strncmp(a, "rows=", 5) && unsigned_value(a + 5, &rows) &&
                 (!rows || rows == 8192 || rows == 16384 || rows == 32768)) { }
        else {
            fprintf(stderr, "invalid llm_session option: %s\n", a);
            return 2;
        }
    }
    int32_t text[NA + NB];
    FILE *f = fopen(argv[2], "rb");
    if (!f || fread(text, sizeof(int32_t), NA + NB, f) != NA + NB)
        return 2;
    fclose(f);
    struct janas_cpu_layout layout;
    if (janas_cpu_layout(NULL, 0, &layout) != 0 || layout.n < 1) {
        fprintf(stderr, "cannot identify available compute CPUs\n");
        return 1;
    }
    int ccpus[JANAS_MAX_CPUS], icpus[JANAS_MAX_CPUS], nc = 0, ni = 0;
    int split = janas_cpu_split(ccpus, &nc, icpus, &ni, JANAS_MAX_CPUS);
    int pinned_io = 0;
    if (split > 0)
        for (int i = 0; i < ni; i++)
            for (int j = 0; j < layout.n; j++)
                if (icpus[i] == layout.cpus[j]) {
                    icpus[pinned_io++] = icpus[i];
                    break;
                }
    const int *io_pin = pinned_io > 0 ? icpus : NULL;
    if (split > 0)
        ni = pinned_io; /* the process's affinity may exclude some I/O CPUs */
    /* Homogeneous CPUs need unpinned I/O workers too. A zero count fails
       model_load before it can produce a useful error message. */
    if (split < 0 || ni < 1)
        ni = 4;
    if (ni > 255)
        ni = 255;
    if (layout.n > 8)
        layout.n = 8;
    if (setenv("JANAS_EXPERT_BITS", "6", 1) ||
        setenv("JANAS_HEAD_BITS", "0", 1) ||
        setenv("JANAS_MISS_SKIP", "0", 1)) {
        perror("quality test environment");
        return 1;
    }
    struct janas_llm_options mo = {.cache_bytes = (uint64_t)20480 << 20,
                                   .n_ctx = 1024,
                                   .compute_cpus = layout.cpus,
                                   .n_compute = layout.n,
                                   .io_cpus = io_pin,
                                   .n_io = ni,
                                   .mtp_path = mtp_path,
                                   .expert_bits = 6,
                                   .head_bits = 0,
                                   .min_quality = JANAS_LLM_QUALITY_HIGH,
                                   .no_npu = 1};
    char err[256] = {0};
    struct janas_llm_model *m =
        janas_llm_model_load(argv[1], &mo, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "%s: %s (compute threads %d, I/O threads %d%s)\n",
                argv[1], err[0] ? err : "model load failed", layout.n, ni,
                io_pin ? ", pinned" : ", unpinned");
        return 1;
    }
    int mode = speculation >= 0 ? speculation
                               : mtp_path ? JANAS_SPEC_MTP : JANAS_SPEC_LOOKUP;
    if (((mode == JANAS_SPEC_HYBRID && !hybrid_policy) || vocab_policy) && planner == 0)
        planner = 1; /* match the generator's resolved hybrid default */
    if (((mode == JANAS_SPEC_MTP || mode == JANAS_SPEC_HYBRID) &&
         !janas_llm_model_has_mtp(m)) ||
        (vocab_policy && mode != JANAS_SPEC_MTP && mode != JANAS_SPEC_HYBRID)) {
        fprintf(stderr, "selected speculation/adaptive head requires an MTP head\n");
        janas_llm_model_free(m);
        return 2;
    }
    printf("settings: compute=%d io=%d planner=%s speculation=%s drafts=%u "
           "draft_vocab=%s rows=%u quality=HIGH expert_bits=6 head_bits=0 "
           "hybrid_policy=%s lookup_drafts=%u mtp_drafts=%u hybrid_probe_percent=%u\n",
           layout.n, ni, planner == 1 ? "cycle" : planner == 2 ? "fixed" : "legacy",
           mode == JANAS_SPEC_HYBRID ? "hybrid" : mode == JANAS_SPEC_MTP ? "mtp" : "lookup",
           drafts, vocab_policy ? "adaptive" : "legacy", rows,
           hybrid_policy ? "cost" : "legacy", lookup_drafts, mtp_drafts, hybrid_probe_percent);
    struct janas_gen_options greedy = {.spec_k = 1, .eos = -1};
    struct janas_gen_options spec = {.spec_mode = mode,
                                     .spec_k = (int)drafts,
                                     .eos = -1,
                                     .spec_planner = planner,
                                     .spec_explore_percent = 20,
                                     .draft_vocab_policy = vocab_policy,
                                     .draft_vocab_rows = rows,
                                     .hybrid_policy = hybrid_policy,
                                     .hybrid_lookup_max_drafts = (int)lookup_drafts,
                                     .hybrid_mtp_max_drafts = (int)mtp_drafts,
                                     .hybrid_probe_percent = (int)hybrid_probe_percent};
    int32_t g1[G1], g2[G2], h1[G1], h2[G2], once[G2];
    int32_t seq[NA + G1 + NB];
    int ok = 1;

    if (run_turns(m, &greedy, text, g1, g2) != 0)
        return 1;
    memcpy(seq, text, NA * sizeof(int32_t));
    memcpy(seq + NA, g1, G1 * sizeof(int32_t));
    memcpy(seq + NA + G1, text + NA, NB * sizeof(int32_t));
    if (run_once(m, &greedy, seq, NA + G1 + NB, once, G2) != 0)
        return 1;
    ok &= same("greedy: turns vs all at once", g2, once, G2);

    if (run_turns(m, &spec, text, h1, h2) != 0)
        return 1;
    ok &= same("greedy + speculation: turn 1", g1, h1, G1);
    ok &= same("greedy + speculation: turn 2", g2, h2, G2);

    struct janas_gen_options sa = greedy, sb = spec;
    sa.temperature = sb.temperature = 1.0f;
    sa.top_p = sb.top_p = 0.95f;
    sa.seed = sb.seed = 42;
    if (run_turns(m, &sa, text, g1, g2) != 0 ||
        run_turns(m, &sb, text, h1, h2) != 0)
        return 1;
    ok &= same("sampling, speculation off vs on: turn 1", g1, h1, G1);
    ok &= same("sampling, speculation off vs on: turn 2", g2, h2, G2);
    int differs = memcmp(g2, once, sizeof(once)) != 0;
    printf("%-44s %s\n", "sampling differs from greedy",
           differs ? "yes" : "no (suspicious)");

    janas_llm_model_free(m);
    printf("llm_session: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
