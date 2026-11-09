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
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#endif

#include "common/sysinfo.h"
#include "janas/llm.h"
#include "../src/llm/generate.h"
#include "../src/llm/model.h"
/* Test-only access: reset mutable draft metadata while retaining loaded
   weights/workers. Production interfaces and runtime are unchanged. */
#include "../src/llm/model_impl.h"

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

#ifndef JANAS_SESSION_MATRIX_UNIT
static int run_turns(struct janas_llm_model *m,
                     const struct janas_gen_options *o, const int32_t *text,
                     int32_t *g1, int32_t *g2)
{
    struct janas_mtp_costs before, after;
    janas_llm_mtp_costs(m, &before);
    struct janas_llm_session *s = janas_llm_session_create(m, o);
    if (!s) return -1;
    if (janas_llm_session_append(s, text, NA) != 0) goto failed;
    for (int i = 0; i < G1; i++)
        if (janas_llm_session_next(s, &g1[i]) != 0)
            goto failed;
    if (janas_llm_session_append(s, text + NA, NB) != 0)
        goto failed;
    for (int i = 0; i < G2; i++)
        if (janas_llm_session_next(s, &g2[i]) != 0)
            goto failed;
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
    if (o->s2_exact) {
        struct janas_gen_s2_metrics s2;
        janas_llm_session_s2_metrics(s, &s2);
        ok &= s2.cycles > 0 && s2.maintenance_fast_calls > 0;
        printf("s2: cycles=%" PRIu64 " roots=%" PRIu64 " branches=%" PRIu64
               " cached=%" PRIu64 " cached_accepted=%" PRIu64 " maintenance=%" PRIu64 " %s\n",
               s2.cycles, s2.root_offered, s2.alternate_builds, s2.cached_uses,
               s2.cached_accepted, s2.maintenance_fast_calls, ok ? "verified" : "FAILED");
    }
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
failed:
    janas_llm_session_destroy(s);
    return -1;
}

static int run_once(struct janas_llm_model *m,
                    const struct janas_gen_options *o, const int32_t *seq,
                    uint32_t n, int32_t *g, int count)
{
    struct janas_llm_session *s = janas_llm_session_create(m, o);
    if (!s) return -1;
    if (janas_llm_session_append(s, seq, n) != 0) goto failed;
    for (int i = 0; i < count; i++)
        if (janas_llm_session_next(s, &g[i]) != 0)
            goto failed;
    janas_llm_session_destroy(s);
    return 0;
failed:
    janas_llm_session_destroy(s);
    return -1;
}

static int same(const char *what, const int32_t *a, const int32_t *b, int n)
{
    int ok = memcmp(a, b, (size_t)n * sizeof(int32_t)) == 0;
    printf("%-44s %s\n", what, ok ? "identical" : "DIFFERENT");
    return ok;
}
#endif

#define SESSION_CASE_MAX 32
#define SESSION_CASE_NAME 80
#define SESSION_ORACLE_TOKENS 32
struct session_case {
    char name[SESSION_CASE_NAME];
    int planner, speculation, vocab_policy, hybrid_policy, s2_exact;
    unsigned drafts, rows, lookup_drafts, mtp_drafts, hybrid_probe_percent;
    unsigned s2_branches, s2_max_nodes, s2_min_gain_percent;
};
static struct session_case case_defaults(void)
{
    struct session_case c = {.speculation = -1, .drafts = 4,
        .lookup_drafts = 8, .mtp_drafts = 4, .hybrid_probe_percent = 2,
        .s2_branches = 2, .s2_max_nodes = 8, .s2_min_gain_percent = 5};
    strcpy(c.name, "single");
    return c;
}
static int case_option(struct session_case *c, const char *a)
{
    if (!strcmp(a, "planner=legacy")) c->planner = 0;
    else if (!strcmp(a, "planner=cycle")) c->planner = 1;
    else if (!strcmp(a, "planner=fixed")) c->planner = 2;
    else if (!strcmp(a, "speculation=mtp")) c->speculation = JANAS_SPEC_MTP;
    else if (!strcmp(a, "speculation=hybrid")) c->speculation = JANAS_SPEC_HYBRID;
    else if (!strcmp(a, "s2_exact=off")) c->s2_exact = 0;
    else if (!strcmp(a, "s2_exact=on")) c->s2_exact = 1;
    else if (!strncmp(a, "s2_branches=", 12) && unsigned_value(a + 12, &c->s2_branches) &&
             c->s2_branches >= 1 && c->s2_branches <= 2) { }
    else if (!strncmp(a, "s2_max_nodes=", 13) && unsigned_value(a + 13, &c->s2_max_nodes) &&
             c->s2_max_nodes >= 1 && c->s2_max_nodes <= 32) { }
    else if (!strncmp(a, "s2_min_gain_percent=", 20) && unsigned_value(a + 20, &c->s2_min_gain_percent) &&
             c->s2_min_gain_percent <= 100) { }
    else if (!strcmp(a, "hybrid_policy=legacy")) c->hybrid_policy = 0;
    else if (!strcmp(a, "hybrid_policy=cost")) c->hybrid_policy = 1;
    else if (!strncmp(a, "lookup_drafts=", 14) && unsigned_value(a + 14, &c->lookup_drafts) &&
             c->lookup_drafts >= 1 && c->lookup_drafts <= JANAS_GEN_MAX_K) { }
    else if (!strncmp(a, "mtp_drafts=", 11) && unsigned_value(a + 11, &c->mtp_drafts) &&
             c->mtp_drafts >= 1 && c->mtp_drafts <= JANAS_GEN_MAX_K) { }
    else if (!strncmp(a, "hybrid_probe_percent=", 21) && unsigned_value(a + 21, &c->hybrid_probe_percent) &&
             c->hybrid_probe_percent <= 100) { }
    else if (!strcmp(a, "draft_vocab=legacy")) c->vocab_policy = 0;
    else if (!strcmp(a, "draft_vocab=adaptive")) c->vocab_policy = 1;
    else if (!strncmp(a, "drafts=", 7) && unsigned_value(a + 7, &c->drafts) &&
             c->drafts >= 1 && c->drafts <= JANAS_GEN_MAX_K) { }
    else if (!strncmp(a, "rows=", 5) && unsigned_value(a + 5, &c->rows) &&
             (!c->rows || c->rows == 8192 || c->rows == 16384 || c->rows == 32768)) { }
    else return -1;
    return 0;
}
static int case_name(const char *name)
{
    size_t n = strlen(name);
    if (!n || n >= SESSION_CASE_NAME) return 0;
    for (size_t i = 0; i < n; i++)
        if (!((name[i] >= 'a' && name[i] <= 'z') ||
              (name[i] >= 'A' && name[i] <= 'Z') ||
              (name[i] >= '0' && name[i] <= '9') ||
              name[i] == '-' || name[i] == '_')) return 0;
    return 1;
}
/* Validate every manifest row before opening the model. Names are restricted
   to JSON-safe ASCII; options use the same parser as the historical CLI. */
static int read_cases(const char *path, const struct session_case *defaults,
                       struct session_case cases[SESSION_CASE_MAX])
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    char line[4096];
    unsigned row = 0, n = 0;
    while (fgets(line, sizeof(line), f)) {
        row++;
        size_t len = strlen(line);
        if (!len || (line[len - 1] != '\n' && !feof(f))) goto bad;
        while (len && (line[len - 1] == '\r' || line[len - 1] == '\n')) line[--len] = 0;
        if (!len || line[0] == '#') continue;
        if (n == SESSION_CASE_MAX) goto bad;
        char *option = strchr(line, '\t');
        if (!option) goto bad;
        *option++ = 0;
        if (!case_name(line)) goto bad;
        for (unsigned i = 0; i < n; i++) if (!strcmp(cases[i].name, line)) goto bad;
        cases[n] = *defaults;
        strcpy(cases[n].name, line);
        for (;;) {
            char *next = strchr(option, '\t');
            if (next) *next++ = 0;
            if (!*option || case_option(&cases[n], option) != 0) goto bad;
            if (!next) break;
            option = next;
        }
        n++;
    }
    if (ferror(f) || !n) goto bad;
    fclose(f);
    return (int)n;
bad:
    fprintf(stderr, "invalid session case list at row %u: %s\n", row, path);
    fclose(f);
    return -1;
}
#ifndef JANAS_SESSION_MATRIX_UNIT
static double session_wall(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif
static uint64_t bytes_hash(uint64_t h, const void *data_, size_t bytes)
{
    const uint8_t *data = data_;
    for (size_t i = 0; i < bytes; i++) h = (h ^ data[i]) * UINT64_C(1099511628211);
    return h;
}
static uint64_t head_hash(const struct janas_llm_model *m)
{
    uint64_t h = UINT64_C(14695981039346656037);
    h = bytes_hash(h, &m->dh_n, sizeof(m->dh_n));
    if (m->dh_n) {
        h = bytes_hash(h, m->dh_ids, (size_t)m->dh_n * sizeof(*m->dh_ids));
        h = bytes_hash(h, m->dh_in, m->n_vocab);
        h = bytes_hash(h, m->dh_w, (size_t)m->dh_n * m->dh_row);
    }
    return h;
}
static uint64_t mtp_kv_clear_hash(struct janas_llm_model *m)
{
    uint64_t h = UINT64_C(14695981039346656037);
    if (!m->mtp || m->asst) return h;
    unsigned slot = m->n_attn;
    size_t rows = (size_t)m->kv_heads[slot] * m->kv_cap[slot];
    size_t bytes = rows * ((size_t)m->kv_hd[slot] << m->kv16);
    void *buffers[] = {m->kcache + m->kv_off[slot], m->vcache + m->kv_off[slot],
        m->kscale + m->ks_off[slot], m->vscale + m->ks_off[slot]};
    const size_t sizes[] = {bytes, bytes, rows * sizeof(float), rows * sizeof(float)};
    for (unsigned i = 0; i < 4; i++) {
        memset(buffers[i], 0, sizes[i]);
        h = bytes_hash(h, buffers[i], sizes[i]);
    }
    return h;
}
struct case_boundary {
    int32_t *ids;
    uint8_t *membership;
    uint32_t count;
    uint64_t head, mtp_kv, oracle;
    float *reference, *scratch;
    size_t oracle_values;
    struct janas_tuner tuner;
};
static void boundary_free(struct case_boundary *b)
{
    free(b->ids); free(b->membership); free(b->reference); free(b->scratch);
    memset(b, 0, sizeof(*b));
}
static int boundary_capture(struct janas_llm_model *m, const int32_t *text,
                             struct case_boundary *b)
{
    memset(b, 0, sizeof(*b));
    /* The fast matrix is deliberately the native CPU validation path. Other
       architectures retain the ordinary one-process-per-case harness. */
    if (m->asst || m->hc_n || m->gpu || m->npu) return -1;
    b->tuner = m->tuner; /* pointer-free tuning observations, not resources */
    b->count = m->dh_n;
    if (b->count) {
        if (!m->dh_ids || !m->dh_in || !m->dh_w || !m->dh_src || !m->dh_row) return -1;
        b->ids = malloc((size_t)b->count * sizeof(*b->ids));
        b->membership = malloc(m->n_vocab);
        if (!b->ids || !b->membership) goto bad;
        memcpy(b->ids, m->dh_ids, (size_t)b->count * sizeof(*b->ids));
        memcpy(b->membership, m->dh_in, m->n_vocab);
        for (uint32_t i = 0; i < b->count; i++)
            if (b->ids[i] < 0 || (uint32_t)b->ids[i] >= m->n_vocab) goto bad;
    }
    b->head = head_hash(m);
    b->mtp_kv = mtp_kv_clear_hash(m);
    b->oracle_values = (size_t)SESSION_ORACLE_TOKENS * m->n_vocab;
    b->reference = malloc(b->oracle_values * sizeof(float));
    b->scratch = malloc(b->oracle_values * sizeof(float));
    if (!b->reference || !b->scratch ||
        janas_llm_model_forward(m, text, SESSION_ORACLE_TOKENS, 0, b->reference, 1) != 0)
        goto bad;
    b->oracle = bytes_hash(UINT64_C(14695981039346656037), b->reference,
                           b->oracle_values * sizeof(float));
    return 0;
bad:
    boundary_free(b);
    return -1;
}
static int boundary_restore(struct janas_llm_model *m, const int32_t *text,
                             const struct case_boundary *b)
{
    janas_llm_mtp_vocab_legacy(m);
    janas_llm_mtp_vocab_free(m); /* discard adaptive observations, not weights */
    m->dh_n = b->count;
    if (b->count) {
        memcpy(m->dh_ids, b->ids, (size_t)b->count * sizeof(*b->ids));
        memcpy(m->dh_in, b->membership, m->n_vocab);
        for (uint32_t i = 0; i < b->count; i++)
            memcpy(m->dh_w + (size_t)i * m->dh_row,
                   m->dh_src + (size_t)b->ids[i] * m->dh_row, m->dh_row);
    }
    m->dh_scored_sum = 0;
    m->dh_scored_rows = 0;
    m->dh_scored_best = m->dh_scored_policy = 0;
    m->dh_scored_epoch = 0;
    m->mtp_maintenance_s = m->mtp_head_s = m->mtp_repack_s = 0;
    m->mtp_head_calls = m->mtp_repacks = m->mtp_vocab_fallbacks = 0;
    memset(&m->runtime_stats, 0, sizeof(m->runtime_stats));
    memset(m->phase, 0, sizeof(m->phase));
    m->tuner = b->tuner;
    uint64_t head = head_hash(m), kv = mtp_kv_clear_hash(m);
    /* pos0 is the production reset boundary: it clears recurrent base/tip
       dependencies and rewrites the causal KV prefix. Compare EVERY target
       logit word, so an omitted mutable state fails this fast-mode gate. */
    int okay = janas_llm_model_forward(m, text, SESSION_ORACLE_TOKENS, 0,
                                      b->scratch, 1) == 0;
    uint64_t oracle = okay ? bytes_hash(UINT64_C(14695981039346656037), b->scratch,
                                        b->oracle_values * sizeof(float)) : 0;
    if (okay) okay = head == b->head && kv == b->mtp_kv &&
        memcmp(b->reference, b->scratch, b->oracle_values * sizeof(float)) == 0;
    m->tuner = b->tuner;
    m->last_pass_s = 0;
    printf("boundary: head=%016" PRIx64 " mtp_kv=%016" PRIx64
           " logits=%016" PRIx64 " %s\n", head, kv, oracle,
           okay ? "identical" : "DIFFERENT");
    return okay ? 0 : -1;
}
static int directory_new(const char *path)
{
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}
static int case_cache(const char *base, const char *name, unsigned index,
                       char *path, size_t capacity)
{
    int n = snprintf(path, capacity, "%s/case-%02u-%s", base, index, name);
    if (n < 0 || (size_t)n >= capacity || directory_new(path) != 0) return -1;
    char directory[2048], source[2048], target[2048];
    n = snprintf(directory, sizeof(directory), "%s/janas", path);
    if (n < 0 || (size_t)n >= sizeof(directory) || directory_new(directory) != 0) return -1;
    n = snprintf(source, sizeof(source), "%s/janas/attn.txt", base);
    if (n < 0 || (size_t)n >= sizeof(source)) return -1;
    n = snprintf(target, sizeof(target), "%s/attn.txt", directory);
    if (n < 0 || (size_t)n >= sizeof(target)) return -1;
    FILE *in = fopen(source, "rb");
    if (!in && errno != ENOENT) return -1;
    if (in) {
        FILE *out = fopen(target, "wb");
        if (!out) { fclose(in); return -1; }
        char buffer[4096]; size_t got;
        int okay = 1;
        while ((got = fread(buffer, 1, sizeof(buffer), in)) != 0)
            if (fwrite(buffer, 1, got, out) != got) { okay = 0; break; }
        if (ferror(in)) okay = 0;
        if (fclose(out) != 0) okay = 0;
        fclose(in);
        if (!okay) return -1;
    }
    return setenv("XDG_CACHE_HOME", path, 1);
}
#ifndef JANAS_SESSION_MATRIX_UNIT
/* Immutable, process-local references. The validator binds the content key
   to executable/model/input identities and the full target sampling controls.
   Failed references and data from other processes are never reused. */
struct session_reference {
    int valid;
    unsigned model_runs;
    int32_t greedy1[G1], greedy2[G2], once[G2], sampled1[G1], sampled2[G2];
    uint64_t key;
};
static int reference_build(struct janas_llm_model *m, const int32_t *text,
                            struct session_reference *r)
{
    memset(r, 0, sizeof(*r));
    struct janas_gen_options o = {.spec_k = 1, .eos = -1};
    int32_t seq[NA + G1 + NB];
    r->model_runs++;
    if (run_turns(m, &o, text, r->greedy1, r->greedy2) != 0) return -1;
    memcpy(seq, text, NA * sizeof(int32_t));
    memcpy(seq + NA, r->greedy1, G1 * sizeof(int32_t));
    memcpy(seq + NA + G1, text + NA, NB * sizeof(int32_t));
    r->model_runs++;
    if (run_once(m, &o, seq, NA + G1 + NB, r->once, G2) != 0 ||
        memcmp(r->greedy2, r->once, sizeof(r->once)) != 0) return -1;
    o.temperature = 1.0f; o.top_p = .95f; o.seed = 42;
    r->model_runs++;
    if (run_turns(m, &o, text, r->sampled1, r->sampled2) != 0) return -1;
    static const char controls[] = "llm_session_reference_v1|NA120|NB40|G112|G224|"
        "CPU_HIGH|expert_bits6|head_bits0|nctx1024|full_target_vocab|"
        "greedy_t0|sample_t1_p0.95_seed42|eos_minus1|no_filters_or_penalties";
    uint64_t key = bytes_hash(UINT64_C(14695981039346656037), controls, sizeof(controls));
    key = bytes_hash(key, text, (NA + NB) * sizeof(*text));
    key = bytes_hash(key, r->greedy1, sizeof(r->greedy1));
    key = bytes_hash(key, r->greedy2, sizeof(r->greedy2));
    key = bytes_hash(key, r->once, sizeof(r->once));
    key = bytes_hash(key, r->sampled1, sizeof(r->sampled1));
    r->key = bytes_hash(key, r->sampled2, sizeof(r->sampled2));
    r->valid = 1;
    return 0;
}
#endif
/* Plain sessions maintain the observed-ID draft head without scoring it.
   Replay only verified reference observations, in canonical order. Preserve
   any extra IDs learned by this case's own greedy verifier and preserve its
   adaptive cost observations before the seeded candidate. */
static void reference_note(struct janas_llm_model *m, const int32_t *text,
                            const int32_t g1[G1], const int32_t g2[G2])
{
    janas_llm_mtp_vocab_legacy(m);
    janas_llm_mtp_vocab_new_reply(m);
    janas_llm_mtp_note_tokens(m, text, NA);
    janas_llm_mtp_note_tokens(m, g1, G1);
    janas_llm_mtp_vocab_new_reply(m);
    janas_llm_mtp_note_tokens(m, text + NA, NB);
    janas_llm_mtp_note_tokens(m, g2, G2);
}
#ifndef JANAS_SESSION_MATRIX_UNIT
static int run_case(struct janas_llm_model *m, struct session_case c,
                     const int32_t *text, int has_mtp_path, int nc, int ni,
                     unsigned *checks, unsigned *model_runs,
                     const struct session_reference *reference)
{
    *checks = 0;
    *model_runs = 0;
    int mode = c.speculation >= 0 ? c.speculation
                                  : has_mtp_path ? JANAS_SPEC_MTP : JANAS_SPEC_LOOKUP;
    if (((mode == JANAS_SPEC_HYBRID && !c.hybrid_policy) || c.vocab_policy) && !c.planner)
        c.planner = 1;
    if (((mode == JANAS_SPEC_MTP || mode == JANAS_SPEC_HYBRID) && !janas_llm_model_has_mtp(m)) ||
        (c.vocab_policy && mode != JANAS_SPEC_MTP && mode != JANAS_SPEC_HYBRID)) {
        fprintf(stderr, "selected speculation/adaptive head requires an MTP head\n");
        return 2;
    }
    printf("settings: compute=%d io=%d planner=%s speculation=%s drafts=%u "
           "draft_vocab=%s rows=%u quality=HIGH expert_bits=6 head_bits=0 "
           "hybrid_policy=%s lookup_drafts=%u mtp_drafts=%u hybrid_probe_percent=%u\n",
           nc, ni, c.planner == 1 ? "cycle" : c.planner == 2 ? "fixed" : "legacy",
           mode == JANAS_SPEC_HYBRID ? "hybrid" : mode == JANAS_SPEC_MTP ? "mtp" : "lookup",
           c.drafts, c.vocab_policy ? "adaptive" : "legacy", c.rows,
           c.hybrid_policy ? "cost" : "legacy", c.lookup_drafts, c.mtp_drafts, c.hybrid_probe_percent);
    struct janas_gen_options greedy = {.spec_k = 1, .eos = -1};
    struct janas_gen_options spec = {.spec_mode = mode, .spec_k = (int)c.drafts, .eos = -1,
        .spec_planner = c.planner, .spec_explore_percent = 20,
        .draft_vocab_policy = c.vocab_policy, .draft_vocab_rows = c.rows,
        .hybrid_policy = c.hybrid_policy, .hybrid_lookup_max_drafts = (int)c.lookup_drafts,
        .hybrid_mtp_max_drafts = (int)c.mtp_drafts, .hybrid_probe_percent = (int)c.hybrid_probe_percent,
        .s2_exact = c.s2_exact, .s2_branches = (int)c.s2_branches,
        .s2_max_nodes = (int)c.s2_max_nodes, .s2_min_gain_percent = (int)c.s2_min_gain_percent};
    int32_t g1[G1],g2[G2],h1[G1],h2[G2],once[G2],seq[NA+G1+NB];
    int okay = 1;
    if (reference) {
        if (!reference->valid) return 1;
        memcpy(g1, reference->greedy1, sizeof(g1));
        memcpy(g2, reference->greedy2, sizeof(g2));
        memcpy(once, reference->once, sizeof(once));
        reference_note(m, text, g1, g2);
    } else {
        (*model_runs)++;
        if (run_turns(m,&greedy,text,g1,g2) != 0) return 1;
        memcpy(seq,text,NA*sizeof(int32_t));
        memcpy(seq+NA,g1,G1*sizeof(int32_t));
        memcpy(seq+NA+G1,text+NA,NB*sizeof(int32_t));
        (*model_runs)++;
        if (run_once(m,&greedy,seq,NA+G1+NB,once,G2) != 0) return 1;
    }
    okay &= same("greedy: turns vs all at once",g2,once,G2); (*checks)++;
    (*model_runs)++;
    if (run_turns(m,&spec,text,h1,h2) != 0) return 1;
    okay &= same("greedy + speculation: turn 1",g1,h1,G1); (*checks)++;
    okay &= same("greedy + speculation: turn 2",g2,h2,G2); (*checks)++;
    struct janas_gen_options sa=greedy,sb=spec;
    sa.temperature=sb.temperature=1.0f;sa.top_p=sb.top_p=.95f;sa.seed=sb.seed=42;
    if (reference) {
        memcpy(g1, reference->sampled1, sizeof(g1));
        memcpy(g2, reference->sampled2, sizeof(g2));
        reference_note(m, text, g1, g2);
    } else {
        (*model_runs)++;
        if (run_turns(m, &sa, text, g1, g2) != 0) return 1;
    }
    (*model_runs)++;
    if (run_turns(m, &sb, text, h1, h2) != 0) return 1;
    okay &= same("sampling, speculation off vs on: turn 1",g1,h1,G1); (*checks)++;
    okay &= same("sampling, speculation off vs on: turn 2",g2,h2,G2); (*checks)++;
    printf("%-44s %s\n","sampling differs from greedy",
           memcmp(g2,once,sizeof(once)) ? "yes" : "no (suspicious)");
    return okay ? 0 : 1;
}
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,"usage: llm_session <model.jns> <tokens.int32> [mtp=<mtp.jns>] "
            "[planner=legacy|cycle|fixed] [speculation=mtp|hybrid] [drafts=1..16] "
            "[draft_vocab=legacy|adaptive] [rows=0|8192|16384|32768] "
            "[hybrid_policy=legacy|cost] [lookup_drafts=1..16] [mtp_drafts=1..16] "
            "[hybrid_probe_percent=0..100] [s2_exact=off|on] [cases=<cases.tsv>]\n");
        return 2;
    }
    struct session_case defaults=case_defaults(),cases[SESSION_CASE_MAX];
    const char *mtp_path=NULL,*cases_path=NULL;
    for(int i=3;i<argc;i++) {
        if(!strncmp(argv[i],"mtp=",4)&&argv[i][4])mtp_path=argv[i]+4;
        else if(!strncmp(argv[i],"cases=",6)&&argv[i][6]&&!cases_path)cases_path=argv[i]+6;
        else if(case_option(&defaults,argv[i])!=0) {
            fprintf(stderr,"invalid llm_session option: %s\n",argv[i]);return 2;
        }
    }
    int count=1;
    if(cases_path) {
        count=read_cases(cases_path,&defaults,cases);
        if(count<0)return 2;
    } else cases[0]=defaults;
    const char *cache_env=getenv("XDG_CACHE_HOME");
    char base_cache[2048]={0};
    if(cases_path) {
        if(!cache_env||!*cache_env||strlen(cache_env)>=sizeof(base_cache)) {
            fprintf(stderr,"cases mode requires an explicit fresh XDG_CACHE_HOME\n");return 2;
        }
        strcpy(base_cache,cache_env);
    }
    int32_t text[NA+NB];
    FILE *f=fopen(argv[2],"rb");
    if(!f)return 2;
    int input_ok=fread(text,sizeof(int32_t),NA+NB,f)==NA+NB;
    fclose(f);if(!input_ok)return 2;
    struct janas_cpu_layout layout;
    if(janas_cpu_layout(NULL,0,&layout)!=0||layout.n<1) {
        fprintf(stderr,"cannot identify available compute CPUs\n");return 1;
    }
    int ccpus[JANAS_MAX_CPUS],icpus[JANAS_MAX_CPUS],nc=0,ni=0;
    int split=janas_cpu_split(ccpus,&nc,icpus,&ni,JANAS_MAX_CPUS),pinned_io=0;
    if(split>0)for(int i=0;i<ni;i++)for(int j=0;j<layout.n;j++)
        if(icpus[i]==layout.cpus[j]){icpus[pinned_io++]=icpus[i];break;}
    const int *io_pin=pinned_io>0?icpus:NULL;
    if(split>0)ni=pinned_io;
    if(split<0||ni<1)ni=4;
    if(ni>255)ni=255;
    if(layout.n>8)layout.n=8;
    if(setenv("JANAS_EXPERT_BITS","6",1)||setenv("JANAS_HEAD_BITS","0",1)||
       setenv("JANAS_MISS_SKIP","0",1)) {perror("quality test environment");return 1;}
    struct janas_llm_options mo={.cache_bytes=(uint64_t)20480<<20,.n_ctx=1024,
        .compute_cpus=layout.cpus,.n_compute=layout.n,.io_cpus=io_pin,.n_io=ni,
        .mtp_path=mtp_path,.expert_bits=6,.head_bits=0,
        .min_quality=JANAS_LLM_QUALITY_HIGH,.no_npu=1};
    char error[256]={0};
    double load_start=session_wall();
    struct janas_llm_model *m=janas_llm_model_load(argv[1],&mo,error,sizeof(error));
    double load_seconds=session_wall()-load_start;
    if(!m) {
        fprintf(stderr,"%s: %s (compute threads %d, I/O threads %d%s)\n",argv[1],
            error[0]?error:"model load failed",layout.n,ni,io_pin?", pinned":", unpinned");return 1;
    }
    struct case_boundary boundary={0};
    struct session_reference reference={0};
    int matrix_ready=1;
    if(cases_path) {
        printf("session-matrix: {\"version\":1,\"cases\":%d,\"model_loads\":1,"
            "\"load_seconds\":%.9f,\"functional_only\":true,\"expert_cache\":\"shared-warm\"}\n",count,load_seconds);
        if(boundary_capture(m,text,&boundary)!=0) {
            fprintf(stderr,"cannot establish native CPU matrix boundary oracle\n");
            matrix_ready=0;
        } else if(reference_build(m,text,&reference)!=0) {
            fprintf(stderr,"cannot establish valid immutable reference transcripts\n");
            matrix_ready=0;
        }
        printf("session-reference: {\"version\":1,\"valid\":%s,\"model_runs\":%u,"
            "\"content_key\":\"%016" PRIx64 "\",\"functional_only\":true}\n",
            matrix_ready?"true":"false",reference.model_runs,reference.key);
    }
    int final=0;
    for(int i=0;i<count;i++) {
        unsigned checks=0,model_runs=0,boundary_calls=0;
        double start=session_wall(),isolation_seconds=0;
        int result=cases_path&&!matrix_ready;
        if(cases_path) {
            printf("session-case-begin: %s\n",cases[i].name);fflush(stdout);
            char path[2048];
            if(matrix_ready) {
                if(case_cache(base_cache,cases[i].name,(unsigned)i,path,sizeof(path))!=0)result=1;
                else {
                    boundary_calls++;
                    if(boundary_restore(m,text,&boundary)!=0)result=1;
                }
            }
            isolation_seconds=session_wall()-start;
        }
        if(!result)result=run_case(m,cases[i],text,mtp_path!=NULL,layout.n,ni,&checks,&model_runs,
                                 cases_path?&reference:NULL);
        printf("llm_session: %s\n",result?"FAILED":"ok");
        if(cases_path) {
            printf("session-case: {\"version\":1,\"name\":\"%s\",\"index\":%d,"
                "\"exit_code\":%d,\"wall_seconds\":%.9f,\"isolation_seconds\":%.9f,"
                "\"checks\":%u,\"reference_reused\":true,\"actual_model_runs\":%u,"
                "\"boundary_forward_calls\":%u,\"content_key\":\"%016" PRIx64 "\","
                "\"functional_only\":true}\n",cases[i].name,i,result,
                session_wall()-start,isolation_seconds,checks,model_runs,boundary_calls,reference.key);
            fflush(stdout);
        }
        if(result)final=1;
    }
    boundary_free(&boundary);
    if(cases_path&&setenv("XDG_CACHE_HOME",base_cache,1)!=0)final=1;
    janas_llm_model_free(m);
    if(cases_path)printf("llm_session_matrix: %s\n",final?"FAILED":"ok");
    return final;
}
#endif
