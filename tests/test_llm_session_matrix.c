/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Model-free proof of the fast real-model harness's manifest and isolation
   seams. The actual target-logit oracle is exercised by llm_session cases=. */
#define JANAS_SESSION_MATRIX_UNIT
#include "llm_session.c"
#ifdef _WIN32
#include <process.h>
#define matrix_pid _getpid
#define matrix_rmdir _rmdir
#else
#include <unistd.h>
#define matrix_pid getpid
#define matrix_rmdir rmdir
#endif

static int failures, forwards, corrupt_oracle, fail_oracle, frees;
static unsigned legacy_calls, reply_calls, note_calls, noted_count;
static int32_t noted[NA + NB + G1 + G2];
#define CHECK(value, description) do { if (!(value)) { \
    fprintf(stderr, "FAIL: %s (line %d)\n", description, __LINE__); failures++; \
} } while (0)

void janas_llm_mtp_vocab_legacy(struct janas_llm_model *m)
{ (void)m; legacy_calls++; }
void janas_llm_mtp_vocab_new_reply(struct janas_llm_model *m)
{ (void)m; reply_calls++; }
void janas_llm_mtp_note_tokens(struct janas_llm_model *m, const int32_t *tokens, uint32_t n)
{
    note_calls++;
    for (uint32_t i = 0; i < n; i++) {
        CHECK(noted_count < sizeof(noted) / sizeof(*noted), "reference observes only bounded canonical tokens");
        if (noted_count < sizeof(noted) / sizeof(*noted)) noted[noted_count++] = tokens[i];
        int32_t id = tokens[i];
        if (id < 0 || (uint32_t)id >= m->n_vocab || m->dh_in[id] || m->dh_n == m->dh_cap) continue;
        m->dh_ids[m->dh_n] = id;
        memcpy(m->dh_w + (size_t)m->dh_n++ * m->dh_row, m->dh_src + (size_t)id * m->dh_row, m->dh_row);
        m->dh_in[id] = 1;
    }
}
void janas_llm_mtp_vocab_free(struct janas_llm_model *m)
{
    if (m->dh_adaptive) frees++;
    free(m->dh_adaptive); m->dh_adaptive = NULL;
}
int janas_llm_model_forward(struct janas_llm_model *m, const int32_t *tokens,
                             uint32_t n, uint32_t pos, float *logits, int all)
{
    forwards++;
    CHECK(n == SESSION_ORACLE_TOKENS && pos == 0 && all == 1,
          "boundary oracle always runs every prefix row from native pos0 reset");
    if (fail_oracle) return -1;
    for (uint32_t row = 0; row < n; row++)
        for (uint32_t v = 0; v < m->n_vocab; v++)
            logits[(size_t)row * m->n_vocab + v] = (float)(tokens[row] * 16 + (int32_t)v);
    if (corrupt_oracle) logits[(size_t)n * m->n_vocab - 1] += 1.0f;
    m->tuner.cls[0].passes += 99;
    return 0;
}
static int manifest(const char *base, const char *content,
                      const struct session_case *defaults,
                      struct session_case cases[SESSION_CASE_MAX])
{
    char path[256];
    snprintf(path, sizeof(path), "%s/cases.tsv", base);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL, "temporary manifest opened");
    if (!f) return -1;
    CHECK(fwrite(content, 1, strlen(content), f) == strlen(content), "temporary manifest written");
    fclose(f);
    int n = read_cases(path, defaults, cases);
    remove(path);
    return n;
}
static void parser_tests(const char *base)
{
    struct session_case defaults = case_defaults(), cases[SESSION_CASE_MAX];
    defaults.s2_exact = 1;
    const char *valid =
        "# full seven-case matrix; global S2 is inherited\r\n"
        "legacy-mtp4\tplanner=legacy\tspeculation=mtp\tdrafts=4\tdraft_vocab=legacy\trows=0\n"
        "cycle-mtp4\tplanner=cycle\tspeculation=mtp\tdrafts=4\tdraft_vocab=legacy\trows=0\n"
        "hybrid-cycle8\tplanner=cycle\tspeculation=hybrid\tdrafts=8\tdraft_vocab=legacy\trows=0\n"
        "hybrid-cycle8-adaptive\tplanner=cycle\tspeculation=hybrid\tdrafts=8\tdraft_vocab=adaptive\trows=0\n"
        "fixed-mtp16-adaptive\tplanner=fixed\tspeculation=mtp\tdrafts=16\tdraft_vocab=adaptive\trows=16384\ts2_max_nodes=32\n"
        "hybrid-cost8\tplanner=legacy\tspeculation=hybrid\tdrafts=8\thybrid_policy=cost\thybrid_probe_percent=0\n"
        "hybrid-cost8-probe\tplanner=legacy\tspeculation=hybrid\tdrafts=8\thybrid_policy=cost\thybrid_probe_percent=2";
    CHECK(manifest(base, valid, &defaults, cases) == 7,
          "all seven original configurations parsed before model loading");
    CHECK(cases[0].planner == 0 && cases[1].planner == 1 &&
          cases[2].speculation == JANAS_SPEC_HYBRID && cases[3].vocab_policy == 1 &&
          cases[4].drafts == 16 && cases[4].s2_max_nodes == 32 &&
          cases[4].rows == 16384 && cases[5].hybrid_probe_percent == 0 &&
          cases[6].hybrid_probe_percent == 2,
          "manifest preserves planner/depth/head/probe coverage, including full width17 stress cap");
    for (unsigned i = 0; i < 7; i++) CHECK(cases[i].s2_exact == 1, "global S2 controls inherited per case");
    const char *invalid[] = {"", "a\n", "a\tdrafts=0\n", "a\tdrafts=17\n",
        "a\tmtp_drafts=42949672960\n", "a\tunknown=x\n", "a\trows=123\n",
        "a\tplanner=legacy\na\tplanner=cycle\n", "bad name\tdrafts=4\n",
        "a\tdrafts=4\t\n", "a\t\tdrafts=4\n", "a\ts2_max_nodes=33\n"};
    int before = forwards;
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(*invalid); i++)
        CHECK(manifest(base, invalid[i], &defaults, cases) < 0,
              "malformed, duplicate, overflowing and unknown controls rejected before model loading");
    CHECK(forwards == before, "manifest validation performs no target work");
    char too_long[4200]; memset(too_long, 'a', sizeof(too_long));
    too_long[1] = '\t'; too_long[sizeof(too_long) - 2] = '\n'; too_long[sizeof(too_long) - 1] = 0;
    CHECK(manifest(base, too_long, &defaults, cases) < 0, "oversized manifest row rejected");
}
static void boundary_tests(void)
{
    struct janas_llm_model m = {0};
    uint8_t source[8 * 4], weights[8 * 4], membership[8] = {0};
    int32_t ids[8] = {1, 3}, tokens[NA + NB];
    uint32_t heads[] = {1, 1}, caps[] = {4, 4}, dimensions[] = {2, 2};
    size_t offsets[] = {0, 8}, scale_offsets[] = {0, 4};
    int8_t keys[16], values[16]; float key_scales[8], value_scales[8];
    memset(keys, 0x7b, sizeof(keys)); memset(values, 0x3c, sizeof(values));
    memset(key_scales, 0x61, sizeof(key_scales)); memset(value_scales, 0x52, sizeof(value_scales));
    for (unsigned i = 0; i < sizeof(source); i++) source[i] = (uint8_t)(i * 7);
    for (unsigned i = 0; i < NA + NB; i++) tokens[i] = (int32_t)(i % 8);
    memcpy(weights, source + 4, 4); memcpy(weights + 4, source + 12, 4);
    membership[1] = membership[3] = 1;
    m.n_vocab = 8; m.dh_n = 2; m.dh_cap = 8; m.dh_row = 4;
    m.dh_w = weights; m.dh_src = source; m.dh_ids = ids; m.dh_in = membership;
    m.mtp = 1; m.n_attn = 1; m.kv_heads = heads; m.kv_cap = caps; m.kv_hd = dimensions;
    m.kv_off = offsets; m.ks_off = scale_offsets;
    m.kcache = keys; m.vcache = values; m.kscale = key_scales; m.vscale = value_scales;
    m.tuner.cls[0].passes = 17;
    struct case_boundary boundary;
    CHECK(boundary_capture(&m, tokens, &boundary) == 0, "initial exact head/KV/full-logit boundary oracle captured");
    memset(weights, 0xa5, sizeof(weights)); ids[2] = 7; membership[7] = 1; m.dh_n = 3;
    memset(keys + 8, 0x22, 8); memset(values + 8, 0x33, 8);
    m.dh_adaptive = malloc(1); m.dh_scored_sum = 1; m.dh_scored_rows = 3;
    m.mtp_head_calls = 19; m.mtp_repacks = 4; m.runtime_stats.expert_assignments = 999;
    CHECK(boundary_restore(&m, tokens, &boundary) == 0,
          "boundary removes prior high-ID/adaptive/head-logit state and restores exact source rows");
    CHECK(m.dh_n == 2 && !membership[7] && ids[0] == 1 && ids[1] == 3 &&
          !memcmp(weights, source + 4, 4) && !memcmp(weights + 4, source + 12, 4) &&
          !m.dh_adaptive && frees == 1 && m.dh_scored_sum == 0 && m.dh_scored_rows == 0 &&
          !m.mtp_head_calls && !m.mtp_repacks && m.tuner.cls[0].passes == 17,
          "fresh head membership/weights/observations restored without shallow-copying resources");
    for (unsigned i = 0; i < 8; i++) CHECK(keys[i] == 0x7b && values[i] == 0x3c,
                                          "MTP reset preserves other KV slots");
    corrupt_oracle = 1;
    CHECK(boundary_restore(&m, tokens, &boundary) != 0,
          "single changed last-row logit word fails boundary comparison");
    corrupt_oracle = 0; fail_oracle = 1;
    CHECK(boundary_restore(&m, tokens, &boundary) != 0,
          "failed target forward cannot establish a valid reused oracle");
    fail_oracle = 0;
    boundary_free(&boundary);
}
static void cache_tests(const char *base)
{
    char seed_dir[256], seed[300], first[512], second[512], target[600], calibration[600];
    snprintf(seed_dir, sizeof(seed_dir), "%s/janas", base);
    CHECK(directory_new(seed_dir) == 0, "private frozen attention seed directory created");
    snprintf(seed, sizeof(seed), "%s/attn.txt", seed_dir);
    FILE *f = fopen(seed, "wb"); CHECK(f != NULL, "private attention seed file opened");
    if (!f) return;
    fputs("8|16/2/256|0|2048 4294967295 1024\n", f); fclose(f);
    CHECK(case_cache(base, "one", 0, first, sizeof(first)) == 0,
          "each case gets its own fresh calibration directory");
    snprintf(target, sizeof(target), "%s/janas/attn.txt", first);
    f = fopen(target, "rb"); CHECK(f != NULL, "case copies only frozen attention calibration");
    if (f) fclose(f);
    snprintf(calibration, sizeof(calibration), "%s/janas/drafts-123.bin", first);
    f = fopen(calibration, "wb"); CHECK(f != NULL, "first case calibration can be learned privately");
    if (f) { fputs("learned", f); fclose(f); }
    CHECK(case_cache(base, "two", 1, second, sizeof(second)) == 0 &&
          !strcmp(getenv("XDG_CACHE_HOME"), second), "second case switches to independent empty draft calibration");
    char absent[600]; snprintf(absent, sizeof(absent), "%s/janas/drafts-123.bin", second);
    f = fopen(absent, "rb"); CHECK(f == NULL, "learned calibration never leaks between cases"); if (f) fclose(f);
    CHECK(case_cache(base, "one", 0, target, sizeof(target)) != 0,
          "existing case cache is refused instead of silently inheriting calibration");
    remove(calibration); remove(seed);
    snprintf(target, sizeof(target), "%s/janas/attn.txt", first); remove(target);
    snprintf(target, sizeof(target), "%s/janas/attn.txt", second); remove(target);
    snprintf(target, sizeof(target), "%s/janas", first); matrix_rmdir(target); matrix_rmdir(first);
    snprintf(target, sizeof(target), "%s/janas", second); matrix_rmdir(target); matrix_rmdir(second);
    matrix_rmdir(seed_dir);
}
static void reference_observation_tests(void)
{
    struct janas_llm_model m = {0};
    uint8_t source[32] = {0}, weights[32] = {0}, membership[8] = {0};
    int32_t ids[8] = {7}, text[NA + NB] = {0}, g1[G1], g2[G2];
    text[0] = 6;
    for (unsigned i = NA; i < NA + NB; i++) text[i] = 3;
    for (unsigned i = 0; i < G1; i++) g1[i] = 1;
    for (unsigned i = 0; i < G2; i++) g2[i] = 2;
    m.n_vocab = 8; m.dh_n = 1; m.dh_cap = 8; m.dh_row = 4;
    m.dh_src = source; m.dh_w = weights; m.dh_in = membership; m.dh_ids = ids;
    membership[7] = 1; /* extra canonical ID from this case's own greedy pass */
    m.dh_adaptive = malloc(1);
    void *observations = m.dh_adaptive;
    unsigned old_legacy = legacy_calls, old_reply = reply_calls;
    int old_forwards = forwards, old_frees = frees;
    reference_note(&m, text, g1, g2);
    const int32_t expected_ids[] = {7, 6, 0, 1, 3, 2};
    CHECK(note_calls == 4 && noted_count == NA + NB + G1 + G2 &&
          noted[0] == 6 && noted[NA] == 1 && noted[NA + G1] == 3 &&
          noted[NA + G1 + NB] == 2 &&
          m.dh_n == 6 && !memcmp(ids, expected_ids, sizeof(expected_ids)),
          "reference replay preserves complete prompt/turn/token order and case-specific buffered IDs");
    CHECK(legacy_calls == old_legacy + 1 && reply_calls == old_reply + 2 &&
          m.dh_adaptive == observations && frees == old_frees && forwards == old_forwards,
          "reused plain reference keeps adaptive observations and adds no scoring or target forward");
    janas_llm_mtp_vocab_free(&m);
}
int main(void)
{
    char base[128]; snprintf(base, sizeof(base), "session-matrix-unit-%lu", (unsigned long)matrix_pid());
    CHECK(directory_new(base) == 0, "isolated test directory created");
    parser_tests(base); boundary_tests(); cache_tests(base); reference_observation_tests();
    CHECK(matrix_rmdir(base) == 0, "temporary test files removed");
    printf("test_llm_session_matrix: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
