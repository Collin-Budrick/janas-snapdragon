/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Public runtime configuration and bounded metrics copies, without weights.
 * The metrics implementation is compiled unchanged with three local data
 * providers, so a successful prefix copy never dereferences a model/session.
 */
#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/chat.h"
#include "llm/spec_config.h"

static struct janas_gen_metrics fixture_gen;
static struct janas_gen_hybrid_metrics fixture_hybrid;
static void fixture_hybrid_metrics(const struct janas_llm_session *s,
                                     struct janas_gen_hybrid_metrics *out)
{ (void)s; *out = fixture_hybrid; }
static struct janas_llm_runtime_stats fixture_runtime;
static struct janas_mtp_costs fixture_mtp;
static void fixture_session_metrics(const struct janas_llm_session *s,
                                      struct janas_gen_metrics *out)
{ (void)s; *out = fixture_gen; }
static void fixture_runtime_stats(const struct janas_llm_model *m,
                                   struct janas_llm_runtime_stats *out)
{ (void)m; *out = fixture_runtime; }
static void fixture_mtp_costs(const struct janas_llm_model *m,
                               struct janas_mtp_costs *out)
{ (void)m; *out = fixture_mtp; }
#define janas_llm_session_hybrid_metrics fixture_hybrid_metrics
#define janas_llm_session_metrics fixture_session_metrics
#define janas_llm_model_runtime_stats fixture_runtime_stats
#define janas_llm_mtp_costs fixture_mtp_costs
#include "../src/llm/api_spec.c"
#undef janas_llm_session_hybrid_metrics
#undef janas_llm_session_metrics
#undef janas_llm_model_runtime_stats
#undef janas_llm_mtp_costs

/* Frozen pre-runtime-controls ABI, commit 5cb4457. These existing layouts
 * must not grow when the independent size/version API gains fields. */
_Static_assert(sizeof(struct janas_llm_chat_params) == 80, "chat params ABI changed");
_Static_assert(offsetof(struct janas_llm_chat_params, seed) == 24, "seed ABI changed");
_Static_assert(sizeof(struct janas_llm_chat_stats) == 192, "chat stats ABI changed");
_Static_assert(offsetof(struct janas_llm_chat_stats, reasoning_cut) == 188,
               "chat stats tail ABI changed");
_Static_assert(sizeof(struct janas_llm_speculation_cycle) == sizeof(struct janas_gen_cycle),
               "cycle metrics mirror changed");
_Static_assert(offsetof(struct janas_llm_speculation_metrics, expert_assignments) -
               offsetof(struct janas_llm_speculation_metrics, cycles) ==
               sizeof(struct janas_gen_metrics), "metrics payload mirror changed");

static int failures;
#define CHECK(c, text) do { if (!(c)) { \
    printf("FAIL: %s (line %d)\n", text, __LINE__); failures++; \
} } while (0)

static const char *const env_names[] = {
    "JANAS_SPEC_PLANNER", "JANAS_SPEC_EXPLORE", "JANAS_DRAFT_VOCAB_POLICY",
    "JANAS_DRAFT_VOCAB_ROWS", "JANAS_HYBRID_POLICY", "JANAS_LOOKUP_DRAFTS",
    "JANAS_MTP_DRAFTS", "JANAS_HYBRID_PROBE_PERCENT"
};

static void clear_env(void)
{
    for (unsigned i = 0; i < sizeof(env_names) / sizeof(env_names[0]); i++) CHECK(unsetenv(env_names[i]) == 0, "clear environment");
}

static void check_unchanged(const struct janas_llm_chat *c,
                             const struct janas_llm_chat *before)
{
    CHECK(memcmp(&c->p, &before->p, sizeof(c->p)) == 0, "invalid update preserves chat params");
    CHECK(memcmp(&c->spec_config, &before->spec_config, sizeof(c->spec_config)) == 0,
          "invalid update preserves runtime config");
    CHECK(c->max_drafts == before->max_drafts, "invalid update preserves draft ceiling");
}

static void invalid_config(struct janas_llm_chat *c,
                            const struct janas_llm_speculation_config *p)
{
    struct janas_llm_chat before = *c;
    CHECK(janas_llm_chat_set_speculation_config(c, p) == JANAS_LLM_EINVAL,
          "invalid public config rejected");
    check_unchanged(c, &before);
}

static void config_tests(struct janas_llm_chat *c)
{
    struct janas_llm_speculation_config p;
    memset(&p, 0xa5, sizeof(p));
    janas_llm_speculation_config_default(&p);
    CHECK(p.size == sizeof(p) && p.version == JANAS_LLM_SPEC_CONFIG_VERSION,
          "config default initializes size and version");
    CHECK(p.mode == JANAS_LLM_SPEC_AUTO && p.max_drafts == 0 &&
          p.planner == JANAS_LLM_SPEC_PLANNER_LEGACY && p.explore_percent == 20 &&
          p.draft_vocab_policy == JANAS_LLM_DRAFT_VOCAB_LEGACY && p.draft_vocab_rows == 0,
          "defaults preserve legacy policy with dormant exploration setting");
    janas_llm_speculation_config_default(NULL);
    CHECK(janas_llm_chat_set_speculation_config(NULL, &p) == JANAS_LLM_EINVAL,
          "null chat rejected");
    invalid_config(c, NULL);
    CHECK(janas_llm_chat_set_speculation_config(c, &p) == JANAS_LLM_OK,
          "default config accepted without a session");
    struct janas_gen_options o;
    CHECK(janas_api_options(c, &o) == JANAS_LLM_OK &&
          o.spec_mode == JANAS_SPEC_LOOKUP && o.spec_planner == 0 &&
          o.draft_vocab_policy == 0, "legacy options remain legacy without MTP");

    struct janas_llm_speculation_config bad = p;
#define BAD(field, value) do { bad = p; bad.field = (value); invalid_config(c, &bad); } while (0)
    BAD(size, 0); BAD(size, sizeof(p) - 1); BAD(size, sizeof(p) + 1);
    BAD(version, 0); BAD(version, JANAS_LLM_SPEC_CONFIG_VERSION + 1);
    BAD(mode, -1); BAD(mode, JANAS_LLM_SPEC_HYBRID + 1);
    BAD(max_drafts, -1); BAD(max_drafts, 17);
    BAD(planner, -1); BAD(planner, 3);
    BAD(explore_percent, -1); BAD(explore_percent, 101);
    BAD(draft_vocab_policy, -1); BAD(draft_vocab_policy, 2);
    BAD(draft_vocab_rows, 1); BAD(draft_vocab_rows, UINT32_MAX);
#undef BAD
    bad = p; bad.mode = JANAS_LLM_SPEC_HYBRID;
    invalid_config(c, &bad);
    bad.mode = JANAS_LLM_SPEC_MTP;
    invalid_config(c, &bad);
    c->llm->has_mtp = 1;
    p.mode = JANAS_LLM_SPEC_HYBRID; p.max_drafts = 16;
    p.draft_vocab_policy = JANAS_LLM_DRAFT_VOCAB_ADAPTIVE;
    p.draft_vocab_rows = 16384;
    CHECK(janas_llm_chat_set_speculation_config(c, &p) == JANAS_LLM_OK &&
          janas_api_options(c, &o) == JANAS_LLM_OK &&
          o.spec_mode == JANAS_SPEC_HYBRID && o.spec_k == 16 && o.spec_planner == 1 &&
          o.spec_explore_percent == 20 && o.draft_vocab_policy == 1 &&
          o.draft_vocab_rows == 16384, "hybrid legacy resolves cycle and preserves head policy");
    p.mode = JANAS_LLM_SPEC_MTP;
    CHECK(janas_llm_chat_set_speculation_config(c, &p) == JANAS_LLM_OK &&
          janas_api_options(c, &o) == JANAS_LLM_OK && o.spec_mode == JANAS_SPEC_MTP &&
          o.spec_planner == JANAS_LLM_SPEC_PLANNER_CYCLE,
          "adaptive head with legacy planner resolves cycle for MTP");
    p.mode = JANAS_LLM_SPEC_HYBRID;
    p.planner = JANAS_LLM_SPEC_PLANNER_FIXED; p.explore_percent = 0;
    CHECK(janas_llm_chat_set_speculation_config(c, &p) == JANAS_LLM_OK &&
          janas_api_options(c, &o) == JANAS_LLM_OK && o.spec_planner == 2 &&
          o.spec_explore_percent == 0, "explicit fixed planner and zero exploration preserved");
    const uint32_t rows[] = {0, 8192, 16384, 32768};
    for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        p.draft_vocab_rows = rows[i];
        CHECK(janas_llm_chat_set_speculation_config(c, &p) == JANAS_LLM_OK,
              "supported draft row budget accepted");
    }
}

static void parser_tests(struct janas_llm_chat *c)
{
    char error[128];
    struct janas_llm_speculation_config p, before;
    clear_env();
    janas_llm_speculation_config_default(&p); before = p;
    CHECK(janas_spec_config_env(&p, error, sizeof(error)) == 0 &&
          memcmp(&p, &before, sizeof(p)) == 0, "absent environment preserves defaults");
    CHECK(!janas_spec_runtime_is_option(NULL) &&
          !janas_spec_runtime_is_option("--unknown") &&
          janas_spec_runtime_is_option("--spec-planner"), "runtime option discovery is exact");
    CHECK(janas_spec_runtime_option("--unknown", "cycle", error, sizeof(error)) == 0,
          "unrelated option left to caller");
    CHECK(janas_spec_runtime_option("--spec-planner", "cycle", error, sizeof(error)) == 1 &&
          janas_spec_runtime_option("--spec-explore", "0", error, sizeof(error)) == 1 &&
          janas_spec_runtime_option("--draft-vocab", "adaptive", error, sizeof(error)) == 1 &&
          janas_spec_runtime_option("--draft-vocab-rows", "8192", error, sizeof(error)) == 1,
          "valid CLI helpers set all runtime controls");
    CHECK(janas_spec_config_env(&p, error, sizeof(error)) == 0 && p.planner == 1 &&
          p.explore_percent == 0 && p.draft_vocab_policy == 1 && p.draft_vocab_rows == 8192,
          "environment controls translate without overwriting explicit zero");
    const char *bad_values[] = {"", "-1", "101", "1x", "1.0", "999999999999999999999999"};
    for (unsigned i = 0; i < sizeof(bad_values) / sizeof(*bad_values); i++) {
        CHECK(janas_spec_runtime_option("--spec-explore", bad_values[i], error, sizeof(error)) == -1 &&
              strcmp(getenv("JANAS_SPEC_EXPLORE"), "0") == 0,
              "invalid CLI numeric value preserves environment");
    }
    CHECK(janas_spec_runtime_option("--draft-vocab", "bogus", NULL, 0) == -1 &&
          strcmp(getenv("JANAS_DRAFT_VOCAB_POLICY"), "adaptive") == 0,
          "invalid enum preserves environment with zero-sized diagnostics");
    CHECK(janas_spec_runtime_option("--draft-vocab-rows", "4096", error, sizeof(error)) == -1 &&
          strcmp(getenv("JANAS_DRAFT_VOCAB_ROWS"), "8192") == 0,
          "unsupported row budget rejected atomically");
    before = p;
    CHECK(setenv("JANAS_SPEC_PLANNER", "fixed", 1) == 0 &&
          setenv("JANAS_DRAFT_VOCAB_ROWS", "16385", 1) == 0, "set late invalid environment");
    CHECK(janas_spec_config_env(&p, error, sizeof(error)) == -1 &&
          memcmp(&p, &before, sizeof(p)) == 0, "late parse error leaves whole config unchanged");
    struct janas_llm_chat saved = *c;
    CHECK(janas_spec_runtime_apply(c, JANAS_LLM_SPEC_HYBRID, 4) == JANAS_LLM_EINVAL,
          "runtime apply rejects malformed environment");
    check_unchanged(c, &saved);
    clear_env();
    CHECK(janas_spec_runtime_apply(c, JANAS_LLM_SPEC_HYBRID, 4) == JANAS_LLM_OK &&
          c->max_drafts == 4, "runtime apply accepts defaults after bad environment removed");
}

static void metrics_tests(struct janas_llm_chat *c)
{
    struct { struct janas_llm_speculation_metrics m; unsigned char tail[16]; } box, before;
    memset(&box, 0xa5, sizeof(box));
    box.m.size = sizeof(box.m); box.m.version = JANAS_LLM_SPEC_METRICS_VERSION;
    before = box;
    CHECK(janas_llm_chat_speculation_metrics(c, &box.m) == JANAS_LLM_EINVAL &&
          memcmp(&box, &before, sizeof(box)) == 0, "metrics without session fail without writes");
    unsigned marker = 0;
    c->s = (struct janas_llm_session *)&marker;
    CHECK(janas_llm_chat_speculation_metrics(NULL, &box.m) == JANAS_LLM_EINVAL &&
          janas_llm_chat_speculation_metrics(c, NULL) == JANAS_LLM_EINVAL,
          "null metrics arguments rejected");
    const uint32_t bad_sizes[] = {0, 4, sizeof(box.m) + 1, UINT32_MAX};
    for (unsigned i = 0; i < sizeof(bad_sizes) / sizeof(bad_sizes[0]); i++) {
        box = before; box.m.size = bad_sizes[i];
        struct janas_llm_speculation_metrics unchanged = box.m;
        CHECK(janas_llm_chat_speculation_metrics(c, &box.m) == JANAS_LLM_EINVAL &&
              memcmp(&box.m, &unchanged, sizeof(unchanged)) == 0,
              "unknown metric size fails without mutation");
    }
    box = before; box.m.version++;
    struct janas_llm_speculation_metrics unchanged = box.m;
    CHECK(janas_llm_chat_speculation_metrics(c, &box.m) == JANAS_LLM_EINVAL &&
          memcmp(&box.m, &unchanged, sizeof(unchanged)) == 0,
          "unknown metrics version rejected atomically");
    fixture_gen.cycles = 10; c->gm0.cycles = 3;
    fixture_gen.source_cycles[2] = 8; c->gm0.source_cycles[2] = 2;
    fixture_gen.lookup_seconds = 3; c->gm0.lookup_seconds = 1;
    fixture_gen.buffered_tokens = 3; c->gm0.buffered_tokens = 20;
    fixture_gen.pending_preparation_seconds = 1.5; c->gm0.pending_preparation_seconds = 8;
    fixture_gen.last.proposal_id = 42; fixture_gen.last.pass_width = 5;
    fixture_runtime.expert_assignments = 100; c->rt0.expert_assignments = 10;
    fixture_runtime.expert_group_widths[4] = 9; c->rt0.expert_group_widths[4] = 2;
    fixture_runtime.gdn_delta_bytes = 4096;
    fixture_mtp.repacks = 5; c->mt0.repacks = 1;
    fixture_mtp.vocab_policy = 1; fixture_mtp.vocab_rows = 16384; fixture_mtp.vocab_epoch = 9;
    box = before;
    CHECK(janas_llm_chat_speculation_metrics(c, &box.m) == JANAS_LLM_OK &&
          box.m.cycles == 7 && box.m.source_cycles[2] == 6 && box.m.lookup_seconds == 2 &&
          box.m.expert_assignments == 90 && box.m.expert_group_widths[4] == 7 &&
          box.m.head_repacks == 4, "metrics report reply-local scalar and array deltas");
    CHECK(box.m.buffered_tokens == 3 && box.m.pending_preparation_seconds == 1.5 &&
          box.m.gdn_delta_bytes == 4096 && box.m.head_vocab_rows == 16384 &&
          box.m.head_vocab_epoch == 9 && box.m.last.proposal_id == 42,
          "current gauges and last cycle are copied without subtraction");
    for (unsigned i = 0; i < sizeof(box.tail); i++)
        CHECK(box.tail[i] == 0xa5, "full-size metrics do not overwrite trailing bytes");
    const uint32_t prefixes[] = {8, 9, offsetof(struct janas_llm_speculation_metrics, last),
        offsetof(struct janas_llm_speculation_metrics, expert_assignments)};
    for (unsigned i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        box = before; box.m.size = prefixes[i];
        for (unsigned call = 0; call < 2; call++) {
            CHECK(janas_llm_chat_speculation_metrics(c, &box.m) == JANAS_LLM_OK &&
                  box.m.size == prefixes[i] &&
                  box.m.version == JANAS_LLM_SPEC_METRICS_VERSION,
                  "metrics prefix preserves caller capacity across repeated calls");
            const unsigned char *bytes = (const unsigned char *)&box;
            for (size_t j = prefixes[i]; j < sizeof(box); j++)
                CHECK(bytes[j] == 0xa5, "repeated prefix copy respects exact caller boundary");
        }
    }
    fixture_gen.cycles = c->gm0.cycles;
    box = before;
    CHECK(janas_llm_chat_speculation_metrics(c, &box.m) == JANAS_LLM_OK &&
          box.m.last.proposal_id == 0 && box.m.last.pass_width == 0,
          "no new cycles never exposes previous reply's last cycle");
    c->s = NULL;
}

static void hybrid_tests(struct janas_llm_chat *c)
{
    clear_env();
    struct janas_llm_hybrid_config p;
    janas_llm_hybrid_config_default(&p);
    CHECK(p.size == sizeof p && p.version == 1 && p.policy == 0 &&
          p.lookup_max_drafts == 8 && p.mtp_max_drafts == 4 && p.probe_percent == 2,
          "independent hybrid defaults preserve old behavior");
    CHECK(janas_llm_chat_set_hybrid_config(c, &p) == JANAS_LLM_OK, "hybrid default accepted");
    struct janas_llm_hybrid_config bad = p;
#define H_BAD(field, value) do { bad = p; bad.field = (value); \
    struct janas_llm_hybrid_config old = c->hybrid_config; \
    CHECK(janas_llm_chat_set_hybrid_config(c, &bad) == JANAS_LLM_EINVAL && \
          memcmp(&old, &c->hybrid_config, sizeof old) == 0, "hybrid invalid atomic"); } while (0)
    H_BAD(size, sizeof p - 1); H_BAD(version, 2); H_BAD(policy, 2);
    H_BAD(lookup_max_drafts, 0); H_BAD(lookup_max_drafts, 17);
    H_BAD(mtp_max_drafts, 0); H_BAD(mtp_max_drafts, 17);
    H_BAD(probe_percent, -1); H_BAD(probe_percent, 101);
#undef H_BAD
    char error[160];
    CHECK(janas_spec_runtime_option("--hybrid-policy", "cost", error, sizeof error) == 1 &&
          janas_spec_runtime_option("--lookup-drafts", "8", error, sizeof error) == 1 &&
          janas_spec_runtime_option("--mtp-drafts", "4", error, sizeof error) == 1 &&
          janas_spec_runtime_option("--hybrid-probe-percent", "2", error, sizeof error) == 1 &&
          janas_hybrid_config_env(&p, error, sizeof error) == 0 && p.policy == 1,
          "hybrid CLI environment round trip");
    struct janas_llm_hybrid_config before = p;
    CHECK(setenv("JANAS_MTP_DRAFTS", "0", 1) == 0 &&
          janas_hybrid_config_env(&p, error, sizeof error) == -1 &&
          memcmp(&p, &before, sizeof p) == 0, "hybrid environment parse atomic");
    clear_env();
    c->llm->has_mtp = 1;
    c->p.speculate = JANAS_LLM_SPEC_HYBRID;
    p = before;
    CHECK(janas_llm_chat_set_hybrid_config(c, &p) == JANAS_LLM_OK, "cost hybrid accepted");
    struct janas_gen_options o;
    CHECK(janas_api_options(c, &o) == JANAS_LLM_OK && o.hybrid_policy == 1 &&
          o.hybrid_lookup_max_drafts == 8 && o.hybrid_mtp_max_drafts == 4 &&
          o.hybrid_probe_percent == 2 && o.spec_planner == JANAS_LLM_SPEC_PLANNER_LEGACY,
          "cost policy preserves legacy fallback option and independent caps");
    c->s = (struct janas_llm_session *)(void *)c;
    fixture_hybrid.decisions = 10; c->hm0.decisions = 3;
    fixture_hybrid.lookup_selected = 5; c->hm0.lookup_selected = 2;
    fixture_hybrid.last.source = 1;
    for (uint32_t n = 8; n <= sizeof(struct janas_llm_hybrid_metrics); n++) {
        unsigned char *box = malloc((size_t)n + 16);
        CHECK(box != NULL, "hybrid metrics allocation");
        if (!box) break;
        memset(box, 0xa5, (size_t)n + 16);
        struct janas_llm_hybrid_metrics *m = (void *)box;
        m->size = n; m->version = 1;
        CHECK(janas_llm_chat_hybrid_metrics(c, m) == JANAS_LLM_OK && m->size == n,
              "hybrid metrics every prefix supported");
        for (unsigned j = 0; j < 16; j++) CHECK(box[n+j] == 0xa5, "hybrid metric guard untouched");
        free(box);
    }
    struct janas_llm_hybrid_metrics m = {.size = sizeof m, .version = 1};
    CHECK(janas_llm_chat_hybrid_metrics(c, &m) == 0 && m.decisions == 7 &&
          m.lookup_selected == 3, "hybrid counters reply-local");
    struct janas_llm_hybrid_metrics unchanged = m;
    m.version = 2; unchanged.version = 2;
    CHECK(janas_llm_chat_hybrid_metrics(c, &m) == JANAS_LLM_EINVAL &&
          memcmp(&m, &unchanged, sizeof m) == 0, "hybrid metrics unknown version atomic");
    c->s = NULL;
}

int main(void)
{
    char *saved_env[sizeof(env_names) / sizeof(env_names[0])] = {0};
    for (unsigned i = 0; i < sizeof(env_names) / sizeof(env_names[0]); i++) {
        const char *v = getenv(env_names[i]);
        if (v) {
            saved_env[i] = malloc(strlen(v) + 1);
            if (!saved_env[i]) return 1;
            strcpy(saved_env[i], v);
        }
    }
    struct janas_llm llm = {0};
    struct janas_llm_chat c = {.llm = &llm};
    janas_llm_chat_params_default(&c.p);
    config_tests(&c);
    parser_tests(&c);
    metrics_tests(&c);
    hybrid_tests(&c);
    for (unsigned i = 0; i < sizeof(env_names) / sizeof(env_names[0]); i++) {
        if (saved_env[i]) CHECK(setenv(env_names[i], saved_env[i], 1) == 0, "restore parent environment");
        else CHECK(unsetenv(env_names[i]) == 0, "restore absent parent environment");
        free(saved_env[i]);
    }
    if (failures) { printf("test_spec_config: %d failures\n", failures); return 1; }
    puts("test_spec_config: ok");
    return 0;
}
