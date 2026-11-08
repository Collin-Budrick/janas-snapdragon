/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Real generator/sampler contracts against a tiny, deterministic model.
 * Target logits depend only on the current token/position, so a block and
 * its token-by-token execution have exactly the same rows. No weights or
 * hardware are used. All model dependencies are local mocks, including
 * unsupported saved-state paths, so ordinary archive linkage stays isolated. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>

#include "llm/model.h"

#define MOCK_VOCAB 16
#define MOCK_CTX 256
struct janas_llm_model {
    uint32_t ctx, period, last_pos, last_n, max_width;
    int bad_drafts, peaky;
    int invalid_calls, invalid_prepare, prepare_calls, feedback_calls;
    uint32_t forward_calls, mtp_calls, mtp_rows, note_rows;
    int32_t last_tokens[JANAS_LLM_MAX_BLOCK];
    int32_t mtp_tokens[MOCK_CTX + JANAS_LLM_MAX_BLOCK];
    float hidden[JANAS_LLM_MAX_BLOCK], mtp_hidden[JANAS_LLM_MAX_BLOCK];
    float *last_logits;
    struct janas_mtp_costs costs;
    struct janas_llm_runtime_stats runtime;
};

#include "../src/llm/generate.c"

static int failures;
#define CHECK(c, text) do { if (!(c)) { \
    printf("FAIL: %s (line %d)\n", text, __LINE__); failures++; \
} } while (0)

static void model_init(struct janas_llm_model *m)
{
    memset(m, 0, sizeof(*m));
    m->ctx = MOCK_CTX;
    m->period = MOCK_VOCAB;
    for (size_t i = 0; i < sizeof(m->mtp_tokens) / sizeof(*m->mtp_tokens); i++)
        m->mtp_tokens[i] = -1;
}

static int32_t next_token(const struct janas_llm_model *m, int32_t token)
{
    return (token + 1) % (int32_t)m->period;
}

static void target_row(const struct janas_llm_model *m, int32_t token,
                       uint32_t pos, float *logits)
{
    for (int i = 0; i < MOCK_VOCAB; i++)
        logits[i] = m->peaky ? -30.0f : -3.0f + (float)((pos + (uint32_t)i) % 3) * 0.1f;
    logits[next_token(m, token)] = 3.5f;
    if (!m->peaky) {
        logits[(token + 3) % MOCK_VOCAB] = 2.7f;
        logits[(token + 7) % MOCK_VOCAB] = 1.8f;
    }
}

uint32_t janas_llm_model_n_vocab(const struct janas_llm_model *m)
{ (void)m; return MOCK_VOCAB; }
uint32_t janas_llm_model_n_ctx(const struct janas_llm_model *m)
{ return m->ctx; }
uint32_t janas_llm_model_hidden_width(const struct janas_llm_model *m)
{ (void)m; return 1; }
uint32_t janas_llm_model_max_block(const struct janas_llm_model *m)
{ (void)m; return JANAS_LLM_MAX_BLOCK; }
uint32_t janas_llm_model_pass_cap(const struct janas_llm_model *m)
{ (void)m; return JANAS_LLM_MAX_BLOCK; }
int janas_llm_model_has_mtp(const struct janas_llm_model *m)
{ (void)m; return 1; }
int janas_llm_mtp_chain_fixed(const struct janas_llm_model *m)
{ (void)m; return 0; }
int janas_llm_model_recurrent(const struct janas_llm_model *m)
{ (void)m; return 1; }
int janas_llm_model_settling(const struct janas_llm_model *m)
{ (void)m; return 0; }
double janas_llm_model_pass_seconds(const struct janas_llm_model *m)
{ (void)m; return 0.001; }
const float *janas_llm_model_hidden(const struct janas_llm_model *m)
{ return m->hidden; }
const float *janas_llm_mtp_hidden(const struct janas_llm_model *m)
{ return m->mtp_hidden; }
int janas_llm_model_cache_file(const struct janas_llm_model *m,
                               const char *what, char *path, size_t len, int create)
{ (void)m; (void)what; (void)path; (void)len; (void)create; return -1; }
uint32_t janas_llm_model_media_cut(const struct janas_llm_model *m,
                                   uint32_t pos, uint32_t n)
{ (void)m; (void)pos; return n; }
void janas_llm_model_media_drop(struct janas_llm_model *m, uint32_t from)
{ (void)m; (void)from; }
uint32_t janas_llm_model_media_count(const struct janas_llm_model *m)
{ (void)m; return 0; }
int janas_llm_model_shift(struct janas_llm_model *m, uint32_t keep,
                          uint32_t drop, uint32_t n_kv)
{ (void)m; return keep <= n_kv && drop <= n_kv - keep ? 0 : -1; }
/* Saved-state paths are deliberately unsupported by this tiny mock. Keep
 * every generator reference local so the ordinary archive link cannot pull
 * the real model object (and its duplicate target functions) into the test. */
struct janas_llm_state *janas_llm_model_state_save(struct janas_llm_model *m,
                                                   uint32_t n)
{ (void)m; (void)n; return NULL; }
int janas_llm_model_state_load(struct janas_llm_model *m,
                               const struct janas_llm_state *st)
{ (void)m; (void)st; return -1; }
uint32_t janas_llm_state_length(const struct janas_llm_state *st)
{ (void)st; return 0; }
size_t janas_llm_state_bytes(const struct janas_llm_state *st)
{ (void)st; return 0; }
void janas_llm_state_free(struct janas_llm_state *st)
{ (void)st; }
int janas_llm_state_write(const struct janas_llm_model *m,
                          const struct janas_llm_state *st, FILE *f)
{ (void)m; (void)st; (void)f; return -1; }
struct janas_llm_state *janas_llm_state_read(const struct janas_llm_model *m, FILE *f)
{ (void)m; (void)f; return NULL; }
void janas_llm_model_runtime_stats(const struct janas_llm_model *m,
                                  struct janas_llm_runtime_stats *out)
{ if (out) *out = m->runtime; }

int janas_llm_model_forward(struct janas_llm_model *m, const int32_t *tokens,
                             uint32_t n, uint32_t pos, float *logits, int all)
{
    if (!n || n > JANAS_LLM_MAX_BLOCK || pos + n > m->ctx ||
        (m->forward_calls && pos &&
         (pos < m->last_pos || pos > m->last_pos + m->last_n))) {
        m->invalid_calls++;
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= MOCK_VOCAB) {
            m->invalid_calls++;
            return -1;
        }
        m->hidden[i] = (float)tokens[i];
        if (logits && (all || i + 1 == n))
            target_row(m, tokens[i], pos + i,
                       logits + (all ? (size_t)i * MOCK_VOCAB : 0));
    }
    memcpy(m->last_tokens, tokens, n * sizeof(*tokens));
    m->last_pos = pos;
    m->last_n = n;
    m->last_logits = logits;
    m->forward_calls++;
    /* Two observed assignments per group: a constant high-reuse context.
     * Counters advance only after target work, never from a future router. */
    m->runtime.expert_assignments += (uint64_t)n * 2;
    m->runtime.expert_groups += n;
    m->runtime.expert_group_widths[2] += n;
    if (n > m->max_width) m->max_width = n;
    return 0;
}

int janas_llm_model_decode(struct janas_llm_model *m, int32_t token,
                            uint32_t pos, float *logits)
{ return janas_llm_model_forward(m, &token, 1, pos, logits, 1); }

int janas_llm_mtp_forward(struct janas_llm_model *m, const int32_t *tokens,
                           const float *hidden, uint32_t n, uint32_t pos,
                           float *logits, int all)
{
    (void)hidden;
    if (!n || n > JANAS_LLM_MAX_BLOCK || pos + n >
        sizeof(m->mtp_tokens) / sizeof(*m->mtp_tokens)) {
        m->invalid_calls++;
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        m->mtp_hidden[i] = (float)tokens[i];
        m->mtp_tokens[pos + i] = tokens[i];
        if (logits && (all || i + 1 == n))
            target_row(m, tokens[i], pos + i + 1,
                       logits + (all ? (size_t)i * MOCK_VOCAB : 0));
    }
    m->mtp_calls++;
    m->mtp_rows += n;
    return 0;
}

int janas_llm_mtp_draft(struct janas_llm_model *m, const int32_t *tokens,
                         const float *hidden, uint32_t n, uint32_t pos,
                         int32_t *token, float *conf)
{
    if (janas_llm_mtp_forward(m, tokens, hidden, n, pos, NULL, 0) != 0)
        return -1;
    *token = m->bad_drafts ? (tokens[n - 1] + 9) % MOCK_VOCAB
                          : next_token(m, tokens[n - 1]);
    *conf = 0.99f;
    m->costs.head_calls++;
    m->costs.head_seconds += 1e-6;
    return 0;
}
void janas_llm_mtp_note_tokens(struct janas_llm_model *m,
                               const int32_t *tokens, uint32_t n)
{ (void)tokens; m->note_rows += n; }
void janas_llm_mtp_costs(const struct janas_llm_model *m, struct janas_mtp_costs *out)
{ *out = m->costs; }
int janas_llm_mtp_vocab_prepare(struct janas_llm_model *m, int policy,
                                uint32_t rows, const float *logits,
                                const int32_t *history, uint32_t n,
                                const int32_t *special, uint32_t ns)
{
    (void)history; (void)special; (void)ns;
    m->prepare_calls++;
    if (n < m->last_pos + 2) {
        m->invalid_prepare++;
        return -1;
    }
    uint32_t row = n - m->last_pos - 2;
    if (row >= m->last_n || logits != m->last_logits + (size_t)row * MOCK_VOCAB)
        m->invalid_prepare++;
    m->costs.vocab_policy = (uint32_t)policy;
    m->costs.vocab_rows = policy ? (rows ? rows : 16384) : 32768;
    return 0;
}
void janas_llm_mtp_vocab_feedback(struct janas_llm_model *m, uint32_t policy,
                                  uint32_t rows, uint32_t drafted,
                                  uint32_t accepted, double seconds)
{
    (void)policy; (void)rows; (void)seconds;
    if (accepted > drafted) m->invalid_calls++;
    m->feedback_calls++;
}
void janas_llm_mtp_vocab_feedback_epoch(struct janas_llm_model *m,
                                       uint64_t epoch, uint32_t policy,
                                       uint32_t rows, uint32_t drafted,
                                       uint32_t accepted, double seconds)
{
    (void)epoch;
    janas_llm_mtp_vocab_feedback(m, policy, rows, drafted, accepted, seconds);
}

void janas_llm_mtp_vocab_legacy(struct janas_llm_model *m)
{ m->costs.vocab_policy = 0; m->costs.vocab_rows = 32768; }
void janas_llm_mtp_vocab_new_reply(struct janas_llm_model *m)
{ (void)m; }
uint32_t janas_llm_mtp_vocab_rows(const struct janas_llm_model *m)
{ return m->costs.vocab_rows; }
int janas_llm_mtp_vocab_policy(const struct janas_llm_model *m)
{ return (int)m->costs.vocab_policy; }

struct filter_state { unsigned accepts, allows; };
static int filter_active(void *ctx)
{ (void)ctx; return 1; }
static int filter_allows(void *ctx, int32_t token)
{
    struct filter_state *f = ctx;
    f->allows++;
    return token != (int32_t)(f->accepts % MOCK_VOCAB);
}
static void filter_accept(void *ctx, int32_t token)
{ (void)token; ((struct filter_state *)ctx)->accepts++; }

static struct janas_gen_options options(int mode, int planner, float temperature)
{
    struct janas_gen_options o = {0};
    o.spec_mode = mode;
    o.spec_planner = planner;
    o.spec_k = 4;
    o.spec_explore_percent = 20;
    o.eos = -1;
    o.temperature = temperature;
    o.top_p = 0.95f;
    o.seed = 42;
    return o;
}

static void canonical_mtp(const struct janas_llm_session *s,
                          const struct janas_llm_model *m)
{
    for (uint32_t i = 0; i + 1 < s->n; i++)
        CHECK(m->mtp_tokens[i] == s->hist[i + 1],
              "MTP canonical prefix maintained through accepted/rejected rows");
}

static void paired_transcript(int mode, int planner, int bad, float temperature,
                              int guarded, int adaptive)
{
    struct janas_llm_model a, b;
    model_init(&a); model_init(&b); b.bad_drafts = bad;
    struct janas_gen_options oa = options(JANAS_SPEC_OFF, 0, temperature);
    struct janas_gen_options ob = options(mode, planner, temperature);
    if (guarded & 2) {
        oa.presence_penalty = ob.presence_penalty = 0.3f;
        oa.frequency_penalty = ob.frequency_penalty = 0.2f;
    }
    ob.draft_vocab_policy = adaptive;
    struct janas_llm_session *sa = janas_llm_session_create(&a, &oa);
    struct janas_llm_session *sb = janas_llm_session_create(&b, &ob);
    CHECK(sa && sb, "paired sessions created");
    if (!sa || !sb) {
        janas_llm_session_destroy(sa); janas_llm_session_destroy(sb); return;
    }
    struct filter_state fa = {0}, fb = {0};
    struct janas_sample_filter f_a = {&fa, filter_active, filter_allows, filter_accept};
    struct janas_sample_filter f_b = {&fb, filter_active, filter_allows, filter_accept};
    if (guarded & 1) {
        janas_llm_session_filter(sa, &f_a);
        janas_llm_session_filter(sb, &f_b);
    }
    int32_t prompt[] = {1, 2, 3, 4, 1, 2, 3, 4};
    int32_t turn[] = {8, 9, 10};
    for (int reply = 0; reply < 2; reply++) {
        const int32_t *p = reply ? turn : prompt;
        uint32_t n = reply ? 3 : 8;
        CHECK(janas_llm_session_append(sa, p, n) == 0 &&
              janas_llm_session_append(sb, p, n) == 0, "same canonical turn appended");
        janas_llm_session_mark(sa); janas_llm_session_mark(sb);
        janas_llm_session_reply_remaining(sa, 72);
        janas_llm_session_reply_remaining(sb, 72);
        for (int i = 0; i < 72; i++) {
            int32_t ta = -1, tb = -1;
            int ra = janas_llm_session_next(sa, &ta);
            int rb = janas_llm_session_next(sb, &tb);
            CHECK(ra == 0 && rb == 0 && ta == tb,
                  "exact greedy/seeded transcript across planner, source and rejection");
        }
        CHECK(sa->n == sb->n && sa->ret == sb->ret &&
              memcmp(sa->hist, sb->hist, sa->n * sizeof(*sa->hist)) == 0,
              "reply hint prevents unread overproduction");
        canonical_mtp(sb, &b);
    }
    struct janas_gen_metrics mt;
    janas_llm_session_metrics(sb, &mt);
    CHECK(mt.delivered_tokens == 144 && mt.produced_tokens == 144 &&
          mt.discarded_tokens == 0 && mt.buffered_tokens == 0,
          "cycle delivery/production counters conserve tokens");
    if (guarded) {
        for (int width = 2; width <= JANAS_GEN_MAX_K + 1; width++)
            CHECK(mt.pass_width_count[width] == 0,
                  "stateful filter/penalty guard uses one target row");
        if (guarded & 1)
            CHECK(fa.accepts == fb.accepts && fa.accepts == 144 && fa.allows == fb.allows,
                  "stateful filter observes only identical target picks");
        if (guarded & 1)
            CHECK(mt.filter_guard_cycles > 0 &&
                  (mt.last.guard_reason & JANAS_GEN_GUARD_FILTER),
                  "filter exactness guard is reported in cycle metrics");
        if (guarded & 2)
            CHECK(mt.penalty_guard_cycles > 0 &&
                  (mt.last.guard_reason & JANAS_GEN_GUARD_PENALTY),
                  "penalty exactness guard is reported in cycle metrics");
    } else if (temperature == 0 && !bad) {
        CHECK(mt.accepted_prefix > 0, "good drafts exercise multi-token acceptance");
    }
    if (mode == JANAS_SPEC_MTP && !guarded) {
        CHECK(mt.proposals > 0 && mt.source_cycles[JANAS_GEN_SOURCE_MTP] > 0 &&
              mt.source_drafted[JANAS_GEN_SOURCE_MTP] > 0 &&
              b.costs.head_calls > 0 && mt.mtp_head_seconds > 0,
              "legacy/cycle/fixed MTP reports proposals, source work and head phase");
    }
    CHECK(!a.invalid_calls && !b.invalid_calls && !b.invalid_prepare,
          "valid context/replay calls and canonical target-row prepare hook");
    if (adaptive && !guarded)
        CHECK(b.prepare_calls > 0, "adaptive hook exercised without changing target logits");
    janas_llm_session_destroy(sa); janas_llm_session_destroy(sb);
}

static void unread_rewind(float temperature)
{
    struct janas_llm_model a, b;
    model_init(&a); model_init(&b); a.peaky = b.peaky = 1;
    struct janas_gen_options oa = options(JANAS_SPEC_OFF, 0, temperature);
    struct janas_gen_options ob = options(JANAS_SPEC_MTP, 2, temperature);
    struct janas_llm_session *sa = janas_llm_session_create(&a, &oa);
    struct janas_llm_session *sb = janas_llm_session_create(&b, &ob);
    CHECK(sa && sb, "unread sessions created");
    if (!sa || !sb) { janas_llm_session_destroy(sa); janas_llm_session_destroy(sb); return; }
    int32_t prompt[] = {1, 2, 3}, next_turn[] = {8, 9};
    CHECK(janas_llm_session_append(sa, prompt, 3) == 0 &&
          janas_llm_session_append(sb, prompt, 3) == 0, "unread prompt appended");
    for (int i = 0; i < 2; i++) {
        int32_t ta, tb;
        CHECK(janas_llm_session_next(sa, &ta) == 0 &&
              janas_llm_session_next(sb, &tb) == 0 && ta == tb,
              "visible prefix identical before cancellation");
    }
    CHECK(sb->ret < sb->n, "fixture creates accepted but unread generated tokens");
    uint32_t unread = sb->n - sb->ret;
    CHECK(janas_llm_session_append(sa, next_turn, 2) == 0 &&
          janas_llm_session_append(sb, next_turn, 2) == 0, "new turn discards unread suffix");
    a.peaky = b.peaky = 0;
    janas_llm_session_mark(sa); janas_llm_session_mark(sb);
    janas_llm_session_reply_remaining(sa, 32); janas_llm_session_reply_remaining(sb, 32);
    for (int i = 0; i < 32; i++) {
        int32_t ta, tb;
        CHECK(janas_llm_session_next(sa, &ta) == 0 &&
              janas_llm_session_next(sb, &tb) == 0 && ta == tb,
              "new-turn sampler sequence exactly rewound after unread suffix");
    }
    struct janas_gen_metrics mt;
    janas_llm_session_metrics(sb, &mt);
    CHECK(mt.discarded_tokens == unread && mt.buffered_tokens == 0,
          "cancelled unread tokens accounted once");
    CHECK(!b.invalid_calls && !b.invalid_prepare, "unread replay stays within previous target call");
    canonical_mtp(sb, &b);
    janas_llm_session_destroy(sa); janas_llm_session_destroy(sb);
}

static void reseed_after_unread(void)
{
    struct janas_llm_model a, b;
    model_init(&a); model_init(&b); a.peaky = b.peaky = 1;
    struct janas_gen_options oa = options(JANAS_SPEC_OFF, 0, 1.0f);
    struct janas_gen_options ob = options(JANAS_SPEC_MTP, 2, 1.0f);
    struct janas_llm_session *sa = janas_llm_session_create(&a, &oa);
    struct janas_llm_session *sb = janas_llm_session_create(&b, &ob);
    CHECK(sa && sb, "reseed sessions created");
    if (!sa || !sb) { janas_llm_session_destroy(sa); janas_llm_session_destroy(sb); return; }
    int32_t prompt[] = {1, 2, 3}, turn[] = {8, 9};
    CHECK(janas_llm_session_append(sa, prompt, 3) == 0 &&
          janas_llm_session_append(sb, prompt, 3) == 0, "reseed prompt appended");
    for (int i = 0; i < 2; i++) {
        int32_t ta, tb;
        CHECK(janas_llm_session_next(sa, &ta) == 0 &&
              janas_llm_session_next(sb, &tb) == 0 && ta == tb,
              "same visible reply before reseeding");
    }
    uint32_t unread = sb->n - sb->ret;
    CHECK(unread > 0, "reseed fixture has old accepted unread tokens");
    oa.seed = ob.seed = UINT64_C(987654321);
    CHECK(janas_llm_session_set_options(sa, &oa) == 0 &&
          janas_llm_session_set_options(sb, &ob) == 0,
          "new seed applied after old unread suffix is discarded");
    CHECK(sb->n == sb->ret && sb->metrics.discarded_tokens == unread,
          "set_options discards old unread tokens exactly once before reseed");
    CHECK(janas_llm_session_append(sa, turn, 2) == 0 &&
          janas_llm_session_append(sb, turn, 2) == 0, "new turn appended after reseeding");
    a.peaky = b.peaky = 0;
    janas_llm_session_reply_remaining(sa, 32); janas_llm_session_reply_remaining(sb, 32);
    for (int i = 0; i < 32; i++) {
        int32_t ta, tb;
        CHECK(janas_llm_session_next(sa, &ta) == 0 &&
              janas_llm_session_next(sb, &tb) == 0 && ta == tb,
              "new-seed sampled turn matches plain reference after cancellation");
    }
    CHECK(sb->metrics.discarded_tokens == unread && !b.invalid_calls,
          "later append does not rewind or discard the old suffix again");
    janas_llm_session_destroy(sa); janas_llm_session_destroy(sb);
}

static void abandoned_cycle_observation(void)
{
    struct janas_llm_model m;
    model_init(&m);
    struct janas_gen_options o = options(JANAS_SPEC_HYBRID, 1, 0);
    struct janas_llm_session *s = janas_llm_session_create(&m, &o);
    CHECK(s != NULL, "abandoned-cycle session created");
    if (!s) return;
    /* One real preceding cycle is awaiting finalization, independently of
     * an already-prepared next proposal that cancellation will never run. */
    s->metrics.cycles = 11;
    s->metrics.cycle_seconds = 3.0;
    s->metrics.last.planner_source = JANAS_GEN_SOURCE_MTP;
    s->metrics.last.source = JANAS_GEN_SOURCE_PLAIN;
    s->metrics.last.warm = 1;
    s->metrics.last.requested_depth = 0;
    s->metrics.last.total_seconds = .5;
    s->metrics.last.delivered = 1;
    s->observation_pending = 1;
    s->observation_reuse = 0;
    s->pending.proposal_id = 77;
    s->pending.source = s->pending.planner_source = JANAS_GEN_SOURCE_MTP;
    s->pending.requested_depth = 4;
    s->pending.warm = 1;
    s->pending.draft_vocab_policy = 1;
    s->pending.draft_vocab_rows = 8192;
    s->pending.lookup_suffix = 3; /* lookup declined and fell back to MTP */
    s->pending.preparation_seconds = .25;
    s->pending_reuse = 1;
    s->n_mdraft = 4;
    janas_llm_session_reset(s);
    const struct janas_cycle_observation *current =
        &s->cycle.bucket[JANAS_GEN_SOURCE_MTP][1][0].depth[0];
    const struct janas_cycle_observation *abandoned =
        &s->cycle.bucket[JANAS_GEN_SOURCE_MTP][1][1].depth[4];
    const struct janas_cycle_observation *context =
        &s->cycle.reuse_bucket[JANAS_GEN_SOURCE_MTP][1][1][1].depth[4];
    const struct janas_cycle_observation *fallback =
        &s->cycle.bucket[JANAS_GEN_SOURCE_LOOKUP][1][0].depth[0];
    CHECK(current->count == 1 && current->seconds == .5 && current->delivered == 1,
          "preceding completed cycle retains independent cost/reward observation");
    CHECK(abandoned->count == 1 && abandoned->seconds == .25 && abandoned->delivered == 0 &&
          context->count == 1 && context->seconds == .25 && context->delivered == 0,
          "cancelled prepared action teaches its exact source, budget and reuse with zero reward");
    CHECK(fallback->count == 1 && fallback->seconds == .25 && fallback->delivered == 0,
          "cancelled hybrid fallback also teaches offered lookup-zero action");
    CHECK(s->metrics.abandoned_preparation_seconds == .25 &&
          s->metrics.cycles == 11 && s->metrics.cycle_seconds == 3.0,
          "abandoned preparation is reported without inventing a verification cycle");
    janas_llm_session_reset(s);
    CHECK(current->count == 1 && abandoned->count == 1 && fallback->count == 1 &&
          s->metrics.abandoned_preparation_seconds == .25,
          "repeated cancellation never double-counts abandoned action or prior cycle");
    janas_llm_session_destroy(s);
}

static void late_hybrid_lookup(void)
{
    struct janas_llm_model m;
    model_init(&m); m.peaky = 1;
    struct janas_gen_options o = options(JANAS_SPEC_HYBRID, 1, 0);
    o.spec_explore_percent = 0;
    struct janas_llm_session *s = janas_llm_session_create(&m, &o);
    CHECK(s != NULL, "late-lookup session created");
    if (!s) return;
    /* MTP completed its initial sweep before lookup had a useful suffix;
     * its mature zero-exploration policy always prefers a nonzero depth. */
    for (unsigned depth = 0; depth <= 4; depth++)
        for (int i = 0; i < 3; i++)
            janas_cycle_observe(&s->cycle, JANAS_GEN_SOURCE_MTP, 1, 0, depth,
                                1.0, depth == 4 ? 200 : 100);
    int32_t first[] = {10, 11, 12}, token;
    CHECK(janas_llm_session_append(s, first, 3) == 0 &&
          janas_llm_session_next(s, &token) == 0, "early target has no lookup continuation");
    CHECK(s->metrics.lookup_hits == 0 && s->metrics.lookup_misses > 0 &&
          s->pending.planner_source == JANAS_GEN_SOURCE_MTP &&
          s->pending.requested_depth == 4,
          "MTP is mature and nonzero before the first lookup hit");
    int32_t repeated[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    CHECK(janas_llm_session_append(s, repeated, 12) == 0, "late repetitive turn appended");
    m.period = 4;
    janas_llm_session_reply_remaining(s, 64);
    for (int i = 0; i < 64; i++)
        CHECK(janas_llm_session_next(s, &token) == 0 && token == i % 4,
              "late lookup preserves canonical target sequence");
    CHECK(s->metrics.lookup_hits > 0 &&
          s->cycle.bucket[JANAS_GEN_SOURCE_LOOKUP][1][0].depth[0].count >= 3 &&
          s->metrics.source_drafted[JANAS_GEN_SOURCE_LOOKUP] > 0,
          "late lookup-zero observations progress to real indexed drafts without random probes");
    CHECK(!m.invalid_calls && !m.invalid_prepare, "late hybrid transitions retain valid target state");
    janas_llm_session_destroy(s);
}

static void lookup_eos_context(void)
{
    struct janas_llm_model m;
    int32_t prompt[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    struct janas_gen_options o;
    struct janas_llm_session *s;
    for (int planner = 0; planner <= 2; planner++) {
        model_init(&m); m.period = 4;
        o = options(planner ? JANAS_SPEC_HYBRID : JANAS_SPEC_LOOKUP, planner, 0);
        s = janas_llm_session_create(&m, &o);
        CHECK(s && janas_llm_session_append(s, prompt, 12) == 0, "hybrid repeated prompt");
        if (!s) return;
        janas_llm_session_reply_remaining(s, 96);
        for (int i = 0; i < 96; i++) {
            int32_t token;
            CHECK(janas_llm_session_next(s, &token) == 0 && token == i % 4,
                  "hybrid indexed continuation preserves greedy target");
        }
        struct janas_gen_metrics mt;
        janas_llm_session_metrics(s, &mt);
        uint64_t suffixes = 0;
        for (int i = 1; i <= 8; i++) suffixes += mt.lookup_suffix_count[i];
        CHECK(mt.lookup_hits > 0 && mt.source_drafted[JANAS_GEN_SOURCE_LOOKUP] > 0 &&
              mt.source_cycles[JANAS_GEN_SOURCE_LOOKUP] > 0 && mt.proposals > 0 &&
              mt.lookup_seconds > 0 && m.mtp_rows >= 96 && (!planner || suffixes > 0),
              "cycle/fixed hybrid executes indexed lookup and maintains canonical MTP state");
        if (planner != 1)
            CHECK(m.costs.head_calls == 0,
                  "fixed hybrid lookup wins before any MTP head/chain");
        canonical_mtp(s, &m);
        janas_llm_session_destroy(s);
    }

    model_init(&m); o = options(JANAS_SPEC_MTP, 2, 0); o.eos = 7;
    s = janas_llm_session_create(&m, &o);
    int32_t start = 3;
    CHECK(s && janas_llm_session_append(s, &start, 1) == 0, "EOS fixture created");
    if (!s) return;
    int32_t token = -1;
    for (int i = 0; i < 4; i++)
        CHECK(janas_llm_session_next(s, &token) == 0 && token == i + 4,
              "EOS appears at identical canonical position");
    CHECK(token == 7 && s->n_mdraft == 0 && s->ret == s->n,
          "no proposal or buffered token survives EOS");
    janas_llm_session_destroy(s);

    model_init(&m); m.ctx = 8; o = options(JANAS_SPEC_MTP, 2, 0);
    s = janas_llm_session_create(&m, &o);
    int32_t near_end[] = {0, 1, 2, 3, 4, 5, 6};
    CHECK(s && janas_llm_session_append(s, near_end, 7) == 0, "context fixture created");
    if (!s) return;
    for (int i = 0; i < 2; i++)
        CHECK(janas_llm_session_next(s, &token) == 0, "remaining context token generated");
    CHECK(janas_llm_session_next(s, &token) == 1 && !m.invalid_calls,
          "target verification never runs past context limit");
    janas_llm_session_destroy(s);

    model_init(&m); s = janas_llm_session_create(&m, &o);
    CHECK(s && janas_llm_session_append(s, &start, 1) == 0, "remaining-one fixture created");
    if (!s) return;
    janas_llm_session_reply_remaining(s, 1);
    CHECK(janas_llm_session_next(s, &token) == 0 && m.costs.head_calls == 0 &&
          s->n_mdraft == 0 && s->ret == s->n,
          "remaining-one hint skips unused head/chain and overproduction");
    janas_llm_session_destroy(s);
}

int main(void)
{
    {
        struct janas_llm_model model;
        model_init(&model);
        struct janas_gen_options options = {.spec_mode = JANAS_SPEC_MTP,
            .spec_k = 4, .eos = -1, .draft_vocab_policy = 1};
        struct janas_llm_session *session = janas_llm_session_create(&model, &options);
        CHECK(session && session->o.spec_planner == 1,
              "adaptive legacy planner resolves cycle on create");
        if (session) {
            options.spec_planner = 2;
            CHECK(janas_llm_session_set_options(session, &options) == 0 &&
                  session->o.spec_planner == 2, "adaptive explicit fixed preserved");
            options.spec_planner = 0;
            CHECK(janas_llm_session_set_options(session, &options) == 0 &&
                  session->o.spec_planner == 1,
                  "adaptive legacy planner resolves cycle on update");
            janas_llm_session_destroy(session);
        }
    }
    for (int sampled = 0; sampled < 2; sampled++) {
        float temp = sampled ? 1.0f : 0.0f;
        paired_transcript(JANAS_SPEC_MTP, 0, 0, temp, 0, 0);
        paired_transcript(JANAS_SPEC_MTP, 1, 0, temp, 0, 0);
        paired_transcript(JANAS_SPEC_MTP, 2, 0, temp, 0, 1);
        paired_transcript(JANAS_SPEC_MTP, 2, 1, temp, 0, 0);
        paired_transcript(JANAS_SPEC_HYBRID, 2, 0, temp, 0, 0);
        paired_transcript(JANAS_SPEC_HYBRID, 1, 0, temp, 3, 1);
        unread_rewind(temp);
    }
    reseed_after_unread();
    abandoned_cycle_observation();
    late_hybrid_lookup();
    lookup_eos_context();
    printf("test_generate_cycle: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
