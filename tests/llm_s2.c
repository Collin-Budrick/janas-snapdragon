/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Real-model S2 branch/session correctness proof, NOT a performance trial.
 * One model is used sequentially for both complete token transcripts.
 * Test-only controller priors and private controller RNG force bounded probes.
 * Native MTP top-two candidates, target forwards, logits and sampler RNG are
 * never replaced, filtered or modified. A real cache use is required.
 *
 * Usage: llm_s2 model.jns mtp.jns tokens.int32 [cache=MiB] [steps=64] [offset=0]
 * Tokens must contain at least 40 native vocabulary IDs. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "janas/llm.h"
#include "llm/model.h"
#include "llm/s2_controller.h"

static struct janas_s2_decision proof_choose(struct janas_s2_controller *,
                                             const struct janas_s2_offer *);
#define janas_s2_choose proof_choose
#include "../src/llm/generate.c"
#undef janas_s2_choose

static unsigned current_case;
#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "real S2 proof failed: case=%u line=%d: %s\n", \
            current_case, __LINE__, #c); exit(1); \
} } while (0)

static uint64_t primed_offers, forced_native_probes;

static struct janas_s2_decision proof_choose(struct janas_s2_controller *p,
                                             const struct janas_s2_offer *o)
{
    if (!p || !o || o->branches < 2 || !o->primary_nodes ||
        o->primary_nodes > 16 || o->vocab >= JANAS_S2_VOCABS ||
        o->reuse >= JANAS_S2_REUSE || o->available_alt_nodes < 2 ||
        o->max_nodes < o->primary_nodes + 2 ||
        !(o->rank2_probability > 0 && o->rank2_probability <= 1))
        return janas_s2_choose(p, o);
    /* Deliberately synthetic scheduling priors, kept OUT of production and
       reported as instrumented. Actual native-head and target outcomes are
       still observed by the unchanged generator. These numbers prove no gain. */
    static struct janas_s2_controller seed, trial;
    janas_s2_init(&seed, p->rng);
    struct janas_s2_observation observation = {
        .cycle = o->cycle, .warm = o->warm, .vocab = o->vocab,
        .reuse = o->reuse, .width = o->primary_nodes + 1,
        .delivered = 1, .full_seconds = .04, .verify_seconds = .03,
        .expert_cpu_seconds = .02, .expert_groups = 1,
        .expert_assignments = o->primary_nodes + 1,
        .root_offered = 1, .rank2_probability = o->rank2_probability};
    for (unsigned i = 0; i < 3; i++) janas_s2_observe(&seed, &observation);
    primed_offers++;
    for (unsigned attempt = 0; attempt < 4096; attempt++) {
        trial = seed;
        trial.rng += attempt;
        struct janas_s2_decision decision = janas_s2_choose(&trial, o);
        if (decision.probe && decision.alt_nodes == 2) {
            *p = trial;
            forced_native_probes++;
            return decision;
        }
    }
    REQUIRE(0); /* A bounded failure is preferable to silently missing probes. */
    return janas_s2_choose(p, o);
}

#define MAX_TRANSCRIPT 512
struct result {
    int32_t token[MAX_TRANSCRIPT];
    unsigned n, live_invalidations, rejected_truncations;
    struct janas_gen_s2_metrics s2;
};

static void inactive(const struct janas_llm_session *s)
{
    REQUIRE(!s->s2_pending.tail_n && !s->s2_pending.cached &&
            !s->s2_current.tail_n && !s->s2_current.matched);
}
static unsigned live(const struct janas_llm_session *s)
{ return s->s2_pending.tail_n + s->s2_current.tail_n; }
static void cleared(struct janas_llm_session *s, struct result *r, unsigned before)
{
    if (s->o.s2_exact) { inactive(s); r->live_invalidations += before > 0; }
}
static void emit(struct janas_llm_session *s, struct result *r, unsigned n,
                   int bounded)
{
    if (bounded) janas_llm_session_reply_remaining(s, n);
    else janas_llm_session_reply_remaining(s, UINT32_MAX);
    for (unsigned i = 0; i < n; i++) {
        REQUIRE(r->n < MAX_TRANSCRIPT);
        REQUIRE(!janas_llm_session_next(s, &r->token[r->n]));
        r->n++;
        if (bounded) {
            REQUIRE(s->n - s->ret <= n - i - 1);
            REQUIRE(s->metrics.last.pass_width <= n - i + s->metrics.last.delivered - 1);
        }
    }
    if (bounded) REQUIRE(s->n == s->ret);
}

static struct result run(struct janas_llm_model *m, const int32_t *text,
                          unsigned prompt, unsigned steps, uint64_t seed_value,
                          int speculative, int32_t stop_id)
{
    struct result r = {0};
    struct janas_gen_options options = {
        .spec_mode = speculative ? JANAS_SPEC_MTP : JANAS_SPEC_OFF,
        .spec_k = 4, .spec_planner = 2, .eos = -1,
        .s2_exact = speculative, .s2_branches = 2, .s2_max_nodes = 8,
        .s2_min_gain_percent = 5, .seed = seed_value,
        .temperature = seed_value ? .8f : 0,
        .top_k = seed_value ? 40 : 0, .top_p = seed_value ? .95f : 0,
        .min_p = seed_value ? .01f : 0};
    struct janas_llm_session *s = janas_llm_session_create(m, &options);
    REQUIRE(s && !janas_llm_session_append(s, text, prompt));
    emit(s, &r, steps, 0);
    /* A new turn drops any buffered suffix and every derived branch. */
    unsigned before = live(s);
    REQUIRE(!janas_llm_session_append(s, text + prompt, 8));
    cleared(s, &r, before);
    emit(s, &r, 12, 0);
    /* A genuinely impossible recurrent rewind must refuse without consuming
       a token. Truncating to ret reconciles only unreturned speculative work. */
    uint32_t old_ret = s->ret, old_n = s->n;
    unsigned char pending[sizeof(s->s2_pending)];
    memcpy(pending, &s->s2_pending, sizeof(pending));
    REQUIRE(janas_llm_session_truncate(s, s->n_kv > 2 ? s->n_kv - 2 : 0) == -1);
    REQUIRE(s->ret == old_ret && s->n == old_n &&
            !memcmp(pending, &s->s2_pending, sizeof(pending)));
    r.rejected_truncations++;
    before = live(s);
    REQUIRE(!janas_llm_session_truncate(s, s->ret));
    if (old_n > old_ret) cleared(s, &r, before); /* ret==n is a valid no-op */
    emit(s, &r, 3, 1);
    /* Snapshots retain canonical state, never derived alternate branches. */
    before = live(s);
    struct janas_llm_saved *saved = janas_llm_session_save(s);
    REQUIRE(saved); cleared(s, &r, before);
    emit(s, &r, 4, 1);
    before = live(s);
    REQUIRE(!janas_llm_session_restore(s, saved)); cleared(s, &r, before);
    /* Session snapshots intentionally do not snapshot the sampler RNG. Reset
       it identically in both transcripts before testing restored execution. */
    REQUIRE(!janas_llm_session_set_options(s, &options));
    emit(s, &r, 4, 1);
    janas_llm_saved_free(saved);
    /* Reset starts a clean real prefix; its first output has no unread target
       rows, yet S2 can already hold an alternate future native-MTP tail. */
    before = live(s); janas_llm_session_reset(s); cleared(s, &r, before);
    REQUIRE(!janas_llm_session_set_options(s, &options));
    REQUIRE(!janas_llm_session_append(s, text, prompt));
    emit(s, &r, 1, 0); REQUIRE(s->n == s->ret);
    before = live(s);
    REQUIRE(!janas_llm_session_slide(s, 4, 4)); cleared(s, &r, before);
    emit(s, &r, 8, 1);
    before = live(s); janas_llm_session_reset(s); cleared(s, &r, before);
    REQUIRE(!janas_llm_session_set_options(s, &options));
    REQUIRE(!janas_llm_session_append(s, text, prompt));
    emit(s, &r, 1, 0); before = live(s);
    REQUIRE(!janas_llm_session_append(s, text + prompt, 8)); cleared(s, &r, before);
    emit(s, &r, 8, 1);
    /* Stop is an actual token from the plain transcript at this same prefix.
       It stays a full-vocabulary target sample and never becomes a draft root. */
    before = live(s); janas_llm_session_reset(s); cleared(s, &r, before);
    options.eos = stop_id;
    REQUIRE(!janas_llm_session_set_options(s, &options));
    REQUIRE(!janas_llm_session_append(s, text, prompt));
    unsigned stop_start = r.n;
    emit(s, &r, 1, 0);
    REQUIRE(r.token[stop_start] == stop_id && s->n == s->ret);
    if (speculative) inactive(s);
    /* Reply clipping is independent of stop handling, including depth zero. */
    janas_llm_session_reset(s); options.eos = -1;
    REQUIRE(!janas_llm_session_set_options(s, &options));
    REQUIRE(!janas_llm_session_append(s, text, prompt));
    emit(s, &r, 2, 1);
    janas_llm_session_s2_metrics(s, &r.s2);
    janas_llm_session_destroy(s);
    return r;
}

static int32_t first_plain(struct janas_llm_model *m, const int32_t *text,
                             unsigned prompt, uint64_t seed_value)
{
    struct janas_gen_options o = {.spec_mode = JANAS_SPEC_OFF, .spec_k = 4,
        .eos = -1, .seed = seed_value, .temperature = seed_value ? .8f : 0,
        .top_k = seed_value ? 40 : 0, .top_p = seed_value ? .95f : 0,
        .min_p = seed_value ? .01f : 0};
    struct janas_llm_session *s = janas_llm_session_create(m, &o);
    int32_t token;
    REQUIRE(s && !janas_llm_session_append(s, text, prompt));
    REQUIRE(!janas_llm_session_next(s, &token));
    janas_llm_session_destroy(s); return token;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: llm_s2 model.jns mtp.jns tokens.int32 [cache=MiB] [steps=64] [offset=0]\n");
        return 2;
    }
    uint64_t cache_mib = 32768; unsigned steps = 64, offset = 0;
    for (int i = 4; i < argc; i++) {
        if (!strncmp(argv[i], "cache=", 6)) cache_mib = strtoull(argv[i] + 6, NULL, 10);
        else if (!strncmp(argv[i], "steps=", 6)) steps = (unsigned)strtoul(argv[i] + 6, NULL, 10);
        else if (!strncmp(argv[i], "offset=", 7)) offset = (unsigned)strtoul(argv[i] + 7, NULL, 10);
        else return 2;
    }
    REQUIRE(cache_mib <= (UINT64_MAX >> 20) && steps >= 16 && steps <= 192);
    int32_t text[1024]; FILE *f = fopen(argv[3], "rb"); REQUIRE(f);
    unsigned nt = (unsigned)fread(text, sizeof(*text), 1024, f); fclose(f);
    REQUIRE(nt >= 40 && offset <= nt - 40);
    if (offset) { nt -= offset; memmove(text, text + offset, nt * sizeof(*text)); }
    unsigned prompt = nt >= 72 ? 64 : nt - 8;
    int cpus[] = {0, 1, 2, 3, 4, 5, 6, 7};
    struct janas_llm_options o = {.cache_bytes = cache_mib << 20, .n_ctx = 512,
        .compute_cpus = cpus, .n_compute = 8, .n_io = 4, .mtp_path = argv[2],
        .expert_bits = 6, .min_quality = JANAS_LLM_QUALITY_HIGH, .no_npu = 1};
    char err[512];
    struct janas_llm_model *m = janas_llm_model_load(argv[1], &o, err, sizeof(err));
    if (!m) { fprintf(stderr, "%s\n", err); return 1; }
    REQUIRE(janas_llm_mtp_branch_capable(m) && janas_llm_model_recurrent(m));
    uint64_t cached = 0, accepted = 0, branches = 0, invalidations = 0;
    for (current_case = 0; current_case < 2; current_case++) {
        uint64_t seed_value = current_case ? UINT64_C(314159) : 0;
        int32_t stop = first_plain(m, text, prompt, seed_value);
        struct result plain = run(m, text, prompt, steps, seed_value, 0, stop);
        struct result exact = run(m, text, prompt, steps, seed_value, 1, stop);
        REQUIRE(plain.n == exact.n && !memcmp(plain.token, exact.token,
                                             plain.n * sizeof(*plain.token)));
        REQUIRE(exact.live_invalidations >= 2 && exact.s2.alternate_builds > 0);
        cached += exact.s2.cached_uses; accepted += exact.s2.cached_accepted;
        branches += exact.s2.alternate_builds; invalidations += exact.live_invalidations;
        printf("{\"case\":%u,\"seed\":%" PRIu64 ",\"passed\":true,"
               "\"instrumented_controller\":true,\"performance_trial\":false,"
               "\"cached_uses\":%" PRIu64 ",\"cached_accepted\":%" PRIu64 ","
               "\"rank2_matches\":%" PRIu64 ",\"branches\":%" PRIu64 ","
               "\"live_invalidations\":%u,\"transcript\":[",
               current_case, seed_value, exact.s2.cached_uses, exact.s2.cached_accepted,
               exact.s2.rank2_matches, exact.s2.alternate_builds, exact.live_invalidations);
        for (unsigned i = 0; i < exact.n; i++) printf("%s%d", i ? "," : "", exact.token[i]);
        puts("]}"); fflush(stdout);
    }
    REQUIRE(cached > 0);
    printf("{\"passed\":true,\"single_model_instance\":true,"
           "\"instrumented_controller\":true,\"performance_trial\":false,"
           "\"primed_offers\":%" PRIu64 ",\"native_probes\":%" PRIu64 ","
           "\"branches\":%" PRIu64 ",\"cached_uses\":%" PRIu64 ","
           "\"cached_accepted\":%" PRIu64 ",\"live_invalidations\":%" PRIu64 "}\n",
           primed_offers, forced_native_probes, branches, cached, accepted, invalidations);
    janas_llm_model_free(m); return 0;
}
