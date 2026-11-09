/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * generate.c - token generation with optional speculative decoding (see
 * generate.h).
 */
#include "generate.h"
#include "history_index.h"
#include "spec_controller.h"
#include "hybrid_controller.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SPEC_K JANAS_GEN_MAX_K
/* passes without drafting before spec_auto retries: doubles after every
   retry that still does not pay, back to the minimum when one does */
#define AUTO_RETRY_MIN 32
#define AUTO_RETRY_MAX 128
/*
 * Turning the drafts off is a heavy thing to do on a light measurement, and
 * the measurement is not even of like with like: the cost of a plain pass is
 * learned only on the passes that make no draft, which are the first of a
 * reply, where the context is shortest and the cache warmest, while the cost
 * of a speculative pass is learned all the way through. Read as it was, that
 * comparison leans towards switching off. So: the drafts must lose by more
 * than a margin, they must lose three judgements running, and the plain cost
 * they are judged against must have been measured in the last few dozen
 * passes - without that, there is nothing honest to compare them with and
 * they stay on. Measured on 21 Sep 2026, before this: one unlucky judgement
 * put the drafts off for 512 passes, longer than the reply, and 1301 tokens
 * came out at 19 token/s where they would have come out at 26.
 */
#define AUTO_MARGIN 0.95 /* the drafts must lose by more than this */
#define AUTO_STRIKES 3   /* judgements running before they go off */
#define AUTO_PROBE                                                             \
    64                 /* passes between two plain ones, to keep the           \
                          baseline a measurement of the work at hand */
#define AUTO_FRESH 128 /* passes: how old the plain cost may be */
#define AUTO_WARM 16   /* speculative passes to learn before judging */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * Lookup draft: the longest suffix of the history (3, 2, then 1 token) that
 * occurred before, most recent occurrence first; the tokens that followed it
 * are the draft. Returns the draft length (0 when nothing matches).
 */
static int lookup_draft(const int32_t *hist, uint32_t n, int k, int32_t *draft)
{
    for (uint32_t len = 3; len >= 1; len--) {
        if (n <= len)
            continue;
        const int32_t *suffix = hist + n - len;
        for (uint32_t start = n - len; start-- > 0;) {
            if (memcmp(hist + start, suffix, len * sizeof(int32_t)) != 0)
                continue;
            uint32_t from = start + len;
            int got = 0;
            while (got < k && from + (uint32_t)got < n) {
                draft[got] = hist[from + (uint32_t)got];
                got++;
            }
            if (got > 0)
                return got;
        }
    }
    return 0;
}

/*
 * Planner of MTP drafts (spec_auto): how many of the available drafts to
 * verify, from costs and odds measured while generating. t[n] is the time
 * of a pass over n tokens (moving average; missing sizes interpolated
 * linearly); the chance that a draft survives is its product of MTP
 * probabilities, calibrated on the outcomes of earlier verifications in
 * bins. A chained draft also costs one step of the MTP block (chain,
 * moving average). The plan maximizes expected tokens per second,
 * (1 + sum of survival chances) / (t(1 + drafts) + chain steps).
 */
#define CAL_BINS 10

struct planner {
    double t[MAX_SPEC_K + 2], chain;
    double tried[CAL_BINS], won[CAL_BINS];
    /* lookup drafts have no probability of their own to calibrate: what is
       measured instead is how often the draft in each position survived,
       which is the same thing the planner needs */
    double lk_tried[MAX_SPEC_K], lk_won[MAX_SPEC_K];
    double dcost; /* what one pass of the drafting model costs */
};

/* Halved when the first position has seen this many drafts, so that the
   estimate follows the text instead of averaging a whole conversation. */
#define LK_WINDOW 192

static int cal_bin(float surv)
{
    int b = (int)(surv * CAL_BINS);
    return b < 0 ? 0 : (b >= CAL_BINS ? CAL_BINS - 1 : b);
}

/*
 * Calibrated survival chance: the bin's record, the raw value as a prior
 * worth two trials. The MTP block's own probabilities are pessimistic -
 * measured on Italian prose with the drafts forced, a guess of 0.2-0.3
 * survived 35% of the time and one of 0.3-0.4, 44%, against a break-even of
 * 29% - but adding an optimism that fades as a bin fills (0.3 / sqrt(n),
 * tried on 20 Sep 2026) made the planner slower, not faster: it spends the
 * drafts it gains on the bins below 0.2, which really are that bad.
 */
static double cal_chance(const struct planner *p, float surv)
{
    int b = cal_bin(surv);
    return (p->won[b] + 2.0 * surv) / (p->tried[b] + 2.0);
}

static double pass_cost(const struct planner *p, int n)
{
    if (p->t[n] > 0)
        return p->t[n];
    int lo = 0, hi = 0; /* nearest measured sizes around n */
    for (int i = n - 1; i >= 1 && !lo; i--)
        if (p->t[i] > 0)
            lo = i;
    for (int i = n + 1; i <= MAX_SPEC_K + 1 && !hi; i++)
        if (p->t[i] > 0)
            hi = i;
    if (lo && hi)
        return p->t[lo] + (p->t[hi] - p->t[lo]) * (n - lo) / (hi - lo);
    int a = lo ? lo : hi;
    if (!a)
        return n; /* nothing measured yet: proportional */
    /* one side only: assume each token adds 30% of a plain pass */
    double one = p->t[1] > 0 ? p->t[1] : p->t[a] / (1.0 + 0.3 * (a - 1));
    return p->t[a] + 0.3 * one * (n - a);
}

/*
 * How often the draft in position i survived, with a prior of one half worth
 * two trials so that a session starts by trying. The value is the chance of
 * reaching that position and being right, not a conditional one: that is
 * what the planner adds up.
 */
static double lk_chance(const struct planner *p, int i)
{
    return (p->lk_won[i] + 1.0) / (p->lk_tried[i] + 2.0);
}

/*
 * How many of the drafts the context offered are worth verifying. Same
 * economics as the planner below, and the same reason to exist: a fixed
 * number can only be right for one kind of text on one kind of model. On a
 * dense model a second token in the pass costs 6% and on a mixture 28% -
 * two tokens route to different experts - while the drafts copied from the
 * context survive 11% of the time on free prose and 65% on a table. Four,
 * which is what this engine asked for until 22 Sep 2026, was slower than no
 * drafts at all on prose: 0.89x on Qwen3-4B, 0.82x on Qwen3-30B.
 */
static int plan_lookup(const struct planner *p, int avail, double per_draft)
{
    double expect = 1.0, best_rate = 1.0 / pass_cost(p, 1);
    int best = 0;
    for (int i = 0; i < avail; i++) {
        expect += lk_chance(p, i);
        double rate = expect / (pass_cost(p, i + 2) + per_draft * (i + 1));
        if (rate > best_rate) {
            best_rate = rate;
            best = i + 1;
        }
    }
    return best;
}

/* At most k drafts, and no more than the pass the model holds worth
   asking for takes (janas_llm_model_pass_cap: the GPU's shaders of a few
   vectors): the planner's line through the costs it has measured cannot
   see the step past them, and paid for one pass to learn it. */
static int drafts_cap(const struct janas_llm_model *m, int k)
{
    uint32_t cap = janas_llm_model_pass_cap(m);
    return cap > 1 && k > (int)cap - 1 ? (int)cap - 1 : k;
}

static int plan_drafts(const struct planner *p, const float *survs, int avail)
{
    double expect = 1.0, best_rate = 1.0 / pass_cost(p, 1);
    int best = 0;
    for (int i = 0; i < avail; i++) {
        expect += cal_chance(p, survs[i]);
        double rate = expect / (pass_cost(p, i + 2) + p->chain * i);
        if (rate > best_rate) {
            best_rate = rate;
            best = i + 1;
        }
    }
    return best;
}

/*
 * After a pass at position pos that accepted acc drafts: the MTP rows of the
 * new tokens (token t_{i+1} with the main model's h_i, at position i, for
 * i = pos .. pos + acc), which keep the block's KV cache complete; then, if
 * drafting, its guess after the last one and k - 1 more, each chained on the
 * block's own hidden state. MTP positions past the accepted ones are
 * overwritten by the next call, as in the main KV cache.
 */
struct mtp_times { double maintenance, head, chain; };
static int mtp_step(struct janas_llm_model *m, const int32_t *hist,
                    uint32_t pos, uint32_t acc, int k, int drafting,
                    float min_conf, struct planner *pl, float *hrow,
                    int32_t *draft, float *survs, int *n_draft,
                    struct mtp_times *times)
{
    struct janas_mtp_costs before, first, after;
    janas_llm_mtp_costs(m, &before);
    double first_start = now();
    if (times) memset(times, 0, sizeof(*times));
    /* a row of hidden state: d_model, or qwen4exp's streams */
    uint32_t dm = janas_llm_model_hidden_width(m);
    *n_draft = 0;
    janas_llm_mtp_note_tokens(m, hist + pos + 1, acc + 1);
    if (!drafting) {
        int rc = janas_llm_mtp_forward(m, hist + pos + 1,
                                     janas_llm_model_hidden(m), acc + 1, pos,
                                     NULL, 0);
        if (times) times->maintenance = now() - first_start;
        return rc;
    }
    float conf;
    if (janas_llm_mtp_draft(m, hist + pos + 1, janas_llm_model_hidden(m),
                            acc + 1, pos, &draft[0], &conf) != 0)
        return -1;
    janas_llm_mtp_costs(m, &first);
    if (times) {
        times->head = first.head_seconds - before.head_seconds;
        times->maintenance = fmax(0, now() - first_start - times->head);
    }
    /* a chained draft counts only if every one before it is accepted too:
       the chain goes on while the product of the probabilities, the
       chance that the draft survives, stays above min_conf */
    float surv = conf;
    if (surv < min_conf)
        return 0; /* not sure enough: a plain pass is cheaper */
    survs[0] = surv;
    memcpy(hrow, janas_llm_mtp_hidden(m) + (size_t)acc * dm,
           dm * sizeof(float));
    int got = 1;
    double chain_start = now();
    for (; got < k; got++) {
        /* planning: chain on only while the last draft is worth verifying
           (the chances only fall along the chain) */
        if (pl && plan_drafts(pl, survs, got) < got)
            break;
        double tc = now();
        uint32_t at = janas_llm_mtp_chain_fixed(m) ? pos + acc
                                                   : pos + acc + (uint32_t)got;
        if (janas_llm_mtp_draft(m, &draft[got - 1], hrow, 1, at, &draft[got],
                                &conf) != 0 ||
            (surv *= conf) < min_conf)
            break; /* past the context, or not sure enough */
        survs[got] = surv;
        memcpy(hrow, janas_llm_mtp_hidden(m), dm * sizeof(float));
        if (pl) {
            double d = now() - tc;
            pl->chain = pl->chain > 0 ? 0.8 * pl->chain + 0.2 * d : d;
        }
    }
    janas_llm_mtp_costs(m, &after);
    if (times) {
        double heads = after.head_seconds - first.head_seconds;
        times->head += heads;
        times->chain = fmax(0, now() - chain_start - heads);
    }
    *n_draft = got;
    return 0;
}

/* What was learned of the tokens generated lately, by position (a ring:
   a pass generates at most MAX_SPEC_K + 1, and they are read soon after) */
#define LP_RING 64

struct janas_llm_session {
    struct janas_llm_model *m;
    struct janas_gen_options o;
    uint32_t nv, cap; /* vocabulary, context */
    /* hist[0 .. n): the sequence; hist[0 .. n_kv) are in the model's state
       (the pending last token is not, until a pass takes it); hist[0 ..
       ret) were appended or returned; a pass rewinds no further than
       call_start, the first position of the model's last forward call */
    int32_t *hist;
    uint32_t n, n_kv, ret, call_start;
    float *logits, *hrow;
    struct janas_sampler *smp;
    struct janas_token_lp *lp; /* LP_RING of them, when logprobs are on */
    uint32_t lp_pos[LP_RING];  /* the position each one belongs to, + 1 */
    int mtp;
    /* speculation */
    double single_cost, spec_cost, gain;
    uint32_t single_at; /* pass at which single_cost was last measured */
    uint32_t spec_seen; /* speculative passes since drafting resumed */
    int could_draft;    /* whether the pass before this one could draft */
    int strikes;        /* judgements running against the drafts */
    uint32_t off_left, retry;
    int32_t mdraft[MAX_SPEC_K];
    float msurv[MAX_SPEC_K];
    int n_mdraft; /* MTP drafts ready for the next pass */
    /* the drafting model: the last position whose KV it holds for certain,
       and room for its logits */
    uint32_t dpos;
    int dpos_set;
    float *dlogits;
    struct planner pl;
    struct janas_gen_stats st;
    struct janas_history_index *index;
    struct janas_cycle_controller cycle;
    struct janas_hybrid_controller hybrid;
    struct planner hybrid_pl[3]; /* adaptive heads do not train legacy bins */
    struct janas_gen_hybrid_metrics hybrid_metrics;
    struct janas_gen_hybrid_decision hybrid_pending, hybrid_current, hybrid_wait_decision;
    struct janas_hybrid_observation hybrid_wait;
    int hybrid_waiting;
    double hybrid_incoming_recovery;
    struct janas_gen_metrics metrics;
    struct janas_gen_cycle pending;
    uint64_t next_proposal_id;
    uint64_t pending_vocab_epoch;
    uint32_t reply_remaining;
    int observation_pending, has_filter;
    unsigned last_reuse, pending_reuse, observation_reuse;
};

static int hybrid_cost(const struct janas_llm_session *s)
{
    return s->o.spec_mode == JANAS_SPEC_HYBRID && s->o.hybrid_policy == 1;
}

static struct planner *mtp_planner(struct janas_llm_session *s)
{
    unsigned key = janas_cycle_vocab_key((unsigned)janas_llm_mtp_vocab_policy(s->m),
                                         janas_llm_mtp_vocab_rows(s->m));
    return key ? &s->hybrid_pl[key - 1] : &s->pl;
}

/* Recovery is fused with the next GDN verification. Its measured pool-time
   row-share estimate is transferred once, without adding elapsed time. */
static void hybrid_settle(struct janas_llm_session *s, double recovery)
{
    if (!s->hybrid_waiting) return;
    if (!isfinite(recovery) || recovery < 0) recovery = 0;
    s->hybrid_wait.seconds += recovery;
    s->hybrid_wait_decision.causal_seconds = s->hybrid_wait.seconds;
    s->hybrid_wait_decision.recovery_seconds += recovery;
    janas_hybrid_observe(&s->hybrid, &s->hybrid_wait);
    if (s->hybrid_wait.source == JANAS_GEN_SOURCE_LOOKUP) {
        s->hybrid_metrics.lookup_seconds += s->hybrid_wait.seconds;
        s->hybrid_metrics.lookup_delivered += s->hybrid_wait.delivered;
    } else {
        s->hybrid_metrics.mtp_seconds += s->hybrid_wait.seconds;
        s->hybrid_metrics.mtp_delivered += s->hybrid_wait.delivered;
    }
    s->hybrid_metrics.recovery_reassigned_seconds += recovery;
    s->hybrid_metrics.last = s->hybrid_wait_decision;
    s->hybrid_waiting = 0;
}

static int cycle_mode(const struct janas_llm_session *s)
{
    return s->o.spec_planner != 0 || s->o.spec_mode == JANAS_SPEC_HYBRID ||
           s->o.draft_vocab_policy != 0;
}

static void finish_cycle(struct janas_llm_session *s)
{
    if (!s->observation_pending) return;
    const struct janas_gen_cycle *c = &s->metrics.last;
    if (hybrid_cost(s)) {
        hybrid_settle(s, 0);
        unsigned source = c->planner_source == JANAS_GEN_SOURCE_LOOKUP
            ? JANAS_GEN_SOURCE_LOOKUP : JANAS_GEN_SOURCE_MTP;
        s->hybrid_wait = (struct janas_hybrid_observation){
            .source = source, .depth = c->pass_width - 1,
            .accepted_prefix = c->accepted_prefix, .delivered = c->delivered,
            .pass_width = c->pass_width, .lookup_suffix = c->lookup_suffix,
            .warm = c->warm,
            .vocab = janas_cycle_vocab_key(c->draft_vocab_policy, c->draft_vocab_rows),
            .reuse = s->observation_reuse,
            .seconds = fmax(0, c->total_seconds - s->hybrid_incoming_recovery),
            .verification_seconds = fmax(0, c->verification_seconds - s->hybrid_incoming_recovery)
        };
        s->hybrid_wait_decision = s->hybrid_current;
        s->hybrid_wait_decision.proposal_id = c->proposal_id;
        s->hybrid_wait_decision.accepted_prefix = c->accepted_prefix;
        s->hybrid_wait_decision.delivered = c->delivered;
        s->hybrid_wait_decision.pass_width = c->pass_width;
        s->hybrid_wait_decision.causal_seconds = s->hybrid_wait.seconds;
        s->hybrid_waiting = 1;
        if (!c->rejected_rows) hybrid_settle(s, 0);
    } else if (s->o.spec_planner == 1) {
        janas_cycle_observe_reuse(&s->cycle, c->planner_source, c->warm,
            janas_cycle_vocab_key(c->draft_vocab_policy, c->draft_vocab_rows),
            c->requested_depth, c->total_seconds, c->delivered, s->observation_reuse);
        /* Lookup depth zero means use the MTP fallback. Its real reward
           and full cost must be observed even when MTP chose a nonzero
           depth, including when the first lookup hit occurs late. */
        if (s->o.spec_mode == JANAS_SPEC_HYBRID && c->lookup_suffix &&
            c->planner_source == JANAS_GEN_SOURCE_MTP)
            janas_cycle_observe_reuse(&s->cycle, JANAS_GEN_SOURCE_LOOKUP, c->warm, 0,
                                 0, c->total_seconds, c->delivered, s->observation_reuse);
    }
    s->observation_pending = 0;
}

static void abandon_proposal(struct janas_llm_session *s)
{
    const struct janas_gen_cycle *p = &s->pending;
    if (p->proposal_id && p->preparation_seconds > 0 && hybrid_cost(s)) {
        struct janas_hybrid_observation o = {
            .source = p->planner_source, .depth = p->requested_depth,
            .lookup_suffix = p->lookup_suffix, .warm = p->warm,
            .vocab = janas_cycle_vocab_key(p->draft_vocab_policy, p->draft_vocab_rows),
            .reuse = s->pending_reuse, .seconds = p->preparation_seconds
        };
        janas_hybrid_observe(&s->hybrid, &o);
        if (o.source == JANAS_GEN_SOURCE_LOOKUP)
            s->hybrid_metrics.lookup_seconds += o.seconds;
        else s->hybrid_metrics.mtp_seconds += o.seconds;
    } else if (p->proposal_id && p->preparation_seconds > 0 && s->o.spec_planner == 1) {
        /* This action already spent its preparation but never reached a
           target pass. Price it once as zero useful output, independently
           of the previous cycle's delivered-token observation. */
        janas_cycle_observe_reuse(&s->cycle, p->planner_source, p->warm,
            janas_cycle_vocab_key(p->draft_vocab_policy, p->draft_vocab_rows),
            p->requested_depth, p->preparation_seconds, 0, s->pending_reuse);
        if (s->o.spec_mode == JANAS_SPEC_HYBRID && p->lookup_suffix &&
            p->planner_source == JANAS_GEN_SOURCE_MTP)
            janas_cycle_observe_reuse(&s->cycle, JANAS_GEN_SOURCE_LOOKUP, p->warm, 0,
                                      0, p->preparation_seconds, 0, s->pending_reuse);
    }
    s->metrics.abandoned_preparation_seconds += s->pending.preparation_seconds;
    memset(&s->pending, 0, sizeof(s->pending));
    memset(&s->hybrid_pending, 0, sizeof(s->hybrid_pending));
    s->n_mdraft = 0;
}

static void delivered(struct janas_llm_session *s)
{
    s->metrics.delivered_tokens++;
    s->metrics.source_delivered[s->metrics.last.source]++;
    s->metrics.last.delivered++;
    s->metrics.last.buffered = s->n - s->ret;
    if (s->reply_remaining != UINT32_MAX && s->reply_remaining)
        s->reply_remaining--;
    if (s->ret == s->n) finish_cycle(s);
}

static void discard_buffered(struct janas_llm_session *s)
{
    uint32_t n = s->n - s->ret;
    s->metrics.discarded_tokens += n;
    s->metrics.source_discarded[s->metrics.last.source] += n;
    s->metrics.last.discarded += n;
    s->metrics.last.buffered = 0;
    finish_cycle(s);
    abandon_proposal(s);
}

static int valid_options(const struct janas_llm_model *m,
                         const struct janas_gen_options *o)
{
    return o && o->spec_mode >= JANAS_SPEC_OFF && o->spec_mode <= JANAS_SPEC_HYBRID &&
           o->spec_planner >= 0 && o->spec_planner <= 2 &&
           o->spec_explore_percent >= 0 && o->spec_explore_percent <= 100 &&
           o->draft_vocab_policy >= 0 && o->draft_vocab_policy <= 1 &&
           (!o->draft_vocab_rows || o->draft_vocab_rows == 8192 ||
            o->draft_vocab_rows == 16384 || o->draft_vocab_rows == 32768) &&
            o->hybrid_policy >= 0 && o->hybrid_policy <= 1 &&
            o->hybrid_lookup_max_drafts >= 0 && o->hybrid_lookup_max_drafts <= MAX_SPEC_K &&
            o->hybrid_mtp_max_drafts >= 0 && o->hybrid_mtp_max_drafts <= MAX_SPEC_K &&
            o->hybrid_probe_percent >= 0 && o->hybrid_probe_percent <= 100 &&
           o->spec_k >= 1 && o->spec_k <= MAX_SPEC_K && o->n_stop >= 0 &&
           o->n_stop <= JANAS_GEN_MAX_STOP && o->temperature >= 0 &&
           o->top_k >= 0 && o->top_p >= 0 && o->min_p >= 0 &&
           ((o->spec_mode != JANAS_SPEC_MTP && o->spec_mode != JANAS_SPEC_HYBRID) ||
            janas_llm_model_has_mtp(m)) &&
           (o->spec_mode != JANAS_SPEC_DRAFT ||
            (o->draft &&
             janas_llm_model_n_vocab(o->draft) == janas_llm_model_n_vocab(m)));
}

/*
 * What the planner has learned about the drafts of this model on this
 * machine, kept between sessions: ten pairs of counts, a few hundred bytes.
 * Without it every chat starts by finding out again which guesses are worth
 * verifying, and pays for the finding out.
 */
static void planner_load(struct janas_llm_session *s)
{
    char path[1024];
    if (janas_llm_model_cache_file(s->m, "drafts", path, sizeof(path), 0) != 0)
        return;
    FILE *f = fopen(path, "rb");
    if (!f)
        return;
    double v[2 * CAL_BINS];
    if (fread(v, sizeof(v), 1, f) == 1)
        for (int b = 0; b < CAL_BINS; b++) {
            s->pl.tried[b] = v[b] >= 0 ? v[b] : 0;
            s->pl.won[b] = v[CAL_BINS + b] >= 0 ? v[CAL_BINS + b] : 0;
            if (s->pl.won[b] > s->pl.tried[b])
                s->pl.won[b] = s->pl.tried[b];
        }
    fclose(f);
}

static void planner_save(const struct janas_llm_session *s)
{
    char path[1024];
    if (janas_llm_model_cache_file(s->m, "drafts", path, sizeof(path), 1) != 0)
        return;
    double v[2 * CAL_BINS];
    double most = 0;
    for (int b = 0; b < CAL_BINS; b++)
        if (s->pl.tried[b] > most)
            most = s->pl.tried[b];
    /* old sessions must not outweigh what the machine does now */
    double shrink = most > 4096 ? 4096 / most : 1.0;
    for (int b = 0; b < CAL_BINS; b++) {
        v[b] = s->pl.tried[b] * shrink;
        v[CAL_BINS + b] = s->pl.won[b] * shrink;
    }
    FILE *f = fopen(path, "wb");
    if (!f)
        return;
    if (fwrite(v, sizeof(v), 1, f) != 1)
        remove(path);
    fclose(f);
}

int janas_llm_session_slide(struct janas_llm_session *s, uint32_t keep,
                            uint32_t drop)
{
    if (!drop || keep + drop > s->n_kv || s->n_kv > s->n)
        return -1;
    if (janas_llm_model_shift(s->m, keep, drop, s->n_kv) != 0)
        return -1;
    memmove(s->hist + keep, s->hist + keep + drop,
            (s->n - keep - drop) * sizeof(*s->hist));
    s->n -= drop;
    s->n_kv -= drop;
    s->ret = s->ret > keep + drop ? s->ret - drop : keep;
    s->call_start = s->n; /* positions moved: no rolling back across this */
    abandon_proposal(s); /* the drafts were for the old positions */
    return 0;
}

/* The sampler's settings from the options; the random sequence restarts. */
static int sampler_setup(struct janas_llm_session *s)
{
    const struct janas_gen_options *o = &s->o;
    struct janas_sample_params sp = {
        .temperature = o->temperature,
        .top_k = o->top_k,
        .top_p = o->top_p,
        .min_p = o->min_p,
        .presence = o->presence_penalty,
        .frequency = o->frequency_penalty,
        .logprobs = o->logprobs ? o->top_logprobs : -1,
    };
    if (o->logprobs && !s->lp) {
        s->lp = malloc(LP_RING * sizeof(*s->lp));
        if (!s->lp)
            return -1;
    }
    janas_sampler_set(s->smp, &sp, o->seed);
    return 0;
}

int janas_llm_session_set_options(struct janas_llm_session *s,
                                  const struct janas_gen_options *o)
{
    if (!valid_options(s->m, o))
        return -1;
    /* Unread tokens were drawn by the old sampler. Reconcile that suffix
       before resetting the random sequence to the new seed. */
    if (s->ret < s->n && janas_llm_session_append(s, NULL, 0) != 0)
        return -1;
    int was_mtp = s->mtp;
    abandon_proposal(s);
    finish_cycle(s);
    if (s->o.spec_mode != o->spec_mode ||
        s->o.hybrid_policy != o->hybrid_policy || s->o.spec_k != o->spec_k ||
        s->o.hybrid_lookup_max_drafts != o->hybrid_lookup_max_drafts ||
        s->o.hybrid_mtp_max_drafts != o->hybrid_mtp_max_drafts ||
        s->o.draft_vocab_policy != o->draft_vocab_policy ||
        s->o.draft_vocab_rows != o->draft_vocab_rows) {
        hybrid_settle(s, 0);
        janas_hybrid_init(&s->hybrid, o->seed);
    }
    s->o = *o;
    if (hybrid_cost(s)) s->o.spec_planner = 0;
    else if ((s->o.spec_mode == JANAS_SPEC_HYBRID || s->o.draft_vocab_policy) &&
        !s->o.spec_planner)
        s->o.spec_planner = 1;
    if (!o->draft_vocab_policy) janas_llm_mtp_vocab_legacy(s->m);
    if (sampler_setup(s) != 0)
        return -1;
    /*
     * The MTP block's KV cache has no rows for what ran without it, so a
     * block turned on mid-sequence waits for a reset. Turning the drafts
     * off is not turning the block off: it goes on noting the tokens, which
     * is one pass of one layer and no vocabulary head, and the drafts can
     * come back whenever they are asked for. Reading the two as one thing
     * cost the drafts for good: /spec off zeroed this, /spec on found
     * was_mtp false and n_kv past zero, and nothing drafted again until the
     * conversation was reset (seen on 21 Sep 2026: 25.3 token/s before the
     * switch, 20.6 after, with no draft in between).
     */
    s->mtp = janas_llm_model_has_mtp(s->m) && (was_mtp || s->n_kv == 0);
    if (!s->mtp || (o->spec_mode != JANAS_SPEC_MTP && o->spec_mode != JANAS_SPEC_HYBRID))
        s->n_mdraft = 0;
    else if (s->n_mdraft > o->spec_k)
        s->n_mdraft = o->spec_k;
    if (o->draft && !s->dlogits) {
        s->dlogits = malloc((size_t)s->nv * sizeof(float));
        if (!s->dlogits)
            return -1;
    }
    return 0;
}

struct janas_llm_session *
janas_llm_session_create(struct janas_llm_model *m,
                         const struct janas_gen_options *o)
{
    if (!valid_options(m, o))
        return NULL;
    struct janas_llm_session *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->m = m;
    s->nv = janas_llm_model_n_vocab(m);
    s->cap = janas_llm_model_n_ctx(m);
    s->hist = malloc((s->cap + MAX_SPEC_K + 2) * sizeof(int32_t));
    s->logits = malloc((size_t)(MAX_SPEC_K + 1) * s->nv * sizeof(float));
    s->hrow = malloc(janas_llm_model_hidden_width(m) * sizeof(float));
    s->smp = janas_sampler_create(s->nv);
    if (o->draft)
        s->dlogits = malloc((size_t)s->nv * sizeof(float));
    if (!s->hist || !s->logits || !s->hrow || !s->smp ||
        (o->draft && !s->dlogits)) {
        janas_llm_session_destroy(s);
        return NULL;
    }
    s->mtp = janas_llm_model_has_mtp(m);
    s->o = *o;
    if (hybrid_cost(s)) s->o.spec_planner = 0;
    else if ((s->o.spec_mode == JANAS_SPEC_HYBRID || s->o.draft_vocab_policy) &&
        !s->o.spec_planner)
        s->o.spec_planner = 1;
    if (sampler_setup(s) != 0) {
        janas_llm_session_destroy(s);
        return NULL;
    }
    s->retry = AUTO_RETRY_MIN;
    s->reply_remaining = UINT32_MAX;
    s->last_reuse = s->pending_reuse = s->observation_reuse = 2;
    janas_cycle_init(&s->cycle, o->seed);
    janas_hybrid_init(&s->hybrid, o->seed);
    if (!o->draft_vocab_policy) janas_llm_mtp_vocab_legacy(m);
    planner_load(s);
    return s;
}

void janas_llm_session_destroy(struct janas_llm_session *s)
{
    if (s)
        free(s->dlogits);
    if (!s)
        return;
    planner_save(s);
    janas_history_index_destroy(s->index);
    free(s->hist);
    free(s->logits);
    free(s->hrow);
    janas_sampler_destroy(s->smp);
    free(s->lp);
    free(s);
}

void janas_llm_session_reset(struct janas_llm_session *s)
{
    discard_buffered(s);
    hybrid_settle(s, 0);
    janas_llm_model_media_drop(s->m, 0); /* the pictures go with it */
    s->dpos_set = 0;
    s->n = s->n_kv = s->ret = s->call_start = 0;
    s->n_mdraft = 0;
    s->mtp = janas_llm_model_has_mtp(s->m);
    s->reply_remaining = UINT32_MAX;
    s->last_reuse = 2;
    janas_llm_mtp_vocab_new_reply(s->m);
    memset(s->lp_pos, 0, sizeof(s->lp_pos));
}

/* A copy of what a session has computed (janas_llm_session_save). */
struct janas_llm_saved {
    int32_t *hist;
    uint32_t n, n_kv;
    struct janas_llm_state *st;  /* the model's state of hist[0 .. n_kv) */
    struct janas_llm_state *dst; /* the drafting model's, or NULL */
    size_t bytes;
};

struct janas_llm_saved *janas_llm_session_save(struct janas_llm_session *s)
{
    /* what was returned, the last token not computed: as after a truncate */
    uint32_t n = s->ret, n_kv = s->n_kv < n ? s->n_kv : n - 1;
    if (n < 2 || n_kv == 0)
        return NULL;
    struct janas_llm_saved *sv = calloc(1, sizeof(*sv));
    if (!sv)
        return NULL;
    sv->n = n;
    sv->n_kv = n_kv;
    sv->hist = malloc(n * sizeof(int32_t));
    sv->st = janas_llm_model_state_save(s->m, n_kv);
    if (!sv->hist || !sv->st) {
        janas_llm_saved_free(sv);
        return NULL;
    }
    memcpy(sv->hist, s->hist, n * sizeof(int32_t));
    sv->bytes =
        sizeof(*sv) + n * sizeof(int32_t) + janas_llm_state_bytes(sv->st);
    /* the drafting model's positions it holds for certain, as far as the
       target's go; without them it reads the conversation again */
    if (s->o.draft && s->dpos_set) {
        uint32_t dn = s->dpos + 1 < n_kv ? s->dpos + 1 : n_kv;
        sv->dst = janas_llm_model_state_save(s->o.draft, dn);
        if (sv->dst)
            sv->bytes += janas_llm_state_bytes(sv->dst);
    }
    return sv;
}

int janas_llm_session_restore(struct janas_llm_session *s,
                              const struct janas_llm_saved *sv)
{
    if (!sv->st || sv->n > s->cap ||
        janas_llm_model_state_load(s->m, sv->st) != 0)
        return -1;
    discard_buffered(s);
    hybrid_settle(s, 0); /* the restored timeline has no old recovery debt */
    memcpy(s->hist, sv->hist, sv->n * sizeof(int32_t));
    s->n = s->ret = sv->n;
    s->n_kv = s->call_start = sv->n_kv;
    s->n_mdraft = 0;
    memset(s->lp_pos, 0, sizeof(s->lp_pos));
    s->dpos_set = 0;
    s->reply_remaining = UINT32_MAX;
    janas_llm_mtp_vocab_new_reply(s->m);
    if (sv->dst && s->o.draft &&
        janas_llm_model_state_load(s->o.draft, sv->dst) == 0) {
        s->dpos = janas_llm_state_length(sv->dst) - 1;
        s->dpos_set = 1;
    }
    return 0;
}

uint32_t janas_llm_saved_tokens(const struct janas_llm_saved *sv,
                                const int32_t **tokens, uint32_t *n_kv)
{
    *tokens = sv->hist;
    if (n_kv)
        *n_kv = sv->n_kv;
    return sv->n;
}

size_t janas_llm_saved_bytes(const struct janas_llm_saved *sv)
{
    return sv->bytes;
}

/*
 * On disk: a mark, the version, n, n_kv, whether the drafting model's state
 * follows, the tokens, then the states (janas_llm_state_write).
 */
#define SAVED_MARK 0x56534b4au /* "JKSV" */

int janas_llm_saved_write(const struct janas_llm_session *s,
                          const struct janas_llm_saved *sv, FILE *f)
{
    int draft = sv->dst && s->o.draft;
    uint32_t h[5] = {SAVED_MARK, 1, sv->n, sv->n_kv, (uint32_t)draft};
    if (fwrite(h, sizeof(h), 1, f) != 1 ||
        fwrite(sv->hist, sizeof(int32_t), sv->n, f) != sv->n ||
        janas_llm_state_write(s->m, sv->st, f) != 0 ||
        (draft && janas_llm_state_write(s->o.draft, sv->dst, f) != 0))
        return -1;
    return 0;
}

struct janas_llm_saved *janas_llm_saved_read(const struct janas_llm_session *s,
                                             FILE *f, int tokens_only)
{
    uint32_t h[5];
    if (fread(h, sizeof(h), 1, f) != 1 || h[0] != SAVED_MARK || h[1] != 1 ||
        h[2] < 2 || h[3] == 0 || h[3] >= h[2] || h[2] > s->cap)
        return NULL;
    struct janas_llm_saved *sv = calloc(1, sizeof(*sv));
    if (!sv)
        return NULL;
    sv->n = h[2];
    sv->n_kv = h[3];
    sv->hist = malloc(sv->n * sizeof(int32_t));
    int bad = !sv->hist || fread(sv->hist, sizeof(int32_t), sv->n, f) != sv->n;
    for (uint32_t i = 0; !bad && i < sv->n; i++)
        bad = sv->hist[i] < 0 || (uint32_t)sv->hist[i] >= s->nv;
    if (!bad && !tokens_only) {
        sv->st = janas_llm_state_read(s->m, f);
        bad = !sv->st || janas_llm_state_length(sv->st) != sv->n_kv;
        /* the drafting model's is a help: without it, it reads again */
        if (!bad && h[4] && s->o.draft)
            sv->dst = janas_llm_state_read(s->o.draft, f);
    }
    if (bad) {
        janas_llm_saved_free(sv);
        return NULL;
    }
    sv->bytes = sizeof(*sv) + sv->n * sizeof(int32_t) +
                (sv->st ? janas_llm_state_bytes(sv->st) : 0) +
                (sv->dst ? janas_llm_state_bytes(sv->dst) : 0);
    return sv;
}

void janas_llm_saved_free(struct janas_llm_saved *sv)
{
    if (!sv)
        return;
    free(sv->hist);
    janas_llm_state_free(sv->st);
    janas_llm_state_free(sv->dst);
    free(sv);
}

uint32_t janas_llm_session_computed(const struct janas_llm_session *s)
{
    return s->n_kv;
}

uint32_t janas_llm_session_length(const struct janas_llm_session *s)
{
    return s->ret;
}

uint32_t janas_llm_session_tokens(const struct janas_llm_session *s,
                                  const int32_t **tokens)
{
    *tokens = s->hist;
    return s->ret;
}

int janas_llm_session_truncate(struct janas_llm_session *s, uint32_t n)
{
    if (n > s->ret)
        return -1;
    if (n == s->ret) /* nothing to drop but what nobody has seen */
        return janas_llm_session_append(s, NULL, 0);
    /* the last kept token is the one the next pass starts from: it must not
       count as computed, or its logits would be missing */
    uint32_t kv = n > 0 ? n - 1 : 0;
    if (kv < s->n_kv && janas_llm_model_recurrent(s->m))
        return -1;
    discard_buffered(s);
    janas_sampler_rewind(s->smp, s->n - s->ret); /* drawn, never returned */
    s->n = s->ret = n;
    janas_llm_model_media_drop(s->m, n); /* pictures past the cut */
    if (s->n_kv > kv)
        s->n_kv = kv;
    if (s->call_start > s->n_kv)
        s->call_start = s->n_kv;
    s->n_mdraft = 0;
    if (s->dpos_set && s->dpos >= s->n_kv)
        s->dpos_set = 0; /* the drafting model starts over */
    return 0;
}

void janas_llm_session_stats(const struct janas_llm_session *s,
                             struct janas_gen_stats *st)
{
    *st = s->st;
    st->forced_tokens = janas_sampler_forced(s->smp);
    for (int b = 0; b < CAL_BINS && b < 10; b++) {
        st->cal_tried[b] = (uint32_t)(s->pl.tried[b] + 0.5);
        st->cal_won[b] = (uint32_t)(s->pl.won[b] + 0.5);
    }
}

void janas_llm_session_metrics(const struct janas_llm_session *s,
                               struct janas_gen_metrics *out)
{
    if (!out) return;
    if (!s) { memset(out, 0, sizeof(*out)); return; }
    *out = s->metrics;
    out->buffered_tokens = s->n - s->ret;
    out->pending_preparation_seconds = s->pending.preparation_seconds;
}

void janas_llm_session_hybrid_metrics(const struct janas_llm_session *s,
                                     struct janas_gen_hybrid_metrics *out)
{
    if (!out) return;
    if (!s) { memset(out, 0, sizeof(*out)); return; }
    *out = s->hybrid_metrics;
    /* A final rejected pass may never require recovery. Expose its completed
       work immediately; the next forward can still add its causal recovery. */
    if (s->hybrid_waiting) {
        const struct janas_hybrid_observation *w = &s->hybrid_wait;
        if (w->source == JANAS_GEN_SOURCE_LOOKUP) {
            out->lookup_seconds += w->seconds;
            out->lookup_delivered += w->delivered;
        } else {
            out->mtp_seconds += w->seconds;
            out->mtp_delivered += w->delivered;
        }
        out->last = s->hybrid_wait_decision;
    }
}

void janas_llm_session_reply_remaining(struct janas_llm_session *s,
                                       uint32_t remaining)
{
    if (s) s->reply_remaining = remaining;
}

static int is_stop(const struct janas_gen_options *o, int32_t t)
{
    if (o->eos >= 0 && t == o->eos)
        return 1;
    for (int i = 0; i < o->n_stop; i++)
        if (t == o->stop[i])
            return 1;
    return 0;
}

int janas_llm_session_append(struct janas_llm_session *s, const int32_t *tokens,
                             uint32_t n)
{
    if (s->ret < s->n) {
        discard_buffered(s);
        /* drop the generated tokens nobody has seen; if the last kept one
           already ran, run it again: its MTP row paired it with a dropped
           token. Each was drawn, so the random sequence goes back as many:
           otherwise a reply cut short with drafts accepted past the cut
           left every later reply of the conversation drawn from other
           numbers than without drafts (1 Oct 2026) */
        janas_sampler_rewind(s->smp, s->n - s->ret);
        s->n = s->ret;
        if (s->n > 0 && s->n_kv > s->n - 1)
            s->n_kv = s->n - 1;
    }
    if (n == 0)
        return 0;
    if (n > s->cap + 1 - s->n)
        return -1;
    /* a new turn: the passes' costs learned again, on the cache and the
       context as they are now (the drafts' odds are kept) */
    memset(s->pl.t, 0, sizeof(s->pl.t));
    s->pl.chain = 0;
    memcpy(s->hist + s->n, tokens, n * sizeof(int32_t));
    s->n += n;
    s->ret = s->n;
    s->n_mdraft = 0; /* they followed the old last token */
    abandon_proposal(s);
    s->reply_remaining = UINT32_MAX;
    janas_llm_mtp_vocab_new_reply(s->m);
    if (s->mtp)
        janas_llm_mtp_note_tokens(s->m, tokens, n);
    return 0;
}

int janas_llm_session_prefill_step(struct janas_llm_session *s)
{
    struct janas_llm_model *m = s->m;
    double t0 = now();
    if (s->n_kv + 1 < s->n) {
        uint32_t p = s->n_kv, b = s->n - 1 - p;
        uint32_t most = janas_llm_model_max_block(m);
        if (b > most)
            b = most;
        b = janas_llm_model_media_cut(m, p, b);
        struct janas_llm_runtime_stats hb, ha;
        int settle = hybrid_cost(s) && s->hybrid_waiting;
        if (settle) janas_llm_model_runtime_stats(m, &hb);
        if (janas_llm_model_forward(m, s->hist + p, b, p, s->logits, 0) != 0) {
            hybrid_settle(s, 0);
            return -1;
        }
        if (settle) {
            janas_llm_model_runtime_stats(m, &ha);
            hybrid_settle(s, ha.gdn_recovery_seconds - hb.gdn_recovery_seconds);
        }
        s->call_start = p;
        /* the MTP block's rows: token t_{i+1} with hidden state h_i, at
           position i, so its attention sees the whole sequence */
        if (s->mtp &&
            janas_llm_mtp_forward(m, s->hist + p + 1, janas_llm_model_hidden(m),
                                  b, p, NULL, 0) != 0)
            return -1;
        s->n_kv += b;
        s->st.prompt_tokens += b;
    }
    s->st.prefill_seconds += now() - t0;
    return s->n_kv + 1 < s->n ? 1 : 0;
}

int janas_llm_session_prefill(struct janas_llm_session *s)
{
    int r;
    while ((r = janas_llm_session_prefill_step(s)) > 0)
        ;
    return r;
}

int janas_llm_session_bias(struct janas_llm_session *s, const int32_t *ids,
                           const float *bias, uint32_t n)
{
    return janas_sampler_bias(s->smp, ids, bias, n);
}

void janas_llm_session_filter(struct janas_llm_session *s,
                              const struct janas_sample_filter *f)
{
    janas_sampler_filter(s->smp, f);
    s->has_filter = f != NULL;
    if (s->has_filter && cycle_mode(s)) abandon_proposal(s);
}

void janas_llm_session_mark(struct janas_llm_session *s)
{
    janas_sampler_mark(s->smp);
}

int janas_llm_session_logprob(const struct janas_llm_session *s, uint32_t pos,
                              struct janas_token_lp *lp)
{
    if (!s->lp || s->lp_pos[pos % LP_RING] != pos + 1)
        return -1;
    *lp = s->lp[pos % LP_RING];
    return 0;
}

/* The token at position at, from its row of logits; -1 when the filter
   allows none. */
static int32_t sample(struct janas_llm_session *s, const float *lg, uint32_t at)
{
    struct janas_token_lp *lp = s->lp ? &s->lp[at % LP_RING] : NULL;
    int32_t t = janas_sampler_pick(s->smp, lg, lp);
    if (lp)
        s->lp_pos[at % LP_RING] = t >= 0 && s->o.logprobs ? at + 1 : 0;
    return t;
}

/*
 * Drafts from a second model. It keeps its own KV cache, and the tokens it
 * has processed are a prefix of the history: the drafts the target accepted
 * were its own, so their KV is right, and only the token the target chose
 * for itself is new to it. So a round costs exactly k of its passes, the
 * first of which is also the catch-up.
 *
 * Returns how many drafts were written, and the seconds one of its passes
 * cost (smoothed by the caller).
 */
static int draft_step(struct janas_llm_session *s, uint32_t pos, int k,
                      int32_t *draft, double *cost, int *cost_n)
{
    struct janas_llm_model *d = s->o.draft;
    uint32_t nv = s->nv;
    int n = 0;
    double spent = 0;
    /*
     * Only passes over one token are counted towards what a draft costs.
     * Catching up on several at once happens when the planner has been
     * refusing drafts, and charging that to the one draft that follows made
     * the refusal feed itself: the first catch-up is the whole prompt, so
     * the first draft looked to cost a hundred milliseconds and no draft was
     * ever worth it again.
     */
    *cost_n = 0;
    /* whatever the drafting model has not seen yet, up to and including the
       token at pos: normally one, the target's own choice of last round */
    uint32_t from = s->dpos_set ? s->dpos + 1 : 0;
    if (from > pos)
        from = pos; /* its KV went past the history: rewrite from here */
    uint32_t most = janas_llm_model_max_block(d);
    for (uint32_t i = from; i <= pos; i += most) {
        uint32_t b = pos - i + 1;
        if (b > most)
            b = most;
        double t0 = now();
        if (janas_llm_model_forward(d, s->hist + i, b, i, s->dlogits, 0) != 0)
            return -1;
        double dt = now() - t0;
        if (b == 1) {
            spent += dt;
            ++*cost_n;
        }
    }
    s->dpos = pos;
    s->dpos_set = 1;
    while (n < k) {
        int32_t best = 0;
        for (uint32_t v = 1; v < nv; v++)
            if (s->dlogits[v] > s->dlogits[best])
                best = (int32_t)v;
        draft[n++] = best;
        if (n == k)
            break;
        double t0 = now();
        if (janas_llm_model_decode(d, best, pos + (uint32_t)n, s->dlogits) != 0)
            return -1;
        spent += now() - t0;
        ++*cost_n;
        s->dpos = pos + (uint32_t)n;
    }
    *cost = spent;
    return n;
}

/* Prepare the next cycle from canonical tokens only. In hybrid mode the
   history index is consulted before any MTP projection or chained state. */
static int prepare_cycle(struct janas_llm_session *s, uint32_t pos,
                          uint32_t acc, int32_t final_token,
                          double *maintenance)
{
    const struct janas_gen_options *o = &s->o;
    uint32_t made = acc + 1;
    uint32_t remaining = s->reply_remaining;
    if (remaining != UINT32_MAX) remaining = remaining > made ? remaining - made : 0;
    int ongoing = !is_stop(o, final_token) && remaining != 0 && s->n <= s->cap;
    unsigned cap = (unsigned)drafts_cap(s->m, o->spec_k);
    /* Stateful filter callbacks and penalty counts have no rollback API.
       Width one preserves their transcript even after an arbitrary caller
       cancellation, rather than advancing state for unread draft outputs. */
    if (s->has_filter || o->presence_penalty != 0 || o->frequency_penalty != 0)
        cap = 0;
    if (cap > s->cap - (s->n <= s->cap ? s->n : s->cap))
        cap = s->cap - (s->n <= s->cap ? s->n : s->cap);
    if (remaining != UINT32_MAX && cap >= remaining) cap = remaining ? remaining - 1 : 0;
    unsigned source = JANAS_GEN_SOURCE_PLAIN, want = 0, matched_suffix = 0;
    unsigned warm = !janas_llm_model_settling(s->m), policy = 0, rows = 0;
    int cost_hybrid = hybrid_cost(s);
    unsigned mtp_cap = cap;
    if (cost_hybrid) {
        unsigned limit = o->hybrid_mtp_max_drafts ? (unsigned)o->hybrid_mtp_max_drafts : 4;
        if (mtp_cap > limit) mtp_cap = limit;
        policy = (unsigned)janas_llm_mtp_vocab_policy(s->m);
        rows = janas_llm_mtp_vocab_rows(s->m);
    }
    struct janas_gen_hybrid_decision hd = {0};
    if (cost_hybrid && !(ongoing && cap)) s->hybrid.cycles++;
    uint64_t vocab_epoch = 0;
    double start = now(), lookup_time = 0, controller_time = 0, repack_time = 0;
    int32_t proposal[MAX_SPEC_K];
    if (ongoing && cap && o->spec_mode != JANAS_SPEC_OFF) {
        if (o->spec_mode == JANAS_SPEC_LOOKUP || o->spec_mode == JANAS_SPEC_HYBRID) {
            double tl = now();
            if (!s->index) s->index = janas_history_index_create(8u * 1024u * 1024u);
            int avail = 0, suffix = 0;
            unsigned lookup_cap = cap;
            if (cost_hybrid) {
                unsigned limit = o->hybrid_lookup_max_drafts ? (unsigned)o->hybrid_lookup_max_drafts : 8;
                if (lookup_cap > limit) lookup_cap = limit;
            }
            /* Lazy synchronization also handles append/truncate/slide/load;
               rebuilding is charged to this proposal, not hidden in a turn. */
            if (s->index && janas_history_index_sync(s->index, s->hist, s->n, 0) == 0)
                avail = janas_history_index_lookup(s->index, s->hist, s->n,
                                                   (int)lookup_cap, proposal, &suffix);
            matched_suffix = (unsigned)suffix;
            lookup_time = now() - tl;
            s->metrics.lookup_seconds += lookup_time;
            if (avail > 0) s->metrics.lookup_hits++; else s->metrics.lookup_misses++;
            source = JANAS_GEN_SOURCE_LOOKUP;
            double tc = now();
            if (cost_hybrid) {
                struct janas_hybrid_offer offer = {
                    .lookup_available = (unsigned)avail, .lookup_suffix = matched_suffix,
                    .lookup_cap = lookup_cap, .mtp_cap = mtp_cap, .warm = warm,
                    .vocab = janas_cycle_vocab_key(policy, rows), .reuse = s->last_reuse,
                    .probe_percent = (unsigned)o->hybrid_probe_percent
                };
                struct janas_hybrid_decision d = janas_hybrid_choose(&s->hybrid, &offer);
                source = d.source;
                want = source == JANAS_GEN_SOURCE_LOOKUP ? d.depth : 0;
                hd.source = d.source; hd.requested_depth = d.depth;
                hd.lookup_available = (unsigned)avail; hd.lookup_suffix = matched_suffix;
                hd.suffix_class = d.suffix_class; hd.reason = d.reason; hd.probe = d.probe;
                hd.warm = warm; hd.vocab_key = offer.vocab;
                hd.lookup_evidence = d.lookup_evidence; hd.mtp_evidence = d.mtp_evidence;
                hd.lookup_utility = d.lookup_utility; hd.mtp_utility = d.mtp_utility;
                hd.predicted_seconds = source == JANAS_GEN_SOURCE_LOOKUP ? d.lookup_seconds : d.mtp_seconds;
                s->hybrid_metrics.decisions++;
                s->hybrid_metrics.offered_matches += avail > 0;
                s->hybrid_metrics.eligible_matches += avail > 0 && lookup_cap > 0;
                s->hybrid_metrics.lookup_selected += source == JANAS_GEN_SOURCE_LOOKUP;
                s->hybrid_metrics.mtp_selected += source == JANAS_GEN_SOURCE_MTP;
                s->hybrid_metrics.probes += d.probe != 0;
                s->hybrid_metrics.cooldown_skips += d.reason == JANAS_HYBRID_COOLDOWN;
                if (d.reason < 12) s->hybrid_metrics.reason_count[d.reason]++;
                if (source == JANAS_GEN_SOURCE_LOOKUP && d.depth <= MAX_SPEC_K)
                    s->hybrid_metrics.lookup_depth_count[d.depth]++;
            } else want = o->spec_planner == 1
                ? janas_cycle_choose_reuse(&s->cycle, source, warm, 0, (unsigned)avail,
                                     (unsigned)o->spec_explore_percent, s->last_reuse)
                : (unsigned)avail;
            controller_time += now() - tc;
        }
        if (cap && (o->spec_mode == JANAS_SPEC_MTP ||
            (o->spec_mode == JANAS_SPEC_HYBRID && want == 0))) {
            source = JANAS_GEN_SOURCE_MTP;
            int32_t special[JANAS_GEN_MAX_STOP + 1];
            uint32_t ns = 0;
            if (o->eos >= 0) special[ns++] = o->eos;
            for (int i = 0; i < o->n_stop; i++) special[ns++] = o->stop[i];
            struct janas_mtp_costs before, after;
            janas_llm_mtp_costs(s->m, &before);
            /* acc is the last VALID target-logit row, including a first
               mismatch. Later rows condition on rejected tokens. */
            (void)janas_llm_mtp_vocab_prepare(s->m, o->draft_vocab_policy,
                o->draft_vocab_rows, s->logits + (size_t)acc * s->nv,
                s->hist, s->n, special, ns);
            janas_llm_mtp_costs(s->m, &after);
            repack_time = after.repack_seconds - before.repack_seconds;
            policy = (unsigned)janas_llm_mtp_vocab_policy(s->m);
            rows = janas_llm_mtp_vocab_rows(s->m);
            double tc = now();
            want = cost_hybrid ? mtp_cap : o->spec_planner == 1
                ? janas_cycle_choose_reuse(&s->cycle, source, warm,
                    janas_cycle_vocab_key(policy, rows), cap,
                    (unsigned)o->spec_explore_percent, s->last_reuse) : cap;
            controller_time += now() - tc;
        } else if (o->spec_mode == JANAS_SPEC_DRAFT) {
            source = JANAS_GEN_SOURCE_DRAFT;
            double tc = now();
            want = o->spec_planner == 1
                ? janas_cycle_choose_reuse(&s->cycle, source, warm, 0, cap,
                                     (unsigned)o->spec_explore_percent, s->last_reuse) : cap;
            controller_time += now() - tc;
            if (want) {
                double spent = 0;
                int count = 0;
                double td = now();
                int got = draft_step(s, s->n - 1, (int)want, proposal, &spent, &count);
                if (got < 0) return -1;
                want = (unsigned)got;
                s->metrics.draft_seconds += now() - td;
                s->st.draft_seconds += spent;
            }
        }
    }
    double preparation = now() - start;
    struct mtp_times mt = {0};
    if (s->mtp && mtp_step(s->m, s->hist, pos, acc, (int)want,
            source == JANAS_GEN_SOURCE_MTP && want > 0,
            cost_hybrid ? o->spec_min_conf : 0,
            cost_hybrid && o->spec_auto ? mtp_planner(s) : NULL,
            s->hrow, s->mdraft, s->msurv, &s->n_mdraft, &mt) != 0)
        return -1;
    *maintenance = mt.maintenance;
    s->metrics.mtp_maintenance_seconds += mt.maintenance;
    s->metrics.mtp_head_seconds += mt.head;
    s->metrics.mtp_chain_seconds += mt.chain;
    s->metrics.mtp_repack_seconds += repack_time;
    s->metrics.controller_seconds += controller_time;
    if (source == JANAS_GEN_SOURCE_MTP) {
        /* note_tokens in mtp_step may have extended the legacy row set. */
        struct janas_mtp_costs resolved;
        janas_llm_mtp_costs(s->m, &resolved);
        policy = (unsigned)janas_llm_mtp_vocab_policy(s->m);
        rows = janas_llm_mtp_vocab_rows(s->m);
        vocab_epoch = resolved.vocab_epoch;
    }
    if (source != JANAS_GEN_SOURCE_MTP) {
        s->n_mdraft = (int)want;
        if (want) memcpy(s->mdraft, proposal, want * sizeof(*proposal));
    }
    for (int i = 0; i < s->n_mdraft; i++)
        if (is_stop(o, s->mdraft[i])) { s->n_mdraft = i + 1; break; }
    memset(&s->pending, 0, sizeof(s->pending));
    if (ongoing) {
        s->pending.proposal_id = ++s->next_proposal_id;
        s->pending.source = s->n_mdraft ? source : JANAS_GEN_SOURCE_PLAIN;
        s->pending.lookup_suffix = matched_suffix; /* offered, even if MTP won */
        s->pending.planner_source = source;
        s->pending.requested_depth = want;
        s->pending.warm = warm;
        s->pending.draft_vocab_policy = policy;
        s->pending.draft_vocab_rows = rows;
        s->pending.preparation_seconds = preparation + mt.head + mt.chain;
        s->pending.controller_seconds = controller_time;
        s->pending_reuse = s->last_reuse;
        s->pending_vocab_epoch = vocab_epoch;
        hd.proposal_id = s->pending.proposal_id;
        hd.vocab_key = janas_cycle_vocab_key(policy, rows);
        s->hybrid_pending = hd;
        s->metrics.proposals += s->n_mdraft > 0;
    } else {
        s->n_mdraft = 0;
    }
    return 0;
}

int janas_llm_session_next(struct janas_llm_session *s, int32_t *token)
{
    if (s->ret < s->n) {
        *token = s->hist[s->ret++];
        delivered(s);
        return 0;
    }
    if (s->n == 0)
        return -1;
    if (s->n > s->cap)
        return 1;
    if (janas_llm_session_prefill(s) != 0)
        return -1;

    struct janas_llm_model *m = s->m;
    const struct janas_gen_options *o = &s->o;
    uint32_t nv = s->nv, pos = s->n - 1;
    double t0 = now();
    int measured = cycle_mode(s);
    int drafts_wanted = o->spec_mode == JANAS_SPEC_MTP || o->spec_mode == JANAS_SPEC_HYBRID;
    int cost_hybrid = hybrid_cost(s);
    int plan = (!measured || cost_hybrid) && s->mtp && drafts_wanted && o->spec_auto;
    int32_t block[MAX_SPEC_K + 1];
    block[0] = s->hist[pos];
    int nd = 0;
    int copies = 0;
    struct janas_gen_cycle current = s->pending;
    uint64_t current_vocab_epoch = current.proposal_id ? s->pending_vocab_epoch : 0;
    unsigned current_reuse = current.proposal_id ? s->pending_reuse : 2;
    s->hybrid_current = s->hybrid_pending;
    memset(&s->hybrid_pending, 0, sizeof(s->hybrid_pending));
    if (!current.proposal_id) {
        current.proposal_id = ++s->next_proposal_id;
        current.source = JANAS_GEN_SOURCE_PLAIN;
        current.planner_source = drafts_wanted ? JANAS_GEN_SOURCE_MTP
            : o->spec_mode == JANAS_SPEC_LOOKUP ? JANAS_GEN_SOURCE_LOOKUP
            : o->spec_mode == JANAS_SPEC_DRAFT ? JANAS_GEN_SOURCE_DRAFT
            : JANAS_GEN_SOURCE_PLAIN;
        current.warm = !janas_llm_model_settling(m);
        current.draft_vocab_policy = (uint32_t)janas_llm_mtp_vocab_policy(m);
        current.draft_vocab_rows = janas_llm_mtp_vocab_rows(m);
    }
    memset(&s->pending, 0, sizeof(s->pending));
    if (measured) {
        nd = s->n_mdraft;
        if (cost_hybrid && current.planner_source == JANAS_GEN_SOURCE_MTP && plan) {
            nd = plan_drafts(mtp_planner(s), s->msurv, s->n_mdraft);
            if (!nd && s->n_mdraft > 0 && s->st.passes % 16 == 0) nd = 1;
        }
        if (s->has_filter || o->presence_penalty != 0 || o->frequency_penalty != 0)
            nd = 0;
        memcpy(block + 1, s->mdraft, (size_t)nd * sizeof(int32_t));
        copies = current.source == JANAS_GEN_SOURCE_LOOKUP ||
                 current.source == JANAS_GEN_SOURCE_DRAFT;
    } else {
        /*
         * spec_auto needs the cost of a plain pass to compare the drafts with,
         * and it must be a pass of the same work: one in AUTO_PROBE makes no
         * draft, so the baseline follows the context as it grows and the cache
         * as it changes. Measured once at the start of a session, as it was, it
         * was the cost of the shortest context and the warmest cache, and the
         * comparison leaned against the drafts for the rest of the session.
         * Only where the drafts come from the text itself: with an MTP block
         * the planner decides how many to use and this judgement never runs.
         */
        int probe =
            o->spec_auto && !plan &&
            (s->single_cost == 0 || s->st.passes - s->single_at >= AUTO_PROBE);
        /* drafting again after a spell without it: what was learned then was
           learned of other work, and judging on it is how the drafts used to
           switch themselves off for good */
        int can_draft = o->spec_mode != JANAS_SPEC_OFF && s->off_left == 0;
        if (can_draft && !s->could_draft) {
            s->spec_cost = s->gain = 0;
            s->spec_seen = 0;
            s->strikes = 0;
            /* and the planner's table of pass costs: those were measured before
               the pause, and the machine has moved since - the expert cache
               warms as a conversation goes on, so a plain pass timed now against
               a block pass timed then is no comparison at all. Left as they
               were, the drafts came back about one time in three (seven runs of
               the sequence on 21 Sep 2026). */
            memset(s->pl.t, 0, sizeof(s->pl.t));
            s->pl.chain = 0;
        }
        s->could_draft = can_draft;
        /* drafts that are copied or predicted by another model: both are judged
           on the rate at which each position has been surviving, and one pass in
           sixteen drafts anyway, because refusing every draft would stop those
           counts from ever moving again */
        copies = o->spec_mode == JANAS_SPEC_LOOKUP || o->spec_mode == JANAS_SPEC_DRAFT;
        if (o->spec_mode == JANAS_SPEC_LOOKUP && s->off_left == 0 && !probe) {
            double tl = now();
            nd = lookup_draft(s->hist, s->n, drafts_cap(m, o->spec_k), block + 1);
            s->metrics.lookup_seconds += now() - tl;
            if (nd > 0) s->metrics.lookup_hits++; else s->metrics.lookup_misses++;
            if (nd > 0) {
                int want = plan_lookup(&s->pl, nd, 0.0);
                if (!want && s->st.passes % 16 == 0)
                    want = 1;
                nd = want;
            }
        }
        if (o->spec_mode == JANAS_SPEC_DRAFT && s->off_left == 0 && !probe) {
            int want = plan_lookup(&s->pl, drafts_cap(m, o->spec_k), s->pl.dcost);
            if (!want && s->st.passes % 16 == 0)
                want = 1;
            if (want > 0) {
                double spent = 0;
                int cost_n = 0;
                double td = now();
                int got = draft_step(s, pos, want, block + 1, &spent, &cost_n);
                if (got < 0)
                    return -1;
                s->metrics.draft_seconds += now() - td;
                nd = got;
                s->st.draft_seconds += spent;
                if (cost_n > 0) {
                    double per = spent / cost_n;
                    s->pl.dcost = s->pl.dcost ? 0.8 * s->pl.dcost + 0.2 * per : per;
                }
            }
        }
        if (s->mtp && drafts_wanted && s->off_left == 0 && !probe) {
            nd = plan ? plan_drafts(&s->pl, s->msurv, drafts_cap(m, s->n_mdraft))
                      : s->n_mdraft;
            /* one pass in sixteen drafts anyway, as the copies do: a pass of
               two tokens timed once while the disk was busy made the planner
               refuse every draft, and with none its cost was never timed again
               - Qwen3-Next-80B with a 4 GiB cache drafted 33 times in a
               session and then never (1 Oct 2026) */
            if (plan && nd == 0 && s->n_mdraft > 0 && s->st.passes % 16 == 0)
                nd = 1;
            memcpy(block + 1, s->mdraft, (size_t)nd * sizeof(int32_t));
        }
        current.preparation_seconds += now() - t0;
        current.requested_depth = (uint32_t)nd;
        current.source = nd == 0 ? JANAS_GEN_SOURCE_PLAIN
            : o->spec_mode == JANAS_SPEC_LOOKUP ? JANAS_GEN_SOURCE_LOOKUP
            : o->spec_mode == JANAS_SPEC_DRAFT ? JANAS_GEN_SOURCE_DRAFT : JANAS_GEN_SOURCE_MTP;
        if (copies && nd > 0) s->metrics.proposals++;
    }
    if ((uint32_t)nd > s->cap - s->n)
        nd = (int)(s->cap - s->n); /* the context ends */
    if (s->reply_remaining != UINT32_MAX && s->reply_remaining > 0 &&
        (uint32_t)nd >= s->reply_remaining)
        nd = (int)s->reply_remaining - 1;
    if (!nd) current.source = JANAS_GEN_SOURCE_PLAIN;
    if (measured && s->has_filter) current.guard_reason |= JANAS_GEN_GUARD_FILTER;
    if (measured && (o->presence_penalty != 0 || o->frequency_penalty != 0))
        current.guard_reason |= JANAS_GEN_GUARD_PENALTY;
    if (s->off_left > 0) {
        s->off_left--;
        s->st.auto_off_passes++;
    }
    struct janas_llm_runtime_stats runtime_before, runtime_after;
    double tr = now();
    janas_llm_model_runtime_stats(m, &runtime_before);
    double runtime_control = now() - tr;
    double tp = now();
    if (janas_llm_model_forward(m, block, (uint32_t)nd + 1, pos, s->logits,
                                1) != 0) {
        hybrid_settle(s, 0);
        return -1;
    }
    double dt = now() - tp;
    tr = now();
    janas_llm_model_runtime_stats(m, &runtime_after);
    s->hybrid_incoming_recovery = 0;
    if (cost_hybrid && s->hybrid_waiting) {
        double recovery = runtime_after.gdn_recovery_seconds - runtime_before.gdn_recovery_seconds;
        recovery = fmin(dt, fmax(0, recovery));
        hybrid_settle(s, recovery);
        s->hybrid_incoming_recovery = recovery;
    }
    if (runtime_after.expert_groups > runtime_before.expert_groups &&
        runtime_after.expert_assignments >= runtime_before.expert_assignments) {
        uint64_t groups = runtime_after.expert_groups - runtime_before.expert_groups;
        uint64_t assignments = runtime_after.expert_assignments - runtime_before.expert_assignments;
        /* Observational only: no future routers/IDs are queried. The next
           proposal uses the group reuse actually seen in this target pass. */
        s->last_reuse = assignments / groups >= 2 ? 1 : 0;
    }
    runtime_control += now() - tr;
    s->metrics.controller_seconds += runtime_control;
    /* what the planner and spec_auto learn: the pass without its waits for
       the disk, and nothing from a pass the tuner ran on an alternative it
       was trying. In a new conversation it tries each configuration in turn
       on the passes of 2 to 16 tokens, those that verify the drafts, and
       learned from the slow ones (279-357 ms a pass where its choice takes
       50), the planner refused the drafts: Qwen3-Next-80B-A3B wrote a reply
       of 64 tokens with 4 drafts at 24.7 token/s, where janas-bench, its
       configuration forced, wrote with drafts at 34.8 (21 Oct 2026). The
       statistics keep the time as it was. */
    double cost = janas_llm_model_pass_seconds(m);
    int learn = cost > 0;
    s->call_start = pos;
    s->st.passes++;
    /* accept the drafts the model would have produced, then its token; a
       stop token ends the pass (and is never counted as accepted) */
    int acc = 0;
    double sampled = 0;
    int32_t t;
    for (;;) {
        double ts = now();
        t = sample(s, s->logits + (size_t)acc * nv, s->n);
        double sampling_time = now() - ts;
        s->st.sample_seconds += sampling_time;
        sampled += sampling_time;
        if (t < 0)
            return -1; /* the filter allows nothing: a broken grammar */
        if (acc < nd && t == block[acc + 1] && !is_stop(o, t)) {
            s->hist[s->n++] = t;
            acc++;
            continue;
        }
        break;
    }
    s->hist[s->n++] = t;
    s->n_kv = pos + (uint32_t)acc + 1;
    s->st.new_tokens += (uint32_t)acc + 1;
    int settling_now = janas_llm_model_settling(s->m);
    for (int i = 0; i < nd; i++) {
        s->st.pos_tried[i]++;
        s->st.pos_won[i] += i < acc;
    }
    if (!settling_now) {
        s->st.pass_count[nd + 1]++;
        s->st.pass_seconds[nd + 1] += dt;
    }
    if ((plan || copies) && !settling_now && learn) {
        struct planner *p = cost_hybrid ? mtp_planner(s) : &s->pl;
        double *tt = &p->t[nd + 1];
        *tt = *tt > 0 ? 0.8 * *tt + 0.2 * cost : cost;
    }
    if (o->spec_mode == JANAS_SPEC_DRAFT && nd > 0)
        /* the drafts the target took were the drafting model's own, so its
           KV holds for them; the last one it never fed back to itself */
        s->dpos = pos + (uint32_t)(acc < nd ? acc : nd - 1);
    if (copies && nd > 0 && !settling_now) {
        for (int i = 0; i < nd; i++) {
            s->pl.lk_tried[i] += 1;
            s->pl.lk_won[i] += i < acc;
        }
        if (s->pl.lk_tried[0] >= LK_WINDOW)
            for (int i = 0; i < MAX_SPEC_K; i++) {
                s->pl.lk_tried[i] *= 0.5;
                s->pl.lk_won[i] *= 0.5;
            }
    }
    /* what a draft of that guessed chance was worth, whoever asked for it:
       the planner only learns about the bins it tries, so drafts made under
       a fixed threshold teach it too */
    if ((!measured || cost_hybrid) && current.source == JANAS_GEN_SOURCE_MTP &&
        (!o->draft_vocab_policy || cost_hybrid))
        for (int i = 0; i < nd; i++) {
            struct planner *p = cost_hybrid ? mtp_planner(s) : &s->pl;
            p->tried[cal_bin(s->msurv[i])] += 1;
            p->won[cal_bin(s->msurv[i])] += i < acc;
        }
    /* a pass run while the expert cache is still filling in the background
       measures the disk, not the drafts: it teaches nothing */
    int settling = settling_now;
    if (nd > 0) {
        s->st.spec_passes++;
        s->st.drafted += (uint32_t)nd;
        s->st.accepted += (uint32_t)acc;
        if (!settling && learn) {
            s->spec_cost =
                s->spec_cost ? 0.8 * s->spec_cost + 0.2 * cost : cost;
            s->gain = s->gain ? 0.8 * s->gain + 0.2 * (acc + 1) : acc + 1;
            s->spec_seen++;
        }
        /* worth it only if tokens per second beat single passes, and only
           on evidence fresh enough to be about the same work (see above) */
        if (!measured && o->spec_auto && !plan && !settling && s->single_cost > 0 &&
            s->spec_seen >= AUTO_WARM &&
            s->st.passes - s->single_at <= AUTO_FRESH) {
            if (s->gain / s->spec_cost < AUTO_MARGIN / s->single_cost) {
                if (++s->strikes >= AUTO_STRIKES) {
                    s->off_left = s->retry;
                    s->retry = s->retry * 2 > AUTO_RETRY_MAX ? AUTO_RETRY_MAX
                                                             : s->retry * 2;
                    s->strikes = 0;
                }
            } else {
                s->strikes = 0;
                s->retry = AUTO_RETRY_MIN;
            }
        }
    } else if (settling || !learn) {
        /* nothing learned while the cache fills, nor from a try */
    } else {
        s->single_cost =
            s->single_cost ? 0.8 * s->single_cost + 0.2 * cost : cost;
        s->single_at = s->st.passes;
    }
    /* the MTP rows of the new tokens always (the block's KV cache must stay
       complete for later turns), drafts only when generation goes on */
    double maintenance = 0;
    if (cost_hybrid && current.planner_source == JANAS_GEN_SOURCE_LOOKUP && nd && !acc) {
        unsigned key = janas_cycle_vocab_key(current.draft_vocab_policy, current.draft_vocab_rows);
        unsigned cls = janas_hybrid_suffix_class(current.lookup_suffix);
        s->hybrid.cooldown_until[!!current.warm][key][cls] = s->hybrid.cycles + 33;
    }
    if (measured) {
        if (prepare_cycle(s, pos, (uint32_t)acc, t, &maintenance) != 0)
            return -1;
    } else if (s->mtp) {
        struct janas_mtp_costs before_prepare, after_prepare;
        janas_llm_mtp_costs(m, &before_prepare);
        double tv = now();
        if (drafts_wanted && !is_stop(o, t)) {
            int32_t special[JANAS_GEN_MAX_STOP + 1];
            uint32_t ns = 0;
            if (o->eos >= 0) special[ns++] = o->eos;
            for (int i = 0; i < o->n_stop; i++) special[ns++] = o->stop[i];
            (void)janas_llm_mtp_vocab_prepare(m, o->draft_vocab_policy,
                o->draft_vocab_rows, s->logits + (size_t)acc * nv,
                s->hist, s->n, special, ns);
        }
        double vocab_preparation = now() - tv;
        janas_llm_mtp_costs(m, &after_prepare);
        s->metrics.mtp_repack_seconds += after_prepare.repack_seconds - before_prepare.repack_seconds;
        struct mtp_times mt;
        int future_k = plan ? drafts_cap(m, o->spec_k) : o->spec_k;
        uint32_t future = s->reply_remaining;
        if (future != UINT32_MAX) {
            future = future > (uint32_t)acc + 1 ? future - (uint32_t)acc - 1 : 0;
            if ((uint32_t)future_k >= future) future_k = future ? (int)future - 1 : 0;
        }
        if (mtp_step(m, s->hist, pos, (uint32_t)acc, future_k,
                      future_k > 0 && drafts_wanted && s->off_left == 0 && !is_stop(o, t),
                     o->spec_min_conf,
                     plan ? &s->pl : NULL, s->hrow, s->mdraft, s->msurv,
                      &s->n_mdraft, &mt) != 0)
            return -1;
        maintenance = mt.maintenance;
        s->metrics.mtp_maintenance_seconds += mt.maintenance;
        s->metrics.mtp_head_seconds += mt.head;
        s->metrics.mtp_chain_seconds += mt.chain;
        s->pending.preparation_seconds = vocab_preparation + mt.head + mt.chain;
        s->pending.proposal_id = ++s->next_proposal_id;
        s->pending.source = JANAS_GEN_SOURCE_MTP;
        s->pending.planner_source = JANAS_GEN_SOURCE_MTP;
        s->pending.draft_vocab_policy = (uint32_t)janas_llm_mtp_vocab_policy(m);
        s->pending.draft_vocab_rows = janas_llm_mtp_vocab_rows(m);
        s->pending_vocab_epoch = after_prepare.vocab_epoch;
        s->metrics.proposals += s->n_mdraft > 0;
    }
    current.pass_width = (uint32_t)nd + 1;
    current.resolved_planner = (uint32_t)o->spec_planner;
    current.accepted_prefix = (uint32_t)acc;
    current.produced = (uint32_t)acc + 1;
    current.buffered = current.produced;
    current.rejected_rows = (uint32_t)(nd - acc);
    current.verification_seconds = dt;
    current.sample_seconds = sampled;
    current.maintenance_seconds = maintenance;
    current.controller_seconds += runtime_control;
    current.total_seconds = current.preparation_seconds + dt + sampled + maintenance + runtime_control;
    if (current.source == JANAS_GEN_SOURCE_MTP && nd > 0)
        janas_llm_mtp_vocab_feedback_epoch(m, current_vocab_epoch, current.draft_vocab_policy,
            current.draft_vocab_rows, (uint32_t)nd, (uint32_t)acc,
            current.total_seconds);
    s->metrics.last = current;
    s->observation_pending = measured;
    s->observation_reuse = current_reuse;
    s->metrics.cycles++;
    s->metrics.verification_rows += current.pass_width;
    s->metrics.accepted_prefix += (uint32_t)acc;
    s->metrics.produced_tokens += current.produced;
    s->metrics.rejected_rows += current.rejected_rows;
    s->metrics.filter_guard_cycles += !!(current.guard_reason & JANAS_GEN_GUARD_FILTER);
    s->metrics.penalty_guard_cycles += !!(current.guard_reason & JANAS_GEN_GUARD_PENALTY);
    s->metrics.source_cycles[current.source]++;
    s->metrics.source_drafted[current.source] += (uint32_t)nd;
    s->metrics.source_accepted[current.source] += (uint32_t)acc;
    s->metrics.source_seconds[current.source] += current.total_seconds;
    if (cost_hybrid && current.planner_source == JANAS_GEN_SOURCE_LOOKUP) {
        s->hybrid_metrics.lookup_drafted += (uint32_t)nd;
        s->hybrid_metrics.lookup_accepted += (uint32_t)acc;
    }
    if (current.source == JANAS_GEN_SOURCE_LOOKUP && current.lookup_suffix <= 8)
        s->metrics.lookup_suffix_count[current.lookup_suffix]++;
    s->metrics.pass_width_count[nd + 1]++;
    s->metrics.prefix_count[acc]++;
    s->metrics.verification_seconds += dt;
    s->metrics.sample_seconds += sampled;
    s->metrics.cycle_seconds += current.total_seconds;
    s->st.decode_seconds += now() - t0;
    *token = s->hist[s->ret++];
    delivered(s);
    return 0;
}

int janas_llm_generate(struct janas_llm_model *m, const int32_t *prompt,
                       uint32_t n_prompt, uint32_t max_new,
                       const struct janas_gen_options *o, int32_t *out,
                       struct janas_gen_stats *st)
{
    memset(st, 0, sizeof(*st));
    if (n_prompt == 0)
        return -1;
    struct janas_llm_session *s = janas_llm_session_create(m, o);
    if (!s)
        return -1;
    int produced = 0;
    if (janas_llm_session_append(s, prompt, n_prompt) != 0)
        produced = -1;
    janas_llm_session_reply_remaining(s, max_new);
    while (produced >= 0 && (uint32_t)produced < max_new) {
        int r = janas_llm_session_next(s, &out[produced]);
        if (r < 0)
            produced = -1;
        if (r != 0)
            break;
        if (is_stop(o, out[produced++]))
            break;
    }
    janas_llm_session_stats(s, st);
    if (produced >= 0)
        st->new_tokens = (uint32_t)produced;
    janas_llm_session_destroy(s);
    return produced;
}
