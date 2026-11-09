/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Source-policy fixtures: no weights, inference or target sampler. */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm/hybrid_controller.h"

static int failures;
#define CHECK(c, msg) do { if (!(c)) { \
    printf("FAIL: %s (line %d)\n", msg, __LINE__); failures++; \
} } while (0)

static struct janas_hybrid_offer offer(unsigned suffix, unsigned probe)
{
    struct janas_hybrid_offer o = {8, suffix, 8, 4, 1, 0, 2, probe};
    return o;
}
static void observe(struct janas_hybrid_controller *p, unsigned source,
                     unsigned depth, unsigned accepted, unsigned delivered,
                     unsigned suffix, double seconds)
{
    struct janas_hybrid_observation o = {
        source, depth, accepted, delivered, depth + 1, suffix, 1, 0, 2,
        seconds, seconds * 0.8
    };
    janas_hybrid_observe(p, &o);
}
static void mtp(struct janas_hybrid_controller *p)
{
    for (unsigned i = 0; i < 3; i++)
        observe(p, JANAS_HYBRID_MTP, 4, 4, 5, 0, 1.0);
}

static void cold_and_caps(void)
{
    struct janas_hybrid_controller p;
    janas_hybrid_init(&p, 0);
    struct janas_hybrid_offer o = offer(8, 0);
    for (unsigned i = 0; i < 100; i++) {
        struct janas_hybrid_decision d = janas_hybrid_choose(&p, &o);
        CHECK(d.source == JANAS_HYBRID_MTP && d.depth == 4 && !d.probe,
              "cold controller preserves calibrated MTP ceiling without a sweep");
    }
    o.probe_percent = 100;
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_MTP,
          "unknown MTP cost defers even forced lookup probes");
    mtp(&p);
    struct janas_hybrid_decision d = janas_hybrid_choose(&p, &o);
    CHECK(d.source == JANAS_HYBRID_LOOKUP && d.depth == 2 && d.probe,
          "long suffix starts shallowly after measured MTP cost");
    o.lookup_suffix = 2;
    CHECK(janas_hybrid_choose(&p, &o).depth == 1,
          "short suffix probes one draft");
    o.lookup_available = 0;
    d = janas_hybrid_choose(&p, &o);
    CHECK(d.source == JANAS_HYBRID_MTP && d.reason == JANAS_HYBRID_NO_MATCH,
          "absent history match falls back without exploration");
    o.lookup_available = 8; o.lookup_cap = 0;
    CHECK(janas_hybrid_choose(&p, &o).reason == JANAS_HYBRID_NO_CAPACITY,
          "reply/context clip can forbid lookup");
    o.lookup_cap = 8; o.lookup_available = 1; o.lookup_suffix = 8;
    CHECK(janas_hybrid_choose(&p, &o).depth == 1,
          "available continuation bounds shallow probing");
    o.lookup_available = UINT_MAX; o.lookup_cap = UINT_MAX;
    o.mtp_cap = UINT_MAX; o.probe_percent = 0;
    CHECK(janas_hybrid_choose(&p, &o).depth == 16,
          "oversized fallback cap remains bounded by global sixteen");
}

static void evidence_and_cost(void)
{
    struct janas_hybrid_controller p;
    struct janas_hybrid_offer o = offer(8, 0);
    janas_hybrid_init(&p, 1); mtp(&p);
    for (unsigned i = 0; i < 2; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 1.0);
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_MTP,
          "two successful lookup proposals do not replace established MTP");
    observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 1.0);
    struct janas_hybrid_decision d = janas_hybrid_choose(&p, &o);
    CHECK(d.source == JANAS_HYBRID_LOOKUP && d.depth == 8 &&
          d.lookup_evidence == 3 && d.mtp_evidence == 3 &&
          d.lookup_utility > d.mtp_utility * 1.05,
          "third compatible observation selects faster complete cycle");
    /* Strong acceptance does not justify expensive target verification. */
    janas_hybrid_init(&p, 1); mtp(&p);
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 3.0);
    d = janas_hybrid_choose(&p, &o);
    CHECK(d.source == JANAS_HYBRID_MTP && d.reason == JANAS_HYBRID_MTP_CHEAPER,
          "full-cycle cost defeats superficially perfect lookup acceptance");
    /* One hundred percent verification success can still be discarded work. */
    janas_hybrid_init(&p, 1); mtp(&p);
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 0, 8, 0.1);
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_MTP,
          "zero-delivery proposals retain cost and cannot appear beneficial");
    /* Small apparent improvement is rejected by hysteresis. */
    janas_hybrid_init(&p, 1); mtp(&p);
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 9.0 / 5.2);
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_MTP,
          "four percent predicted advantage does not replace MTP");
}

static void progressive_probes(void)
{
    struct janas_hybrid_controller p;
    struct janas_hybrid_offer o = offer(8, 100);
    janas_hybrid_init(&p, 3); mtp(&p);
    CHECK(janas_hybrid_choose(&p, &o).depth == 2,
          "new class starts at two, rather than probing eight");
    for (unsigned i = 0; i < 2; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 2, 2, 3, 8, 1.0);
    CHECK(janas_hybrid_choose(&p, &o).depth == 2,
          "two observations cannot expand the tested ceiling");
    observe(&p, JANAS_HYBRID_LOOKUP, 2, 1, 2, 8, 1.0);
    CHECK(janas_hybrid_choose(&p, &o).depth == 4,
          "three observations and two-thirds full acceptance permit doubling");
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 4, 4, 5, 8, 2.0);
    CHECK(janas_hybrid_choose(&p, &o).depth == 8,
          "next qualified prefix doubles to eight");
    o.lookup_cap = 7;
    CHECK(janas_hybrid_choose(&p, &o).depth == 7,
          "odd depth seven is eligible and respects configured ceiling");
    janas_hybrid_init(&p, 3); mtp(&p);
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 2, 1, 2, 8, 1.0);
    CHECK(janas_hybrid_choose(&p, &o).depth == 2,
          "partial prefixes cannot justify deeper exploratory work");
}

static void cooldown_and_isolation(void)
{
    struct janas_hybrid_controller p;
    struct janas_hybrid_offer o = offer(8, 100);
    janas_hybrid_init(&p, 4); mtp(&p);
    observe(&p, JANAS_HYBRID_LOOKUP, 2, 0, 1, 8, 1.0);
    for (unsigned i = 0; i < 32; i++) {
        struct janas_hybrid_decision d = janas_hybrid_choose(&p, &o);
        CHECK(d.source == JANAS_HYBRID_MTP && d.reason == JANAS_HYBRID_COOLDOWN,
              "first-draft rejection suspends class for thirty-two cycles");
    }
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_LOOKUP,
          "bounded shallow reevaluation resumes after cooldown");
    /* A long-match success must not make a short match look reliable. */
    janas_hybrid_init(&p, 4); mtp(&p);
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 1.0);
    o.probe_percent = 0; o.lookup_suffix = 1;
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_MTP,
          "suffix reliability classes do not leak acceptance evidence");
    o.lookup_suffix = 8; o.warm = 0;
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_MTP,
          "settling and warm acceptance/cost estimates are separate");
    o.warm = 1; o.vocab = 1;
    CHECK(janas_hybrid_choose(&p, &o).source == JANAS_HYBRID_MTP,
          "draft-vocabulary policy changes cannot reuse incompatible evidence");
    CHECK(janas_hybrid_suffix_class(1) == 0 &&
          janas_hybrid_suffix_class(2) == 1 &&
          janas_hybrid_suffix_class(3) == 1 &&
          janas_hybrid_suffix_class(4) == 2 &&
          janas_hybrid_suffix_class(8) == 2,
          "documented suffix classes match boundaries");
}

static void exploration_and_refresh(void)
{
    struct janas_hybrid_controller a, b;
    struct janas_hybrid_offer o = offer(8, 2);
    janas_hybrid_init(&a, 17); janas_hybrid_init(&b, 17);
    mtp(&a); mtp(&b);
    unsigned probes = 0;
    srand(555); int expected = rand(); srand(555);
    for (unsigned i = 0; i < 20000; i++) {
        struct janas_hybrid_decision x = janas_hybrid_choose(&a, &o);
        struct janas_hybrid_decision y = janas_hybrid_choose(&b, &o);
        CHECK(memcmp(&x, &y, sizeof(x)) == 0,
              "independent equal seeds reproduce selection and diagnostics");
        probes += x.probe;
        CHECK(!x.probe || x.depth == 2, "cold probes remain shallow");
    }
    CHECK(rand() == expected, "controller leaves ambient sampler RNG untouched");
    CHECK(probes > 300 && probes < 500,
          "two percent exploration stays near selected bounded frequency");
    for (unsigned i = 0; i < 3; i++)
        observe(&a, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 1.0);
    unsigned refreshed = 0, ordinary = 0;
    for (unsigned i = 0; i < 20000; i++) {
        struct janas_hybrid_decision x = janas_hybrid_choose(&a, &o);
        if (x.reason == JANAS_HYBRID_MTP_REFRESH) refreshed++;
        else if (x.source == JANAS_HYBRID_LOOKUP) ordinary++;
    }
    CHECK(refreshed > 300 && refreshed < 500 && ordinary > 19500,
          "dominant lookup shares its exploration allowance with MTP refresh");
}

static void prefix_cost_and_reuse(void)
{
    struct janas_hybrid_controller p;
    struct janas_hybrid_offer o = offer(8, 0);
    janas_hybrid_init(&p, 8); mtp(&p);
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 1.0);
    o.lookup_cap = 7;
    struct janas_hybrid_decision d = janas_hybrid_choose(&p, &o);
    CHECK(d.source == JANAS_HYBRID_LOOKUP && d.depth == 7 && d.evidence == 3,
          "verified wider chains also provide shorter-prefix evidence including seven");
    /* Train target width three under distinct PREVIOUS observed group classes.
       Current router output is never an input to this policy. */
    struct janas_hybrid_observation x = {
        JANAS_HYBRID_MTP, 2, 2, 3, 3, 0, 1, 0, 0, 1.0, 0.8
    };
    for (unsigned i = 0; i < 3; i++) janas_hybrid_observe(&p, &x);
    x.reuse = 1; x.seconds = 0.25; x.verification_seconds = 0.1;
    for (unsigned i = 0; i < 3; i++) janas_hybrid_observe(&p, &x);
    o.lookup_cap = 2; o.reuse = 0;
    double slower = janas_hybrid_choose(&p, &o).lookup_seconds;
    o.reuse = 1;
    double faster = janas_hybrid_choose(&p, &o).lookup_seconds;
    CHECK(faster < slower,
          "past expert-group class adjusts measured width costs with aggregate backoff");
}

static void invalid_and_recency(void)
{
    struct janas_hybrid_controller p, before;
    janas_hybrid_init(&p, 11); before = p;
    struct janas_hybrid_observation x = {
        JANAS_HYBRID_LOOKUP, 2, 2, 3, 3, 8, 1, 0, 2, 1.0, 0.8
    };
    x.seconds = NAN; janas_hybrid_observe(&p, &x);
    x.seconds = INFINITY; janas_hybrid_observe(&p, &x);
    x.seconds = 0; janas_hybrid_observe(&p, &x);
    x.seconds = 1; x.accepted_prefix = 3; janas_hybrid_observe(&p, &x);
    x.accepted_prefix = 2; x.depth = 17; janas_hybrid_observe(&p, &x);
    x.depth = 2; x.pass_width = 18; janas_hybrid_observe(&p, &x);
    x.pass_width = 3; x.vocab = 4; janas_hybrid_observe(&p, &x);
    CHECK(memcmp(&p, &before, sizeof(p)) == 0,
          "invalid measurements never poison learned state");
    struct janas_hybrid_offer o = offer(8, 100); o.vocab = 4;
    (void)janas_hybrid_choose(&p, &o);
    CHECK(memcmp(&p, &before, sizeof(p)) == 0,
          "invalid vocabulary leaves private clock and RNG unchanged");
    for (unsigned i = 0; i < 65; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 2, 2, 3, 8, 1.0);
    struct janas_hybrid_sample *s = &p.lookup[1][0][2][2];
    CHECK(s->count == 33 && s->seconds == 33 && s->delivered == 99 && s->full == 33,
          "recency decay bounds samples and preserves measured rates");
}

static void abandoned_preparation(void)
{
    struct janas_hybrid_controller p;
    struct janas_hybrid_offer o = offer(8, 0);
    janas_hybrid_init(&p, 18); mtp(&p);
    for (unsigned i = 0; i < 3; i++)
        observe(&p, JANAS_HYBRID_LOOKUP, 8, 8, 9, 8, 1.0);
    struct janas_hybrid_observation x = {
        JANAS_HYBRID_LOOKUP, 8, 0, 0, 0, 8, 1, 0, 2, 100.0, 0
    };
    double evidence = p.lookup[1][0][2][8].count;
    double full = p.lookup[1][0][2][8].full;
    janas_hybrid_observe(&p, &x);
    CHECK(p.lookup[1][0][2][8].count == evidence &&
          p.lookup[1][0][2][8].full == full &&
          !p.cooldown_until[1][0][2] &&
          p.lookup[1][0][2][8].abandoned_seconds == 100.0,
          "abandoned preparation charges cost without inventing acceptance evidence or rejection");
    struct janas_hybrid_decision d = janas_hybrid_choose(&p, &o);
    CHECK(d.depth < 8 || d.source == JANAS_HYBRID_MTP,
          "abandoned preparation cannot be omitted from useful-output utility");
    x.source = JANAS_HYBRID_MTP; x.lookup_suffix = 0; x.depth = 4;
    janas_hybrid_observe(&p, &x);
    CHECK(p.mtp[1][0].count == 3 && p.mtp[1][0].abandoned_seconds == 100.0,
          "MTP abandonment preserves measured policy count and adds exact wasted cost");
}

int main(void)
{
    cold_and_caps(); evidence_and_cost(); progressive_probes();
    cooldown_and_isolation(); exploration_and_refresh();
    prefix_cost_and_reuse(); invalid_and_recency(); abandoned_preparation();
    printf("test_hybrid_controller: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
