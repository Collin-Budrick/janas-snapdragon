/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Cycle-planner policy fixtures; no model, target sampler or inference. */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/spec_controller.h"

static int failures;
#define CHECK(c, text) do { if (!(c)) { \
    printf("FAIL: %s (line %d)\n", text, __LINE__); failures++; \
} } while (0)

static void train(struct janas_cycle_controller *p, unsigned source,
                  unsigned warm, unsigned vocab, unsigned cap,
                  unsigned winning_depth)
{
    for (unsigned k = 0; k <= cap; k++)
        for (unsigned i = 0; i < 3; i++)
            janas_cycle_observe(p, source, warm, vocab, k, 1.0,
                                k == winning_depth ? 200 : 100);
}

static void initial_sweep(void)
{
    struct janas_cycle_controller p;
    for (unsigned cap = 0; cap <= JANAS_CYCLE_MAX_K; cap++) {
        janas_cycle_init(&p, 123);
        uint64_t rng = p.rng;
        for (unsigned round = 0; round < 3; round++) {
            for (unsigned depth = 0; depth <= cap; depth++) {
                unsigned k = janas_cycle_choose(&p, 1, 1, 0, cap, 100);
                CHECK(k == depth, "initial sweep balances every eligible depth");
                CHECK(p.rng == rng, "initial sweep consumes no exploration RNG");
                janas_cycle_observe(&p, 1, 1, 0, k, 1.0, k + 1);
            }
        }
        for (unsigned depth = 0; depth <= cap; depth++)
            CHECK(p.bucket[1][1][0].depth[depth].count == 3,
                  "three observations per depth before exploitation");
        CHECK(janas_cycle_choose(&p, 1, 1, 0, cap, 0) == cap,
              "complete sweep chooses empirically fastest depth");
    }
    janas_cycle_init(&p, 5);
    for (int i = 0; i < 3; i++)
        janas_cycle_observe(&p, 1, 0, 0, 0, 1.0, 100);
    for (int i = 0; i < 2; i++)
        janas_cycle_observe(&p, 1, 0, 0, 1, 1.0, 1000);
    uint64_t rng = p.rng;
    CHECK(janas_cycle_choose(&p, 1, 0, 0, 1, 20) == 1 && p.rng == rng,
          "two observations remain an initial probe, not exploitation");
    janas_cycle_observe(&p, 1, 0, 0, 1, 1.0, 1000);
    CHECK(janas_cycle_choose(&p, 1, 0, 0, 1, 0) == 1 && p.rng != rng,
          "third observation enables empirical selection");
}

static void hysteresis(void)
{
    struct janas_cycle_controller p;
    janas_cycle_init(&p, 1);
    for (int i = 0; i < 3; i++) {
        janas_cycle_observe(&p, 1, 1, 0, 0, 1.0, 100);
        janas_cycle_observe(&p, 1, 1, 0, 1, 1.0, 104);
    }
    CHECK(janas_cycle_choose(&p, 1, 1, 0, 1, 0) == 0,
          "four percent gain retains incumbent");
    janas_cycle_init(&p, 1);
    for (int i = 0; i < 3; i++) {
        janas_cycle_observe(&p, 1, 1, 0, 0, 1.0, 100);
        janas_cycle_observe(&p, 1, 1, 0, 1, 1.0, 105);
    }
    CHECK(janas_cycle_choose(&p, 1, 1, 0, 1, 0) == 0,
          "exact five percent does not exceed hysteresis");
    janas_cycle_init(&p, 1);
    for (int i = 0; i < 3; i++) {
        janas_cycle_observe(&p, 1, 1, 0, 0, 1.0, 100);
        janas_cycle_observe(&p, 1, 1, 0, 1, 1.0, 106);
    }
    CHECK(janas_cycle_choose(&p, 1, 1, 0, 1, 0) == 1 &&
          p.bucket[1][1][0].incumbent == 1,
          "six percent gain changes incumbent");
    /* A tied depth must not cause arbitrary switching. */
    p.bucket[1][1][0].depth[0] = p.bucket[1][1][0].depth[1];
    CHECK(janas_cycle_choose(&p, 1, 1, 0, 1, 0) == 1,
          "equal measured rate keeps incumbent");
}

static void exploration(void)
{
    struct janas_cycle_controller a, b, noise;
    janas_cycle_init(&a, 0);
    janas_cycle_init(&b, 0);
    janas_cycle_init(&noise, 42);
    train(&a, 2, 1, 2, 8, 0);
    train(&b, 2, 1, 2, 8, 0);
    train(&noise, 2, 1, 2, 8, 8);
    unsigned seen[9] = {0}, non_incumbent = 0;
    srand(8912);
    int expected_rand = rand();
    srand(8912);
    for (int i = 0; i < 10000; i++) {
        unsigned k = janas_cycle_choose(&a, 2, 1, 2, 8, 20);
        (void)janas_cycle_choose(&noise, 2, 1, 2, 8, 100);
        CHECK(k == janas_cycle_choose(&b, 2, 1, 2, 8, 20),
              "fixed seed exploration is deterministic and per-controller");
        CHECK(k <= 8, "exploration respects eligible cap");
        if (k <= 8)
            seen[k]++;
        if (k)
            non_incumbent++;
    }
    CHECK(rand() == expected_rand, "controller never consumes ambient RNG");
    CHECK(non_incumbent > 1400 && non_incumbent < 2200,
          "twenty percent exploration yields bounded expected probe coverage");
    for (unsigned k = 1; k <= 8; k++)
        CHECK(seen[k] > 100, "twenty percent exploration probes every action");
    for (int i = 0; i < 100; i++)
        CHECK(janas_cycle_choose(&a, 2, 1, 2, 8, 0) == 0,
              "zero exploration retains measured winner");
    /* Clamp oversized percentages exactly like 100 percent. */
    janas_cycle_init(&a, 18);
    janas_cycle_init(&b, 18);
    train(&a, 2, 0, 3, 4, 0);
    train(&b, 2, 0, 3, 4, 0);
    for (int i = 0; i < 100; i++)
        CHECK(janas_cycle_choose(&a, 2, 0, 3, 4, UINT_MAX) ==
              janas_cycle_choose(&b, 2, 0, 3, 4, 100),
              "exploration percentage clamps to one hundred");
}

static void isolation_and_caps(void)
{
    struct janas_cycle_controller p;
    janas_cycle_init(&p, 7);
    train(&p, 1, 1, 1, 8, 8);
    CHECK(janas_cycle_choose(&p, 1, 1, 1, 8, 0) == 8,
          "trained bucket chooses winner");
    CHECK(janas_cycle_choose(&p, 2, 1, 1, 8, 0) == 0 &&
          janas_cycle_choose(&p, 1, 0, 1, 8, 0) == 0 &&
          janas_cycle_choose(&p, 1, 1, 2, 8, 0) == 0,
          "source, warmness and vocabulary observations are isolated");
    CHECK(janas_cycle_choose(&p, 1, 99, 1, 8, 0) == 8,
          "nonzero warmness selects warm bucket");
    for (unsigned cap = 0; cap < 8; cap++)
        CHECK(janas_cycle_choose(&p, 1, 1, 1, cap, 100) <= cap,
              "short context/reply cap excludes overlong actions");
    janas_cycle_init(&p, 3);
    train(&p, 3, 0, 0, JANAS_CYCLE_MAX_K, JANAS_CYCLE_MAX_K);
    CHECK(janas_cycle_choose(&p, 3, 0, 0, UINT_MAX, 0) == JANAS_CYCLE_MAX_K,
          "oversized depth cap clamps to maximum sixteen");
    CHECK(janas_cycle_choose(&p, 3, 0, 0, 0, 100) == 0,
          "zero remaining draft capacity selects plain target pass");
    CHECK(janas_cycle_vocab_key(0, 32768) == 0 &&
          janas_cycle_vocab_key(1, 8192) == 1 &&
          janas_cycle_vocab_key(1, 16384) == 2 &&
          janas_cycle_vocab_key(1, 32768) == 3,
          "legacy and resolved adaptive budgets have separate keys");
}

static void observations(void)
{
    struct janas_cycle_controller p, before;
    janas_cycle_init(&p, 8);
    before = p;
    janas_cycle_observe(&p, JANAS_CYCLE_SOURCES, 0, 0, 0, 1.0, 1);
    janas_cycle_observe(&p, 0, 0, JANAS_CYCLE_VOCABS, 0, 1.0, 1);
    janas_cycle_observe(&p, 0, 0, 0, JANAS_CYCLE_MAX_K + 1, 1.0, 1);
    janas_cycle_observe(&p, 0, 0, 0, 0, 0.0, 1);
    janas_cycle_observe(&p, 0, 0, 0, 0, -1.0, 1);
    janas_cycle_observe(&p, 0, 0, 0, 0, NAN, 1);
    janas_cycle_observe(&p, 0, 0, 0, 0, INFINITY, 1);
    CHECK(memcmp(&p, &before, sizeof(p)) == 0,
          "invalid observations do not mutate learned state");
    CHECK(janas_cycle_choose(&p, JANAS_CYCLE_SOURCES, 0, 0, 16, 20) == 0 &&
          janas_cycle_choose(&p, 0, 0, JANAS_CYCLE_VOCABS, 16, 20) == 0 &&
          memcmp(&p, &before, sizeof(p)) == 0,
          "invalid buckets do not mutate policy or RNG");
    for (int i = 0; i < 3; i++) {
        janas_cycle_observe(&p, 1, 0, 0, 0, 1.0, 100);
        janas_cycle_observe(&p, 1, 0, 0, 1, 1.0, 0);
    }
    CHECK(p.bucket[1][0][0].depth[1].count == 3 &&
          p.bucket[1][0][0].depth[1].seconds == 3.0 &&
          janas_cycle_choose(&p, 1, 0, 0, 1, 0) == 0,
          "zero-delivery cycles retain their full wasted cost");
    for (int i = 0; i < 65; i++)
        janas_cycle_observe(&p, 2, 0, 0, 4, 1.0, 7);
    struct janas_cycle_observation *o = &p.bucket[2][0][0].depth[4];
    CHECK(o->count == 33 && o->seconds == 33.0 && o->delivered == 231.0,
          "bounded recency preserves measured rate during decay");
}

static void reuse_contexts(void)
{
    struct janas_cycle_controller p;
    janas_cycle_init(&p, 45);
    /* Identical aggregate workload, opposite reuse-conditioned winners. */
    for (unsigned k = 0; k <= 4; k++) {
        for (int i = 0; i < 3; i++) {
            janas_cycle_observe_reuse(&p, 1, 1, 2, k, 1.0,
                                      k == 1 ? 300 : 100, 0);
            janas_cycle_observe_reuse(&p, 1, 1, 2, k, 1.0,
                                      k == 4 ? 300 : 100, 1);
        }
        CHECK(p.bucket[1][1][2].depth[k].count == 6 &&
              p.reuse_bucket[1][1][2][0].depth[k].count == 3 &&
              p.reuse_bucket[1][1][2][1].depth[k].count == 3,
              "reuse observation teaches aggregate and selected context once");
    }
    CHECK(janas_cycle_choose_reuse(&p, 1, 1, 2, 4, 0, 0) == 1 &&
          janas_cycle_choose_reuse(&p, 1, 1, 2, 4, 0, 1) == 4,
          "past expert reuse selects opposite measured optimal depths");
    CHECK(janas_cycle_choose_reuse(&p, 1, 1, 2, 4, 0, 2) ==
          janas_cycle_choose(&p, 1, 1, 2, 4, 0),
          "unknown reuse retains aggregate policy");
    janas_cycle_init(&p, 46);
    for (unsigned k = 0; k <= 4; k++)
        for (int i = 0; i < 10; i++)
            janas_cycle_observe(&p, 1, 1, 0, k, 1.0, k == 4 ? 200 : 100);
    for (int i = 0; i < 2; i++)
        janas_cycle_observe_reuse(&p, 1, 1, 0, 1, 1.0, 300, 0);
    CHECK(janas_cycle_choose_reuse(&p, 1, 1, 0, 4, 0, 0) == 4,
          "two context observations back off to mature aggregate estimate");
    janas_cycle_observe_reuse(&p, 1, 1, 0, 1, 1.0, 300, 0);
    CHECK(janas_cycle_choose_reuse(&p, 1, 1, 0, 4, 0, 0) == 1 &&
          janas_cycle_choose(&p, 1, 1, 0, 4, 0) == 4,
          "third context observation may improve on its aggregate backoff");
    struct janas_cycle_bucket before[2];
    memcpy(before, p.reuse_bucket[1][1][0], sizeof(before));
    janas_cycle_observe_reuse(&p, 1, 1, 0, 2, 1.0, 100, UINT_MAX);
    CHECK(memcmp(before, p.reuse_bucket[1][1][0], sizeof(before)) == 0 &&
          p.bucket[1][1][0].depth[2].count == 11,
          "unknown context adds aggregate observation only");
    /* A fresh context must preserve the same balanced initial sweep. */
    janas_cycle_init(&p, 47);
    for (unsigned round = 0; round < 3; round++)
        for (unsigned k = 0; k <= 4; k++) {
            unsigned got = janas_cycle_choose_reuse(&p, 2, 0, 3, 4, 20, 1);
            CHECK(got == k, "reuse policy preserves balanced initial sweep");
            janas_cycle_observe_reuse(&p, 2, 0, 3, got, 1.0, got + 1, 1);
        }
    struct janas_cycle_controller twin = p;
    srand(124);
    int expected = rand();
    srand(124);
    for (unsigned i = 0; i < 1000; i++)
        CHECK(janas_cycle_choose_reuse(&p, 2, 0, 3, 4, 20, 1) ==
              janas_cycle_choose_reuse(&twin, 2, 0, 3, 4, 20, 1),
              "reuse probes retain deterministic independent exploration stream");
    CHECK(rand() == expected, "reuse policy leaves ambient RNG untouched");
}

int main(void)
{
    initial_sweep();
    hysteresis();
    exploration();
    isolation_and_caps();
    observations();
    reuse_contexts();
    printf("test_spec_controller: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
