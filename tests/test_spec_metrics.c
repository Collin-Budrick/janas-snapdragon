/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Versioned metrics getter: repeatable bounded prefixes and sample deltas.
   No model, session inference or production archive is used by this fixture. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/api_spec.c"

static struct janas_gen_metrics generation;
static struct janas_llm_runtime_stats runtime;
static struct janas_mtp_costs head;

int32_t janas_api_fail(int32_t code, const char *fmt, ...)
{ (void)fmt; return code; }
void janas_llm_session_metrics(const struct janas_llm_session *s,
                                struct janas_gen_metrics *out)
{ (void)s; *out = generation; }
void janas_llm_model_runtime_stats(const struct janas_llm_model *m,
                                   struct janas_llm_runtime_stats *out)
{ (void)m; *out = runtime; }
void janas_llm_mtp_costs(const struct janas_llm_model *m,
                         struct janas_mtp_costs *out)
{ (void)m; *out = head; }

static int failures;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static void prefixes(janas_llm_chat *c)
{
    enum { GUARD = 32 };
    for (size_t n = 2 * sizeof(uint32_t);
         n <= sizeof(struct janas_llm_speculation_metrics); n++) {
        unsigned char *p = malloc(n + GUARD);
        CHECK(p);
        if (!p) return;
        memset(p, 0xa5, n + GUARD);
        struct janas_llm_speculation_metrics *out = (void *)p;
        out->size = (uint32_t)n;
        out->version = JANAS_LLM_SPEC_METRICS_VERSION;
        for (int call = 0; call < 2; call++) {
            CHECK(janas_llm_chat_speculation_metrics(c, out) == JANAS_LLM_OK);
            CHECK(out->size == n); /* safe to reuse WITHOUT resetting size */
            CHECK(out->version == JANAS_LLM_SPEC_METRICS_VERSION);
            for (size_t i = n; i < n + GUARD; i++) CHECK(p[i] == 0xa5);
            /* A broken first call has changed the capacity; fail safely
               before permitting its second write beyond this allocation. */
            if (out->size != n) break;
        }
        free(p);
    }
}

static void invalid(janas_llm_chat *c, uint32_t size, uint32_t version)
{
    struct janas_llm_speculation_metrics out, before;
    memset(&out, 0xa5, sizeof out);
    out.size = size, out.version = version;
    memcpy(&before, &out, sizeof out);
    CHECK(janas_llm_chat_speculation_metrics(c, &out) == JANAS_LLM_EINVAL);
    CHECK(!memcmp(&out, &before, sizeof out));
}

int main(void)
{
    janas_llm llm = {0};
    janas_llm_chat c = {.llm = &llm,
        .s = (struct janas_llm_session *)(void *)&generation};
    generation.cycles = 10, c.gm0.cycles = 3;
    generation.produced_tokens = 20, c.gm0.produced_tokens = 4;
    generation.buffered_tokens = 2;
    generation.pending_preparation_seconds = 0.125;
    generation.last.resolved_planner = JANAS_LLM_SPEC_PLANNER_CYCLE;
    generation.source_cycles[2] = 8, c.gm0.source_cycles[2] = 2;
    runtime.expert_groups = 30, c.rt0.expert_groups = 10;
    runtime.gdn_delta_bytes = 1024;
    runtime.gdn_core_seconds = 1.25, c.rt0.gdn_core_seconds = 0.25;
    head.repacks = 9, c.mt0.repacks = 4;
    head.vocab_epoch = 17, head.vocab_policy = 1, head.vocab_rows = 16384;
    struct {
        unsigned char before[16];
        struct janas_llm_speculation_metrics metrics;
        unsigned char after[16];
    } guarded;
    memset(&guarded, 0xa5, sizeof guarded);
    guarded.metrics.size = sizeof guarded.metrics;
    guarded.metrics.version = JANAS_LLM_SPEC_METRICS_VERSION;
    CHECK(janas_llm_chat_speculation_metrics(&c, &guarded.metrics) == JANAS_LLM_OK);
    CHECK(guarded.metrics.size == sizeof guarded.metrics);
    CHECK(guarded.metrics.cycles == 7 && guarded.metrics.produced_tokens == 16);
    CHECK(guarded.metrics.source_cycles[2] == 6);
    CHECK(guarded.metrics.expert_groups == 20);
    CHECK(guarded.metrics.gdn_core_seconds == 1.0);
    CHECK(guarded.metrics.head_repacks == 5);
    CHECK(guarded.metrics.buffered_tokens == 2);
    CHECK(guarded.metrics.pending_preparation_seconds == 0.125);
    CHECK(guarded.metrics.gdn_delta_bytes == 1024);
    CHECK(guarded.metrics.head_vocab_epoch == 17);
    CHECK(guarded.metrics.last.resolved_planner == JANAS_LLM_SPEC_PLANNER_CYCLE);
    for (size_t i = 0; i < 16; i++)
        CHECK(guarded.before[i] == 0xa5 && guarded.after[i] == 0xa5);
    prefixes(&c);
    invalid(&c, 0, JANAS_LLM_SPEC_METRICS_VERSION);
    invalid(&c, 4, JANAS_LLM_SPEC_METRICS_VERSION);
    invalid(&c, sizeof guarded.metrics + 1, JANAS_LLM_SPEC_METRICS_VERSION);
    invalid(&c, sizeof guarded.metrics, 0);
    invalid(NULL, sizeof guarded.metrics, JANAS_LLM_SPEC_METRICS_VERSION);
    CHECK(janas_llm_chat_speculation_metrics(&c, NULL) == JANAS_LLM_EINVAL);
    c.s = NULL;
    invalid(&c, sizeof guarded.metrics, JANAS_LLM_SPEC_METRICS_VERSION);
    if (failures) return 1;
    puts("test_spec_metrics: full layout/all prefixes twice/guards/deltas/gauges/invalid ok");
    return 0;
}
