/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <stddef.h>
#include <string.h>
#include "chat.h"

_Static_assert(offsetof(struct janas_llm_speculation_metrics, expert_assignments) -
               offsetof(struct janas_llm_speculation_metrics, cycles) ==
               sizeof(struct janas_gen_metrics), "public generation metrics layout");

int32_t janas_llm_chat_speculation_metrics(
    const janas_llm_chat *c, struct janas_llm_speculation_metrics *out)
{
    if (!c || !c->s || !out || out->size < 2 * sizeof(uint32_t) ||
        out->size > sizeof(*out) || out->version != JANAS_LLM_SPEC_METRICS_VERSION)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid speculation metrics layout");
    uint32_t requested = out->size;
    struct janas_gen_metrics g;
    janas_llm_session_metrics(c->s, &g);
    struct janas_llm_speculation_metrics v = {0};
    v.size = requested; v.version = JANAS_LLM_SPEC_METRICS_VERSION;
    memcpy(&v.cycles, &g, sizeof(g));
#define D(f) v.f = g.f >= c->gm0.f ? g.f - c->gm0.f : g.f
    D(cycles); D(proposals); D(verification_rows); D(accepted_prefix);
    D(produced_tokens); D(delivered_tokens); D(discarded_tokens);
    D(rejected_rows); D(lookup_hits); D(lookup_misses);
    D(filter_guard_cycles); D(penalty_guard_cycles);
    D(cycle_seconds); D(lookup_seconds); D(mtp_maintenance_seconds);
    D(mtp_head_seconds); D(mtp_chain_seconds); D(mtp_repack_seconds);
    D(draft_seconds); D(verification_seconds); D(sample_seconds);
    D(controller_seconds); D(abandoned_preparation_seconds);
#undef D
#define A(f, n) for (unsigned i = 0; i < (n); i++) \
    v.f[i] = g.f[i] >= c->gm0.f[i] ? g.f[i] - c->gm0.f[i] : g.f[i]
    A(source_cycles, 4); A(source_drafted, 4); A(source_accepted, 4);
    A(source_delivered, 4); A(source_discarded, 4); A(source_seconds, 4);
    A(pass_width_count, 18); A(prefix_count, 17); A(lookup_suffix_count, 9);
#undef A
    /* Pending preparation and buffered tokens are current gauges. */
    if (!v.cycles) memset(&v.last, 0, sizeof(v.last));
    struct janas_llm_runtime_stats rt;
    struct janas_mtp_costs mt;
    janas_llm_model_runtime_stats(c->llm->m, &rt);
    janas_llm_mtp_costs(c->llm->m, &mt);
#define R(f) v.f = rt.f >= c->rt0.f ? rt.f - c->rt0.f : rt.f
    R(expert_assignments); R(expert_groups);
    R(gdn_chain_tokens); R(gdn_journal_replay_tokens);
    R(gdn_scalar_replay_tokens); R(gdn_konly_replay_tokens);
    R(gdn_core_seconds); R(gdn_conv_seconds);
    R(cpu_groups); R(cpu_tasks);
    R(cpu_prepared_rows); R(cpu_prepared_row_vectors);
    R(cpu_established_rows); R(cpu_established_row_vectors);
    R(cpu_panel_views); R(cpu_panel_unique); R(cpu_panel_dedup_hits);
    R(cpu_panel_bytes); R(cpu_panel_prepare_seconds);
#undef R
    for (unsigned i = 0; i < 18; i++)
        v.expert_group_widths[i] = rt.expert_group_widths[i] - c->rt0.expert_group_widths[i];
    v.gdn_delta_bytes = rt.gdn_delta_bytes;
    v.cpu_panel_peak_bytes = rt.cpu_panel_peak_bytes;
    v.head_repacks = mt.repacks - c->mt0.repacks;
    v.head_vocab_fallbacks = mt.vocab_fallbacks - c->mt0.vocab_fallbacks;
    v.head_vocab_epoch = mt.vocab_epoch;
    v.head_vocab_policy = mt.vocab_policy; v.head_vocab_rows = mt.vocab_rows;
    memcpy(out, &v, requested);
    return JANAS_LLM_OK;
}

_Static_assert(sizeof(struct janas_gen_hybrid_metrics) ==
               sizeof(struct janas_llm_hybrid_metrics) - 2 * sizeof(uint32_t),
               "public hybrid metrics layout");
int32_t janas_llm_chat_hybrid_metrics(const janas_llm_chat *c,
                                      struct janas_llm_hybrid_metrics *out)
{
    if (!c || !c->s || !out || out->size < 2 * sizeof(uint32_t) ||
        out->size > sizeof(*out) || out->version != JANAS_LLM_HYBRID_METRICS_VERSION)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid hybrid metrics layout");
    uint32_t requested = out->size;
    struct janas_gen_hybrid_metrics g;
    janas_llm_session_hybrid_metrics(c->s, &g);
    struct janas_llm_hybrid_metrics v = {0};
    v.size = requested; v.version = JANAS_LLM_HYBRID_METRICS_VERSION;
    memcpy(&v.decisions, &g, sizeof(g));
#define D(f) v.f = g.f >= c->hm0.f ? g.f - c->hm0.f : g.f
    D(decisions); D(offered_matches); D(eligible_matches);
    D(lookup_selected); D(mtp_selected); D(probes); D(cooldown_skips);
    D(lookup_drafted); D(lookup_accepted); D(lookup_delivered); D(mtp_delivered);
    D(lookup_seconds); D(mtp_seconds); D(recovery_reassigned_seconds);
#undef D
    for (unsigned i = 0; i < 12; i++)
        v.reason_count[i] = g.reason_count[i] >= c->hm0.reason_count[i] ?
            g.reason_count[i] - c->hm0.reason_count[i] : g.reason_count[i];
    for (unsigned i = 0; i < 17; i++)
        v.lookup_depth_count[i] = g.lookup_depth_count[i] >= c->hm0.lookup_depth_count[i] ?
            g.lookup_depth_count[i] - c->hm0.lookup_depth_count[i] : g.lookup_depth_count[i];
    if (!v.decisions) memset(&v.last, 0, sizeof(v.last));
    memcpy(out, &v, requested);
    return JANAS_LLM_OK;
}

_Static_assert(sizeof(struct janas_llm_s2_metrics) -
               offsetof(struct janas_llm_s2_metrics, cycles) ==
               sizeof(struct janas_gen_s2_metrics), "public S2 metrics mirror layout");
int32_t janas_llm_chat_s2_metrics(const janas_llm_chat *c,
                                  struct janas_llm_s2_metrics *out)
{
    if (!c || !c->s || !out || out->size < 2 * sizeof(uint32_t) ||
        out->size > sizeof(*out) || out->version != JANAS_LLM_S2_METRICS_VERSION)
        return janas_api_fail(JANAS_LLM_EINVAL, "invalid S2 metrics layout");
    uint32_t requested = out->size;
    struct janas_gen_s2_metrics g;
    janas_llm_session_s2_metrics(c->s, &g);
    struct janas_llm_s2_metrics v = {0};
    v.size = requested; v.version = JANAS_LLM_S2_METRICS_VERSION;
    memcpy(&v.cycles, &g, sizeof(g));
#define D(f) v.f = g.f >= c->s2m0.f ? g.f - c->s2m0.f : g.f
    D(cycles);
    D(primary_proposals);
    D(root_offered);
    D(rank2_matches);
    D(alternate_builds);
    D(alternate_nodes);
    D(cached_uses);
    D(cached_nodes);
    D(discarded_nodes);
    D(pruned);
    D(probes);
    D(maintenance_fast_calls);
    D(cached_accepted);
    D(alternate_seconds);
    D(cached_cycle_seconds);
    D(controller_seconds);
#undef D
    memcpy(out, &v, requested);
    return JANAS_LLM_OK;
}
