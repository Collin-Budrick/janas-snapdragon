/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spec_json.h"

void janas_spec_metrics_json(struct janas_buf *b,
                              const struct janas_llm_speculation_metrics *m)
{
    janas_buf_puts(b, "{\"schema\":1,\"cycle_scope\":\"active_decode_with_incoming_preparation\","
        "\"source_order\":[\"plain\",\"lookup\",\"mtp\",\"draft\"],"
        "\"expert_scope\":\"whole_chat_sum_per_layer_per_pass_main_plus_mtp_shared_included\","
        "\"gdn_scope\":\"whole_chat_main_plus_mtp\","
        "\"cpu_scope\":\"whole_chat_main_plus_mtp_synchronous_cpu_groups\","
        "\"cpu_row_scope\":\"logical_matrix_rows;row_vectors_equals_rows_times_nv\","
        "\"cpu_panel_scope\":\"eligible_allocated_views_are_not_execution;prepared_rows_prove_dispatch\","
        "\"cpu_established_scope\":\"existing_kernels_including_legacy_q6_i8mm\","
        "\"cpu_panel_bytes_scope\":\"sum_allocated_workspace_including_metadata_per_group\","
        "\"cpu_panel_peak_bytes_scope\":\"maximum_since_model_load_gauge\","
        "\"cpu_panel_prepare_seconds_scope\":\"allocation_plus_packing_wall_time\","
        "\"timing_relationship\":\"preparation_contains_controller_repack_head_chain;"
        "recovery_is_nested_in_target_verification;pending_and_buffered_are_gauges\"");
#define U(f) janas_buf_printf(b, ",\"" #f "\":%llu", (unsigned long long)m->f)
#define F(f) janas_buf_printf(b, ",\"" #f "\":%.9g", m->f)
    U(cycles); U(proposals); U(verification_rows); U(accepted_prefix);
    U(produced_tokens); U(delivered_tokens); U(discarded_tokens);
    U(rejected_rows); U(lookup_hits); U(lookup_misses);
    U(filter_guard_cycles); U(penalty_guard_cycles); U(buffered_tokens);
    U(expert_assignments); U(expert_groups);
    U(gdn_chain_tokens); U(gdn_journal_replay_tokens);
    U(gdn_scalar_replay_tokens); U(gdn_konly_replay_tokens); U(gdn_delta_bytes);
    U(head_repacks); U(head_vocab_fallbacks); U(head_vocab_epoch);
    U(head_vocab_policy); U(head_vocab_rows);
    U(cpu_groups); U(cpu_tasks);
    U(cpu_prepared_rows); U(cpu_prepared_row_vectors);
    U(cpu_established_rows); U(cpu_established_row_vectors);
    U(cpu_panel_views); U(cpu_panel_unique); U(cpu_panel_dedup_hits);
    U(cpu_panel_bytes); U(cpu_panel_peak_bytes);
    F(cycle_seconds); F(lookup_seconds); F(mtp_maintenance_seconds);
    F(mtp_head_seconds); F(mtp_chain_seconds); F(mtp_repack_seconds); F(draft_seconds);
    F(verification_seconds); F(sample_seconds); F(controller_seconds);
    F(abandoned_preparation_seconds); F(pending_preparation_seconds);
    F(gdn_core_seconds); F(gdn_conv_seconds);
    F(cpu_panel_prepare_seconds);
#undef U
#undef F
#define A(f, n) do { janas_buf_puts(b, ",\"" #f "\":["); \
    for (unsigned i = 0; i < (n); i++) \
        janas_buf_printf(b, "%s%llu", i ? "," : "", (unsigned long long)m->f[i]); \
    janas_buf_puts(b, "]"); } while (0)
    A(source_cycles, 4); A(source_drafted, 4); A(source_accepted, 4);
    A(source_delivered, 4); A(source_discarded, 4);
    A(pass_width_count, 18); A(prefix_count, 17); A(lookup_suffix_count, 9);
    A(expert_group_widths, 18);
#undef A
    janas_buf_puts(b, ",\"source_seconds\":[");
    for (unsigned i = 0; i < 4; i++) janas_buf_printf(b, "%s%.9g", i ? "," : "", m->source_seconds[i]);
    janas_buf_puts(b, "],\"last\":{");
#define L(f) janas_buf_printf(b, "\"" #f "\":%llu,", (unsigned long long)m->last.f)
    L(proposal_id); L(source); L(planner_source); L(requested_depth); L(pass_width);
    L(accepted_prefix); L(produced); L(delivered); L(buffered); L(discarded);
    L(rejected_rows); L(draft_vocab_policy); L(draft_vocab_rows); L(warm);
    L(guard_reason); L(lookup_suffix); L(resolved_planner);
#undef L
    janas_buf_printf(b, "\"preparation_seconds\":%.9g,\"verification_seconds\":%.9g,"
        "\"sample_seconds\":%.9g,\"maintenance_seconds\":%.9g,"
        "\"controller_seconds\":%.9g,\"total_seconds\":%.9g}}",
        m->last.preparation_seconds, m->last.verification_seconds,
        m->last.sample_seconds, m->last.maintenance_seconds,
        m->last.controller_seconds, m->last.total_seconds);
}

void janas_hybrid_metrics_json(struct janas_buf *b,
                                const struct janas_llm_hybrid_metrics *m)
{
    janas_buf_puts(b, "{\"schema\":1,\"scope\":\"current_reply\","
        "\"reason_order\":[\"no_match\",\"no_evidence\",\"mtp_cheaper\",\"lookup_faster\","
        "\"lookup_probe\",\"mtp_refresh\",\"cooldown\",\"no_capacity\"],"
        "\"cost_scope\":\"causal_full_cycles_recovery_reassigned_once\",\"recovery_attribution\":\"fused_gdn_row_share_estimate\"");
#define U(f) janas_buf_printf(b, ",\"" #f "\":%llu", (unsigned long long)m->f)
#define F(f) janas_buf_printf(b, ",\"" #f "\":%.9g", m->f)
    U(decisions); U(offered_matches); U(eligible_matches);
    U(lookup_selected); U(mtp_selected); U(probes); U(cooldown_skips);
    U(lookup_drafted); U(lookup_accepted); U(lookup_delivered); U(mtp_delivered);
    F(lookup_seconds); F(mtp_seconds); F(recovery_reassigned_seconds);
#undef U
#undef F
#define A(f, n) do { janas_buf_puts(b, ",\"" #f "\":["); \
    for (unsigned i = 0; i < (n); i++) \
        janas_buf_printf(b, "%s%llu", i ? "," : "", (unsigned long long)m->f[i]); \
    janas_buf_puts(b, "]"); } while (0)
    A(reason_count, 12); A(lookup_depth_count, 17);
#undef A
    janas_buf_puts(b, ",\"last\":{");
#define L(f) janas_buf_printf(b, "\"" #f "\":%llu,", (unsigned long long)m->last.f)
    L(proposal_id); L(source); L(requested_depth); L(lookup_available); L(lookup_suffix);
    L(suffix_class); L(reason); L(probe); L(warm); L(vocab_key);
    L(lookup_evidence); L(mtp_evidence); L(accepted_prefix); L(delivered); L(pass_width);
#undef L
    janas_buf_printf(b, "\"lookup_utility\":%.9g,\"mtp_utility\":%.9g,"
        "\"predicted_seconds\":%.9g,\"causal_seconds\":%.9g,\"recovery_seconds\":%.9g}}",
        m->last.lookup_utility, m->last.mtp_utility, m->last.predicted_seconds,
        m->last.causal_seconds, m->last.recovery_seconds);
}
