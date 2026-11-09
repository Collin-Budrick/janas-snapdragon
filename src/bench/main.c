/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * janas-bench - measures what this machine does with a model and reports
 * it: prefill and decoding speed for every configuration the engine can
 * use (thread counts, GPU), on the same text, and the choices the engine
 * makes from them. The measurements also fill the engine's tuning profile
 * (~/.cache/janas/), so the next start is tuned already.
 *
 * Usage: janas-bench [<model>] [--models <dir>] [--mtp <file> | --no-mtp]
 *                    [--prompt <tokens>] [--gen <tokens>] [--ctx <tokens>]
 *                    [--cache <GiB>] [--reserve <GiB>]
 *                    [--rounds <n>] (default 3, alternating the order)
 *                    [--no-npu] (without the NPU, as JANAS_NPU=0;
 *                    with one, its configurations are measured too)
 *                    [--prompt-on <devices>] [--drafts-on <devices>]
 *                    [--reply-on <devices>] (as in janas-chat: the
 *                    devices the engine may choose for each kind of pass;
 *                    the rows measured are the configurations it may
 *                    choose for some kind)
 *                    [--npu-int8] (the NPU's rows with eight-bit
 *                    activations)
 *                    [--gpu-profile] (the GPU's own timestamps of each
 *                    phase of a pass on the card, in the report: as
 *                    JANAS_GPU_TOKEN_PROFILE=1)
 *                    [--settle <seconds>] (after the warm-up, wait up to
 *                    that long for the cache's background filling to end:
 *                    the configurations measured on a warm cache, as a
 *                    chat runs after its first minutes)
 *
 * With the fast prompt the GPU reads a prompt's blocks in every
 * configuration, "+ GPU" or not, and the report says so under the table;
 * "+ GPU" is its part in the shorter passes.
 *
 * The model is taken as janas-chat takes it (common/model_pick.c): a .jns
 * file, a name of the catalog (qwen3-4b), downloaded if it is not there
 * yet, or nothing, and then the list to choose from. Its prediction file
 * comes with it as in the chat: the one the catalog names, or for a file
 * of no catalog the one found beside it (--no-mtp: without).
 *
 * The prompt is prose that does not repeat itself (only a prompt longer
 * than the text repeats it, and the report says so). The drafts of the
 * prediction block are measured as a chat uses them: a question in the
 * model's chat format and the reply it writes, greedily and sampled as the
 * chat samples, each with the share of drafts that survived. Until 26 Sep
 * 2026 they were measured continuing the prompt, one paragraph repeated,
 * and so is any raw continuation of prose: a model copies the paragraphs it
 * has seen, the drafts guess the copy, and "with MTP" came out near twice
 * the plain rate on Qwen3.5-2B (95% of the drafts kept) where its chat got
 * a tenth more (40% kept).
 */
#include <stdarg.h>
#include <errno.h>
#include <limits.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "common/devices.h"
#include "common/env_names.h"
#include "common/model_pick.h"
#include "common/pool.h"
#include "common/sysinfo.h"
#include "llm/quant.h"
#include "common/words.h"
#include "janas/llm.h"
#include "llm/spec_config.h"
#include "llm/spec_json.h"
#include "llm/api_internal.h"
#include "llm/expert_cache.h"
#include "llm/generate.h"
#include "llm/gpu.h"
#include "llm/json.h"
#include "llm/tuner.h"

#ifndef JANAS_VERSION
#define JANAS_VERSION "unknown"
#endif

/* Plain English prose on several subjects, none of it repeated: what comes
   after it has to be written, not copied. */
static const char *TEXT =
    "The history of computing is a history of trade-offs between memory and "
    "arithmetic. Early machines stored a few thousand words, and every "
    "program was written around what fitted. Caches, virtual memory and "
    "solid-state disks each moved the boundary, but never removed it: the "
    "fastest code still reads the fewest bytes.\n\n"
    "On the west coast of Sardinia the wind comes from the north-west for "
    "most of the year, and the villages turn their backs to it. Stone walls "
    "divide the fields into small, irregular plots, some older than any "
    "written record of the families who farm them. In late spring the "
    "hillsides are yellow with broom, and by August they are the colour of "
    "straw.\n\n"
    "A good recipe for bread asks for less than people expect: flour, water, "
    "salt, a little yeast, and time. The dough should rest overnight in a "
    "cool room, where slow fermentation builds flavour that no shortcut can "
    "imitate. Bakers argue about hydration and flour the way musicians "
    "argue about tuning.\n\n"
    "The committee met on a Tuesday and, after two hours of discussion, "
    "postponed the decision on the new library until the budget for the "
    "following year was known. Three members voted against the delay, "
    "noting that construction costs had risen by a fifth since the "
    "project was first proposed.\n\n"
    "Birds that migrate at night navigate by the stars, by the Earth's "
    "magnetic field and, near the end of their journey, by smell. Young "
    "birds on their first flight make more mistakes than adults, and many "
    "of them end the autumn far from where their species usually winters.\n\n"
    "When the train finally left the station, Marta opened the letter her "
    "grandmother had given her that morning. It was short, written in pencil, "
    "and it began with a question she had not expected.";

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static FILE *report;

/* Printed to the terminal and to the report file. */
static void out(const char *fmt, ...) __attribute__((format(gnu_printf, 1, 2)));
static void out(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    if (report) {
        va_start(ap, fmt);
        vfprintf(report, fmt, ap);
        va_end(ap);
    }
}

struct run {
    double prefill, decode, mtp, mtp_s; /* token/s; MTP greedy and sampled */
    double hits;                        /* experts served from the cache, % */
    uint64_t drafted, accepted, drafted_s, accepted_s;
};

/* Prefill then n_gen tokens, greedy, without drafts: one token a pass,
   whatever it says, costs the same. */
static int measure(struct janas_llm_model *m, const int32_t *prompt,
                   uint32_t n_prompt, uint32_t n_gen, double *prefill,
                   double *decode)
{
    struct janas_gen_options o = {
        .spec_mode = JANAS_SPEC_OFF, .spec_k = 1, .spec_auto = 1, .eos = -1};
    struct janas_llm_session *s = janas_llm_session_create(m, &o);
    if (!s || janas_llm_session_append(s, prompt, n_prompt) != 0)
        return -1;
    /* without the NPU's compilations (a matrix's first block, once a
       session): said apart in the report */
    double k0 = janas_llm_model_npu_compile_seconds(m), t0 = now();
    if (janas_llm_session_prefill(s) != 0)
        return -1;
    double t1 = now();
    t0 += janas_llm_model_npu_compile_seconds(m) - k0;
    for (uint32_t i = 0; i < n_gen; i++) {
        int32_t t;
        if (janas_llm_session_next(s, &t) != 0)
            return -1;
    }
    double t2 = now();
    janas_llm_session_destroy(s);
    if (prefill)
        *prefill = (n_prompt - 1) / (t1 - t0);
    *decode = n_gen / (t2 - t1);
    return 0;
}

/* An ordinary request, nothing in it to copy from. */
static const char *QUESTION =
    "Explain in plain words how bread is made at home, from the flour to "
    "the oven, for someone who has never baked.";

/* A reply of n_gen tokens to QUESTION through the chat, drafts from the
   MTP block, greedy or sampled as the chat samples (with a fixed seed):
   its rate, and the drafts made and kept. */
static int measure_chat(janas_llm *llm, uint32_t n_gen, int sampled,
                        double *rate, uint64_t *drafted, uint64_t *accepted)
{
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    if (!sampled)
        cp.temperature = 0;
    cp.seed = 20260926;
    cp.speculate = 1;
    cp.thinking = 0;
    cp.max_reply = (int32_t)n_gen;
    janas_llm_chat *c;
    if (janas_llm_chat_create(llm, &cp, &c) != JANAS_LLM_OK)
        return -1;
    char buf[256];
    int32_t len, rc = janas_llm_chat_send(c, QUESTION, -1);
    while (rc == JANAS_LLM_OK)
        rc = janas_llm_chat_next(c, buf, sizeof(buf), &len);
    struct janas_llm_chat_stats st = {.size = sizeof(st)};
    if (rc != JANAS_LLM_DONE || janas_llm_chat_stats(c, &st) != JANAS_LLM_OK ||
        st.output_seconds <= 0) {
        janas_llm_chat_destroy(c);
        return -1;
    }
    *rate = st.output_tokens / st.output_seconds;
    *drafted += st.drafted;
    *accepted += st.accepted;
    janas_llm_chat_destroy(c);
    return 0;
}

/* Opt-in chat benchmark: every speculation strategy receives the same
   formatted prompt, greedy sampler, seed and output-token limit. The old
   table below remains a hardware/kernel diagnostic, not a speculation
   speedup comparison (its raw continuation and chat workloads differ). */
struct fair_options {
    int enabled, json, speculation, drafts, threads, warmups, profile_phases;
    int retain_controller;
    uint64_t seed;
    const char *prompt_file;
};

static const char *spec_name(int mode)
{
    static const char *const names[] = {"none", "auto", "lookup", "mtp", "hybrid"};
    return mode >= 0 && mode <= 4 ? names[mode] : "unknown";
}

static int fair_threads(int n)
{
    if (!n)
        return 0;
    struct janas_cpu_layout l;
    if (janas_cpu_layout(NULL, 0, &l) != 0 || n > l.n)
        return -1;
#ifdef _WIN32
    DWORD_PTR mask = 0;
    for (int i = 0; i < n; i++) {
        if (l.cpus[i] >= (int)(8 * sizeof mask))
            return -1;
        mask |= (DWORD_PTR)1 << l.cpus[i];
    }
    return SetProcessAffinityMask(GetCurrentProcess(), mask) ? 0 : -1;
#else
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < n; i++) {
        if (l.cpus[i] >= CPU_SETSIZE)
            return -1;
        CPU_SET(l.cpus[i], &mask);
    }
    return sched_setaffinity(0, sizeof mask, &mask);
#endif
}

static void json_text(struct janas_buf *b, const char *key, const char *text)
{
    janas_buf_printf(b, "\"%s\":", key);
    janas_json_write_str(b, text ? text : "", text ? strlen(text) : 0);
    janas_buf_puts(b, ",");
}

static int json_line(struct janas_buf *b)
{
    if (b->oom)
        return -1;
    janas_buf_puts(b, "}\n");
    if (b->oom || fwrite(b->p, 1, b->n, stdout) != b->n)
        return -1;
    fflush(stdout);
    return 0;
}

struct fair_profile {
    double start[JANAS_PH_COUNT], tail[JANAS_PH_COUNT], end[JANAS_PH_COUNT];
    double began, tail_began, ended;
    uint32_t tokens_before_tail;
    int has_tail, pool_trace, expert_trace;
};

static int fair_trace_enabled(const char *name)
{
    const char *value = getenv(name);
    return value && atoi(value) > 0;
}

/* model_phases also prints/resets EXPTRACE when enabled. Pool and expert
   details are nested inside these aggregate phases, never added to them. */
static void fair_profile_start(struct fair_profile *s,
                                struct janas_llm_model *m,
                                const char *phase, int index)
{
    memset(s, 0, sizeof(*s));
    s->pool_trace = fair_trace_enabled("JANAS_POOLTRACE");
    s->expert_trace = fair_trace_enabled("JANAS_EXPTRACE");
    if (s->expert_trace)
        fprintf(stderr, "janas-bench profile %s %d: discard earlier expert counters\n",
                phase, index);
    janas_llm_model_phases(m, s->start);
    if (s->pool_trace)
        janas_pool_trace_reset();
    s->began = now();
}

static void fair_profile_tail(struct fair_profile *s,
                               struct janas_llm_model *m,
                               const char *phase, int index, uint32_t tokens)
{
    double elapsed = now() - s->began;
    char label[160];
    snprintf(label, sizeof label, "%s %d prompt and first output call", phase,
             index);
    if (s->expert_trace)
        fprintf(stderr, "janas-bench profile %s: expert detail\n", label);
    janas_llm_model_phases(m, s->tail);
    if (s->pool_trace) {
        janas_pool_trace_report(label, elapsed);
        janas_pool_trace_reset();
    }
    s->tokens_before_tail = tokens;
    s->has_tail = 1;
    s->tail_began = now();
}

static void fair_profile_end(struct fair_profile *s,
                              struct janas_llm_model *m,
                              const char *phase, int index)
{
    s->ended = now();
    char label[160];
    snprintf(label, sizeof label, "%s %d %s", phase, index,
             s->has_tail ? "reply tail after first output call" : "whole chat");
    if (s->expert_trace)
        fprintf(stderr, "janas-bench profile %s: expert detail\n", label);
    janas_llm_model_phases(m, s->end);
    if (s->pool_trace)
        janas_pool_trace_report(label,
                                s->ended - (s->has_tail ? s->tail_began : s->began));
}

static void fair_phase_json(struct janas_buf *b, const double *start,
                             const double *end)
{
    static const char *const names[JANAS_PH_COUNT] = {
        "qkv", "attention", "wo", "router", "experts", "output", "ssm", "mix"};
    janas_buf_puts(b, "{");
    for (int i = 0; i < JANAS_PH_COUNT; i++)
        janas_buf_printf(b, "%s\"%s\":%.9g", i ? "," : "", names[i],
                         end[i] - start[i]);
    janas_buf_puts(b, "}");
}

static void fair_profile_json(struct janas_buf *b, const struct fair_profile *s,
                               uint32_t output_tokens)
{
    janas_buf_printf(b,
        ",\"profile_phases\":{\"units\":\"seconds\","
        "\"scope\":\"aggregate_model_wall_time\","
        "\"detail_relationship\":\"expert_and_pool_traces_are_nested_not_additive\","
        "\"diagnostic_timings\":true,\"pool_trace_stderr\":%s,"
        "\"expert_trace_stderr\":%s,\"chat_wall_seconds\":%.9g,"
        "\"whole_chat\":",
        s->pool_trace ? "true" : "false", s->expert_trace ? "true" : "false",
        s->ended - s->began);
    fair_phase_json(b, s->start, s->end);
    janas_buf_printf(b,
        ",\"reply_tail\":{\"available\":%s,"
        "\"scope\":\"after_first_output_call_to_done\","
        "\"excludes_first_output_call_model_work\":true,"
        "\"output_tokens_before_tail\":%u,\"output_tokens\":%u,"
        "\"wall_seconds\":%.9g,\"phase_seconds\":",
        s->has_tail ? "true" : "false", s->tokens_before_tail,
        s->has_tail && output_tokens >= s->tokens_before_tail
            ? output_tokens - s->tokens_before_tail : 0,
        s->has_tail ? s->ended - s->tail_began : 0);
    fair_phase_json(b, s->has_tail ? s->tail : s->end, s->end);
    janas_buf_puts(b, "}}");
}

static void fair_profile_print(const struct fair_profile *s,
                                const char *phase, int index)
{
    static const char *const names[JANAS_PH_COUNT] = {
        "qkv", "attention", "wo", "router", "experts", "output", "ssm", "mix"};
    printf("  profile %s %d whole chat, phase seconds:", phase, index);
    for (int k = 0; k < JANAS_PH_COUNT; k++)
        printf(" %s %.6f", names[k], s->end[k] - s->start[k]);
    if (s->has_tail) {
        printf("\n  reply tail after first output call, phase seconds:");
        for (int k = 0; k < JANAS_PH_COUNT; k++)
            printf(" %s %.6f", names[k], s->end[k] - s->tail[k]);
    }
    puts("\n  diagnostic run; expert/pool detail is nested, not additional time");
}

struct fair_cache {
    uint64_t capacity, arena, slots, slot_bytes;
    struct janas_expert_cache_stats st;
};

struct fair_caches {
    struct fair_cache main, mtp, total;
    int embedded_mtp, separate_mtp;
};

static void fair_cache_read(struct janas_expert_cache *c, struct fair_cache *s)
{
    memset(s, 0, sizeof(*s));
    if (!c)
        return;
    janas_expert_cache_arena(c, &s->arena);
    s->capacity =
        s->arena >= JANAS_ARENA_GUARD ? s->arena - JANAS_ARENA_GUARD : 0;
    s->slots = janas_expert_cache_slots(c);
    s->slot_bytes = janas_expert_cache_slot_bytes(c);
    janas_expert_cache_stats(c, &s->st);
}

static struct janas_expert_cache_stats
fair_cache_sum(const struct janas_expert_cache_stats *a,
               const struct janas_expert_cache_stats *b)
{
    return (struct janas_expert_cache_stats){
        .requests = a->requests + b->requests,
        .hits = a->hits + b->hits,
        .misses = a->misses + b->misses,
        .bytes_read = a->bytes_read + b->bytes_read,
        .warm_bytes = a->warm_bytes + b->warm_bytes,
        .io_seconds = a->io_seconds + b->io_seconds,
        .wait_seconds = a->wait_seconds + b->wait_seconds,
        .prefetch_reads = a->prefetch_reads + b->prefetch_reads,
        .prefetch_used = a->prefetch_used + b->prefetch_used,
        .prefetch_bytes = a->prefetch_bytes + b->prefetch_bytes};
}

static void fair_caches_total(struct fair_caches *s)
{
    s->total.capacity = s->main.capacity + s->mtp.capacity;
    s->total.arena = s->main.arena + s->mtp.arena;
    s->total.st = fair_cache_sum(&s->main.st, &s->mtp.st);
}

static void fair_caches_read(struct janas_llm_model *m, struct fair_caches *s)
{
    memset(s, 0, sizeof(*s));
    struct janas_expert_cache *sidecar = janas_llm_model_mtp_cache(m);
    fair_cache_read(janas_llm_model_cache(m), &s->main);
    fair_cache_read(sidecar, &s->mtp);
    s->separate_mtp = sidecar != NULL;
    s->embedded_mtp = janas_llm_model_has_mtp(m) && !sidecar;
    /* _stats and _arena already include embedded MTP's wider partitions.
       Only a distinct sidecar is added here. */
    fair_caches_total(s);
}

static struct janas_expert_cache_stats
fair_cache_delta(const struct janas_expert_cache_stats *end,
                 const struct janas_expert_cache_stats *start)
{
    return (struct janas_expert_cache_stats){
        .requests = end->requests - start->requests,
        .hits = end->hits - start->hits,
        .misses = end->misses - start->misses,
        .bytes_read = end->bytes_read - start->bytes_read,
        .warm_bytes = end->warm_bytes - start->warm_bytes,
        .io_seconds = end->io_seconds - start->io_seconds,
        .wait_seconds = end->wait_seconds - start->wait_seconds,
        .prefetch_reads = end->prefetch_reads - start->prefetch_reads,
        .prefetch_used = end->prefetch_used - start->prefetch_used,
        .prefetch_bytes = end->prefetch_bytes - start->prefetch_bytes};
}

static void fair_cache_json(struct janas_buf *b, const char *name,
                            const struct fair_cache *s, int slots)
{
    const struct janas_expert_cache_stats *c = &s->st;
    janas_buf_printf(
        b,
        "\"%s\":{\"capacity_bytes\":%llu,\"arena_bytes\":%llu,"
        "\"requests\":%llu,\"hits\":%llu,\"misses\":%llu,"
        "\"logical_demand_read_bytes\":%llu,\"logical_prefetch_read_bytes\":%llu,"
        "\"logical_warm_read_bytes\":%llu,\"logical_read_bytes\":%llu,"
        "\"prefetch_reads\":%llu,\"prefetch_used\":%llu,"
        "\"io_seconds\":%.9g,\"wait_seconds\":%.9g",
        name, (unsigned long long)s->capacity, (unsigned long long)s->arena,
        (unsigned long long)c->requests, (unsigned long long)c->hits,
        (unsigned long long)c->misses, (unsigned long long)c->bytes_read,
        (unsigned long long)c->prefetch_bytes,
        (unsigned long long)c->warm_bytes,
        (unsigned long long)(c->bytes_read + c->prefetch_bytes + c->warm_bytes),
        (unsigned long long)c->prefetch_reads,
        (unsigned long long)c->prefetch_used, c->io_seconds, c->wait_seconds);
    if (slots)
        janas_buf_printf(
            b,
            ",\"main_partition_slots\":%llu,\"main_partition_slot_bytes\":%llu",
            (unsigned long long)s->slots, (unsigned long long)s->slot_bytes);
    janas_buf_puts(b, "}");
}

static void fair_caches_json(struct janas_buf *b, const struct fair_caches *s,
                             uint64_t requested, int delta)
{
    janas_buf_printf(
        b,
        ",\"expert_cache\":{\"main_allocation\":\"%s\",\"main_requested_bytes\":%llu,"
        "\"mtp_storage\":\"%s\",\"mtp_separate_allocation\":\"%s\","
        "\"main_includes_embedded_mtp\":%s,\"counter_scope\":\"%s\","
        "\"read_bytes_scope\":\"logical_file_reads_not_physical_ssd\","
        "\"physical_ssd_throughput_measured\":false,"
        "\"io_seconds_scope\":\"sum_of_cache_read_wall_times\",",
        requested ? "explicit" : "auto", (unsigned long long)requested,
        s->separate_mtp   ? "separate"
        : s->embedded_mtp ? "shared_main"
                          : "none",
        s->separate_mtp ? "auto" : "none", s->embedded_mtp ? "true" : "false",
        delta ? "sample_delta" : "cache_lifetime");
    fair_cache_json(b, "main", &s->main, 1);
    janas_buf_puts(b, ",");
    fair_cache_json(b, "mtp_separate", &s->mtp, 1);
    janas_buf_puts(b, ",");
    fair_cache_json(b, "total", &s->total, 0);
    janas_buf_puts(b, "}");
}

static void fair_cache_print(const char *name, const struct fair_cache *s)
{
    const struct janas_expert_cache_stats *c = &s->st;
    printf(
        "  %s cache: capacity %llu bytes, arena %llu bytes; requests/hits/misses "
        "%llu/%llu/%llu; logical demand/prefetch/warm reads %llu/%llu/%llu bytes; "
        "I/O sum %.3f s, wait %.3f s\n",
        name, (unsigned long long)s->capacity, (unsigned long long)s->arena,
        (unsigned long long)c->requests, (unsigned long long)c->hits,
        (unsigned long long)c->misses, (unsigned long long)c->bytes_read,
        (unsigned long long)c->prefetch_bytes,
        (unsigned long long)c->warm_bytes, c->io_seconds, c->wait_seconds);
}

static void fair_accel_json(struct janas_buf *b,
                            const struct janas_llm_accel_counters *end,
                            const struct janas_llm_accel_counters *start)
{
    struct janas_llm_accel_counters zero = {0};
    const int delta = start != NULL;
    if (!start)
        start = &zero;
    janas_buf_printf(
        b,
        ",\"accelerator_counters\":{\"count_scope\":\"%s\","
        "\"npu_products\":%llu,\"npu_ffns\":%llu,\"npu_layer_passes\":%llu,"
        "\"npu_prepared_plans\":%llu,\"npu_weight_bytes\":%llu,"
        "\"npu_weight_bytes_scope\":\"current_unique_resident\","
        "\"gpu_product_submissions\":%llu,\"gpu_products\":%llu,\"gpu_whole_passes\":%llu}",
        delta ? "sample_delta" : "model_lifetime",
        (unsigned long long)(end->npu_products - start->npu_products),
        (unsigned long long)(end->npu_ffns - start->npu_ffns),
        (unsigned long long)(end->npu_layer_passes - start->npu_layer_passes),
        (unsigned long long)(end->npu_prepared_plans -
                             start->npu_prepared_plans),
        (unsigned long long)end->npu_weight_bytes,
        (unsigned long long)(end->gpu_product_submissions -
                             start->gpu_product_submissions),
        (unsigned long long)(end->gpu_products - start->gpu_products),
        (unsigned long long)(end->gpu_whole_passes - start->gpu_whole_passes));
}

static int fair_bench(const char *path, struct janas_llm_params *p,
                      uint32_t n_gen, int rounds, const struct fair_options *f)
{
    char *owned = NULL;
    const char *prompt = QUESTION;
    if (f->prompt_file) {
        FILE *in = fopen(f->prompt_file, "rb");
        if (!in) {
            fprintf(stderr, "janas-bench: prompt: %s\n", strerror(errno));
            return 1;
        }
        owned = malloc((1 << 20) + 1);
        if (!owned) {
            fclose(in);
            return 1;
        }
        size_t n = fread(owned, 1, (1 << 20) + 1, in);
        int bad = ferror(in) || n == 0 || n > (1 << 20) || memchr(owned, 0, n);
        fclose(in);
        if (bad) {
            fprintf(
                stderr,
                "janas-bench: prompt must be 1..1048576 bytes without NUL\n");
            free(owned);
            return 2;
        }
        owned[n] = 0;
        prompt = owned;
    }
    if (fair_threads(f->threads) != 0) {
        fprintf(stderr, "janas-bench: cannot select %d allowed CPU threads\n",
                f->threads);
        free(owned);
        return 2;
    }
    /* No previous workload's expert profile, automatic quality reductions or
       alternative thread trials hidden in a timed sample. Device restrictions
       still select the requested prompt/draft/reply hardware. */
    setenv("JANAS_TUNE", "0", 1);
    setenv("JANAS_TUNE_THREADS", "0", 1);
    setenv("JANAS_WARM", "0", 1);
    setenv("JANAS_MISS_SKIP", "0", 1);
    setenv("JANAS_EXPERT_BITS", "6", 1);
    setenv("JANAS_HEAD_BITS", "0", 1);
    unsetenv("JANAS_EXPERTS");
    p->no_preload = 1;
    p->min_quality = JANAS_LLM_QUALITY_HIGH;
    p->expert_bits = 6;
    p->head_bits = 0;
    janas_llm *llm = NULL;
    janas_llm_chat *c = NULL;
    double t0 = now();
    int32_t rc = janas_llm_open(path, p, &llm);
    double load_seconds = now() - t0;
    if (rc != JANAS_LLM_OK) {
        fprintf(stderr, "janas-bench: %s\n", janas_llm_last_error());
        free(owned);
        return 1;
    }
    struct janas_llm_model *m = janas_llm_internal_model(llm);
    struct janas_gpu *gpu = janas_llm_model_gpu(m);
    const char *npu = janas_llm_model_npu_name(m);
    struct janas_buf b = {0};
    struct fair_caches caches_setup;
    struct janas_llm_accel_counters accel_setup;
    fair_caches_read(m, &caches_setup);
    janas_llm_model_accel_counters(m, &accel_setup);
    int resolved = f->speculation == JANAS_LLM_SPEC_AUTO
                       ? (janas_llm_model_has_mtp(m) ? JANAS_LLM_SPEC_MTP
                                                     : JANAS_LLM_SPEC_LOOKUP)
                       : f->speculation;
    if (f->json) {
        janas_buf_puts(&b, "{\"schema\":1,\"kind\":\"setup\",");
        json_text(&b, "version", JANAS_VERSION);
        json_text(&b, "cpu_kernels", janas_q4k_kernel_name());
        json_text(&b, "model", path);
        json_text(&b, "speculation", spec_name(f->speculation));
        json_text(&b, "speculation_resolved", spec_name(resolved));
        json_text(&b, "fair_controller_protocol", f->retain_controller
                  ? "retained_across_fresh_conversations" : "fresh_chat_per_sample");
        janas_buf_printf(&b, "\"fair_retain_controller\":%s,\"conversation_cache_reuse\":false,",
                         f->retain_controller ? "true" : "false");
        json_text(&b, "history_index_scope", "canonical_tokens_of_current_sample");
        json_text(&b, "target_sampler_seed_scope", "reset_each_sample");
        struct janas_llm_speculation_config sc;
        janas_llm_speculation_config_default(&sc);
        char config_error[160];
        if (janas_spec_config_env(&sc, config_error, sizeof config_error) != 0) {
            fprintf(stderr, "janas-bench: %s\n", config_error); goto failed;
        }
        struct janas_llm_hybrid_config hc;
        janas_llm_hybrid_config_default(&hc);
        if (janas_hybrid_config_env(&hc, config_error, sizeof config_error) != 0) {
            fprintf(stderr, "janas-bench: %s\n", config_error); goto failed;
        }
        json_text(&b, "hybrid_policy", hc.policy ? "cost" : "legacy");
        json_text(&b, "hybrid_fallback_planner", hc.policy ? "legacy" : "configured");
        janas_buf_printf(&b, "\"lookup_drafts\":%d,\"mtp_drafts\":%d,\"hybrid_probe_percent\":%d,",
                         hc.lookup_max_drafts, hc.mtp_max_drafts, hc.probe_percent);
        struct janas_llm_s2_config s2;
        janas_llm_s2_config_default(&s2);
        if (janas_s2_config_env(&s2, config_error, sizeof config_error) != 0) {
            fprintf(stderr, "janas-bench: %s\n", config_error); goto failed;
        }
        json_text(&b, "s2_exact", s2.exact ? "on" : "off");
        janas_buf_printf(&b, "\"s2_branches\":%d,\"s2_max_nodes\":%d,\"s2_min_gain_percent\":%d,",
                         s2.branches, s2.max_nodes, s2.min_gain_percent);
        const char *const planners[] = {"legacy", "cycle", "fixed"};
        json_text(&b, "spec_planner", planners[sc.planner]);
        json_text(&b, "draft_vocab", sc.draft_vocab_policy ? "adaptive" : "legacy");
        janas_buf_printf(&b, "\"spec_explore\":%d,\"draft_vocab_rows\":%u,", sc.explore_percent, sc.draft_vocab_rows);
        json_text(&b, "gpu", gpu ? janas_gpu_name(gpu) : "");
        json_text(&b, "gpu_unavailable_reason",
                  gpu ? "" : janas_llm_model_gpu_why(m));
        json_text(&b, "npu", npu);
        json_text(&b, "npu_unavailable_reason",
                  npu ? "" : janas_llm_model_npu_why(m));
        janas_buf_printf(
            &b,
            "\"load_seconds\":%.9g,\"seed\":%llu,\"context_tokens\":%u,"
            "\"generation_limit\":%u,\"requested_threads\":%d,"
            "\"cache_bytes_requested\":%llu,\"reserve_bytes\":%llu,"
            "\"draft_tokens_cap\":%d,\"quality\":\"high\","
            "\"os_cache_state\":\"uncontrolled\","
            "\"thread_budget_scope\":\"process_affinity\","
            "\"peak_rss_scope\":\"process_lifetime\","
            "\"cache_bytes_resolved\":%llu,\"mtp_cache_bytes_resolved\":%llu,"
            "\"cache_total_bytes_resolved\":%llu,"
            "\"cache_resolved_scope\":\"allocated_capacity_excluding_arena_guards\"",
            load_seconds, (unsigned long long)f->seed, p->n_ctx, n_gen,
            f->threads, (unsigned long long)p->cache_bytes,
            (unsigned long long)p->reserve_bytes, f->drafts,
            (unsigned long long)caches_setup.main.capacity,
            (unsigned long long)caches_setup.mtp.capacity,
            (unsigned long long)caches_setup.total.capacity);
        fair_caches_json(&b, &caches_setup, p->cache_bytes, 0);
        fair_accel_json(&b, &accel_setup, NULL);
        if (f->profile_phases)
            janas_buf_puts(&b, ",\"profile_phases_enabled\":true");
        if (json_line(&b) != 0)
            goto failed;
        janas_buf_free(&b);
    } else {
        printf(
            "fair chat benchmark: %s, %d drafts ceiling, seed %llu, %u tokens\n",
            spec_name(f->speculation), f->drafts, (unsigned long long)f->seed,
            n_gen);
        printf(
            "load %.3f s; quality high; fresh process does not imply cold OS/SSD cache\n",
            load_seconds);
        printf("controller protocol: %s; fresh conversation and sampler seed each sample\n",
               f->retain_controller ? "retained measurements" : "fresh chat per sample");
        printf("expert cache allocation: main %s requested %llu -> %llu bytes; "
               "MTP %s -> %llu separate bytes; total %llu bytes%s\n",
               p->cache_bytes ? "explicit" : "auto",
               (unsigned long long)p->cache_bytes,
               (unsigned long long)caches_setup.main.capacity,
               caches_setup.separate_mtp   ? "separate auto"
               : caches_setup.embedded_mtp ? "shared with main"
                                           : "none",
               (unsigned long long)caches_setup.mtp.capacity,
               (unsigned long long)caches_setup.total.capacity,
               caches_setup.embedded_mtp ? " (embedded MTP already included)"
                                         : "");
        fair_cache_print("main (at setup)", &caches_setup.main);
        fair_cache_print("MTP separate (at setup)", &caches_setup.mtp);
        fair_cache_print("total (at setup)", &caches_setup.total);
        puts(
            "expert read bytes are logical file reads, not physical SSD throughput; "
            "peak RSS is the process-lifetime high-water mark");
    }
    for (int i = 0; i < 1 + f->warmups + rounds; i++) {
        const char *phase = i == 0            ? "fresh_process"
                            : i <= f->warmups ? "warmup"
                                              : "warm";
        struct janas_llm_chat_params cp;
        janas_llm_chat_params_default(&cp);
        cp.temperature = 0;
        cp.seed = f->seed;
        cp.speculate = f->speculation;
        cp.thinking = 0;
        cp.recap = 0;
        cp.max_reply = (int32_t)n_gen;
        if (c) {
            /* Reset tokens and computed context, retaining only session-local
               cost estimates. The index resynchronizes against this sample's
               canonical history before its next query. Reapply params so the
               target sampler starts from the same seed as a fresh chat. */
            rc = janas_llm_chat_reset(c);
            if (rc == JANAS_LLM_OK)
                rc = janas_llm_chat_set_params(c, &cp);
        } else {
            rc = janas_llm_chat_create(llm, &cp, &c);
        }
        if (rc != JANAS_LLM_OK)
            goto failed;
        if (janas_llm_chat_speculation(c, f->speculation, f->drafts) !=
                JANAS_LLM_OK ||
            janas_llm_chat_system(c, "You are a helpful assistant.", -1) !=
                JANAS_LLM_OK) {
            goto failed;
        }
        double compile0 = janas_llm_model_npu_compile_seconds(m);
        struct fair_caches caches0, caches1;
        struct janas_llm_accel_counters accel0, accel1;
        fair_caches_read(m, &caches0);
        janas_llm_model_accel_counters(m, &accel0);
        uint64_t hash = 14695981039346656037ull;
        uint64_t output_bytes = 0;
        char text[1024];
        int32_t len;
        struct fair_profile profile;
        if (f->profile_phases)
            fair_profile_start(&profile, m, phase, i);
        rc = janas_llm_chat_send(c, prompt, -1);
        while (rc == JANAS_LLM_OK) {
            rc = janas_llm_chat_next(c, text, sizeof text, &len);
            if (f->profile_phases && !profile.has_tail) {
                struct janas_llm_chat_stats progress = {.size = sizeof progress};
                if (janas_llm_chat_stats(c, &progress) == JANAS_LLM_OK &&
                    progress.output_tokens > 0)
                    fair_profile_tail(&profile, m, phase, i,
                                      progress.output_tokens);
            }
            if (rc == JANAS_LLM_OK)
                for (int32_t j = 0; j < len; j++) {
                    hash = (hash ^ (unsigned char)text[j]) * 1099511628211ull;
                    output_bytes++;
                }
        }
        struct janas_llm_chat_stats st = {.size = sizeof st};
        if (rc != JANAS_LLM_DONE ||
            janas_llm_chat_stats(c, &st) != JANAS_LLM_OK) {
            goto failed;
        }
        if (st.cached_tokens != 0) {
            fprintf(stderr, "janas-bench: fresh-conversation sample reused %u prompt tokens\n",
                    st.cached_tokens);
            goto failed;
        }
        if (f->profile_phases)
            fair_profile_end(&profile, m, phase, i);
        fair_caches_read(m, &caches1);
        janas_llm_model_accel_counters(m, &accel1);
        struct fair_caches cache_delta = caches1;
        cache_delta.main.st =
            fair_cache_delta(&caches1.main.st, &caches0.main.st);
        cache_delta.mtp.st = fair_cache_delta(&caches1.mtp.st, &caches0.mtp.st);
        fair_caches_total(&cache_delta);
        const struct janas_expert_cache_stats *cs = &cache_delta.total.st;
        double rate =
            st.output_seconds > 0 ? st.output_tokens / st.output_seconds : 0;
        if (f->json) {
            janas_buf_puts(&b, "{\"schema\":1,\"kind\":\"sample\",");
            json_text(&b, "phase", phase);
            json_text(&b, "speculation", spec_name(f->speculation));
            json_text(&b, "speculation_resolved", spec_name(resolved));
            json_text(&b, "fair_controller_protocol", f->retain_controller
                      ? "retained_across_fresh_conversations" : "fresh_chat_per_sample");
            janas_buf_printf(
                &b,
                "\"index\":%d,\"output_tokens\":%u,\"output_bytes\":%llu,"
                "\"output_hash\":\"%016llx\",\"input_tokens\":%u,"
                "\"prompt_tokens\":%u,\"cached_tokens\":%u,\"finish\":%d,"
                "\"input_seconds\":%.9g,\"output_seconds\":%.9g,"
                "\"total_seconds\":%.9g,\"first_token_seconds\":%.9g,"
                "\"output_tokens_per_second\":%.9g,"
                "\"drafted\":%u,\"accepted\":%u,\"auto_off_passes\":%u,"
                "\"input_threads\":%d,\"output_threads\":%d,"
                "\"peak_rss_bytes\":%llu,\"peak_rss_scope\":\"process_lifetime\","
                "\"context_bytes\":%llu,"
                "\"expert_requests\":%llu,\"expert_hits\":%llu,\"expert_misses\":%llu,"
                "\"expert_bytes_read\":%llu,\"expert_prefetch_bytes\":%llu,"
                "\"expert_warm_bytes\":%llu,\"expert_logical_read_bytes\":%llu,"
                "\"expert_counters_scope\":\"main_plus_separate_mtp\","
                "\"expert_io_seconds\":%.9g,\"expert_wait_seconds\":%.9g,"
                "\"npu_compile_seconds\":%.9g,\"gpu_policy_used\":%d,\"npu_policy_used\":%d",
                i, st.output_tokens, (unsigned long long)output_bytes,
                (unsigned long long)hash, st.input_tokens, st.prompt_tokens,
                st.cached_tokens, st.finish, st.input_seconds,
                st.output_seconds, st.total_seconds, st.first_token_seconds,
                rate, st.drafted, st.accepted, st.auto_off_passes,
                st.input_threads, st.output_threads,
                (unsigned long long)st.peak_rss_bytes,
                (unsigned long long)st.context_bytes,
                (unsigned long long)cs->requests, (unsigned long long)cs->hits,
                (unsigned long long)cs->misses,
                (unsigned long long)cs->bytes_read,
                (unsigned long long)cs->prefetch_bytes,
                (unsigned long long)cs->warm_bytes,
                (unsigned long long)(cs->bytes_read + cs->prefetch_bytes +
                                     cs->warm_bytes),
                cs->io_seconds, cs->wait_seconds,
                janas_llm_model_npu_compile_seconds(m) - compile0,
                janas_llm_model_gpu_used(m), janas_llm_model_npu_used(m));
            fair_caches_json(&b, &cache_delta, p->cache_bytes, 1);
            fair_accel_json(&b, &accel1, &accel0);
            struct janas_llm_speculation_metrics sm = {
                .size = sizeof sm, .version = JANAS_LLM_SPEC_METRICS_VERSION};
            if (janas_llm_chat_speculation_metrics(c, &sm) != JANAS_LLM_OK) {
                janas_llm_chat_destroy(c); goto failed;
            }
            janas_buf_puts(&b, ",\"speculation_metrics\":");
            janas_spec_metrics_json(&b, &sm);
            struct janas_llm_hybrid_metrics hm = {
                .size = sizeof hm, .version = JANAS_LLM_HYBRID_METRICS_VERSION};
            if (janas_llm_chat_hybrid_metrics(c, &hm) != JANAS_LLM_OK) goto failed;
            janas_buf_puts(&b, ",\"hybrid_metrics\":");
            janas_hybrid_metrics_json(&b, &hm);
            struct janas_llm_s2_metrics s2m = {
                .size = sizeof s2m, .version = JANAS_LLM_S2_METRICS_VERSION};
            if (janas_llm_chat_s2_metrics(c, &s2m) != JANAS_LLM_OK) goto failed;
            janas_buf_puts(&b, ",\"s2_metrics\":");
            janas_s2_metrics_json(&b, &s2m);

            if (f->profile_phases)
                fair_profile_json(&b, &profile, st.output_tokens);
            if (json_line(&b) != 0) {
                janas_llm_chat_destroy(c);
                goto failed;
            }
            janas_buf_free(&b);
        } else {
            printf("%s %d: %u tokens, %.3f tok/s, TTFT %.3f s, drafts %u/%u, "
                   "logical expert reads %llu bytes, wait %.3f s, "
                   "peak RSS %llu bytes (process lifetime)\n",
                   phase, i, st.output_tokens, rate, st.first_token_seconds,
                   st.accepted, st.drafted,
                   (unsigned long long)(cs->bytes_read + cs->prefetch_bytes +
                                        cs->warm_bytes),
                   cs->wait_seconds, (unsigned long long)st.peak_rss_bytes);
            fair_cache_print("main", &cache_delta.main);
            fair_cache_print("MTP separate", &cache_delta.mtp);
            fair_cache_print("total", &cache_delta.total);
            printf(
                "  accelerator inference: NPU products %llu, FFNs %llu, layer passes %llu; "
                "GPU submissions/products/whole passes %llu/%llu/%llu; "
                "NPU prepared plans +%llu (setup), resident weights %llu bytes\n",
                (unsigned long long)(accel1.npu_products - accel0.npu_products),
                (unsigned long long)(accel1.npu_ffns - accel0.npu_ffns),
                (unsigned long long)(accel1.npu_layer_passes -
                                     accel0.npu_layer_passes),
                (unsigned long long)(accel1.gpu_product_submissions -
                                     accel0.gpu_product_submissions),
                (unsigned long long)(accel1.gpu_products - accel0.gpu_products),
                (unsigned long long)(accel1.gpu_whole_passes -
                                     accel0.gpu_whole_passes),
                (unsigned long long)(accel1.npu_prepared_plans -
                                     accel0.npu_prepared_plans),
                (unsigned long long)accel1.npu_weight_bytes);
            if (f->profile_phases)
                fair_profile_print(&profile, phase, i);
        }
        if (!f->retain_controller) {
            janas_llm_chat_destroy(c); /* history is never reused by lookup */
            c = NULL;
        }
    }
    janas_llm_chat_destroy(c);
    c = NULL;
    janas_llm_close(llm);
    free(owned);
    return 0;
failed:
    fprintf(stderr, "janas-bench: fair benchmark failed: %s\n",
            janas_llm_last_error());
    janas_buf_free(&b);
    janas_llm_chat_destroy(c);
    janas_llm_close(llm);
    free(owned);
    return 1;
}

int main(int argc, char **argv)
{
    janas_env_check("janas-bench");
    if (argc > 1 &&
        (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        fprintf(stderr,
                "usage: janas-bench [<model>] [--models <dir>] "
                "[--mtp <file> | --no-mtp] [--prompt <tokens>] "
                "[--gen <tokens>] [--ctx <tokens>] [--cache <GiB>] "
                "[--reserve <GiB>] [--rounds <n>] [--settle <seconds>] "
                "[--no-npu] [--prompt-on <devices>] [--drafts-on <devices>] "
                "[--reply-on <devices>] [--npu-int8] [--gpu-profile]\n"
                "fair chat mode: --speculation none|lookup|mtp|hybrid|auto "
                "[--draft-tokens <1..16>] [--prompt-file <utf8>] "
                "[--seed <n>] [--threads <n>] [--warmups <n>] [--json] "
                "[--profile-phases] [--fair-retain-controller]\n"
                JANAS_SPEC_RUNTIME_HELP
                "--profile-phases: diagnostic phase wall times per chat and "
                "reply tail (first output call excluded); JANAS_POOLTRACE=1 "
                "and JANAS_EXPTRACE=1 add nested detail on stderr. "
                "Repeat without profiling for throughput.\n"
                "--fair-retain-controller: retain planner measurements across "
                "fresh conversations; off by default.\n"
                "<model>: a .jns file, or a name of the models Janas knows "
                "(downloaded if it is not there yet); none: the list\n"
                "<devices>: auto (default), cpu, gpu, npu, npu+gpu (both in "
                "each pass), or a list apart by commas to choose from\n");
        return 2;
    }
    struct janas_llm_params p;
    janas_llm_params_default(&p);
    p.mode = JANAS_LLM_MODE_MAX; /* every configuration is measured */
    p.n_ctx = 4096;
    uint32_t n_prompt = 256, n_gen = 64;
    int rounds = 3;
    long long settle = 0;
    const char *model = NULL, *models = NULL;
    int no_mtp = 0;
    struct fair_options fair = {.speculation = JANAS_LLM_SPEC_AUTO,
                                .warmups = 1, .seed = 20260926};
    static const char *const options[] = {
        "--mtp",       "--no-mtp",   "--models",   "--prompt",
        "--gen",       "--ctx",      "--rounds",   "--cache",
        "--reserve",   "--settle",   "--no-npu",   "--prompt-on",
        "--drafts-on", "--reply-on", "--npu-int8", "--gpu-profile",
        "--speculation", "--draft-tokens", "--prompt-file", "--seed",
        "--threads", "--warmups", "--json", "--profile-phases", "--fair-retain-controller",
        "--spec-planner", "--spec-explore", "--draft-vocab", "--draft-vocab-rows"};
    for (int i = 1; i < argc; i += 2) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        char why[200] = "";
        long long x = 0;
        double r = 0;
        if (a[0] != '-' && !model) {
            model = a;
            i--;
            continue;
        }
        if (janas_spec_runtime_is_option(a)) {
            if (janas_spec_runtime_option(a, v, why, sizeof why) < 0) {
                fprintf(stderr, "janas-bench: %s\n", why); return 2;
            }
            fair.enabled = 1;
            continue;
        }
        if (strcmp(a, "--no-mtp") == 0) {
            no_mtp = 1;
            i--;
            continue;
        }
        if (strcmp(a, "--no-npu") == 0) {
            setenv("JANAS_NPU", "0", 1); /* read when the model opens */
            i--;
            continue;
        }
        if (strcmp(a, "--npu-int8") == 0) {
            setenv("JANAS_NPU_A8", "1", 1); /* so is this */
            i--;
            continue;
        }
        if (strcmp(a, "--gpu-profile") == 0) {
            setenv("JANAS_GPU_TOKEN_PROFILE", "1", 1); /* and this */
            i--;
            continue;
        }
        if (strcmp(a, "--json") == 0) {
            fair.enabled = fair.json = 1;
            i--;
            continue;
        }
        if (strcmp(a, "--profile-phases") == 0) {
            fair.enabled = fair.profile_phases = 1;
            i--;
            continue;
        }
        if (strcmp(a, "--fair-retain-controller") == 0) {
            fair.enabled = fair.retain_controller = 1;
            i--;
            continue;
        }
        int dk = janas_devices_option(a, v, why, sizeof why);
        if (dk < 0) {
            fprintf(stderr, "janas-bench: %s\n", why);
            return 2;
        }
        if (dk > 0)
            continue;
        if (strcmp(a, "--speculation") == 0 && v) {
            fair.enabled = 1;
            if (strcmp(v, "none") == 0)
                fair.speculation = JANAS_LLM_SPEC_NONE;
            else if (strcmp(v, "auto") == 0)
                fair.speculation = JANAS_LLM_SPEC_AUTO;
            else if (strcmp(v, "lookup") == 0)
                fair.speculation = JANAS_LLM_SPEC_LOOKUP;
            else if (strcmp(v, "mtp") == 0)
                fair.speculation = JANAS_LLM_SPEC_MTP;
            else if (strcmp(v, "hybrid") == 0)
                fair.speculation = JANAS_LLM_SPEC_HYBRID;
            else
                snprintf(why, sizeof why, "expected none, lookup, mtp, hybrid or auto");
        } else if (strcmp(a, "--prompt-file") == 0 && v) {
            fair.enabled = 1;
            fair.prompt_file = v;
        } else if (strcmp(a, "--draft-tokens") == 0 ||
                   strcmp(a, "--threads") == 0 ||
                   strcmp(a, "--warmups") == 0 || strcmp(a, "--seed") == 0) {
            fair.enabled = 1;
            long long lo = strcmp(a, "--warmups") == 0 ? 0 : 1;
            long long hi = strcmp(a, "--draft-tokens") == 0 ? JANAS_GEN_MAX_K
                           : strcmp(a, "--threads") == 0 ? JANAS_MAX_CPUS
                           : strcmp(a, "--warmups") == 0 ? 1000 : LLONG_MAX;
            if (janas_word_int(v, lo, hi, &x, why, sizeof why) == 0) {
                if (strcmp(a, "--draft-tokens") == 0)
                    fair.drafts = (int)x;
                else if (strcmp(a, "--threads") == 0)
                    fair.threads = (int)x;
                else if (strcmp(a, "--warmups") == 0)
                    fair.warmups = (int)x;
                else
                    fair.seed = (uint64_t)x;
            }
        } else if (strcmp(a, "--mtp") == 0 && v)
            p.mtp_path = v;
        else if (strcmp(a, "--models") == 0 && v)
            models = v;
        else if (strcmp(a, "--prompt") == 0 || strcmp(a, "--gen") == 0 ||
                 strcmp(a, "--ctx") == 0 || strcmp(a, "--rounds") == 0) {
            if (janas_word_int(v, 1, UINT32_MAX, &x, why, sizeof why) == 0) {
                if (a[2] == 'p')
                    n_prompt = (uint32_t)x;
                else if (a[2] == 'g')
                    n_gen = (uint32_t)x;
                else if (a[2] == 'c')
                    p.n_ctx = (uint32_t)x;
                else
                    rounds = x < 1000 ? (int)x : 1000;
            }
        } else if (strcmp(a, "--settle") == 0) {
            janas_word_int(v, 0, 86400, &settle, why, sizeof why);
        } else if (strcmp(a, "--cache") == 0 || strcmp(a, "--reserve") == 0) {
            if (janas_word_real(v, 0, 1 << 20, &r, why, sizeof why) == 0)
                *(a[2] == 'c' ? &p.cache_bytes : &p.reserve_bytes) =
                    (uint64_t)(r * (1 << 30));
        } else {
            janas_unknown_word(why, sizeof why, a, "an option", options,
                               sizeof options / sizeof *options);
            fprintf(stderr, "janas-bench: %s\n", why);
            return 2;
        }
        if (why[0] || !v) {
            fprintf(stderr, "janas-bench: %s: %s\n", a,
                    why[0] ? why : "a value is missing");
            return 2;
        }
    }
    if (n_prompt < 2 || n_gen < 1) {
        fprintf(stderr, "janas-bench: --prompt 2 or more, --gen 1 or more\n");
        return 2;
    }
    /* the generated tokens twice (the drafts' rounds after the plain
       ones) and 64 for the chat's turns: issue #27 met it without
       knowing why */
    if (!fair.enabled && (uint64_t)n_prompt + 2ull * n_gen + 64 > p.n_ctx) {
        fprintf(stderr,
                "janas-bench: --ctx %u is too small for --prompt %u and "
                "--gen %u: it needs %u tokens (the prompt, the generated "
                "tokens twice, 64 more)\n",
                (unsigned)p.n_ctx, (unsigned)n_prompt, (unsigned)n_gen,
                (unsigned)(n_prompt + 2 * n_gen + 64));
        return 2;
    }
    if (no_mtp && p.mtp_path) {
        fprintf(stderr, "janas-bench: --mtp and --no-mtp together\n");
        return 2;
    }
    if (fair.enabled && n_gen > INT32_MAX) {
        fprintf(stderr, "janas-bench: fair --gen must fit in int32\n");
        return 2;
    }
    if (models)
        setenv("JANAS_MODELS", models, 1);
    struct janas_model_pick mp;
    if (janas_model_pick("janas-bench", model, &mp) != 0)
        return 1;
    static char mtp_found[4400];
    if (!p.mtp_path && !no_mtp &&
        janas_model_pick_mtp(&mp, mtp_found, sizeof(mtp_found))) {
        p.mtp_path = mtp_found;
        const char *b = strrchr(mtp_found, '/');
        fprintf(stderr,
                "multi-token prediction from %s, found beside the model "
                "(--no-mtp: without)\n",
                b ? b + 1 : mtp_found);
    }
    if (fair.enabled)
        return fair_bench(mp.path, &p, n_gen, rounds, &fair);

    char path[512], stamp[32];
    time_t tt = time(NULL);
    strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", localtime(&tt));
    snprintf(path, sizeof(path), "janas-bench-%s.txt", stamp);
    report = fopen(path, "w");

    /* the machine */
    char cpu[128];
    struct utsname un;
    struct janas_cpu_layout lay;
    janas_cpu_name(cpu, sizeof(cpu));
    uname(&un);
    out("janas-bench %s, %s\n", JANAS_VERSION, stamp);
    out("system:  %s %s %s\n", un.sysname, un.release, un.machine);
    out("CPU:     %s", cpu);
    if (janas_cpu_layout(NULL, 0, &lay) == 0)
        out(" (%d performance cores, %d threads on them, %d usable in all)",
            lay.n_p_cores, lay.n_p_threads, lay.n);
    char feat[512];
    janas_cpu_features(feat, sizeof(feat));
    out("\nCPU has: %s", feat[0] ? feat : "(not known)");
    out("\nkernels: %s%s", janas_q4k_kernel_name(),
        getenv("JANAS_KERNELS") ? " (JANAS_KERNELS set)" : "");
    int mains = janas_on_mains();
    out("\nmemory:  %.1f GiB, %.1f GiB available; power: %s\n",
        (double)janas_mem_total() / (1 << 30),
        (double)janas_mem_available() / (1 << 30),
        mains < 0 ? "unknown" : (mains ? "mains" : "battery"));

    janas_llm *llm;
    if (janas_llm_open(mp.path, &p, &llm) != JANAS_LLM_OK) {
        out("cannot open the model: %s\n", janas_llm_last_error());
        return 1;
    }
    struct janas_llm_model *m = janas_llm_internal_model(llm);
    struct janas_gpu *gpu = janas_llm_model_gpu(m);
    char desc[1024]; /* 256 cut it short (issue #47) */
    int32_t dl;
    janas_llm_describe(llm, desc, sizeof(desc), &dl);
    if (gpu)
        out("GPU:     %s\n", janas_gpu_name(gpu));
    else
        out("GPU:     none usable: %s\n", janas_llm_model_gpu_why(m));
    const char *npu = janas_llm_model_npu_name(m);
    if (npu)
        out("NPU:     %s\n", npu);
    else
        out("NPU:     none usable: %s\n", janas_llm_model_npu_why(m));
    out("model:   %s\n", desc);
    /* the user's devices, as this machine has them (the engine said on
       stderr what it could not take) */
    int on_p = janas_llm_model_devices(m, 2),
        on_d = janas_llm_model_devices(m, 1),
        on_r = janas_llm_model_devices(m, 0);
    if (on_p || on_d || on_r) {
        char dp[64], dd[64], dr[64];
        janas_devices_name(on_p, dp, sizeof dp);
        janas_devices_name(on_d, dd, sizeof dd);
        janas_devices_name(on_r, dr, sizeof dr);
        out("devices: prompts %s, drafts %s, replies %s\n", dp, dd, dr);
    }

    /* the prompt: the text once, repeated only when more is asked for */
    struct janas_tokenizer *tok = janas_llm_internal_tokenizer(llm);
    size_t tl = strlen(TEXT), reps = n_prompt / 100 + 2;
    int32_t once[1024];
    long n_once = janas_tokenizer_encode(tok, TEXT, tl, 0, once, 1024);
    char *text = malloc(tl * reps + 1);
    for (size_t r = 0; r < reps; r++)
        memcpy(text + r * tl, TEXT, tl);
    text[tl * reps] = 0;
    int32_t *prompt = malloc((tl * reps + 1) * sizeof(int32_t));
    long nt =
        janas_tokenizer_encode(tok, text, tl * reps, 0, prompt, tl * reps + 1);
    if (nt < (long)n_prompt) {
        out("prompt too short\n");
        return 1;
    }
    int has_mtp = janas_llm_model_has_mtp(m);
    struct janas_llm_chat_params cp;
    janas_llm_chat_params_default(&cp);
    out("test:    prompt %u tokens of prose%s, %u generated tokens, greedy",
        n_prompt,
        n_once >= (long)n_prompt ? " (not repeated)"
                                 : " (repeated: the text is shorter)",
        n_gen);
    if (has_mtp)
        out(";\n         with MTP drafts: a chat reply of as many tokens, "
            "greedy and sampled\n         as the chat samples (temperature "
            "%.2g, top-k %d, top-p %.2g), no reasoning",
            cp.temperature, cp.top_k, cp.top_p);
    out("\n\n");

    /* warm-up: the expert cache filled by the same work */
    double d;
    out("warming up the expert cache...\n");
    if (measure(m, prompt, n_prompt, n_gen, NULL, &d) != 0) {
        out("generation failed\n");
        return 1;
    }
    if (settle > 0) {
        /* the filling runs while nobody asks: wait until it has stopped
           (not filling, and no expert more for 3 s: it may not have
           resumed yet when first asked) */
        struct timespec t0, t1, nap = {.tv_sec = 1};
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t last = janas_llm_model_warm_done(m);
        int still = 0;
        do {
            nanosleep(&nap, NULL);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            uint64_t now_in = janas_llm_model_warm_done(m);
            still =
                janas_llm_model_settling(m) || now_in != last ? 0 : still + 1;
            last = now_in;
        } while (still < 3 && t1.tv_sec - t0.tv_sec < settle);
        out("waited %lld s for the cache's filling (%llu experts in)%s\n",
            (long long)(t1.tv_sec - t0.tv_sec), (unsigned long long)last,
            still < 3 ? ", not over" : "");
        if (measure(m, prompt, n_prompt, n_gen, NULL, &d) != 0) {
            out("generation failed\n");
            return 1;
        }
    }

    int th[JANAS_TUNE_MAX_CANDS], gp[JANAS_TUNE_MAX_CANDS],
        np[JANAS_TUNE_MAX_CANDS], al[JANAS_TUNE_MAX_CANDS];
    int nc =
        janas_llm_model_candidates(m, th, gp, np, al, JANAS_TUNE_MAX_CANDS);
    /* the rows: the configurations the engine may choose for some kind of
       pass, every one unless the devices are limited - the NPU's for
       blocks only, the GPU's for replies only where it takes single
       tokens, the CPU alone for blocks only (its replies are the row
       without "+ GPU") */
    int rows[JANAS_TUNE_MAX_CANDS], n_rows = 0, npu_row = -1;
    int gpu_one = gpu && janas_gpu_min_n(gpu) <= 1;
    for (int c = 0; c < nc; c++) {
        rows[c] = !on_p || on_p >> (gp[c] | np[c] << 1) & 1;
        if (!np[c] && !al[c])
            rows[c] |= (!on_d || on_d >> gp[c] & 1) ||
                       ((!gp[c] || gpu_one) && (!on_r || on_r >> gp[c] & 1));
        n_rows += rows[c];
        if (rows[c] && np[c] && npu_row < 0)
            npu_row = c;
    }
    /* the NPU's copies of the weights and its graphs made (and with
       --npu-int8 the scales of the activations taken) before the rows, as
       a chat has them after its first prompt */
    if (npu_row >= 0) {
        out("warming up the NPU...\n");
        janas_llm_model_force(m, npu_row);
        if (measure(m, prompt, n_prompt, n_gen, NULL, &d) != 0) {
            out("generation failed\n");
            return 1;
        }
        janas_llm_model_force(m, -1);
    }
    struct run sum[JANAS_TUNE_MAX_CANDS];
    memset(sum, 0, sizeof(sum));
    int has_cache[JANAS_TUNE_MAX_CANDS] = {0};
    uint64_t drafted[JANAS_TUNE_MAX_CANDS] = {0}, accepted[JANAS_TUNE_MAX_CANDS] = {0},
             drafted_s[JANAS_TUNE_MAX_CANDS] = {0}, accepted_s[JANAS_TUNE_MAX_CANDS] = {0};
    /* several rounds, every other one in reverse order: heat and cache
       drift affect every configuration alike */
    out("measuring %d configurations, %d rounds...\n", n_rows, rounds);
    for (int k = 0; k < rounds * nc; k++) {
        int c = (k / nc) % 2 ? nc - 1 - k % nc : k % nc;
        if (!rows[c])
            continue;
        janas_llm_model_force(m, c);
        struct janas_expert_cache_stats s0, s1;
        janas_expert_cache_stats(janas_llm_model_cache(m), &s0);
        struct run r = {0};
        if (measure(m, prompt, n_prompt, n_gen, &r.prefill, &r.decode) != 0 ||
            (has_mtp && !al[c] &&
             (measure_chat(llm, n_gen, 0, &r.mtp, &drafted[c], &accepted[c]) != 0 ||
              measure_chat(llm, n_gen, 1, &r.mtp_s, &drafted_s[c], &accepted_s[c]) !=
                  0))) {
            out("generation failed\n");
            return 1;
        }
        janas_expert_cache_stats(janas_llm_model_cache(m), &s1);
        uint64_t req = s1.requests - s0.requests;
        r.hits = req ? 100.0 * (double)(s1.hits - s0.hits) / (double)req : 0;
        has_cache[c] |= req > 0;
        sum[c].prefill += r.prefill / rounds;
        sum[c].decode += r.decode / rounds;
        sum[c].mtp += r.mtp / rounds;
        sum[c].mtp_s += r.mtp_s / rounds;
        sum[c].hits += r.hits / rounds;
    }
    out("\n%-22s %12s %12s %12s %12s %9s\n", "configuration", "prefill",
        "decode", has_mtp ? "MTP greedy" : "", has_mtp ? "MTP sampled" : "",
        "cache");
    for (int c = 0; c < nc; c++) {
        const struct run *r = &sum[c];
        char name[64];
        if (!rows[c])
            continue;
        snprintf(name, sizeof(name), "%d threads%s", th[c],
                 al[c]            ? ", CPU alone"
                 : gp[c] && np[c] ? " + GPU + NPU"
                 : gp[c]          ? " + GPU"
                 : np[c]          ? " + NPU"
                                  : "");
        char mt[32] = "", ms[32] = "", dec[32] = "-";
        if (has_mtp && !al[c]) {
            snprintf(mt, sizeof(mt), "%.1f tok/s", r->mtp);
            snprintf(ms, sizeof(ms), "%.1f tok/s", r->mtp_s);
        }
        /* no expert asked for (a dense model): no cache to speak of */
        char hit[16] = "-";
        if (has_cache[c])
            snprintf(hit, sizeof(hit), "%.1f%%", r->hits);
        /* the CPU alone: a block's row; its replies are the row of the
           same threads without "+ GPU" */
        if (!al[c])
            snprintf(dec, sizeof(dec), "%.1f tok/s", r->decode);
        out("%-22s %8.1f tok/s %12s %12s %12s %9s\n", name, r->prefill, dec, mt,
            ms, hit);
    }
    /* the fast prompt runs on the GPU in every row but the CPU alone's:
       the rows without "+ GPU" are not the CPU alone (issue #16 read them
       so, and found the GPU of no help where it read the prompt 5.2 times
       faster) */
    int cpu_rows = 0, alone_rows = 0;
    for (int c = 0; c < nc; c++) {
        cpu_rows += rows[c] && !gp[c] && !al[c];
        alone_rows += rows[c] && al[c];
    }
    /* a card with memory of its own takes every pass whole, in every row
       (issue #54: a "12 threads" row at 834 tokens/s where the CPU alone
       read 152) */
    if (janas_llm_model_gpu_whole(m))
        out("note:    on this card every pass of the model runs whole on "
            "the GPU, in every row%s: the threads feed it and compute a "
            "mixture's experts the card does not hold; "
            "JANAS_GPU=0 measures the CPU alone, JANAS_GPU_WHOLE=0 the "
            "products shared with the CPU\n",
            alone_rows ? " but a prompt's in \"CPU alone\"" : "");
    else if (cpu_rows && janas_llm_model_gpu_prompts(m))
        out("note:    in every row%s the GPU reads a prompt's blocks of %d "
            "tokens or more (the fast prompt); \"+ GPU\" also gives it a "
            "part in the shorter passes%s\n",
            alone_rows ? " but \"CPU alone\"" : "", JANAS_GPU_FAST_N,
            gpu && janas_gpu_min_n(gpu) > 1
                ? ", from 2 tokens on: a reply's single tokens are the "
                  "CPU's in every row"
                : "");
    if (npu_row >= 0)
        out("note:    \"+ NPU\": the NPU computes the products of blocks of "
            "%s tokens or more whose weights stay in memory (attention, "
            "recurrent layers, shared experts, a dense model's "
            "feed-forwards; a dense model's layers whole where nine in ten "
            "fit%s), the CPU%s the "
            "rest%s; its preparation (copies of the weights, graphs: %.1f s "
            "in all) is in a warm-up of its own, not in the rows\n",
            getenv("JANAS_NPU_MIN_N") ? getenv("JANAS_NPU_MIN_N") : "64",
            getenv("JANAS_NPU_A8") && atoi(getenv("JANAS_NPU_A8")) > 0
                ? ", with eight-bit activations"
                : "",
            janas_llm_model_gpu_prompts(m) ? " and the GPU's fast prompt" : "",
            janas_llm_model_gpu_prompts(m)
                ? " (--prompt-on npu for the CPU and the NPU alone)"
                : "",
            janas_llm_model_npu_compile_seconds(m));
    if (npu_row >= 0 && !janas_llm_model_npu_auto(m) && !on_p)
        out("note:    the engine does not give the NPU this model's blocks by "
            "itself: in products on Gemma 4 the NPU's answers measured below "
            "the CPU's against llama.cpp's (26B-A4B, E2B); --prompt-on npu "
            "asks for it\n");
    janas_llm_model_force(m, -1);
    char tune[512];
    int32_t tn;
    janas_llm_set_mode(llm, JANAS_LLM_MODE_MAX);
    janas_llm_tuning(llm, tune, sizeof(tune), &tn);
    /* the result is the row the engine chose for each kind of pass - what
       the chat and the server will run - not the fastest of each column,
       which could mix configurations no pass uses together */
    int row[3];
    for (int k = 0; k < 3; k++) {
        int t = janas_llm_model_threads(m, k),
            g = (janas_llm_model_gpu_used(m) >> k) & 1,
            u = (janas_llm_model_npu_used(m) >> k) & 1,
            a = k == 2 && janas_llm_model_blocks_alone(m);
        row[k] = 0;
        for (int c = 0; c < nc; c++)
            if (rows[c] && th[c] == t && gp[c] == g && np[c] == u && al[c] == a)
                row[k] = c;
    }
    out("\nresult:  prefill %.1f tok/s, decode %.1f tok/s", sum[row[2]].prefill,
        sum[row[0]].decode);
    if (has_mtp)
        out(", with MTP %.1f tok/s greedy, %.1f tok/s sampled", sum[row[1]].mtp,
            sum[row[1]].mtp_s);
    out("\n         (the configuration the engine chose for each kind of "
        "pass, as it runs in a chat)");
    if (has_mtp)
        for (int c = 0; c < nc; c++)
            if (rows[c] && (drafted[c] || drafted_s[c]))
                out("\nMTP row %d: %d threads%s%s, drafts kept %.1f%% greedy "
                    "(%llu/%llu), %.1f%% sampled (%llu/%llu)", c, th[c],
                    gp[c] ? " + GPU" : "", np[c] ? " + NPU" : "",
                    drafted[c] ? 100.0 * (double)accepted[c] / drafted[c] : 0,
                    (unsigned long long)accepted[c], (unsigned long long)drafted[c],
                    drafted_s[c] ? 100.0 * (double)accepted_s[c] / drafted_s[c] : 0,
                    (unsigned long long)accepted_s[c], (unsigned long long)drafted_s[c]);
    out("\nchoices: %s\n", tune);
    /* what the card did: a tester's report has to tell, without the card
       here, where a discrete GPU's time went */
    char gr[8192];
    janas_llm_model_gpu_report(m, gr, sizeof(gr));
    if (gr[0])
        out("%s", gr);
    janas_llm_close(llm); /* saves the tuning profile */
    if (report) {
        fclose(report);
        printf("\nreport written to %s\n", path);
    }
    free(text);
    free(prompt);
    return 0;
}
