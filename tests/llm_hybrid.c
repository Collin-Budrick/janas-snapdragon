/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Real-model semantic copy / topic-switch check. No model changes.
   llm_hybrid model.jns mtp.jns [probe=2] [copy_turns=4] [copy_tokens=256]
   Three chats per sampler: plain, calibrated MTP4, cost hybrid8/MTP4.
   All output bytes, token ids and token parts must agree across sources.
   Counters report actual lookup use; no lookup benefit is assumed. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "janas/llm.h"

#define MAX_TURNS 10
#define MAX_TOKENS 512
#define MAX_TEXT 65536

struct result {
    int n, bytes;
    int32_t ids[MAX_TOKENS], parts[MAX_TOKENS];
    char text[MAX_TEXT];
};
static struct result reference[MAX_TURNS];

static int failed(const char *what, int code)
{
    if (code < 0) {
        fprintf(stderr, "%s: %d %s\n", what, code, janas_llm_last_error());
        return 1;
    }
    return 0;
}
static uint64_t hash(const void *data, size_t n)
{
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    const unsigned char *p = data;
    while (n--) { h ^= *p++; h *= UINT64_C(0x100000001b3); }
    return h;
}
static void json_string(const char *s, int n)
{
    putchar('"');
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if (c < 32) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}
static int same(const struct result *a, const struct result *b)
{
    return a->n == b->n && a->bytes == b->bytes &&
           !memcmp(a->text, b->text, (size_t)a->bytes) &&
           !memcmp(a->ids, b->ids, (size_t)a->n * sizeof(a->ids[0])) &&
           !memcmp(a->parts, b->parts, (size_t)a->n * sizeof(a->parts[0]));
}
static int reply(janas_llm_chat *chat, struct result *out)
{
    char piece[4096];
    memset(out, 0, sizeof(*out));
    for (;;) {
        int32_t n = 0;
        int rc = janas_llm_chat_next(chat, piece, sizeof(piece), &n);
        if (failed("next", rc) || n < 0 || out->bytes > MAX_TEXT - n) return 0;
        memcpy(out->text + out->bytes, piece, (size_t)n);
        out->bytes += n;
        if (rc == JANAS_LLM_DONE) break;
    }
    out->n = janas_llm_chat_tokens(chat);
    if (out->n < 1 || out->n > MAX_TOKENS) return 0;
    for (int i = 0; i < out->n; i++) {
        int32_t len;
        int rc = janas_llm_chat_token(chat, i, &out->ids[i], &out->parts[i],
                                     NULL, piece, sizeof(piece), &len);
        if (failed("token", rc)) return 0;
    }
    return 1;
}
static int run(janas_llm *llm, int sampler, int mode, int probe,
               int copy_turns, int copy_tokens, uint64_t *lookup, uint64_t *fallback)
{
    const char *names[] = {"plain", "mtp4", "hybrid-cost8"};
    const char *first =
        "Copy the following line exactly, repeatedly, for 100 lines. "
        "Do not number the lines, add explanations or use a code fence.\n"
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n";
    const char *again =
        "Continue copying exactly the same line for another 100 lines. "
        "Output only the repeated line, without numbering or explanations.";
    const char *different =
        "Stop copying. Change subject completely. In clear prose, explain why "
        "ocean tides vary through the lunar month and how a bay changes their height. "
        "Do not repeat the preceding words or use a list.";
    const char *resume =
        "Return to the earlier copying task. Copy this exact line 100 times, "
        "without numbering or explanations:\n"
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n";
    struct janas_llm_chat_params params;
    janas_llm_chat_params_default(&params);
    params.temperature = sampler ? 0.7f : 0;
    params.top_k = 0; params.top_p = sampler ? 0.95f : 1;
    params.seed = 20260926; params.thinking = 0; params.recap = 0;
    params.max_reply = copy_tokens;
    params.speculate = mode == 0 ? JANAS_LLM_SPEC_NONE :
                       mode == 1 ? JANAS_LLM_SPEC_MTP : JANAS_LLM_SPEC_HYBRID;
    janas_llm_chat *chat = NULL;
    if (failed("chat_create", janas_llm_chat_create(llm, &params, &chat))) return 0;
    struct janas_llm_speculation_config spec;
    janas_llm_speculation_config_default(&spec);
    spec.mode = params.speculate; spec.max_drafts = mode == 2 ? 8 : 4;
    spec.planner = JANAS_LLM_SPEC_PLANNER_LEGACY;
    if (failed("speculation", janas_llm_chat_set_speculation_config(chat, &spec))) goto fail;
    struct janas_llm_hybrid_config config;
    janas_llm_hybrid_config_default(&config);
    config.policy = JANAS_LLM_HYBRID_COST;
    config.lookup_max_drafts = 8; config.mtp_max_drafts = 4; config.probe_percent = probe;
    if (failed("hybrid_config", janas_llm_chat_set_hybrid_config(chat, &config)) ||
        failed("system", janas_llm_chat_system(chat, "Follow the user's instructions precisely.", -1))) goto fail;
    for (int turn = 0; turn < copy_turns + 2; turn++) {
        const char *phase = turn < copy_turns ? "copy" : turn == copy_turns ? "switch" : "return";
        const char *prompt = !turn ? first : turn < copy_turns ? again : turn == copy_turns ? different : resume;
        params.max_reply = turn < copy_turns ? copy_tokens : turn == copy_turns ? 64 : 128;
        if (failed("params", janas_llm_chat_set_params(chat, &params)) ||
            failed("send", janas_llm_chat_send(chat, prompt, -1))) goto fail;
        struct result actual;
        if (!reply(chat, &actual)) goto fail;
        int identity = !mode || same(&reference[turn], &actual);
        if (!mode) reference[turn] = actual;
        struct janas_llm_chat_stats st = {.size = sizeof(st)};
        struct janas_llm_speculation_metrics sm = {.size = sizeof(sm), .version = JANAS_LLM_SPEC_METRICS_VERSION};
        struct janas_llm_hybrid_metrics hm = {.size = sizeof(hm), .version = JANAS_LLM_HYBRID_METRICS_VERSION};
        if (failed("stats", janas_llm_chat_stats(chat, &st)) ||
            failed("spec_metrics", janas_llm_chat_speculation_metrics(chat, &sm)) ||
            failed("hybrid_metrics", janas_llm_chat_hybrid_metrics(chat, &hm))) goto fail;
        if (mode == 2) {
            if (turn != copy_turns) *lookup += hm.lookup_accepted;
            else *fallback += hm.mtp_selected;
        }
        printf("{\"schema\":1,\"kind\":\"semantic\",\"mode\":\"%s\",\"sampler\":\"%s\","
               "\"phase\":\"%s\",\"turn\":%d,\"probe_percent\":%d,\"identity_verified\":%s,"
               "\"output_tokens\":%d,\"output_bytes\":%d,\"output_hash\":\"%016" PRIx64 "\","
               "\"token_hash\":\"%016" PRIx64 "\",\"input_threads\":%d,\"output_threads\":%d,"
               "\"input_seconds\":%.9g,\"output_seconds\":%.9g,\"tokens_per_second\":%.9g,"
               "\"verification_rows\":%" PRIu64 ",\"lookup_selected\":%" PRIu64 ","
               "\"mtp_selected\":%" PRIu64 ",\"probes\":%" PRIu64 ",\"cooldown_skips\":%" PRIu64 ","
               "\"lookup_faster\":%" PRIu64 ",\"lookup_drafted\":%" PRIu64 ",\"lookup_accepted\":%" PRIu64 ","
               "\"lookup_delivered\":%" PRIu64 ",\"mtp_delivered\":%" PRIu64 ",\"output_text\":",
               names[mode], sampler ? "seeded" : "greedy", phase, turn, probe,
               identity ? "true" : "false", actual.n, actual.bytes,
               hash(actual.text, (size_t)actual.bytes), hash(actual.ids, (size_t)actual.n * sizeof(actual.ids[0])),
               st.input_threads, st.output_threads, st.input_seconds, st.output_seconds,
               st.output_seconds > 0 ? st.output_tokens / st.output_seconds : 0,
               sm.verification_rows, hm.lookup_selected, hm.mtp_selected, hm.probes,
               hm.cooldown_skips, hm.reason_count[JANAS_LLM_HYBRID_REASON_LOOKUP_FASTER],
               hm.lookup_drafted, hm.lookup_accepted, hm.lookup_delivered, hm.mtp_delivered);
        json_string(actual.text, actual.bytes); puts("}"); fflush(stdout);
        if (!identity || st.input_threads != 8 || st.output_threads != 8 ||
            st.finish == JANAS_LLM_FINISH_CONTEXT) goto fail;
    }
    janas_llm_chat_destroy(chat);
    return 1;
fail:
    janas_llm_chat_destroy(chat);
    return 0;
}

int main(int argc, char **argv)
{
    int probe = 2, copy_turns = 4, copy_tokens = 256;
    if (argc < 3) {
        fprintf(stderr, "usage: llm_hybrid model.jns mtp.jns [probe=2] [copy_turns=4] [copy_tokens=256]\n");
        return 2;
    }
    for (int i = 3; i < argc; i++) {
        int *target = NULL; const char *value = NULL;
        if (!strncmp(argv[i], "probe=", 6)) { target = &probe; value = argv[i] + 6; }
        else if (!strncmp(argv[i], "copy_turns=", 11)) { target = &copy_turns; value = argv[i] + 11; }
        else if (!strncmp(argv[i], "copy_tokens=", 12)) { target = &copy_tokens; value = argv[i] + 12; }
        else return 2;
        char *end; long parsed = strtol(value, &end, 10);
        if (!*value || *end || parsed < 0 || parsed > 512) return 2;
        *target = (int)parsed;
    }
    if (probe > 100 || copy_turns < 1 || copy_turns > MAX_TURNS - 2 ||
        copy_tokens < 1 || copy_tokens > MAX_TOKENS || copy_turns * copy_tokens > 1280) return 2;
#ifdef _WIN32
    if (!SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)255)) return 1;
#endif
    if (setenv("JANAS_GPU", "0", 1) || setenv("JANAS_NPU", "0", 1) ||
        setenv("JANAS_TUNE", "0", 1) || setenv("JANAS_TUNE_THREADS", "0", 1) ||
        setenv("JANAS_WARM", "0", 1) || setenv("JANAS_MISS_SKIP", "0", 1) ||
        setenv("JANAS_EXPERT_BITS", "6", 1) || setenv("JANAS_HEAD_BITS", "0", 1) ||
        setenv("JANAS_PREFETCH", "0", 1) || setenv("JANAS_PREFETCH_PREFILL", "0", 1) ||
        setenv("JANAS_GDN_CHAIN", "1", 1) || setenv("JANAS_GDN_JOURNAL", "1", 1) ||
        setenv("JANAS_GDN_K_ONLY", "1", 1)) return 1;
    /* Keep component experiments out of this production-policy check. */
    if (unsetenv("JANAS_CPU_TILES") || unsetenv("JANAS_KV") ||
        unsetenv("JANAS_ATTN") || unsetenv("JANAS_KERNELS")) return 1;
    struct janas_llm_params model;
    janas_llm_params_default(&model);
    model.n_ctx = 2048; model.cache_bytes = UINT64_C(32) << 30;
    model.reserve_bytes = UINT64_C(12) << 30; model.mtp_path = argv[2];
    model.no_preload = 1; model.expert_bits = 6; model.head_bits = 0;
    model.min_quality = JANAS_LLM_QUALITY_HIGH;
    janas_llm *llm = NULL;
    if (failed("open", janas_llm_open(argv[1], &model, &llm))) return 1;
    uint64_t lookup = 0, fallback = 0;
    int ok = 1;
    for (int sampler = 0; sampler < 2 && ok; sampler++)
        for (int mode = 0; mode < 3 && ok; mode++)
            ok = run(llm, sampler, mode, probe, copy_turns, copy_tokens, &lookup, &fallback);
    janas_llm_close(llm);
    printf("{\"schema\":1,\"kind\":\"summary\",\"identity_verified\":%s,"
           "\"accepted_copy_lookup_tokens\":%" PRIu64 ",\"switch_mtp_selections\":%" PRIu64 ","
           "\"lookup_exercised\":%s,\"fallback_exercised\":%s}\n",
           ok ? "true" : "false", lookup, fallback, lookup ? "true" : "false", fallback ? "true" : "false");
    fprintf(stderr, "llm_hybrid: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
