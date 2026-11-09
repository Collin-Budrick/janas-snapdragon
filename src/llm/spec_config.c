/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spec_config.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const options[] = {
    "--spec-planner", "--spec-explore", "--draft-vocab", "--draft-vocab-rows",
    "--hybrid-policy", "--lookup-drafts", "--mtp-drafts", "--hybrid-probe-percent"
};
static const char *const variables[] = {
    "JANAS_SPEC_PLANNER", "JANAS_SPEC_EXPLORE", "JANAS_DRAFT_VOCAB_POLICY",
    "JANAS_DRAFT_VOCAB_ROWS", "JANAS_HYBRID_POLICY", "JANAS_LOOKUP_DRAFTS",
    "JANAS_MTP_DRAFTS", "JANAS_HYBRID_PROBE_PERCENT"
};
static int option_id(const char *name)
{
    for (int i = 0; i < 8; i++) if (!strcmp(name, options[i])) return i;
    return -1;
}
int janas_spec_runtime_is_option(const char *name)
{ return name && option_id(name) >= 0; }
static int parse(int id, const char *value, int *out, char *error, size_t len)
{
    if (!value) goto invalid;
    if (id == 0 || id == 2 || id == 4) {
        const char *const planners[] = {"legacy", "cycle", "fixed"};
        const char *const heads[] = {"legacy", "adaptive"};
        const char *const hybrid[] = {"legacy", "cost"};
        int count = id == 0 ? 3 : 2;
        for (int i = 0; i < count; i++)
            if (!strcmp(value, id == 0 ? planners[i] : id == 2 ? heads[i] : hybrid[i])) { *out = i; return 0; }
    } else {
        char *end = NULL;
        errno = 0;
        long n = strtol(value, &end, 10);
        if (!errno && end != value && *end == 0 && n >= 0 &&
            (((id == 1 || id == 7) && n <= 100) ||
              ((id == 5 || id == 6) && n >= 1 && n <= 16) ||
             (id == 3 && (n == 0 || n == 8192 || n == 16384 || n == 32768)))) {
            *out = (int)n; return 0;
        }
    }
invalid:
    snprintf(error, len, "invalid value for %s", options[id]);
    return -1;
}
int janas_spec_runtime_option(const char *name, const char *value,
                              char *error, size_t len)
{
    int id = name ? option_id(name) : -1, parsed;
    if (id < 0) return 0;
    if (parse(id, value, &parsed, error, len) != 0) return -1;
    if (setenv(variables[id], value, 1) != 0) {
        snprintf(error, len, "cannot set %s", options[id]); return -1;
    }
    return 1;
}
int janas_spec_config_env(struct janas_llm_speculation_config *config,
                          char *error, size_t len)
{
    struct janas_llm_speculation_config copy = *config;
    for (int i = 0; i < 4; i++) {
        const char *value = getenv(variables[i]);
        int parsed;
        if (!value) continue;
        if (parse(i, value, &parsed, error, len) != 0) return -1;
        if (i == 0) copy.planner = parsed;
        else if (i == 1) copy.explore_percent = parsed;
        else if (i == 2) copy.draft_vocab_policy = parsed;
        else copy.draft_vocab_rows = (uint32_t)parsed;
    }
    *config = copy;
    return 0;
}
int janas_hybrid_config_env(struct janas_llm_hybrid_config *config,
                             char *error, size_t len)
{
    struct janas_llm_hybrid_config copy = *config;
    for (int i = 4; i < 8; i++) {
        const char *value = getenv(variables[i]);
        int parsed;
        if (!value) continue;
        if (parse(i, value, &parsed, error, len) != 0) return -1;
        if (i == 4) copy.policy = parsed;
        else if (i == 5) copy.lookup_max_drafts = parsed;
        else if (i == 6) copy.mtp_max_drafts = parsed;
        else copy.probe_percent = parsed;
    }
    *config = copy;
    return 0;
}

int32_t janas_spec_runtime_apply(janas_llm_chat *chat, int32_t mode, int32_t cap)
{
    struct janas_llm_speculation_config config;
    janas_llm_speculation_config_default(&config);
    struct janas_llm_hybrid_config hybrid;
    janas_llm_hybrid_config_default(&hybrid);
    char error[160];
    if (janas_spec_config_env(&config, error, sizeof error) != 0 ||
        janas_hybrid_config_env(&hybrid, error, sizeof error) != 0)
        config.version = 0; /* setter provides the public error contract */
    config.mode = mode; config.max_drafts = cap;
    int32_t rc = janas_llm_chat_set_speculation_config(chat, &config);
    return rc == JANAS_LLM_OK ? janas_llm_chat_set_hybrid_config(chat, &hybrid) : rc;
}
