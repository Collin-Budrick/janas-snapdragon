/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_LLM_SPEC_CONFIG_H
#define JANAS_LLM_SPEC_CONFIG_H
#include <stddef.h>
#include "janas/llm.h"
#define JANAS_SPEC_RUNTIME_HELP \
    "  --spec-planner M    legacy, cycle or fixed (default legacy)\n" \
    "  --spec-explore N    exploratory cycles, 0..100 (cycle default 20)\n" \
    "  --draft-vocab M     legacy or adaptive (default legacy)\n" \
    "  --draft-vocab-rows N  0 auto, 8192, 16384 or 32768\n"
int janas_spec_runtime_is_option(const char *name);
/* 1 handled, 0 not a runtime option, -1 invalid (environment unchanged). */
int janas_spec_runtime_option(const char *name, const char *value,
                              char *error, size_t error_len);
int janas_spec_config_env(struct janas_llm_speculation_config *config,
                          char *error, size_t error_len);
int32_t janas_spec_runtime_apply(janas_llm_chat *chat, int32_t mode, int32_t cap);
#endif
