/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_LLM_SPEC_JSON_H
#define JANAS_LLM_SPEC_JSON_H
#include "janas/llm.h"
#include "json.h"
/* One object only; callers own the key, comma and outer envelope. */
void janas_spec_metrics_json(struct janas_buf *out,
                              const struct janas_llm_speculation_metrics *m);
#endif
