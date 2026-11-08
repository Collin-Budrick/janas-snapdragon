/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Public speculation controls translated into generator options, without
 * model weights or inference. Invalid requests must not change a chat. */
#include <stdio.h>
#include <string.h>

#include "llm/chat.h"

static int failures;
#define CHECK(c, text) do { if (!(c)) { printf("FAIL: %s\n", text); failures++; } } while (0)

int main(void)
{
    struct janas_llm llm = {0};
    struct janas_llm_chat c = {.llm = &llm};
    struct janas_gen_options o;
    janas_llm_chat_params_default(&c.p);
    CHECK(c.p.speculate == 1, "legacy default remains enabled");
    CHECK(janas_api_options(&c, &o) == JANAS_LLM_OK &&
          o.spec_mode == JANAS_SPEC_LOOKUP && o.spec_k == 4,
          "legacy enabled without head uses lookup");
    c.p.speculate = 0;
    CHECK(janas_api_options(&c, &o) == JANAS_LLM_OK && o.spec_mode == JANAS_SPEC_OFF,
          "legacy disabled remains off");
    llm.has_mtp = 1;
    c.p.speculate = 1;
    CHECK(janas_api_options(&c, &o) == JANAS_LLM_OK &&
          o.spec_mode == JANAS_SPEC_MTP && o.spec_k == 8,
          "legacy enabled with head uses MTP");
    CHECK(janas_llm_chat_speculation(&c, JANAS_LLM_SPEC_LOOKUP, 2) == JANAS_LLM_OK &&
          janas_api_options(&c, &o) == JANAS_LLM_OK &&
          o.spec_mode == JANAS_SPEC_LOOKUP && o.spec_k == 2,
          "lookup remains selectable with an MTP head loaded");
    llm.has_mtp = 0;
    CHECK(janas_llm_chat_speculation(&c, JANAS_LLM_SPEC_MTP, 4) == JANAS_LLM_EINVAL &&
          strstr(janas_llm_last_error(), "MTP head") &&
          c.p.speculate == JANAS_LLM_SPEC_LOOKUP && c.max_drafts == 2,
          "missing head fails explicitly and preserves earlier settings");
    CHECK(janas_llm_chat_speculation(&c, 9, 4) == JANAS_LLM_EINVAL &&
          c.p.speculate == JANAS_LLM_SPEC_LOOKUP && c.max_drafts == 2,
          "unknown mode does not mutate a chat");
    CHECK(janas_llm_chat_speculation(&c, JANAS_LLM_SPEC_LOOKUP, 17) == JANAS_LLM_EINVAL &&
          c.max_drafts == 2, "oversized chain rejected");
    CHECK(janas_llm_chat_speculation(&c, JANAS_LLM_SPEC_LOOKUP, -1) == JANAS_LLM_EINVAL &&
          c.max_drafts == 2, "negative chain rejected");
    CHECK(janas_llm_chat_speculation(&c, JANAS_LLM_SPEC_LOOKUP, 0) == JANAS_LLM_OK &&
          janas_api_options(&c, &o) == JANAS_LLM_OK && o.spec_k == 4,
          "zero restores the default ceiling");
    CHECK(janas_llm_chat_speculation(NULL, JANAS_LLM_SPEC_NONE, 0) == JANAS_LLM_EINVAL,
          "null chat rejected");
    if (failures) {
        printf("test_speculation: %d failures\n", failures);
        return 1;
    }
    printf("test_speculation: ok\n");
    return 0;
}
