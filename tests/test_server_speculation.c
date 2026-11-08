/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/* Per-job speculation JSON, without a model, HTTP server or live chat. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "server/chat_out.c"

/* These paths are not used by the response fixture. */
void srv_job_release(struct srv_job *j) { (void)j; abort(); }
int srv_store_put(struct srv_store *st, const char *kind, const char *id,
                  int64_t created, const char *json, const char *aux)
{
    (void)st, (void)kind, (void)id, (void)created, (void)json, (void)aux;
    abort();
}

static int failures;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static yyjson_doc *usage(struct srv_job *j)
{
    yyjson_mut_doc *d = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(d);
    yyjson_mut_doc_set_root(d, root);
    out_usage(d, root, j);
    /* Writing must use the job snapshot already copied into the JSON doc,
       even if a later choice or request now changes the source counters. */
    uint64_t saved = j->speculation.cycles;
    j->speculation.cycles = 0;
    size_t n = 0;
    char *text = yyjson_mut_write(d, 0, &n);
    yyjson_mut_doc_free(d);
    yyjson_doc *parsed = text ? yyjson_read(text, n, 0) : NULL;
    free(text);
    j->speculation.cycles = saved;
    CHECK(parsed);
    return parsed;
}

static yyjson_val *get_usage(yyjson_doc *d)
{
    return yyjson_obj_get(yyjson_doc_get_root(d), "usage");
}

int main(void)
{
    struct srv_job j = {0};
    j.prompt_tokens = 23, j.completion_tokens = 7, j.reasoning_tokens = 2;
    j.st.cached_tokens = 3, j.st.context_used = 30, j.st.context_size = 1024;
    j.st.input_seconds = 0.5, j.st.output_seconds = 1.25;
    j.speculation.size = sizeof(j.speculation);
    j.speculation.version = JANAS_LLM_SPEC_METRICS_VERSION;
    j.speculation.cycles = UINT64_C(9007199254740993);
    j.speculation.verification_rows = 17;
    j.speculation.gdn_journal_replay_tokens = 4;
    j.speculation.gdn_delta_bytes = 9 * 1024 * 1024;
    j.speculation.gdn_core_seconds = 0.25;
    j.speculation.last.accepted_prefix = 4;
    j.speculation.source_cycles[2] = 9;
    j.speculation_valid = 1;
    yyjson_doc *d = usage(&j);
    if (d) {
        yyjson_val *u = get_usage(d), *jn = yyjson_obj_get(u, "janas");
        yyjson_val *m = yyjson_obj_get(jn, "speculation");
        CHECK(yyjson_get_uint(yyjson_obj_get(u, "prompt_tokens")) == 23);
        CHECK(yyjson_get_uint(yyjson_obj_get(u, "completion_tokens")) == 7);
        CHECK(yyjson_get_uint(yyjson_obj_get(u, "total_tokens")) == 30);
        CHECK(yyjson_get_uint(yyjson_obj_get(jn, "context_used")) == 30);
        CHECK(yyjson_get_real(yyjson_obj_get(jn, "output_seconds")) == 1.25);
        CHECK(yyjson_is_obj(m));
        CHECK(yyjson_get_uint(yyjson_obj_get(m, "schema")) == 1);
        CHECK(yyjson_get_uint(yyjson_obj_get(m, "cycles")) ==
              UINT64_C(9007199254740993));
        CHECK(yyjson_get_uint(yyjson_obj_get(m, "verification_rows")) == 17);
        CHECK(yyjson_get_uint(yyjson_obj_get(m, "gdn_journal_replay_tokens")) == 4);
        CHECK(yyjson_get_real(yyjson_obj_get(m, "gdn_core_seconds")) == 0.25);
        CHECK(yyjson_get_uint(yyjson_arr_get(yyjson_obj_get(m, "source_cycles"), 2)) == 9);
        CHECK(yyjson_get_uint(yyjson_obj_get(yyjson_obj_get(m, "last"),
                                          "accepted_prefix")) == 4);
        CHECK(yyjson_equals_str(yyjson_obj_get(m, "reply_scope"), "first_choice"));
        CHECK(yyjson_is_str(yyjson_obj_get(m, "cycle_scope")));
        CHECK(yyjson_is_str(yyjson_obj_get(m, "expert_scope")));
        yyjson_doc_free(d);
    }
    j.speculation_valid = 0;
    d = usage(&j);
    if (d) {
        CHECK(!yyjson_obj_get(yyjson_obj_get(get_usage(d), "janas"), "speculation"));
        CHECK(yyjson_get_uint(yyjson_obj_get(get_usage(d), "total_tokens")) == 30);
        yyjson_doc_free(d);
    }
    /* Invalid optional diagnostics must not make the response invalid JSON. */
    j.speculation_valid = 1;
    j.speculation.gdn_core_seconds = NAN;
    d = usage(&j);
    if (d) {
        CHECK(!yyjson_obj_get(yyjson_obj_get(get_usage(d), "janas"), "speculation"));
        yyjson_doc_free(d);
    }
    if (failures) return 1;
    puts("test_server_speculation: snapshot/object/scopes/legacy usage ok");
    return 0;
}
