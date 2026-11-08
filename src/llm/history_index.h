/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Bounded, session-local lookup of canonical token-history continuations. */
#ifndef JANAS_LLM_HISTORY_INDEX_H
#define JANAS_LLM_HISTORY_INDEX_H

#include <stddef.h>
#include <stdint.h>

#define JANAS_HISTORY_INDEX_MAX_BYTES ((size_t)8 * 1024 * 1024)
#define JANAS_HISTORY_INDEX_MAX_SUFFIX 8
#define JANAS_HISTORY_INDEX_MAX_DRAFT 16

struct janas_history_index;

/* All owned storage, including metadata, fits in min(budget_bytes, 8 MiB).
 * A budget too small for a useful index, or an allocation failure, returns
 * NULL. A bounded tail is retained when the history exceeds its capacity. */
struct janas_history_index *janas_history_index_create(size_t budget_bytes);
void janas_history_index_destroy(struct janas_history_index *index);
size_t janas_history_index_bytes(const struct janas_history_index *index);

/* Supply only canonical tokens, never unverified proposals. base identifies
 * the absolute position of tokens[0]. Append is incremental after checking
 * the retained prefix; truncate, slide and replacement rebuild the index.
 * NULL tokens is valid only for n == 0. Returns 0, or -1 for invalid input
 * (which also clears the index). No allocation occurs during sync/lookup. */
int janas_history_index_sync(struct janas_history_index *index,
                             const int32_t *tokens, size_t n, uint64_t base);

/* Query the same canonical history most recently supplied to sync. Search
 * suffixes from length 8 down to 1; use the latest exact prior occurrence
 * with at least one known continuation token. The current suffix itself is
 * excluded, but an earlier overlapping occurrence is valid. Returns 0 on
 * no match/invalid arguments; matched_suffix is then 0. max_draft is 1..16.
 * Copies at most max_draft tokens into out. */
int janas_history_index_lookup(const struct janas_history_index *index,
                              const int32_t *tokens, size_t n, int max_draft,
                              int32_t *out, int *matched_suffix);

#endif
