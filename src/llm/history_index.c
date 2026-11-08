/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "history_index.h"

#include <stdlib.h>
#include <string.h>

/* References point into the retained token copy. Exact token comparisons
 * resolve hash collisions, so hashes never establish a proposal's identity. */
struct history_entry {
    uint32_t hash, end;
    uint8_t length;
    uint8_t padding[3];
};

struct janas_history_index {
    struct history_entry *entries;
    int32_t *tokens;
    size_t bytes, slots, capacity, count, input_n, input_start;
    uint64_t base;
    int ready;
};

static uint32_t suffix_hash(const int32_t *tokens, size_t n)
{
    uint32_t h = UINT32_C(2166136261);
    for (size_t i = 0; i < n; i++) {
        h ^= (uint32_t)tokens[i];
        h *= UINT32_C(16777619);
        h ^= h >> 16;
    }
    return h;
}

static void clear_index(struct janas_history_index *index)
{
    memset(index->entries, 0, index->slots * sizeof(*index->entries));
    index->count = index->input_n = index->input_start = 0;
    index->base = 0;
    index->ready = 0;
}

struct janas_history_index *janas_history_index_create(size_t budget_bytes)
{
    size_t budget = budget_bytes < JANAS_HISTORY_INDEX_MAX_BYTES ?
                    budget_bytes : JANAS_HISTORY_INDEX_MAX_BYTES;
    /* At most 8 suffix keys per token, with a maximum 1/2 table load. */
    size_t slots = 256;
    size_t need = sizeof(struct janas_history_index) +
                  slots * sizeof(struct history_entry) +
                  (slots / 16) * sizeof(int32_t);
    if (need > budget)
        return NULL;
    while (slots <= SIZE_MAX / 2) {
        size_t next = slots * 2;
        size_t next_need = sizeof(struct janas_history_index) +
                           next * sizeof(struct history_entry) +
                           (next / 16) * sizeof(int32_t);
        if (next_need > budget)
            break;
        slots = next;
        need = next_need;
    }
    struct janas_history_index *index = calloc(1, need);
    if (!index)
        return NULL;
    index->bytes = need;
    index->slots = slots;
    index->capacity = slots / 16;
    index->entries = (struct history_entry *)(index + 1);
    index->tokens = (int32_t *)(index->entries + slots);
    return index;
}

void janas_history_index_destroy(struct janas_history_index *index)
{
    free(index);
}

size_t janas_history_index_bytes(const struct janas_history_index *index)
{
    return index ? index->bytes : 0;
}

static void insert_suffix(struct janas_history_index *index, size_t end,
                          size_t length)
{
    const int32_t *key = index->tokens + end - length;
    uint32_t hash = suffix_hash(key, length);
    size_t slot = hash & (index->slots - 1);
    for (size_t probes = 0; probes < index->slots; probes++) {
        struct history_entry *entry = index->entries + slot;
        if (!entry->length ||
            (entry->hash == hash && entry->length == length &&
             memcmp(index->tokens + entry->end - length, key,
                    length * sizeof(*key)) == 0)) {
            entry->hash = hash;
            entry->end = (uint32_t)end;
            entry->length = (uint8_t)length;
            return;
        }
        slot = (slot + 1) & (index->slots - 1);
    }
}

int janas_history_index_sync(struct janas_history_index *index,
                             const int32_t *tokens, size_t n, uint64_t base)
{
    if (!index)
        return -1;
    if ((!tokens && n) || n > UINT64_MAX - base ||
        n > SIZE_MAX / sizeof(*tokens)) {
        clear_index(index);
        return -1;
    }
    size_t start = n > index->capacity ? n - index->capacity : 0;
    size_t count = n - start;
    int append = index->ready && base == index->base &&
                 n >= index->input_n && start == index->input_start &&
                 (!index->count ||
                  memcmp(tokens + start, index->tokens,
                         index->count * sizeof(*tokens)) == 0);
    size_t first_end;
    if (append) {
        first_end = index->count ? index->count : 1;
        if (count > index->count)
            memcpy(index->tokens + index->count, tokens + start + index->count,
                   (count - index->count) * sizeof(*tokens));
    } else {
        memset(index->entries, 0, index->slots * sizeof(*index->entries));
        if (count)
            memcpy(index->tokens, tokens + start, count * sizeof(*tokens));
        first_end = 1;
    }
    /* Excluding end == count prevents the current suffix matching itself.
     * Append inserts the old terminal suffix once its continuation exists. */
    for (size_t end = first_end; end < count; end++) {
        size_t max_length = end < JANAS_HISTORY_INDEX_MAX_SUFFIX ?
                            end : JANAS_HISTORY_INDEX_MAX_SUFFIX;
        for (size_t length = 1; length <= max_length; length++)
            insert_suffix(index, end, length);
    }
    index->count = count;
    index->input_n = n;
    index->input_start = start;
    index->base = base;
    index->ready = 1;
    return 0;
}

int janas_history_index_lookup(const struct janas_history_index *index,
                              const int32_t *tokens, size_t n, int max_draft,
                              int32_t *out, int *matched_suffix)
{
    if (matched_suffix)
        *matched_suffix = 0;
    if (!index || !index->ready || !tokens || !out ||
        n != index->input_n || !index->count || max_draft < 1 ||
        max_draft > JANAS_HISTORY_INDEX_MAX_DRAFT)
        return 0;
    /* Do not return data from a stale snapshot if the caller replaced its
     * canonical history without syncing it first. */
    if (memcmp(tokens + index->input_start, index->tokens,
               index->count * sizeof(*tokens)) != 0)
        return 0;
    size_t length = index->count < JANAS_HISTORY_INDEX_MAX_SUFFIX ?
                    index->count : JANAS_HISTORY_INDEX_MAX_SUFFIX;
    for (; length; length--) {
        const int32_t *key = index->tokens + index->count - length;
        uint32_t hash = suffix_hash(key, length);
        size_t slot = hash & (index->slots - 1);
        for (size_t probes = 0; probes < index->slots; probes++) {
            const struct history_entry *entry = index->entries + slot;
            if (!entry->length)
                break;
            if (entry->hash == hash && entry->length == length &&
                entry->end < index->count &&
                memcmp(index->tokens + entry->end - length, key,
                       length * sizeof(*key)) == 0) {
                size_t got = index->count - entry->end;
                if (got > (size_t)max_draft)
                    got = (size_t)max_draft;
                memcpy(out, index->tokens + entry->end, got * sizeof(*out));
                if (matched_suffix)
                    *matched_suffix = (int)length;
                return (int)got;
            }
            slot = (slot + 1) & (index->slots - 1);
        }
    }
    return 0;
}
