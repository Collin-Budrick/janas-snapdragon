/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/history_index.h"

/* Local ownership/failure checks exercise the implementation without changing
 * its API or asking the host allocator to exhaust memory. The normal archive
 * linker does not pull a second history_index.o into this test. */
static unsigned allocation_calls;
static size_t last_allocation_bytes;
static int fail_allocation;
static void *index_test_calloc(size_t n, size_t bytes)
{
    allocation_calls++;
    if (bytes && n > SIZE_MAX / bytes)
        return NULL;
    last_allocation_bytes = n * bytes;
    return fail_allocation ? NULL : calloc(n, bytes);
}
#define calloc index_test_calloc
#include "../src/llm/history_index.c"
#undef calloc

static int failures;
#define CHECK(c, text) do { if (!(c)) { \
    printf("FAIL: %s (line %d)\n", text, __LINE__); failures++; \
} } while (0)

static int scan(const int32_t *tokens, size_t n, int cap,
                int32_t *out, int *prefix)
{
    *prefix = 0;
    for (size_t len = n < 8 ? n : 8; len; len--) {
        for (size_t end = n; end-- > len;) {
            if (memcmp(tokens + end - len, tokens + n - len,
                       len * sizeof(*tokens)))
                continue;
            size_t count = n - end;
            if (count > (size_t)cap)
                count = (size_t)cap;
            memcpy(out, tokens + end, count * sizeof(*out));
            *prefix = (int)len;
            return (int)count;
        }
    }
    return 0;
}

static void compare(struct janas_history_index *index, const int32_t *tokens,
                    size_t n, uint64_t base, const char *text)
{
    int32_t want[16] = {0}, got[16] = {0};
    int wp, gp;
    CHECK(janas_history_index_sync(index, tokens, n, base) == 0, text);
    int wn = scan(tokens, n, 16, want, &wp);
    int gn = janas_history_index_lookup(index, tokens, n, 16, got, &gp);
    CHECK(gn == wn && gp == wp &&
          memcmp(got, want, (size_t)wn * sizeof(*want)) == 0, text);
}

static void context_allocations(void)
{
    unsigned before = allocation_calls;
    CHECK(janas_history_index_create_for_tokens(SIZE_MAX, 0) == NULL &&
          janas_history_index_create_for_tokens(0, 2048) == NULL &&
          janas_history_index_create_for_tokens(64, 2048) == NULL &&
          allocation_calls == before,
          "zero tokens and unusable budgets fail before allocation");
    struct janas_history_index *minimum =
        janas_history_index_create_for_tokens(JANAS_HISTORY_INDEX_MAX_BYTES, 1);
    struct janas_history_index *context =
        janas_history_index_create_for_tokens(JANAS_HISTORY_INDEX_MAX_BYTES, 2048);
    struct janas_history_index *rounded =
        janas_history_index_create_for_tokens(JANAS_HISTORY_INDEX_MAX_BYTES, 2049);
    struct janas_history_index *legacy = janas_history_index_create(JANAS_HISTORY_INDEX_MAX_BYTES);
    CHECK(minimum && context && rounded && legacy, "context allocation fixtures created");
    if (!minimum || !context || !rounded || !legacy) goto done;
    CHECK(minimum->slots == 256 && minimum->capacity == 16,
          "tiny positive requests keep the minimum useful table");
    CHECK(context->capacity == 2048 && context->slots == 32768 &&
          context->bytes > 384u * 1024u && context->bytes < 400u * 1024u &&
          context->bytes < legacy->bytes / 15,
          "2048-token context owns about 392 KiB instead of the maximum table");
    CHECK(rounded->capacity == 4096 && rounded->slots == context->slots * 2 &&
          rounded->bytes <= legacy->bytes,
          "2049-token request rounds to the next permitted capacity");
    size_t exact = context->bytes;
    struct janas_history_index *fit = janas_history_index_create_for_tokens(exact, 2048);
    struct janas_history_index *less = janas_history_index_create_for_tokens(exact - 1, 2048);
    struct janas_history_index *limited = janas_history_index_create_for_tokens(exact, 2049);
    CHECK(fit && less && limited && fit->capacity == 2048 && fit->bytes == exact &&
          less->capacity == 1024 && less->bytes < exact && limited->capacity == 2048,
          "insufficient budget chooses the largest affordable bounded tail");
    janas_history_index_destroy(fit); janas_history_index_destroy(less);
    janas_history_index_destroy(limited);
    const size_t budgets[] = {64, 4096, exact - 1, exact,
                             JANAS_HISTORY_INDEX_MAX_BYTES, SIZE_MAX};
    for (size_t i = 0; i < sizeof(budgets) / sizeof(*budgets); i++) {
        struct janas_history_index *old = janas_history_index_create(budgets[i]);
        struct janas_history_index *huge = janas_history_index_create_for_tokens(budgets[i], SIZE_MAX);
        CHECK((!old && !huge) || (old && huge && old->bytes == huge->bytes &&
              old->capacity == huge->capacity && old->slots == huge->slots &&
              old->bytes <= JANAS_HISTORY_INDEX_MAX_BYTES),
              "SIZE_MAX request safely retains exact legacy budget behavior");
        janas_history_index_destroy(old); janas_history_index_destroy(huge);
    }
    size_t ignored;
    CHECK(allocation_bytes(SIZE_MAX, &ignored) == -1 &&
          allocation_bytes((SIZE_MAX - sizeof(struct janas_history_index)) /
                           sizeof(struct history_entry), &ignored) == -1,
          "both entry and token-storage size additions reject overflow");
    before = allocation_calls;
    fail_allocation = 1;
    struct janas_history_index *failed =
        janas_history_index_create_for_tokens(SIZE_MAX, 2048);
    CHECK(failed == NULL && allocation_calls == before + 1 &&
          last_allocation_bytes == exact,
          "context allocation failure returns NULL after one bounded request");
    CHECK(janas_history_index_create(4096) == NULL,
          "legacy allocation failure still returns NULL");
    fail_allocation = 0;
    failed = janas_history_index_create_for_tokens(exact, 2048);
    CHECK(failed && failed->bytes == exact && failed->capacity == 2048,
          "allocation failure leaves no persistent state preventing retry");
    janas_history_index_destroy(failed);
done:
    janas_history_index_destroy(minimum); janas_history_index_destroy(context);
    janas_history_index_destroy(rounded); janas_history_index_destroy(legacy);
}

static void context_lookup_boundaries(void)
{
    struct janas_history_index *context =
        janas_history_index_create_for_tokens(JANAS_HISTORY_INDEX_MAX_BYTES, 2048);
    struct janas_history_index *rounded =
        janas_history_index_create_for_tokens(JANAS_HISTORY_INDEX_MAX_BYTES, 2049);
    struct janas_history_index *legacy = janas_history_index_create(JANAS_HISTORY_INDEX_MAX_BYTES);
    CHECK(context && rounded && legacy, "bounded-context lookup fixtures created");
    if (!context || !rounded || !legacy) goto done;
    int32_t hist[2049], rejected[2048], out[16];
    int prefix;
    for (size_t n = 0; n <= 2048; n++) {
        if (n) hist[n - 1] = (int32_t)((n * 17 + n / 11) % 53);
        compare(context, hist, n, 0, "context-sized incremental history matches exact scan");
        compare(legacy, hist, n, 0, "legacy creator matches the same context-sized scan");
    }
    memcpy(rejected, hist, sizeof(rejected));
    for (size_t n = 2040; n < 2048; n++) rejected[n] = (int32_t)(900000 + n);
    CHECK(janas_history_index_lookup(context, rejected, 2048, 16, out, &prefix) == 0 &&
          janas_history_index_lookup(legacy, rejected, 2048, 16, out, &prefix) == 0,
          "unverified replacement suffix cannot query a canonical-history index");
    compare(context, hist, 2048, 0, "rejected suffix leaves canonical context lookup unchanged");
    compare(legacy, hist, 2048, 0, "legacy canonical lookup agrees after rejected suffix");
    compare(context, rejected, 2048, 0, "replacement snapshot sync is exact within context");
    compare(context, hist, 2048, 0, "restoring canonical history removes replacement snapshot keys");
    compare(legacy, rejected, 2048, 0, "legacy replacement snapshot sync is exact");
    compare(legacy, hist, 2048, 0, "legacy restored snapshot agrees with bounded context");
    const size_t lengths[] = {2047, 1024, 17, 8, 1, 0, 2048};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        compare(context, hist, lengths[i], 0, "context truncation/resync matches exact scan");
        compare(legacy, hist, lengths[i], 0, "legacy truncation/resync matches the same scan");
    }
    compare(context, hist + 37, 2011, 37, "context-sized sliding history matches scan");
    compare(legacy, hist + 37, 2011, 37, "legacy sliding history matches the same scan");
    /* Beyond the configured window, the new creator intentionally retains a
       bounded tail. A request of 2049 must retain this first occurrence; a
       capacity of 2048 must exclude it. */
    for (size_t i = 0; i < 2048; i++) hist[i] = (int32_t)(10000 + i);
    hist[2048] = hist[0];
    compare(legacy, hist, 2049, 0, "legacy full history includes boundary occurrence");
    compare(rounded, hist, 2049, 0, "rounded 2049-token request retains the full exact match");
    CHECK(janas_history_index_sync(context, hist, 2049, 0) == 0 &&
          janas_history_index_lookup(context, hist, 2049, 16, out, &prefix) == 0 &&
          prefix == 0,
          "2048-token capacity excludes an occurrence just outside its retained tail");
    compare(context, hist + 1, 2048, 1, "sliding to that same retained tail remains exact");
    compare(context, hist, 2048, 0, "context boundary can be resynchronized after tail sliding");
done:
    janas_history_index_destroy(context); janas_history_index_destroy(rounded);
    janas_history_index_destroy(legacy);
}

int main(void)
{
    context_allocations();
    context_lookup_boundaries();
    CHECK(janas_history_index_create(0) == NULL, "zero budget rejected");
    CHECK(janas_history_index_create(64) == NULL, "tiny budget rejected");
    CHECK(janas_history_index_bytes(NULL) == 0, "NULL owns no bytes");
    struct janas_history_index *index =
        janas_history_index_create(JANAS_HISTORY_INDEX_MAX_BYTES + 1);
    CHECK(index != NULL, "default allocation succeeds");
    if (!index)
        return 1;
    CHECK(janas_history_index_bytes(index) <= JANAS_HISTORY_INDEX_MAX_BYTES,
          "all storage fits 8 MiB");
    int32_t hist[512], out[16];
    int prefix;
    CHECK(janas_history_index_sync(index, NULL, 0, 0) == 0,
          "empty history valid");
    CHECK(janas_history_index_lookup(index, NULL, 0, 16, out, &prefix) == 0 &&
          prefix == 0, "empty history has no proposal");
    for (int len = 1; len <= 8; len++) {
        size_t n = 0;
        for (int i = 0; i < len; i++)
            hist[n++] = 100 + i;
        hist[n++] = 900;
        hist[n++] = 901;
        for (int i = 0; i < len; i++)
            hist[n++] = 100 + i;
        compare(index, hist, n, 0, "suffix length 1..8 exact");
        CHECK(janas_history_index_lookup(index, hist, n, 16, out, &prefix) >= 2 &&
              prefix == len && out[0] == 900 && out[1] == 901,
              "longest suffix chosen");
    }
    int32_t repeated[] = {1, 2, 3, 8, 1, 2, 3, 9, 1, 2, 3};
    compare(index, repeated, sizeof(repeated) / sizeof(*repeated), 0,
            "latest occurrence selected");
    out[1] = -12345;
    CHECK(janas_history_index_lookup(index, repeated, 11, 1, out, &prefix) == 1 &&
          out[0] == 9 && out[1] == -12345 && prefix == 3,
          "latest continuation and exact write cap");
    /* These distinct length-2 keys have the same 32-bit hash. Neither may
     * overwrite the other's latest occurrence or establish an inexact hit. */
    int32_t collision[] = {100, 101, 700, 200, -1408781466, 900, 100, 101};
    compare(index, collision, 8, 0, "full hash collision resolved exactly");
    CHECK(janas_history_index_lookup(index, collision, 8, 16, out, &prefix) > 0 &&
          prefix == 2 && out[0] == 700, "collision preserves exact continuation");
    for (size_t n = 0; n <= 512; n++) {
        if (n)
            hist[n - 1] = (int32_t)((n * 17 + n / 11) % 23);
        compare(index, hist, n, 0, "incremental append matches scan");
    }
    for (size_t n = 512; n > 0; n -= n > 19 ? 19 : n)
        compare(index, hist, n, 0, "truncate matches scan");
    compare(index, hist + 37, 400, 37, "slide matches scan");
    for (size_t n = 0; n < 512; n++)
        hist[n] = (int32_t)(n % 7);
    compare(index, hist, 512, 0, "load replacement matches scan");
    hist[10] ^= 123;
    CHECK(janas_history_index_lookup(index, hist, 512, 16, out, &prefix) == 0,
          "unsynced changed history rejected");
    compare(index, hist, 512, 0, "same length changed history rebuilds");
    int32_t arbitrary[] = {INT32_MIN, INT32_MAX, -1, INT32_MIN, INT32_MAX};
    compare(index, arbitrary, 5, 0, "arbitrary token bit patterns safe");
    CHECK(janas_history_index_lookup(index, arbitrary, 5, 0, out, &prefix) == 0 &&
          janas_history_index_lookup(index, arbitrary, 5, 17, out, &prefix) == 0,
          "invalid draft bounds rejected");
    CHECK(janas_history_index_sync(index, NULL, 1, 0) == -1 &&
          janas_history_index_lookup(index, arbitrary, 5, 16, out, &prefix) == 0,
          "invalid sync clears old results");
    CHECK(janas_history_index_sync(index, arbitrary, 5, UINT64_MAX - 3) == -1,
          "absolute position overflow rejected");
    CHECK(janas_history_index_sync(NULL, arbitrary, 5, 0) == -1,
          "NULL index rejected");
    CHECK(janas_history_index_lookup(NULL, arbitrary, 5, 16, out, &prefix) == 0,
          "NULL lookup rejected");
    /* Exercise the bounded-tail rebuild path with the minimum useful budget.
     * Its 16-token tail must behave exactly like a scan over that tail. */
    struct janas_history_index *small = janas_history_index_create(4096);
    CHECK(small != NULL && janas_history_index_bytes(small) <= 4096,
          "small index stays within explicit budget");
    if (small) {
        for (size_t n = 32; n <= 512; n++) {
            int32_t expected[16] = {0};
            int ep;
            CHECK(janas_history_index_sync(small, hist, n, 0) == 0,
                  "bounded tail sync");
            int en = scan(hist + n - 16, 16, 16, expected, &ep);
            int gn = janas_history_index_lookup(small, hist, n, 16, out, &prefix);
            CHECK(en == gn && ep == prefix &&
                  memcmp(out, expected, (size_t)en * sizeof(*out)) == 0,
                  "bounded tail matches retained scan");
        }
        janas_history_index_destroy(small);
    }
    janas_history_index_destroy(index);
    janas_history_index_destroy(NULL);
    printf("test_history_index: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
