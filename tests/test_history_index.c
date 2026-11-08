/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/history_index.h"

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

int main(void)
{
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
