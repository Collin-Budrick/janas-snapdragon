"""Evaluator failure/exit-status regressions with a four-logit mock model.

Run with Python 3 and a host C compiler, or set LLM_EVAL_FIXTURE_EXE to the
same mock linked with tests/llm_eval.c for the native target. No model is read.
"""

import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOCK = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "janas/llm.h"
#include "llm/model.h"
struct janas_llm_model { unsigned calls; };
static struct janas_llm_model model;
static int mode(const char *s) {
    const char *v = getenv("EVAL_FIXTURE_MODE");
    return v && !strcmp(v, s);
}
struct janas_llm_model *janas_llm_model_load(
    const char *path, const struct janas_llm_options *o, char *err, size_t n) {
    (void)path;
    if (o->n_compute != 8 || o->n_io != 4 ||
        o->compute_cpus[7] != 7 || o->io_cpus[0] != 8 ||
        o->expert_bits != 6 || o->head_bits != 0 ||
        o->min_quality != JANAS_LLM_QUALITY_HIGH) {
        snprintf(err, n, "unexpected fixture options");
        return NULL;
    }
    model.calls = 0;
    return &model;
}
void janas_llm_model_free(struct janas_llm_model *m) { (void)m; }
uint32_t janas_llm_model_n_vocab(const struct janas_llm_model *m) {
    (void)m; return mode("zero_vocab") ? 0 : 4;
}
uint32_t janas_llm_model_max_block(const struct janas_llm_model *m) {
    (void)m; return 64;
}
static void row(int32_t t, uint32_t p, float *out) {
    out[0] = 0.0f; out[1] = (float)t; out[2] = (float)p; out[3] = 10.0f;
}
int janas_llm_model_decode(struct janas_llm_model *m, int32_t t,
                            uint32_t p, float *out) {
    ++m->calls;
    if ((mode("decode_fail") && p == 2) ||
        (mode("self_fail") && m->calls > 7)) return -1;
    row(t, p, out);
    if ((mode("decode_nan") && p == 2) ||
        (mode("self_nan") && m->calls > 7)) out[1] = NAN;
    return 0;
}
int janas_llm_model_forward(struct janas_llm_model *m, const int32_t *t,
    uint32_t n, uint32_t p, float *out, int all) {
    (void)m; (void)all;
    if (mode("forward_fail")) return -1;
    for (uint32_t i = 0; i < n; ++i) row(t[i], p + i, out + 4 * i);
    if (mode("block_diff")) out[1] += 1;
    if (mode("rejected_diff")) out[4 * (n - 1) + 1] += 1;
    if (mode("replay_diff") && p) out[1] += 1;
    if (mode("block_inf")) out[1] = INFINITY;
    if (mode("signed_zero")) out[0] = -0.0f;
    return 0;
}
struct janas_expert_cache *janas_llm_model_cache(struct janas_llm_model *m) {
    return (struct janas_expert_cache *)m;
}
void janas_expert_cache_stats(const struct janas_expert_cache *c,
                              struct janas_expert_cache_stats *out) {
    (void)c; memset(out, 0, sizeof(*out));
}
void janas_llm_model_phases(const struct janas_llm_model *m,
                            double out[JANAS_PH_COUNT]) {
    (void)m; memset(out, 0, JANAS_PH_COUNT * sizeof(*out));
}
'''


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="janas-eval-fixture-")
        cls.directory = Path(cls.temp.name)
        cls.exe = Path(os.environ.get("LLM_EVAL_FIXTURE_EXE", ""))
        if not os.environ.get("LLM_EVAL_FIXTURE_EXE"):
            mock = cls.directory / "mock.c"
            mock.write_text(MOCK, encoding="utf-8")
            cls.exe = cls.directory / ("llm_eval.exe" if os.name == "nt" else "llm_eval")
            built = subprocess.run(
                shlex.split(os.environ.get("CC", "cc"))
                + ["-std=c11", "-D_POSIX_C_SOURCE=200809L", "-Wall", "-Wextra",
                   "-Werror", "-I" + str(ROOT / "src"), "-I" + str(ROOT / "include"),
                   str(ROOT / "tests/llm_eval.c"), str(mock), "-lm", "-o", str(cls.exe)],
                capture_output=True, text=True,
            )
            if built.returncode:
                raise RuntimeError("mock evaluator compilation failed:\n" + built.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.tokens = self.directory / "tokens.int32"
        self.tokens.write_bytes(struct.pack("=7i", *range(7)))
        self.ref = self.directory / "reference.f32"
        self.ref.write_bytes(b"".join(struct.pack("=4f", 0, p, p, 10) for p in range(7)))

    def run_eval(self, *args, mode="", token_path=None):
        env = os.environ.copy()
        env["EVAL_FIXTURE_MODE"] = mode
        return subprocess.run(
            [str(self.exe), "fixture.jns", str(token_path or self.tokens), *map(str, args)],
            capture_output=True, text=True, env=env, timeout=10,
        )

    def test_valid_plain_and_overlapping_short_tail(self):
        for args in [(), ("block=4",), ("block=4", "reject=2"),
                     (self.ref, "strict=1", "block=4", "reject=2")]:
            with self.subTest(args=args):
                p = self.run_eval(*args)
                self.assertEqual(p.returncode, 0, p.stderr)

    def test_every_model_failure_and_nonfinite_result_fails(self):
        for mode in ["decode_fail", "decode_nan", "self_fail", "self_nan",
                     "forward_fail", "block_inf", "zero_vocab"]:
            with self.subTest(mode=mode):
                p = self.run_eval("block=4", "reject=2", mode=mode)
                self.assertEqual(p.returncode, 1, p.stderr)

    def test_mismatch_including_rejected_rows_replay_and_signed_zero_fails(self):
        for mode in ["block_diff", "rejected_diff", "replay_diff", "signed_zero"]:
            with self.subTest(mode=mode):
                p = self.run_eval("block=4", "reject=2", mode=mode)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn("block/replay mismatch", p.stderr)

    def test_external_numeric_differences_allowed_but_strict_fails(self):
        self.ref.write_bytes(struct.pack("=28f", *([2.0] * 28)))
        self.assertEqual(self.run_eval(self.ref, "block=4").returncode, 0)
        p = self.run_eval(self.ref, "strict=1")
        self.assertEqual(p.returncode, 1)
        self.assertIn("strict reference mismatch", p.stderr)

    def test_reference_open_short_read_and_nonfinite_fail(self):
        self.assertEqual(self.run_eval(self.directory / "missing").returncode, 1)
        for data in [b"", b"\0" * 15, b"\0" * (6 * 16),
                     struct.pack("=28f", *([float("nan")] * 28))]:
            with self.subTest(length=len(data)):
                self.ref.write_bytes(data)
                self.assertEqual(self.run_eval(self.ref).returncode, 1)

    def test_token_bounds_alignment_and_open_failure(self):
        for data in [b"", b"\0", b"\0" * (4096 * 4 + 1), b"\0" * (4097 * 4)]:
            with self.subTest(length=len(data)):
                self.tokens.write_bytes(data)
                self.assertEqual(self.run_eval().returncode, 1)
        self.assertEqual(self.run_eval(token_path=self.directory / "missing").returncode, 1)

    def test_invalid_numbers_cpu_lists_and_block_relations(self):
        for args in [("block=0",), ("block=-1",), ("block=4x",), ("block=4097",),
                     ("block=65",), ("reject=0",), ("block=4", "reject=4"),
                     ("block=65", "reject=1"), ("strict=1",), ("strict=2",),
                     ("bdump=x",), ("dump=",), ("bogus=1",),
                     ("-", "-1"), ("-", "18446744073709551615"),
                     ("-", "1x"), ("-", "1", ""),
                     ("-", "1", "0,1", ""), ("-", "1", "0,,1"),
                     ("-", "1", "0,1,"), ("-", "1", "0,0"),
                     ("-", "1", "-1"), ("-", "1", "2147483648")]:
            with self.subTest(args=args):
                self.assertEqual(self.run_eval(*args).returncode, 2)

    def test_dump_exact_coverage_and_io_failures(self):
        dump = self.directory / "decode.f32"
        bdump = self.directory / "block.f32"
        p = self.run_eval("block=4", "reject=2", "dump=" + str(dump),
                          "bdump=" + str(bdump))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(dump.read_bytes(), self.ref.read_bytes())
        self.assertEqual(bdump.read_bytes(), self.ref.read_bytes())
        for option in ["dump=", "bdump="]:
            p = self.run_eval("block=4", option + str(self.directory / "absent/out"))
            self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(self.run_eval("dump=" + str(self.tokens)).returncode, 2)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_buffered_dump_close_failure_fails(self):
        for option in ["dump=/dev/full", "bdump=/dev/full"]:
            with self.subTest(option=option):
                self.assertEqual(self.run_eval("block=4", option).returncode, 1)

    def test_self_check_coverage_limit_is_explicit(self):
        self.tokens.write_bytes(struct.pack("=257i", *range(257)))
        p = self.run_eval("block=64")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("256/257 input positions covered", p.stdout)


if __name__ == "__main__":
    unittest.main()
