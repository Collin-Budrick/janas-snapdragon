# Reproducing the Snapdragon runtime checks

These standard-library Python tools run the CPU correctness and performance
protocols described in [the runtime guide](../docs/snapdragon-runtime.md).
Both default to `--plan`. Execution requires native Windows ARM64 Python and
an explicit `--execute`; each driver runs children sequentially.

Build the native tools using [the Snapdragon build instructions](../docs/snapdragon.md).
Place the original main JNS and MTP sidecar under `models/`, or supply
`--model` and `--mtp`. Their filenames, sizes and SHA-256 pins are in
[snapdragon-runtime-models.json](snapdragon-runtime-models.json). The tools
use original expert weights, HIGH quality and CPU execution.

## Correctness gates

```powershell
python tools/validate-snapdragon-runtime.py --plan
python tools/validate-snapdragon-runtime.py --plan --create-reference
python tools/validate-snapdragon-runtime.py --execute --create-reference
```

The final command first creates `models/cpu64.qref`, with the experimental
tiles and DeltaNet controls disabled. It accepts that reference only when
its complete SHA-256 and logit digest match the published receipt. It then
runs the validation matrix. An existing reference is never overwritten.
For a preserved baseline tool, add
`--baseline-quality-exe D:/baseline/quality_check.exe`. To reuse a reference,
omit `--create-reference` and use `--reference D:/validation/cpu64.qref`.

The matrix contains:

- Native `test*.exe` units, excluding the two standalone Hexagon hardware
  probes. Explicit platform skips remain visible in the results.
- The original 64-position teacher-forced reference at batches
  1, 3, 5, 8, 9 and 17, context 1024, eight compute threads and 24576 MiB cache.
  Every comparison must have finite logits, the original digest, full top-1
  agreement and zero logit, NLL and KL errors.
- Strict 32-position block/rejection checks for `(block, reject)` values
  `(5,1)`, `(9,4)` and `(17,8)`. These keep prefixes of 4, 5 and 9 rows;
  every computed row, including rejected rows, is checked.
- Five two-turn session configurations: legacy MTP4, cycle MTP4, hybrid
  cycle8, hybrid cycle8 with adaptive vocabulary, and fixed MTP16 with an
  adaptive 16384-row head. Each checks the original five greedy/seeded
  comparisons. The fixed case must report an actual 17-row verification.

Session input comes from the public
[frozen token corpus](benchmark-prompts/qwen3-next-code.tokens.json), whose
first 160 IDs match the original check. A hybrid case can use lookup without
using its MTP head; the results label that head as unexercised. The fixed MTP
case must demonstrate head execution and repacking.

Use `--stage units|quality|eval|session` for a subset. `--out` selects a new
output directory; existing directories are rejected. Each process has a
600-second timeout. Validation forces `JANAS_CPU_TILES=1` and the three
DeltaNet controls to exercise the candidate configuration, while clearing
inherited `JANAS_*` settings. Logs retain requested controls and actual
runtime evidence. These checks establish correctness, not throughput.

`--verify-model-sha256` independently streams both model files before the
checks. Without it, validation records their receipt hashes and verifies
file sizes; it does not claim to have recomputed the large-file hashes.

## Performance screen and paired comparison

Preserve a control `janas-bench.exe` before rebuilding the candidate. Pass
that file explicitly; a baseline manifest can additionally bind its hash
and source commit. The published baseline hash is retained as provenance,
and a different explicit control is labelled as a different baseline.

```powershell
python tools/benchmark-snapdragon-runtime.py screen --plan --baseline-exe D:/baseline/janas-bench.exe
python tools/benchmark-snapdragon-runtime.py screen --execute --baseline-exe D:/baseline/janas-bench.exe --verify-model-sha256 --output results/screen
python tools/benchmark-snapdragon-runtime.py paired --plan --baseline-exe D:/baseline/janas-bench.exe --candidate-case hybrid8-adaptive
python tools/benchmark-snapdragon-runtime.py paired --execute --baseline-exe D:/baseline/janas-bench.exe --candidate-case hybrid8-adaptive --verify-model-sha256 --output results/paired
```

The screen uses code and prose, one fresh sample and two warm samples per
case. The paired protocol uses public code, prose and long prompts in ABBA
process order. All runs request 128 output tokens, context 2048, eight compute
threads, 32 GiB expert cache, 12 GiB reserve and greedy seed 20260926. Each
binary receives five warm samples per workload, split
3+2 across its two processes. A fresh process does not establish a cold SSD
or Windows file cache.

`--cases` selects screen cases; `--workloads` selects an ordered subset of
`code,prose,long`. Paired runs accept `--candidate-case` or a validated JSON
object through `--candidate-controls`. `--bench-exe`, `--model`, `--mtp` and
`--baseline-manifest` override local paths. A compact baseline manifest is:

```json
{
  "source_commit": "the-control-source-commit",
  "baseline": {
    "path": "janas-bench.exe",
    "sha256": "the-actual-64-character-lowercase-sha256"
  }
}
```

The binary path is relative to the manifest. The supplied hash must match
the executable. A completed prior download receipt can replace
`--verify-model-sha256` through `--model-receipt`. Fresh verification occurs
before timing and can affect the OS file cache. `--resume` validates the
manifest and completed ordered prefix before continuing an existing output.

The drivers preserve exact commands, executable hashes, input receipts,
cleared/inherited control names, stdout/stderr, raw results and validation
errors. Output hashes, token counts and runtime counters must support a
comparison before it qualifies. Logical expert-read bytes are not physical
SSD throughput; peak RSS is a process-lifetime measurement.

## Driver tests

```powershell
python -m unittest discover -s tools -p "test_*snapdragon_runtime.py"
```

These tests use synthetic records and temporary files. They check planning,
ABBA scheduling, input identities, environment isolation and rejection of
invalid comparisons without loading model weights or launching inference.
