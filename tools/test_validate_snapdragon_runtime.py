"""Model-free protocol checks for the public native validation driver."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("validate-snapdragon-runtime.py")


def load():
    spec = importlib.util.spec_from_file_location("validate_snapdragon_fixture", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.v = load()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.v.OUT = self.root / "results"
        self.v.OUT.mkdir()

    def result(self, name, text, code=0):
        stdout = self.v.OUT / (name + ".stdout.log")
        stderr = self.v.OUT / (name + ".stderr.log")
        stdout.write_text(text, encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        return {"name": name, "stdout": str(stdout), "stderr": str(stderr),
                "passed": code == 0, "exit_code": code}

    def quality_record(self, batch):
        return dict(finite_logits=True, hardware_execution_verified=True, positions=64,
                    batch=batch, context=1024, cache_mib=24576, vocab=self.v.VOCAB,
                    threads=8, profile="cpu", profile_effective="cpu", expert_bits=6,
                    min_quality=2, tuning=False, expert_skip=False, head_requantized=False,
                    gpu_selected=0, npu_selected=0, comparison="compared_to_cpu",
                    logit_fnv1a64=self.v.EXPECTED_DIGEST, top1_agreement=1,
                    nll_delta=0, logit_max_abs=0, logit_mae=0, logit_rmse=0,
                    mean_kl_ref_to_candidate=0, nll=3.39, perplexity=29.76,
                    reference_nll=3.39, reference_perplexity=29.76,
                    accelerator_counters=dict.fromkeys(("npu_products", "npu_ffns",
                        "npu_layer_passes", "gpu_product_submissions", "gpu_products",
                        "gpu_whole_passes"), 0))

    def fast_inputs(self):
        self.v.BIN=self.root/"bin";self.v.BIN.mkdir(exist_ok=True)
        for name in ("quality_check.exe","llm_session.exe","test_fixture.exe"):
            (self.v.BIN/name).write_bytes(("fixture "+name).encode())
        for key,name in (("MODEL","model.jns"),("MTP","mtp.jns"),("QREF","reference.qref"),("CODE","tokens.json"),("RECEIPT","receipt.json")):
            path=self.root/name;path.write_bytes(("fixture "+name).encode());setattr(self.v,key,path)
        (self.v.OUT/"session-code160.int32").write_bytes(b"frozen token fixture")
        self.v.CACHE_SOURCE_IDENTITY={"sha256":"fixture-source","files":{}}

    def mojo_module(self):
        path = self.root / "optional-mojo.dll"
        path.write_bytes(b"fixture Mojo module; never loaded")
        return path

    def test_mojo_is_explicit_and_inherited_environment_is_cleared(self):
        inherited = {"JANAS_MOJO": "1", "janas_mojo_library": "hostile.dll"}
        with patch.dict(os.environ, inherited):
            env = self.v.environment()
            self.assertFalse(any(k.upper().startswith("JANAS_MOJO") for k in env))
            self.assertEqual(os.environ["JANAS_MOJO"], "1")
            module = self.mojo_module()
            self.v.MOJO_LIBRARY = module
            for builder in (self.v.environment, self.v.raw_model_environment):
                selected = builder()
                self.assertEqual(selected["JANAS_MOJO"], "1")
                self.assertEqual(selected["JANAS_MOJO_LIBRARY"], str(module))

    def test_mojo_flag_requires_absolute_existing_module_and_plan_stays_inert(self):
        self.fast_inputs()
        (self.v.BIN / "test_mojo.exe").write_bytes(b"fixture test; never executed")
        module = self.mojo_module()
        self.assertEqual(self.v.mojo_input(module), {"path": str(module.resolve()),
            "bytes": module.stat().st_size, "sha256": self.v.sha(module)})
        for path in (Path("relative.dll"), self.root / "missing.dll", self.root):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "absolute path.*existing"):
                self.v.mojo_input(path)
        argv = [str(SCRIPT), "--plan", "--stage", "units", "--bin-dir", str(self.v.BIN),
                "--mojo-library", str(module), "--out", str(self.root / "uncreated")]
        with patch.object(sys, "argv", argv), patch.object(self.v.subprocess, "Popen",
                side_effect=AssertionError("plan launched a child")), contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(self.v.main(), 0)
        value = json.loads(text.getvalue())
        self.assertFalse((self.root / "uncreated").exists())
        self.assertEqual(value["mojo_module"]["sha256"], self.v.sha(module))
        self.assertIn("unit-test_mojo", [c["check_id"] for c in value["cases"]])
        self.assertEqual(value["excluded_optional_modules"], [])
        self.assertEqual(value["requested_janas_environment"]["JANAS_MOJO"], "1")

    def test_optional_mojo_unit_selection_is_separate_from_hardware_helpers(self):
        self.fast_inputs()
        (self.v.BIN / "test_mojo.exe").write_bytes(b"fixture")
        (self.v.BIN / "test_hx_probe.exe").write_bytes(b"fixture")
        plain = self.v.plan("units")
        self.assertEqual([c["check_id"] for c in plain["cases"]], ["unit-test_fixture"])
        self.assertEqual(plain["excluded_optional_modules"], ["test_mojo.exe"])
        self.assertIn("test_hx_probe.exe", plain["excluded_hardware_helpers"])
        def fake(name, argv, env, **kwargs):
            if Path(argv[0]).name == "test_mojo.exe":
                self.assertEqual(env["JANAS_MOJO"], "1")
                output = json.dumps(dict(test="mojo", cases=1341, compared_words=144570,
                                         failures=0, skipped=False)) + "\n"
            else:
                output = Path(argv[0]).stem + ": ok\n"
            return self.result(name, output)
        with patch.object(self.v, "run", fake):
            self.assertTrue(self.v.units())
        units = json.loads((self.v.OUT / "units.json").read_text())
        self.assertEqual(units["count"], 1)
        self.assertEqual(units["excluded_optional_modules"], ["test_mojo.exe"])
        self.v.MOJO_LIBRARY = self.mojo_module()
        with patch.object(self.v, "run", fake):
            self.assertTrue(self.v.units())
        units = json.loads((self.v.OUT / "units.json").read_text())
        self.assertEqual(units["count"], 2)
        self.assertEqual(units["excluded_optional_modules"], [])

    def test_explicit_mojo_requires_its_built_unit_and_a_complete_non_skipped_matrix(self):
        self.fast_inputs()
        self.v.MOJO_LIBRARY = self.mojo_module()
        with self.assertRaisesRegex(ValueError, "test_mojo.exe"):
            self.v.plan("units")
        path = self.v.BIN / "test_mojo.exe"
        path.write_bytes(b"fixture")
        good = dict(test="mojo", cases=1341, compared_words=144570, failures=0, skipped=False)
        for changes in ({}, {"skipped": True}, {"cases": 0}, {"cases": True},
                        {"compared_words": 0}, {"failures": False}, {"failures": 1}):
            value = dict(good, **changes)
            r = self.result("unit-test_mojo", json.dumps(value) + "\n")
            with self.subTest(changes=changes):
                self.assertEqual(self.v.validate_unit_result(r, path)["passed"], not changes)

    def test_mojo_proof_cache_hashes_module_content_even_when_size_and_mtime_match(self):
        self.fast_inputs()
        self.v.INCREMENTAL_CACHE = self.root / "proof-cache"
        module = self.mojo_module()
        self.v.MOJO_LIBRARY = module
        exe = self.v.BIN / "test_fixture.exe"
        env = self.v.environment()
        identity = self.v.case_identity("unit-test_fixture", [exe], env)
        self.assertEqual(identity["mojo_module"]["sha256"], self.v.sha(module))
        self.assertNotIn("mojo_module", self.v.case_identity("unit-test_fixture", [exe], {}))
        r = self.result("unit-test_fixture", "test_fixture: ok\n")
        r.update(cache_identity=identity, command=[str(exe)], wall_seconds=1, validation_errors=[])
        self.v.save_validated(r)
        self.assertTrue(self.v.cache_load("unit-test_fixture", [exe], env)["proof_reused"])
        st = module.stat()
        original = module.read_bytes()
        module.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        os.utime(module, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(module.stat().st_size, st.st_size)
        self.assertNotEqual(identity, self.v.case_identity("unit-test_fixture", [exe], env))
        self.assertIsNone(self.v.cache_load("unit-test_fixture", [exe], env))

    def test_mojo_source_identity_includes_kernel_and_build_script(self):
        root = self.root / "source"
        (root / "tools").mkdir(parents=True)
        (root / "src/mojo").mkdir(parents=True)
        runner = root / "tools/validate-snapdragon-runtime.py"
        runner.write_text("validator fixture")
        (root / "build.sh").write_text("C build fixture")
        (root / "tools/build-mojo.sh").write_text("Mojo build fixture")
        (root / "src/mojo/quant_panel.mojo").write_text("Mojo leaf fixture")
        with patch.object(self.v, "ROOT", root), patch.object(self.v, "__file__", str(runner)):
            before = self.v.source_identity()
            for relative in ("tools/build-mojo.sh", "src/mojo/quant_panel.mojo"):
                self.assertEqual(before["files"][relative], self.v.sha(root / relative))
                original = self.v.source_identity()["sha256"]
                with (root / relative).open("a") as out:
                    out.write(" changed")
                self.assertNotEqual(original, self.v.source_identity()["sha256"])

    def test_mojo_raw_proof_import_requires_recorded_matching_module_identity(self):
        self.fast_inputs()
        self.v.INCREMENTAL_CACHE = self.root / "proof-cache"
        self.v.MOJO_LIBRARY = self.mojo_module()
        source = self.root / "historical"
        source.mkdir()
        exe = self.v.BIN / "test_fixture.exe"
        stdout, stderr = source / "unit.stdout.log", source / "unit.stderr.log"
        stdout.write_text("test_fixture: ok\n")
        stderr.write_text("")
        record = dict(name="unit-test_fixture", passed=True, validation_errors=[], exit_code=0,
            command=[str(exe)], binary_sha256=self.v.sha(exe),
            janas_environment={k:v for k,v in self.v.environment().items() if k.startswith("JANAS_")},
            stdout=str(stdout), stderr=str(stderr), wall_seconds=1)
        (source / "summary.json").write_text(json.dumps(dict(passed=True)))
        planned = {"cases": [dict(check_id="unit-test_fixture", stage="units", command=[str(exe)])]}
        for bound in (None, self.v.mojo_identity(self.v.environment())):
            if bound is not None:
                record["mojo_module"] = bound
            (source / "units.json").write_text(json.dumps(dict(results=[record])))
            imported = self.v.import_raw_proofs(source, planned)
            self.assertEqual(imported["results"][0]["imported"], bound is not None)
            if bound is None:
                self.assertIn("Mojo module identity", imported["results"][0]["reason"])

    def test_mojo_module_mutation_during_child_invalidates_the_new_execution(self):
        self.fast_inputs()
        self.v.MOJO_LIBRARY = module = self.mojo_module()
        class Process:
            pid = 12345
            def wait(self, timeout):
                module.write_bytes(b"changed fixture module")
                return 0
        with (patch.object(self.v.subprocess, "Popen", return_value=Process()),
              patch.object(self.v.subprocess, "CREATE_NEW_PROCESS_GROUP", 0, create=True),
              contextlib.redirect_stdout(io.StringIO())):
            result = self.v.run("unit-test_fixture", [self.v.BIN / "test_fixture.exe"], self.v.environment())
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["passed"])
        self.assertFalse(result["mojo_module_unchanged"])

    def session_text(self,case):
        name,planner,mode,drafts,vocab,rows=case
        text=f"settings: compute=8 io=4 planner={planner} speculation={mode} drafts={drafts} draft_vocab={vocab} rows={rows} quality=HIGH expert_bits=6 head_bits=0\n"
        for n in range(2):
            text+=(f"runtime: mode={4 if mode=='hybrid' else 2} planner={dict(legacy=0,cycle=1,fixed=2)[planner]} cycles=32 max_width={17 if drafts==16 else 5} lookup_attempts=8 mtp_drafts=32 head_calls=32 repacks={2 if vocab=='adaptive' else 0} head_policy=0 head_rows=16384 head_epoch=2 verified\n")
            if name.startswith("hybrid-cost"):
                text+="hybrid: policy=cost decisions=12 offered=8 lookup=1 mtp=11 probes=1 cooldown=0 accepted=1 verified\n"
        for marker in ("greedy: turns vs all at once","greedy + speculation: turn 1","greedy + speculation: turn 2","sampling, speculation off vs on: turn 1","sampling, speculation off vs on: turn 2"):
            text+=marker+" identical\n"
        return text+"sampling differs from greedy yes\nboundary: head=0123456789abcdef mtp_kv=0123456789abcdef logits=0123456789abcdef identical\nllm_session: ok\n"

    def test_fast_plan_is_inert_and_has_identical_logical_coverage(self):
        self.fast_inputs();serial=self.v.plan("all");self.v.FAST=True;fast=self.v.plan("all")
        self.assertEqual([c["check_id"] for c in serial["cases"]],[c["check_id"] for c in fast["cases"]])
        self.assertTrue(fast["full_requested_coverage"]);self.assertTrue(fast["models_sequential"])
        drivers=fast["driver_invocations"]
        quality=next(c for c in drivers if c["stage"]=="quality")
        self.assertIn("1,3,5,8,9,17",quality["command"])
        session=next(c for c in drivers if c["stage"]=="session")
        self.assertEqual(len(session["logical_checks"]),7);self.assertEqual(session["reference_model_runs"],3)
        self.assertEqual(session["candidate_model_runs"],14)
        self.assertFalse((self.v.OUT/"session-cases.tsv").exists())

    def test_fast_quality_checks_every_width_and_rejects_partial_or_reordered_matrix(self):
        self.fast_inputs();self.v.FAST=True
        for bad in (None,"missing","reorder","drift","nan"):
            calls=[]
            def fake(name,argv,env,**kwargs):
                calls.append(argv);batches=[int(x) for x in argv[argv.index("--batches")+1].split(",")]
                records=[]
                for n,b in enumerate(batches):
                    r=self.quality_record(b);r.update(completed=True,exit_code=0,matrix_index=n,matrix_count=len(batches),max_batch_capacity=max(batches),model_load_count=1,functional_only=True,sequence_restart="forward_position_zero",actual_batches=[min(b,64-i) for i in range(0,64,b)],width_seconds=.1)
                    records.append(r)
                if bad=="missing":records.pop()
                if bad=="reorder":records.reverse()
                if bad=="drift":records[2]["logit_max_abs"]=.001
                if bad=="nan":records[2]["nll"]=float("nan")
                r=self.result(name,"\n".join(map(json.dumps,records))+"\n");r.update(wall_seconds=1,binary_sha256="fixture")
                return r
            with patch.object(self.v,"run",fake):self.assertEqual(self.v.quality(),bad is None)
            self.assertEqual(len(calls),1)
            rows=json.loads((self.v.OUT/"quality.json").read_text())["results"]
            self.assertEqual([r["name"] for r in rows],[f"quality64-batch{b}" for b in self.v.BATCHES])

    def test_fast_session_requires_all_cases_shared_references_and_boundary_oracle(self):
        self.fast_inputs();self.v.FAST=True;self.v.S2_CONTROLS["s2_exact"]="on"
        for bad in (None,"key","boundary","case"):
            def fake(name,argv,env,**kwargs):
                manifest=Path(next(x[6:] for x in argv if isinstance(x,str) and x.startswith("cases=")))
                lines=manifest.read_text().splitlines();self.assertEqual(len(lines),7)
                self.assertIn("s2_max_nodes=32",lines[4]);self.assertIn("s2_max_nodes=8",lines[0])
                text="runtime: mode=0 planner=0 cycles=32 verified\nruntime: mode=0 planner=0 cycles=32 verified\n"
                text+='session-reference: '+json.dumps(dict(version=1,valid=True,model_runs=3,content_key="0123456789abcdef",functional_only=True))+"\n"
                for n,case in enumerate(self.v.SESSION_CASES):
                    text+="session-case-begin: "+case[0]+"\n"
                    body=self.session_text(case)
                    if bad=="boundary" and n==2:body=body.replace("boundary:","removed:")
                    text+=body
                    r=dict(name=case[0],index=n,exit_code=0,wall_seconds=.1,checks=5,functional_only=True,reference_reused=True,actual_model_runs=2,boundary_forward_calls=1,content_key="0123456789abcdef")
                    if bad=="key" and n==2:r["content_key"]="ffffffffffffffff"
                    if bad!="case" or n!=2:text+="session-case: "+json.dumps(r)+"\n"
                r=self.result(name,text);r.update(wall_seconds=1,binary_sha256="fixture",cache_identity={"fixture":"binding"});return r
            with patch.object(self.v,"run",fake):self.assertEqual(self.v.session(),bad is None)
            rows=json.loads((self.v.OUT/"sessions.json").read_text())["results"]
            self.assertEqual(len(rows),7)
            if bad != "case": self.assertTrue(all(r["reference_reused"] is True for r in rows))

    def test_parallel_units_are_bounded_cpu8_and_keep_input_order(self):
        self.fast_inputs();self.v.FAST=True;self.v.UNIT_WORKERS=2
        for n in range(5):(self.v.BIN/f"test_{n}.exe").write_bytes(b"fixture")
        def fake(name,argv,env,**kwargs):return self.result(name,Path(argv[0]).stem+": ok\n")
        with patch.object(self.v,"run",fake),patch.object(self.v,"cpu8_parent") as affinity:
            self.assertTrue(self.v.units());affinity.assert_called_once()
        result=json.loads((self.v.OUT/"units.json").read_text())
        self.assertEqual(result["workers"],2);self.assertEqual(result["cpu_mask"],255)
        self.assertEqual([r["name"] for r in result["results"]],sorted(r["name"] for r in result["results"]))

    def test_explicit_selection_marks_subset_and_rejects_unknown_ids(self):
        self.v.SELECT_CHECKS={"session-fixed-mtp16-adaptive"};self.v.FAST=True
        plan=self.v.plan("session");self.assertFalse(plan["full_requested_coverage"])
        self.assertEqual(len(plan["cases"]),1);self.assertEqual(len(plan["omitted_checks"]),6)
        self.v.SELECT_CHECKS={"typo"}
        with self.assertRaisesRegex(ValueError,"unknown"):self.v.plan("session")

    def test_cache_binds_generated_content_models_controls_sources_and_output_paths(self):
        self.fast_inputs()
        cases=self.v.OUT/"session-cases.tsv";cases.write_text("one\tplanner=legacy\n")
        argv=[self.v.BIN/"llm_session.exe",self.v.MODEL,self.v.OUT/"session-code160.int32","cases="+str(cases)]
        first=self.v.case_identity("session-matrix",argv,{"JANAS_S2_EXACT":"on"},255)
        cases.write_text("two\tplanner=legacy\n")
        self.assertNotEqual(first,self.v.case_identity("session-matrix",argv,{"JANAS_S2_EXACT":"on"},255))
        cases.write_text("one\tplanner=legacy\n")
        for path in (self.v.MODEL,self.v.QREF,self.v.CODE,self.v.RECEIPT):
            before=self.v.case_identity("session-matrix",argv,{"JANAS_S2_EXACT":"on"},255)
            path.write_bytes(path.read_bytes()+b"x")
            self.assertNotEqual(before,self.v.case_identity("session-matrix",argv,{"JANAS_S2_EXACT":"on"},255))
        old=self.v.OUT;new=self.root/"other-output";new.mkdir();self.v.OUT=new
        for p in old.glob("*.int32"): (new/p.name).write_bytes(p.read_bytes())
        (new/cases.name).write_bytes(cases.read_bytes())
        newargv=[argv[0],argv[1],new/"session-code160.int32","cases="+str(new/cases.name)]
        same=self.v.case_identity("session-matrix",newargv,{"JANAS_S2_EXACT":"on"},255)
        self.v.OUT=old
        self.assertEqual(same,self.v.case_identity("session-matrix",argv,{"JANAS_S2_EXACT":"on"},255))
        self.assertNotEqual(same,self.v.case_identity("session-matrix",argv,{"JANAS_S2_EXACT":"off"},255))
        self.v.CACHE_SOURCE_IDENTITY={"sha256":"different"}
        self.assertNotEqual(same,self.v.case_identity("session-matrix",argv,{"JANAS_S2_EXACT":"on"},255))

    def test_cache_reuses_only_intact_validated_proofs_and_invalidates_new_failure(self):
        self.fast_inputs();self.v.INCREMENTAL_CACHE=self.root/"proof-cache"
        exe=self.v.BIN/"test_fixture.exe";identity=self.v.case_identity("unit-test_fixture",[exe],{})
        r=self.result("unit-test_fixture","test_fixture: ok\n");r.update(cache_identity=identity,command=[str(exe)],wall_seconds=1,validation_errors=[])
        self.v.save_validated(r)
        cached=self.v.cache_load("unit-test_fixture",[exe],{})
        self.assertTrue(cached["proof_reused"]);self.assertFalse(cached["performed_new_execution"])
        self.assertEqual(cached["wall_seconds"],1)
        folder=self.v.INCREMENTAL_CACHE/self.v.identity_key(identity)
        (folder/"stdout.log").write_text("test_fixture: forged\n")
        self.assertIsNone(self.v.cache_load("unit-test_fixture",[exe],{}))
        self.v.save_validated(r)
        r["passed"]=False;r["validation_errors"]=["new failure"]
        self.v.save_validated(r)
        self.assertIsNone(self.v.cache_load("unit-test_fixture",[exe],{}))

    def test_explicit_raw_proof_import_revalidates_and_rejects_changed_artifacts(self):
        self.fast_inputs();self.v.INCREMENTAL_CACHE=self.root/"proof-cache"
        source=self.root/"historical";source.mkdir();exe=self.v.BIN/"test_fixture.exe"
        stdout=source/"unit.stdout.log";stderr=source/"unit.stderr.log"
        stdout.write_text("test_fixture: ok\n");stderr.write_text("")
        record=dict(name="unit-test_fixture",passed=True,validation_errors=[],exit_code=0,
            command=[str(exe)],binary_sha256=self.v.sha(exe),janas_environment={k:v for k,v in self.v.environment().items() if k.startswith("JANAS_")},
            stdout=str(stdout),stderr=str(stderr),wall_seconds=1)
        (source/"summary.json").write_text(json.dumps(dict(passed=True)))
        (source/"units.json").write_text(json.dumps(dict(results=[record])))
        plan={"cases":[dict(check_id="unit-test_fixture",stage="units",command=[str(exe)])]}
        imported=self.v.import_raw_proofs(source,plan)
        self.assertTrue(imported["results"][0]["imported"])
        hit=self.v.cache_load("unit-test_fixture",[exe],self.v.environment())
        self.assertTrue(hit["proof_reused"]);self.assertFalse(hit["performed_new_execution"])
        self.assertTrue(hit["imported_raw_proof"]["revalidated_by_current_checker"])
        exe.write_bytes(b"changed executable")
        imported=self.v.import_raw_proofs(source,plan)
        self.assertFalse(imported["results"][0]["imported"])
        self.assertIn("executable SHA",imported["results"][0]["reason"])
        (source/"summary.json").write_text(json.dumps(dict(passed=False)))
        with self.assertRaisesRegex(ValueError,"complete passing"):self.v.import_raw_proofs(source,plan)

    def test_model_proof_import_requires_completed_earlier_unchanged_payload_attestation(self):
        self.fast_inputs()
        self.v.MODEL_PINS={role:(path.stat().st_size,self.v.sha(path)) for role,path in (("main",self.v.MODEL),("mtp",self.v.MTP))}
        with self.assertRaisesRegex(ValueError,"requires"):self.v.import_model_attestation(None,"2026-11-09T00:00:00+00:00")
        records={path.name:dict(sha256_verified=True,sha256=self.v.sha(path),bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,verified_utc="2026-11-08T00:00:00+00:00") for path in (self.v.MODEL,self.v.MTP)}
        receipt=self.root/"completed-model-receipt.json";receipt.write_text(json.dumps(dict(complete=True,files=records)))
        self.assertTrue(self.v.import_model_attestation(receipt,"2026-11-09T00:00:00+00:00")["complete"])
        with self.assertRaisesRegex(ValueError,"predates"):self.v.import_model_attestation(receipt,"2026-11-07T00:00:00+00:00")
        self.v.MODEL.write_bytes(b"changed payload")
        with self.assertRaisesRegex(ValueError,"unchanged payload"):self.v.import_model_attestation(receipt,"2026-11-09T00:00:00+00:00")

    def test_failed_whole_quality_matrix_never_caches_partial_passes(self):
        self.fast_inputs();self.v.FAST=True;self.v.INCREMENTAL_CACHE=self.root/"proof-cache"
        def fake(name,argv,env,**kwargs):
            rows=[]
            for n,b in enumerate(self.v.BATCHES):
                r=self.quality_record(b);r.update(completed=True,exit_code=0,matrix_index=n,matrix_count=6,max_batch_capacity=17,model_load_count=1,functional_only=True,sequence_restart="forward_position_zero",actual_batches=[min(b,64-i) for i in range(0,64,b)],width_seconds=.1)
                if b==5:r["logit_max_abs"]=.1
                rows.append(r)
            r=self.result(name,"\n".join(map(json.dumps,rows))+"\n");r.update(wall_seconds=1,binary_sha256="fixture",cache_identity={"fixture":"quality"});return r
        with patch.object(self.v,"run",fake):self.assertFalse(self.v.quality())
        self.assertFalse(list(self.v.INCREMENTAL_CACHE.glob("*/manifest.json")))

    def test_s2_fixed16_stress_has_explicit32_node_override_only(self):
        self.v.S2_CONTROLS.update(s2_exact="on",s2_max_nodes=8)
        self.assertIn("s2_max_nodes=32",self.v.session_controls("fixed-mtp16-adaptive"))
        self.assertIn("s2_max_nodes=8",self.v.session_controls("legacy-mtp4"))
        self.assertEqual(self.v.S2_CONTROLS["s2_max_nodes"],8)
        case=next(c for c in self.v.plan("session")["cases"] if c["name"]=="fixed-mtp16-adaptive")
        self.assertEqual(case["s2_controls_effective"]["s2_max_nodes"],32)
        self.assertIsNotNone(case["stress_override"])

    def test_s2_validation_is_separately_opt_in_and_recorded(self):
        self.assertFalse(any(k.startswith("JANAS_S2_") for k in self.v.environment()))
        self.assertFalse(any(v.startswith("s2_") for v in self.v.session_controls("legacy-mtp4")))
        self.v.S2_CONTROLS.update(s2_exact="on", s2_branches=1)
        controls=self.v.session_controls("legacy-mtp4")
        self.assertIn("s2_exact=on",controls);self.assertIn("s2_branches=1",controls)
        self.assertEqual(self.v.environment()["JANAS_S2_EXACT"],"on")
        plan=self.v.plan("session")
        self.assertEqual(plan["s2_controls"]["s2_exact"],"on")
        self.assertTrue(all("s2_exact=on" in c["command"] for c in plan["cases"]))
        args=self.v.parser().parse_args(["--plan","--s2-exact","on","--s2-branches","1"])
        self.assertEqual(args.s2_exact,"on")

    def test_source_relative_root_and_plan_never_launches_or_creates_output(self):
        out = self.root / "uncreated"
        argv = [str(SCRIPT), "--plan", "--stage", "quality", "--out", str(out),
                "--model", str(self.root / "main.jns"),
                "--reference", str(self.root / "reference.qref")]
        with patch.object(sys, "argv", argv), patch.object(self.v.subprocess, "Popen",
                side_effect=AssertionError("plan launched a process")), contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(self.v.main(), 0)
        plan = json.loads(text.getvalue())
        self.assertEqual(self.v.ROOT, SCRIPT.resolve().parents[1])
        self.assertFalse(plan["launches_executables"])
        self.assertFalse(out.exists())
        self.assertEqual([r["batch"] for r in plan["cases"]], [1, 3, 5, 8, 9, 17])
        self.assertTrue(all(str(self.root / "main.jns") in r["command"] for r in plan["cases"]))
        self.assertEqual(plan["reference"], str((self.root / "reference.qref").resolve()))
        self.assertNotIn("frozen-chat80b-20261108", json.dumps(plan))

    def test_inherited_controls_are_cleared_without_mutating_parent(self):
        with patch.dict(os.environ, {"JANAS_UNKNOWN_CONTROL": "parent", "JANAS_KV": "float"}):
            env = self.v.environment()
            self.assertNotIn("JANAS_UNKNOWN_CONTROL", env)
            self.assertNotIn("JANAS_KV", env)
            self.assertEqual(os.environ["JANAS_UNKNOWN_CONTROL"], "parent")
            self.assertTrue(all(env[k] == "1" for k in (
                "JANAS_CPU_TILES", "JANAS_GDN_CHAIN", "JANAS_GDN_JOURNAL", "JANAS_GDN_K_ONLY")))

    def test_receipt_and_frozen_tokens_are_public_and_self_consistent(self):
        receipt = self.v.load_receipt(self.v.RECEIPT)
        self.assertEqual(receipt["models"]["main"]["bytes"], 48411922432)
        corpus = json.loads(self.v.CODE.read_text(encoding="utf-8"))
        raw = struct.pack("<" + "i" * len(corpus["tokens"]), *corpus["tokens"])
        self.assertEqual(len(corpus["tokens"]), 276)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), corpus["int32_le_sha256"])
        self.assertEqual(hashlib.sha256(raw[:640]).hexdigest(),
                         self.v.EXPECTED_INPUTS["session-code160.int32"])
        self.assertNotIn("C:\\Users", json.dumps(corpus))

    def test_ambiguous_or_incomplete_receipts_are_rejected(self):
        path = self.root / "receipt.json"
        for text in ('{"schema":1,"schema":1}', '{"schema":true}',
                     '{"schema":1,"models":{}}', '{"schema":NaN}'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.v.load_receipt(path)

    def test_reference_generation_refuses_to_overwrite(self):
        self.v.QREF = self.root / "cpu64.qref"
        self.v.QREF.write_bytes(b"preserve existing input")
        with patch.object(self.v, "run", side_effect=AssertionError("unexpected model run")):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                self.v.create_reference()
        self.assertEqual(self.v.QREF.read_bytes(), b"preserve existing input")

    def test_execute_on_non_windows_rejects_before_creating_output(self):
        out = self.root / "uncreated"
        fake_os = types.SimpleNamespace(name="posix", getpid=os.getpid, environ=os.environ)
        with patch.object(self.v, "os", fake_os), patch.object(sys, "argv",
                [str(SCRIPT), "--execute", "--stage", "units", "--out", str(out)]), \
                patch.object(self.v.subprocess, "Popen", side_effect=AssertionError("spawned")), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.v.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(out.exists())

    def test_quality_rejects_drift_and_nonfinite_even_on_exit_zero(self):
        for change in ({}, {"nll_delta": 0.01}, {"nll": float("nan")}):
            calls = []
            def fake(name, argv, env, **unused):
                batch = int(argv[argv.index("--batch") + 1])
                record = self.quality_record(batch)
                record.update(change)
                calls.append(batch)
                return self.result(name, json.dumps(record) + "\n")
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.quality(), not change)
            self.assertEqual(calls, [1, 3, 5, 8, 9, 17] if not change else [1])

    def test_eval_requires_actual_rejection_prefix_marker(self):
        for wrong in (False, True):
            def fake(name, argv, env, expected_exit=(0,), **unused):
                if name == "eval-cli-capabilities":
                    result = self.result(name, "usage block=B reject=R strict=1\n", 2)
                    result["passed"] = 2 in expected_exit
                    return result
                block = int(next(a[6:] for a in argv if isinstance(a, str) and a.startswith("block=")))
                reject = int(next(a[7:] for a in argv if isinstance(a, str) and a.startswith("reject=")))
                return self.result(name, f"blocks of {block} keeping {block-reject+wrong}: passes\n"
                    f"blocks of {block}: 32/32 positions bit-identical to token by token, max |diff| 0;\n"
                    "32/32 input positions covered\n"
                    "top-1 agreement 32/32, max |diff| 0.000, mean |diff| 0.0000\n")
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.evaluation(), not wrong)

    def test_session_requires_width17_and_does_not_claim_unused_adaptive_head(self):
        for wrong_width in (False, True):
            def fake(name, argv, env, **unused):
                values = dict(a.split("=", 1) for a in argv if isinstance(a, str) and "=" in a)
                planner, mode = values["planner"], values["speculation"]
                drafts, rows = int(values["drafts"]), int(values["rows"])
                adaptive = values["draft_vocab"] == "adaptive"
                text = (f"settings: compute=8 io=4 planner={planner} speculation={mode} drafts={drafts} "
                        f"draft_vocab={values['draft_vocab']} rows={rows} quality=HIGH expert_bits=6 head_bits=0\n")
                for i in range(4):
                    selected = i % 2 == 1
                    mtp = selected and not (adaptive and mode == "hybrid")
                    resolved = {"legacy": 0, "cycle": 1, "fixed": 2}[planner] if selected else 0
                    width = (16 if wrong_width else 17) if selected and drafts == 16 else 5 if selected else 1
                    text += (f"runtime: mode={4 if selected and mode=='hybrid' else 2 if selected else 0} "
                        f"planner={resolved} cycles=36 max_width={width} "
                        f"lookup_attempts={4 if selected and mode=='hybrid' else 0} mtp_drafts={32 if mtp else 0} "
                        f"head_calls={32 if mtp else 0} repacks={2 if mtp and adaptive else 0} "
                        "head_policy=0 head_rows=16384 head_epoch=2 verified\n")
                    if selected and values.get("hybrid_policy") == "cost":
                        text += "hybrid: policy=cost decisions=12 offered=8 lookup=1 mtp=11 probes=1 cooldown=0 accepted=1 verified\n"
                for marker in ("greedy: turns vs all at once", "greedy + speculation: turn 1",
                    "greedy + speculation: turn 2", "sampling, speculation off vs on: turn 1",
                    "sampling, speculation off vs on: turn 2"):
                    text += marker + " identical\n"
                return self.result(name, text + "sampling differs from greedy yes\nllm_session: ok\n")
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.session(), not wrong_width)
            records = json.loads((self.v.OUT / "sessions.json").read_text())["results"]
            hybrid = next(r for r in records if r["name"] == "session-hybrid-cycle8-adaptive")
            self.assertFalse(hybrid["adaptive_execution_verified"])

    def test_cost_hybrid_plan_separates_normal_and_forced_probe_validation(self):
        cases = self.v.plan("session")["cases"]
        normal = next(c for c in cases if c["name"] == "hybrid-cost8")
        probe = next(c for c in cases if c["name"] == "hybrid-cost8-probe")
        for case in (normal, probe):
            self.assertIn("hybrid_policy=cost", case["command"])
            self.assertIn("lookup_drafts=8", case["command"])
            self.assertIn("mtp_drafts=4", case["command"])
            self.assertEqual(case["planner"], "legacy")
        self.assertIn("hybrid_probe_percent=2", normal["command"])
        self.assertIn("hybrid_probe_percent=100", probe["command"])

    def test_semantic_hybrid_requires_all_sources_both_samplers_and_actual_lookup(self):
        for lookup_seen in (False, True):
            def fake(name, argv, env, **unused):
                records = []
                for sampler in ("greedy", "seeded"):
                    for mode in ("plain", "mtp4", "hybrid-cost8"):
                        for turn, phase in enumerate(["copy"] * 4 + ["switch", "return"]):
                            records.append(dict(schema=1, kind="semantic", sampler=sampler, mode=mode,
                                turn=turn, phase=phase, identity_verified=True, input_threads=8, output_threads=8))
                records.append(dict(schema=1, kind="summary", identity_verified=True,
                                    lookup_exercised=lookup_seen, fallback_exercised=True))
                return self.result(name, "\n".join(json.dumps(v) for v in records))
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.semantic_hybrid(), lookup_seen)
        planned = self.v.plan("hybrid")["cases"]
        self.assertEqual(len(planned), 1)
        self.assertIn("probe=2", planned[0]["command"])

    def test_attention_calibration_seeds_identical_private_copies_without_touching_original(self):
        path = self.root / "attn.txt"
        data = b"Snapdragon_X|8|16/2/256|0|2048 4294967295 1024\n"
        path.write_bytes(data)
        frozen = self.v.attention_input(path)
        for name in ("first", "second"):
            seed = self.v.seed_attention_calibration(self.root / name, frozen)
            copied = Path(seed["private_path"])
            self.assertEqual(copied.read_bytes(), data)
            copied.write_bytes(b"private changes remain private")
        self.assertEqual(path.read_bytes(), data)
        path.write_bytes(b"Snapdragon_X|8|16/2/256|0|2048 1024 1024\n")
        with self.assertRaises(RuntimeError):
            self.v.seed_attention_calibration(self.root / "third", frozen)

    def test_model_files_with_same_basename_are_validated_by_role(self):
        self.v.MODEL = self.root / "main" / "weights.jns"
        self.v.MTP = self.root / "mtp" / "weights.jns"
        for path, data in ((self.v.MODEL, b"a"), (self.v.MTP, b"bb")):
            path.parent.mkdir()
            path.write_bytes(data)
        self.v.MODEL_PINS = {"main": (1, "a" * 64), "mtp": (2, "b" * 64)}
        records = self.v.model_metadata()
        self.assertEqual([r["size"] for r in records], [1, 2])
        self.assertTrue(all(not r["sha256_recomputed_by_runner"] for r in records))


if __name__ == "__main__":
    unittest.main()
