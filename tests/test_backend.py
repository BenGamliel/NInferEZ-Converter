from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / "backend"), str(ROOT / "vendor"), str(ROOT / "tests")]
from core import Context, Cancelled, convert, inspect, probe, analyze, _checkpoint
from fixtures import checkpoint, checkpoint_gguf, draft_checkpoint, write_safetensors
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ez-tests-", dir=ROOT / "build")
        self.root = Path(self.tmp.name)
        self.events = []
        self.ctx = Context(self.events.append, threading.Event())

    def tearDown(self):
        self.tmp.cleanup()

    def request(self, source, **extra):
        return {"source": str(source), "output": str(self.root / "result.ninfer"), "device": "cpu", "profile": "auto", **extra}

    def test_dense_real_source_all_profiles_and_progress(self):
        src, _, _ = checkpoint(self.root / "מודל עם רווחים")
        for profile in ("auto", "q4", "q5", "q6", "q8", "bf16", "fp8", "preserve"):
            with self.subTest(profile=profile):
                req = self.request(src, profile=profile, output=str(self.root / (profile + ".ninfer")), rowsPerChunk=7)
                result = inspect(req, self.ctx)
                self.assertTrue(result["compatible"])
                report = convert(req, self.ctx)
                self.assertFalse(report["runtimeVerified"])
                with Artifact(req["output"]) as artifact:
                    self.assertEqual(set(artifact.directory.components), {"text"})
        progress = [e for e in self.events if e["type"] == "progress"]
        self.assertTrue(progress)
        self.assertEqual(progress[-1]["percent"], 100)

    def test_moe_optional_components_and_dimension_driven_recipe(self):
        src, _, _ = checkpoint(self.root / "moe", moe=True, mtp=True, vision=True)
        req = self.request(src, components=["text", "mtp", "vision"])
        result = inspect(req, self.ctx)
        self.assertEqual(set(result["components"]), {"text", "mtp", "vision"})
        convert(req, self.ctx)
        with Artifact(req["output"]) as artifact:
            self.assertIn("text/layers/0/moe/experts/2/down", artifact.directory.bindings)

    def test_generic_gguf_dense_moe_mapping_against_independent_hf_values(self):
        from contextlib import ExitStack
        for moe in (False, True):
            src, cfg, tensors = checkpoint(self.root / ("ggmoe" if moe else "ggdense"), moe=moe)
            gg = checkpoint_gguf(src, cfg, tensors)
            req = self.request(gg, configFolder=str(src), profile="bf16", output=str(self.root / ("moe.ninfer" if moe else "dense.ninfer")))
            from tools.convert.qwen3_5 import build_model
            from tools.convert.sources.safetensors import SafetensorsSource
            with ExitStack() as stack:
                ggmodel, _, _ = _checkpoint(req, stack, self.ctx)
                original = build_model(stack.enter_context(SafetensorsSource(src)))
                for name, parameter in original.parameters.items():
                    with self.subTest(moe=moe, parameter=name):
                        torch.testing.assert_close(ggmodel.parameters[name].source.values(), parameter.source.values(), atol=2e-7, rtol=1e-5)
            self.assertTrue(inspect(req, self.ctx)["compatible"])
            convert(req, self.ctx)

    def test_no_overwrite_and_failed_source_no_published_output(self):
        src, _, _ = checkpoint(self.root / "bad")
        (src / "model.safetensors").write_bytes(b"broken")
        req = self.request(src)
        with self.assertRaises(Exception): convert(req, self.ctx)
        self.assertFalse(Path(req["output"]).exists())
        self.assertFalse(list(self.root.glob(".ezc-*")))
        target = Path(req["output"])
        target.write_bytes(b"user data")
        with self.assertRaises(FileExistsError): convert(req, self.ctx)
        self.assertEqual(target.read_bytes(), b"user data")

    def test_auto_mixed_nvfp4_fp8_and_small_projection_upcast(self):
        from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
        for modelopt in (False, True):
            src, _, tensors = checkpoint(self.root / ("modelopt" if modelopt else "gptq"))
            gate = "model.layers.1.mlp.gate_proj"
            n, k = tensors.pop(gate + ".weight").shape
            codes = torch.arange(n * (k // 2), dtype=torch.int64).reshape(n, k // 2).remainder(256).to(torch.uint8)
            scales = torch.full((n, k // 16), 2.0, dtype=torch.float8_e4m3fn)
            tensors[gate + (".weight" if modelopt else ".weight_packed")] = codes
            tensors[gate + ".weight_scale"] = scales
            tensors[gate + (".weight_scale_2" if modelopt else ".weight_global_scale")] = torch.tensor([2.0])
            tensors[gate + (".input_scale" if modelopt else ".input_global_scale")] = torch.tensor([4.0])
            small = "model.layers.0.linear_attn.in_proj_a"
            sn, sk = tensors.pop(small + ".weight").shape
            tensors[small + ".weight_packed"] = torch.full((sn, sk // 2), 0x22, dtype=torch.uint8)
            tensors[small + ".weight_scale"] = torch.ones((sn, sk // 16), dtype=torch.float8_e4m3fn)
            tensors[small + ".weight_global_scale"] = torch.tensor([1.0])
            tensors[small + ".input_global_scale"] = torch.tensor([1.0])
            output = "lm_head"
            on, ok = tensors[output + ".weight"].shape
            tensors[output + ".weight"] = torch.ones((on, ok), dtype=torch.float8_e4m3fn)
            tensors[output + ".weight_scale"] = torch.tensor([.25])
            write_safetensors(src / "model.safetensors", tensors)
            req = self.request(src, output=str(self.root / ("modelopt.ninfer" if modelopt else "gptq.ninfer")))
            self.assertTrue(inspect(req, self.ctx)["compatible"])
            convert(req, self.ctx)
            with Artifact(req["output"]) as artifact:
                obj = binding_parts(artifact.directory.bindings["text/layers/1/mlp/gate"], artifact.by_id)[0][0]
                stored_codes, stored_scales, divisor = decode_nvfp4_words(artifact.read_object(obj), (n, k))
                self.assertTrue(torch.equal(stored_codes, codes))
                self.assertTrue(torch.equal(stored_scales, scales.view(torch.uint8)))
                self.assertEqual(divisor.item(), .5 if modelopt else 2.0)
                small_obj = binding_parts(artifact.directory.bindings["text/layers/0/gdn/a_projection"], artifact.by_id)[0][0]
                self.assertEqual(artifact.by_id[small_obj].format, "bf16")

    def test_cancel_during_writing_removes_staging(self):
        src, _, _ = checkpoint(self.root / "cancel")
        req = self.request(src)
        def emit(event):
            self.events.append(event)
            if event["type"] == "progress": self.ctx.cancel.set()
        self.ctx.emit = emit
        with self.assertRaises(Cancelled): convert(req, self.ctx)
        self.assertFalse(Path(req["output"]).exists())
        self.assertFalse(list(self.root.glob(".ezc-*")))

    def test_separate_quantized_text_does_not_replace_vision_or_mtp_source(self):
        src, cfg, tensors = checkpoint(self.root / "separate", mtp=True, vision=True)
        gate = "model.layers.1.mlp.gate_proj"
        text = {k: v for k, v in tensors.items() if not k.startswith(("model.visual.", "mtp."))}
        n, k = text.pop(gate + ".weight").shape
        text[gate + ".weight_packed"] = torch.full((n, k // 2), 0x22, dtype=torch.uint8)
        text[gate + ".weight_scale"] = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn)
        text[gate + ".weight_global_scale"] = torch.tensor([1.0])
        text[gate + ".input_global_scale"] = torch.tensor([1.0])
        quant = self.root / "quantized"; quant.mkdir()
        (quant / "config.json").write_text(json.dumps(cfg))
        write_safetensors(quant / "model.safetensors", text)
        vision = self.root / "vision"; vision.mkdir()
        (vision / "config.json").write_text(json.dumps(cfg))
        write_safetensors(vision / "model.safetensors", {k: v for k, v in tensors.items() if k.startswith("model.visual.")})
        write_safetensors(src / "model.safetensors", {k: v for k, v in tensors.items() if not k.startswith("model.visual.")})
        req = self.request(src, components=["text", "vision", "mtp"], sources={"quantized":str(quant), "vision":str(vision)})
        self.assertTrue(inspect(req, self.ctx)["compatible"])
        convert(req, self.ctx)
        with Artifact(req["output"]) as artifact:
            obj = binding_parts(artifact.directory.bindings["text/layers/1/mlp/gate"], artifact.by_id)[0][0]
            self.assertEqual(artifact.by_id[obj].format, "nvfp4")
            self.assertIn("vision/patch_embedding", artifact.directory.bindings)
            self.assertIn("mtp/input_projection", artifact.directory.bindings)

    def test_lossless_repack_shards(self):
        src, _, _ = checkpoint(self.root / "repack")
        req = self.request(src, profile="bf16", output=str(self.root / "original.ninfer"), maxFileBytes=32768)
        convert(req, self.ctx)
        req2 = self.request(Path(req["output"]), output=str(self.root / "copied.ninfer"), maxFileBytes=65536)
        convert(req2, self.ctx)
        with Artifact(req["output"]) as first, Artifact(req2["output"]) as second:
            self.assertGreater(len(first.directory.files), 1)
            self.assertEqual(first.directory.bindings, second.directory.bindings)
            for obj in first.objects: self.assertEqual(first.read_object(obj.id), second.read_object(obj.id))

    def test_mismatched_architecture_and_geometry_rejected(self):
        src, cfg, tensors = checkpoint(self.root / "incompatible")
        cfg["architectures"] = ["LlamaForCausalLM"]
        (src / "config.json").write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "unsupported"): inspect(self.request(src), self.ctx)

    def test_discovery_missing_files_and_individual_resource_recovery(self):
        src, _, _ = checkpoint(self.root / "resources")
        moved = self.root / "matching files"; moved.mkdir()
        resources = {}
        for role in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "config.json"):
            shutil.move(src / role, moved / role)
            if role != "config.json": resources[role] = str(moved / role)
        req = self.request(src)
        initial = probe(req, self.ctx)
        self.assertFalse(initial["ready"])
        self.assertEqual(sum(r["status"] == "missing" for r in initial["requirements"]), 5)
        req.update(configFile=str(moved / "config.json"), resources=resources)
        self.assertTrue(probe(req, self.ctx)["ready"])
        choices = analyze(req, self.ctx)
        self.assertEqual(len(choices["options"]), 8)
        self.assertTrue(all(o["supported"] for o in choices["options"]))
        q4 = next(o["plan"] for o in choices["options"] if o["profile"] == "q4")
        self.assertIn("q4_g64_fp16", q4["formats"])
        self.assertIn("q8_g32_fp16", q4["formats"])
        self.assertIn("bf16", q4["formats"])
        self.assertTrue(q4["tensorPlan"])
        self.assertFalse(Path(req["output"]).exists())
        convert({**req, "profile": "q4", "expectedPlanSignature": q4["planSignature"], "expectedFileState": choices["fileState"]}, self.ctx)

    def test_analysis_reports_unavailable_gguf_preserve_and_artifact_repack(self):
        src, cfg, tensors = checkpoint(self.root / "scan-gguf")
        gg = checkpoint_gguf(src, cfg, tensors)
        scan = analyze(self.request(gg, configFolder=str(src)), self.ctx)
        preserve = next(o for o in scan["options"] if o["profile"] == "preserve")
        self.assertFalse(preserve["supported"])
        self.assertIn("preservation", preserve["reason"])
        bf16 = next(o for o in scan["options"] if o["profile"] == "bf16")
        q4 = next(o for o in scan["options"] if o["profile"] == "q4")
        self.assertGreater(bf16["plan"]["encodedBytes"], q4["plan"]["encodedBytes"])
        req = self.request(src, profile="bf16")
        convert(req, self.ctx)
        repack = analyze(self.request(Path(req["output"])), self.ctx)
        self.assertEqual([o["profile"] for o in repack["options"]], ["preserve"])
        with self.assertRaisesRegex(ValueError, "repacking"):
            inspect(self.request(Path(req["output"]), profile="q4"), self.ctx)
        with self.assertRaisesRegex(ValueError, "repacking"):
            convert(self.request(Path(req["output"]), output=str(self.root / "bad-q4.ninfer"), profile="q4"), self.ctx)

    def test_v2_upgrade_layout_respects_selected_shard_limit(self):
        from tools.upgrade_ninfer_v2_to_v3 import layout
        for limit in (8192, 12345, 32768):
            directory = {"objects": []}
            data, start = layout(Path("upgraded.ninfer"), directory, 75000, limit)
            files = directory["files"]
            self.assertEqual(sum(f["payload_bytes"] for f in files), 75000)
            self.assertLessEqual(start + files[0]["payload_bytes"], limit)
            self.assertTrue(all(4096 + f["payload_bytes"] <= limit for f in files[1:]))
            self.assertEqual(json.loads(data)["files"], files)
        with self.assertRaisesRegex(ValueError, "at least"):
            layout(Path("bad.ninfer"), {}, 75000, 4096)

    def test_scan_plan_changes_and_tensor_overrides_are_checked(self):
        src, _, _ = checkpoint(self.root / "pin")
        req = self.request(src, profile="q4", tensorOverrides=[{"selector": "text/layers/*/mlp/down", "format": "q8_g32_fp16", "method": "grouped_absmax"}])
        plan = inspect(req, self.ctx)
        self.assertTrue(all(w["format"] == "q8_g32_fp16" for w in plan["tensorPlan"] if any(n.endswith("/mlp/down") for n in w["parameters"])))
        with self.assertRaisesRegex(ValueError, "plan changed"):
            convert({**req, "profile": "q8", "expectedPlanSignature": plan["planSignature"]}, self.ctx)
        state = probe(req, self.ctx)["fileState"]
        (src / "chat_template.jinja").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed since"):
            convert({**req, "expectedPlanSignature": plan["planSignature"], "expectedFileState": state}, self.ctx)
        self.assertFalse(Path(req["output"]).exists())
        self.assertFalse(list(self.root.glob(".ezc-*")))

    def test_optional_requirements_and_analysis_cancellation(self):
        src, _, _ = checkpoint(self.root / "optional", vision=True, mtp=True)
        req = self.request(src, components=["text", "vision", "mtp", "dflash2"])
        first = probe(req, self.ctx)
        self.assertFalse(first["ready"])
        self.assertTrue(any(r["key"] == "source:dflash2" and r["status"] == "missing" for r in first["requirements"]))
        req["components"] = ["text", "vision", "mtp"]
        scan = analyze(req, self.ctx)
        self.assertTrue(all(o["supported"] for o in scan["options"]))
        self.ctx.cancel.set()
        with self.assertRaises(Cancelled): analyze(req, self.ctx)

    def test_missing_indexed_shard_is_attached_without_changing_index(self):
        src, _, tensors = checkpoint(self.root / "indexed")
        original = src / "model-00001-of-00001.safetensors"
        index = src / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {name: original.name for name in tensors}}))
        moved = self.root / "downloaded-shard.safetensors"
        shutil.move(src / "model.safetensors", moved)
        req = self.request(src)
        first = probe(req, self.ctx)
        self.assertFalse(first["ready"])
        self.assertTrue(any(r["key"] == "shard:" + str(original.resolve()) for r in first["requirements"]))
        before = index.read_bytes()
        req["shardOverrides"] = {str(original.resolve()): str(moved)}
        self.assertTrue(probe(req, self.ctx)["ready"])
        self.assertTrue(inspect(req, self.ctx)["compatible"])
        convert(req, self.ctx)
        self.assertEqual(before, index.read_bytes())
        self.assertFalse(original.exists())

    def test_all_optional_component_subsets_dense_moe_and_proposal(self):
        from itertools import combinations
        for moe in (False, True):
            src, cfg, _ = checkpoint(self.root / ("all-moe" if moe else "all-dense"), moe=moe, vision=True, mtp=True)
            sources = {key: str(draft_checkpoint(self.root / (str(moe) + key), src, cfg, key)) for key in ("dflash", "dflash2")}
            optional = ["vision", "mtp", "dflash", "dflash2"]
            for count in range(5):
                for subset in combinations(optional, count):
                    for profile in ("auto", "q4"):
                        with self.subTest(moe=moe, subset=subset, profile=profile):
                            req = self.request(src, components=["text", *subset], profile=profile, sources={k: v for k, v in sources.items() if k in subset})
                            self.assertTrue(inspect(req, self.ctx)["compatible"])
            ranking = self.root / (str(moe) + "-ranking.i64")
            np.arange(cfg["vocab_size"], dtype=np.int64).tofile(ranking)
            req = self.request(src, components=["text", *optional], sources=sources, proposal=True, proposalRows=8,
                               ranking=str(ranking), output=str(self.root / (str(moe) + "-all.ninfer")))
            options = analyze(req, self.ctx)
            self.assertTrue(all(o["supported"] for o in options["options"]), str(options["options"]))
            convert(req, self.ctx)
            with Artifact(req["output"]) as artifact:
                self.assertEqual(set(artifact.directory.components), {"text", *optional})
                self.assertIn("proposal/head", artifact.directory.bindings)

    def test_uppercase_weights_and_utf8_bom_resources(self):
        src, _, _ = checkpoint(self.root / "bom")
        weight = src / "WEIGHTS.SAFETENSORS"
        (src / "model.safetensors").rename(weight)
        for role in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
            path = src / role
            path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        req = self.request(weight)
        self.assertTrue(probe(req, self.ctx)["ready"])
        self.assertTrue(inspect(req, self.ctx)["compatible"])
        convert(req, self.ctx)
        with Artifact(req["output"]) as artifact:
            self.assertFalse(artifact.read_object("resource/text/generation_config.json").startswith(b"\xef\xbb\xbf"))

    def test_standalone_native_backend_unicode(self):
        exe = ROOT / "dist" / "Portable" / "NInferEZ.Converter.Backend.exe"
        if not exe.is_file(): self.skipTest("build the backend host first")
        src, _, _ = checkpoint(self.root / "עברית spaces")
        env = os.environ.copy()
        env["DOTNET_BUNDLE_EXTRACT_BASE_DIR"] = str(self.root / "dotnet-cache")
        run = subprocess.run([str(exe), "inspect"], input=json.dumps(self.request(src), ensure_ascii=False), capture_output=True, text=True, encoding="utf-8", timeout=60, env=env)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        events = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertTrue(events[-1]["success"])
        request_file = self.root / "request.json"
        request_file.write_text(json.dumps(self.request(src, output=str(self.root / "native עברית.ninfer"), maxFileBytes=32768)), encoding="utf-8")
        relative = request_file.relative_to(ROOT)
        run = subprocess.run([str(exe), "convert", "--request", str(relative)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, env=env)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertTrue((self.root / "native עברית.ninfer").is_file())
        with Artifact(self.root / "native עברית.ninfer") as artifact:
            self.assertGreater(len(artifact.directory.files), 1)
            for obj in artifact.objects: self.assertEqual(len(artifact.read_object(obj.id)), obj.bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
