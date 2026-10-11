"""Portable conversion API. No dependency on the graphical application."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import torch

from tools.artifact.reader import Artifact
from tools.artifact.inspect import artifact_summary
from tools.artifact.schema import ResourceSpec, TensorSpec, TensorObject, plan_objects
from tools.artifact.writer import ArtifactWriter
from tools.artifact.tensor_output import TensorOutput
from tools.convert.__main__ import SourceInputs, _function
from tools.convert.official_recipes import RECIPES, _dense_groupwise, _optional, Q4, Q5, Q6, Q8, FP8
from tools.convert.gguf_blocks import RECIPES as GGUF_RECIPES
from tools.convert.ternary import RECIPES as TERNARY_RECIPES
from tools.convert.qwen3_5 import build_model, text_config
from tools.convert.recipe import Recipe
from tools.convert.methods import cast_direct, fp8_row_maxabs, import_encoded, grouped_absmax
from tools.convert.proposal import add_official_proposal
from tools.convert.sources.safetensors import SafetensorsSource

VERSION = "1.0.0"
ENGINE_REVISION = "51cdf17108d28d7b772dfcc56a80dcf5de517291"
PROFILES = ["auto", "q4", "q5", "q6", "q8", "bf16", "fp8", "preserve", "recipe"]
RECIPES_ALL = sorted({**RECIPES, **GGUF_RECIPES, **TERNARY_RECIPES})


class Cancelled(Exception):
    pass


@dataclass
class Context:
    emit: object
    cancel: threading.Event

    def check(self):
        if self.cancel.is_set():
            raise Cancelled("Conversion cancelled. No incomplete model was published.")

    def stage(self, message):
        self.check()
        self.emit({"type": "stage", "message": message})


def capabilities():
    from tools.artifact.formats import NUMERIC_FORMATS
    return {
        "version": VERSION, "engineRevision": ENGINE_REVISION,
        "architectures": ["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
                          "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
        "profiles": PROFILES, "recipes": RECIPES_ALL,
        "components": ["text", "vision", "mtp", "dflash", "dflash2"],
        "inputs": ["Safetensors checkpoint/folder/index", "GGUF + HF config/tokenizer folder",
                   "v3 .ninfer (lossless repack)", "official v2 .ninfer (upgrade)"],
        "devices": ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
        "formats": sorted(NUMERIC_FORMATS), "methods": ["cast_direct", "grouped_absmax", "grouped_mse", "fp8_row_maxabs", "import_encoded"],
        "resourceRoles": ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json"],
        "runtimeVerification": "Use validate with a compatible ninfer-serve.exe; structural checks do not run CUDA operations.",
    }


def _path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Choose {label}.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _artifact_version(path):
    with path.open("rb") as f:
        raw = f.read(8)
    if raw in (b"NINFER\x00\x02", b"NINFER\x00\x03"):
        return raw[-1]
    raise ValueError("Not a supported .ninfer entry file. Select the entry, not a shard.")


def _select_components(request):
    selected = request.get("components", ["text"])
    if isinstance(selected, str):
        selected = selected.split(",")
    if not isinstance(selected, list) or len(selected) != len(set(selected)):
        raise ValueError("components must be a unique list")
    return tuple(selected)


def _float_checkpoint_profile(model, recipe, profile):
    """Dimension-driven policies, not model-name allowlists."""
    if profile == "bf16":
        return
    if profile == "auto":
        if "num_experts" not in model.config:
            _dense_groupwise(model, recipe, Q8)
        else:
            _optional(model, recipe)
            for name, p in model.parameters.items():
                if not name.startswith("text/") or not p.projection:
                    continue
                if name.endswith(("/gdn/a_projection", "/gdn/b_projection", "/moe/router", "/moe/shared_score")):
                    continue
                fmt = (Q5 if name.endswith("/down") else Q4) if "/moe/experts/" in name else Q8
                recipe.assign(name, format=fmt, method=grouped_absmax)
        return
    fmt = {"q4": Q4, "q5": Q5, "q6": Q6, "q8": Q8, "fp8": FP8}[profile]
    method = fp8_row_maxabs if profile == "fp8" else grouped_absmax
    for name, p in model.parameters.items():
        if (not p.projection and name != "text/token_embedding") or len(p.shape) != 2:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection", "/moe/router", "/moe/shared_score")):
            continue
        # Vocabulary endpoints retain Q8 with integer profiles, as in the engine recipes.
        target = Q8 if name in ("text/token_embedding", "text/output_head") and profile != "fp8" else fmt
        recipe.assign(name, format=target, method=method)


def _preserve_checkpoint(model, recipe, store):
    """Detect encoding per logical projection; no fixed layer-count split."""
    from tools.convert.official_recipes import _moe_encoded_source
    for name, p in model.parameters.items():
        if not p.projection or len(p.shape) != 2:
            continue
        source = recipe.selections[name][0].source
        if name.startswith("text/") and ("/moe/experts/" in name or "/moe/shared/" in name):
            try:
                source = _moe_encoded_source(store, p, name)
            except ValueError:
                pass
        # Default matrix_source is lazy even for ordinary FP16, so probe the encoding.
        try:
            words = source.read_encoded(0, 1) if source.read_encoded else None
        except ValueError as e:
            if "does not provide encoded" not in str(e) and "no NVFP4 source" not in str(e):
                raise
            words = None
        if words is None:
            continue
        fmt = words.format
        direct_only = name.endswith(("/gdn/a_projection", "/gdn/b_projection", "/moe/router", "/moe/shared_score"))
        if direct_only or fmt == "nvfp4" and (p.shape[0] % 128 or p.shape[1] % 64):
            # Small projections cannot inhabit the engine's NVFP4 128-row tiles;
            # recover values into the existing BF16 route, as in the Quasar converter.
            recipe.assign(name, source=source, format="bf16", method=cast_direct)
        elif fmt == "nvfp4" and name in ("text/output_head", "text/token_embedding"):
            recipe.assign(name, source=source, format=FP8, method=fp8_row_maxabs, activation_policy="AllowA8")
        else:
            recipe.assign(name, source=source, format=fmt, method=import_encoded,
                          activation_policy="AllowA4" if fmt == "nvfp4" else "AllowA8" if fmt == FP8 else "A16Only")
    for group in model.packing_groups:
        if all("/moe/experts/" in name for name in group):
            formats = {recipe.selections[n][0].format for n in group}
            if formats == {"nvfp4"}:
                recipe.group(group)


class ConfiguredGGUF:
    """HF-shaped source over GGUF, with explicit exporter transforms and bounded expert reads."""
    def __init__(self, path, base, stack, components, vision=None, mtp=None):
        from gguf_adapter import create_store
        self.store = create_store(path, base, stack, components, vision, mtp)


def _checkpoint(request, stack, ctx):
    from planning import normalize_request, open_source
    request = normalize_request(request)
    source = _path(request.get("source"), "a model file or checkpoint folder")
    profile = request.get("profile", "auto").lower()
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}")
    components = _select_components(request)
    resources = request.get("resources", {})
    paths = {k: _path(v, f"{k} source") for k, v in request.get("sources", {}).items() if v}
    base_path = request.get("model") or request.get("configFolder")
    if source.suffix.lower() == ".gguf":
        base_path = _path(base_path or str(source.parent), "the matching config/tokenizer folder")
        base = stack.enter_context(open_source(base_path, request))
        if not base.config:
            raise ValueError("GGUF requires matching config.json and tokenizer files. Choose their folder or individual files.")
        # Preserve the established exact block/ternary route where its geometry applies.
        config = text_config(base.config, mtp="mtp" in components)
        native = config.get("hidden_size") == 5120 and config.get("num_hidden_layers") == 64 and "num_experts" not in config
        ternary = False
        if native and profile in ("auto", "preserve"):
            from tools.convert.sources.gguf import GGUFFile
            try:
                with GGUFFile(source) as probe:
                    ternary = any(info.type_name in ("PQ2_0", "PTQ1_0") for info in probe.tensors.values())
                    if not ternary:
                        from tools.convert.gguf_blocks import validate as validate_blocks
                        validate_blocks(probe)
                        if "mtp" in components and paths.get("mtp"):
                            native = False
            except (ValueError, KeyError, NotImplementedError) as error:
                if profile == "preserve":
                    raise ValueError("Exact GGUF block preservation is unavailable: " + str(error)) from error
                native = False
        if profile == "recipe" or native and profile in ("auto", "preserve"):
            paths.setdefault("gguf", source)
            inputs = SourceInputs(base, paths, stack, file_overrides=request.get("shardOverrides"))
            if profile == "recipe":
                recipe_name = request.get("recipe", "")
            else:
                if ternary:
                    paths["ternary"] = source
                    inputs = SourceInputs(base, paths, stack, file_overrides=request.get("shardOverrides"))
                recipe_name = "bonsai2_27b_ternary" if ternary else "qwen3_8_27b_gguf"
            companions = {k: inputs[k] for k in ("dflash", "dflash2", "mtp") if k in components and k in paths}
            model = build_model(base, components=components, companions=companions, resource_overrides=resources)
            recipe = Recipe(model)
            _function(recipe_name)(model, recipe, inputs)
        else:
            vision_gguf = paths.get("vision") if paths.get("vision") and paths["vision"].suffix.lower() == ".gguf" else None
            mtp_gguf = paths.get("mtp") if paths.get("mtp") and paths["mtp"].suffix.lower() == ".gguf" else None
            base = ConfiguredGGUF(source, base, stack, components, vision_gguf, mtp_gguf).store
            inputs = SourceInputs(base, paths, stack)
            companions = {k: inputs[k] for k in ("dflash", "dflash2") if k in components}
            if "mtp" in components:
                companions["mtp"] = inputs["mtp"] if "mtp" in paths and mtp_gguf is None else base
            model = build_model(base, components=components, companions=companions, resource_overrides=resources)
            recipe = Recipe(model)
            if "vision" in components and "vision" in paths and vision_gguf is None:
                for name in model.parameters:
                    if name.startswith("vision/"):
                        recipe.assign(name, source=model.source(name, inputs["vision"]))
            # Re-encoding arbitrary GGUF geometries uses the same architecture's value transforms.
            # Exact block copies are exposed through the geometry-qualified official recipe.
            if profile == "preserve":
                raise ValueError("Exact GGUF block preservation requires a geometry-qualified recipe. Choose Auto to re-encode this geometry, or supply a custom recipe.")
            _float_checkpoint_profile(model, recipe, profile)
            recipe_name = "generic_gguf_" + profile
    else:
        if source.is_file() and source.name == "config.json":
            source = source.parent
        base = stack.enter_context(open_source(source, request))
        if not base.config:
            raise ValueError("The checkpoint needs config.json beside its Safetensors/index file.")
        inputs = SourceInputs(base, paths, stack, file_overrides=request.get("shardOverrides"))
        companions = {k: inputs[k] for k in ("dflash", "dflash2", "mtp") if k in components and k in paths}
        model = build_model(base, components=components, companions=companions, resource_overrides=resources)
        recipe = Recipe(model)
        if profile == "recipe":
            recipe_name = request.get("recipe", "")
            if not recipe_name:
                raise ValueError("Choose a built-in recipe or a custom Python recipe.")
            _function(recipe_name)(model, recipe, inputs)
        else:
            recipe_name = "architecture_" + profile
            if "vision" in components and "vision" in paths:
                from tools.convert.gguf_blocks import _vision_source
                vision_store = _vision_source(model, inputs)
                for name in model.parameters:
                    if name.startswith("vision/"):
                        recipe.assign(name, source=model.source(name, vision_store))
            # For a separate quantized source, apply the same per-tensor encoding policy.
            selected = inputs["quantized"] if "quantized" in paths else base
            if selected is not base:
                for name in model.parameters:
                    if name.startswith("text/"):
                        recipe.assign(name, source=model.source(name, selected))
            quantized = any(selected.describe(n).dtype in ("U8", "F8_E4M3") for n in selected.weight_map if n.endswith((".weight", ".weight_packed")))
            if profile == "preserve" or profile == "auto" and quantized:
                _preserve_checkpoint(model, recipe, selected)
            else:
                _float_checkpoint_profile(model, recipe, profile)
    if request.get("proposal"):
        options = {"rows": int(request.get("proposalRows", 131072))}
        if request.get("ranking"):
            options["ranking"] = _path(request["ranking"], "token ranking file")
        add_official_proposal(recipe, **options)
    if request.get("override"):
        _function(request["override"])(model, recipe, inputs)
    for entry in request.get("tensorOverrides", []):
        if not isinstance(entry, dict) or not entry.get("selector"):
            raise ValueError("Each tensor override requires a selector")
        allowed = {"selector", "format", "layout", "method", "parameters", "activationPolicy"}
        if set(entry) - allowed:
            raise ValueError("Unknown tensor override fields: " + str(sorted(set(entry) - allowed)))
        recipe.assign(entry["selector"], **{key: value for key, value in entry.items() if key in ("format", "layout", "method", "parameters") and value},
                      activation_policy=entry.get("activationPolicy") or None)
    # Check cancellation on every preflight probe and every source chunk. This
    # also covers custom conversion methods that use the public source API.
    from dataclasses import replace
    from tools.convert.sources.logical import LogicalSource
    for name, selections in recipe.selections.items():
        checked = []
        for selection in selections:
            src = selection.source
            def values(a, b, src=src):
                ctx.check()
                result = src.values(a, b)
                ctx.check()
                return result
            def encoded(a, b, src=src):
                ctx.check()
                result = src.read_encoded(a, b)
                ctx.check()
                return result
            wrapped = LogicalSource(src.shape, src.label, values, encoded if src.read_encoded else None, src.weight_divisor, src.input_divisor)
            checked.append(replace(selection, source=wrapped))
        recipe.selections[name] = checked
    ctx.check()
    return model, recipe, recipe_name


def inspect(request, ctx):
    source = _path(request.get("source"), "a model file or checkpoint folder")
    ctx.stage("Checking architecture, resources, weights and conversion plan")
    if source.suffix.lower() == ".ninfer":
        if request.get("profile", "auto") not in ("auto", "preserve"):
            raise ValueError("NInfer input supports lossless repacking or v2 upgrade only; select Auto or Preserve, not a quantization profile.")
        version = _artifact_version(source)
        if version == 2:
            from tools.upgrade_ninfer_v2_to_v3 import read_v2, make_directory
            directory, _, _ = read_v2(source)
            make_directory(directory["identity"], directory["objects"])
            result = {"compatible": True, "inputVersion": 2, "mode": "upgrade", "runtimeVerified": False}
        elif version == 3:
            with Artifact(source) as artifact:
                result = {"compatible": True, "mode": "repack", **artifact_summary(artifact), "runtimeVerified": False}
        else:
            raise ValueError(f"Unsupported NInfer version {version}")
    else:
        with ExitStack() as stack:
            model, recipe, recipe_name = _checkpoint(request, stack, ctx)
            prepared = recipe.prepare(device="cpu", rows_per_chunk=_rows(request))
            ctx.check()
            objects = plan_objects([w.spec for w in prepared.weights])
            sizes = {o.id: o.bytes for o in objects}
            tensor_plan = [{"parameters": list(w.parameters), "shape": list(w.spec.shape), "format": w.spec.format,
                            "method": w.method_name, "methodParameters": w.method_parameters, "encodedBytes": sizes[w.spec.id]} for w in prepared.weights]
            result = {
                "compatible": True, "mode": "convert", "architecture": model.config["architectures"][0],
                "recipe": recipe_name, "components": list(model.components), "parameters": len(model.parameters),
                "weightObjects": len(prepared.weights), "encodedBytes": sum(o.bytes for o in objects),
                "formats": dict(Counter(w.spec.format for w in prepared.weights)), "runtimeVerified": False,
                "operations": dict(Counter(w.method_name for w in prepared.weights)), "tensorPlan": tensor_plan,
            }
            identity = {"recipe": recipe_name, "components": model.components, "tensors": tensor_plan, "bindings": prepared.bindings, "uses": prepared.uses,
                        "resources": {k: hashlib.sha256(v).hexdigest() for k, v in model.resources.items()}}
            result["planSignature"] = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    if "planSignature" not in result:
        result["planSignature"] = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()
    result["message"] = "Conversion plan is valid. GPU execution has not been verified; use Test in engine after conversion."
    ctx.emit({"type": "compatibility", **result})
    return result


def probe(request, ctx):
    from planning import probe as run
    return run(request, ctx)


def analyze(request, ctx):
    from planning import analyze as run
    return run(request, ctx)


def _rows(request):
    value = int(request.get("rowsPerChunk", 512))
    if not 1 <= value <= 65536:
        raise ValueError("rowsPerChunk must be between 1 and 65536")
    return value


def _maximum_file(request):
    value = int(request.get("maxFileBytes", 32_000_000_000))
    if value < 8192:
        raise ValueError("maxFileBytes must be at least 8192")
    return value


def _native_inspect(path, ctx):
    root = Path(__file__).resolve().parents[2]
    exe = root / "app" / "inspector" / "ninfer-inspect.exe"
    if not exe.is_file():
        return None
    ctx.check()
    # The upstream inspector serializes path.string() as UTF-8; an absolute
    # Unicode Windows path can instead contain ACP bytes. Inspect relative to
    # the staging directory with its ASCII entry name.
    result = subprocess.run([str(exe), "--model", path.name, "--json"], cwd=path.parent,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if result.returncode:
        raise ValueError(f"Engine inspector rejected the converted file (exit {result.returncode}): " + (result.stderr or result.stdout).strip())
    summary = json.loads(result.stdout)
    summary["path"] = str(path)
    return summary


def _progress_writer(writer, ctx, total):
    """Wrap actual byte writes, including row-chunk writes inside large matrices."""
    original = writer.write_region
    written = 0
    last = 0.0
    def write(object_id, offset, data):
        nonlocal written, last
        ctx.check()
        original(object_id, offset, data)
        written += len(data)
        now = time.monotonic()
        if now - last > .15 or written >= total:
            ctx.emit({"type": "progress", "phase": "writing", "writtenBytes": written, "totalBytes": total,
                      "percent": min(100, round(100 * written / max(1, total), 2))})
            last = now
    writer.write_region = write


def _write_checkpoint(request, output, ctx):
    device = request.get("device", "cpu")
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("This portable runtime is CPU-only. Install the optional CUDA runtime before choosing CUDA.")
    with ExitStack() as stack:
        model, recipe, recipe_name = _checkpoint(request, stack, ctx)
        ctx.stage("Preparing weights and output layout")
        prepared = recipe.prepare(device=device, rows_per_chunk=_rows(request))
        ctx.check()
        specs = [ResourceSpec(k, len(v)) for k, v in model.resources.items()] + [w.spec for w in prepared.weights] + [s for s, _ in prepared.auxiliaries]
        provenance = {"converter": "NInferEZ Converter", "version": VERSION, "engine_revision": ENGINE_REVISION,
                      "recipe": recipe_name, "profile": request.get("profile", "auto")}
        with ArtifactWriter(output, specs, components=model.components, bindings=prepared.bindings, uses=prepared.uses,
                            metadata={"name": request.get("name") or output.stem}, provenance=provenance,
                            max_file_bytes=_maximum_file(request)) as writer:
            _progress_writer(writer, ctx, sum(o.bytes for o in writer.by_id.values()))
            for k, v in model.resources.items():
                writer.write_object(k, v)
            for i, w in enumerate(prepared.weights):
                ctx.stage(f"{i + 1}/{len(prepared.weights)}: {w.parameters[0]} ({w.spec.format})")
                w.prepared.produce(TensorOutput(writer, w.spec.id))
            for spec, data in prepared.auxiliaries:
                writer.write_object(spec.id, data)
            ctx.check()
        return provenance


def _repack(source, output, request, ctx):
    with Artifact(source) as artifact:
        specs = [TensorSpec(o.id, o.shape, o.format, o.layout, o.divisors) if isinstance(o, TensorObject) else ResourceSpec(o.id, o.bytes, o.encoding) for o in artifact.objects]
        with ArtifactWriter(output, specs, components=artifact.directory.components,
                            bindings=artifact.directory.bindings, uses=artifact.directory.uses,
                            metadata=artifact.directory.metadata, provenance=artifact.directory.provenance,
                            max_file_bytes=_maximum_file(request)) as writer:
            _progress_writer(writer, ctx, sum(obj.bytes for obj in artifact.objects))
            for obj in artifact.objects:
                ctx.stage(f"Copying {obj.id} without changing its encoding")
                for begin in range(0, obj.bytes, 4 * 1024 * 1024):
                    ctx.check()
                    count = min(4 * 1024 * 1024, obj.bytes - begin)
                    writer.write_region(obj.id, begin, artifact.read_range(obj.offset + begin, count))


def convert(request, ctx):
    source = _path(request.get("source"), "a model file or checkpoint folder")
    target = Path(request.get("output", "")).expanduser()
    if not request.get("output") or target.suffix.lower() != ".ninfer":
        raise ValueError("Choose an output filename ending in .ninfer")
    target = target.resolve()
    if target == source or target.exists() or Path(str(target) + ".conversion.json").exists():
        raise FileExistsError("Output exists or is the input. Choose a new filename; existing files are never overwritten.")
    target.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    # Private staging on the target filesystem. Cleanup is confined to this job's new directory.
    staging = Path(tempfile.mkdtemp(prefix=".ezc-", dir=target.parent))
    # Entry names are not embedded in the artifact directory. For Unicode
    # output names, use an ASCII staging entry and shard names, then rename
    # only the entry when publishing. This also works on FAT without hardlinks.
    staged = staging / (target.name if target.name.isascii() else f"artifact-{staging.name[5:]}.ninfer")
    published = []
    try:
        ctx.stage("Checking source before conversion")
        if source.suffix.lower() == ".ninfer" and request.get("profile", "auto") not in ("auto", "preserve"):
            raise ValueError("NInfer input supports lossless repacking or v2 upgrade only; select Auto or Preserve, not a quantization profile.")
        if request.get("expectedPlanSignature"):
            from planning import verify_dependencies
            verify_dependencies(request.get("expectedFileState", {}))
            quiet = Context(lambda event: ctx.emit(event) if event["type"] == "stage" else None, ctx.cancel)
            current = inspect(request, quiet)
            if current.get("planSignature") != request["expectedPlanSignature"]:
                raise ValueError("The conversion plan changed since it was reviewed. Scan/check again before converting.")
        provenance = {}
        if source.suffix.lower() == ".ninfer":
            version = _artifact_version(source)
            if version == 3:
                _repack(source, staged, request, ctx)
            elif version == 2:
                from tools.upgrade_ninfer_v2_to_v3 import upgrade
                upgrade(source, staged, max_file_bytes=_maximum_file(request), check_cancel=ctx.check, progress=lambda done, total: ctx.emit({"type": "progress", "phase": "upgrading", "writtenBytes": done, "totalBytes": total, "percent": round(100 * done / total, 2)}))
            else:
                raise ValueError(f"Unsupported artifact version {version}")
        else:
            provenance = _write_checkpoint(request, staged, ctx)
        ctx.stage("Verifying the output with the artifact reader and engine inspector")
        with Artifact(staged) as artifact:
            summary = artifact_summary(artifact)
            files = [staged] + [staging / f.path for f in artifact.directory.files[1:]]
        native = _native_inspect(staged, ctx)
        report = {"converter": VERSION, "engineRevision": ENGINE_REVISION, "seconds": round(time.monotonic() - start, 3),
                  "output": str(target), "summary": summary, "conversion": provenance, "nativeInspected": native is not None,
                  "runtimeVerified": False, "files": [target.name] + [f.name for f in files[1:]]}
        report["summary"]["path"] = str(target)
        if request.get("validateEngine"):
            report["runtimeValidation"] = validate({"source": str(staged), "engine": request["validateEngine"], "timeoutSeconds": int(request.get("validateTimeoutSeconds", 600))}, ctx)
            report["runtimeVerified"] = True
        report_path = staging / (target.name + ".conversion.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        # Publish shards/report first, entry last, with atomic no-overwrite semantics.
        for file in files[1:] + [report_path, files[0]]:
            ctx.check()
            destination = target if file == files[0] else target.parent / file.name
            if os.name == "nt":
                os.rename(file, destination)
            else:
                os.link(file, destination)
            published.append(destination)
        ctx.emit({"type": "result", "success": True, "output": str(target), "report": str(target) + ".conversion.json",
                  "runtimeVerified": report["runtimeVerified"], "seconds": report["seconds"], "summary": summary})
        return report
    except BaseException:
        for file in reversed(published):
            file.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging)


def validate(request, ctx):
    """Real generation in a caller-selected engine package; owns only its own child process."""
    import socket
    import queue
    import urllib.request
    # Loopback validation must not use machine/global HTTP proxy settings.
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    source = _path(request.get("source"), "a converted .ninfer file")
    exe = _path(request.get("engine"), "a compatible ninfer-serve.exe")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx.stage("Testing model loading and generation in the selected engine")
    # Work beside the model: this inherits the user's selected folder permissions,
    # avoids sandbox/private system-temp ACLs, and keeps calibration isolated.
    with tempfile.TemporaryDirectory(prefix=".ez-runtime-", dir=source.parent) as work, tempfile.TemporaryFile(mode="w+b", dir=work) as log:
        args = [str(exe), str(source), "--host", "127.0.0.1", "--port", str(port), "--max-context", "256", "--kv-capacity", "256", "--kv-dtype", "int8"]
        process = subprocess.Popen(args, cwd=work, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + int(request.get("timeoutSeconds", 600))
            base = f"http://127.0.0.1:{port}"
            last_status = 0.0
            while time.monotonic() < deadline:
                ctx.check()
                if process.poll() is not None:
                    log.seek(0)
                    raise ValueError("Engine validation failed: " + log.read().decode("utf-8", errors="replace")[-5000:])
                try:
                    with http.open(base + "/health", timeout=1) as response:
                        health = json.load(response)
                    if health.get("status") in ("ok", "ready") or health.get("ready") is True:
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() - last_status >= 5:
                    ctx.stage("Waiting for engine initialization and warmup")
                    last_status = time.monotonic()
                ctx.cancel.wait(.3)
            else:
                log.seek(0)
                raise TimeoutError("Engine did not become ready within the validation timeout: " + log.read().decode("utf-8", errors="replace")[-5000:])
            with http.open(base + "/v1/models", timeout=10) as response:
                model = json.load(response)["data"][0]["id"]
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({"model": model, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 2}).encode(), headers={"Content-Type": "application/json"})
            # Keep Cancel responsive while the server performs generation.
            response_queue = queue.Queue()
            def generate():
                try:
                    with http.open(req, timeout=120) as response:
                        response_queue.put((True, json.load(response)))
                except Exception as error:
                    response_queue.put((False, error))
            threading.Thread(target=generate, daemon=True).start()
            while True:
                ctx.check()
                try:
                    success, answer = response_queue.get(timeout=.25)
                    if not success:
                        raise answer
                    break
                except queue.Empty:
                    pass
            if not answer.get("choices"):
                raise ValueError("Engine did not return a generation result")
            ctx.check()
            result = {"runtimeVerified": True, "engine": str(exe), "model": model,
                      "tested": "text loading, warmup, prefill and decode", "optionalComponentsTested": False}
            ctx.emit({"type": "validation", **result})
            return result
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
