"""Read-only discovery, missing-file guidance, and comparison of actual conversion plans."""
from collections import Counter
from pathlib import Path
import json
import re

from core import Context, Cancelled, _path, _artifact_version, inspect, PROFILES
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject
from tools.convert.qwen3_5 import text_config
from tools.convert.resources import TEXT_RESOURCES, VISION_RESOURCES
from tools.convert.sources.safetensors import SafetensorsSource


def normalize_request(request):
    result = dict(request)
    source = Path(request.get("source", ""))
    root = source if source.is_dir() else source.parent
    config_file = request.get("configFile")
    folder = request.get("configFolder") or request.get("model")
    if config_file and not folder:
        result["configFolder"] = str(Path(config_file).parent)
    resources = dict(request.get("resources", {}))
    resource_folder = request.get("resourceFolder") or folder
    if not resource_folder and config_file:
        resource_folder = str(Path(config_file).parent)
    if resource_folder:
        roles = TEXT_RESOURCES + (VISION_RESOURCES if "vision" in request.get("components", ["text"]) else ())
        for role in roles:
            resources.setdefault(role, str(Path(resource_folder) / role))
    result["resources"] = resources
    return result


def configure_source(base, request):
    path = request.get("configFile")
    if not path and (request.get("configFolder") or request.get("model")):
        path = str(Path(request.get("configFolder") or request["model"]) / "config.json")
    if path:
        value = json.loads(_path(path, "the matching config.json").read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("config.json must be a JSON object")
        base.config = value


def open_source(path, request):
    class Config:
        config = None
    value = Config()
    if request.get("configFile") or request.get("configFolder") or request.get("model"):
        configure_source(value, request)
    return SafetensorsSource(path, config_override=value.config, file_overrides=request.get("shardOverrides"))


def _file_record(key, label, path, help_text, kind="file"):
    file = Path(path)
    present = file.is_dir() if kind == "folder" else file.is_file() and file.stat().st_size > 0
    return {"key": key, "label": label, "path": str(file), "status": "ready" if present else "missing",
            "kind": kind, "help": help_text}


def dependency_state(request):
    """Record file sizes/timestamps, including indexes and companions, without hashing large weights."""
    files = set()
    paths = [request.get(k) for k in ("source", "configFolder", "resourceFolder", "configFile", "model", "ranking")]
    paths += list(request.get("sources", {}).values()) + list(request.get("resources", {}).values())
    paths += list(request.get("shardOverrides", {}).values())
    if request.get("proposal") and not request.get("ranking"):
        from tools.convert.proposal import DEFAULT_RANKING
        paths.append(str(DEFAULT_RANKING))
    source = Path(request.get("source", ""))
    if source.is_file() and source.suffix.lower() != ".ninfer":
        # Include sibling metadata and index-selected shards of a single weight input.
        paths += [str(source.parent / role) for role in ("config.json", *TEXT_RESOURCES, *VISION_RESOURCES)]
    for value in paths:
        if not value:
            continue
        path = Path(value).resolve()
        if path.is_file():
            files.add(path)
            if path.suffix == ".ninfer":
                if _artifact_version(path) == 3:
                    with Artifact(path) as artifact:
                        files.update(path.parent / f.path for f in artifact.directory.files[1:])
            elif path.name.endswith(".index.json"):
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                files.update(path.parent / name for name in value.get("weight_map", {}).values())
        elif path.is_dir():
            # Standard sources are flat; follow explicit shard indexes when needed.
            files.update(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in (".json", ".jinja", ".safetensors", ".gguf"))
            for index in path.glob("*.safetensors.index.json"):
                value = json.loads(index.read_text(encoding="utf-8-sig"))
                files.update(path / name for name in value.get("weight_map", {}).values())
    for key in ("recipe", "override"):
        value = request.get(key, "")
        from tools.convert.__main__ import _recipe_parts
        path = Path(_recipe_parts(value)[0]) if value else None
        if path and path.is_file():
            files.add(path.resolve())
    return {str(p): {"bytes": p.stat().st_size, "mtimeNs": p.stat().st_mtime_ns} for p in sorted(files) if p.is_file()}


def verify_dependencies(state):
    for name, expected in state.items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or path.stat().st_mtime_ns != expected["mtimeNs"]:
            raise ValueError(f"Source/resource changed since the scan: {name}. Scan/check again.")


def probe(request, ctx):
    ctx.stage("Reading source headers and required files")
    source = _path(request.get("source"), "a model file or checkpoint folder")
    request = normalize_request(request)
    requirements, issues = [], []
    info = {"container": "Safetensors", "formats": {}, "alreadyQuantized": False}
    root = source if source.is_dir() else source.parent
    if source.suffix.lower() == ".ninfer":
        version = _artifact_version(source)
        info["container"] = f"NInfer v{version}"
        if version == 3:
            with Artifact(source) as artifact:
                tensors = [o for o in artifact.objects if isinstance(o, TensorObject)]
                info.update(formats=dict(Counter(o.format for o in tensors)), weightBytes=sum(o.bytes for o in tensors),
                            weightTensors=len(tensors), components=list(artifact.directory.components))
        return {"ready": True, "source": info, "requirements": [], "issues": [], "fileState": dependency_state(request),
                "message": "NInfer inputs can be repacked losslessly or upgraded from v2. Re-quantization is not supported."}
    config_file = request.get("configFile") or str(Path(request.get("configFolder") or request.get("model") or root) / "config.json")
    requirements.append(_file_record("configFile", "Model configuration", config_file,
                                    "Select config.json from this exact model/checkpoint. A tokenizer alone cannot define its architecture."))
    config = None
    if requirements[0]["status"] == "ready":
        try:
            config = json.loads(Path(config_file).read_text(encoding="utf-8-sig"))
            cfg = text_config(config, mtp="mtp" in request.get("components", []))
            info["architecture"] = cfg["architectures"][0]
            info["layers"] = cfg["num_hidden_layers"]
            info["availableComponents"] = ["text"] + (["vision"] if config.get("vision_config") else []) + (["mtp"] if cfg.get("mtp_num_hidden_layers") or config.get("mtp_num_hidden_layers") else [])
        except Exception as error:
            requirements[0].update(status="invalid", help=str(error))
            issues.append({"message": str(error), "key": "configFile"})
    if source.suffix.lower() == ".gguf":
        from tools.convert.sources.gguf import GGUFFile
        info["container"] = "GGUF"
        try:
            with GGUFFile(source) as gguf:
                tensors = list(gguf.tensors.values())
                info.update(formats=dict(Counter(i.type_name for i in tensors)), weightBytes=sum(i.nbytes for i in tensors),
                            weightTensors=len(tensors), alreadyQuantized=any(i.type_name not in ("F32", "F16", "BF16") for i in tensors))
                if any(gguf.data_offset + i.offset + i.nbytes > source.stat().st_size for i in tensors):
                    issues.append({"message": "GGUF data is truncated. Select a complete model file.", "key": "source"})
        except KeyError:
            # Input codecs need not have an exact engine block-copy codec.
            from tools.convert.sources.gguf_source import GGUFSource
            with GGUFSource(source) as raw:
                tensors = list(raw._tensors.values())
                info.update(formats=dict(Counter(i.tensor_type.name for i in tensors)), weightBytes=sum(i.data.nbytes for i in tensors),
                            weightTensors=len(tensors), alreadyQuantized=any(i.tensor_type.name not in ("F32", "F16", "BF16") for i in tensors))
    else:
        selected = request.get("sources", {}).get("quantized", str(source))
        selected = Path(selected)
        if selected.name == "config.json":
            selected = selected.parent
        # A missing/corrupt config is a requirement; defer weight probing until it is corrected.
        if config is not None and requirements[0]["status"] == "ready":
            with SafetensorsSource(selected, config_override=config, file_overrides=request.get("shardOverrides")) as store:
                tensors = []
                for original, file in store.shards.items():
                    if not file.is_file() or file.stat().st_size == 0:
                        requirements.append(_file_record("shard:" + original, "Missing weight shard: " + Path(original).name, file,
                                                        "Choose the exact missing Safetensors shard. The index and original files are not modified."))
                for name in store.weight_map:
                    ctx.check()
                    if store.weight_map[name].is_file() and store.weight_map[name].stat().st_size > 0:
                        tensors.append(store.describe(name))
                info.update(formats=dict(Counter(i.dtype for i in tensors)), weightBytes=sum(i.bytes for i in tensors),
                            weightTensors=len(tensors), alreadyQuantized=any(i.dtype in ("F8_E4M3", "U8", "I8") for i in tensors))
                if not tensors and not request.get("recipe") and not any(r["key"].startswith("shard:") for r in requirements):
                    issues.append({"message": "No Safetensors weights found. Select the weights/index or the checkpoint folder.", "key": "source"})
    resource_root = Path(request.get("resourceFolder") or request.get("configFolder") or Path(config_file).parent)
    roles = TEXT_RESOURCES + (VISION_RESOURCES if "vision" in request.get("components", ["text"]) else ())
    for role in roles:
        path = request.get("resources", {}).get(role, str(resource_root / role))
        item = _file_record("resource:" + role, role, path, "Choose this file from the same model. Token IDs and template must match the weights.")
        if item["status"] == "ready" and role.endswith(".json"):
            try:
                if not isinstance(json.loads(Path(path).read_text(encoding="utf-8-sig")), dict):
                    raise ValueError("Expected a JSON object")
            except Exception as error:
                item.update(status="invalid", help=str(error))
        requirements.append(item)
    for component in ("dflash", "dflash2"):
        if component in request.get("components", []) and not request.get("sources", {}).get(component):
            requirements.append({"key": "source:" + component, "label": component.upper() + " checkpoint", "path": "", "status": "missing", "kind": "folder", "help": "This optional component requires its own compatible draft checkpoint."})
    ready = not issues and all(r["status"] == "ready" for r in requirements)
    return {"ready": ready, "source": info, "requirements": requirements, "issues": issues,
            "fileState": dependency_state(request), "message": "Ready to compare conversion plans." if ready else "Provide the missing or corrected files, then check again."}


def analyze(request, ctx):
    discovery = probe(request, ctx)
    if not discovery["ready"]:
        return {"compatible": False, "needsInput": True, **discovery, "options": []}
    source = Path(request["source"])
    profiles = ["preserve"] if discovery["source"]["container"] == "NInfer v3" else ["auto"] if source.suffix.lower() == ".ninfer" else ["auto", "preserve", "q4", "q5", "q6", "q8", "bf16", "fp8"]
    if request.get("recipe") and source.suffix.lower() != ".ninfer":
        profiles.append("recipe")
    options, missing = [], []
    quiet = Context(lambda e: ctx.emit(e) if e["type"] == "stage" else None, ctx.cancel)
    for index, profile in enumerate(profiles):
        ctx.stage(f"Preparing option {index + 1}/{len(profiles)}: {profile.upper()}")
        try:
            plan = inspect({**request, "profile": profile}, quiet)
            operations = plan.get("operations", {})
            requantizes = any(k in operations for k in ("grouped_absmax", "grouped_mse", "fp8_row_maxabs"))
            warnings = []
            if requantizes:
                warnings.append("Quantization can reduce answer quality; its effect cannot be measured by this structural scan.")
            if discovery["source"]["alreadyQuantized"] and requantizes:
                warnings.append("The source is already quantized. Re-encoding cannot restore lost precision and may introduce further error, even at a higher bit-width.")
            if "cast_direct" in operations:
                warnings.append("Some parameters are stored directly or cast to the engine's required precision. Preserve does not imply every original tensor is copied byte-for-byte.")
            if "import_encoded" in operations:
                warnings.append("Encoded-import weights preserve their supported numeric encoding; layout and resource normalization can still change.")
            option = {"profile": profile, "supported": True, "plan": plan, "warnings": warnings,
                      "quality": "Possible precision loss" if requantizes or "cast_direct" in operations else "Existing encoding retained"}
        except Cancelled:
            raise
        except Exception as error:
            text = str(error)
            match = re.search(r"requires source '([^']+)'", text)
            if match:
                key = "source:" + match[1]
                if not any(r["key"] == key for r in missing):
                    missing.append({"key": key, "label": match[1] + " source", "path": "", "status": "missing", "kind": "fileOrFolder", "help": text})
            option = {"profile": profile, "supported": False, "reason": text, "errorClass": type(error).__name__}
        ctx.check()
        options.append(option)
        ctx.emit({"type": "option", **option})
        ctx.emit({"type": "progress", "phase": "analysis", "percent": round(100 * (index + 1) / len(profiles), 2)})
    return {**discovery, "compatible": any(o["supported"] for o in options), "options": options,
            "requirements": discovery["requirements"] + missing, "runtimeVerified": False,
            "message": "Select and review a supported option. Weight sizes exclude file overhead and inference VRAM; no quality/GPU benchmark has been run."}
