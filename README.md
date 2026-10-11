# NInferEZ Converter

Convert supported checkpoints into NInfer artifacts using a guided wizard or manual controls. A standalone Windows x64 desktop application and JSON backend share the same conversion engine.

**Using [NInferEZ Manager](https://github.com/BenGamliel/NInferEZ-Manager)? Its Converter page provides the same workflow without opening another application.**

[Download 1.0.0](https://github.com/BenGamliel/NInferEZ-Converter/releases/tag/v1.0.0) · [Supported inputs](docs/SUPPORTED-INPUTS.md) · [Backend API](docs/BACKEND.md) · [Engine](https://github.com/BenGamliel/NInferEZ-Engine) · [Models](https://huggingface.co/2beng2)

<img src="assets/branding/converter.png" alt="NInferEZ Converter" width="96">

## Download and run

Extract the complete [Portable ZIP](https://github.com/BenGamliel/NInferEZ-Converter/releases/download/v1.0.0/NInferEZ-Converter-1.0.0-Portable-win-x64.zip) into a writable folder and open `NInferEZ.Converter.exe`. End users do not need a separate Python, .NET or CUDA Toolkit installation.

The bundled conversion runtime is **CPU-only** and works offline on local sources. Conversion still consumes CPU, RAM and disk. Optional inference-engine testing uses that engine's GPU and is explicitly selected. The standalone release is Portable; no separate Converter installer is included.

## Guided and manual workflows

**Guided wizard:** select a source, attach matching resources, inspect available options, choose a prepared plan and review the destination. Missing files and unsupported choices are explained before conversion.

**Manual setup:** profiles, components, source mappings, recipes, tensor policies, chunk/shard limits and optional engine validation. Request import/export preserves advanced options.

Both modes use the existing backend with progress and cancellation. Changed sources or plans require a fresh compatibility check. Custom Python recipes execute code with the user's permissions; only use recipes you trust.

## Supported conversions

Support follows source geometry and importer coverage, not a filename alone:

- Compatible Qwen3.5 Dense/MoE geometry, including supported Qwen3.6/3.8 derivatives.
- Safetensors checkpoints/indexed shards with matching configuration, tokenizer and frontend resources.
- Supported compressed sources, including native NVFP4 and row-/tensor-scaled FP8 layouts.
- Supported GGUF encodings, accompanied by matching HF configuration/tokenizer resources.
- Native v3 repacking and qualified older-container import paths.
- Optional vision, MTP and DFlash/DFlash2 when compatible companion sources are supplied.

See [supported inputs](docs/SUPPORTED-INPUTS.md) and the backend's prepared plans for exact choices. Output-format support does not imply an importer exists for every architecture. Unsupported mappings are rejected rather than guessed.

Higher-bit output cannot restore information already lost in compressed sources. Output-size estimates are not quality scores, speed predictions or total VRAM estimates.

## Independent backend

```powershell
.\NInferEZ.Converter.Backend.exe capabilities
.\NInferEZ.Converter.Backend.exe probe --request check.json
.\NInferEZ.Converter.Backend.exe analyze --request check.json
.\NInferEZ.Converter.Backend.exe convert --request convert.json
.\NInferEZ.Converter.Backend.exe validate --request validate.json
```

See [BACKEND.md](docs/BACKEND.md) for schemas and NDJSON events. The EXE locates its runtime relative to itself; callers need not use a particular working directory.

## Validation and source

Structural validation uses the artifact reader and a host-only engine inspector. It is distinct from GPU execution and answer-quality evaluation. See [QA](docs/QA.md).

Building requires .NET SDK 10, and Python 3.12 plus `uv` when preparing a new runtime. Pass `-PythonSource` explicitly to `scripts/build.ps1 -PrepareRuntime`; the build script contains no personal default path. `scripts/package.py` writes manifests, checksums and the archive.

## License and credits

Application additions and the engine-derived converter use [Apache-2.0](LICENSE). The vendored converter is pinned to Engine revision `51cdf17108d28d7b772dfcc56a80dcf5de517291`; [vendor changes](docs/VENDOR-CHANGES.md) record adaptations and retain upstream authorship.

Packages include Python, PyTorch, NumPy, GGUF, .NET, native-inspector and runtime licenses/notices. Converted weights retain their original model licenses. NInferEZ is an independent distribution.
