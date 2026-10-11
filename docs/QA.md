# Release validation — 2026-10-11

NInferEZ Converter 1.0.0, Windows x64, app-local Python 3.12.10, CPU PyTorch
2.8.0+cpu and self-contained .NET 10.0.9. Converter sources are pinned to engine
revision `51cdf17108d28d7b772dfcc56a80dcf5de517291`.

The 1.0.0 release keeps the previously qualified conversion backend and UI behavior. Both Windows host executables were rebuilt with product version 1.0.0; the existing 18 CPU-only regression tests passed again for this release. Historical GPU checks below belong to the preceding conversion qualification, not a new GPU run. No GPU conversion or inference test was performed during release packaging.

The organized release has four root files and four functional directories.
Both EXEs embed their compressed .NET runtime; no loose .NET DLLs or translation
directories remain beside them. The final relocation check also asserts this
layout and exercises first-launch native runtime extraction inside a job-owned
cache with inherited user access.
Flat Python standard-library modules and encodings are bundled in Python's
canonical `runtime/python/python312.zip`; package/resource directories retain
their normal layout. Build-machine bytecode caches are removed.

## Automated and desktop checks

The approved converter artwork is embedded in both executable icons and in the
frontend's header and window icons. Its transparent PNG is retained under
`assets/branding` in the source project; the ICO includes ten sizes from 16 to
256 pixels. Windows-extracted 32-pixel icons from both published EXEs match the
source ICO pixel for pixel. The rebuilt frontend was launched and its rendered
header/window icon reviewed (`build/icon-frontend.png`). Branding is bundled
inside the EXEs and adds no loose files to the Portable root.

Backend tests using the packaged Python runtime cover:

The final backend suite passes 18 tests.

- Actual complete Dense source tensors through every exposed output profile,
  including per-chunk byte progress and variable dimensions.
- MoE expert bindings with optional Vision and MTP components.
- Independent HF versus GGUF tensor values, including GDN inverse exporter
  transforms, Dense and MoE expert-bank mappings.
- Mixed GPTQ/ModelOpt NVFP4 and FP8 source encodings, exact preserved code/scales,
  and small GDN projections promoted to the engine's direct BF16 route.
- Separate quantized text and Vision sources, retaining the base/companion MTP
  source and the selected text's encoded NVFP4 representation.
- Lossless sharded v3 repacking, comparing every object's bytes and bindings.
- v2 upgrade layout boundaries honor the selected shard limit; invalid quantization
  profiles are rejected for existing NInfer inputs at plan-check time.
- Rejection of an unrelated architecture, corrupt input, and existing destinations.
- Cancellation during writing, rollback and removal of unpublished staging.
- Native backend EXE inspection and full conversion, a caller-relative JSON request,
  Unicode model/output paths and readable multiple output shards.

The native WPF frontend completed its compatibility and conversion workflow
against a real fixture through the independent backend EXE, including a Unicode
output name. Completion, output existence, enabled controls and 100% progress were
checked. The rendered window was visually reviewed for clipping and overflow.

Wizard/manual coverage includes missing external config/tokenizer recovery,
read-only missing indexed-shard attachment, actual profile previews and exceptions,
source/plan change rejection, analysis cancellation, UTF-8 BOM resources and
uppercase weight extensions. All 16 subsets of Vision/MTP/DFlash/DFlash2 are
planned with Auto and Q4 for Dense and MoE (64 combinations). Each architecture
also compares all eight standard profiles with all optional components and a
proposal head, and completes a full structurally inspected conversion.
The GUI EXE tests explicit Wizard selection, required-file recovery, manual JSON
import, MoE/Vision/MTP, a per-weight assignment, Unicode paths and output sharding.
Wizard and manual artifact-repacking flows additionally compare all object bytes,
metadata and components with their source artifacts.
Results are in `build/wizard-backend-tests.txt` and `build/frontend-tests.json`.

The real 27B GSQ-RCO GGUF was scanned through all eight profiles. All plans were
prepared successfully, with actual mixed-format distributions and different size
estimates. This scan writes no converted model and adds no GPU qualification to
the earlier real-model conversion evidence below.
The compact record is `build/real/wizard-summary.json`: Auto/Preserve 7.83 GiB,
Q4 14.60 GiB, Q5 17.43 GiB, Q6 20.27 GiB, Q8 26.64 GiB, BF16 50.10 GiB and
FP8 25.08 GiB of encoded weights. This already-quantized source illustrates why
choosing Q4 can expand a model and cannot restore precision lost during earlier
quantization.

The release ZIP is additionally extracted into another directory. The extracted
backend and frontend are exercised from a different working directory with their
app-local runtimes. `build/relocation-results.json` records that check.

## Real model and engine

Source: `RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ2_XS.gguf`, with its matching local
HF config/tokenizer resources. Profile Auto, text component, CPU conversion.

- Source size: 8,446,433,824 bytes; converted entry: 8,419,682,048 bytes.
- 64-layer Dense architecture; 849 objects, 963 logical bindings, 609 declared uses.
- Mixed GSQ-RCO block formats were preserved through the geometry-qualified recipe.
- Conversion and structural inspection took about 31 seconds on this machine.
- The bundled inspector accepted the artifact. Inspector build ID:
  `v0.1.0-preview.1-1-g69c75330`, `sm120a`.
- A caller-selected `NInferEZ-Engine-0.1.0-preview.1-sm120a-windows-x64`
  `ninfer-serve.exe` loaded it on an NVIDIA RTX 5090 and returned a successful
  chat-completion result after warmup, prefill and decode.
- The saved report has `runtimeVerified: true`. Original model inputs and engine
  files were retained. Test models and reports are outside the distributed ZIP.

The first sandboxed runtime test could not reach its loopback engine API and
encountered private system-temp cleanup restrictions. Validation now bypasses
global proxies and uses an isolated temporary directory beside the selected model.
The final real-engine test passed outside those sandbox network restrictions.

## Limits of this evidence

This is a functional portable release, not an exhaustive matrix of every model,
quantization, GPU, driver or optional inference path. The real-model test checks
text execution, not numerical equivalence, answer quality, Vision, MTP or DFlash
runtime workloads. Those optional components have structural fixture coverage
where listed and require workload-specific GPU qualification. v2 upgrades, custom
recipes and every upstream named recipe have not each been GPU-qualified here.

A supported output codec does not make arbitrary input tensor names or containers
importable. Unknown mappings fail explicitly; custom recipes/source adapters can
extend the conversion API. Future engine architectures require updated adapters.

The EXEs are unsigned. All runtime dependencies and redistribution notices travel
with the ZIP; no Python/.NET installation, model weights or credentials are required
or included. The recipient must retain the full extracted directory layout.
