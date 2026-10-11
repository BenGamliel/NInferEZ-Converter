# Standalone backend API

`NInferEZ.Converter.Backend.exe` is independent of the GUI. Commands are
`capabilities`, `probe`, `analyze`, `inspect`, `convert`, `validate`, and `raw`.
For JSON commands, supply one JSON object through stdin (one line), or use
`--request PATH.json`. All JSON commands emit newline-delimited UTF-8 JSON on stdout;
human diagnostics and tracebacks go to stderr. No network listener is required.

The backend supports multiple independent processes. Each conversion owns its own
staging directory. A successful entry file is published last, and an existing output
or shard is never replaced. Destination-directory write access and free space are required.

## Request fields

| Field | Type / default | Purpose |
| --- | --- | --- |
| `source` | required string | Checkpoint folder, Safetensors/index/config file, GGUF, or `.ninfer` entry |
| `output` | required for conversion | New `.ninfer` destination |
| `profile` | `auto` | `auto`, `q4`, `q5`, `q6`, `q8`, `bf16`, `fp8`, `preserve`, `recipe` |
| `configFolder` / `model` | string | Matching config/tokenizer folder for GGUF |
| `configFile` | string | Individual matching config.json override, including Safetensors |
| `resourceFolder` | string | Default frontend resource folder; individual overrides take precedence |
| `shardOverrides` | object | Expected absolute Safetensors shard filename → actual file location |
| `components` | `["text"]` | Any selection containing text and optional vision/mtp/dflash/dflash2 |
| `sources` | object | Named paths: quantized, vision, mtp, dflash, dflash2; additional names for recipes |
| `recipe` | string | Built-in recipe or custom `FILE.py[:function]`, with profile=recipe |
| `override` | string | Optional Python recipe applied after the base and proposal |
| `resources` | object | Frontend role-to-path overrides, e.g. chat_template.jinja |
| `device` | `cpu` | `cpu`, `auto`, or `cuda` (only with a separately configured CUDA-enabled runtime) |
| `rowsPerChunk` | 512 | Bounded conversion chunk; 1–65536 |
| `maxFileBytes` | 32000000000 | Maximum output shard size |
| `name` | output stem | Public model name in artifact metadata |
| `proposal` | false | Add proposal head using the bundled official token ranking |
| `proposalRows` | 131072 | Number of proposal rows, bounded by the actual tokenizer domain |
| `ranking` | string | Optional replacement token ranking file |
| `validateEngine` | string | Matching engine EXE; verify text generation before publishing |
| `validateTimeoutSeconds` | 600 | Timeout for the optional conversion-time engine test |
| `tensorOverrides` | array | Ordered per-parameter/wildcard assignments applied after profile/proposal/Python override |
| `expectedPlanSignature` | string | Reject conversion if its prepared plan differs from the reviewed plan |
| `expectedFileState` | object | Scanner file size/timestamp records; changed dependencies require rescan |

`probe` returns `ready`, source encodings/size, a `requirements` checklist and
issues without producing model files. Missing/invalid files are data, not a
successful conversion. Requirements have a `key`, `kind`, `path`, `status` and
help text. Keys are `configFile`, `resource:ROLE`, `source:NAME` or
`shard:EXPECTED_ABSOLUTE_PATH`.

`analyze` first probes, then prepares Auto/Preserve/Q4/Q5/Q6/Q8/BF16/FP8 separately
and includes a configured Recipe option. Its `options` contain `supported`, a
plan or failure reason, and qualitative precision warnings. A `needsInput` result
with no options asks the UI to collect required files. A specific unsupported
recipe can request an additional named source while other options remain usable.
Existing v3 NInfer artifacts offer repacking only; official v2 artifacts offer
upgrade. Quantization profiles for `.ninfer` inputs are rejected explicitly.

Plans include `tensorPlan`, `operations`, `formats`, `encodedBytes` and a
`planSignature`. Sizes cover encoded weights, excluding final file overhead and
inference memory. Physical object counts are not logical parameter counts.
These plans do not run a quality benchmark or GPU operations. File-state checks
detect size/timestamp changes; they are not cryptographic hashes of all weights.

Example manual assignment:

```json
{"tensorOverrides": [{"selector": "text/layers/*/mlp/down", "format": "q8_g32_fp16",
  "method": "grouped_absmax", "layout": "auto", "activationPolicy": "A16Only", "parameters": {}}]}
```

Registered format/method choices are listed by `capabilities`. Shape, consumer,
source-encoding and layout restrictions still apply. Python recipes retain the
full upstream source/method API. Existing NInfer artifacts retain their embedded
components/resources; their repack/upgrade path does not apply such overrides.

Example `convert.json`:

```json
{
  "source": "D:\\Models\\my-checkpoint",
  "output": "D:\\Models\\my-model.ninfer",
  "profile": "auto",
  "components": ["text"],
  "device": "cpu",
  "rowsPerChunk": 512
}
```

GGUF example:

```json
{
  "source": "D:\\Models\\model.gguf",
  "configFolder": "D:\\Models\\matching-config-and-tokenizer",
  "output": "D:\\Models\\model.ninfer",
  "profile": "auto",
  "components": ["text", "vision"],
  "sources": {"vision": "D:\\Models\\mmproj.gguf"}
}
```

For `validate`, pass `{ "source": "D:\\Models\\model.ninfer", "engine":
"D:\\Engine\\ninfer-serve.exe" }`. An optional `timeoutSeconds` defaults to 600.
The selected engine is launched on a loopback port with a small explicit KV capacity,
and terminated after the test. Only the process launched by this backend is stopped.
It tests text inference, not all optional workload paths.

## Events and exit codes

| Event | Meaning |
| --- | --- |
| `stage` | Actual current phase/weight name; may be indeterminate |
| `progress` | Actual bytes written, total encoded bytes and percent |
| `compatibility` | Prepared conversion plan, components, formats and encoded weight estimate |
| `option` | One supported or unsupported candidate during analysis |
| `validation` | Real text inference result in caller-selected engine |
| `result` | Successful completion; output/report paths for conversion |
| `error` | Failure with message and exception class |
| `cancelled` | Cancellation completed; unpublished work cleaned |

Exit 0 means success, 1 failure, 2 cancellation. `runtimeVerified` is false unless
real engine validation passed. The saved `.conversion.json` report records this distinction.
The progress bar can reach 100% for writing before verification/publication finishes;
the `result` event is the completion signal.

Send a subsequent line `cancel` through the same stdin pipe to cancel. Ctrl+C is also
handled by the console host. Do not kill a running process if graceful cleanup is needed;
abrupt OS termination can leave a `.ezc-*` staging folder. Such a folder belongs to the
interrupted job, while existing model files remain unchanged.

`raw` passes remaining arguments to the upstream CLI; its output is the upstream human
format and does not use the JSON event/publication layer. The frontend uses the JSON
commands. Custom recipes can use every bundled converter method and artifact codec.
