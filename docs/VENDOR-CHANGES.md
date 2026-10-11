# Converter source provenance

Source: local NInferEZ Engine checkout, revision
`51cdf17108d28d7b772dfcc56a80dcf5de517291`, derived from the upstream lineage
recorded in `vendor/UPSTREAM.md`. Included modules: tools/convert, tools/artifact,
chat templates, the official proposal ranking and the v2 upgrade tool. Original
engine source/build trees were not changed for this project.

Local modifications:

- `vendor/tools/artifact/writer.py`: use Windows atomic no-overwrite rename instead
  of hard links, so Portable output publication works on NTFS and FAT/exFAT targets.
- `vendor/tools/convert/sources/gguf_source.py`: release the GGUF memory mapping
  explicitly on context exit so users can move/delete the input after conversion.
- `vendor/tools/upgrade_ninfer_v2_to_v3.py`: chunk progress/cancellation hooks and
  Windows no-overwrite rename, plus an optional maximum shard size. Default CLI
  behavior remains unchanged.
- `vendor/tools/convert/sources/safetensors.py`: optional external configuration
  and shard-location overrides without changing source files/indexes; UTF-8 BOM
  metadata and case-insensitive weight extensions.
- `vendor/tools/convert/__main__.py`: optional shard overrides propagated into
  named sources; default upstream callers keep their existing behavior.
- `vendor/tools/convert/resources.py`: strip an optional UTF-8 BOM from packaged
  resource bytes while preserving the original input file.

The new backend uses the existing codecs and architecture builder. Its added
dimension-driven policies, GGUF adapter and transactional publication are in
`src/backend`; no CUDA kernels or source quantization mathematics were changed.
