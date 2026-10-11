# Supported conversion inputs

Support is determined by the source adapters and target architecture. A recognized container does not guarantee compatibility with every model in that container.

## Safetensors

Select the complete folder or shard index and supply matching configuration, tokenizer and chat-template resources. Missing indexed shards may be attached from other locations. Compressed sources need an adapter that understands their encoded tensors and quantization metadata.

## GGUF

Supply a supported GGUF and matching HF configuration/tokenizer resources. Unrelated metadata can produce an incompatible plan. When supported by the recipe, existing encoded blocks can be imported without quantizing the trunk again.

## Native artifacts

Supported v3 artifacts can be inspected and repacked. Older formats require a qualified importer; renaming an extension is not conversion. A valid artifact is not universal GPU qualification.

## Components and validation

Vision, MTP, DFlash and DFlash2 are included only when selected and backed by compatible sources. The wizard identifies required companion files. Text-only output does not acquire these features merely because the source family supports them.

Advanced mode exposes real backend options and trusted custom Python recipes. Recipes execute code with user permissions. Structural checks run on CPU; optional inference testing requires a compatible engine/GPU and sufficient VRAM. Successful conversion alone does not establish answer quality or runtime performance.
