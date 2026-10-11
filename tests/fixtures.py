from __future__ import annotations
import json
from pathlib import Path
import struct
import numpy as np
import torch


def write_safetensors(path, tensors):
    header, data = {}, bytearray()
    for name, array in tensors.items():
        if isinstance(array, torch.Tensor):
            dtype = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8"}[array.dtype]
            shape, raw = list(array.shape), array.contiguous().view(torch.uint8).numpy().tobytes()
        else:
            array = np.asarray(array, dtype=np.float32)
            dtype, shape, raw = "F32", list(array.shape), array.tobytes()
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data.extend(raw)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def checkpoint(root, *, moe=False, mtp=False, vision=False, hybrid=True):
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"architectures": ["Qwen3_5MoeForCausalLM" if moe else "Qwen3_5ForCausalLM"],
        "hidden_size": 128, "vocab_size": 16, "num_hidden_layers": 2, "max_position_embeddings": 256,
        "layer_types": ["linear_attention" if hybrid else "full_attention", "full_attention"],
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 6, "linear_key_head_dim": 16,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "rope_parameters": {"partial_rotary_factor": .5, "mrope_section": [4, 2, 2]}}
    if moe:
        cfg.update(num_experts=3, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64)
    else:
        cfg["intermediate_size"] = 256
    if mtp:
        cfg["mtp_num_hidden_layers"] = 1
    if vision:
        cfg["vision_config"] = {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": 16}
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for name, value in {"tokenizer.json": {"model": {"vocab": {str(i): i for i in range(16)}}},
                        "tokenizer_config.json": {}, "generation_config.json": {},
                        "preprocessor_config.json": {"patch_size": 2, "temporal_patch_size": 2, "merge_size": 2},
                        "video_preprocessor_config.json": {"patch_size": 2, "temporal_patch_size": 2, "merge_size": 2}}.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    # Capture the architecture's physical HF source names/shapes without duplicating its mapping.
    from tools.convert.qwen3_5 import _Builder, build_model
    from tools.convert.sources.safetensors import SafetensorsSource
    collected = {}
    original = _Builder.add
    def capture(self, name, store, source_name, shape, **kwargs):
        native_shape = tuple(kwargs.get("source_shape") or shape)
        if source_name in collected and collected[source_name] != native_shape:
            raise AssertionError("inconsistent shared source shape")
        collected[source_name] = native_shape
        return original(self, name, store, source_name, shape, **kwargs)
    _Builder.add = capture
    try:
        with SafetensorsSource(root) as source:
            build_model(source, components=("text",) + (("mtp",) if mtp else ()) + (("vision",) if vision else ()))
    finally:
        _Builder.add = original
    rng = np.random.default_rng(20261010)
    tensors = {name: rng.normal(0, .02, size=shape).astype(np.float32) for name, shape in collected.items()}
    write_safetensors(root / "model.safetensors", tensors)
    return root, cfg, tensors


def draft_checkpoint(root, target_root, target_config, backend):
    """Small complete DFlash/DFlash2 sources using the official architecture builder."""
    root.mkdir(parents=True, exist_ok=True)
    h = target_config["hidden_size"]
    config = {"architectures": ["DFlash2DraftModel" if backend == "dflash2" else "DFlashDraftModel"],
              "model_type": "qwen3", "hidden_act": "silu", "attention_bias": False, "is_causal": False,
              "hidden_size": h, "vocab_size": target_config["vocab_size"], "num_target_layers": target_config["num_hidden_layers"],
              "intermediate_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
              "num_hidden_layers": 1, "max_position_embeddings": 256,
              "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 0}}
    if backend == "dflash2":
        config["dflash_config"].update(conv_kernel_size=3, conv_group_size=16, selector_rank=8, selector_top_k=8)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    from tools.convert.qwen3_5 import _Builder, build_model
    from tools.convert.sources.safetensors import SafetensorsSource
    collected, original = {}, _Builder.add
    with SafetensorsSource(target_root) as base, SafetensorsSource(root) as draft:
        def capture(self, name, store, source_name, shape, **kwargs):
            if store is draft:
                collected[source_name] = tuple(kwargs.get("source_shape") or shape)
            return original(self, name, store, source_name, shape, **kwargs)
        _Builder.add = capture
        try: build_model(base, components=("text", backend), companions={backend: draft})
        finally: _Builder.add = original
    rng = np.random.default_rng(13)
    write_safetensors(root / "model.safetensors", {name: rng.normal(0, .02, size=shape).astype(np.float32) for name, shape in collected.items()})
    return root


def checkpoint_gguf(root, cfg, hf):
    """Independent synthetic exporter; use explicit inverse transforms as the oracle."""
    from tools.convert.sources.gguf import write_gguf
    h, layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    moe = "num_experts" in cfg
    arch = "qwen35moe" if moe else "qwen35"
    tensors = {"token_embd.weight": hf["model.embed_tokens.weight"], "output.weight": hf["lm_head.weight"], "output_norm.weight": hf["model.norm.weight"] + 1}
    def tiled(a, dim, d):
        k, v = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        a = np.asarray(a)
        shape = a.shape
        expanded = list(shape[:dim]) + [k, v // k, d] + list(shape[dim+1:])
        return a.reshape(expanded).swapaxes(dim, dim + 1).reshape(shape).copy()
    for i, kind in enumerate(cfg["layer_types"]):
        p, g = f"model.layers.{i}.", f"blk.{i}."
        tensors[g + "attn_norm.weight"] = hf[p + "input_layernorm.weight"] + 1
        tensors[g + "post_attention_norm.weight"] = hf[p + "post_attention_layernorm.weight"] + 1
        if kind == "full_attention":
            for leaf, native in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output"), ("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
                value = hf[p + "self_attn." + leaf + ".weight"]
                tensors[g + native + ".weight"] = value + 1 if leaf.endswith("norm") else value
        else:
            prefix = p + "linear_attn."
            qk = 2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
            value = hf[prefix + "in_proj_qkv.weight"]
            tensors[g + "attn_qkv.weight"] = np.concatenate([value[:qk], tiled(value[qk:], 0, cfg["linear_value_head_dim"])])
            for leaf, native, d in (("in_proj_z.weight", "attn_gate.weight", cfg["linear_value_head_dim"]),
                                    ("in_proj_a.weight", "ssm_alpha.weight", 1), ("in_proj_b.weight", "ssm_beta.weight", 1), ("dt_bias", "ssm_dt.bias", 1)):
                tensors[g + native] = tiled(hf[prefix + leaf], 0, d)
            tensors[g + "ssm_a"] = tiled(-np.exp(hf[prefix + "A_log"]), 0, 1)
            tensors[g + "ssm_out.weight"] = tiled(hf[prefix + "out_proj.weight"], 1, cfg["linear_value_head_dim"])
            conv = hf[prefix + "conv1d.weight"][:, 0, :]
            tensors[g + "ssm_conv1d.weight"] = np.concatenate([conv[:qk], tiled(conv[qk:], 0, cfg["linear_value_head_dim"])])
            tensors[g + "ssm_norm.weight"] = hf[prefix + "norm.weight"]
        if moe:
            bank = hf[p + "mlp.experts.gate_up_proj"]
            ir = cfg["moe_intermediate_size"]
            tensors[g + "ffn_gate_exps.weight"], tensors[g + "ffn_up_exps.weight"] = bank[:, :ir], bank[:, ir:]
            tensors[g + "ffn_down_exps.weight"] = hf[p + "mlp.experts.down_proj"]
            for hf_leaf, gg_leaf in (("gate", "ffn_gate_inp"), ("shared_expert_gate", "ffn_gate_inp_shexp"), ("shared_expert.gate_proj", "ffn_gate_shexp"), ("shared_expert.up_proj", "ffn_up_shexp"), ("shared_expert.down_proj", "ffn_down_shexp")):
                tensors[g + gg_leaf + ".weight"] = hf[p + "mlp." + hf_leaf + ".weight"]
        else:
            for role in ("gate", "up", "down"):
                tensors[g + "ffn_" + role + ".weight"] = hf[p + "mlp." + role + "_proj.weight"]
    records = [(name, tuple(value.shape), 0, value.astype(np.float32).tobytes()) for name, value in tensors.items()]
    path = root / "fixture.gguf"
    write_gguf(path, {"general.architecture": arch, arch + ".embedding_length": h, arch + ".block_count": layers,
                     arch + ".attention.head_count": cfg["num_attention_heads"], arch + ".attention.head_count_kv": cfg["num_key_value_heads"], arch + ".attention.key_length": cfg["head_dim"]}, records)
    return path
