"""Dimension-driven GGUF adapter for the engine's Dense and MoE architectures.

Uses the upstream, verified dense/GDN/MTP mappings. Expert banks use bounded
virtual gate/up views; their entire concatenated bank is never materialized.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import prod
from pathlib import Path
import torch

from tools.convert.qwen3_5 import text_config
from tools.convert.sources.gguf_source import (
    GGUFSource, HFAliasSource, standard_dense_name_map, qwen35_linear_attention_name_map,
    qwen35_mtp_name_map, qwen35_vision_name_map,
)


@dataclass
class Info:
    shape: tuple
    dtype: str = "F32"


class MappedStore:
    def __init__(self, base, aliases, virtual, config):
        self.path, self.root, self.config = base.path, base.root, config
        self.base, self.aliases, self.virtual = base, aliases, virtual
        self.weight_map = {name: self.path for alias in aliases for name in alias._names}
        self.weight_map.update({name: self.path for name in virtual})

    def has(self, name):
        return name in self.virtual or any(a.has(name) for a in self.aliases) or self.base.has(name)

    def describe(self, name):
        if name in self.virtual:
            return Info(self.virtual[name][0])
        for alias in self.aliases:
            if alias.has(name):
                return Info(alias.describe(name).shape)
        return self.base.describe(name)

    def read_flat(self, name, begin=0, end=None):
        if name in self.virtual:
            shape, read = self.virtual[name]
            end = prod(shape) if end is None else end
            if not 0 <= begin <= end <= prod(shape):
                raise ValueError(f"Invalid GGUF source range for {name}")
            return read(begin, end)
        for alias in self.aliases:
            if alias.has(name):
                return alias.read_flat(name, begin, end)
        return self.base.read_flat(name, begin, end)


def _merge(target, update):
    for a, b in zip(target, update):
        a.update(b)


def _expert_virtual(raw, gate, up, down, e, intermediate, h):
    for name, shape in ((gate, (e, intermediate, h)), (up, (e, intermediate, h)), (down, (e, h, intermediate))):
        if raw.describe(name).shape != shape:
            raise ValueError(f"{name}: expected expert bank shape {shape}, got {raw.describe(name).shape}")
    width = intermediate * h
    def read_gate_up(begin, end):
        parts = []
        while begin < end:
            expert, offset = divmod(begin, 2 * width)
            source, offset = (gate, offset) if offset < width else (up, offset - width)
            count = min(end - begin, width - offset)
            parts.append(raw.read_flat(source, expert * width + offset, expert * width + offset + count))
            begin += count
        return torch.cat(parts) if parts else torch.empty(0)
    return ((e, 2 * intermediate, h), read_gate_up), ((e, h, intermediate), lambda a, b: raw.read_flat(down, a, b))


def create_store(path, base, stack, components, vision_path=None, mtp_path=None):
    raw = stack.enter_context(GGUFSource(path))
    cfg = text_config(base.config, mtp="mtp" in components)
    metadata = raw.metadata()
    arch = metadata.get("general.architecture")
    expected_arch = "qwen35moe" if "num_experts" in cfg else "qwen35"
    if arch != expected_arch:
        raise ValueError(f"GGUF architecture {arch!r} differs from matching config architecture {expected_arch!r}")
    for ggkey, hfkey in (("embedding_length", "hidden_size"), ("attention.head_count", "num_attention_heads"),
                         ("attention.head_count_kv", "num_key_value_heads"), ("attention.key_length", "head_dim")):
        value = metadata.get(arch + "." + ggkey)
        if value is not None and value != cfg.get(hfkey):
            raise ValueError(f"GGUF {ggkey}={value} differs from config {hfkey}={cfg.get(hfkey)}")
    blocks = metadata.get(arch + ".block_count")
    if blocks not in (cfg["num_hidden_layers"], cfg["num_hidden_layers"] + 1):
        raise ValueError("GGUF layer count differs from the config")
    prefix = "model.language_model." if "text_config" in base.config else "model."
    maps = standard_dense_name_map(cfg["num_hidden_layers"], tie_word_embeddings=cfg["tie_word_embeddings"], text_prefix=prefix)
    virtual = {}
    for i, layer in enumerate(cfg["layer_types"]):
        if layer == "linear_attention":
            _merge(maps, qwen35_linear_attention_name_map(i, num_k_heads=cfg["linear_num_key_heads"],
                num_v_heads=cfg["linear_num_value_heads"], head_k_dim=cfg["linear_key_head_dim"],
                head_v_dim=cfg["linear_value_head_dim"], text_prefix=prefix))
        if "num_experts" in cfg:
            hf, gg = prefix + f"layers.{i}.mlp.", f"blk.{i}."
            bank, down = _expert_virtual(raw, gg + "ffn_gate_exps.weight", gg + "ffn_up_exps.weight",
                gg + "ffn_down_exps.weight", cfg["num_experts"], cfg["moe_intermediate_size"], cfg["hidden_size"])
            virtual[hf + "experts.gate_up_proj"] = bank
            virtual[hf + "experts.down_proj"] = down
            maps[0].update({hf + "gate.weight": gg + "ffn_gate_inp.weight",
                hf + "shared_expert_gate.weight": gg + "ffn_gate_inp_shexp.weight",
                hf + "shared_expert.gate_proj.weight": gg + "ffn_gate_shexp.weight",
                hf + "shared_expert.up_proj.weight": gg + "ffn_up_shexp.weight",
                hf + "shared_expert.down_proj.weight": gg + "ffn_down_shexp.weight"})
    # Third-party exporters use both names; source facts choose the actual one.
    for hf, gg in list(maps[0].items()):
        if not raw.has(gg) and gg.endswith("post_attention_norm.weight"):
            alternative = gg.replace("post_attention_norm.weight", "ffn_norm.weight")
            if raw.has(alternative):
                maps[0][hf] = alternative
    aliases = [stack.enter_context(HFAliasSource(raw, maps[0], transforms=maps[1], shapes=maps[2]))]
    if "mtp" in components:
        mtp_raw = stack.enter_context(GGUFSource(mtp_path)) if mtp_path else raw
        index = cfg["num_hidden_layers"] if not mtp_path else (int(mtp_raw.metadata().get(arch + ".block_count", 1)) - 1)
        mm = qwen35_mtp_name_map(index)
        aliases.append(stack.enter_context(HFAliasSource(mtp_raw, mm[0], transforms=mm[1], shapes=mm[2])))
        if "num_experts" in cfg:
            hf, gg = "mtp.layers.0.mlp.", f"blk.{index}."
            bank, down = _expert_virtual(mtp_raw, gg + "ffn_gate_exps.weight", gg + "ffn_up_exps.weight",
                gg + "ffn_down_exps.weight", cfg["num_experts"], cfg["moe_intermediate_size"], cfg["hidden_size"])
            virtual[hf + "experts.gate_up_proj"], virtual[hf + "experts.down_proj"] = bank, down
            aliases[-1]._names.update({hf + "gate.weight": gg + "ffn_gate_inp.weight",
                hf + "shared_expert_gate.weight": gg + "ffn_gate_inp_shexp.weight",
                hf + "shared_expert.gate_proj.weight": gg + "ffn_gate_shexp.weight",
                hf + "shared_expert.up_proj.weight": gg + "ffn_up_shexp.weight",
                hf + "shared_expert.down_proj.weight": gg + "ffn_down_shexp.weight"})
    if "vision" in components and vision_path:
        vc = base.config["vision_config"]
        vr = stack.enter_context(GGUFSource(vision_path))
        vm = qwen35_vision_name_map(num_blocks=vc["depth"], hidden_size=vc["hidden_size"],
            in_channels=vc.get("in_channels", 3), temporal_patch_size=vc.get("temporal_patch_size", 2), patch_size=vc["patch_size"])
        aliases.append(stack.enter_context(HFAliasSource(vr, vm[0], transforms=vm[1], shapes=vm[2], composites=vm[3])))
    return MappedStore(base, aliases, virtual, base.config)
