#!/usr/bin/env python3
"""Offline PTQ for OneRec Decoder Attention (static W8A8).

Quantizes Decoder Self / Cross Q,K,V,O per-channel to int8 and writes
`deq_scale` (FP32), `input_scale` (BF16), `input_offset` (int8).

MoE / Encoder tensors are copied unchanged.

Usage:
  python3 tools/onerec_attn_w8a8_ptq.py \\
      --input-dir graph3 --output-dir graph3_attn_w8a8 \\
      --calibrate-json rmsnorm_absmax.json

  python3 tools/onerec_attn_w8a8_ptq.py --self-test

`--placeholder-input-scale` is only for plumbing. Do not use it for
accuracy sign-off; calibrate with 2-seq x beam=512 RMSNorm outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    safe_open = None
    save_file = None


DECODER_ATTN_SUFFIXES = (
    "layer.0.SelfAttention.q.weight",
    "layer.0.SelfAttention.k.weight",
    "layer.0.SelfAttention.v.weight",
    "layer.0.SelfAttention.o.weight",
    "layer.1.EncDecAttention.q.weight",
    "layer.1.EncDecAttention.k.weight",
    "layer.1.EncDecAttention.v.weight",
    "layer.1.EncDecAttention.o.weight",
)

SHARED_INPUT_SCALE_GROUPS = (
    (
        "layer.0.SelfAttention.q.weight",
        "layer.0.SelfAttention.k.weight",
        "layer.0.SelfAttention.v.weight",
    ),
)


def is_decoder_attn_weight(name: str) -> bool:
    if not name.startswith("decoder."):
        return False
    return any(name.endswith(suffix) for suffix in DECODER_ATTN_SUFFIXES)


def quantize_per_channel_int8(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (int8_weight [n,k], fp32 per-channel weight scale [n])."""
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
    weight_fp32 = weight.detach().to(torch.float32)
    absmax = weight_fp32.abs().amax(dim=1).clamp(min=1e-8)
    scale = absmax / 127.0
    quant = torch.clamp(torch.round(weight_fp32 / scale.unsqueeze(1)), -127, 127)
    return quant.to(torch.int8), scale.contiguous()


def deq_scale(input_scale: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return (input_scale.reshape(()).to(torch.float32) * weight_scale).contiguous()


def default_input_scale_key(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".input_scale"


def resolve_input_scale(
    weight_name: str,
    calibrated: Dict[str, float],
    placeholder: Optional[float],
) -> float:
    keys = [
        default_input_scale_key(weight_name),
        weight_name,
        weight_name[: -len(".weight")],
    ]
    for key in keys:
        if key in calibrated:
            return float(calibrated[key])
    if placeholder is None:
        raise KeyError(
            f"no calibrated input_scale for {weight_name}; "
            "pass --calibrate-json or --placeholder-input-scale"
        )
    return float(placeholder)


def apply_shared_qkv_scales(scales: Dict[str, float]) -> None:
    for group in SHARED_INPUT_SCALE_GROUPS:
        present = [name for name in group if name in scales]
        if not present:
            continue
        shared = max(scales[name] for name in present)
        for name in group:
            scales[name] = shared


def iter_weight_files(model_dir: str) -> List[str]:
    files: List[str] = []
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            continue
        # graph3 also stores rec lookup tables as *.bin; only convert
        # actual model checkpoints.
        if name.endswith(".safetensors"):
            files.append(path)
        elif name in {"pytorch_model.bin", "model.bin", "model.pt", "model.pth"}:
            files.append(path)
        elif name.startswith("pytorch_model") and name.endswith(".bin"):
            files.append(path)
    return files


def load_tensors(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        if safe_open is None:
            raise RuntimeError("safetensors is required to read " + path)
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
        return tensors
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"unsupported checkpoint type in {path}: {type(obj)}")
    return {str(k): v for k, v in obj.items() if torch.is_tensor(v)}


def save_tensors(path: str, tensors: Dict[str, torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".safetensors"):
        if save_file is None:
            raise RuntimeError("safetensors is required to write " + path)
        save_file(tensors, path)
        return
    torch.save(tensors, path)


def copy_sidecar_files(src_dir: str, dst_dir: str, skip_names: Iterable[str]) -> None:
    skip = set(skip_names)
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name in skip:
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            continue
        # graph3 sidecars include multi-GB rec lookup tables. Link them
        # instead of copying so a plumbing checkpoint does not duplicate them.
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)


def quantize_state_dict(
    state: Dict[str, torch.Tensor],
    calibrated: Dict[str, float],
    placeholder: Optional[float],
) -> Tuple[Dict[str, torch.Tensor], int]:
    input_scales: Dict[str, float] = {}
    for name in state:
        if is_decoder_attn_weight(name):
            input_scales[name] = resolve_input_scale(name, calibrated, placeholder)
    apply_shared_qkv_scales(input_scales)

    out = dict(state)
    converted = 0
    for name, tensor in state.items():
        if not is_decoder_attn_weight(name):
            continue
        q_weight, w_scale = quantize_per_channel_int8(tensor)
        prefix = name[: -len(".weight")]
        a_scale = torch.tensor([input_scales[name]], dtype=torch.bfloat16)
        out[name] = q_weight
        out[prefix + ".input_scale"] = a_scale
        out[prefix + ".input_offset"] = torch.zeros(1, dtype=torch.int8)
        out[prefix + ".deq_scale"] = deq_scale(a_scale, w_scale)
        converted += 1
    return out, converted


def run_self_test() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 16, dtype=torch.bfloat16)
    q_weight, w_scale = quantize_per_channel_int8(weight)
    assert q_weight.dtype == torch.int8
    assert w_scale.dtype == torch.float32
    assert q_weight.shape == weight.shape
    assert w_scale.shape == (32,)

    input_scale = torch.tensor(0.5, dtype=torch.bfloat16)
    descale = deq_scale(input_scale, w_scale)
    assert descale.dtype == torch.float32
    assert descale.shape == (32,)
    expected = 0.5 * w_scale
    if not torch.allclose(descale, expected, rtol=1e-5, atol=1e-6):
        raise AssertionError("deq_scale != input_scale * s_w")

    recon = q_weight.to(torch.float32) * w_scale.unsqueeze(1)
    cosine = torch.nn.functional.cosine_similarity(
        weight.flatten().float(), recon.flatten(), dim=0
    ).item()
    if cosine < 0.99:
        raise AssertionError(f"weight recon cosine {cosine:.6f} < 0.99")

    fake = {
        "decoder.block.0.layer.0.SelfAttention.q.weight": weight,
        "decoder.block.0.layer.0.SelfAttention.k.weight": torch.randn(
            8, 16, dtype=torch.bfloat16
        ),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(
            16, 16, dtype=torch.bfloat16
        ),
        "decoder.block.0.layer.2.ffn.router.weight": torch.randn(4, 16),
    }
    out, n = quantize_state_dict(fake, {}, placeholder=1.0)
    if n != 2:
        raise AssertionError(f"expected 2 converted tensors, got {n}")
    if out["encoder.block.0.layer.0.SelfAttention.q.weight"].dtype != torch.bfloat16:
        raise AssertionError("encoder attention must stay BF16")
    q_deq = out["decoder.block.0.layer.0.SelfAttention.q.deq_scale"]
    k_deq = out["decoder.block.0.layer.0.SelfAttention.k.deq_scale"]
    if q_deq.dtype != torch.float32 or k_deq.dtype != torch.float32:
        raise AssertionError("deq_scale must stay FP32")
    q_scale = out["decoder.block.0.layer.0.SelfAttention.q.input_scale"]
    k_scale = out["decoder.block.0.layer.0.SelfAttention.k.input_scale"]
    if float(q_scale) != float(k_scale):
        raise AssertionError("packed Self Q/K must share input_scale")
    print("onerec_attn_w8a8_ptq self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", help="source model directory")
    parser.add_argument("--output-dir", help="destination model directory")
    parser.add_argument(
        "--calibrate-json",
        help="JSON map of tensor name / input_scale key -> RMSNorm absmax",
    )
    parser.add_argument(
        "--placeholder-input-scale",
        type=float,
        help="plumbing-only fallback input_scale; not for accuracy sign-off",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if not args.input_dir or not args.output_dir:
        print("--input-dir and --output-dir are required", file=sys.stderr)
        return 2

    calibrated: Dict[str, float] = {}
    if args.calibrate_json:
        with open(args.calibrate_json, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        calibrated = {str(k): float(v) for k, v in raw.items()}

    if args.placeholder_input_scale is not None:
        print(
            "WARNING: --placeholder-input-scale is for plumbing only; "
            "do not use the output checkpoint for accuracy sign-off.",
            file=sys.stderr,
        )

    files = iter_weight_files(args.input_dir)
    if not files:
        print(f"no weight files found in {args.input_dir}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    copy_sidecar_files(
        args.input_dir, args.output_dir, skip_names={os.path.basename(f) for f in files}
    )

    converted_total = 0
    for src in files:
        dst = os.path.join(args.output_dir, os.path.basename(src))
        state = load_tensors(src)
        out, converted = quantize_state_dict(
            state, calibrated, args.placeholder_input_scale
        )
        save_tensors(dst, out)
        converted_total += converted
        print(f"wrote {dst} ({converted} attention tensors quantized)")

    if converted_total == 0:
        print("no decoder attention weights were quantized", file=sys.stderr)
        return 1
    print(f"done, quantized {converted_total} decoder attention weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
