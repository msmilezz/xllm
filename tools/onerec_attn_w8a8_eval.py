#!/usr/bin/env python3
"""Evaluate OneRec Attention W8A8: linear cosine and online beam overlap.

Usage:
  python3 tools/onerec_attn_w8a8_eval.py cosine \\
      --bf16-dir graph3 --w8a8-dir graph3_attn_w8a8 \\
      --eval-pkl 260330_3B_5/eval_runner_inputs.pkl

  python3 tools/onerec_attn_w8a8_eval.py calibrate \\
      --bf16-dir graph3 --eval-pkl 260330_3B_5/eval_runner_inputs.pkl \\
      --output-json graph3_attn_w8a8_rmsnorm_absmax.json

  python3 tools/onerec_attn_w8a8_eval.py dump \\
      --out /tmp/bf16_beams.json --max-samples 4

  python3 tools/onerec_attn_w8a8_eval.py overlap \\
      --bf16-json /tmp/bf16_beams.json --w8a8-json /tmp/w8a8_beams.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.onerec_attn_w8a8_ptq import (  # noqa: E402
    DECODER_ATTN_SUFFIXES,
    is_decoder_attn_weight,
    load_tensors,
    iter_weight_files,
)

RMS_EPS = 1e-6
COSINE_GATE = 0.99


def load_dir(model_dir: str) -> Dict[str, torch.Tensor]:
    files = iter_weight_files(model_dir)
    if not files:
        raise FileNotFoundError(f"no weight files in {model_dir}")
    merged: Dict[str, torch.Tensor] = {}
    for path in files:
        merged.update(load_tensors(path))
    return merged


def attn_weight_names(state: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(name for name in state if is_decoder_attn_weight(name))


def prefix_of(weight_name: str) -> str:
    return weight_name[: -len(".weight")]


def layer_norm_name_for(weight_name: str) -> Optional[str]:
    # decoder.block_xx.layer.0.SelfAttention.q.weight
    #   -> decoder.block_xx.layer.0.layer_norm.weight
    # decoder.block_xx.layer.1.EncDecAttention.k.weight (skipNorm for K/V)
    #   -> None for K/V; Q/O use layer.1.layer_norm.weight
    parts = weight_name.split(".")
    try:
        layer_idx = parts.index("layer")
    except ValueError:
        return None
    sub = parts[layer_idx + 1]
    proj = parts[-2]
    if sub == "1" and proj in {"k", "v"}:
        return None
    parts[layer_idx + 2 :] = ["layer_norm", "weight"]
    return ".".join(parts)


def rms_norm(x: torch.Tensor, gamma: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x32 * rms
    if gamma is not None:
        y = y * gamma.reshape(-1).to(torch.float32)
    return y


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 1.0 if float(a.norm()) == 0.0 and float(b.norm()) == 0.0 else 0.0
    return float(torch.dot(a, b) / denom)


def load_eval_hidden(eval_pkl: str, max_tokens: int = 4096) -> torch.Tensor:
    import pickle

    with open(eval_pkl, "rb") as handle:
        data = pickle.load(handle)
    section = data.get("after_truncation", data)
    chunks: List[np.ndarray] = []
    for key in ("instruct_embedding", "inputs_embeds"):
        if key not in section:
            continue
        arr = np.asarray(section[key], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            pass
        else:
            continue
        chunks.append(arr)
    if not chunks:
        raise KeyError(f"no hidden-like tensors in {eval_pkl}")
    hidden = np.concatenate(chunks, axis=0)
    if hidden.shape[0] > max_tokens:
        hidden = hidden[:max_tokens]
    return torch.from_numpy(np.ascontiguousarray(hidden))


def w8a8_linear(
    x: torch.Tensor,
    weight_i8: torch.Tensor,
    input_scale: torch.Tensor,
    deq: torch.Tensor,
    input_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scale = float(input_scale.reshape(-1)[0].to(torch.float32))
    scale = max(scale, 1e-8)
    offset = 0.0
    if input_offset is not None and input_offset.numel() > 0:
        offset = float(input_offset.reshape(-1)[0].to(torch.float32))
    x_q = torch.clamp(torch.round(x / scale + offset), -128, 127).to(torch.int8)
    acc = x_q.to(torch.int32) @ weight_i8.to(torch.int32).transpose(0, 1)
    return acc.to(torch.float32) * deq.reshape(1, -1).to(torch.float32)


def cmd_cosine(args: argparse.Namespace) -> int:
    bf16 = load_dir(args.bf16_dir)
    w8a8 = load_dir(args.w8a8_dir)
    names = attn_weight_names(w8a8)
    if not names:
        print("no decoder attention int8 weights found", file=sys.stderr)
        return 1
    hidden = load_eval_hidden(args.eval_pkl, args.max_tokens)
    print(
        f"eval hidden tokens={hidden.shape[0]} dim={hidden.shape[1]} "
        f"layers={len(names)}"
    )

    rows: List[dict] = []
    fail = 0
    for name in names:
        w_bf16 = bf16[name].to(torch.float32)
        w_i8 = w8a8[name]
        prefix = prefix_of(name)
        deq = w8a8[prefix + ".deq_scale"]
        a_scale = w8a8[prefix + ".input_scale"]
        a_off = w8a8.get(prefix + ".input_offset")
        w_scale = deq.to(torch.float32) / a_scale.reshape(-1)[0].to(torch.float32).clamp(
            min=1e-8
        )
        recon = w_i8.to(torch.float32) * w_scale.reshape(-1, 1)
        w_cos = cosine(w_bf16, recon)

        gamma_name = layer_norm_name_for(name)
        gamma = bf16.get(gamma_name) if gamma_name else None
        x = hidden if gamma is None else rms_norm(hidden, gamma, RMS_EPS)
        y_ref = x @ w_bf16.transpose(0, 1)
        y_q = w8a8_linear(x, w_i8, a_scale, deq, a_off)
        y_cos = cosine(y_ref, y_q)

        # What cosine would be with a per-tensor scale fitted on this x.
        fitted = float(x.abs().amax() / 127.0)
        fitted = max(fitted, 1e-8)
        y_fit = w8a8_linear(
            x,
            w_i8,
            torch.tensor([fitted], dtype=torch.float32),
            (torch.tensor(fitted) * w_scale).to(torch.float32),
            None,
        )
        y_fit_cos = cosine(y_ref, y_fit)

        ok_w = w_cos >= COSINE_GATE
        ok_y = y_cos >= COSINE_GATE
        if not ok_w or not ok_y:
            fail += 1
        rows.append(
            {
                "name": name,
                "weight_cosine": w_cos,
                "gemm_cosine_ckpt_scale": y_cos,
                "gemm_cosine_fitted_scale": y_fit_cos,
                "ckpt_input_scale": float(a_scale.reshape(-1)[0].to(torch.float32)),
                "fitted_input_scale": fitted,
                "norm": gamma_name or "skipNorm",
            }
        )
        print(
            f"{'PASS' if ok_w and ok_y else 'FAIL'} {name} "
            f"W={w_cos:.6f} GEMM={y_cos:.6f} fittedGEMM={y_fit_cos:.6f} "
            f"scale={float(a_scale.reshape(-1)[0]):.6g}->{fitted:.6g} "
            f"norm={gamma_name or 'skipNorm'}"
        )

    print(
        f"summary: {len(rows) - fail}/{len(rows)} pass "
        f"(gate {COSINE_GATE}, ckpt-scale GEMM)"
    )
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.output_json}")
    return 0 if fail == 0 else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    bf16 = load_dir(args.bf16_dir)
    names = attn_weight_names(
        {k: v for k, v in bf16.items() if k.endswith(".weight")}
    )
    hidden = load_eval_hidden(args.eval_pkl, args.max_tokens)
    scales: Dict[str, float] = {}
    for name in names:
        if not name.startswith("decoder."):
            continue
        gamma_name = layer_norm_name_for(name)
        gamma = bf16.get(gamma_name) if gamma_name else None
        x = hidden if gamma is None else rms_norm(hidden, gamma, RMS_EPS)
        scales[name] = float(x.abs().amax() / 127.0)
        scales[prefix_of(name) + ".input_scale"] = scales[name]
        print(f"{name} input_scale={scales[name]:.8g} norm={gamma_name or 'skipNorm'}")
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(scales, handle, indent=2)
    print(f"wrote {args.output_json} ({len(names)} tensors)")
    return 0


def _client_module():
    client_dir = os.path.join(
        os.path.dirname(__file__), "..", "260330_3B_5"
    )
    sys.path.insert(0, os.path.abspath(client_dir))
    import read_and_call_2 as client  # type: ignore

    return client


def cmd_dump(args: argparse.Namespace) -> int:
    client = _client_module()
    ns = argparse.Namespace(
        api_url=args.api_url,
        model=client.DEFAULT_MODEL,
        max_samples=args.max_samples,
        sample_indices=args.sample_indices,
        beam_width=args.beam_width,
        max_tokens=None,
        prompt=client.DEFAULT_PROMPT,
        temperature=0.0,
        request_timeout=300.0,
        keep_leading_zero=False,
        skip_instruct_embedding=False,
        bearer_token=os.environ.get("OPENAI_API_KEY", "dummy_key"),
        input_section=client.DEFAULT_INPUT_SECTION,
        output_tensor_name=client.DEFAULT_OUTPUT_TENSOR_NAME,
    )

    import pickle

    input_pkl = args.input_pkl
    gen_pkl = args.gen_ids_pkl
    with open(input_pkl, "rb") as handle:
        inputs_data = pickle.load(handle)
    with open(gen_pkl, "rb") as handle:
        gen_ids_data = pickle.load(handle)

    section = client.parse_input_sections(ns.input_section)[0]
    samples = client.collect_samples(
        inputs_data,
        gen_ids_data,
        trim_leading_zero=not ns.keep_leading_zero,
        input_section=section,
    )
    indices = client.parse_sample_indices(ns.sample_indices, len(samples))
    selected = client.select_samples(samples, indices, ns.max_samples)

    dumped = []
    for sample in selected:
        payload = client.build_payload(sample, ns, ns.beam_width)
        beams, summaries, latency, _ = client.call_service(
            payload=payload,
            api_url=ns.api_url,
            bearer_token=ns.bearer_token,
            output_tensor_name=ns.output_tensor_name,
            request_timeout=ns.request_timeout,
        )
        dumped.append(
            {
                "index": sample.index,
                "latency": latency,
                "beams": beams,
                "summaries": summaries,
            }
        )
        print(
            f"sample {sample.index}: beams={len(beams)} "
            f"width={len(beams[0]) if beams else 0} latency={latency:.3f}s"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(dumped, handle)
    print(f"wrote {args.out}")
    return 0


def overlap_sets(
    a: Sequence[Sequence[int]], b: Sequence[Sequence[int]], k: int
) -> Tuple[int, int, float]:
    top_a = [tuple(row) for row in a[:k]]
    top_b = [tuple(row) for row in b[:k]]
    inter = len(set(top_a) & set(top_b))
    return inter, k, inter / float(k) if k else 0.0


def cmd_overlap(args: argparse.Namespace) -> int:
    with open(args.bf16_json, "r", encoding="utf-8") as handle:
        bf16 = json.load(handle)
    with open(args.w8a8_json, "r", encoding="utf-8") as handle:
        w8a8 = json.load(handle)
    by_w = {item["index"]: item for item in w8a8}
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    print(f"samples bf16={len(bf16)} w8a8={len(w8a8)} ks={ks}")
    for item in bf16:
        other = by_w.get(item["index"])
        if other is None:
            print(f"sample {item['index']}: missing W8A8 dump")
            continue
        a, b = item["beams"], other["beams"]
        exact, unique = (
            sum((Counter(map(tuple, a)) & Counter(map(tuple, b))).values()),
            len(set(map(tuple, a)) & set(map(tuple, b))),
        )
        print(
            f"sample {item['index']}: exact={exact}/{len(a)} "
            f"unique={unique} bf16={len(a)} w8a8={len(b)}"
        )
        for k in ks:
            inter, _, ratio = overlap_sets(a, b, min(k, len(a), len(b)))
            print(f"  overlap@{k}={inter}/{min(k, len(a), len(b))} ({ratio * 100:.2f}%)")
            print(f"  bf16_first3={a[:3]}")
            print(f"  w8a8_first3={b[:3]}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cosine_p = sub.add_parser("cosine")
    cosine_p.add_argument("--bf16-dir", required=True)
    cosine_p.add_argument("--w8a8-dir", required=True)
    cosine_p.add_argument("--eval-pkl", required=True)
    cosine_p.add_argument("--max-tokens", type=int, default=4096)
    cosine_p.add_argument("--output-json")

    cal_p = sub.add_parser("calibrate")
    cal_p.add_argument("--bf16-dir", required=True)
    cal_p.add_argument("--eval-pkl", required=True)
    cal_p.add_argument("--output-json", required=True)
    cal_p.add_argument("--max-tokens", type=int, default=4096)

    dump_p = sub.add_parser("dump")
    dump_p.add_argument("--out", required=True)
    dump_p.add_argument("--api-url", default="http://localhost:8096/v1/completions")
    dump_p.add_argument("--input-pkl", default="260330_3B_5/eval_runner_inputs.pkl")
    dump_p.add_argument("--gen-ids-pkl", default="260330_3B_5/eval_runner_gen_ids.pkl")
    dump_p.add_argument("--max-samples", type=int, default=4)
    dump_p.add_argument("--sample-indices", default="0,1,2,3")
    dump_p.add_argument("--beam-width", type=int, default=512)

    ov_p = sub.add_parser("overlap")
    ov_p.add_argument("--bf16-json", required=True)
    ov_p.add_argument("--w8a8-json", required=True)
    ov_p.add_argument("--ks", default="10,50,256,512")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "cosine":
        return cmd_cosine(args)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    if args.cmd == "dump":
        return cmd_dump(args)
    if args.cmd == "overlap":
        return cmd_overlap(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
