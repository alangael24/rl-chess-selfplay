#!/usr/bin/env python3
"""Export native NNUE integer weights for C-side search/inference.

Binary layout (little-endian):
  magic[8]      = b"NNUV1\\0\\0\\0"
  int32 dims[3] = (halfkp_features=41024, accum=256, hidden=32)
  int32 search_depth
  in_bias int16[256]
  in_w    int16[41024, 256]
  l1_w    int8[32, 512]
  l1_b    int32[32]
  l2_w    int8[32, 32]
  l2_b    int32[32]
  out_w   int8[32]
  out_b   int32[1]

The current training checkpoint does not contain this exact architecture yet, so this
exporter can synthesize a deterministic starter NNUE (optionally seeded from checkpoint).
"""

import argparse
import hashlib
import os
import struct

import numpy as np
import torch


MAGIC = b"NNUV1\0\0\0"
HALFKP_FEATURES = 41024
ACCUM = 256
HIDDEN = 32


def _load_state_dict(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "policy_state_dict" in ckpt:
        return ckpt["policy_state_dict"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _quantize_i8(x: np.ndarray):
    max_abs = float(np.max(np.abs(x)))
    if max_abs < 1e-12:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = 127.0 / max_abs
    q = np.clip(np.rint(x * scale), -127, 127).astype(np.int8)
    return q, scale


def _seed_from_checkpoint(path):
    h = hashlib.sha256(path.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .qpol path")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional PyTorch checkpoint to seed output-layer weights")
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-weight-span", type=int, default=2,
                        help="Random int16 half-range for input table, e.g. 2 => [-2, 2]")
    args = parser.parse_args()

    seed = int(args.seed)
    if args.checkpoint:
        seed ^= _seed_from_checkpoint(args.checkpoint)
    rng = np.random.default_rng(seed)

    span = max(1, int(args.input_weight_span))
    in_bias = np.zeros((ACCUM,), dtype=np.int16)
    in_w = rng.integers(-span, span + 1, size=(HALFKP_FEATURES, ACCUM), dtype=np.int16)

    l1_w = rng.integers(-4, 5, size=(HIDDEN, ACCUM * 2), dtype=np.int8)
    l1_b = np.zeros((HIDDEN,), dtype=np.int32)
    l2_w = rng.integers(-4, 5, size=(HIDDEN, HIDDEN), dtype=np.int8)
    l2_b = np.zeros((HIDDEN,), dtype=np.int32)
    out_w = rng.integers(-8, 9, size=(HIDDEN,), dtype=np.int8)
    out_b = np.int32(0)

    if args.checkpoint:
        sd = _load_state_dict(args.checkpoint, device="cpu")
        if "critic.weight" in sd:
            critic_w = sd["critic.weight"].detach().cpu().numpy().astype(np.float32)[0]
            critic_b = float(sd.get("critic.bias", torch.tensor([0.0])).detach().cpu().numpy()[0])
            q, scale = _quantize_i8(critic_w[:HIDDEN])
            out_w[:] = q
            out_b = np.int32(np.rint(critic_b * scale))

        if "backbone.2.weight" in sd:
            w2 = sd["backbone.2.weight"].detach().cpu().numpy().astype(np.float32)
            q2, _ = _quantize_i8(w2[:HIDDEN, :HIDDEN])
            l2_w[:] = q2
        if "backbone.2.bias" in sd:
            b2 = sd["backbone.2.bias"].detach().cpu().numpy().astype(np.float32)
            l2_b[:] = np.rint(b2[:HIDDEN]).astype(np.int32)

        if "backbone.0.weight" in sd:
            w1 = sd["backbone.0.weight"].detach().cpu().numpy().astype(np.float32)
            q1, _ = _quantize_i8(w1[:HIDDEN, : min(w1.shape[1], ACCUM * 2)])
            l1_w[:, :q1.shape[1]] = q1
        if "backbone.0.bias" in sd:
            b1 = sd["backbone.0.bias"].detach().cpu().numpy().astype(np.float32)
            l1_b[:] = np.rint(b1[:HIDDEN]).astype(np.int32)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iii", HALFKP_FEATURES, ACCUM, HIDDEN))
        f.write(struct.pack("<i", max(1, int(args.search_depth))))
        f.write(in_bias.tobytes(order="C"))
        f.write(in_w.tobytes(order="C"))
        f.write(l1_w.tobytes(order="C"))
        f.write(l1_b.astype(np.int32).tobytes(order="C"))
        f.write(l2_w.tobytes(order="C"))
        f.write(l2_b.astype(np.int32).tobytes(order="C"))
        f.write(out_w.tobytes(order="C"))
        f.write(struct.pack("<i", int(out_b)))

    print("Exported native NNUE qpolicy")
    print(f"  output:       {out_path}")
    print(f"  search_depth: {max(1, int(args.search_depth))}")
    print(f"  seed:         {seed}")
    if args.checkpoint:
        print(f"  checkpoint:   {args.checkpoint}")


if __name__ == "__main__":
    main()
