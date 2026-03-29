#!/usr/bin/env python3
"""
Export Depth Anything V2 Metric Indoor Small from HuggingFace to ONNX.

This runs on Mac (needs torch + transformers). The resulting ONNX is then
compiled to a tinygrad pickle by compile_depth_model.py.

Usage:
  python selfdrive/depthd/export_depth_model.py

Output:
  depth_model/depth_anything_v2_metric_indoor_small.onnx
"""
import os
import sys
from pathlib import Path

import torch
import numpy as np

HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Use 518x518 — the canonical DINOv2 input size (14*37)
INPUT_H, INPUT_W = 518, 518

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "depth_model"
ONNX_PATH = MODELS_DIR / "depth_anything_v2_metric_indoor_small.onnx"


def export():
  MODELS_DIR.mkdir(parents=True, exist_ok=True)

  if ONNX_PATH.exists():
    print(f"ONNX already exists at {ONNX_PATH}")
    return ONNX_PATH

  from transformers import AutoModelForDepthEstimation

  print(f"Loading {HF_MODEL_ID}...")
  model = AutoModelForDepthEstimation.from_pretrained(HF_MODEL_ID)
  model.eval()
  print("Model loaded")

  dummy_input = torch.randn(1, 3, INPUT_H, INPUT_W)

  print(f"Exporting to ONNX ({INPUT_H}x{INPUT_W})...")
  torch.onnx.export(
    model,
    (dummy_input,),
    str(ONNX_PATH),
    input_names=["pixel_values"],
    output_names=["predicted_depth"],
    opset_version=17,
    dynamic_axes=None,  # static shape for tinygrad compilation
  )

  sz = ONNX_PATH.stat().st_size
  print(f"Exported to {ONNX_PATH} ({sz / 1e6:.1f} MB)")

  # Quick validation
  print("Validating ONNX...")
  import onnxruntime as ort
  sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
  inp = {"pixel_values": np.random.randn(1, 3, INPUT_H, INPUT_W).astype(np.float32)}
  out = sess.run(None, inp)[0]
  print(f"  Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}] meters")

  # Compare with torch
  with torch.no_grad():
    torch_out = model(dummy_input).predicted_depth.numpy()
  onnx_out = sess.run(None, {"pixel_values": dummy_input.numpy()})[0]
  diff = np.abs(torch_out - onnx_out).max()
  print(f"  Max diff torch vs ONNX: {diff:.6f}")

  return ONNX_PATH


if __name__ == "__main__":
  export()
  print(f"\nDone! ONNX at {ONNX_PATH}")
  print(f"Next: python selfdrive/depthd/compile_depth_model.py")
