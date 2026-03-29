#!/usr/bin/env python3
"""Export MiDaS v2.1 Small to ONNX at 256x256.

Much faster than Depth Anything V2 (518x518):
  - ~12x fewer parameters
  - ~4x smaller input
  - Should run <1s on TICI CPU vs ~5s for Depth Anything V2
"""
import os
import torch
import numpy as np

OUT_H, OUT_W = 256, 256
MODEL_TYPE = "MiDaS_small"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "depth_model")
os.makedirs(OUT_DIR, exist_ok=True)

ONNX_PATH = os.path.join(OUT_DIR, "midas_small_256.onnx")
SIM_PATH = os.path.join(OUT_DIR, "midas_small_256.sim.onnx")


def main():
  print(f"Loading {MODEL_TYPE} from torch hub...")
  model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE, trust_repo=True)
  model.eval()

  # Count params
  n_params = sum(p.numel() for p in model.parameters())
  print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

  # Dummy input
  dummy = torch.randn(1, 3, OUT_H, OUT_W)

  # Quick test
  with torch.no_grad():
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Should be (1, 256, 256)

  # Export to ONNX
  print(f"Exporting to {ONNX_PATH}...")
  torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["depth"],
    opset_version=17,
    dynamic_axes=None,  # fixed shape for max perf
  )
  print(f"ONNX saved: {os.path.getsize(ONNX_PATH) / 1e6:.1f} MB")

  # Simplify
  try:
    import onnxsim
    import onnx
    print("Simplifying with onnxsim...")
    model_onnx = onnx.load(ONNX_PATH)
    model_sim, check = onnxsim.simplify(model_onnx)
    assert check, "Simplification failed"
    onnx.save(model_sim, SIM_PATH)
    print(f"Simplified ONNX: {os.path.getsize(SIM_PATH) / 1e6:.1f} MB")
  except ImportError:
    print("onnxsim not installed, skipping simplification")
    import shutil
    shutil.copy(ONNX_PATH, SIM_PATH)

  # Verify with onnxruntime
  import onnxruntime as ort
  sess = ort.InferenceSession(SIM_PATH)
  inp = np.random.randn(1, 3, OUT_H, OUT_W).astype(np.float32)
  out = sess.run(None, {"input": inp})[0]
  print(f"ORT output shape: {out.shape}, range: [{out.min():.2f}, {out.max():.2f}]")
  print(f"\nDone! Deploy {SIM_PATH} to /data/depth_model/ on device")


if __name__ == "__main__":
  main()
