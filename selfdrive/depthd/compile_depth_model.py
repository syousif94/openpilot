#!/usr/bin/env python3
"""
Compile Depth Anything V2 Metric Indoor Small for tinygrad.

Takes the ONNX model (exported by export_depth_model.py) and compiles it
into a tinygrad TinyJit pickle.

Usage:
  # First export ONNX on Mac (needs torch + transformers):
  python selfdrive/depthd/export_depth_model.py

  # Then compile (runs on Mac or device):
  python selfdrive/depthd/compile_depth_model.py

Outputs:
  depth_model/depth_anything_v2_metric_indoor_small_tinygrad.pkl
"""
import os, sys, pickle
import numpy as np

# Auto-detect device
from openpilot.system.hardware import TICI
os.environ.setdefault('DEV', 'QCOM' if TICI else 'CPU')
if TICI:
  os.environ.setdefault('FLOAT16', '1')
os.environ.setdefault("JIT_BATCH_SIZE", "0")

from pathlib import Path
from tinygrad import Tensor, TinyJit, Context, GlobalCounters, Device, dtypes
from tinygrad.helpers import DEBUG
from tinygrad.nn.onnx import OnnxRunner

if TICI:
  MODELS_DIR = Path("/data/depth_model")
else:
  MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "depth_model"

ONNX_PATH = MODELS_DIR / "depth_anything_v2_metric_indoor_small.onnx"
PKL_PATH = MODELS_DIR / "depth_anything_v2_metric_indoor_small_tinygrad.pkl"

# Depth Anything V2 Small canonical input size (14*37 = 518)
INPUT_H, INPUT_W = 518, 518


def simplify_model(onnx_path: Path) -> Path:
  """Run onnxsim to fold constants for tinygrad compatibility."""
  simplified_path = onnx_path.with_suffix(".sim.onnx")
  if simplified_path.exists():
    print(f"Simplified ONNX already exists at {simplified_path}")
    return simplified_path

  print("Simplifying ONNX model with onnxsim...")
  import onnx
  from onnx import numpy_helper
  from onnxsim import simplify
  model = onnx.load(str(onnx_path))
  input_name = model.graph.input[0].name
  model_sim, check = simplify(model, overwrite_input_shapes={input_name: [1, 3, INPUT_H, INPUT_W]})
  if not check:
    print("WARNING: onnxsim could not validate simplified model, using it anyway")

  # Fix Resize nodes: tinygrad expects (X, roi, scales, sizes) but onnxsim may
  # produce (X, scales) with only 2 inputs. Insert empty roi tensor.
  empty_roi_name = "__empty_roi__"
  added_roi = False
  for node in model_sim.graph.node:
    if node.op_type == "Resize" and len(node.input) == 2:
      if not added_roi:
        model_sim.graph.initializer.append(
          numpy_helper.from_array(np.zeros(0, dtype=np.float32), name=empty_roi_name)
        )
        added_roi = True
      data_in, scales_in = node.input[0], node.input[1]
      del node.input[:]
      node.input.extend([data_in, empty_roi_name, scales_in])

  onnx.save(model_sim, str(simplified_path))
  print(f"Saved simplified model to {simplified_path}")
  return simplified_path


def compile_model(onnx_path: Path):
  # Simplify first to resolve dynamic Resize ops (skipped if already .sim.onnx)
  if not str(onnx_path).endswith(".sim.onnx"):
    onnx_path = simplify_model(onnx_path)

  print(f"Loading ONNX model from {onnx_path}...")
  run_onnx = OnnxRunner(str(onnx_path))
  print("Loaded model")

  # Get input spec
  input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
  input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}
  input_types = {k: (dtypes.float32 if v is dtypes.float16 else v) for k, v in input_types.items()}

  print(f"Model inputs: {input_shapes}")
  print(f"Model input types: {input_types}")

  # Create dummy inputs
  Tensor.manual_seed(100)
  inputs = {k: Tensor(Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize().numpy(), device='NPY')
            for k, shp in sorted(input_shapes.items())}
  print("Created test tensors")

  # Wrap in TinyJit for compiled execution
  run_onnx_jit = TinyJit(
    lambda **kwargs: next(iter(run_onnx({k: v.to(Device.DEFAULT) for k, v in kwargs.items()}).values())).cast('float32'),
    prune=True
  )

  # Warm up JIT (3 runs to capture and validate)
  test_val = None
  for i in range(3):
    GlobalCounters.reset()
    print(f"  warmup run {i}...")
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      ret = run_onnx_jit(**inputs).numpy()
    if i == 1:
      test_val = np.copy(ret)

  print(f"Captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  np.testing.assert_equal(test_val, ret, "JIT validation failed!")
  print("JIT validated successfully")

  # Save compiled model
  with open(PKL_PATH, "wb") as f:
    pickle.dump(run_onnx_jit, f)

  pkl_sz = PKL_PATH.stat().st_size
  print(f"Compiled pickle: {PKL_PATH} ({pkl_sz / 1e6:.1f} MB)")
  print(f"Device: {Device.DEFAULT}")
  print("**** compile done ****")
  return inputs, test_val


def test_compiled(inputs, test_val):
  """Verify the pickled model produces the same output."""
  print("Testing compiled pickle...")
  with open(PKL_PATH, "rb") as f:
    loaded = pickle.load(f)

  ret = loaded(**inputs).numpy()
  np.testing.assert_equal(test_val, ret, "Pickle output mismatch!")
  print("Pickle test passed!")


if __name__ == "__main__":
  sim_path = ONNX_PATH.with_suffix(".sim.onnx")
  if sim_path.exists():
    # Pre-simplified ONNX available (e.g. uploaded to device) — use it directly
    inputs, test_val = compile_model(sim_path)
  elif ONNX_PATH.exists():
    inputs, test_val = compile_model(ONNX_PATH)
  else:
    print(f"ONNX not found at {ONNX_PATH} or {sim_path}")
    print(f"Run first: python selfdrive/depthd/export_depth_model.py")
    sys.exit(1)
  test_compiled(inputs, test_val)
  print(f"\nDone! Model files in {MODELS_DIR}/")
  print(f"  ONNX:   {ONNX_PATH}")
  print(f"  Pickle: {PKL_PATH}")
