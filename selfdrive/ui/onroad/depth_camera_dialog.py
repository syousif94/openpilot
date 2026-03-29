"""
Depth Camera Dialog: Full-screen relative depth estimation view.
Uses MiDaS v2.1 Small via onnxruntime (~2 fps on device).
Tap to probe distance at any point.  Close button (X) to exit.

Occupancy grid: integrates multiple depth frames with IMU/livePose data
to build a persistent 2D bird's-eye-view occupancy map.  Browse to
http://<device-ip>:8099 to see the live grid.

On device (TICI): Uses the driver camera via VisionIPC + CPU onnxruntime.
On Mac: Uses the webcam via OpenCV + CPU onnxruntime.

Before use, export the model:
  python selfdrive/depthd/export_depth_model.py   (Mac, needs torch)
"""
import os
import sys
import time
import threading
import numpy as np
import pyray as rl

from openpilot.system.hardware import TICI

if TICI:
  # onnxruntime installed to /data/onnxrt on device
  if '/data/onnxrt' not in sys.path:
    sys.path.insert(0, '/data/onnxrt')
  from msgq.visionipc import VisionStreamType
  from openpilot.selfdrive.ui.mici.onroad.cameraview import CameraView
  from openpilot.system.ui.widgets.nav_widget import NavWidget
  from openpilot.selfdrive.ui.ui_state import device, ui_state

import onnxruntime as ort
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import gui_label
from openpilot.common.swaglog import cloudlog

if TICI:
  import cereal.messaging as messaging
  from openpilot.selfdrive.ui.onroad.occupancy_grid import OccupancyGrid
  from openpilot.selfdrive.ui.onroad.occupancy_web import OccupancyWebServer

# Depth map resolution — must match what the model was compiled with
DEPTH_W = 256
DEPTH_H = 256

# Model file paths
MODEL_NAME = "midas_small"
_DEPTH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "depth_model")
if TICI:
  DEPTH_ONNX_PATH = f"/data/depth_model/{MODEL_NAME}.sim.onnx"
else:
  # Prefer the onnxsim-inlined ONNX (all weights embedded); fall back to original
  _onnx_sim = os.path.join(_DEPTH_DIR, f"{MODEL_NAME}.sim.onnx")
  _onnx_orig = os.path.join(_DEPTH_DIR, f"{MODEL_NAME}.onnx")
  DEPTH_ONNX_PATH = _onnx_sim if os.path.exists(_onnx_sim) else _onnx_orig

# ImageNet normalization (DINOv2 backbone)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# Depth display: adaptive min/max per frame
# Smoothing factor for range (0 = no smoothing, 1 = frozen)
DEPTH_RANGE_SMOOTH = 0.8

# Default camera intrinsics for driver camera (AR/OX fisheye on tici/mici)
# Overridden at runtime if liveCalibration is available
_DEFAULT_DCAM_INTRINSICS = np.array([
  [567.0, 0.0, 964.0],
  [0.0, 567.0, 604.0],
  [0.0, 0.0, 1.0],
], dtype=np.float64)

DEFAULT_CAM_HEIGHT = 1.2   # metres above ground

# Close button (top-right corner)
CLOSE_BTN_SIZE = 60
CLOSE_BTN_MARGIN = 20

# Probe crosshair / label
PROBE_CROSSHAIR_SIZE = 20
PROBE_TEXT_SIZE = 36

# Turbo-like colormap: 8 key colors, far (purple/blue) → close (orange/red)
_COLORMAP_KEYS = np.array([
  [48, 18, 59],    # deep purple (farthest)
  [69, 91, 205],   # blue
  [32, 164, 134],  # teal
  [94, 201, 58],   # green
  [189, 204, 31],  # yellow-green
  [254, 188, 43],  # yellow
  [246, 108, 24],  # orange
  [176, 14, 14],   # red (closest)
], dtype=np.uint8)


def _build_colormap_lut() -> np.ndarray:
  """Build a 256-entry RGBA lookup table."""
  lut = np.zeros((256, 4), dtype=np.uint8)
  n = len(_COLORMAP_KEYS)
  for i in range(256):
    t = 1.0 - (i / 255.0)
    idx_f = t * (n - 1)
    lo, hi = int(idx_f), min(int(idx_f) + 1, n - 1)
    frac = idx_f - lo
    c = _COLORMAP_KEYS[lo] * (1.0 - frac) + _COLORMAP_KEYS[hi] * frac
    lut[i, :3] = c.astype(np.uint8)
    lut[i, 3] = 160  # semi-transparent
  return lut


COLORMAP_LUT = _build_colormap_lut()


def _nv12_to_rgb_small(data: np.ndarray, width: int, height: int,
                       stride: int, uv_offset: int, out_w: int, out_h: int) -> np.ndarray:
  """Convert NV12 to a small RGB array by subsampling.

  NV12 layout: Y plane (height rows x stride) then interleaved UV plane
  (height/2 rows x stride).  Each UV sample covers a 2x2 block of Y pixels.
  We upsample UV to full Y resolution via np.repeat before applying the
  same subsampling steps so the shapes always match.
  """
  step_y = max(1, height // out_h)
  step_x = max(1, width // out_w)

  # Y plane — full resolution
  y_plane = data[:uv_offset].reshape(-1, stride)[:height, :width]
  y_sub = y_plane[::step_y, ::step_x][:out_h, :out_w].astype(np.float32)

  # UV plane — half resolution, upsample to full so shapes match Y
  uv_size = stride * (height // 2)
  uv_data = data[uv_offset:uv_offset + uv_size].reshape(height // 2, stride)[:, :width]
  u_half = uv_data[:, 0::2].astype(np.float32) - 128.0   # (H/2, W/2)
  v_half = uv_data[:, 1::2].astype(np.float32) - 128.0

  # Nearest-neighbour upsample: repeat rows x2, cols x2 → (H, W)
  u_full = np.repeat(np.repeat(u_half, 2, axis=0), 2, axis=1)[:height, :width]
  v_full = np.repeat(np.repeat(v_half, 2, axis=0), 2, axis=1)[:height, :width]

  # Subsample with the same steps as Y
  u_sub = u_full[::step_y, ::step_x][:out_h, :out_w]
  v_sub = v_full[::step_y, ::step_x][:out_h, :out_w]

  r = np.clip(y_sub + 1.402 * v_sub, 0, 255)
  g = np.clip(y_sub - 0.344 * u_sub - 0.714 * v_sub, 0, 255)
  b = np.clip(y_sub + 1.772 * u_sub, 0, 255)
  return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
  """Preprocess RGB (H,W,3 uint8) → NCHW float32 with ImageNet normalisation."""
  img = rgb.astype(np.float32) / 255.0
  img = (img - IMAGENET_MEAN) / IMAGENET_STD
  return img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)   # (1, 3, H, W)


class _DepthMixin:
  """Shared depth inference + drawing logic, mixed into both TICI and Mac dialogs."""

  def _init_depth(self):
    # Model state — ort.InferenceSession on all platforms
    self._depth_run = None
    self._model_loading = True
    self._model_error: str | None = None

    # Depth outputs (guarded by _depth_lock)
    self._depth_rgba: np.ndarray | None = None
    self._depth_meters: np.ndarray | None = None
    self._depth_lock = threading.Lock()

    # Adaptive depth range for colormap
    self._depth_min = 0.0
    self._depth_max = 10.0

    # Depth overlay texture
    self._depth_texture: rl.Texture | None = None
    self._tex_w = 0
    self._tex_h = 0

    # Performance
    self._last_inference_time = 0.0
    self._fps_text = ""

    # Asynchronous inference
    self._inference_busy = False

    # Probe state
    self._probe_screen_pos: tuple[float, float] | None = None
    self._probe_depth: float | None = None
    self._current_rect: rl.Rectangle | None = None

    # ── Occupancy grid (TICI only) ────────────────────────────────
    self._occ_grid = None
    self._occ_web = None
    self._sm = None
    self._last_pose_t: float | None = None
    self._last_vel = np.zeros(3, dtype=np.float64)
    self._last_yaw_rate = 0.0
    self._last_gyro = [0.0, 0.0, 0.0]   # raw gyro xyz for web UI
    self._last_accel = [0.0, 0.0, 0.0]  # raw accel xyz for web UI
    self._cam_height = DEFAULT_CAM_HEIGHT
    self._cam_intrinsics = _DEFAULT_DCAM_INTRINSICS.copy()

    if TICI:
      self._occ_grid = OccupancyGrid()
      self._occ_web = OccupancyWebServer(self._occ_grid, get_imu=self._get_imu_data)
      self._occ_web.start()
      try:
        self._sm = messaging.SubMaster(['livePose', 'liveCalibration', 'accelerometer', 'gyroscope'])
      except Exception:
        cloudlog.warning("depth: could not subscribe to livePose/IMU — grid will accumulate without ego-motion compensation")

    # Load model in background
    self._load_thread = threading.Thread(target=self._load_model_bg, daemon=True)
    self._load_thread.start()

  def _load_model_bg(self):
    try:
      onnx_path = os.path.realpath(DEPTH_ONNX_PATH)
      if not os.path.exists(onnx_path):
        self._model_error = (
          f"ONNX not found: {onnx_path}\n"
          "Run: python selfdrive/depthd/export_depth_model.py"
        )
        cloudlog.error(f"depth: {self._model_error}")
        self._model_loading = False
        return
      cloudlog.info(f"depth: loading ONNX from {onnx_path}")
      opts = ort.SessionOptions()
      if TICI:
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4
      providers = ['CPUExecutionProvider']
      if not TICI:
        all_providers = ort.get_available_providers()
        cloudlog.info(f"depth: onnxruntime providers: {all_providers}")
        if 'CoreMLExecutionProvider' in all_providers:
          providers = all_providers
      try:
        self._depth_run = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
      except Exception:
        cloudlog.warning("depth: failed with providers, falling back to CPU")
        self._depth_run = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
      # warmup
      dummy = np.zeros((1, 3, DEPTH_H, DEPTH_W), dtype=np.float32)
      self._depth_run.run(None, {'input': dummy})
      cloudlog.info("depth: onnxruntime warmup complete")

    except Exception as e:
      self._model_error = f"Failed to load model: {e}"
      cloudlog.exception("depth: model load failed")
    finally:
      self._model_loading = False

  def _get_imu_data(self) -> dict:
    """Return latest IMU readings for the web UI."""
    return {
      'gyro': self._last_gyro,
      'accel': self._last_accel,
    }

  def _update_pose(self) -> tuple[float, float, float]:
    """Poll livePose/IMU and return (delta_x, delta_y, delta_yaw) since last call.

    On Mac (no cereal), returns (0, 0, 0) — grid accumulates without motion.
    """
    dx, dy, dyaw = 0.0, 0.0, 0.0
    sm = self._sm
    if sm is None:
      return dx, dy, dyaw

    try:
      sm.update(0)  # non-blocking poll

      now = time.monotonic()
      dt = 0.0
      if self._last_pose_t is not None:
        dt = min(now - self._last_pose_t, 0.5)  # clamp to avoid large jumps
      self._last_pose_t = now

      if dt < 1e-6:
        return dx, dy, dyaw

      # Prefer livePose (fused IMU + visual odometry at 20 Hz)
      if sm.recv_frame.get('livePose', 0) > 0 and sm.valid.get('livePose', False):
        pose = sm['livePose']
        if hasattr(pose, 'velocityDevice') and pose.velocityDevice.valid:
          vx = pose.velocityDevice.x   # forward
          vy = pose.velocityDevice.y   # right (we negate for left-positive)
          dx = vx * dt
          dy = -vy * dt

        if hasattr(pose, 'angularVelocityDevice') and pose.angularVelocityDevice.valid:
          yaw_rate = pose.angularVelocityDevice.z
          dyaw = yaw_rate * dt

      # Fallback: raw gyroscope for yaw (reliable), raw accel for tilt info
      elif sm.recv_frame.get('gyroscope', 0) > 0:
        gyro = sm['gyroscope']
        if hasattr(gyro, 'gyro') and len(gyro.gyro.v) > 2:
          # Z-axis gyro = yaw rate (rotation around vertical axis)
          yaw_rate = gyro.gyro.v[2]
          dyaw = yaw_rate * dt

          # Store raw gyro/accel for web UI visualization
          self._last_gyro = [float(gyro.gyro.v[i]) for i in range(3)]

        if sm.recv_frame.get('accelerometer', 0) > 0:
          accel = sm['accelerometer']
          if hasattr(accel, 'acceleration') and len(accel.acceleration.v) >= 3:
            self._last_accel = [float(accel.acceleration.v[i]) for i in range(3)]

      # Update camera height from calibration
      if sm.recv_frame.get('liveCalibration', 0) > 0:
        calib = sm['liveCalibration']
        if hasattr(calib, 'height') and len(calib.height) > 0:
          self._cam_height = float(calib.height[0])

    except Exception:
      cloudlog.exception("depth: pose update error")

    return dx, dy, dyaw

  def _run_depth_on_rgb(self, rgb: np.ndarray):
    """Run depth on (H,W,3) uint8 RGB. Stores inverse depth + RGBA overlay."""
    if self._depth_run is None:
      return

    t0 = time.monotonic()
    try:
      # Letterbox: pad shorter dim to make square 256x256 for model
      h_in, w_in = rgb.shape[:2]
      pad_top = (DEPTH_H - h_in) // 2
      pad_bot = DEPTH_H - h_in - pad_top
      pad_left = (DEPTH_W - w_in) // 2
      pad_right = DEPTH_W - w_in - pad_left
      if pad_top > 0 or pad_bot > 0 or pad_left > 0 or pad_right > 0:
        rgb_sq = np.pad(rgb, ((max(pad_top,0), max(pad_bot,0)), (max(pad_left,0), max(pad_right,0)), (0, 0)), mode='constant')
      else:
        rgb_sq = rgb
      inp = _preprocess_rgb(rgb_sq)

      out = self._depth_run.run(None, {'input': inp})[0].squeeze()

      # Crop out letterbox padding from depth output
      if pad_top > 0 or pad_bot > 0:
        out = out[pad_top:pad_top + h_in, :]
      if pad_left > 0 or pad_right > 0:
        out = out[:, pad_left:pad_left + w_in]

      inv_depth = out.astype(np.float32)  # MiDaS inverse depth (higher = closer)

      # Adaptive colormap: smooth the display range across frames
      lo, hi = float(inv_depth.min()), float(inv_depth.max())
      alpha = DEPTH_RANGE_SMOOTH
      self._depth_min = self._depth_min * alpha + lo * (1 - alpha)
      self._depth_max = self._depth_max * alpha + hi * (1 - alpha)
      rng = max(self._depth_max - self._depth_min, 0.1)
      # Normalise so 0 = close, 1 = far (invert for colormap: 0=purple=close, 255=red=far)
      depth_norm = 1.0 - np.clip((inv_depth - self._depth_min) / rng, 0.0, 1.0)
      depth_u8 = (depth_norm * 255).astype(np.uint8)
      rgba = np.ascontiguousarray(COLORMAP_LUT[depth_u8][:, ::-1])

      with self._depth_lock:
        self._depth_rgba = rgba
        self._depth_meters = inv_depth

      # ── Feed occupancy grid (TICI only) ────────────────────────
      if self._occ_grid is not None:
        try:
          dx, dy, dyaw = self._update_pose()
          self._occ_grid.integrate(
            inv_depth,
            self._cam_intrinsics,
            delta_x=dx, delta_y=dy, delta_yaw=dyaw,
            cam_height=self._cam_height,
            is_driver_cam=True,
          )
        except Exception:
          cloudlog.exception("depth: occupancy grid integration error")

    except Exception:
      cloudlog.exception("depth inference error")

    dt = time.monotonic() - t0
    self._last_inference_time = dt
    self._fps_text = f"{1.0 / max(dt, 1e-6):.0f} fps  ({dt * 1000:.0f} ms)"

  def _submit_inference(self, rgb: np.ndarray):
    """Submit a frame for background inference. Drops frame if busy."""
    if self._inference_busy or self._depth_run is None:
      return
    self._inference_busy = True

    def _worker():
      try:
        self._run_depth_on_rgb(rgb)
      finally:
        self._inference_busy = False

    threading.Thread(target=_worker, daemon=True).start()

  def _probe_at_screen(self, x: float, y: float):
    """Read depth value at a screen coordinate."""
    rect = self._current_rect
    if rect is None:
      return
    with self._depth_lock:
      dm = self._depth_meters
    if dm is None:
      return
    rel_x = (x - rect.x) / rect.width
    rel_y = (y - rect.y) / rect.height
    if not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
      return
    dh, dw = dm.shape[:2]
    px = min(int(rel_x * dw), dw - 1)
    py = min(int(rel_y * dh), dh - 1)
    self._probe_depth = float(dm[py, px])
    self._probe_screen_pos = (x, y)

  def _is_close_btn_hit(self, x: float, y: float) -> bool:
    """Return True if (x,y) falls inside the close button."""
    rect = self._current_rect
    if rect is None:
      return False
    bx = rect.x + rect.width - CLOSE_BTN_SIZE - CLOSE_BTN_MARGIN
    by = rect.y + CLOSE_BTN_MARGIN
    return bx <= x <= bx + CLOSE_BTN_SIZE and by <= y <= by + CLOSE_BTN_SIZE

  def _draw_depth_overlay(self, rect: rl.Rectangle):
    with self._depth_lock:
      rgba = self._depth_rgba
    if rgba is None:
      return

    h, w = rgba.shape[:2]
    if self._depth_texture is None or self._tex_w != w or self._tex_h != h:
      if self._depth_texture is not None:
        rl.unload_texture(self._depth_texture)
      img = rl.gen_image_color(w, h, rl.BLANK)
      self._depth_texture = rl.load_texture_from_image(img)
      rl.unload_image(img)
      self._tex_w, self._tex_h = w, h

    rl.update_texture(self._depth_texture, rl.ffi.cast("void *", rgba.ctypes.data))
    src = rl.Rectangle(0, 0, float(w), float(h))
    dst = rl.Rectangle(rect.x, rect.y, rect.width, rect.height)
    rl.draw_texture_pro(self._depth_texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

  def _draw_close_button(self, rect: rl.Rectangle):
    """Draw a circular X close button in the top-right corner."""
    bx = rect.x + rect.width - CLOSE_BTN_SIZE - CLOSE_BTN_MARGIN
    by = rect.y + CLOSE_BTN_MARGIN
    cx = int(bx + CLOSE_BTN_SIZE / 2)
    cy = int(by + CLOSE_BTN_SIZE / 2)
    rl.draw_circle(cx, cy, CLOSE_BTN_SIZE // 2, rl.Color(0, 0, 0, 150))
    m = 18
    rl.draw_line_ex(
      rl.Vector2(bx + m, by + m),
      rl.Vector2(bx + CLOSE_BTN_SIZE - m, by + CLOSE_BTN_SIZE - m),
      3, rl.WHITE,
    )
    rl.draw_line_ex(
      rl.Vector2(bx + CLOSE_BTN_SIZE - m, by + m),
      rl.Vector2(bx + m, by + CLOSE_BTN_SIZE - m),
      3, rl.WHITE,
    )

  def _draw_probe(self):
    """Draw the probe crosshair and depth label."""
    if self._probe_screen_pos is None or self._probe_depth is None:
      return
    x, y = self._probe_screen_pos
    d = self._probe_depth

    s = PROBE_CROSSHAIR_SIZE
    rl.draw_line_ex(rl.Vector2(x - s, y), rl.Vector2(x + s, y), 2, rl.WHITE)
    rl.draw_line_ex(rl.Vector2(x, y - s), rl.Vector2(x, y + s), 2, rl.WHITE)
    rl.draw_circle(int(x), int(y), 4, rl.WHITE)

    font = gui_app.font(FontWeight.BOLD)
    text = f"{d:.0f}"
    ts = rl.measure_text_ex(font, text, PROBE_TEXT_SIZE, 1)
    rect = self._current_rect
    lx = x + 15
    ly = y - ts.y - 5
    if rect:
      if lx + ts.x > rect.x + rect.width - 10:
        lx = x - ts.x - 15
      if ly < rect.y + 10:
        ly = y + 10
    pad = 6
    rl.draw_rectangle(int(lx - pad), int(ly - pad),
                      int(ts.x + 2 * pad), int(ts.y + 2 * pad),
                      rl.Color(0, 0, 0, 170))
    rl.draw_text_ex(font, text, rl.Vector2(lx, ly), PROBE_TEXT_SIZE, 1, rl.WHITE)

  def _draw_hud(self, rect: rl.Rectangle):
    backend = "ONNX"
    font = gui_app.font(FontWeight.BOLD)
    margin = 30

    rl.draw_text_ex(font, "DEPTH VIEW", rl.Vector2(rect.x + margin, rect.y + margin), 48, 1, rl.WHITE)

    if self._fps_text:
      info = f"{self._fps_text}  [{backend}]"
      rl.draw_text_ex(font, info, rl.Vector2(rect.x + margin, rect.y + margin + 56), 32, 1,
                      rl.Color(200, 200, 200, 220))

    hint = "Tap to measure  |  X to close"
    hint_size = rl.measure_text_ex(font, hint, 28, 1)
    rl.draw_text_ex(font, hint, rl.Vector2(
      rect.x + margin,
      rect.y + rect.height - hint_size.y - margin,
    ), 28, 1, rl.Color(200, 200, 200, 180))

    # Occupancy grid info (TICI only)
    if self._occ_grid is not None and self._occ_web is not None:
      occ_frames = self._occ_grid._frame_count
      occ_text = f"grid: {occ_frames} frames  |  http://*:{self._occ_web._port}"
      rl.draw_text_ex(font, occ_text, rl.Vector2(rect.x + margin, rect.y + margin + 90), 24, 1,
                      rl.Color(100, 255, 100, 200))

    # Colormap legend with adaptive range
    bar_w, bar_h = 20, 120
    lx = int(rect.x + rect.width - bar_w - margin)
    ly = int(rect.y + rect.height - bar_h - margin - 50)
    for i in range(bar_h):
      idx = int((i / bar_h) * 255)
      c = COLORMAP_LUT[idx]
      rl.draw_line(lx, ly + i, lx + bar_w, ly + i, rl.Color(int(c[0]), int(c[1]), int(c[2]), 220))

    font_sm = gui_app.font(FontWeight.MEDIUM)
    rl.draw_text_ex(font_sm, "close", rl.Vector2(lx - 24, ly - 22), 20, 1, rl.WHITE)
    rl.draw_text_ex(font_sm, "far", rl.Vector2(lx - 24, ly + bar_h + 4), 20, 1, rl.WHITE)

    self._draw_close_button(rect)
    self._draw_probe()

  def _render_status(self, rect, has_camera: bool) -> bool:
    """Render loading/error states. Returns True if should skip normal rendering."""
    if not has_camera:
      gui_label(rect, tr("camera starting"), font_size=100,
                font_weight=FontWeight.BOLD, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return True
    if self._model_loading:
      gui_label(rect, "Loading depth model...", font_size=64,
                font_weight=FontWeight.BOLD, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return True
    if self._model_error:
      gui_label(rect, self._model_error, font_size=36,
                font_weight=FontWeight.BOLD, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return True
    return False

  def _close_depth(self):
    if self._occ_web is not None:
      self._occ_web.stop()
      self._occ_web = None
    if self._depth_texture is not None:
      rl.unload_texture(self._depth_texture)
      self._depth_texture = None


# ---------------------------------------------------------------------------
# TICI (on-device): use VisionIPC + CameraView
# ---------------------------------------------------------------------------
if TICI:
  class _DepthCameraView(CameraView):
    """CameraView with zoomed driver view ratio for depth preview."""
    def _calc_frame_matrix(self, rect: rl.Rectangle):
      base = super()._calc_frame_matrix(rect)
      driver_view_ratio = 1.5
      base[0, 0] *= driver_view_ratio
      base[1, 1] *= driver_view_ratio
      return base

  class DepthCameraDialog(NavWidget, _DepthMixin):
    """Full-screen depth view using the driver-facing camera (device).
    Follows the mici DriverCameraDialog pattern: NavWidget + composition."""

    def __init__(self):
      super().__init__()
      self._camera_view = _DepthCameraView("camerad", VisionStreamType.VISION_STREAM_DRIVER)
      self._init_depth()
      device.add_interactive_timeout_callback(gui_app.pop_widget)
      ui_state.params.put_bool("IsDriverViewEnabled", True)

    def show_event(self):
      super().show_event()
      device.set_override_interactive_timeout(300)

    def hide_event(self):
      super().hide_event()
      ui_state.params.put_bool("IsDriverViewEnabled", False)
      device.set_override_interactive_timeout(None)

    def _handle_mouse_release(self, pos):
      self._probe_at_screen(pos.x, pos.y)

    def __del__(self):
      self.close()

    def _update_state(self):
      if self._camera_view:
        self._camera_view._update_state()
      super()._update_state()

    def _render(self, rect):
      self._current_rect = rect
      rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

      self._camera_view._render(rect)

      has_frame = self._camera_view.frame is not None
      if not has_frame:
        gui_label(rect, tr("camera starting"), font_size=54, font_weight=FontWeight.BOLD,
                  alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
        rl.end_scissor_mode()
        return

      if self._render_status(rect, has_frame):
        rl.end_scissor_mode()
        return

      frame = self._camera_view.frame
      # Preserve camera aspect ratio: subsample to DEPTH_W x proportional height
      cam_aspect = frame.width / max(frame.height, 1)
      sub_h = max(1, round(DEPTH_W / cam_aspect))
      rgb = _nv12_to_rgb_small(
        frame.data, frame.width, frame.height,
        frame.stride, frame.uv_offset, DEPTH_W, sub_h,
      )
      self._submit_inference(rgb)
      self._draw_depth_overlay(rect)
      self._draw_hud(rect)

      rl.end_scissor_mode()

    def close(self):
      self._close_depth()
      if self._camera_view:
        self._camera_view.close()
        self._camera_view = None


# ---------------------------------------------------------------------------
# Mac / non-TICI: use OpenCV webcam
# ---------------------------------------------------------------------------
else:
  import cv2

  class DepthCameraDialog(Widget, _DepthMixin):
    """Full-screen depth view using the Mac webcam (local dev)."""

    def __init__(self):
      Widget.__init__(self)
      self._init_depth()

      self._cap: cv2.VideoCapture | None = None
      self._cam_texture: rl.Texture | None = None
      self._cam_rgba: np.ndarray | None = None
      self._cam_w = 0
      self._cam_h = 0
      self._has_frame = False

      # Must open webcam on main thread for macOS AVFoundation auth
      try:
        self._cap = cv2.VideoCapture(0)
        if self._cap.isOpened():
          cloudlog.info("depth: webcam opened")
        else:
          cloudlog.error("depth: failed to open webcam")
      except Exception:
        cloudlog.exception("depth: webcam error")

    def _handle_mouse_release(self, pos):
      if self._is_close_btn_hit(pos.x, pos.y):
        gui_app.pop_widget()
      else:
        self._probe_at_screen(pos.x, pos.y)

    def __del__(self):
      self.close()

    def _render(self, rect):
      self._current_rect = rect
      rl.draw_rectangle_rec(rect, rl.BLACK)

      # Read webcam frame
      rgb_small = None
      if self._cap is not None and self._cap.isOpened():
        ret, frame_bgr = self._cap.read()
        if ret:
          self._has_frame = True
          # Flip horizontally (mirror, like driver cam)
          frame_bgr = cv2.flip(frame_bgr, 1)
          frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

          # Draw camera feed as background
          h, w = frame_rgb.shape[:2]
          rgba = np.zeros((h, w, 4), dtype=np.uint8)
          rgba[:, :, :3] = frame_rgb
          rgba[:, :, 3] = 255

          if self._cam_texture is None or self._cam_w != w or self._cam_h != h:
            if self._cam_texture is not None:
              rl.unload_texture(self._cam_texture)
            img = rl.gen_image_color(w, h, rl.BLANK)
            self._cam_texture = rl.load_texture_from_image(img)
            rl.unload_image(img)
            self._cam_w, self._cam_h = w, h

          rl.update_texture(self._cam_texture, rl.ffi.cast("void *", rgba.ctypes.data))
          src = rl.Rectangle(0, 0, float(w), float(h))
          dst = rl.Rectangle(rect.x, rect.y, rect.width, rect.height)
          rl.draw_texture_pro(self._cam_texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

          # Resize for depth model
          rgb_small = cv2.resize(frame_rgb, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)

      if self._render_status(rect, self._has_frame):
        return -1

      if rgb_small is not None:
        self._submit_inference(rgb_small)
      self._draw_depth_overlay(rect)
      self._draw_hud(rect)
      return -1

    def close(self):
      self._close_depth()
      if self._cam_texture is not None:
        rl.unload_texture(self._cam_texture)
        self._cam_texture = None
      if self._cap is not None:
        self._cap.release()
        self._cap = None


if __name__ == "__main__":
  gui_app.init_window("Depth Camera")
  dialog = DepthCameraDialog()

  # Override pop_widget to set a close flag
  _should_close = False
  _orig_pop = gui_app.pop_widget
  def _flag_close():
    global _should_close
    _should_close = True
  gui_app.pop_widget = _flag_close

  gui_app.push_widget(dialog)
  try:
    for _ in gui_app.render():
      if _should_close:
        break
      if TICI:
        ui_state.update()
  finally:
    gui_app.pop_widget = _orig_pop
    dialog.close()
