"""
Occupancy Grid: accumulates depth observations from a monocular camera
into a 2D bird's-eye-view (BEV) occupancy grid, integrating pose changes
from IMU/livePose to properly fuse multiple frames.

Coordinate system (BEV grid):
  - X → forward (positive ahead of the device)
  - Y → left (positive to the left)
  - Cell value: log-odds occupancy  (>0 = occupied, <0 = free, 0 = unknown)

The grid is ego-centric and shifts as the device moves so the device
is always at a fixed position in the grid.
"""

import math
import threading
import time

import numpy as np

# ── Grid parameters ──────────────────────────────────────────────────
GRID_SIZE = 200          # cells per side (200 × 200)
CELL_SIZE = 0.10         # metres per cell  → 20 m × 20 m grid
MAX_RANGE = 8.0          # max depth to project (metres)
MIN_RANGE = 0.3          # minimum depth to consider

# Device sits at this cell in the grid (bottom-centre)
ORIGIN_X = 20            # 20 cells = 2 m behind, giving 18 m forward view
ORIGIN_Y = GRID_SIZE // 2  # centred left-right

# Log-odds update values
L_OCC = 0.85             # observation: occupied
L_FREE = -0.40           # observation: free (ray passed through)
L_MAX = 5.0              # clamp
L_MIN = -3.0

# Decay: slowly forget old observations so stale obstacles fade
DECAY_RATE = 0.97        # per integration step


class OccupancyGrid:
  """Thread-safe 2D occupancy grid with pose-compensated accumulation."""

  def __init__(self):
    self._lock = threading.Lock()
    self._grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Cumulative ego pose in world frame (x=forward, y=left, yaw)
    self._world_x = 0.0
    self._world_y = 0.0
    self._world_yaw = 0.0

    # Pose trail: list of (world_x, world_y, world_yaw, timestamp)
    self._trail: list[tuple[float, float, float, float]] = []
    self._trail_max = 500  # max trail points

    self._last_update = time.monotonic()
    self._frame_count = 0

  # ── Public API ───────────────────────────────────────────────────

  def integrate(self, depth_map: np.ndarray, intrinsics: np.ndarray,
                delta_x: float, delta_y: float, delta_yaw: float,
                cam_height: float = 1.2, is_driver_cam: bool = True):
    """
    Integrate one depth frame into the grid.

    Args:
      depth_map:  (H, W) float32 — metric depth in metres (or MiDaS
                  inverse-depth; see _inverse_depth_to_metric).
      intrinsics: 3×3 camera intrinsic matrix K.
      delta_x, delta_y, delta_yaw:  ego-motion since last call
                  (metres forward, metres left, radians CCW).
      cam_height: camera height above ground (metres).
      is_driver_cam: if True, the camera faces backward; we flip the
                     point cloud X axis so it maps into the BEV correctly.
    """
    with self._lock:
      # 1. Shift grid to compensate ego-motion
      if abs(delta_x) > 1e-4 or abs(delta_y) > 1e-4 or abs(delta_yaw) > 1e-4:
        self._shift_grid(delta_x, delta_y, delta_yaw)

      # 2. Decay old evidence
      self._grid *= DECAY_RATE

      # 3. Back-project depth → 3D points in camera frame
      pts_cam = self._backproject(depth_map, intrinsics)
      if pts_cam.shape[0] == 0:
        return

      # 4. Camera frame → device/road frame
      #    Camera: x=right, y=down, z=forward
      #    Device/road BEV: x=forward, y=left
      pts_x = pts_cam[:, 2]   # forward = cam Z
      pts_y = -pts_cam[:, 0]  # left = -cam X
      pts_z = -pts_cam[:, 1]  # up   = -cam Y

      if is_driver_cam:
        # Driver cam faces backward; flip forward axis
        pts_x = -pts_x

      # Filter: only keep points near ground plane (reasonable height)
      ground_mask = (pts_z < cam_height + 0.5) & (pts_z > -0.5)
      pts_x = pts_x[ground_mask]
      pts_y = pts_y[ground_mask]

      # 5. Convert to grid cells and update
      gx = (pts_x / CELL_SIZE + ORIGIN_X).astype(np.int32)
      gy = (pts_y / CELL_SIZE + ORIGIN_Y).astype(np.int32)

      valid = (gx >= 0) & (gx < GRID_SIZE) & (gy >= 0) & (gy < GRID_SIZE)
      gx = gx[valid]
      gy = gy[valid]

      # Mark occupied cells
      np.add.at(self._grid, (gx, gy), L_OCC)

      # Bresenham-lite: mark free space along rays from origin to each occupied cell
      self._mark_free_rays(gx, gy)

      # Clamp
      np.clip(self._grid, L_MIN, L_MAX, out=self._grid)

      self._frame_count += 1
      self._last_update = time.monotonic()

      # Record trail point
      self._trail.append((self._world_x, self._world_y, self._world_yaw, self._last_update))
      if len(self._trail) > self._trail_max:
        self._trail = self._trail[-self._trail_max:]

  def get_grid(self) -> np.ndarray:
    """Return a copy of the current occupancy grid (float32, GRID_SIZE×GRID_SIZE)."""
    with self._lock:
      return self._grid.copy()

  def get_probability_grid(self) -> np.ndarray:
    """Return occupancy probabilities [0, 1] from log-odds."""
    g = self.get_grid()
    return 1.0 / (1.0 + np.exp(-g))

  def get_binary_grid(self, threshold: float = 0.65) -> np.ndarray:
    """Return a binary uint8 grid (255=occupied, 128=unknown, 0=free)."""
    prob = self.get_probability_grid()
    out = np.full_like(prob, 128, dtype=np.uint8)
    out[prob > threshold] = 255
    out[prob < (1.0 - threshold)] = 0
    return out

  def get_metadata(self) -> dict:
    """Return grid metadata for the web UI."""
    return {
      "grid_size": GRID_SIZE,
      "cell_size": CELL_SIZE,
      "origin_x": ORIGIN_X,
      "origin_y": ORIGIN_Y,
      "world_x": self._world_x,
      "world_y": self._world_y,
      "world_yaw": self._world_yaw,
      "frame_count": self._frame_count,
      "max_range": MAX_RANGE,
    }

  def get_trail(self) -> list[tuple[float, float, float, float]]:
    """Return the pose trail as [(world_x, world_y, world_yaw, time), ...]."""
    with self._lock:
      return list(self._trail)

  def reset(self):
    with self._lock:
      self._grid[:] = 0
      self._world_x = 0.0
      self._world_y = 0.0
      self._world_yaw = 0.0
      self._trail.clear()
      self._frame_count = 0

  # ── Internals ────────────────────────────────────────────────────

  def _backproject(self, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project depth map to 3D points in camera frame.

    Returns (N, 3) float32 array of [x_cam, y_cam, z_cam].
    """
    h, w = depth.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Scale intrinsics if depth map resolution differs from the native resolution
    scale_x = w / (cx * 2)  # approximate
    scale_y = h / (cy * 2)
    fx_s = fx * scale_x
    fy_s = fy * scale_y
    cx_s = cx * scale_x
    cy_s = cy * scale_y

    # Subsample for speed: take every Nth pixel
    step = max(1, min(w, h) // 64)
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    xv, yv = np.meshgrid(xs, ys)

    d = depth[yv, xv].astype(np.float32)

    # Check if this is MiDaS inverse depth (high values) vs metric depth
    median_d = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    if median_d > 50:
      # Likely MiDaS inverse depth — convert with a rough scale
      # MiDaS relative inverse depth; we normalize to approximate metric depth
      safe_d = np.where(d > 1e-3, d, 1e-3)
      d = median_d * 0.5 / safe_d  # rough heuristic: median → ~0.5 m
      d = np.clip(d, MIN_RANGE, MAX_RANGE)

    # Filter valid depth
    mask = (d > MIN_RANGE) & (d < MAX_RANGE) & np.isfinite(d)
    d = d[mask]
    xp = xv[mask].astype(np.float32)
    yp = yv[mask].astype(np.float32)

    # Unproject
    z = d
    x = (xp - cx_s) / fx_s * z
    y = (yp - cy_s) / fy_s * z

    return np.stack([x, y, z], axis=-1)

  def _shift_grid(self, dx: float, dy: float, dyaw: float):
    """Shift/rotate the grid to compensate ego-motion.

    For small motions we approximate with a translation shift (in cells).
    Rotation is handled by rotating all evidence around the origin cell.
    """
    # Update world pose
    c, s = math.cos(self._world_yaw), math.sin(self._world_yaw)
    self._world_x += c * dx - s * dy
    self._world_y += s * dx + c * dy
    self._world_yaw += dyaw

    # Translate grid (shift in cells)
    shift_gx = int(round(dx / CELL_SIZE))
    shift_gy = int(round(dy / CELL_SIZE))

    if shift_gx != 0 or shift_gy != 0:
      self._grid = np.roll(self._grid, -shift_gx, axis=0)
      self._grid = np.roll(self._grid, -shift_gy, axis=1)

      # Zero out the region that rolled in (new unknown space)
      if shift_gx > 0:
        self._grid[-shift_gx:, :] = 0
      elif shift_gx < 0:
        self._grid[:-shift_gx, :] = 0  # type: ignore
      if shift_gy > 0:
        self._grid[:, -shift_gy:] = 0
      elif shift_gy < 0:
        self._grid[:, :-shift_gy] = 0  # type: ignore

    # Simple rotation using scipy-free affine (small angle approximation)
    if abs(dyaw) > 1e-4:
      from scipy.ndimage import rotate as ndi_rotate
      # Rotate around the ego origin cell
      rotated = ndi_rotate(self._grid, np.degrees(dyaw), reshape=False,
                           order=1, mode='constant', cval=0.0)
      self._grid = rotated.astype(np.float32)

  def _mark_free_rays(self, gx: np.ndarray, gy: np.ndarray):
    """Fast vectorized free-space marking using DDA-lite.

    For each occupied cell, mark cells along the ray from the ego origin
    as free (except the endpoint).
    """
    if len(gx) == 0:
      return

    # Unique endpoints to avoid redundant work
    endpoints = np.unique(np.stack([gx, gy], axis=-1), axis=0)
    if len(endpoints) > 500:
      # Subsample for speed
      idx = np.random.choice(len(endpoints), 500, replace=False)
      endpoints = endpoints[idx]

    ox, oy = ORIGIN_X, ORIGIN_Y

    for ex, ey in endpoints:
      # Simple line walk (Bresenham-ish with numpy)
      n_steps = max(abs(ex - ox), abs(ey - oy))
      if n_steps < 2:
        continue
      ts = np.linspace(0, 1, n_steps, endpoint=False)[1:]  # skip origin, skip endpoint
      rx = (ox + ts * (ex - ox)).astype(np.int32)
      ry = (oy + ts * (ey - oy)).astype(np.int32)
      valid = (rx >= 0) & (rx < GRID_SIZE) & (ry >= 0) & (ry < GRID_SIZE)
      rx, ry = rx[valid], ry[valid]
      np.add.at(self._grid, (rx, ry), L_FREE)
