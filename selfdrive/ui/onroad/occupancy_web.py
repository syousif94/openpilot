"""
Occupancy Grid Web Server: serves a real-time 3D/2D visualisation of the
occupancy grid over the network.

Usage:
  Starts automatically when the DepthCameraDialog opens.
  Browse to http://<device-ip>:8099 to see the live grid.

Endpoints:
  GET /              → HTML+JS viewer (Three.js)
  GET /grid.json     → current grid data + metadata as JSON
  GET /grid.png      → current grid as a PNG image
"""

import io
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

from openpilot.selfdrive.ui.onroad.occupancy_grid import (
  OccupancyGrid, GRID_SIZE, CELL_SIZE, ORIGIN_X, ORIGIN_Y,
)

WEB_PORT = 8099


def _grid_to_png_bytes(grid: 'OccupancyGrid') -> bytes:
  """Render the grid as a PNG image (grayscale: 0=free, 128=unknown, 255=occupied)."""
  binary = grid.get_binary_grid()
  # Flip X axis so forward is up in the image
  binary = binary[::-1, :]

  try:
    from PIL import Image
    img = Image.fromarray(binary, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()
  except ImportError:
    # Fallback: minimal PNG encoder (uncompressed)
    return _minimal_png(binary)


def _minimal_png(data: np.ndarray) -> bytes:
  """Produce a minimal valid PNG from a 2D uint8 array without PIL."""
  import struct
  import zlib

  h, w = data.shape
  raw = b''
  for row in range(h):
    raw += b'\x00' + data[row].tobytes()  # filter byte 0 (None)
  compressed = zlib.compress(raw)

  def chunk(ctype, cdata):
    c = ctype + cdata
    return struct.pack('>I', len(cdata)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

  sig = b'\x89PNG\r\n\x1a\n'
  ihdr = struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
  return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')


# ── HTML / JS viewer ────────────────────────────────────────────────
_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occupancy Grid — openpilot</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; color: #eee; font-family: 'SF Mono', 'Fira Code', monospace; overflow: hidden; }
  #hud {
    position: absolute; top: 12px; left: 16px; z-index: 10;
    font-size: 13px; line-height: 1.7; background: rgba(0,0,0,0.65);
    padding: 10px 16px; border-radius: 8px; pointer-events: none;
    backdrop-filter: blur(4px);
  }
  #hud b { color: #0f0; }
  #imu-panel {
    position: absolute; top: 12px; right: 16px; z-index: 10;
    font-size: 12px; line-height: 1.6; background: rgba(0,0,0,0.65);
    padding: 10px 14px; border-radius: 8px; pointer-events: none;
    backdrop-filter: blur(4px); min-width: 180px;
  }
  #imu-panel .label { color: #888; }
  #imu-panel .val { color: #0ff; font-weight: bold; }
  canvas { display: block; width: 100vw; height: 100vh; }
  #status { position: absolute; bottom: 12px; right: 16px; font-size: 11px; color: #666; z-index: 10; }
</style>
</head>
<body>
<div id="hud">
  <div>Occupancy Grid <b>LIVE</b></div>
  <div id="info">connecting…</div>
</div>
<div id="imu-panel">
  <div><span class="label">gyro:</span> <span class="val" id="gyro-val">—</span></div>
  <div><span class="label">accel:</span> <span class="val" id="accel-val">—</span></div>
  <div><span class="label">yaw:</span> <span class="val" id="yaw-val">—</span></div>
  <div><span class="label">pos:</span> <span class="val" id="pos-val">—</span></div>
</div>
<div id="status"></div>
<canvas id="c"></canvas>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const status = document.getElementById('status');

const GRID = """ + str(GRID_SIZE) + """;
const CELL = """ + str(CELL_SIZE) + """;
const OX = """ + str(ORIGIN_X) + """;
const OY = """ + str(ORIGIN_Y) + """;

let gridData = null;
let meta = {};
let trail = [];
let imu = { gyro: [0,0,0], accel: [0,0,0] };

// Smooth interpolation targets
let displayYaw = 0;
let targetYaw = 0;

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

function cellColor(prob) {
  if (prob < 0.35) {
    const t = prob / 0.35;
    return `rgb(${Math.round(5 + t * 15)},${Math.round(15 + t * 35)},${Math.round(40 + t * 30)})`;
  } else if (prob < 0.65) {
    return '#111';
  } else {
    const t = (prob - 0.65) / 0.35;
    const r = Math.round(200 + t * 55);
    const g = Math.round(100 - t * 80);
    const b = Math.round(15 + t * 10);
    return `rgb(${r},${g},${b})`;
  }
}

function lerp(a, b, t) { return a + (b - a) * t; }

// Smooth angle interpolation
function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

function draw() {
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);

  if (!gridData) {
    ctx.fillStyle = '#444';
    ctx.font = '20px monospace';
    ctx.fillText('Waiting for data...', 40, 60);
    return;
  }

  // Smoothly interpolate yaw
  displayYaw = lerpAngle(displayYaw, targetYaw, 0.15);

  const pad = 40;
  const cellPx = Math.min((W - 2*pad) / GRID, (H - 2*pad) / GRID);
  const gridPx = GRID * cellPx;
  const centerX = W / 2;
  const centerY = H / 2;

  // Ego position in grid pixel space (before rotation)
  const egoGx = OX;
  const egoGy = OY;

  ctx.save();
  // Move canvas center to screen center, rotate around ego
  ctx.translate(centerX, centerY);
  ctx.rotate(-displayYaw);  // rotate grid so device heading is always "up"

  // Offset so ego cell is at origin
  const gridOffX = -egoGy * cellPx - cellPx/2;
  const gridOffY = (egoGx - GRID + 1) * cellPx + cellPx/2;

  // Draw grid cells
  for (let gx = 0; gx < GRID; gx++) {
    for (let gy = 0; gy < GRID; gy++) {
      const idx = gx * GRID + gy;
      const prob = gridData[idx];
      if (prob > 0.4 && prob < 0.6) continue;

      const screenX = gridOffX + gy * cellPx;
      const screenY = gridOffY + (GRID - 1 - gx) * cellPx;

      ctx.fillStyle = cellColor(prob);
      ctx.fillRect(screenX, screenY, cellPx + 0.5, cellPx + 0.5);
    }
  }

  // Range rings
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 0.5;
  for (let r = 2; r <= 10; r += 2) {
    const rpx = r / CELL * cellPx;
    ctx.beginPath();
    ctx.arc(0, 0, rpx, 0, 2*Math.PI);
    ctx.stroke();
    // Label
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.font = '10px monospace';
    ctx.fillText(r + 'm', 4, -rpx + 12);
  }

  ctx.restore();

  // Draw trail in screen space (world coordinates → screen)
  if (trail.length > 1) {
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(-displayYaw);

    const wx0 = meta.world_x || 0;
    const wy0 = meta.world_y || 0;

    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0,255,100,0.4)';
    ctx.lineWidth = 2;
    for (let i = 0; i < trail.length; i++) {
      // Trail points are world coords; convert relative to current ego
      const relX = trail[i][0] - wx0;  // forward delta
      const relY = trail[i][1] - wy0;  // left delta
      // In grid space: relX → up (negative Y on screen), relY → left (negative X)
      const sx = -relY / CELL * cellPx;
      const sy = -relX / CELL * cellPx;
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // Trail dots
    for (let i = 0; i < trail.length; i++) {
      const relX = trail[i][0] - wx0;
      const relY = trail[i][1] - wy0;
      const sx = -relY / CELL * cellPx;
      const sy = -relX / CELL * cellPx;
      const age = i / trail.length;
      ctx.fillStyle = `rgba(0,255,100,${0.1 + age * 0.5})`;
      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, 2*Math.PI);
      ctx.fill();
    }

    ctx.restore();
  }

  // Draw ego marker (always at center, always pointing up)
  ctx.save();
  ctx.translate(centerX, centerY);
  const sz = Math.max(cellPx * 4, 12);
  // Glow
  ctx.shadowColor = '#0f0';
  ctx.shadowBlur = 15;
  ctx.fillStyle = '#0f0';
  ctx.beginPath();
  ctx.moveTo(0, -sz);
  ctx.lineTo(-sz * 0.5, sz * 0.3);
  ctx.lineTo(0, sz * 0.1);
  ctx.lineTo(sz * 0.5, sz * 0.3);
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  // Outline
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();

  // Scale info
  ctx.fillStyle = '#555';
  ctx.font = '12px monospace';
  const mPerGrid = (GRID * CELL).toFixed(0);
  ctx.fillText(`${mPerGrid}m × ${mPerGrid}m | cell=${CELL}m`, 16, H - 16);

  // IMU visualization: small attitude indicator
  drawAttitudeIndicator(W - 110, H - 110, 40);
}

function drawAttitudeIndicator(cx, cy, r) {
  const gx = imu.gyro[0] || 0;
  const gy = imu.gyro[1] || 0;
  const gz = imu.gyro[2] || 0;

  ctx.save();
  ctx.translate(cx, cy);

  // Background circle
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, 2*Math.PI);
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Gyro activity bars (show rotation rates)
  const maxRate = 3.0; // rad/s
  // X = pitch (red)
  const bh = r * 0.6;
  ctx.fillStyle = `rgba(255,80,80,${Math.min(1, Math.abs(gx)/maxRate * 0.8 + 0.2)})`;
  ctx.fillRect(-r + 4, -bh * Math.min(1, gx/maxRate), 6, bh * Math.min(1, Math.abs(gx)/maxRate));
  // Y = roll (blue)
  ctx.fillStyle = `rgba(80,80,255,${Math.min(1, Math.abs(gy)/maxRate * 0.8 + 0.2)})`;
  ctx.fillRect(-r + 14, -bh * Math.min(1, gy/maxRate), 6, bh * Math.min(1, Math.abs(gy)/maxRate));
  // Z = yaw (green) — arc
  ctx.beginPath();
  ctx.arc(0, 0, r * 0.75, -Math.PI/2, -Math.PI/2 + Math.min(Math.PI, Math.abs(gz)) * Math.sign(gz));
  ctx.strokeStyle = `rgba(0,255,0,${Math.min(1, Math.abs(gz)/maxRate * 0.8 + 0.2)})`;
  ctx.lineWidth = 3;
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.fillText('IMU', -10, r + 14);

  ctx.restore();
}

async function fetchGrid() {
  try {
    const resp = await fetch('/grid.json');
    if (!resp.ok) throw new Error(resp.statusText);
    const obj = await resp.json();
    meta = obj.meta;
    trail = obj.trail || [];
    if (obj.imu) imu = obj.imu;

    gridData = new Float32Array(obj.grid.length);
    for (let i = 0; i < obj.grid.length; i++) {
      gridData[i] = 1.0 / (1.0 + Math.exp(-obj.grid[i]));
    }

    targetYaw = meta.world_yaw || 0;

    info.innerHTML = `frames: <b>${meta.frame_count}</b> | ` +
      `pose: (${meta.world_x.toFixed(2)}, ${meta.world_y.toFixed(2)}) yaw=${(meta.world_yaw * 180/Math.PI).toFixed(1)}°`;

    // Update IMU readouts
    const g = imu.gyro;
    const a = imu.accel;
    document.getElementById('gyro-val').textContent =
      `${g[0].toFixed(2)} ${g[1].toFixed(2)} ${g[2].toFixed(2)}`;
    document.getElementById('accel-val').textContent =
      `${a[0].toFixed(1)} ${a[1].toFixed(1)} ${a[2].toFixed(1)}`;
    document.getElementById('yaw-val').textContent =
      `${(meta.world_yaw * 180/Math.PI).toFixed(1)}°`;
    document.getElementById('pos-val').textContent =
      `${meta.world_x.toFixed(2)}, ${meta.world_y.toFixed(2)}`;

    lastFetch = performance.now();
  } catch(e) {
    status.textContent = 'fetch error: ' + e.message;
  }
}

function loop() {
  draw();
  requestAnimationFrame(loop);
}

setInterval(fetchGrid, 150);  // ~7 Hz
fetchGrid();
loop();
</script>
</body>
</html>"""


class _GridHandler(BaseHTTPRequestHandler):
  """HTTP handler — serves the web viewer and grid data."""

  grid: OccupancyGrid | None = None  # set externally before starting server
  get_imu = None  # callback: () -> dict with gyro/accel, set externally

  def log_message(self, fmt, *args):
    pass  # suppress access logs

  def do_GET(self):
    if self.path == '/' or self.path == '/index.html':
      self._serve_html()
    elif self.path == '/grid.json':
      self._serve_json()
    elif self.path == '/grid.png':
      self._serve_png()
    else:
      self.send_error(404)

  def _serve_html(self):
    data = _HTML_PAGE.encode('utf-8')
    self.send_response(200)
    self.send_header('Content-Type', 'text/html; charset=utf-8')
    self.send_header('Content-Length', str(len(data)))
    self.send_header('Cache-Control', 'no-cache')
    self.end_headers()
    self.wfile.write(data)

  def _serve_json(self):
    grid = self.__class__.grid
    if grid is None:
      self.send_error(503, 'Grid not initialized')
      return
    raw = grid.get_grid()
    meta = grid.get_metadata()
    trail = grid.get_trail()
    # Downsample trail for JSON size
    trail_out = [[round(t[0], 3), round(t[1], 3), round(t[2], 3)] for t in trail[::max(1, len(trail) // 200)]]
    obj = {
      'meta': meta,
      'grid': raw.flatten().tolist(),
      'trail': trail_out,
    }
    # Add IMU data if available
    get_imu = self.__class__.get_imu
    if get_imu:
      obj['imu'] = get_imu()
    data = json.dumps(obj).encode('utf-8')
    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', str(len(data)))
    self.send_header('Cache-Control', 'no-cache')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.end_headers()
    self.wfile.write(data)

  def _serve_png(self):
    grid = self.__class__.grid
    if grid is None:
      self.send_error(503, 'Grid not initialized')
      return
    data = _grid_to_png_bytes(grid)
    self.send_response(200)
    self.send_header('Content-Type', 'image/png')
    self.send_header('Content-Length', str(len(data)))
    self.send_header('Cache-Control', 'no-cache')
    self.end_headers()
    self.wfile.write(data)


class OccupancyWebServer:
  """Manages the HTTP server lifecycle in a background thread."""

  def __init__(self, grid: OccupancyGrid, port: int = WEB_PORT, get_imu=None):
    self._grid = grid
    self._port = port
    self._get_imu = get_imu
    self._server: HTTPServer | None = None
    self._thread: threading.Thread | None = None

  def start(self):
    _GridHandler.grid = self._grid
    _GridHandler.get_imu = self._get_imu
    self._server = HTTPServer(('0.0.0.0', self._port), _GridHandler)
    self._server.timeout = 1.0
    self._thread = threading.Thread(target=self._run, daemon=True, name='occ-web')
    self._thread.start()

    from openpilot.common.swaglog import cloudlog
    cloudlog.info(f"occupancy web server: http://0.0.0.0:{self._port}")

  def _run(self):
    while self._server:
      try:
        self._server.handle_request()
      except Exception:
        break

  def stop(self):
    if self._server:
      self._server.shutdown()
      self._server = None
    if self._thread:
      self._thread.join(timeout=3)
      self._thread = None
