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
<title>Depth Profile — openpilot</title>
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
  <div>Depth Profile <b>LIVE</b></div>
  <div id="info">connecting…</div>
</div>
<div id="imu-panel">
  <div><span class="label">gyro:</span> <span class="val" id="gyro-val">—</span></div>
  <div><span class="label">accel:</span> <span class="val" id="accel-val">—</span></div>
</div>
<div id="status"></div>
<canvas id="c"></canvas>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const status = document.getElementById('status');

let depthData = null;
let imu = { gyro: [0,0,0], accel: [0,0,0] };

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

function draw() {
  const W = canvas.width;
  const H = canvas.height;
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);

  if (!depthData) {
    ctx.fillStyle = '#444';
    ctx.font = '20px monospace';
    ctx.fillText('Waiting for depth data...', 40, 60);
    return;
  }

  const closest = depthData.closest;
  const farthest = depthData.farthest;
  const cols = closest.length;
  if (cols === 0) return;

  // Find global min/max across both lines for Y scaling
  let gMin = Infinity, gMax = -Infinity;
  for (let i = 0; i < cols; i++) {
    if (farthest[i] < gMin) gMin = farthest[i];
    if (closest[i] > gMax) gMax = closest[i];
  }
  const range = gMax - gMin || 1;

  // Chart area
  const pad = { top: 80, bottom: 50, left: 70, right: 30 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 0.5;
  const nGridY = 6;
  for (let i = 0; i <= nGridY; i++) {
    const y = pad.top + (i / nGridY) * ch;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + cw, y);
    ctx.stroke();
    const val = gMin + (i / nGridY) * range;
    ctx.fillStyle = '#555';
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(1), pad.left - 8, y + 4);
  }

  // X axis labels
  ctx.fillStyle = '#555';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  const nGridX = 8;
  for (let i = 0; i <= nGridX; i++) {
    const x = pad.left + (i / nGridX) * cw;
    const col = Math.round((i / nGridX) * (cols - 1));
    ctx.fillText(col, x, pad.top + ch + 20);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + ch);
    ctx.stroke();
  }

  // Fill area between the two lines
  ctx.beginPath();
  for (let i = 0; i < cols; i++) {
    const x = pad.left + (i / (cols - 1)) * cw;
    const y = pad.top + ((closest[i] - gMin) / range) * ch;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = cols - 1; i >= 0; i--) {
    const x = pad.left + (i / (cols - 1)) * cw;
    const y = pad.top + ((farthest[i] - gMin) / range) * ch;
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(100, 180, 255, 0.08)';
  ctx.fill();

  // Draw farthest line (blue)
  ctx.beginPath();
  ctx.strokeStyle = '#4488ff';
  ctx.lineWidth = 2;
  for (let i = 0; i < cols; i++) {
    const x = pad.left + (i / (cols - 1)) * cw;
    const y = pad.top + ((farthest[i] - gMin) / range) * ch;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw closest line (red/orange)
  ctx.beginPath();
  ctx.strokeStyle = '#ff6633';
  ctx.lineWidth = 2;
  for (let i = 0; i < cols; i++) {
    const x = pad.left + (i / (cols - 1)) * cw;
    const y = pad.top + ((closest[i] - gMin) / range) * ch;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Legend
  const lx = pad.left + 10;
  const ly = pad.top + 20;
  ctx.font = '13px monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#ff6633';
  ctx.fillRect(lx, ly - 8, 16, 3);
  ctx.fillText('closest', lx + 22, ly);
  ctx.fillStyle = '#4488ff';
  ctx.fillRect(lx, ly + 14, 16, 3);
  ctx.fillText('farthest', lx + 22, ly + 22);

  // Axis labels
  ctx.fillStyle = '#888';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('pixel column', pad.left + cw / 2, H - 10);
  ctx.save();
  ctx.translate(14, pad.top + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('inverse depth (MiDaS)', 0, 0);
  ctx.restore();
}

async function fetchData() {
  try {
    const resp = await fetch('/grid.json');
    if (!resp.ok) throw new Error(resp.statusText);
    const obj = await resp.json();
    if (obj.imu) imu = obj.imu;
    if (obj.depth) depthData = obj.depth;

    // Debug: log full IMU payload including debug info
    if (obj.imu && obj.imu.debug) {
      console.log('IMU debug:', JSON.stringify(obj.imu.debug));
    }
    console.log('IMU gyro:', imu.gyro, 'accel:', imu.accel);

    const g = imu.gyro;
    const a = imu.accel;
    document.getElementById('gyro-val').textContent =
      g[0].toFixed(2) + ' ' + g[1].toFixed(2) + ' ' + g[2].toFixed(2);
    document.getElementById('accel-val').textContent =
      a[0].toFixed(1) + ' ' + a[1].toFixed(1) + ' ' + a[2].toFixed(1);

    const cols = depthData ? depthData.closest.length : 0;
    info.innerHTML = 'columns: <b>' + cols + '</b>';
  } catch(e) {
    status.textContent = 'fetch error: ' + e.message;
  }
}

function loop() {
  draw();
  requestAnimationFrame(loop);
}

setInterval(fetchData, 150);
fetchData();
loop();
</script>
</body>
</html>"""


class _GridHandler(BaseHTTPRequestHandler):
  """HTTP handler — serves the web viewer and grid data."""

  grid: OccupancyGrid | None = None  # set externally before starting server
  get_imu = None  # callback: () -> dict with gyro/accel, set externally
  get_depth = None  # callback: () -> dict with closest/farthest columns, set externally

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
    # Add depth column data if available
    get_depth = self.__class__.get_depth
    if get_depth:
      depth = get_depth()
      if depth:
        obj['depth'] = depth
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

  def __init__(self, grid: OccupancyGrid, port: int = WEB_PORT, get_imu=None, get_depth=None):
    self._grid = grid
    self._port = port
    self._get_imu = get_imu
    self._get_depth = get_depth
    self._server: HTTPServer | None = None
    self._thread: threading.Thread | None = None

  def start(self):
    _GridHandler.grid = self._grid
    _GridHandler.get_imu = self._get_imu
    _GridHandler.get_depth = self._get_depth
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
