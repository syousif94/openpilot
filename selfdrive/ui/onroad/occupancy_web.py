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
  #imu-panel .axis { color: #888; margin-right: 2px; }
  #imu-panel .val { color: #0ff; font-weight: bold; margin-right: 8px; }
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
  <div class="label">gyro (rad/s)</div>
  <div><span class="axis">X</span><span class="val" id="gx">—</span> <span class="axis">Y</span><span class="val" id="gy">—</span> <span class="axis">Z</span><span class="val" id="gz">—</span></div>
  <div class="label" style="margin-top:4px">accel (m/s²)</div>
  <div><span class="axis">X</span><span class="val" id="ax">—</span> <span class="axis">Y</span><span class="val" id="ay">—</span> <span class="axis">Z</span><span class="val" id="az">—</span></div>
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

// Dead-reckoning state for minimap
let drHeading = 0;          // radians (0 = forward)
let drX = 0, drY = 0;      // world-frame position (m)
let drVx = 0, drVy = 0;    // world-frame velocity (m/s)
let drPath = [{x:0, y:0}]; // accumulated trail
let lastDRTime = null;

// Gravity calibration: collect samples before integrating
const GRAV_CAL_SAMPLES = 30;  // ~4.5 s at 150 ms poll
let gravCalBuf = [];           // [{x,y,z}, ...]
let gravX = 0, gravY = 0, gravZ = 0;
let gravCalibrated = false;

// Gyro bias calibration (same window)
let gyroBiasBuf = [];
let gyroBiasX = 0, gyroBiasY = 0, gyroBiasZ = 0;

// Gyro high-pass state (removes slow bias drift)
let hpGz = 0, prevRawGz = 0;

function updateDeadReckoning(gyro, accel) {
  const now = performance.now() / 1000;
  if (lastDRTime === null) { lastDRTime = now; prevRawGz = gyro[2]; return; }
  const dt = Math.min(now - lastDRTime, 0.5);
  lastDRTime = now;
  if (dt < 0.001) return;

  // ── Calibration phase: accumulate gravity + gyro bias ──
  if (!gravCalibrated) {
    gravCalBuf.push({x: accel[0], y: accel[1], z: accel[2]});
    gyroBiasBuf.push({x: gyro[0], y: gyro[1], z: gyro[2]});
    if (gravCalBuf.length >= GRAV_CAL_SAMPLES) {
      gravX = gravCalBuf.reduce((s, v) => s + v.x, 0) / gravCalBuf.length;
      gravY = gravCalBuf.reduce((s, v) => s + v.y, 0) / gravCalBuf.length;
      gravZ = gravCalBuf.reduce((s, v) => s + v.z, 0) / gravCalBuf.length;
      gyroBiasX = gyroBiasBuf.reduce((s, v) => s + v.x, 0) / gyroBiasBuf.length;
      gyroBiasY = gyroBiasBuf.reduce((s, v) => s + v.y, 0) / gyroBiasBuf.length;
      gyroBiasZ = gyroBiasBuf.reduce((s, v) => s + v.z, 0) / gyroBiasBuf.length;
      gravCalibrated = true;
      prevRawGz = gyro[2] - gyroBiasZ;
    }
    return;
  }

  // ── Heading from bias-corrected, high-pass-filtered gyro Z ──
  const rawGz = gyro[2] - gyroBiasZ;
  const hpAlpha = 0.98;
  hpGz = hpAlpha * (hpGz + rawGz - prevRawGz);
  prevRawGz = rawGz;
  const gz = Math.abs(hpGz) > 0.008 ? hpGz : 0;
  drHeading += gz * dt;

  // ── Linear acceleration (raw minus calibrated gravity) ──
  let lax = accel[0] - gravX;
  let lay = accel[1] - gravY;

  // Slowly track gravity drift (very conservative)
  gravX += 0.002 * (accel[0] - gravX);
  gravY += 0.002 * (accel[1] - gravY);
  gravZ += 0.002 * (accel[2] - gravZ);

  // Stationary check: if total accel magnitude is close to 1g, device isn't moving
  const totalAccel = Math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2);
  const isStationary = Math.abs(totalAccel - 9.81) < 0.3;

  // Noise gate
  const linMag = Math.sqrt(lax * lax + lay * lay);
  if (linMag < 0.5 || isStationary) { lax = 0; lay = 0; }

  // Rotate to world frame by current heading
  const ch = Math.cos(drHeading), sh = Math.sin(drHeading);
  const wax = lax * ch - lay * sh;
  const way = lax * sh + lay * ch;

  // Integrate velocity with very strong decay
  const decay = Math.exp(-10.0 * dt);
  drVx = drVx * decay + wax * dt;
  drVy = drVy * decay + way * dt;

  // ZUPT: zero velocity when speed is negligible
  const speed = Math.sqrt(drVx * drVx + drVy * drVy);
  if (speed < 0.02 || isStationary) { drVx = 0; drVy = 0; }

  // Integrate position
  drX += drVx * dt;
  drY += drVy * dt;

  // Only record point if we actually moved
  const lastPt = drPath[drPath.length - 1];
  const moved = Math.sqrt((drX - lastPt.x) ** 2 + (drY - lastPt.y) ** 2);
  if (moved > 0.005) {
    drPath.push({x: drX, y: drY});
    if (drPath.length > 4000) drPath = drPath.slice(-4000);
  }
}

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

  // ── Minimap ──
  drawMinimap();
}

function drawMinimap() {
  if (drPath.length < 2) return;

  const mW = 200, mH = 200;
  const mx = canvas.width - mW - 16, my = canvas.height - mH - 30;

  // Background
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  ctx.fillRect(mx, my, mW, mH);
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.strokeRect(mx, my, mW, mH);

  // Compute path bounds
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  for (const p of drPath) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  const pad = 20;
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const maxRange = Math.max(rangeX, rangeY, 0.5); // at least 0.5 m
  const scale = (Math.min(mW, mH) - 2 * pad) / maxRange;
  const cenX = (minX + maxX) / 2;
  const cenY = (minY + maxY) / 2;

  // Draw path (green line)
  ctx.beginPath();
  ctx.strokeStyle = '#0f0';
  ctx.lineWidth = 1.5;
  for (let i = 0; i < drPath.length; i++) {
    const sx = mx + mW / 2 + (drPath[i].x - cenX) * scale;
    const sy = my + mH / 2 - (drPath[i].y - cenY) * scale;
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();

  // Start marker (dim circle)
  const first = drPath[0];
  const fx = mx + mW / 2 + (first.x - cenX) * scale;
  const fy = my + mH / 2 - (first.y - cenY) * scale;
  ctx.beginPath();
  ctx.arc(fx, fy, 3, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,255,0,0.3)';
  ctx.fill();

  // Current position dot
  const last = drPath[drPath.length - 1];
  const lx = mx + mW / 2 + (last.x - cenX) * scale;
  const ly = my + mH / 2 - (last.y - cenY) * scale;
  ctx.beginPath();
  ctx.arc(lx, ly, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#0f0';
  ctx.fill();

  // Bearing triangle (shows device facing direction)
  const hLen = 14;
  const tipX = lx + Math.sin(drHeading) * hLen;
  const tipY = ly - Math.cos(drHeading) * hLen;
  const baseAng = 2.5;  // half-angle of triangle base (radians)
  const baseLen = 6;
  const b1x = lx + Math.sin(drHeading + baseAng) * baseLen;
  const b1y = ly - Math.cos(drHeading + baseAng) * baseLen;
  const b2x = lx + Math.sin(drHeading - baseAng) * baseLen;
  const b2y = ly - Math.cos(drHeading - baseAng) * baseLen;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(b1x, b1y);
  ctx.lineTo(b2x, b2y);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,255,0,0.6)';
  ctx.fill();
  ctx.strokeStyle = '#0f0';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Bearing reading (degrees, 0=north/forward, CW positive)
  const bearDeg = ((drHeading * 180 / Math.PI) % 360 + 360) % 360;
  ctx.fillStyle = '#0f0';
  ctx.font = '11px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(bearDeg.toFixed(0) + '\u00b0', mx + mW - 6, my + 14);

  // ── Scale bar ──
  const barMaxPx = mW * 0.4;
  const barMaxM = barMaxPx / scale;
  const niceSteps = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
  let scaleM = niceSteps[0];
  for (const s of niceSteps) {
    if (s <= barMaxM) scaleM = s;
    else break;
  }
  const scalePx = scaleM * scale;
  const sbx = mx + 8, sby = my + mH - 12;

  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(sbx, sby); ctx.lineTo(sbx + scalePx, sby); ctx.stroke();
  // Endcaps
  ctx.beginPath(); ctx.moveTo(sbx, sby - 4); ctx.lineTo(sbx, sby + 4); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sbx + scalePx, sby - 4); ctx.lineTo(sbx + scalePx, sby + 4); ctx.stroke();

  ctx.fillStyle = '#ccc';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  let label = scaleM >= 1 ? scaleM + ' m' : Math.round(scaleM * 100) + ' cm';
  ctx.fillText(label, sbx + scalePx + 5, sby + 3);

  // Title
  ctx.fillStyle = '#888';
  ctx.font = '11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('IMU path', mx + 6, my + 14);
}

async function fetchData() {
  try {
    const resp = await fetch('/grid.json');
    if (!resp.ok) throw new Error(resp.statusText);
    const obj = await resp.json();
    if (obj.imu) imu = obj.imu;
    if (obj.depth) depthData = obj.depth;

    updateDeadReckoning(imu.gyro, imu.accel);

    const g = imu.gyro;
    const a = imu.accel;
    document.getElementById('gx').textContent = g[0].toFixed(3);
    document.getElementById('gy').textContent = g[1].toFixed(3);
    document.getElementById('gz').textContent = g[2].toFixed(3);
    document.getElementById('ax').textContent = a[0].toFixed(2);
    document.getElementById('ay').textContent = a[1].toFixed(2);
    document.getElementById('az').textContent = a[2].toFixed(2);

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
