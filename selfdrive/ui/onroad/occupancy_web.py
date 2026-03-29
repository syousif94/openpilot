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
  #container { display: flex; flex-direction: column; width: 100vw; height: 100vh; }
  #cam-section {
    flex: 0 0 55%; position: relative; background: #111;
    display: flex; justify-content: center; align-items: center;
    overflow: hidden;
  }
  #depth-cam { max-width: 100%; max-height: 100%; object-fit: contain; display: none; }
  #cam-placeholder { color: #555; font-size: 16px; position: absolute; }
  #chart-section { flex: 1; position: relative; min-height: 0; overflow: hidden; }
  #status { position: absolute; bottom: 4px; right: 8px; font-size: 11px; color: #666; z-index: 10; }
</style>
</head>
<body>
<div id="container">
  <div id="cam-section">
    <img id="depth-cam" alt="Depth Camera">
    <div id="cam-placeholder">Waiting for depth frames…</div>
    <div id="hud">
      <div>Depth Profile <b>LIVE</b></div>
      <div id="info">connecting…</div>
    </div>
    <div id="imu-panel">
      <div class="label">orientation (°)</div>
      <div><span class="axis">Y</span><span class="val" id="yaw">—</span> <span class="axis">P</span><span class="val" id="pitch">—</span> <span class="axis">R</span><span class="val" id="roll">—</span></div>
      <div class="label" style="margin-top:4px">gyro (rad/s)</div>
      <div><span class="axis">X</span><span class="val" id="gx">—</span> <span class="axis">Y</span><span class="val" id="gy">—</span> <span class="axis">Z</span><span class="val" id="gz">—</span></div>
      <div class="label" style="margin-top:4px">accel (m/s²)</div>
      <div><span class="axis">X</span><span class="val" id="ax">—</span> <span class="axis">Y</span><span class="val" id="ay">—</span> <span class="axis">Z</span><span class="val" id="az">—</span></div>
    </div>
  </div>
  <div id="chart-section">
    <canvas id="c"></canvas>
    <div id="status"></div>
  </div>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const status = document.getElementById('status');

let depthData = null;
let imu = { gyro: [0,0,0], accel: [0,0,0] };

// Dead-reckoning state for minimap
let drX = 0, drY = 0;      // world-frame position (m)
let drVx = 0, drVy = 0;    // world-frame velocity (m/s)
let drPath = [{x:0, y:0}]; // accumulated trail
let lastDRTime = null;

// ── Madgwick AHRS filter ──
// Quaternion [w, x, y, z] — starts as identity (no rotation)
let q0 = 1, q1 = 0, q2 = 0, q3 = 0;
const madgwickBeta = 0.1;  // filter gain — higher = more accel trust, less gyro drift
let madgwickInitialised = false;

// Gyro bias: calibrated from first N samples while stationary
const BIAS_SAMPLES = 20;
let biasBuf = [];
let gbiasX = 0, gbiasY = 0, gbiasZ = 0;
// Gravity calibration for position integration
let gravCalBuf = [];
let gravX = 0, gravY = 0, gravZ = 0;
let calibrated = false;

function madgwickUpdate(gx, gy, gz, ax, ay, az, dt) {
  // Normalise accelerometer
  let norm = Math.sqrt(ax*ax + ay*ay + az*az);
  if (norm < 0.01) return;  // can't determine gravity direction
  norm = 1.0 / norm;
  ax *= norm; ay *= norm; az *= norm;

  // Auxiliary variables
  const _2q0 = 2*q0, _2q1 = 2*q1, _2q2 = 2*q2, _2q3 = 2*q3;
  const _4q0 = 4*q0, _4q1 = 4*q1, _4q2 = 4*q2;
  const _8q1 = 8*q1, _8q2 = 8*q2;
  const q0q0 = q0*q0, q1q1 = q1*q1, q2q2 = q2*q2, q3q3 = q3*q3;

  // Gradient descent corrective step (objective: align estimated gravity with measured)
  let s0 = _4q0*q2q2 + _2q2*ax + _4q0*q1q1 - _2q1*ay;
  let s1 = _4q1*q3q3 - _2q3*ax + 4*q0q0*q1 - _2q0*ay - _4q1 + _8q1*q1q1 + _8q1*q2q2 + _4q1*az;
  let s2 = 4*q0q0*q2 + _2q0*ax + _4q2*q3q3 - _2q3*ay - _4q2 + _8q2*q1q1 + _8q2*q2q2 + _4q2*az;
  let s3 = 4*q1q1*q3 - _2q1*ax + 4*q2q2*q3 - _2q2*ay;
  // Normalise step
  norm = 1.0 / Math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3 + 1e-12);
  s0 *= norm; s1 *= norm; s2 *= norm; s3 *= norm;

  // Apply feedback (gyro rate minus correction)
  const beta = madgwickBeta;
  const qDot0 = 0.5*(-q1*gx - q2*gy - q3*gz) - beta*s0;
  const qDot1 = 0.5*(q0*gx + q2*gz - q3*gy)  - beta*s1;
  const qDot2 = 0.5*(q0*gy - q1*gz + q3*gx)  - beta*s2;
  const qDot3 = 0.5*(q0*gz + q1*gy - q2*gx)  - beta*s3;

  // Integrate
  q0 += qDot0 * dt;
  q1 += qDot1 * dt;
  q2 += qDot2 * dt;
  q3 += qDot3 * dt;

  // Normalise quaternion
  norm = 1.0 / Math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
  q0 *= norm; q1 *= norm; q2 *= norm; q3 *= norm;
}

function getYaw() {
  // Extract yaw (heading around Z) from quaternion
  return Math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3));
}

function getRoll() {
  return Math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2));
}

function getPitch() {
  const sinp = 2*(q0*q2 - q3*q1);
  return Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI/2 : Math.asin(sinp);
}

// Rotate a vector from sensor frame to world frame using the quaternion
function rotateToWorld(sx, sy, sz) {
  // q * v * q^-1  (for unit quaternion, q^-1 = conjugate)
  // Expanded for efficiency:
  const _2q0 = 2*q0, _2q1 = 2*q1, _2q2 = 2*q2, _2q3 = 2*q3;
  const q0q0 = q0*q0, q1q1 = q1*q1, q2q2 = q2*q2, q3q3 = q3*q3;

  const wx = (q0q0 + q1q1 - q2q2 - q3q3)*sx + (_2q1*q2 - _2q0*q3)*sy + (_2q1*q3 + _2q0*q2)*sz;
  const wy = (_2q1*q2 + _2q0*q3)*sx + (q0q0 - q1q1 + q2q2 - q3q3)*sy + (_2q2*q3 - _2q0*q1)*sz;
  const wz = (_2q1*q3 - _2q0*q2)*sx + (_2q2*q3 + _2q0*q1)*sy + (q0q0 - q1q1 - q2q2 + q3q3)*sz;

  return [wx, wy, wz];
}

function updateDeadReckoning(gyro, accel) {
  const now = performance.now() / 1000;
  if (lastDRTime === null) { lastDRTime = now; return; }
  const dt = Math.min(now - lastDRTime, 0.5);
  lastDRTime = now;
  if (dt < 0.001) return;

  // ── Calibration phase ──
  if (!calibrated) {
    biasBuf.push({x: gyro[0], y: gyro[1], z: gyro[2]});
    gravCalBuf.push({x: accel[0], y: accel[1], z: accel[2]});
    if (biasBuf.length >= BIAS_SAMPLES) {
      gbiasX = biasBuf.reduce((s,v) => s+v.x, 0) / biasBuf.length;
      gbiasY = biasBuf.reduce((s,v) => s+v.y, 0) / biasBuf.length;
      gbiasZ = biasBuf.reduce((s,v) => s+v.z, 0) / biasBuf.length;
      gravX = gravCalBuf.reduce((s,v) => s+v.x, 0) / gravCalBuf.length;
      gravY = gravCalBuf.reduce((s,v) => s+v.y, 0) / gravCalBuf.length;
      gravZ = gravCalBuf.reduce((s,v) => s+v.z, 0) / gravCalBuf.length;
      // Initialise quaternion from accelerometer (tilt only)
      const norm = Math.sqrt(gravX*gravX + gravY*gravY + gravZ*gravZ);
      const anx = gravX/norm, any = gravY/norm, anz = gravZ/norm;
      const pitch = Math.asin(-anx);
      const roll = Math.atan2(any, anz);
      // Build quaternion from Euler (yaw=0)
      const cr = Math.cos(roll/2), sr = Math.sin(roll/2);
      const cp = Math.cos(pitch/2), sp = Math.sin(pitch/2);
      q0 = cr*cp; q1 = sr*cp; q2 = cr*sp; q3 = -sr*sp;
      calibrated = true;
    }
    return;
  }

  // Bias-corrected gyro
  const gx = gyro[0] - gbiasX;
  const gy = gyro[1] - gbiasY;
  const gz = gyro[2] - gbiasZ;

  // ── Run Madgwick filter ──
  madgwickUpdate(gx, gy, gz, accel[0], accel[1], accel[2], dt);

  // ── Position dead reckoning using quaternion-rotated accel ──
  // Rotate raw accelerometer to world frame using Madgwick quaternion
  const [wax, way, waz] = rotateToWorld(accel[0], accel[1], accel[2]);

  // Subtract gravity in world frame (always straight down)
  let lax = wax;
  let lay = way;
  let laz = waz - 9.81;

  // Stationary check: if total accel magnitude ≈ 1g AND linear accel is small
  const linMag3 = Math.sqrt(lax*lax + lay*lay + laz*laz);
  const totalAccel = Math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2);
  const isStationary = linMag3 < 0.4 && Math.abs(totalAccel - 9.81) < 0.4;

  // Noise gate on horizontal linear acceleration
  const linMagH = Math.sqrt(lax*lax + lay*lay);
  if (linMagH < 0.3 || isStationary) { lax = 0; lay = 0; }

  // Integrate velocity with strong decay
  const decay = Math.exp(-8.0 * dt);
  drVx = drVx * decay + lax * dt;
  drVy = drVy * decay + lay * dt;

  // ZUPT
  const speed = Math.sqrt(drVx*drVx + drVy*drVy);
  if (speed < 0.02 || isStationary) { drVx = 0; drVy = 0; }

  drX += drVx * dt;
  drY += drVy * dt;

  const lastPt = drPath[drPath.length - 1];
  const moved = Math.sqrt((drX - lastPt.x)**2 + (drY - lastPt.y)**2);
  if (moved > 0.005) {
    drPath.push({x: drX, y: drY});
    if (drPath.length > 4000) drPath = drPath.slice(-4000);
  }
}

function resize() {
  const section = document.getElementById('chart-section');
  canvas.width = section.clientWidth;
  canvas.height = section.clientHeight;
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

  // Bearing triangle (shows device facing direction via Madgwick yaw)
  const heading = getYaw();
  const hLen = 14;
  const tipX = lx + Math.sin(heading) * hLen;
  const tipY = ly - Math.cos(heading) * hLen;
  const baseAng = 2.5;  // half-angle of triangle base (radians)
  const baseLen = 6;
  const b1x = lx + Math.sin(heading + baseAng) * baseLen;
  const b1y = ly - Math.cos(heading + baseAng) * baseLen;
  const b2x = lx + Math.sin(heading - baseAng) * baseLen;
  const b2y = ly - Math.cos(heading - baseAng) * baseLen;
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
  const bearDeg = ((heading * 180 / Math.PI) % 360 + 360) % 360;
  // Also show pitch/roll from Madgwick
  const pitchDeg = getPitch() * 180 / Math.PI;
  const rollDeg = getRoll() * 180 / Math.PI;
  ctx.fillStyle = '#0f0';
  ctx.font = '11px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('yaw ' + bearDeg.toFixed(0) + '\u00b0', mx + mW - 6, my + 14);
  ctx.fillStyle = '#0aa';
  ctx.fillText('P' + pitchDeg.toFixed(0) + '\u00b0 R' + rollDeg.toFixed(0) + '\u00b0', mx + mW - 6, my + 28);

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
    // Madgwick Euler angles
    if (calibrated) {
      const yDeg = ((getYaw() * 180 / Math.PI) % 360 + 360) % 360;
      const pDeg = getPitch() * 180 / Math.PI;
      const rDeg = getRoll() * 180 / Math.PI;
      document.getElementById('yaw').textContent = yDeg.toFixed(1);
      document.getElementById('pitch').textContent = pDeg.toFixed(1);
      document.getElementById('roll').textContent = rDeg.toFixed(1);
    }
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

// Depth camera frame streaming
const depthImg = document.getElementById('depth-cam');
const camPlaceholder = document.getElementById('cam-placeholder');
let prevFrameUrl = null;
async function fetchFrame() {
  try {
    const resp = await fetch('/frame.jpg');
    if (!resp.ok) return;
    const blob = await resp.blob();
    if (prevFrameUrl) URL.revokeObjectURL(prevFrameUrl);
    prevFrameUrl = URL.createObjectURL(blob);
    depthImg.src = prevFrameUrl;
    depthImg.style.display = 'block';
    camPlaceholder.style.display = 'none';
  } catch(e) {}
}
setInterval(fetchFrame, 250);
fetchFrame();

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
  get_depth_frame = None  # callback: () -> bytes|None (JPEG frame), set externally

  def log_message(self, fmt, *args):
    pass  # suppress access logs

  def do_GET(self):
    if self.path == '/' or self.path == '/index.html':
      self._serve_html()
    elif self.path == '/grid.json':
      self._serve_json()
    elif self.path == '/grid.png':
      self._serve_png()
    elif self.path == '/frame.jpg':
      self._serve_frame()
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

  def _serve_frame(self):
    get_fn = self.__class__.get_depth_frame
    if not get_fn:
      self.send_error(503, 'Depth frame callback not set')
      return
    jpeg = get_fn()
    if jpeg is None:
      self.send_error(503, 'No frame available yet')
      return
    self.send_response(200)
    self.send_header('Content-Type', 'image/jpeg')
    self.send_header('Content-Length', str(len(jpeg)))
    self.send_header('Cache-Control', 'no-cache')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.end_headers()
    self.wfile.write(jpeg)


class OccupancyWebServer:
  """Manages the HTTP server lifecycle in a background thread."""

  def __init__(self, grid: OccupancyGrid, port: int = WEB_PORT, get_imu=None, get_depth=None, get_depth_frame=None):
    self._grid = grid
    self._port = port
    self._get_imu = get_imu
    self._get_depth = get_depth
    self._get_depth_frame = get_depth_frame
    self._server: HTTPServer | None = None
    self._thread: threading.Thread | None = None

  def start(self):
    _GridHandler.grid = self._grid
    _GridHandler.get_imu = self._get_imu
    _GridHandler.get_depth = self._get_depth
    _GridHandler.get_depth_frame = self._get_depth_frame
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
