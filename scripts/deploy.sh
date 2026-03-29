#!/bin/bash
# Deploy openpilot changes to comma device via rsync
#
# Usage:
#   ./scripts/deploy.sh [device-ip]
#   COMMA_IP=192.168.1.50 ./scripts/deploy.sh
#
# Prerequisites:
#   - SSH enabled on device (Settings → Developer → SSH)
#   - Your SSH key added on device (Settings → Developer → SSH Keys)
#   - Device is accessible on the network

set -e

COMMA_IP="${1:-${COMMA_IP:-comma}}"
COMMA_USER="comma"
REMOTE_PATH="/data/openpilot"

echo "==> Deploying to ${COMMA_USER}@${COMMA_IP}:${REMOTE_PATH}"
SSH="ssh -o StrictHostKeyChecking=no"

# 1. Depth camera dialog (new file — uses tinygrad for GPU inference)
echo "--- syncing depth_camera_dialog.py"
rsync -avz --progress \
  --exclude='__pycache__' \
  -e "$SSH" \
  selfdrive/ui/onroad/depth_camera_dialog.py \
  "${COMMA_USER}@${COMMA_IP}:${REMOTE_PATH}/selfdrive/ui/onroad/"

# 2. Device settings panel (modified — added Depth Camera button)
echo "--- syncing device.py"
rsync -avz --progress \
  --exclude='__pycache__' \
  -e "$SSH" \
  selfdrive/ui/layouts/settings/device.py \
  "${COMMA_USER}@${COMMA_IP}:${REMOTE_PATH}/selfdrive/ui/layouts/settings/"

# 3. Depth model compile script + ONNX model
echo "--- syncing depthd/"
rsync -avz --progress \
  --exclude='__pycache__' \
  --exclude='*.pkl' \
  -e "$SSH" \
  selfdrive/depthd/ \
  "${COMMA_USER}@${COMMA_IP}:${REMOTE_PATH}/selfdrive/depthd/"

# 4. Upload the simplified ONNX to the device's depth_model dir
#    (the device can't run export_depth_model.py — no torch/transformers)
echo "--- syncing depth model ONNX"
$SSH "${COMMA_USER}@${COMMA_IP}" "mkdir -p /data/depth_model"
rsync -avz --progress \
  -e "$SSH" \
  depth_model/depth_anything_v2_metric_indoor_small.sim.onnx \
  "${COMMA_USER}@${COMMA_IP}:/data/depth_model/"

echo ""
echo "==> Files deployed. Now compile the depth model on device..."
$SSH "${COMMA_USER}@${COMMA_IP}" "source /usr/local/venv/bin/activate && cd ${REMOTE_PATH} && PYTHONPATH=${REMOTE_PATH} python3 selfdrive/depthd/compile_depth_model.py"

echo ""
echo "==> Restarting openpilot..."
$SSH "${COMMA_USER}@${COMMA_IP}" "sudo systemctl restart comma"

echo "==> Done!"
echo "    Go to Settings → Device → 'Depth Camera' → tap PREVIEW"
echo "    (vehicle must be off)"
