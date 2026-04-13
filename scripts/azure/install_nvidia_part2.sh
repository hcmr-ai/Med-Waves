#!/bin/bash
set -e

echo "=== Verifying kernel ==="
KERNEL=$(uname -r)
if [[ "$KERNEL" != "6.8.0-1025-azure" ]]; then
    echo "ERROR: Wrong kernel: $KERNEL — expected 6.8.0-1025-azure"
    exit 1
fi
echo "Kernel OK: $KERNEL"

echo "=== Stopping GDM ==="
sudo systemctl stop gdm 2>/dev/null || true

echo "=== Unloading any existing nvidia modules ==="
sudo rmmod nvidia_uvm 2>/dev/null || true
sudo rmmod nvidia-drm 2>/dev/null || true
sudo rmmod nvidia-modeset 2>/dev/null || true
sudo rmmod nvidia 2>/dev/null || true

echo "=== Downloading NVIDIA GRID driver (vGPU 18.5 / CUDA 12.8) ==="
wget -q --show-progress -P /tmp \
  https://download.microsoft.com/download/0541e1a5-dff2-4b8c-a79c-96a7664b1d49/NVIDIA-Linux-x86_64-570.195.03-grid-azure.run
chmod +x /tmp/NVIDIA-Linux-x86_64-570.195.03-grid-azure.run

echo "=== Installing NVIDIA driver ==="
sudo /tmp/NVIDIA-Linux-x86_64-570.195.03-grid-azure.run --silent

echo "=== Enabling persistence mode ==="
sudo nvidia-smi -pm 1

echo "=== Cleaning up ==="
rm /tmp/NVIDIA-Linux-x86_64-570.195.03-grid-azure.run

echo "=== Verifying installation ==="
nvidia-smi

echo "=== Installing Poetry ==="
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/home/azureuser/.local/bin:$PATH"' >> ~/.bashrc
export PATH="/home/azureuser/.local/bin:$PATH"

sudo chown azureuser:azureuser /mnt
