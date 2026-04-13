#!/bin/bash
set -e

echo "=== 1. Update & install build tools ==="
sudo apt-get update
sudo apt-get install -y build-essential

echo "=== 2. Install kernel 6.8 ==="
sudo apt-get install -y linux-image-6.8.0-1025-azure linux-headers-6.8.0-1025-azure

echo "=== 3. Hold kernel packages ==="
sudo apt-mark hold linux-image-6.17.0-1008-azure linux-headers-6.17.0-1008-azure linux-image-azure linux-headers-azure

echo "=== 4. Set kernel 6.8 as default ==="
sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-1025-azure"/' /etc/default/grub
sudo update-grub

echo "=== 5. Blacklist Nouveau drivers ==="
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

echo "=== 6. Disable conflicting services ==="
sudo systemctl disable gdm 2>/dev/null || true
sudo systemctl disable nvidia-gridd 2>/dev/null || true

echo "=== 7. Rebooting in 5 seconds... ==="
echo "After reboot, run: bash install_nvidia_part2.sh"
sleep 5
sudo reboot