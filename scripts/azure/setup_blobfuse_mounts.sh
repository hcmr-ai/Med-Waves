"""
AZURE_STORAGE_KEY='your-key-here' ./scripts/setup_blobfuse_mounts.sh
"""
set -euo pipefail

# Reusable provisioning script for blobfuse2 mounts.
# Defaults are set for the Med-WAV storage account and containers.

STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-medwavdatastorageneu}"
CONTAINER_MAIN="${CONTAINER_MAIN:-medwav-data}"
CONTAINER_SCALERS="${CONTAINER_SCALERS:-scalers}"

MOUNT_MAIN="${MOUNT_MAIN:-/mnt/blobstorage}"
MOUNT_SCALERS="${MOUNT_SCALERS:-/mnt/blobstorage-scalers}"

CACHE_ROOT="${CACHE_ROOT:-/mnt/blobfuse2-cache}"
MAIN_CACHE_PATH="${CACHE_ROOT}/${CONTAINER_MAIN}"
SCALERS_CACHE_PATH="${CACHE_ROOT}/${CONTAINER_SCALERS}"

CONFIG_MAIN="${CONFIG_MAIN:-$HOME/blobfuse2-${CONTAINER_MAIN}.yaml}"
CONFIG_SCALERS="${CONFIG_SCALERS:-$HOME/blobfuse2-${CONTAINER_SCALERS}.yaml}"

require_sudo() {
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Error: sudo is required." >&2
    exit 1
  fi
}

install_blobfuse2_if_missing() {
  if command -v blobfuse2 >/dev/null 2>&1; then
    echo "blobfuse2 already installed: $(blobfuse2 --version)"
    return
  fi

  if [[ ! -f /etc/os-release ]]; then
    echo "Error: cannot detect OS (/etc/os-release missing)." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "Error: this installer currently supports Ubuntu only." >&2
    exit 1
  fi

  local version_id="${VERSION_ID:-}"
  if [[ -z "$version_id" ]]; then
    echo "Error: could not detect Ubuntu VERSION_ID." >&2
    exit 1
  fi

  local ms_deb="/tmp/packages-microsoft-prod.deb"
  local repo_url="https://packages.microsoft.com/config/ubuntu/${version_id}/packages-microsoft-prod.deb"

  echo "Adding Microsoft package feed for Ubuntu ${version_id}..."
  sudo apt-get update
  sudo apt-get install -y wget ca-certificates
  wget "$repo_url" -O "$ms_deb"
  sudo dpkg -i "$ms_deb"
  rm -f "$ms_deb"

  echo "Installing blobfuse2..."
  sudo apt-get update
  sudo apt-get install -y blobfuse2 fuse3
  echo "Installed: $(blobfuse2 --version)"
}

read_storage_key() {
  if [[ -n "${AZURE_STORAGE_KEY:-}" ]]; then
    STORAGE_KEY="$AZURE_STORAGE_KEY"
    return
  fi

  read -r -s -p "Enter storage account key for ${STORAGE_ACCOUNT}: " STORAGE_KEY
  echo
  if [[ -z "$STORAGE_KEY" ]]; then
    echo "Error: storage account key is empty." >&2
    exit 1
  fi
}

write_config() {
  local config_path="$1"
  local container_name="$2"
  local cache_path="$3"

  cat > "$config_path" <<EOF
logging:
  type: syslog
  level: log_warning
components:
  - libfuse
  - file_cache
  - azstorage
file_cache:
  path: $cache_path
  timeout-sec: 120
  max-size-mb: 4096
azstorage:
  type: block
  account-name: $STORAGE_ACCOUNT
  container: $container_name
  mode: key
  account-key: $STORAGE_KEY
EOF
  chmod 600 "$config_path"
}

mount_one() {
  local mount_path="$1"
  local config_path="$2"

  if mountpoint -q "$mount_path"; then
    echo "Already mounted: $mount_path"
    return
  fi

  blobfuse2 mount "$mount_path" --config-file="$config_path"
  echo "Mounted: $mount_path"
}

main() {
  require_sudo
  install_blobfuse2_if_missing
  read_storage_key

  sudo mkdir -p "$MOUNT_MAIN" "$MOUNT_SCALERS"
  sudo mkdir -p "$MAIN_CACHE_PATH" "$SCALERS_CACHE_PATH"
  sudo chown -R "$USER:$USER" "$MOUNT_MAIN" "$MOUNT_SCALERS" "$CACHE_ROOT"

  write_config "$CONFIG_MAIN" "$CONTAINER_MAIN" "$MAIN_CACHE_PATH"
  write_config "$CONFIG_SCALERS" "$CONTAINER_SCALERS" "$SCALERS_CACHE_PATH"

  mount_one "$MOUNT_MAIN" "$CONFIG_MAIN"
  mount_one "$MOUNT_SCALERS" "$CONFIG_SCALERS"

  echo
  echo "Verification:"
  mountpoint -q "$MOUNT_MAIN" && echo " - OK: $MOUNT_MAIN"
  mountpoint -q "$MOUNT_SCALERS" && echo " - OK: $MOUNT_SCALERS"
  echo
  echo "Config files:"
  echo " - $CONFIG_MAIN"
  echo " - $CONFIG_SCALERS"
}

main "$@"
