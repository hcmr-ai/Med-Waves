#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/install_repo_skills.sh [options]

Install all repo skills from ./skills into Codex, Claude, and Cursor skill directories.

Options:
  --source-dir PATH        Source skills directory. Default: skills
  --targets LIST          Comma-separated targets: codex,claude,cursor
                          Default: codex,claude,cursor
  --mode MODE             Install mode: symlink or copy. Default: symlink
  --force                 Replace conflicting existing installs
  --dry-run               Print planned actions without changing anything
  -h, --help              Show this help

Environment overrides:
  CODEX_SKILLS_DIR        Default: $HOME/.codex/skills
  CLAUDE_SKILLS_DIR       Default: $HOME/.claude/skills
  CURSOR_SKILLS_DIR       Default: $HOME/.cursor/skills

Examples:
  scripts/install_repo_skills.sh
  scripts/install_repo_skills.sh --mode copy
  scripts/install_repo_skills.sh --targets codex,claude
  CURSOR_SKILLS_DIR="$HOME/.agents/skills" scripts/install_repo_skills.sh --targets cursor
EOF
}

SOURCE_DIR="skills"
TARGETS_RAW="codex,claude,cursor"
MODE="symlink"
FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --source-dir" >&2; exit 1; }
      SOURCE_DIR="$2"
      shift 2
      ;;
    --targets)
      [[ $# -ge 2 ]] || { echo "Missing value for --targets" >&2; exit 1; }
      TARGETS_RAW="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "Missing value for --mode" >&2; exit 1; }
      MODE="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ "$MODE" == "symlink" || "$MODE" == "copy" ]] || {
  echo "Invalid --mode '$MODE'. Expected 'symlink' or 'copy'." >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR_ABS="$(cd "$REPO_ROOT" && cd "$SOURCE_DIR" 2>/dev/null && pwd)" || {
  echo "Source skills directory not found: $SOURCE_DIR" >&2
  exit 1
}

SKILL_DIRS=()
while IFS= read -r skill_dir; do
  SKILL_DIRS+=("$skill_dir")
done < <(find "$SOURCE_DIR_ABS" -mindepth 1 -maxdepth 1 -type d | sort)

[[ ${#SKILL_DIRS[@]} -gt 0 ]] || {
  echo "No skill directories found in $SOURCE_DIR_ABS" >&2
  exit 1
}

for skill_dir in "${SKILL_DIRS[@]}"; do
  [[ -f "$skill_dir/SKILL.md" ]] || {
    echo "Skill directory missing SKILL.md: $skill_dir" >&2
    exit 1
  }
done

CODEX_SKILLS_DIR="${CODEX_SKILLS_DIR:-$HOME/.codex/skills}"
CLAUDE_SKILLS_DIR="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
CURSOR_SKILLS_DIR="${CURSOR_SKILLS_DIR:-$HOME/.cursor/skills}"

IFS=',' read -r -a TARGETS <<< "$TARGETS_RAW"
[[ ${#TARGETS[@]} -gt 0 ]] || {
  echo "No targets resolved from --targets" >&2
  exit 1
}

log() {
  printf '%s\n' "$*"
}

run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

ensure_dir() {
  local dir="$1"
  run_cmd mkdir -p "$dir"
}

remove_path() {
  local path="$1"
  run_cmd rm -rf "$path"
}

install_one() {
  local target_name="$1"
  local target_root="$2"
  local source_skill_dir="$3"
  local skill_name
  local dest_path
  local source_real
  local dest_real

  skill_name="$(basename "$source_skill_dir")"
  dest_path="$target_root/$skill_name"
  source_real="$(cd "$source_skill_dir" && pwd)"

  ensure_dir "$target_root"

  if [[ -e "$dest_path" || -L "$dest_path" ]]; then
    if [[ -L "$dest_path" ]]; then
      dest_real="$(readlink "$dest_path" 2>/dev/null || true)"
      if [[ "$MODE" == "symlink" && "$dest_real" == "$source_real" ]]; then
        log "[$target_name] already installed: $skill_name"
        return 0
      fi
    fi

    if [[ $FORCE -ne 1 ]]; then
      echo "[$target_name] destination exists, use --force to replace: $dest_path" >&2
      return 1
    fi

    log "[$target_name] replacing existing install: $skill_name"
    remove_path "$dest_path"
  fi

  if [[ "$MODE" == "symlink" ]]; then
    log "[$target_name] linking $skill_name -> $dest_path"
    run_cmd ln -s "$source_skill_dir" "$dest_path"
  else
    log "[$target_name] copying $skill_name -> $dest_path"
    run_cmd cp -R "$source_skill_dir" "$dest_path"
  fi
}

for target in "${TARGETS[@]}"; do
  target="${target//[[:space:]]/}"
  [[ -n "$target" ]] || continue
  case "$target" in
    codex)
      target_root="$CODEX_SKILLS_DIR"
      ;;
    claude)
      target_root="$CLAUDE_SKILLS_DIR"
      ;;
    cursor)
      target_root="$CURSOR_SKILLS_DIR"
      ;;
    *)
      echo "Unknown target '$target'. Expected codex, claude, or cursor." >&2
      exit 1
      ;;
  esac
  log "Installing repo skills into $target_root"

  for skill_dir in "${SKILL_DIRS[@]}"; do
    install_one "$target" "$target_root" "$skill_dir"
  done
done

log "Completed skill installation."
