#!/usr/bin/env bash

set -euo pipefail

REPO="jingkaihe/nano-agent"
DEFAULT_BRANCH="main"
DEST_DIR="${HOME}/.local/bin"
BRANCH="${DEFAULT_BRANCH}"
SCRIPT_NAME="nano-agent"

usage() {
    cat <<EOF
Install ${SCRIPT_NAME} into ~/.local/bin by default.

Usage:
  ./install.sh [--branch <name>] [--dest-dir <dir>] [--repo <owner/repo>]

Options:
  --branch <name>    Git branch to install from (default: ${DEFAULT_BRANCH})
  --dest-dir <dir>   Installation directory (default: ${DEST_DIR})
  --repo <repo>      GitHub repository to install from (default: ${REPO})
  -h, --help         Show this help text

Examples:
  ./install.sh
  ./install.sh --branch feat/sandbox
  ./install.sh --dest-dir /usr/local/bin
EOF
}

log() {
    printf '[install] %s\n' "$*"
}

die() {
    printf '[install] error: %s\n' "$*" >&2
    exit 1
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)
            [[ $# -ge 2 ]] || die "--branch requires a value"
            BRANCH="$2"
            shift 2
            ;;
        --dest-dir)
            [[ $# -ge 2 ]] || die "--dest-dir requires a value"
            DEST_DIR="$2"
            shift 2
            ;;
        --repo)
            [[ $# -ge 2 ]] || die "--repo requires a value"
            REPO="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

have_cmd git || die "git is required"
have_cmd install || die "install is required"

TMP_DIR="$(mktemp -d)"
CLONE_DIR="${TMP_DIR}/repo"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

REPO_URL="https://github.com/${REPO}.git"

log "cloning ${REPO}@${BRANCH}"
git clone --depth=1 --branch "$BRANCH" --single-branch "$REPO_URL" "$CLONE_DIR" >/dev/null 2>&1

SOURCE_PATH="${CLONE_DIR}/${SCRIPT_NAME}"
[[ -f "$SOURCE_PATH" ]] || die "could not find ${SCRIPT_NAME} in cloned repository"

mkdir -p "$DEST_DIR"
install -m 0755 "$SOURCE_PATH" "$DEST_DIR/$SCRIPT_NAME"

log "installed to $DEST_DIR/$SCRIPT_NAME"

if ! have_cmd uv; then
    log "warning: uv is not installed; ${SCRIPT_NAME} needs uv on PATH to run"
    log "install uv from https://docs.astral.sh/uv/ and then run ${SCRIPT_NAME} --help"
fi

if [[ ":$PATH:" != *":${DEST_DIR}:"* ]]; then
    log "warning: ${DEST_DIR} is not currently on PATH"
fi

log "done"
