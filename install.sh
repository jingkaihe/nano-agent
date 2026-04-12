#!/usr/bin/env bash

set -euo pipefail

REPO="jingkaihe/nano-agent"
DEFAULT_BRANCH="main"
BRANCH="${DEFAULT_BRANCH}"
PACKAGE_NAME="nano-agent"

usage() {
    cat <<EOF
Install ${PACKAGE_NAME} with uv tool install.

Usage:
  ./install.sh [--branch <name>] [--repo <owner/repo>]

Options:
  --branch <name>    Git branch to install from (default: ${DEFAULT_BRANCH})
  --repo <repo>      GitHub repository to install from (default: ${REPO})
  -h, --help         Show this help text

Examples:
  ./install.sh
  ./install.sh --branch feat/sandbox
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

have_cmd uv || die "uv is required"

REPO_URL="git+https://github.com/${REPO}"
PACKAGE_SPEC="${REPO_URL}@${BRANCH}"

log "installing ${PACKAGE_NAME} from ${REPO}@${BRANCH}"
uv tool install --force "$PACKAGE_SPEC"

BIN_DIR="$(uv tool dir --bin)"
log "installed to ${BIN_DIR}/${PACKAGE_NAME}"

if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
    log "warning: ${BIN_DIR} is not currently on PATH"
fi

log "done"
