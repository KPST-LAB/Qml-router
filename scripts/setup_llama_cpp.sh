#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#------------------------------------------------------------
#setup_llama_cpp.sh
#Purpose : fetch llama.cpp at the commit used in the paper (build b10141, 0d47ea742),
#          apply patches/llama.cpp-qml-session.patch and build llama-server with
#          the qml-router session store (LLAMA_QML_SESSION=ON)
#Usage   : scripts/setup_llama_cpp.sh [--cuda ARCH | --cpu] [--src PATH_OR_URL] [--jobs N]
#          --cuda ARCH  CUDA build for the given architecture (89 = RTX 4090,
#                       87 = Jetson AGX Orin, 90 = H100/H200); default --cuda 89
#          --cpu        build without CUDA
#          --src        llama.cpp repository to clone from
#                       (default https://github.com/ggml-org/llama.cpp)
#Output  : third_party/llama.cpp/build/bin/llama-server
#------------------------------------------------------------
set -euo pipefail

UPSTREAM_COMMIT=0d47ea742
SRC=https://github.com/ggml-org/llama.cpp
CUDA=ON
ARCH=89
JOBS=$(nproc)

while [ $# -gt 0 ]; do
  case "$1" in
    --cuda) CUDA=ON; ARCH="$2"; shift 2 ;;
    --cpu)  CUDA=OFF; shift ;;
    --src)  SRC="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '3,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
DST=$REPO/third_party/llama.cpp
PATCH=$REPO/patches/llama.cpp-qml-session.patch

if [ ! -d "$DST/.git" ]; then
  git clone "$SRC" "$DST"
fi
git -C "$DST" checkout --quiet "$UPSTREAM_COMMIT"

if git -C "$DST" apply --check "$PATCH" 2>/dev/null; then
  git -C "$DST" apply "$PATCH"
  echo "patch applied"
elif git -C "$DST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied"
else
  echo "patch does not apply to $UPSTREAM_COMMIT; check the checkout in $DST" >&2
  exit 1
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_QML_SESSION=ON -DQML_ROUTER_DIR="$REPO" -DLLAMA_CURL=OFF)
if [ "$CUDA" = ON ]; then
  CMAKE_ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH")
else
  CMAKE_ARGS+=(-DGGML_CUDA=OFF)
fi

cmake -S "$DST" -B "$DST/build" "${CMAKE_ARGS[@]}"
cmake --build "$DST/build" --target llama-server -j "$JOBS"
echo "built: $DST/build/bin/llama-server"
