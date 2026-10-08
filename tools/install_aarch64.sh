#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════
# WBench installation for aarch64 + Blackwell (DGX Station GB300 / Spark)
#
# tools/install.sh pins torch 2.4.0 and xformers 0.0.27.post2: a 2024 stack for x86 GPUs.
# That torch has no kernels for Blackwell, and that xformers has no aarch64 wheel (it
# compiles flash-attention from source and fails). This variant:
#   - installs a current CUDA torch from the PyTorch index, then FAILS FAST if CUDA is not
#     usable (so a CPU-only torch can never go unnoticed),
#   - skips xformers (optional; the code falls back to PyTorch SDPA),
#   - builds torch-scatter from source instead of using the x86-only prebuilt wheel,
#   - re-checks that torch is still the CUDA build after every install step.
# Everything else matches tools/install.sh.
#
# Usage:  bash tools/install_aarch64.sh [env_name] [torch_cuda_tag]
#   env_name        default: wbench
#   torch_cuda_tag  default: cu130  (GB300 needs CUDA 12.9 or newer; cu128/cu129 also exist)
# Optional environment variables:
#   PYTHON_VERSION  default 3.10
#   TORCH_CUDA_ARCH_LIST  set only to force the MegaSAM/torch-scatter build architecture
#                         (unset = autodetect from the visible GPU)
# ═══════════════════════════════════════════════════════════════════

if [ "$(uname -m)" != "aarch64" ]; then
    echo "[ERROR] This script is for aarch64 machines (found $(uname -m)). Use tools/install.sh."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".gitmodules" ]; then
    echo ""
    echo "[0/7] Initializing git submodules ..."
    git submodule sync --recursive
    git submodule update --init --recursive
fi

ENV_NAME="${1:-wbench}"
TORCH_CUDA="${2:-cu130}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX="https://download.pytorch.org/whl/$TORCH_CUDA"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           WBench Environment Installation (aarch64)          ║"
echo "║  Env: $ENV_NAME | Python $PYTHON_VERSION | torch from $TORCH_CUDA"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Create conda environment ─────────────────────────────────────
echo "[1/7] Creating conda environment: $ENV_NAME ..."
conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
CONDA_PREFIX=$(conda env list | grep "^$ENV_NAME " | awk '{print $NF}')
PY="$CONDA_PREFIX/bin/python"
PIP="$CONDA_PREFIX/bin/pip"

echo "       Python: $($PY --version)"
echo "       Path:   $CONDA_PREFIX"

# ── 2. Setup libstdc++ (fixes GLIBCXX_3.4.29 not found) ─────────────
echo ""
echo "[2/7] Linking system libstdc++ (GLIBCXX_3.4.29) ..."
SYSTEM_LIBSTDCXX="/usr/local/conda/lib/libstdc++.so.6"
if [ -f "$SYSTEM_LIBSTDCXX" ]; then
    cp -f "$SYSTEM_LIBSTDCXX" "$CONDA_PREFIX/lib/libstdc++.so.6"
    echo "       Copied $SYSTEM_LIBSTDCXX → $CONDA_PREFIX/lib/"
else
    echo "       [WARN] $SYSTEM_LIBSTDCXX not found, may get GLIBCXX errors later"
fi

# Abort unless torch is a CUDA build that can see the GPU. Run after every install step: a
# later package that depends on torch must not be able to replace it with a CPU build.
check_cuda_torch() {
    "$PY" - <<'EOF' || { echo "[ERROR] torch is not a working CUDA build (see above). Aborting."; exit 1; }
import sys
import torch

print(f"       torch {torch.__version__} | CUDA build: {torch.version.cuda} | "
      f"available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(1)
print(f"       device: {torch.cuda.get_device_name(0)} | capability: {torch.cuda.get_device_capability(0)}")
EOF
}

# ── 3. Install uv and PyTorch ───────────────────────────────────────
echo ""
echo "[3/7] Installing uv + PyTorch from $TORCH_INDEX ..."
$PIP install uv
UV="$CONDA_PREFIX/bin/uv"

# --index-url (not --extra-index-url): with PyPI also in play, "best match" can pick a newer
# CPU-only aarch64 torch from PyPI over the CUDA build.
$UV pip install --python "$PY" torch torchvision --index-url "$TORCH_INDEX"
check_cuda_torch

# ── 4. Remaining Python packages ────────────────────────────────────
echo ""
echo "[4/7] Installing remaining Python packages ..."
$UV pip install --python "$PY" \
    -r tools/requirements.txt \
    --index-strategy unsafe-best-match
check_cuda_torch

# ── 5. Extra packages ───────────────────────────────────────────────
echo ""
echo "[5/7] Installing torch-scatter (source build) and tools deps ..."
echo "       xformers: skipped on purpose (optional; falls back to PyTorch SDPA)."

# The prebuilt wheel index (data.pyg.org) has no aarch64 wheel for this torch: build locally.
$UV pip install --python "$PY" setuptools wheel
$UV pip install --python "$PY" torch-scatter --no-build-isolation \
    || echo "       [WARN] torch-scatter build failed (needed for MegaSAM only)"

$UV pip install --python "$PY" \
    wandb iopath hydra-core fire "trl<1.0" \
    --index-strategy unsafe-best-match
check_cuda_torch

# ── 6. Build MegaSAM CUDA extensions ────────────────────────────────
echo ""
echo "[6/7] Building MegaSAM CUDA extensions (lietorch + droid_backends) ..."
if [ -f "third_party/mega-sam/base/setup.py" ]; then
    cd third_party/mega-sam/base
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" "$PY" setup.py install && \
        echo "       MegaSAM build: OK" || \
        echo "       [WARN] MegaSAM build failed (navigation metrics won't work)"
    cd ../../..
else
    echo "       [SKIP] third_party/mega-sam not found (run: git submodule update --init --recursive)"
fi

# ── 7. Download model weights ───────────────────────────────────────
echo ""
echo "[7/7] Downloading model weights ..."
if [ -f "tools/download_weights.py" ]; then
    "$PY" tools/download_weights.py || echo "       [WARN] Weight download failed. Run manually: python tools/download_weights.py"
else
    echo "       [SKIP] tools/download_weights.py not found"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Installation complete!                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Activate:  conda activate $ENV_NAME"
echo "  Run:       export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  Verify:    python tools/verify_install.py"
echo ""
echo "  If you see 'GLIBCXX not found' errors, make sure LD_LIBRARY_PATH is set."
echo ""
