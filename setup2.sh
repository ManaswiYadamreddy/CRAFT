#!/bin/bash
# =============================================================================
# setup.sh — CRAFT/OSDFace environment setup for BU SCC
#
# Run this script on any node (CPU or GPU) to install all dependencies.
# GPU is NOT needed for installation.
#
# Usage:
#   bash setup.sh
#
# After setup, activate with:
#   conda activate /projectnb/cs585/students/hannasam/craft-env
# =============================================================================

set -e   # exit immediately on any error

ENV_DIR="/projectnb/cs585/students/hannasam/craft-env"
PYTHON_VERSION="3.10"

echo "============================================="
echo " CRAFT Environment Setup — BU SCC"
echo "============================================="

# -----------------------------------------------------------------------------
# Step 1: Create conda environment in project directory
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Creating conda environment at ${ENV_DIR}..."

conda create --prefix ${ENV_DIR} python=${PYTHON_VERSION} -y

echo "Environment created. Activating..."
conda activate ${ENV_DIR}

# Confirm activation
echo "Active environment: ${CONDA_PREFIX}"

# -----------------------------------------------------------------------------
# Step 2: Install PyTorch 2.4.0 pinned to CUDA 12.1
# Use +cu121 suffix to prevent pip from resolving to a newer patch version
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Installing PyTorch 2.4.0+cu121..."

pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify torch version
python -c "import torch; print(f'  torch {torch.__version__} installed')"

# -----------------------------------------------------------------------------
# Step 3: Install core pinned dependencies
# These must match exact versions — API changes between versions break the code
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Installing pinned core dependencies..."

pip install \
    diffusers==0.27.2 \
    transformers==4.37.2 \
    accelerate==0.31.0 \
    peft==0.12.0 \
    tokenizers==0.15.2 \
    safetensors \
    huggingface_hub

# -----------------------------------------------------------------------------
# Step 4: Install remaining dependencies unpinned
# These are flexible — latest versions are fine
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Installing remaining dependencies..."

# Image processing
pip install \
    pillow \
    opencv-python \
    imageio \
    tifffile

# Metrics — eval.py
pip install \
    pyiqa \
    basicsr \
    facexlib

# Scientific / ML utilities
pip install \
    numpy \
    scipy \
    omegaconf \
    einops \
    ftfy \
    sentencepiece

# Training utilities
pip install \
    tensorboard \
    tqdm \
    loralib

# Wavelet utilities (wavelet_color_fix.py, common.py)
pip install \
    pytorch-wavelets \
    PyWavelets

# Misc
pip install \
    ptflops \
    numba \
    shapely \
    icecream

# xformers — must be installed LAST and pinned to match torch 2.4.0
# 0.0.28.post1 is the build for torch 2.4.x
# Note: you may see a conflict warning saying it prefers torch==2.4.1
# This is a warning only — it works fine with 2.4.0
echo ""
echo "Installing xformers (pinned to torch 2.4.0 build)..."
pip install xformers==0.0.28.post1 \
    --index-url https://download.pytorch.org/whl/cu121

# Re-pin torch after xformers in case it tried to upgrade it
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 --force-reinstall \
    --no-deps   # --no-deps prevents torchvision from pulling in a newer torch

# -----------------------------------------------------------------------------
# Step 5: Verify all key imports
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Verifying installation..."

python - <<'EOF'
import sys

packages = [
    ("torch",        lambda: __import__("torch").__version__),
    ("torchvision",  lambda: __import__("torchvision").__version__),
    ("diffusers",    lambda: __import__("diffusers").__version__),
    ("transformers", lambda: __import__("transformers").__version__),
    ("peft",         lambda: __import__("peft").__version__),
    ("accelerate",   lambda: __import__("accelerate").__version__),
    ("safetensors",  lambda: __import__("safetensors").__version__),
    ("pyiqa",        lambda: __import__("pyiqa").__version__),
    ("basicsr",      lambda: __import__("basicsr").__version__),
    ("facexlib",     lambda: "ok"),
    ("cv2",          lambda: __import__("cv2").__version__),
    ("PIL",          lambda: __import__("PIL").__version__),
    ("xformers",     lambda: __import__("xformers").__version__),
]

all_ok = True
for name, get_version in packages:
    try:
        version = get_version()
        print(f"  ✓  {name:<20} {version}")
    except Exception as e:
        print(f"  ✗  {name:<20} FAILED — {e}")
        all_ok = False

import torch
print(f"\n  CUDA available: {torch.cuda.is_available()}")
print(f"  torch version:  {torch.__version__}  (expected: 2.4.0+cu121)")

if not all_ok:
    print("\nSome packages failed to import. Check errors above.")
    sys.exit(1)
else:
    print("\nAll imports successful.")
EOF

echo ""
echo "============================================="
echo " Setup complete!"
echo ""
echo " To activate in future sessions:"
echo "   conda activate ${ENV_DIR}"
echo "============================================="