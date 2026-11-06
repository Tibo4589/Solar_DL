#!/bin/bash
# =============================================================================
# Source this at the start of EVERY session on the VSC (login node, interactive
# session or inside a job):      source env.sh
# The first time, it also creates your Python environment (takes ~5-10 minutes).
# =============================================================================

# 1. Software: the same PyTorch module as in the labs (check with your TA that it still is the right one)
module purge
module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl

# 2. The SHARED project folder (data + runs + outputs). ONE path, the same for both of you.
#    It lives in the $VSC_DATA of one of you and is shared through a VSC group (guide, step 0.2).
#    TODO: replace by the real path, e.g. /data/leuven/3xx/vsc3xxxx/solar_project
export SOLAR_ROOT=${SOLAR_ROOT:-/data/leuven/CHANGE/ME/solar_project}

# 3. Caches for downloaded pretrained weights: NOT in $VSC_HOME (its quota is small)
export HF_HOME=$SOLAR_ROOT/cache/huggingface
export TORCH_HOME=$SOLAR_ROOT/cache/torch

# 4. A personal virtual environment ON TOP of the module (reuses its PyTorch), for the extra packages
VENV=$VSC_DATA/venvs/solar
if [ ! -d "$VENV" ]; then
    echo "Creating $VENV (first time only)..."
    python -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"
else
    source "$VENV/bin/activate"
fi

# 5. Group-writable files, so your partner can overwrite/extend what you create in SOLAR_ROOT
umask 002

echo "SOLAR_ROOT=$SOLAR_ROOT"
python -c "import torch; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
