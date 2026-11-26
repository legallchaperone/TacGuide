# Source this before any TacGuide command on the training machine:  source training/tacguide_env.sh
# Keeps the venv, caches, wandb files and weights on the data disk, never in $HOME.
export TG=${TG:-/data2/cwx/TacGuide}
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME=$TG/cache
export UV_CACHE_DIR=$TG/cache/uv
export UV_PYTHON_INSTALL_DIR=$TG/cache/uv-python
export UV_PROJECT_ENVIRONMENT=$TG/venv
export HF_HOME=$TG/cache/hf
export OPENPI_DATA_HOME=$TG/cache/openpi
export MODELSCOPE_CACHE=$TG/cache/modelscope
export WANDB_DIR=$TG/wandb
export WANDB_CACHE_DIR=$TG/cache/wandb
export WANDB_PROJECT=TacGuide
export TORCH_HOME=$TG/cache/torch
export TMPDIR=$TG/cache/tmp
mkdir -p "$TMPDIR"
# The machine has 384 cores. Unlimited intra-op threads during a model load pushed the load above 190
# and every job of the account was killed (2026-11-26). Keep CPU thread pools small unless overridden.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
