#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != '--locked' ]]; then
  exec bash /mnt/shared-storage-user/yushan/asserts/gpu_lock.sh -n bash "$0" --locked
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${TRAIN_INFRA_REFERENCE:?Set the exact XTuner run directory to replay}"
gpu_usage="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)"
if [[ -n "$gpu_usage" ]]; then
  printf 'GPU occupied:\n%s\n' "$gpu_usage" >&2
  exit 75
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/automodel:$repo_root"
export PATH="/mnt/shared-storage-user/yushan/conda-envs/pt29_glm2_automodel/bin:$PATH"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WORK_DIR="${WORK_DIR:-/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/validation/train_infra/runs/automodel_$(date -u +%Y%m%dT%H%M%S)_$$}"
mkdir -p "$(dirname -- "$WORK_DIR")"
mkdir "$WORK_DIR"
export TRITON_CACHE_DIR="$WORK_DIR/triton_cache" TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/inductor_cache"
cp experiments/train_infra/run_automodel.py experiments/train_infra/replay.py experiments/train_infra/run_automodel.sh "$WORK_DIR/"
git rev-parse HEAD > "$WORK_DIR/experiment_sha.txt"
git -C /mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/automodel rev-parse HEAD > "$WORK_DIR/automodel_sha.txt"
git status --porcelain=v1 > "$WORK_DIR/git_status.txt"
git diff HEAD --binary > "$WORK_DIR/code.patch"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > "$WORK_DIR/gpus_before.csv"
(
  while true; do
    date -u +%FT%TZ
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    sleep 10
  done
) > "$WORK_DIR/gpu_processes.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
set +e
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m experiments.train_infra.run_automodel --reference "$TRAIN_INFRA_REFERENCE" --out "$WORK_DIR" \
  2>&1 | tee "$WORK_DIR/training.log"
train_exit=${PIPESTATUS[0]}
set -e
printf '%s\n' "$train_exit" > "$WORK_DIR/exit_code.txt"
exit "$train_exit"
