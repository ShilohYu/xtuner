#!/usr/bin/env bash
set -euo pipefail

# 所有训练进程由同一个共享锁保护；退出75表示资源忙，本次未启动训练。
if [[ ${1:-} != '--locked' ]]; then
  exec bash /mnt/shared-storage-user/yushan/asserts/gpu_lock.sh -n bash "$0" --locked
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_usage="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits)"
# 驱动本身可占几 MiB；只把实际计算进程视为占卡。
if [[ -n "$gpu_usage" ]]; then
  printf 'GPU occupied:\n%s\n' "$gpu_usage" >&2
  exit 75
fi
export PYTHONPATH="$repo_root"
export PATH="/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin:$PATH"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_INFRA_LENGTH="${TRAIN_INFRA_LENGTH:-65536}" TRAIN_INFRA_SP="${TRAIN_INFRA_SP:-1}"
export TRAIN_INFRA_COMPILE="${TRAIN_INFRA_COMPILE:-1}" TRAIN_INFRA_STEPS="${TRAIN_INFRA_STEPS:-}"
export WORK_DIR="${WORK_DIR:-/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/validation/train_infra/runs/xtuner_$(date -u +%Y%m%dT%H%M%S)_$$}"
mkdir -p "$(dirname -- "$WORK_DIR")"
mkdir "$WORK_DIR"
export TRITON_CACHE_DIR="$WORK_DIR/triton_cache" TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/inductor_cache"
cp experiments/train_infra/config.py experiments/train_infra/run_xtuner.sh experiments/train_infra/record.py "$WORK_DIR/"
git rev-parse HEAD > "$WORK_DIR/code_sha.txt"
git status --porcelain=v1 > "$WORK_DIR/git_status.txt"
git diff HEAD --binary > "$WORK_DIR/code.patch"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv > "$WORK_DIR/gpus_before.csv"
python -u - <<'PY' > "$WORK_DIR/preflight.log" 2>&1
import json, os, platform
from pathlib import Path
import torch, transformers, xtuner
from xtuner.v1.utils import Config
cfg = Config.fromfile('experiments/train_infra/config.py')['trainer']
out = Path(os.environ['WORK_DIR'])
(out / 'resolved_config.json').write_text(cfg.model_dump_json(indent=2, serialize_as_any=True))
keys = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'NCCL_DEBUG',
        'XTUNER_TORCH_COMPILE', 'TRAIN_INFRA_LENGTH', 'TRAIN_INFRA_SP',
        'TRAIN_INFRA_STEPS', 'TRAIN_INFRA_COMPILE', 'TRAIN_INFRA_DATA', 'PYTHONPATH']
manifest = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'transformers': transformers.__version__, 'xtuner_file': xtuner.__file__,
            'env': {k: os.environ.get(k, '<unset>') for k in keys}, 'cuda_devices': torch.cuda.device_count()}
assert manifest['cuda_devices'] == 8, manifest
(out / 'environment.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY
# 记录持续显存占用，以识别运行中外部任务的资源干扰。
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
  -m experiments.train_infra.record --config experiments/train_infra/config.py \
  2>&1 | tee "$WORK_DIR/training.log"
train_exit=${PIPESTATUS[0]}
set -e
printf '%s\n' "$train_exit" > "$WORK_DIR/exit_code.txt"
exit "$train_exit"
