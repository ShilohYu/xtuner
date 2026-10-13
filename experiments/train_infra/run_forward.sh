#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} != --locked ]]; then
  exec bash /mnt/shared-storage-user/yushan/asserts/gpu_lock.sh bash "$0" --locked "$@"
fi
shift
: "${1:?Usage: run_forward.sh NEW_OUTPUT_DIRECTORY FIXTURE_SAFETENSORS XTUNER_REFERENCE_RUN}"
diagnostic_out="$1"
fixture="$2"
reference="$3"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU occupied; diagnostic not started' >&2
  exit 75
fi
mkdir "$diagnostic_out"
cp experiments/train_infra/{probe_forward.py,vision_attention.py,run_automodel.py,run_forward.sh} "$diagnostic_out/"
git rev-parse HEAD > "$diagnostic_out/experiment_sha.txt"
git diff HEAD --binary > "$diagnostic_out/code.patch"
git -C /mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/automodel rev-parse HEAD > "$diagnostic_out/automodel_sha.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XTUNER_USE_FA3=0 XTUNER_DETERMINISTIC=false PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/automodel:$repo_root"
export TRITON_CACHE_DIR="$diagnostic_out/triton_cache" TORCHINDUCTOR_CACHE_DIR="$diagnostic_out/inductor_cache"
for backend in xtuner automodel; do
  if [[ "$backend" == xtuner ]]; then
    python_bin=/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python
  else
    python_bin=/mnt/shared-storage-user/yushan/conda-envs/pt29_glm2_automodel/bin/python
  fi
  set +e
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m experiments.train_infra.probe_forward --backend "$backend" --fixture "$fixture" \
    --reference "$reference" --out "$diagnostic_out/$backend" > "$diagnostic_out/$backend.log" 2>&1
  result=$?
  set -e
  printf '%s\n' "$result" > "$diagnostic_out/${backend}_exit_code.txt"
  if [[ "$result" != 0 ]]; then exit "$result"; fi
done
