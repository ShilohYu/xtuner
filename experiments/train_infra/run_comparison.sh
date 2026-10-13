#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} != --locked ]]; then
  exec bash /mnt/shared-storage-user/yushan/asserts/gpu_lock.sh -n bash "$0" --locked "$@"
fi
shift
: "${1:?Usage: run_comparison.sh NEW_OUTPUT_DIRECTORY}"
comparison_out="$1"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
mkdir "$comparison_out"
export TRAIN_INFRA_LENGTH=8192 TRAIN_INFRA_STEPS=20 TRAIN_INFRA_SP=1 TRAIN_INFRA_COMPILE=0
export TRAIN_INFRA_FREEZE_VISION=1 TRAIN_INFRA_NATIVE_TEXT=1 TRAIN_INFRA_XTUNER_HF_NORM=1
export TRAIN_INFRA_PROFILE_STEP=1000 XTUNER_USE_FA3=0 XTUNER_DETERMINISTIC=false
export TRAIN_INFRA_LOGITS_FIXTURE=/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/validation/train_infra/forward_alignment_20261013/pack_rank0_step0.safetensors
env | sort | rg '^(TRAIN_INFRA_|XTUNER_)' > "$comparison_out/settings.env"
git rev-parse HEAD > "$comparison_out/experiment_sha.txt"
cp -r experiments/train_infra "$comparison_out/source"
export WORK_DIR="$comparison_out/xtuner_e2e"
set +e
bash experiments/train_infra/run_xtuner.sh --locked > "$comparison_out/xtuner_launch.log" 2>&1
xtuner_exit=$?
set -e
printf '%s\n' "$xtuner_exit" > "$comparison_out/xtuner_exit_code.txt"
automodel_exit=1
if [[ "$xtuner_exit" == 0 ]]; then
  export TRAIN_INFRA_REFERENCE="$WORK_DIR" WORK_DIR="$comparison_out/automodel_e2e"
  set +e
  bash experiments/train_infra/run_automodel.sh --locked > "$comparison_out/automodel_launch.log" 2>&1
  automodel_exit=$?
  set -e
  printf '%s\n' "$automodel_exit" > "$comparison_out/automodel_exit_code.txt"
fi
export PYTHONPATH="$repo_root" OMP_NUM_THREADS=1
analysis_python=/mnt/shared-storage-user/yushan/conda-envs/pt29_glm2_automodel/bin/python
"$analysis_python" -m experiments.train_infra.analyze_runs --runs "$comparison_out" --out "$comparison_out/plots" > "$comparison_out/analysis.log" 2>&1
if [[ "$xtuner_exit" == 0 && "$automodel_exit" == 0 ]]; then
  "$analysis_python" -m experiments.train_infra.plot_training_logits --runs "$comparison_out" --out "$comparison_out/plots" > "$comparison_out/logits_analysis.log" 2>&1
else
  exit 1
fi
