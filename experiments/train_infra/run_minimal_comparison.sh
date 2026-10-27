#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} != --locked ]]; then
  exec bash /mnt/shared-storage-user/yushan/asserts/gpu_lock.sh -n bash "$0" --locked
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
raw=/tmp/glm53_minimal_alignment_20261014
persistent=/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/validation/train_infra/minimal_alignment_runs_20261014
analysis_python=/mnt/shared-storage-user/yushan/conda-envs/pt29_glm2_automodel/bin/python
export TRAIN_INFRA_STEPS=5 TRAIN_INFRA_WARMUP=1 TRAIN_INFRA_SP=1 TRAIN_INFRA_COMPILE=0
export TRAIN_INFRA_FREEZE_VISION=1 TRAIN_INFRA_NATIVE_TEXT=1 TRAIN_INFRA_XTUNER_HF_NORM=0
export TRAIN_INFRA_ALIGNED_PRECISION=1 PERMUTE_COMPUTE_DTYPE=fp32
export TRAIN_INFRA_PROFILE_STEP=1000 XTUNER_USE_FA3=0 XTUNER_DETERMINISTIC=false
unset TRAIN_INFRA_AM_HF_GATED_NORM
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
for modality in text image video_frames; do
  cp "$raw/fixtures/$modality.json" "$persistent/"
done
cp "$raw/fixtures/manifest.json" "$persistent/fixtures_manifest.json"
git diff HEAD --binary > "$persistent/code.patch"
cp -r experiments/train_infra "$persistent/source"
for modality in text image video_frames; do
  comparison_out="$raw/$modality"
  mkdir "$comparison_out"
  export TRAIN_INFRA_FIXED_SAMPLE="$raw/fixtures/$modality.pt"
  export TRAIN_INFRA_LOGITS_FIXTURE="$raw/fixtures/$modality.safetensors"
  TRAIN_INFRA_LENGTH="$("$analysis_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pack_length"])' "$raw/fixtures/$modality.json")"
  export TRAIN_INFRA_LENGTH
  env | sort | rg '^(TRAIN_INFRA_|XTUNER_|PERMUTE_COMPUTE_DTYPE)' > "$comparison_out/settings.env"
  export WORK_DIR="$comparison_out/xtuner_e2e"
  echo "START $modality XTuner $(date -u +%FT%TZ)"
  bash experiments/train_infra/run_xtuner.sh --locked > "$comparison_out/xtuner_launch.log" 2>&1
  export TRAIN_INFRA_REFERENCE="$WORK_DIR" WORK_DIR="$comparison_out/automodel_e2e"
  echo "START $modality AutoModel $(date -u +%FT%TZ)"
  bash experiments/train_infra/run_automodel.sh --locked > "$comparison_out/automodel_launch.log" 2>&1
  export PYTHONPATH="$repo_root"
  "$analysis_python" -m experiments.train_infra.analyze_runs --runs "$comparison_out" --out "$persistent/$modality/plots" > "$comparison_out/analysis.log" 2>&1
  "$analysis_python" -m experiments.train_infra.plot_training_logits --runs "$comparison_out" --out "$persistent/$modality/plots" > "$comparison_out/logits_analysis.log" 2>&1
  for backend in xtuner_e2e automodel_e2e; do
    mkdir -p "$persistent/$modality/$backend"
    cp "$comparison_out/$backend/"*.json "$comparison_out/$backend/"*.jsonl "$comparison_out/$backend/"*.txt "$persistent/$modality/$backend/"
    cp "$comparison_out/$backend/training.log" "$persistent/$modality/$backend/"
  done
  echo "COMPLETE $modality $(date -u +%FT%TZ)"
done
