# Add AutoModel to the existing GLM-5.3 HF/XTuner accuracy test

Set `GLM53_AUTOMODEL_PYTHON` and run the original pytest entry point. The
class-scoped fixture runs a single-GPU AutoModel reference once, waits for that
process to release its GPU, then lets `DeterministicDDPTestCase` launch the
existing XTuner workers. No manual reference-generation command or external
`torchrun` is needed. Unset the variable for the original HF/XTuner comparison.

```bash
GLM_5_3_FLASH_PATH=/absolute/path/to/GLM-5.3-Flash-25B \
GLM53_AUTOMODEL_PYTHON=/absolute/path/to/automodel/bin/python \
XTUNER_TEST_WORLD_SIZE=8 \
/absolute/path/to/xtuner-env/bin/python -m pytest -s -v \
  tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy
```

This runs the original EP1/4/8 parameterization against HF and the same AM
EP1/CP1 reference. It does **not** enable AutoModel EP/CP or add XTuner SP.
For one GPU, use `XTUNER_TEST_WORLD_SIZE=1` and select only:

```text
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy[None-1]
```

## Model-only substitution

AutoModel loads through
`NeMoAutoModelForImageTextToText.from_pretrained(..., force_hf=False)`.
The helper checks that the native NeMo GLM class was loaded. No AutoModel
trainer, recipe, dataset, collator or optimizer is involved. Its dependencies
remain in its own interpreter; neither environment is changed by the test.

The original four text cases and two image cases are extracted into one shared
helper. Images still come from `tests/resource/mscoco_twocat_000000039769.jpg`
(single image and image plus horizontal flip). Every test invocation starts
from these raw cases. AutoModel preprocessing runs in the pytest interpreter
through a CPU tensor pipe, preserving HF's tokenizer, processor and chat
template even when the model environment has a different Transformers version.
No pre-tokenized input fixtures are required or persisted.

Temporary reference results live under pytest's temporary directory and are
rebuilt on every invocation. Before comparing results, workers verify input
hashes (including pixels/grid), labels, sample order, positions and token counts.
The reference includes interpreter/package/backend metadata and sampled logits.

HF's original `output.loss` and XTuner's public `CELossConfig` path remain.
AM computes external FP32 CE with exactly one next-token shift. Visual-token
labels are ignored; these are the existing whole-text labels, not assistant-only
SFT supervision. All three pairs use the original loss checks (mean relative
error <3%, cosine >0.97) and sampled-logits checks (per-case relative L2 <5%,
cosine >0.998, full vocabulary at the last eight supervised positions).
Only eval/no-grad forward is tested: no backward, updates, packing or MTP.

Attention choices remain explicit: HF eager; XTuner Torch DSA/indexer and
FlashAttention vision; AutoModel cuDNN/FlashMLA sparse MLA, native Torch KPool,
FLA KDA and SDPA vision. These are not claims of identical kernels. Production
XTuner TileLang/FlashMLA experiments are outside this existing accuracy test.

## Shared-machine environment

| Component | Validated path / version |
|---|---|
| HF/XT Python | `/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python` |
| HF/XT packages | PyTorch `2.9.1+cu128`, Transformers `5.17.0` |
| AM Python | `/mnt/shared-storage-user/llmrazor-share/zhangxinsen/dev_envs/automodel/bin/python` |
| AM packages | PyTorch `2.10.0+cu130`, Transformers `5.15.1`, FLA `0.4.2` |
| AM source | `/mnt/shared-storage-user/llmrazor-share/zhangxinsen/Automodel` |
| AM commit | `fda1bf266c78e3723599b9441f1c93b94f26380f` |
| Checkpoint | `/mnt/shared-storage-user/llmrazor-share/model/GLM-5.3-Flash-25B` |

`GLM53_AUTOMODEL_PYTHON` may also point to an executable wrapper ending in
`exec /absolute/path/to/automodel/bin/python "$@"` when that environment needs
its own CUDA library paths or source `PYTHONPATH`. Keep these exports inside the
wrapper so CUDA 13 settings do not leak into the HF/XT CUDA 12.8 process.
The shared machine's self-contained wrapper is:
`/mnt/shared-storage-user/llmrazor-share/zhangxinsen/data/xtuner-automodel-loss-compare-data/forward-parity/am-pr-simplified-20261010/am-python`.
It configures the existing AM environment without sourcing or changing it.

Local Transformers 5.15.1 preprocessing did not supply `image_grid_thw` for
these cases. Using the pytest interpreter for preprocessing avoids that version
mismatch without installing packages or substituting an AM recipe template.

## Validation (2026-10-10, reduced checkpoint, H200)

- CPU input/result contract regressions: 7 passed.
- One pytest command on eight GPUs: original EP1/4/8 cases, 3 passed.
  AM ran the six reference cases exactly once before workers started.
- One-GPU invocation selecting `[None-1]`: 1 passed, including automatic AM execution.
- All three pairwise loss/logits checks passed at the existing thresholds.
  Across the eight-GPU suite, maximum sampled relative L2 was 2.5741% HF→XT,
  3.5018% HF→AM and 2.4247% AM→XT; minimum cosine was 0.9993867.
- Ruff, formatting and diff whitespace checks passed. Existing optional-extension,
  deprecation and collective-shutdown warnings were retained.

These results validate the automatic pytest entry point, not AM EP/CP or
training/backward. Earlier distributed experiment artifacts remain separate.

## Packed training diagnostic findings (2026-10-13)

These findings use a real 8K XTuner pack and the four-layer checkpoint. They
are separate from the earlier six-case eval test. Freeze vision/projector,
disable compile, keep FP32 exception parameters, and verify input/weight hashes
before comparing. Exclude padding from text error statistics.

- **Vision:** FA2 varlen on both sides plus `cast_forward_inputs=False` preserves
  FP32 visual RoPE. Visual stages/features then match bitwise. Text adapters must
  affect language instances only: native text/vision share an RMSNorm class.
- **Text RMSNorm:** XTuner's default BF16 `F.rms_norm` differs from HF's explicit
  FP32 statistics/normalization, cast, then affine. On recorded layer-zero input,
  relative L2 is 0.2804%; native AutoModel matches HF bitwise. A GLM-specific
  HF-compatible XTuner path merits validation; replacing AM norm with XT norm
  alone is not a correctness fix. An experiment-only XT norm adapter matches HF
  forward and both input/weight gradients in the CPU regression.
- **Dense MLP:** native AM clamped SwiGLU promotes gate/up to FP32; XTuner and HF
  preserve BF16 and match bitwise on the real input. Sharing BF16 activation
  explains/eliminates this difference, but is an AM experiment adaptation.
- **mHC residual mix:** fused XT vs AM/HF matmul relative L2 is 0.1717% on identical
  inputs. Against FP32 math, errors are 0.1852% XT and 0.2110% AM/HF; this does not
  establish an XT formula bug. Exact matching needs consistent rounding/order.
- **DSA projections:** whole-pack XT vs document/32-token AM `q_a` GEMM differs
  by 0.0130% relative L2. Errors against FP32 GEMM are nearly equal (0.165927%
  and 0.165927%); shape-dependent rounding alone does not justify an XT rewrite.
- **DSA key selection:** original native selection differs on 2.38% of real
  token rows. Forcing identical key sets leaves output relative L2 0.2057%; key
  selection alone is not the full explanation. KPool scoring/tie precision
  remains under investigation.
- **Sparse attention:** with identical Q/KV/key sets, XT vs AM output relative
  L2 is 0.0247%. Sorting XT indices to AM's order makes outputs bitwise equal;
  five sampled rows have nearly identical error against FP32 softmax (0.1662%).
  No attention-formula bug is established. Reusing all XT DSA logic is only a
  counterfactual experiment, not an independent correctness reference; the
  proposed DSA adapter was withdrawn and not committed.
- **Acceptance:** the shared-text experiment at `20a736a65` aligns layer zero
  bitwise, but supervised logits still differ by 6.85% relative L2 / 5.55% top-1.
  It is not a native-text parity result. Backward/full training remains unverified.
  Grad norm records are pre-clipping global L2; a complete training run requires
  every rank to have all expected continuous steps. Historical AM has 6/20
  updates (stopped), XT 20/20; they are not curves for the latest fixes.
