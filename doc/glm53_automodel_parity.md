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

## 固定输入逐模块对齐（2026-10-13 恢复执行）

**最新验收：显式精度配置下，四层 8K 的实际八卡前向已对齐。7384 个真实 token 的四层 hidden states、final norm 及全部 18 个文本抓点逐位一致；1261×154880 监督 logits 数值完全一致，relL2、max abs、top-1 分歧均为 0。logits 保存 dtype 不同（XT FP32 / AM BF16），因此只称数值一致。**

XT 普通 RMSNorm 的 GLM 精度修复已落在工作树并通过回归/生产等价验收。其余对齐使用显式配置：保留双方原生 KDA/DSA attention 与 AM expert GEMM，统一已定位的精度边界、投影形状、索引 dtype、排序及合并原语。原生默认配置及完整训练尚未对齐；当前结果覆盖固定 checkpoint/pack、四层、EP1/SP1、train/no_grad。此前各阶段误差保留在后文作为因果证据。

本文层号从 0 开始：layer0 是第一层（KDA + dense MLP），layer1 是第二层（DSA + MoE）。
总结和过程证据已同步至[飞书最新精度定位章节](https://aicarrier.feishu.cn/wiki/EPJ5wY8IHi8LOvki3KycNx5En5e#doxcnfAvAjAyCYsWF23WguU4G3e)，按结论、三实现表格、因果干预、后续验收组织。
本节替代此前未完成的 `layer0_modules_latest_20261013` 探针记录。共享结果根目录为
`/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/validation/train_infra/layer0_modules_resumed_20261013/`。

### 版本、输入与独立参考

- XT 基线 HEAD `49af42b54f5c92f6ec25bf649212f14212b55f67`；实时远端
  main `ac2d64993163d18edb0e1777cc10a95985e2d516`、F6
  `b5816bd1d100fd8da19abcaa6a3a8cf112bf356d`、FP32 remesh
  `561425e0cf56561f18fcb57cc69f1c7f9fc4f05f` 均已包含，没有新上游提交需合并。
- AutoModel `eb11d83ece9952b12e2f770df9fc524c1cbfe6fb`；HF Transformers
  `5.17.0`，两侧运行环境的 PyTorch 均为 `2.9.1+cu128`。
- 四层 checkpoint 的 `/tmp` 原件已丢失；恢复引用张鑫森
  `gradnorm-repro-20261010-104322/checkpoint` 的持久化副本，**12 个分片全部重新核对
  原始 SHA256**。配置、tokenizer、模板也通过原始 hash，记录见 `recovery.json`。
- 用 XT 训练日志 `xtuner_20261013T045424_1102069/inputs_rank0.jsonl` 的第 0 步源偏移
  重建完整 8192-token pack，input_ids / labels / cu_seqlens / pixel_values /
  image_grid_thw / mm_token_type_ids 六项 hash 全部匹配。含 7384 个真实 token、
  808 padding、22976 图像 patches、1261 个监督位置。
- 重新执行两侧 8 卡前向：同源权重、同 pack，冻结视觉/projector，train/no_grad，
  compile=False，EP/SP1，0 optimizer step。视觉全部抓点及拼接输入逐位一致；
  观测 hook 不改变 logits。独立单卡 layer0 回放逐位复现两侧整网 layer0 抓点。
- **参考确为 HF 原生 Torch**：先前失败源于 HF 装饰器自动分派至 causal-conv/FLA。
  现在 `inspect.unwrap` 并校验函数代码文件；卷积和递推均按原始 cu_seqlens 隔离文档，
  core 按 8 个独立 head 限制临时内存，并与未分组的前 128 token 验证逐位一致。
  HF core 自己保留 FP32 q/k norm、BF16 beta 等语义，不拿 XT/FLA 替代 HF oracle。
- XT 对照基线保留此前 HF 普通 norm 适配，**原生 norm 另做同输入与整层干预**。
  新生产修正和该适配的等价性另经 8 卡实测验证。普通 norm 与 gated norm 是不同模块。
- upstream FP32 gate/mHC 分块投影是 **opt-in**；本次实际两个 block size 均为 **0**。
  先前交接中“合并后自动转 FP32 投影”的描述不成立。

### 三实现对照表

下表为同一真实模块输入、相同权重、排除 padding 的相对 L2；数值均为百分比。
XT–HF、AM–HF 各以 HF 输出为分母。它们是局部回放，不能相加或外推成整网收益。
完整 32 行数值、dtype、最大绝对误差、相等比例在 `comparison_fp64stats.json`。
误差范数统一 FP64 归约，模型计算仍保持原 dtype；整网图表见 `forward_fp64stats/`。

| 模块 | XTuner | AutoModel | HF 原生 | XT–HF | AM–HF | 判断 |
|---|---|---|---|---:|---:|---|
| mHC pre / collapse | FP32 F.rms_norm 后投影、Sinkhorn | 显式 FP32 norm 后同构计算 | 同 AM | 0.001421% | 0% | 最早极小首差；未证实公式错误 |
| 普通 input RMSNorm | 原先 BF16 F.rms_norm；本次修成 HF 精度边界 | FP32 norm → BF16 → affine | 同 AM | 原先 0.280411%；修后 0% | 0% | 已修 XT 的 GLM 文本路径 |
| q/k/v/b 线性投影 | BF16 GEMM | BF16 GEMM | BF16 GEMM | 0% | 0% | 全部逐位一致 |
| q/k/v 短卷积 | FLA/custom op | FLA | Torch conv1d + SiLU | 0.00106–0.00174% | 同 XT | XT–AM 逐位一致；极小后端舍入 |
| forget gate | FP32 fused gate | 同源 FLA gate | FP32 Torch sigmoid/exp | 0.000006% | 同 XT | XT–AM 逐位一致 |
| beta sigmoid | FP32 sigmoid | FP32 sigmoid | BF16 输出 | 0.171795% | 同 XT | HF 的低精度边界，不能据此要求 XT 降精度 |
| output gate / o_proj | BF16 GEMM | BF16 GEMM | BF16 GEMM | 0% | 0% | 逐位一致；FP32 gate 未启用 |
| KDA core | FLA 派生，固定 gate 扫描布局 | FLA chunk | FP32 Torch chunk 递推 | 0.383262% | 0.383520% | XT–AM 0.032565%，仍需区分内核舍入 |
| gated RMSNorm | 融合 FP32 norm / affine / sigmoid，末次 cast | norm 和 sigmoid 分别提前舍入 BF16 | 全程 FP32，末次 cast | 0.001667% | 0.379792% | AM 是此处主要偏差；HF 适配后 AM–HF 为 0 |
| mHC attention post | 融合 FP32 累加后 cast | BF16 乘法和 matmul 的分步舍入 | 同 AM | 0.171697% | 0% | 精度/顺序差异，未证实 XT 公式错误 |
| dense MLP | BF16 clamped SwiGLU | gate/up 升 FP32 算激活再 cast | BF16 clamped SwiGLU | 0% | 0.261853% | XT 与 HF 逐位一致；不应反向改 XT |
| mHC FFN post | 同上 | 同上 | 同上 | 0.138018% | 0% | 放大与抵消依赖真实输入 |
| 完整 KDA，同一入口 | XT 原生 KDA | AM 原生 KDA | HF 原生 packed KDA | 0.500992% | 0.616646% | XT–AM 0.453257% |
| 完整 layer0，同一残差入口 | XT + HF 普通 norm | AM 原生 | HF 原生 packed layer0 | 0.409201% | 0.437034% | XT–AM 0.404925% |

实现入口：XT `xtuner/v1/module/decoder_layer/mhc.py`、`module/attention/kda.py`、
`ops/kda/chunk_kda.py`、`model/moe/glm53/glm53.py`；AM
`nemo_automodel/components/models/glm5_next/layers.py`（普通 norm、mHC、gated norm、KDA）及
`components/moe/layers.py`（clamped MLP）；HF `transformers/models/glm5_next/modeling_glm5_next.py`。
HF 函数真实代码位置同时保存在 JSON 的 `hf_eager_functions` 字段。

### 因果干预与对齐顺序

1. **XT 普通 RMSNorm：已落地修正。** GLM 的 `_configure_model_specific_layers` 在编译/
   FSDP 包装前为文本 RMSNorm 设置 FP32 统计与归一化、转回输入 dtype、再乘权重的函数。
   覆盖 dense/MoE、DSA q_a/kv_a、最终 norm 和 MTP；权重布局不变，保留 DTensor 处理，
   不修改视觉、其他模型或 gated norm。原生普通 norm 的整层 XT–HF 为 0.438985%，
   HF 精度流程为 0.409201%；仅修普通 norm 不能令整层逐位一致。
2. **AM gated RMSNorm 是下一项最明确的精度边界修正。** 只替换 AM 的 o_norm 为实际 HF
   模块，整层 XT–AM 从 0.404925% 降至 0.312679%（约 23%）；AM–HF 从 0.437034%
   降至 0.394379%。仅改普通 norm 无变化，单改 AM MLP 只降至 0.400508%。
   已有 opt-in AM 实验适配及前后向回归；当前没有改 AM checkout。此前此适配的 20 步
   实验没有持续的末步收益，不能用局部改善承诺训练/logits 同比例改善。
3. **mHC 与 KDA 采用参考精度模式作诊断，不盲目撤销稳定内核。** 仅将 XT mHC post
   换成 HF 表达式，XT–HF 整层降至 0.389490%；pre 换成 HF 并无净改善。仅开启
   FP32 gate block64 得到 0.418227%，也不能当作当前配置的对齐修复。
4. **KDA 已排除“只是 q/k norm cast”这一单一解释。** 相同核心输入下，用 FLA 的实际
   l2norm 输出喂 HF 且禁用其内部归一化，XT–HF 从 0.383262% 仅降至 0.347862%。
   第一文档前 128 token 对独立 FP64 递推：HF 原生 0.163832%，HF 使用 FLA q/k
   0.245900%，XT 0.375306%，AM 0.375291%。这说明 HF 原生参考在该子样本更接近
   FP64，也说明 XT/AM 很接近；不足以判为 XT 独有公式 bug。见 `kda_qk_boundary.json`。
   历史同输入扫描替换已将 XT/AM core 对齐；固定扫描布局还承担 SP 稳定性，替换前须验证
   跨 SP/反向。下一步拆解 chunk 内部舍入，并用双侧实际输入复核关键边界。
5. **整网继续追 layer1 DSA/KPool→MoE。** 本轮真实 logits 仍为 10.1316% relL2、
   9.5956% top-1 分歧，CE token mean XT=13.4222965、AM=13.4179258。
   历史固定输入已证明同输入路由专家集合相同，而上游约 0.61% 扰动使 9.40% 真实 token
   换专家，是 layer1 的主要放大源。KPool 选 key 和 sparse attention 的归约顺序应分开
   测；不能强制相同路由来宣称修复，不能因 HF 与 XT 不逐位相等就重写其数学公式。

### 本轮修正验收与边界

统计纠正：此前 FP32 CPU norm 在大张量归约中产生偏差，旧整网 logits relL2 为
10.2694%，同一份输出使用 FP64 归约为 **10.1316%**；这不是模型改善。
layer0 CPU 旧统计 0.405899% 修正为 0.404925%，与 GPU 模块回放的 FP64 统计一致。
旧统计文件保留；新增统计回归覆盖 FP32 norm 溢出而真实相对误差有限的情形（2 passed）。

- CPU：普通 norm 的 FP32/FP16/BF16 前向、输入梯度及权重梯度与实际 HF 逐位相同；
  GLM builder 覆盖范围通过（4 passed，3 CUDA cases 在 CPU 环境跳过）。
- 持 GPU 锁：上述三 dtype 的 CPU/GPU 测试、GLM builder、合成 HF checkpoint 公共加载
  路径、upstream 投影完整回归、HF 原生 packed oracle，共 **16 passed**；没有 GPU skip。
  命令和 warning 在 `gpu_regressions.log`，不将合成 checkpoint 说成真实大模型验收。
- 真实四层 checkpoint / 8 卡：**关闭实验 HF norm hook，使用本次生产修正，58 个全部抓点
  （含 1261×154880 全监督 logits）与此前适配输出逐位一致**，元数据确认 hook 数为 0。
  见 `production_norm_acceptance.json`。视觉仍逐位对齐。
- 本轮没有重跑 20 步训练，也没有把 forward/no_grad 当作逐参数梯度或 optimizer 对齐。
  compile=True、64K、未冻结视觉训练仍未由本轮验证。历史 20 步两侧完整运行的证据保留在
  `validation/train_infra/logits_every_step_20261013/`，不代表新代码训练验收。
- 旧 `/tmp` dump 已丢失，恢复目录内只有重新生成的 pack，不能声称历史大文件仍存在。
  新整网 dump：`/tmp/glm53_forward_resumed_20261013`；新模块 dump：
  `/tmp/glm53_layer0_modules_fp64stats_20261013`（保留旧统计 run）；无 hook 修正验收 dump：
  `/tmp/glm53_normfix_forward_20261013`。它们是临时文件；持久化 JSON、日志、图表、
  恢复来源及脚本在本节开头的共享结果目录。

复跑逐模块诊断（需先恢复 pack/整网抓点，输出目录必须不存在）：

```bash
GPU_LOCK_DEVICES=0 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
PYTHONPATH="$PWD:/mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/automodel" \
bash /mnt/shared-storage-user/yushan/asserts/gpu_lock.sh -n \
/mnt/shared-storage-user/yushan/conda-envs/pt29_glm2_automodel/bin/python \
-m experiments.train_infra.probe_layer0_modules \
--source /tmp/glm53_forward_resumed_20261013 --out /tmp/glm53_layer0_fresh
```

## 第二层（layer1）DSA：固定输入与因果验证

2026-10-13 再核对 upstream main，仍为 `ac2d6499`，已包含于当前 HEAD。
同一 checkpoint、同一 8192-token pack，分别用 XT 和 AM 的 layer1 input norm 抓点回放，
两侧 DSA 均逐位复现各自整网输出。该探针为 train/no_grad/SP1，不覆盖 backward 或 CP/SP。
这里的 HF 是 `Glm5NextTextAttention` 的原生 eager 路径，按文档隔离，不替换其 attention。
以下误差均排除 808 个 padding，使用 7384 个真实 token 和 FP64 范数归约。

### 三实现比较

| 环节 | XTuner | AutoModel | HF 原生 | 实测 |
|---|---|---|---|---|
| query 投影 | 整 pack GEMM | 每文档内 32-token GEMM | 每文档 GEMM | 固定 `q_a_proj` 输入时 XT–AM 为 0.012902%；相同输入的 `q_b_proj` 为 0 |
| compressed KV | 整 pack 投影、norm | 每文档投影、norm | 投影、norm 后展开 K/V | XT–AM 的 KV 投影和 norm 均逐位一致 |
| KPool 选 key | TileLang score + Torch topk | 按 32-token query 分块打分 | Torch score + topk | 原生 XT–AM 有 163/7384 行 key 集合不同；XT–HF 仅 1 行 |
| attention 核心 | absorbed MLA，FlashMLA | absorbed MLA，包装层排序 key 后 FlashMLA | 展开 K/V，BF16 attention matmul、FP32 softmax 后转 BF16 | 固定 q/kv/key 集合时 XT–AM 为 0.024745%；统一 key 顺序后逐位一致 |
| value 展开 / o_proj | 整 pack | 每文档 | attention 输出后 o_proj | 固定各自模块输入，XT–AM 均逐位一致 |
| 完整 DSA，同一 XT 输入 | 原生 XT（含已修普通 norm） | 原生 AM | 原生 HF eager | XT–AM 0.207635%；XT–HF 0.336725%；AM–HF 0.339230% |
| 完整 DSA，同一 AM 输入 | 同上 | 同上 | 此入口未重跑 HF | XT–AM 0.208284% |

完整 DSA 的 XT–AM 以 XT 输出为分母，XT–HF/AM–HF 以 HF 为分母。
`projection_geometry.*` 是从同一 DSA 入口向下传播后的中间结果，不是每个子模块都固定了输入：
例如 q_b 输出有 0.047245% 差异，但单独固定 q_b 输入后逐位一致。
query absorption 累积差异为 0.083902%，固定输入后也为 0；不能把这两项误判为额外算子错误。

### 单变量与组合干预

| 诊断干预 | DSA 输出相对误差 | 结论 |
|---|---:|---|
| AM 仅将 query chunk 从 32 改为整文档 | 对 XT 0.115430% | GEMM 分块是可测原因之一；此项也会影响依赖 query 的 indexer |
| XT 仅将 TileLang indexer 改为 Torch | 对 AM 0.207632% | 仅 1 行 key 改变；Torch XT 与 HF 的 key 集合全部一致 |
| XT 仅固定 AM key 集合 | 对 AM 0.204304% | 原生入口下，key 集合不是主要独立来源 |
| XT 仅固定 HF key 集合 | 对 HF 0.336759% | 不能用 key 选择解释 absorbed 与 HF eager 的整体差异 |
| XT 仅使用 AM 的 q_a 投影结果，其余仍走 XT | 对 AM 0.116085% | q_a 的分块首差影响 query 和 indexer 两条路径 |
| 上一项再固定 AM key 集合及顺序 | 对 AM **0，逐位一致** | 固定输入下的 DSA 剩余差异已闭环，不需要替换整个注意力模块 |

最后一行是**诊断干预，不是原生模型已对齐**；不把强制 key 选择写进生产实现。
各项不可相加：投影会影响选 key，因此改变 q_a 后的 indexer 差异与原生入口下并不相同。
另将相同 q/kv/key 集合按 AM 的每文档方式启动 sparse kernel，结果与整 pack 逐位一致；
不能把剩余误差笼统归因于所有 per-document 计算。

当前证据不支持修改 XT DSA 的数学公式。对齐优先级为：先处理 layer0 已定位的精度边界，
再评估两侧统一 query 投影分块和 key 顺序的参考模式，同时验证吞吐、backward 与 CP/SP；
继续检查 layer1 MoE 的上游扰动放大。原生全网 logits 仍以此前 10.1316% relL2 为准，
本次没有新跑整网训练，不能用局部干预的 0 误差声称训练已对齐。

### 复现与产物

- 脚本：`experiments/train_infra/probe_layer1_dsa.py`；内含权重加载、原生回放、文档范围、
  相同 sparse 输入排序、最终组合干预的断言。Ruff 和 diff whitespace 检查通过。
- 持久化小产物：`validation/train_infra/layer1_dsa_resumed_20261013/`（workspace 根目录下）。
  最终结果为 `interventions_comparison.json`、`interventions_run.log`、`manifest.json` 和脚本副本；
  初次结果和投影分解结果保留，不能把失败的 `stages_run.log` 当成完成记录。
- 最新原始大张量：`/tmp/glm53_layer1_dsa_interventions_20261013`，pod 重建后可能丢失。
- 采集修正：`cu_seqlens` 为 padding 也分配了文档；AM 实际先计算 8192 行再 mask。
  探针按 `packed.doc_ids > 0` 对齐中间输出，统计按 padding mask 筛至 7384 行。
  原失败是探针对中间结果行数的假设错误，未改 AM 的 padding 行为。

复跑命令沿用上一节环境与 GPU lock，将 module 改为
`experiments.train_infra.probe_layer1_dsa`，并指定不存在的 `--out /tmp/glm53_layer1_dsa_fresh`。

## KDA chunk 内部精度：128-token 子样本干预

范围仅为第一文档前 128 token、全部 64 heads、真实 layer0 core 输入；不能替代 8K 或 backward 验收。
`probe_kda_intra_precision.py` 直接组合 FLA 的 intra、recurrent、output 阶段，逐位复现 XT 原生输出。
固定 FLA 已归一化 q/k 与 XT gate prefix，独立计算 FP32 chunk 因子；其 FP32 输出对同一已舍入 q/k
的 FP64 逐 token 递推误差为 0.00001677%，验证因子分解正确。

下表则统一以**未提前舍入 q/k 归一化的 FP64 递推**为参考，数值均为百分比：

| 计算路径 / 干预 | 相对 L2 |
|---|---:|
| XT 原生 | 0.375306% |
| HF 原生 | 0.163832% |
| 仅 u 中 v×beta 不提前 BF16，仍使用 native Akk | 0.359388% |
| 仅 w 中 k×beta×decay 不提前 BF16，仍使用 native Akk | 0.383699% |
| u/w 同时取消中间乘法 cast | 0.373114% |
| FP32 求 Akk 的逆后转 BF16，余下 native | 0.345486% |
| FP32 计算 Aqk 后转 BF16，余下 native | 0.366000% |
| 所有 intra 因子 FP32 计算后转 BF16，recurrent/output 仍 native | 0.288411% |
| FP32 chunk 参考，保留 FLA 已舍入 q/k | 0.245898% |

**多处精度边界共同贡献残差；单点提高精度不保证净收益。** 全部 intra 参考因子在该子样本有改善，
但与 HF 仍不同，且使用 Torch 重建，不能当作生产修复。保留原乘法 BF16 边界的 Torch 控制组
与 XT 最终输出差 0.001907%；这项后端残差已单列，不将干预结果全部归因于一次 cast。
该子样本阶段尚未验证完整 pack、梯度、吞吐或 SP 稳定性；完整 pack 的后续结果见下节。

持久化结果：`validation/train_infra/layer0_modules_resumed_20261013/kda_intra_precision_valid.json`
及 `.log`、脚本副本。早期 `kda_intra_precision.json` 的 Aqk 阶段指标未屏蔽未初始化上三角，
含 NaN，不能作为有效完整报告；输出指标本身有限。最终脚本只统计内核实际使用的下三角，
并拒绝序列化非有限值。原生回放、独立递推和输出有限性自检均通过。

### 扩展至完整 8K：提高单侧精度不等于两框架对齐

随后使用 `probe_kda_packed_precision.py` 覆盖全部 8192 个位置、7 个文档段、64 heads，
统计排除 padding，仅包含 7384 个真实 token；文档间重置状态，验证非整 64-token 尾部。
原生分阶段输出逐位复现 XT；XT/AM 对 HF 的两组基线数值与此前 `kda_qk_boundary.json`
精确一致。公共因子计算提取后，128-token 回归的所有已记录指标也精确不变。

| KDA core 路径 | 对 HF 原生 | 对 FP64 递推 | 对 AM 原生 |
|---|---:|---:|---:|
| XT 原生 | 0.383262% | 0.342654% | 0.032565% |
| AM 原生 FLA | 0.383520% | 0.342763% | 0% |
| HF 原生 | 0% | 0.181335% | 0.383782% |
| XT 全部 intra 因子 FP32 计算后存 BF16，其余 native | 0.335255% | 0.293252% | 0.344296% |

**这项 XT 单边改动不作为第一层对齐修复。** 它提高了对高精度参考的接近程度，但扩大了与
原生 AM 的差异；两侧 core 原本非常接近。后续优先统一已明确的 gated norm 等框架间边界，
并在整层验证；不能将局部 FP64 改善称为 XT–AM 第一层或整网的改善。

重建 FP32 因子对固定已舍入 q/k 的 FP64 递推误差为 0.00001537%；保留原 BF16 乘法边界
的 Torch 控制组与 XT 输出差 0.011593%，后端残差单列。最终指标有限，自检通过。
结果位于 `layer0_modules_resumed_20261013/kda_packed_threeway.json`、`.log`、
`kda_packed_manifest.json`，包含全部单变量干预。仅覆盖 KDA core 前向，未改生产内核，
未完成整层或 backward 验收；第一层原生误差仍为 0.404925%。


## 第一层累计精度配置：保留双方原生 KDA 核心

**结论：诊断配置的第一层前向已逐位一致；原生默认配置尚未对齐。**
同一真实 pack 的 7384 个有效 token、同一权重与层入口，按下表累计统一精度边界。
普通 RMSNorm 沿用已通过生产等价验收的 HF 精度流程；不是将整段 XT KDA 换成 FLA。

| 累计配置 | 整层 XT–AM relL2 |
|---|---:|
| 修正普通 RMSNorm 后的基线 | 0.404925% |
| AM gated norm 使用 FP32 计算，末次 cast | 0.312679% |
| 再统一 AM dense MLP 为 HF BF16 SwiGLU | 0.303551% |
| 再统一 XT mHC pre 的显式 FP32 normalization | 0.302931% |
| 再统一 XT mHC post 为 HF 原生分步表达式 | 0.177469% |
| 再统一 XT gated norm 的 FP32 表达式 | 0.177411% |
| 最后 AM gate prefix 使用 XT 固定扫描布局 | **0%，逐位一致** |

实验断言两侧 KDA 原生入口未变、所有 Parameter 对象身份未变。最后一步仅替换
AM FLA forward 的 `chunk_local_cumsum`，保留 XT 的 SP 稳定扫描策略；没有替换双方
KDA 整体模块或核心。各项是累计干预，不能将相邻差值解释为独立可加的误差贡献。

此配置两侧对独立 HF 原生整层的 relL2 均为 **0.389464%**；框架间对齐不意味着与
HF 的不同 KDA 数值路径也逐位一致。当前结果为 SP1、8K 真实 pack 的精度诊断，
未改变生产默认配置，也不能自动推导整网 logits 或训练对齐。

前向证据：`validation/train_infra/layer0_modules_resumed_20261013/cumulative_boundaries_comparison.json`
及 `cumulative_boundaries.log`。脚本为 `experiments/train_infra/probe_layer0_modules.py`。

### 第一层反向验证：前向一致尚未延伸到所有梯度

使用相同 BF16 随机输出余切（seed=20261013，scale=1e-3，padding 置零），
对完整第一层分别求输入及全部 26 个可训练参数的 VJP；开启梯度后的前向也断言逐位一致。
**所有梯度有限，20/26 个参数梯度逐位一致；仍有 6 个参数及输入梯度存在差异。**

| 梯度 | XT–AM relL2（XT 为分母） | 最大绝对误差 |
|---|---:|---:|
| 输入 | 0.00007684% | 3.81470e-6 |
| KDA dt_bias | 0.162214% | 2.40365e-5 |
| mHC attention fn | 0.00225444% | 4.76837e-7 |
| beta b_proj.weight | 0.00132636% | 2.38419e-7 |
| mHC attention base | 0.00008971% | 1.03610e-8 |
| mHC attention scale | 0.00003148% | 7.45058e-8 |
| KDA A_log | 0.00000794% | 1.86265e-9 |
| 其余 20 个参数 | 0%，逐位一致 | 0 |

最大剩余项是 gate 的 dt_bias 反向。这里只测一次确定性余切，不包含 optimizer step、
整网训练、64K、SP>1 或 compile=True；不能据此宣布训练对齐。结果见
`cumulative_backward_comparison.json`、`cumulative_backward_fixed.log`。首次反向尝试
因局部 `args` 列表遮蔽 argparse Namespace 而提前退出，已改名修正并完整重跑，失败日志保留。

### gate 反向单项干预：dt_bias 与 A_log 已逐位一致

源码给出了可检验的原因：FLA `kda_gate_bwd` 先把逐 token `dg` cast 到 gate 输入 dtype，
再对 token 求和得到 `dt_bias` 梯度。AM 传 BF16 gate，XT 已通过 `g.float()` 保留 FP32
求和。仅将 AM fused gate 输入转为 FP32，原生 FLA kernel 保留，前向仍逐位一致。

完整第一层实测：`dt_bias` 梯度误差由 **0.162214% 降至 0**，`A_log` 也变为逐位一致；
**22/26 个参数梯度逐位一致**。剩余为 attention mHC 的 fn/base/scale 和 beta
`b_proj.weight`；最大参数 relL2 为 mHC fn 的 **0.00225444%**，输入梯度仍为
**0.00007684%**。两次运行原累计配置的全部梯度指标精确复现。

该干预支持 AM gate 梯度提前 BF16 舍入的定位，不需要撤销 XT 的 FP32 gate 修正。
尚未定位的微小残差不能仅凭量级判为可忽略；当前仍不是完整反向逐位验收。
证据为 `cumulative_backward_gate_comparison.json` 和 `cumulative_backward_gate.log`。

### 暂停前新增证据：反向残差起点已定位到 KDA 的 db 出口

完整第一层再次重跑，原累计配置及 AM FP32 gate 干预的全部梯度指标精确复现。
有效 token 上，KDA backward 的 q/k/v/g/beta/Aqk/Akk/do 输入逐位一致；Aqk/Akk
仅比较内核使用的下三角，未初始化的上三角已屏蔽。dq/dk/dv/dg 出口逐位一致，
仅 db（beta 梯度）出口出现 relL2 **0.00000305084%**、max abs **1.16415e-10** 的差异。
beta 投影的输入和前向输出逐位一致，反向输出梯度 relL2 为 **0.000189632%**。
因此原 4 个参数及输入梯度残差的首个已观测位置是 KDA db 出口，尚不能归因于投影本身。

进一步观测：XT 向公共 FLA backward 传入 dht=None，AM 传入全零 dht 张量
（AM 请求返回最终状态但模型不使用该输出）。这是待验证的候选原因，尚未做移除全零
dht 的单项实验，不能先宣称它已解释残差。下一步只改变该边界，保留原生核心、
相同梯度余切及前向逐位断言，比较 db、所有参数和输入梯度。

证据：`backward_stages_comparison.json`、`backward_stages_valid.log`。首次阶段抓点
比较误触未初始化上三角，被 finite 断言终止；修正统计掩码后完整重跑成功。
`backward_stages.log` 和原失败目录保留，不作为完成验收。本小节尚未同步飞书。


### 第一层反向闭环：索引 dtype → autotune 布局 → beta 归约

**结论：统一精度与索引 dtype 的诊断配置，第一层前向、输入梯度及全部 26 个参数梯度
均逐位一致；两个独立输出余切 seed=20261013、20261014 均通过。原生默认配置未改变。**

沿着残差逐步向内定位，得到以下因果链；所有输出误差仍使用 FP64 范数。

| 检查或干预 | 证据与结果 |
|---|---|
| 将 AM 全零 dht 改成 None | 全部梯度指标完全不变，排除该候选原因 |
| 比较 KDA backward 及 intra 输入 | q/k/v/g/beta/Aqk/Akk/do、dAqk/dAkk 有效元素全部逐位一致 |
| 分解 beta 梯度归约 | db_base 逐位一致；db2 relL2 0.00000821491%；最终 db relL2 0.00000305084% |
| 核对元数据 | cu_seqlens 数值相同；XT 的 cu_seqlens/chunk_indices 为 int32，AM 为 int64 |
| 核对实际 Triton autotune 配置 | XT：4 warps、3 stages；AM：1 warp、3 stages |
| 仅将 AM backward 的两项索引转为 int32 | 断言转换无损，AM 也选择 4 warps、3 stages；db2/db 全部逐位一致 |
| 整层验证 | 原前向逐位一致保持；输入和 26/26 参数梯度全部逐位一致，所有值有限 |
| 第二组余切复核 | seed=20261014 重跑完整流程，同样全部逐位一致 |

这不是索引数值或公式错误，而是整数 dtype 进入 Triton 特化/自动调优后选择了不同
浮点归约布局。微小 FP32 db 差异在后续 BF16 cast 时跨越舍入边界，传播到 beta 投影和
attention mHC；单项索引干预使整条链上的残差同时消失。没有用对方梯度覆盖结果，
也没有替换 KDA core。元数据转换只在本次 AM backward 诊断入口生效，保持真实
最终状态梯度行为（不依赖已排除的 dht=None 干预）。

正式落地策略：普通 RMSNorm 的 GLM 精度 bug 已修；保留 XT FP32 gate 梯度求和及
SP 稳定扫描。统一比较双方的 metadata dtype 应作为数值对齐契约的一部分，但不能
把当前 SP1 的入口适配直接当作 CP/SP 多卡验收。mHC 与 gated norm 融合策略的统一
仍是显式精度配置，不因为诊断逐位一致就自动改动生产默认内核。

验证范围：同一真实 8K pack、7384 有效 token、固定 checkpoint、两组 BF16 余切
（scale=1e-3，padding 置零）；只做 VJP，没有 optimizer step、整网训练、64K、
SP 多卡或 compile=True 验收。原生第一层仍为 0.404925%，整网监督 logits 为 10.1316%。

持久化证据：`backward_no_dht_comparison.json`、`backward_beta_reduction_comparison.json`、
`backward_intra_launch_comparison.json`、`backward_int32_indices_comparison.json`、
`backward_int32_seed2_comparison.json`，对应日志、各次运行的原始探针源码及 SHA256
均收录于 `cumulative_manifest.json`。恢复后再次核对 upstream main 仍为 ac2d6499。


## 可复用配置：AM 向 XT 当前精度流程对齐，保留原生 attention core

新增 `experiments/train_infra/precision_profile.py` 的显式上下文
`automodel_native_core_precision(text)`。它保留双方原生 KDA/DSA 核心，以及 AM
模型类、权重加载、参数和 routed experts；只统一已定位的 mHC/gated norm/SwiGLU
计算步骤、gate FP32 输入、固定 gate prefix 和 int32 packed metadata。
退出上下文会恢复所有替换的方法；单层实测恢复后的 AM 输出逐位复现原始基线。

**该配置保持 XT 当前前向逐位不变；AM 第一层前向、输入和全部 26 个参数梯度
逐位一致（seed=20261013）。** 这是对 XT 当前精度流程的适配，区别于上一节为诊断
而将 XT mHC 等表达式改成 HF 形式的累计配置。普通 RMSNorm 使用已通过生产等价
验收的 HF 精度流程；没有恢复旧的 XT 普通 norm。

新增的反向发现：第一次可复用配置前向已逐位一致，但输入梯度 relL2 为 0.434150%，
部分注意力参数梯度约 0.4–0.7%。定位到 mHC 共享 view 的反向合流位置：XT 同时将
flattened view 用于 norm 分支和加权 collapse；AM 原写法将 collapse 直接连接到原
streams。前向值相同，BF16 梯度合流顺序不同。让 AM 两个分支复用同一个 flattened
view 后，输入及所有参数梯度逐位一致。没有替换梯度数值；该变化只统一计算图结构。

实际八卡 FSDP 前向新增 `probe_forward.py --backend automodel --native-text
--native-core-precision` 入口。上下文安装时检查参数身份；FSDP 在执行中会按需替换
Parameter 对象，因此结束时检查名称/形状/可训练状态，非 FSDP 仍检查对象身份。
首次完整前向虽已产出 logits，但退出身份断言失败，保留为未通过记录；修正该运行时
检查后完整重跑退出码为 0。输入、视觉权重、未观测/抓点前向等既有检查均保留。

证据：`xt_native_profile_comparison.json`（只通过前向的初版）、
`xt_native_profile_views_comparison.json`（完整单层通过）、
`native_core_precision_manifest.json`、对应日志和源码副本。整网结果见随后补充。


### 可复用配置的实际八卡整网结果

**第一层已在真实八卡 FSDP 前向中逐位一致；最终 logits relL2 从 10.1316% 降到
6.4831%，top-1 分歧从 9.5956% 降到 5.8684%。整网仍未对齐。**
XT 使用无实验 norm hook 的生产 RMSNorm 修正版基线；AM 使用上述显式配置，
保留原生 attention cores、routed experts 和权重。原生 AM 默认配置的旧误差仍保留
作为控制组，不能把配置收益写成默认行为已改变。

| 抓点（真实 token，最终 logits 为全部监督位置） | 原生 AM 对 XT | 显式精度配置 AM 对 XT |
|---|---:|---:|
| 第一层 layer0 | 0.404925% | 0.000000%，逐位一致 |
| 第二层 layer1 | 8.256200% | 5.476202% |
| 第三层 layer2 | 8.811579% | 5.915335% |
| 第四层 layer3 | 8.637021% | 5.822803% |
| 最终 norm | 10.037281% | 6.940419% |
| 全部监督 logits | 10.131602% | 6.483103% |
| top-1 分歧 | 9.5956% | 5.8684% |

双方输入 hash 一致，347 个视觉参数加载审计通过，抓点前后输出一致。比较覆盖
1261 个监督位置 × 154880 词表，范数使用 FP64。实验为 train/no_grad、compile=False、
EP1/SP1、八卡 FSDP；没有 optimizer step。精度配置适用于已验证的 CP1 场景，
对 CP>1 显式拒绝，避免把当前结果冒充 CP/SP 多卡验收。

下一步从新整网的第二层输入重新定位 DSA + MoE。现有 DSA 因果实验已说明 q_a
投影几何、选 key 集合和顺序会影响结果，但不能直接将旧输入下的干预收益外推为
新整网收益；也不通过整段替换 DSA 或恢复已撤销的 grouped-router 修改来掩盖差异。

完成证据：`native_core_precision_forward/comparison.json`、
`native_core_precision_forward/forward_logits_hidden_states.png`、
`native_core_precision_forward/supervised_token_errors.csv`、
`native_core_precision_forward_valid.log` 和 manifest。首次退出断言失败的日志仍保留。


## 第二层：共同输入下 DSA 残差闭环与 MoE 放大

**结论：第二层 DSA 已在显式诊断配置下前向逐位一致；此前第二层整层误差
5.476202% 主要受到 MoE 离散专家选择放大的影响。新 DSA 配置的八卡结果见本节末尾。**
本节以已完成的第一层精度配置八卡 dump 为输入，第一层输出及第二层 input norm
在双方之间逐位相同。原生 DSA 重放各自逐位复现整网抓点，排除了重放环境漂移。
比较覆盖 7384 个真实 token，范数以 FP64 归约。

### 三种实现与对齐选择

| 计算边界 | XTuner 当前实现 | AutoModel 原生实现 | HF 原生实现 | 本次显式配置 |
|---|---|---|---|---|
| q_a 投影 | 整个 packed sequence 一次 GEMM | 文档内每 32 个 query 一次 GEMM | 本次 replay 对每个文档一次 GEMM | AM 用自己的权重做 full-pack，再切片供原生 query 路径使用 |
| indexer wk/weights/compress 投影 | full-pack | wk/compress 按文档，weights 按 query chunk | 输入序列范围内投影 | AM 三项自有投影统一 full-pack，保留原生 prepare_pools/select |
| indexer score/top-k | TileLang fused score、top-k | FP32 Torch einsum/top-k | FP32 Torch einsum/top-k | XT 显式选择已有 Torch 参考实现 |
| sparse key 顺序 | 原 top-k/pool 顺序 | cuDNN 适配层整理后的 key 顺序 | 构造布尔 mask，按原序列坐标取 key | XT 对自有选中 key 排序，invalid 放尾部 |
| attention core | absorbed MLA，FlashMLA/cuDNN | 原生 absorbed MLA，FlashMLA/cuDNN | 展开 K/V，eager attention | 保留 XT/AM 各自原生稀疏核心 |
| MoE 专家选择 | 分组 sigmoid gate | 分组 sigmoid gate | 分组 sigmoid gate | 同输入三方集合一致；未恢复已撤销的 grouped-router 修改 |

HF 是独立公式与选择集合对照，不把 AM/XT 的 absorbed sparse core 与 HF eager
宣称为相同内核。上表描述本次 fixture/replay 的调用范围，并非全场景性能结论。

### 单项到累计干预

| 干预组合 | DSA 输出 relL2 |
|---|---:|
| 相同输入、原生 XT / AM | 0.20763544% |
| AM 自有 q_a 改 full-pack，XT key 排序 | 0.01319740% |
| 上项再将 XT indexer 改 Torch | 0.01239902% |
| AM q_a + wk/weights/compress 全部 full-pack，XT 原始 key 顺序 | 0.11542969% |
| 全部 full-pack + XT key 排序，保留 TileLang score | 0.00452059% |
| 全部 full-pack + XT Torch score + key 排序 | **0，逐位一致** |
| 可复用 native_dsa_precision 上下文 | **0，逐位一致；退出后双方原始输出均逐位恢复** |

每边都使用自身权重与本次输入，没有替换为另一边已存储的投影值、选 key 集合或输出。
本次 reusable helper 保留 AM 原生 prepare_pools，压缩 gate 的 full-pack 结果通过
仅匹配该权重身份的临时线性入口提供；上下文退出恢复所有方法。
该 helper 明确限制 no_grad、batch1、SP1/CP1，拒绝把前向诊断入口直接用于训练。

这里的投影几何、score 归约与 key 顺序改变的是有限精度计算，不构成已证实的数学
公式错误。继续保留 TileLang 生产默认；Torch indexer 会物化大规模 FP32 分数张量，
只适合当前诊断，不能直接外推到 64K 生产配置。

### MoE 对上游差异的放大

在接入新 DSA 配置之前，MoE 输入 relL2 为 0.28340932%。固定同一输入后，
XT/AM/HF 的专家集合完全相同；路由权重差异比例仅 5.68068e-8（max abs 8.94070e-8）。
但使用各自上游输入时，**328/7384 个 token（4.4420%）改变专家集合**。

| 公共专家数学重放 | routed 输出 relL2 |
|---|---:|
| 各自输入 + 各自路由 | 9.5960465% |
| 使用 AM 输入但固定 XT 专家集合，权重重新计算 | 0.5590537% |
| 完全相同 XT 输入/路由，仅使用 AM activation | 0.3488494% |

重放覆盖全部 288 个专家，FP32 weighting/accumulation，但不含 shared expert 和 mHC，
也不是 native grouped GEMM/combine 的验收。结论是先消除上游 DSA 输入差异，再定位
同输入条件下的专家 activation/combine 精度，不能由这组数学重放断言生产 MoE 已对齐。

证据目录：validation/train_infra/layer1_dsa_resumed_20261013/。新证据包括
aligned_input_comparison.json、own_projections_comparison.json、all_projections_comparison.json、
reusable_profile_comparison.json、moe_aligned_input_comparison.json 及对应日志。
2026-10-13 再次只读核对 upstream main=ac2d6499，已为当前 HEAD 的祖先；AM 仓库保持干净。


### DSA 可复用配置的八卡整网验收

**双方执行与退出均为 0；第二层 attention 与 MoE 输入逐位一致。整网监督 logits
relL2 1.615920%，top-1 分歧 2.6170%。** 这是新的双方显式配置：XT 改用 Torch
indexer 与自有 key 排序；AM 沿用第一层精度配置并统一自有 DSA 投影的 full-pack
计算形状。因为 XT 基线也改变，不能将此列误读为“只改 AM 后对固定 XT 默认”的结果。

| 抓点 / 指标 | 仅第一层精度配置 | 再加双方 DSA 配置 |
|---|---:|---:|
| layer0 | 0，逐位一致 | 0，逐位一致 |
| layer1.self_attn | 0.207635% | 0，逐位一致 |
| layer1.post_attention_layernorm | 0.283409% | 0，逐位一致 |
| layer1 | 5.476202% | 0.388129% |
| layer2 | 5.915335% | 1.477646% |
| layer3 | 5.822803% | 1.607154% |
| final norm | 6.940419% | 2.010676% |
| 全监督 logits | 6.483103% | 1.615920% |
| top-1 分歧 | 5.8684% | 2.6170% |

重新对新整网抓点运行原生三方 router：MoE 输入逐位相同，专家集合差异从 328 个
真实 token 变成 **0/7384**；路由权重 max abs 仍仅 8.9407e-8。最常用专家的同输入
activation XT–AM relL2 为 0.291260%，down projection 后为 0.364555%；XT activation
与 HF 逐位相同。下一步保持路由/矩阵乘法/合并不变，仅把 AM routed activation
切换为 XT/HF BF16 精度，测量整网剩余误差。

347 个视觉参数审计、输入 hash、抓点与未抓点前向一致性均通过。比较全部
1261×154880 监督 logits；XT dump dtype 为 FP32、AM 为 BF16，比较按数值进行，
不因此改变输入或模型计算精度。范围仍是四层、8K pack、EP1/SP1、train/no_grad；
没有 optimizer step、完整反向、64K 或多 SP 验收。退出时已有 unwaited collective
警告保留在日志中，进程退出码 0，不能据此前向结果宣称训练同步正确。

持久化：dsa_precision_forward/ 下 JSON/CSV/图及双方 metadata/视觉审计，
dsa_precision_forward_{xtuner,automodel}.log、moe_dsa_profile_comparison.json、
dsa_precision_manifest.json。大 dump 为 /tmp/glm53_dsa_precision_forward_20261013/。


### 固定 XT 基线再统一 AM routed activation

专家激活单变量八卡实验：固定上一项 XT 基线，只把 AM routed SwiGLU 从 FP32 中间计算改为 XT/HF BF16 舍入，保留原生专家矩阵乘法、路由和合并。真实专家输入上的新入口输出逐位匹配 XT/HF，整网两次前向一致，执行与退出码均为 0。

结果：layer0、第二层 DSA 和 MoE 输入仍逐位一致；第二层整层误差从 0.388129% 降至 0.325893%，layer2=1.334441%，layer3=1.500422%，final norm=1.949152%。全监督 logits 从 1.615920% 降至 1.514695%，top-1 分歧从 2.6170% 降至 2.3791%。

解释与下一步：激活精度确有贡献，但没有解释全部残差。下一步在相同输入/专家集合下拆分 native grouped GEMM、token 排列、padding 参与范围、路由权重归一化和输出合并；这些仍是待验证候选，不能先称作 XT bug。已确认的生产修复仍只有 GLM 普通 RMSNorm，未恢复 grouped-router 改动。

证据：layer1_dsa_resumed_20261013/dsa_moe_activation_forward/ 下完整 JSON/CSV/图、metadata/视觉审计；moe_activation_manifest.json、moe_activation_check_comparison.json 和对应日志/source/hash。raw=/tmp/glm53_dsa_moe_activation_forward_20261013/automodel，XT 复用 /tmp/glm53_dsa_precision_forward_20261013/xtuner。范围是八卡四层 8K 前向；不含整网反向、优化器步或多 SP 验收。


## MoE 剩余误差：原生矩阵乘法一致，差异在权重与合并精度

**固定第二层真实 MoE 输入、原生专家选择及 checkpoint 权重后，全部 288 个专家的
两次 grouped GEMM 均逐位一致；原生输出合并 relL2 为 0.470361%。**
这一组比较使用实际 8192-token 路由形状，统计覆盖 7384×8 个有效专家输入。
双方原生 permute 的行顺序和专家槽位相同；去除 padding 路由不改变实测差异。
它是固定边界的 native primitive 比较，随后另用 AM GroupedExperts 原生 forward
复现，再验证精度上下文，不拿早先“公共数学重放”代替原生执行。

### 三实现的关键精度边界

| 项目 | XTuner 当前路径 | AutoModel 原生路径 | HF 原生路径 |
|---|---|---|---|
| 专家 top-k 排序 | sorted=True | sorted=False | sorted=False |
| 路由权重归一化 | FP32，按上述槽位顺序求和 | FP32，按上述槽位顺序求和 | FP32，按上述槽位顺序求和 |
| 专家投影 | Triton grouped GEMM | torch._grouped_mm | 按专家 F.linear |
| 激活 | BF16 clamp/silu/product | FP32 中间计算，最后回 BF16 | BF16 clamp/silu/product |
| routed 合并 | grouped_gemm CUDA unpermute；默认路由权重/乘法/逐项累加均 BF16 | FP32 权重相乘、scatter 到槽位后 FP32 sum，最后回 BF16 | 专家循环中乘 FP32 权重后回 BF16，再 index_add 到 BF16 final |
| 已有高精度开关 | PERMUTE_COMPUTE_DTYPE=fp32 | 原生已 FP32；仍有计算顺序差异 | 本次作为独立实现对照，不强行替换其整个 MoE |

固定同一组 routed 专家结果，逐项候选比较发现：只有“先将 probs 转 BF16，BF16
乘积，再逐槽位 BF16 累加”能逐位复现 XT 默认 combine。FP32 probs + BF16 product
或 BF16 product + FP32 sum 都不能复现。安装扩展的来源记录指向 GroupedGEMM；
对应 CUDA 源码也明确在未设置 PERMUTE_COMPUTE_DTYPE=fp32 时选择 BF16 TCompute。
源码快照和数值指纹互相支持；没有修改共享扩展、Python 环境或其他工作树。

### 原生 FP32 开关与完整八卡结果

新进程设置 PERMUTE_COMPUTE_DTYPE=fp32 后，同输入 combine relL2 从 **0.470361%
降至 0.00163008%**。开关在扩展中由静态 getenv 读取，必须在新进程启动前设置，
不能依赖已运行进程中的临时环境切换。新 probe metadata 显式记录该环境变量。
XT 现有 autotest 与部分 example 也使用这个开关，因此这里首先归类为运行精度配置
差异，不直接把扩展既有 BF16 默认策略改成全局生产 bug fix。

固定此前 AM DSA+activation 配置，只改变 XT 的原生 combine 设置，八卡执行和完整
比较均退出 0。layer0 仍逐位一致，layer1=0.00146462%，layer2=0.596735%，
layer3=0.826024%，final norm=1.242414%；全监督 logits=**0.973999%**，top-1 分歧
**1.665345%**。这是进一步改变 XT 精度配置后的新比较，不是旧默认模型的误差。

### 剩余微差与可复用 AM reduction 配置

同输入同权重时，把 AM 原生 top-k 改为 sorted=True，保持专家选择算法、分组及
logits 不变，即可让路由权重与 XT 逐位一致。当前检查明确限制 n_groups=1，
没有恢复用户要求撤销的 grouped-router 修改，也不把该断言外推到分组路由。

FP32 combine 仍有小量乘加/归约顺序差异：普通 Torch sequential sum 和 addcmul
均未在真实输入上完全复现 CUDA 输出。独立混合正负 BF16 合成用例中，显式 Triton
FP32 FMA 扫描逐位匹配原生 CUDA；禁用 FMA 则有 51 个坐标不同。该合成用例只用于
识别舍入机制，不作为真实整网或反向通过的证据。

可复用 automodel_moe_reduction_precision(text) 保留 AM native GroupedExperts.forward
及其真实 permute/GEMM，捕获其自己的专家结果和槽位，使用已有 XT FP32 unpermute
统一合并原语。所有数值来自 AM 本次自己的计算；没有引用 XT 保存的专家输出。
当前实验会额外执行一次合并，属于显式 forward/no_grad/EP1 精度诊断，尚不是性能
优化实现。它还保留原生 router，只设置 top-k 排序，并在退出时恢复所有方法。

**单层真实数据验收：AM 原生 GroupedExperts 逐位复现独立重放；开启新配置后，
7384 个真实 token 的 routed 输出与 XT 逐位一致；退出配置后原始 AM 输出恢复。**
八卡整网验证已完成，见下一节；单层 routed 验收本身不构成训练验收。

证据：moe_native_kernels_comparison.json、moe_combine_rounding_comparison.json、
moe_combine_fp32_comparison.json、moe_fp32_order_comparison.json、
moe_reduction_profile_comparison.json、moe_combine_scan_synthetic.log，及对应源码、日志、
moe_combine_manifest.json / moe_reduction_manifest.json。完整八卡 FP32 combine
报告位于 moe_fp32_combine_forward/；原始大 dump 的准确路径见根 AGENTS.md。


## 最终四层前向验收与复现配置

**结论：真实八卡 FSDP 的四层前向已在显式精度配置下对齐；没有遗漏监督词表坐标。**
XT raw 为 /tmp/glm53_moe_fp32_combine_forward_20261013/xtuner，AM raw 为
/tmp/glm53_moe_reduction_forward_20261013/automodel。两边运行、退出、完整比较均为 0。

| 验收项 | 结果 |
|---|---|
| 全部输入 hash | 一致 |
| 347 个视觉参数及视觉加载 | 审计通过 |
| 四层 hidden states、final norm | 7384 个真实 token 逐位一致 |
| 全部文本抓点 | 18/18 个抓点的真实 token 逐位一致 |
| 全监督 logits | 1261×154880 个坐标数值一致；relL2=max abs=0 |
| logits dtype | XT FP32、AM BF16，不能称跨 dtype 的 bitwise |
| top-1 与逐 token CE | top-1 分歧 0；双方 mean CE=13.419907569885254 |
| 抓点是否改变结果 | 两边均逐位复现未观测 logits |
| 参数/模型 | AM repo 未改，原生 attention/专家 GEMM 保留；FSDP 参数结构检查通过 |

### 对齐配置组成

1. XT 使用当前 GLM 普通 RMSNorm 修正；双方在新进程启动前设置
   PERMUTE_COMPUTE_DTYPE=fp32，沿用原生扩展的 FP32 combine。
2. AM --native-core-precision：统一 mHC/gated norm/SwiGLU/gate FP32、64-token
   prefix 与 int32 metadata，保留原生 KDA/DSA 核心。
3. 双方 --dsa-precision：XT 自有 Torch indexer 与 key 排序，AM 自有 full-pack
   q_a/wk/weights/compress 投影，保留原生文档隔离与 sparse core。
4. AM --routed-activation-precision：只统一 routed SwiGLU 的 BF16 精度边界。
5. AM --moe-reduction-precision：原生 top-k 使用 sorted=True；原生 GroupedExperts
   完成自己的 GEMM，再按自身槽位与权重使用 XT 的原生 FP32 combine。

这些步骤不使用另一边的历史 tensor 覆盖输出。最终配置共享部分数值原语，因此是
“明确数值契约下的对齐”，不宣称两边所有原生默认实现天然逐位一致。最后一步当前
还会额外执行一次合并；Torch indexer 也会物化 FP32 分数，不直接作为 64K 生产配置。

复现入口为 experiments/train_infra/probe_forward.py。公共参数固定为：

```text
--fixture /tmp/forward_alignment_20261013/pack_rank0_step0.safetensors
--reference /mnt/shared-storage-user/yushan/GLM-5.3-flash-integration/validation/train_infra/runs/xtuner_20261013T045424_1102069
```

从 XT worktree 启动，设置 PYTHONPATH=XT_WORKTREE:ROOT/automodel，以及
OMP_NUM_THREADS=1、HF_HUB_OFFLINE=1、XTUNER_USE_FA3=0、PERMUTE_COMPUTE_DTYPE=fp32。
通过 /mnt/shared-storage-user/yushan/asserts/gpu_lock.sh 串行运行两个八卡任务：

```text
XT_PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m experiments.train_infra.probe_forward --backend xtuner --dsa-precision [公共参数] --out /tmp/新的目录/xtuner
AM_PYTHON -m torch.distributed.run --standalone --nproc_per_node=8 -m experiments.train_infra.probe_forward --backend automodel --native-text --native-core-precision --dsa-precision --routed-activation-precision --moe-reduction-precision [公共参数] --out /tmp/新的目录/automodel
```

XT_PYTHON=/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2/bin/python；
AM_PYTHON=/mnt/shared-storage-user/yushan/conda-envs/pt29_glm2_automodel/bin/python。
用同一 plot_forward.py 比较两边目录并持久化小型 JSON/CSV/图；大 dump 继续写 /tmp。

### 修复与后续生产落地边界

已证实并修改的 XT bug 是普通 RMSNorm 的 GLM 精度流程（16 行）；回归包含
HF 同输入前向、输入/权重梯度与 builder 安装范围，另有无适配 hook 的八卡 58 抓点
等价验收。没有恢复用户撤销的 grouped-router 修改，也没有把有限精度策略差异
都标成数学 bug。最新 upstream main ac2d6499 已为实验 HEAD 49af42b54 的祖先。

下一阶段若用于实际训练，需要将诊断配置整理为可维护且无重复计算的实现，并验收
整网反向/优化器、多输入、长序列、EP/SP、compile 与吞吐。当前不据此前向结果声称
这些范围已经完成。第一层已有独立 VJP 逐位验收，但不能替代新全配置的整网反向。

最终证据：layer1_dsa_resumed_20261013/moe_reduction_forward/ 下完整比较、逐 token
CSV、图、双方 metadata/视觉审计；moe_reduction_manifest.json、moe_reduction_forward.log
及精确运行源码/hash。新实验证据、失败候选和原始默认结果均保留，不能只保留零误差结论。


## 对齐配置 20 步训练对照（2026-10-14）

两侧八个 rank 均完成 20/20 步、exit 0，固定输入 21 个 logits 点齐全。初始监督 logits 数值完全一致；第 20 次更新后逐 token 相对 L2 最大/平均/中位数为 17.0723% / 4.4371% / 4.3074%，全局 relL2 5.4823%。因此初始前向对齐没有扩展为完整训练对齐。

末步 loss：XT=8.83555031，AM=8.83311653；clip 前全局 grad norm：XT=9.47808838，AM=9.54917583。

复现：`TRAIN_INFRA_ALIGNED_PRECISION=1 bash experiments/train_infra/run_comparison.sh NEW_OUTPUT_DIRECTORY`；脚本启动前设置 `PERMUTE_COMPUTE_DTYPE=fp32`，显式启用精度适配的实验反向。当前默认关闭 balancing auxiliary loss，两侧一致。第一步 lr=0 后 logits 已分歧，后续需区分 router bias/状态更新和参数梯度的贡献。

证据：`validation/train_infra/aligned_training_20261014/`（相对 integration 根目录）；图为 `plots/loss_logits_grad_norm.png`。3 项预检通过，8-rank 训练及输入回放验收通过；不覆盖 64K、多 EP/SP、compile 或解冻视觉。
