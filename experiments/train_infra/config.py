"""GLM 四层对照：优化器跟随 glm53_flash_4layer_train_8gpu.py 的 Muon 配置。"""

import os
from pathlib import Path

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import Glm53VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig
from xtuner.v1.train import TrainerConfig


data_root = Path("/mnt/shared-storage-user/llmrazor-share/data/train_infra_10_per_dataset")
model_path = "/tmp/GLM-5.3-Flash-4layer"
processor_path = "/tmp/GLM-5.3-Flash-4layer-with-processor"
work_dir = os.environ["WORK_DIR"]
length = int(os.environ.get("TRAIN_INFRA_LENGTH", "65536"))
sp_size = int(os.environ.get("TRAIN_INFRA_SP", "1"))
compile_enabled = os.environ.get("TRAIN_INFRA_COMPILE", "1") == "1"
total_step = int(os.environ["TRAIN_INFRA_STEPS"]) if os.environ.get("TRAIN_INFRA_STEPS") else None
# 【模型差异】Qwen 接收像素面积；GLM 接收合并后的 image token 数。
image_token_budget = 16777216 // (14 * 2) ** 2
loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="square")
model_cfg = Glm53BaseConfig.from_hf(model_path)
model_cfg.text_config.ep_size = 1
model_cfg.text_config.dispatcher = "deepep"
model_cfg.text_config.lm_loss_cfg = loss_cfg
model_cfg.text_config.attention.sparse_mla_backend = "flash_mla_cudnn"
model_cfg.text_config.attention.indexer_backend = "tilelang"
model_cfg.vision_config.attn_impl = "flash_attention"
model_cfg.freeze_vision = model_cfg.freeze_projector = os.environ.get("TRAIN_INFRA_FREEZE_VISION", "0") == "1"
model_cfg.compile_cfg = compile_enabled
model_cfg.text_config.compile_cfg = compile_enabled

dataset_config = [{
    "dataset": DatasetConfig(
        name="train_infra_10_per_dataset",
        # 【数据差异】原 meta_data + OSS 改为同一抽样集的完整本地媒体副本。
        anno_path=os.environ.get("TRAIN_INFRA_DATA", str(data_root / "train_infra_localdata.jsonl")),
        media_root=str(data_root),
        class_name="VLMJsonlDataset", sample_ratio=1.0,
        enable_sequential_sampler=True,
        cache_tag="train_infra_qwen_aligned",
        cache_dir=str(Path(work_dir) / "jsonl_cache"),
    ),
    "tokenize_fn": Glm53VLTokenizeFnConfig(
        processor_path=processor_path, max_length=length,
        max_frames=24, max_pixels=image_token_budget,
        llm_pack_weight=-3.2, visual_pack_weight=5.0, debug=True,
    ),
}]
trainer = TrainerConfig(
    model_cfg=model_cfg, load_from=model_path, tokenizer_path=model_path,
    # 【验证差异】每次从同一权重开始，禁止读取旧实验 checkpoint。
    strict_load=True, auto_resume=False,
    sp_size=sp_size, global_batch_size=8, intra_layer_micro_batch=1,
    optim_cfg=MuonConfig(lr=8e-6, weight_decay=0.1),
    lr_cfg=LRConfig(lr_type="cosine", warmup_ratio=float(os.environ.get("TRAIN_INFRA_WARMUP", "0.1")), lr_min=1e-6),
    loss_cfg=loss_cfg,
    fsdp_cfg=FSDPConfig(ep_size=1, recompute_ratio=1.0,
                       torch_compile=compile_enabled, checkpoint_preserve_rng_state=False),
    dataloader_cfg=DataloaderConfig(
        dataset_config_list=dataset_config,
        pack_max_length=length, pack_level="soft", pack_to_max_length=True,
        pack_extra_buffer_size=20, num_workers=4,
        collator="glm53_vl_sft_collator",
        collator_kwargs={"image_token_id": model_cfg.image_token_id,
                         "merge_unit": model_cfg.vision_config.spatial_merge_size**2},
    ),
    total_step=total_step, total_epoch=None if total_step else 2,
    hf_interval=1000, hf_max_keep=1000, checkpoint_interval=1000,
    checkpoint_maxkeep=1000, snapshot_interval=500,
    # 【验证差异】保留日志、输入和参数更新指标，不保存数十 GB 的权重副本。
    debug_skip_save=True, work_dir=work_dir,
    exp_tracker="tensorboard", profile_step=int(os.environ.get("TRAIN_INFRA_PROFILE_STEP", "10")), profile_memory=True,
)
