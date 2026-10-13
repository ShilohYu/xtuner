"""Reproduce the missing-language-layer heuristic on the real native model graph."""

import unittest

import torch
from torch.distributed.pipelining import microbatch
from torch.nn.attention.flex_attention import BlockMask

microbatch.BlockMask = BlockMask

from experiments.train_infra.run_automodel import register_glm_layers
from nemo_automodel.components.distributed import parallelizer
from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration


class LayerGroupsTest(unittest.TestCase):
    def test_real_meta_glm_graph_includes_both_towers_after_fsdp_subclassing(self):
        cfg = Glm5NextConfig.from_pretrained("/tmp/GLM-5.3-Flash-4layer-with-processor")
        with torch.device("meta"):
            model = Glm5NextForConditionalGeneration(cfg)
        original = parallelizer._get_model_layer_group_specs
        try:
            legacy = parallelizer.get_model_layer_groups(model)
            self.assertEqual(len(legacy["unknown"]), 24)
            register_glm_layers(parallelizer, Glm5NextForConditionalGeneration)
            groups = parallelizer.get_model_layer_groups(model)
            self.assertEqual({key: len(value) for key, value in groups.items()}, {"language": 4, "vision": 24})
            # AutoModel's checkpoint mixin constructs a distinct class with the same name.
            model.__class__ = type("Glm5NextForConditionalGeneration", (Glm5NextForConditionalGeneration,), {})
            mixed = parallelizer.get_model_layer_groups(model)
            self.assertEqual({key: len(value) for key, value in mixed.items()}, {"language": 4, "vision": 24})
            model.__class__ = type("FSDPGlm5NextForConditionalGeneration", (Glm5NextForConditionalGeneration,), {})
            wrapped = parallelizer.get_model_layer_groups(model)
            self.assertEqual({key: len(value) for key, value in wrapped.items()}, {"language": 4, "vision": 24})
        finally:
            parallelizer._get_model_layer_group_specs = original


if __name__ == "__main__":
    unittest.main()
