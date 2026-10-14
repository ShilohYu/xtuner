import unittest

import torch

from experiments.train_infra.record import recording_collator
from experiments.train_infra.replay import square_weights
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContext, CELossKwargs


class ReplayTests(unittest.TestCase):
    def test_square_weights_match_native_unequal_documents_and_padding(self):
        labels = torch.tensor([[1, -100, 2, 3, 4, 5, -100, -100]])
        boundaries = torch.tensor([0, 2, 6, 8], dtype=torch.int32)
        context = CELossContext(CELossConfig(loss_reduction="square"), CELossKwargs(shifted_labels=labels))
        native = CELossContext.build_batches([context], [boundaries])[0].loss_kwargs.loss_weight
        weights = square_weights(labels, boundaries)
        torch.testing.assert_close(weights / weights.sum(), native, rtol=0, atol=0)
        self.assertEqual(float(weights[0, 0]), 1.0)
        self.assertEqual(float(weights[0, 2]), 0.5)
        self.assertEqual(float(weights[0, 6:].sum()), 0.0)

    def test_provenance_follows_dropped_pack_tail(self):
        def sample(offset, count):
            return {"_source_offset": offset, "input_ids": [1] * count,
                    "labels": [1] * count, "num_tokens": count,
                    "mm_token_type_ids": torch.zeros(1, count, dtype=torch.long)}
        batches = recording_collator([[sample(10, 6), sample(20, 6)]],
                                     pack_max_length=8, padding_token_idx=0,
                                     image_token_id=99, merge_unit=4)
        self.assertEqual(batches[0]["_source_offsets"], [10])
        self.assertEqual(batches[0]["seq_ctx"].num_padding, 3)


if __name__ == "__main__":
    unittest.main()
