"""Keep explicit sparse inputs intact while adapting the installed FlashMLA API."""

import unittest
from types import SimpleNamespace

from experiments.train_infra.run_automodel import register_flash_mla_compat


class FlashMLACompatTest(unittest.TestCase):
    def test_legacy_signature_forwards_sparse_contract_and_rejects_nonzero_indexer(self):
        received = []

        def legacy(q, kv, indices, scale, *, topk_length, attn_sink, d_v):
            received.append((q, kv, indices, scale, topk_length, attn_sink, d_v))
            return "output", "max", "lse"

        module = SimpleNamespace(_FLASH_MLA_SPARSE_FWD=legacy)
        self.assertTrue(register_flash_mla_compat(module))
        tensors = [object() for _ in range(5)]
        output = module._FLASH_MLA_SPARSE_FWD(*tensors[:3], 0.25, topk_length=tensors[3],
                                            attn_sink=tensors[4], d_v=512, indexer_topk=0)
        self.assertEqual(output, ("output", "max", "lse"))
        self.assertEqual(received, [(*tensors[:3], 0.25, tensors[3], tensors[4], 512)])
        with self.assertRaises(AssertionError):
            module._FLASH_MLA_SPARSE_FWD(indexer_topk=1)

    def test_current_signature_is_unchanged(self):
        def current(*args, indexer_topk=0, **kwargs):
            return indexer_topk

        module = SimpleNamespace(_FLASH_MLA_SPARSE_FWD=current)
        self.assertFalse(register_flash_mla_compat(module))
        self.assertIs(module._FLASH_MLA_SPARSE_FWD, current)


if __name__ == "__main__":
    unittest.main()
