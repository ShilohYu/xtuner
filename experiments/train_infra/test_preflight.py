"""Guard against passing corrupt supervision or media/token correspondence."""

import copy
import unittest

import torch
from PIL import Image

from experiments.train_infra.preflight import check_image_geometry, check_item


class PreflightContracts(unittest.TestCase):
    def test_exif_rotation_is_distinct_from_bad_encoded_geometry(self):
        image = Image.new("RGB", (4, 2))
        image.getexif()[274] = 6
        self.assertTrue(check_image_geometry(image, [4, 2]))
        with self.assertRaises(AssertionError):
            check_image_geometry(image, [2, 4])

    def test_media_and_supervision_corruption_are_rejected(self):
        item = {"input_ids": [1, 99, 3], "labels": [-100, -100, 3], "num_tokens": 3,
                "image_grid_thw": torch.tensor([[1, 2, 2]]), "num_img_tokens": [4],
                "pixel_values": torch.zeros(4, 12)}
        self.assertEqual(check_item(item, 99, 1, 4), 1)
        bad = copy.deepcopy(item)
        bad["labels"][1] = 99
        with self.assertRaises(AssertionError):
            check_item(bad, 99, 1, 4)
        bad = copy.deepcopy(item)
        bad["image_grid_thw"] = torch.tensor([[1, 2, 4]])
        with self.assertRaises(AssertionError):
            check_item(bad, 99, 1, 4)


if __name__ == "__main__":
    unittest.main()
