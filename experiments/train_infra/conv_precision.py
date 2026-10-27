"""AM-only diagnostic: BF16 convolution forward with FP32 weight gradients."""
from contextlib import contextmanager
from unittest.mock import patch

import torch


def _fp32_conv(self, x, *, cu_seqlens=None, **kwargs):
    from xtuner.v1.ops.kda.causal_conv1d import causal_conv1d

    assert self.weight.dtype == torch.float32, "Pin convolution compute/reduce dtype before FSDP wrapping"
    assert kwargs.get("cache") is None and kwargs.get("cp_context") is None
    assert not kwargs.get("output_final_state", False)
    # The shared kernel rounds weight values for forward, retaining FP32 dW in backward.
    return causal_conv1d(x=x, weight=self.weight.squeeze(1), activation="silu", cu_seqlens=cu_seqlens)


@contextmanager
def automodel_conv_precision():
    from nemo_automodel.components.models.glm5_next.layers import _TorchShortConvolution

    with patch.object(_TorchShortConvolution, "forward", _fp32_conv):
        yield {"scope": "AM diagnostic, EP1/SP1, no cache",
               "forward": "activation-dtype weight values; unchanged convolution formula",
               "weight_compute_dtype": "float32", "gradient_reduce_dtype": "float32",
               "kernel": "XT native KDA convolution; XT source unchanged"}
