"""Share kernel implementations while keeping native AutoModel modules and loading."""

from types import MethodType

import torch.nn.functional as F


def _text_rms_norm(self, hidden_states):
    from xtuner.v1.ops import rms_norm

    return rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)


def install_text_rms_norm(text):
    from nemo_automodel.components.models.glm5_next.layers import Glm5NextRMSNorm, _TorchRMSNormGated

    count = 0
    for module in text.modules():
        if isinstance(module, Glm5NextRMSNorm) and not isinstance(module, _TorchRMSNormGated):
            # ViT shares this class but has its own FP32-statistics/cast/affine order.
            module.forward = MethodType(_text_rms_norm, module)
            count += 1
    assert count, "No native text RMSNorm modules selected"
    return count


def install_shared_text_kernels(layers):
    from xtuner.v1.ops import rms_norm
    from xtuner.v1.ops.gated_deltanet.rms_norm_gated import rms_norm_gated
    from xtuner.v1.ops.kda import get_causal_conv1d_fn, get_chunk_kda_fn, get_fused_kda_gate_fn

    def unweighted_norm(self, hidden_states):
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight=None, eps=self.eps)

    def gated_norm(self, hidden_states, gate):
        return rms_norm_gated(hidden_states, gate, self.weight, None, activation="sigmoid",
                              eps=self.variance_epsilon)

    layers.Glm5NextUnweightedRMSNorm.forward = unweighted_norm
    layers._TorchRMSNormGated.forward = gated_norm
    chunk_kda = get_chunk_kda_fn()

    def shared_chunk(**kwargs):
        # Native GLM discards the final state; XT's compile-friendly kernel omits it.
        assert kwargs.get("cp_context") is None, "Shared text-kernel diagnostic requires CP1"
        kwargs["output_final_state"] = False
        return chunk_kda(**kwargs)

    layers._chunk_kda = shared_chunk
    layers._fused_kda_gate = get_fused_kda_gate_fn()
    layers._fla_causal_conv1d = get_causal_conv1d_fn()
    return {"rms_norm": rms_norm.__module__ + "." + rms_norm.__name__,
            "rms_norm_scope": "language module instances only; native vision unchanged",
            "chunk_kda": chunk_kda.__module__ + "." + chunk_kda.__name__,
            "forget_gate": layers._fused_kda_gate.__module__ + "." + layers._fused_kda_gate.__name__,
            "conv1d": layers._fla_causal_conv1d.__module__ + "." + layers._fla_causal_conv1d.__name__,
            "gated_norm": rms_norm_gated.__module__ + "." + rms_norm_gated.__name__}
