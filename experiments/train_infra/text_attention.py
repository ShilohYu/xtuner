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


def _text_mlp(self, x):
    from xtuner.v1.ops.act_fn import get_gated_act_fn

    activation = get_gated_act_fn("silu", self.swiglu_limit)
    return self.down_proj(activation(self.gate_proj(x), self.up_proj(x)))


def install_text_mlp(text):
    from nemo_automodel.components.moe.layers import MLP

    count = 0
    for module in text.modules():
        if isinstance(module, MLP) and module.activation == "swiglu" and module.swiglu_limit > 0:
            # Scope to text dense/shared experts; leave vision and routed expert kernels intact.
            module.forward = MethodType(_text_mlp, module)
            count += 1
    assert count, "No native clamped text MLP modules selected"
    return count


def _decoder_with_shared_hc_post(self, hidden_streams, *, packed_context, padding_mask=None, **kwargs):
    from xtuner.v1.module.decoder_layer.mhc import hc_post

    residual = hidden_streams
    post, comb, collapsed = self.attn_hc(hidden_streams)
    update = self.self_attn(self.input_layernorm(collapsed), packed_context=packed_context,
                            padding_mask=padding_mask, **kwargs)
    hidden_streams = hc_post(update, residual, post, comb)
    residual = hidden_streams
    post, comb, collapsed = self.ffn_hc(hidden_streams)
    update = self.post_attention_layernorm(collapsed)
    update = self.mlp(update, padding_mask) if self.is_moe_layer else self.mlp(update)
    return hc_post(update, residual, post, comb)


def install_shared_text_kernels(layers):
    from xtuner.v1.ops import rms_norm
    from xtuner.v1.module.decoder_layer.mhc import hc_post
    from xtuner.v1.ops.gated_deltanet.rms_norm_gated import rms_norm_gated
    from xtuner.v1.ops.kda import get_causal_conv1d_fn, get_chunk_kda_fn, get_fused_kda_gate_fn

    def unweighted_norm(self, hidden_states):
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight=None, eps=self.eps)

    def gated_norm(self, hidden_states, gate):
        return rms_norm_gated(hidden_states, gate, self.weight, None, activation="sigmoid",
                              eps=self.variance_epsilon)

    layers.Glm5NextUnweightedRMSNorm.forward = unweighted_norm
    layers._TorchRMSNormGated.forward = gated_norm
    # Keep the native submodules/parameters; only replace the two residual-mix expressions.
    layers.Glm5NextDecoderLayer.forward = _decoder_with_shared_hc_post
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
            "hc_post": hc_post.__module__ + "." + hc_post.__name__,
            "chunk_kda": chunk_kda.__module__ + "." + chunk_kda.__name__,
            "forget_gate": layers._fused_kda_gate.__module__ + "." + layers._fused_kda_gate.__name__,
            "conv1d": layers._fla_causal_conv1d.__module__ + "." + layers._fla_causal_conv1d.__name__,
            "gated_norm": rms_norm_gated.__module__ + "." + rms_norm_gated.__name__}
