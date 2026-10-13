"""Select the same external FlashAttention 2 ViT kernel as the XTuner reference."""

def install_flash_attention_2(vision, flash_varlen):
    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        sequence = hidden_states.shape[0]
        query, key, value = self.qkv(hidden_states).view(sequence, 3, self.num_heads, self.head_dim).unbind(1)
        query, key = self.q_norm(query), self.k_norm(key)
        cos, sin = (item.unsqueeze(-2).float() for item in position_embeddings)
        query_float, key_float = query.float(), key.float()
        query = (query_float * cos + vision._rotate_half(query_float) * sin).to(query.dtype)
        key = (key_float * cos + vision._rotate_half(key_float) * sin).to(key.dtype)
        max_length = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        output = flash_varlen(
            query, key, value, cu_seqlens, cu_seqlens, max_length, max_length,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scaling, causal=False, deterministic=False,
            return_attn_probs=True,
        )[0]
        return self.proj(output.reshape(sequence, -1).contiguous())

    # Native GLM ViT hardcodes SDPA; the generic backend.attn setting only selects text attention.
    vision.Glm5NextVisionAttention.forward = forward
