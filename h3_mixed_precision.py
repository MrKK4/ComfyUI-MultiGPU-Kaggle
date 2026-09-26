"""fp16 compute for MiniMax H3 on GPUs without bf16 (Turing / T4).

ComfyUI lists H3's inference dtypes as bf16/fp32, so a T4 runs the whole DiT in
fp32. Plain fp16 overflows in three places, each kept wide here:
  * the raw Qwen3-VL text states (massive activations) -> text preprocess in fp32
  * the residual stream between blocks (peaks ~4.6e5) -> residual in fp32
  * the MLP fc2 and attention out_proj outputs (>6.5e4) -> those int8 linears run on a
    1/S-scaled fp16 input and are rescaled in fp32; exact, because comfy_kitchen's int8
    linear quantizes activations per row and neither layer has a bias.
Measured on 2x T4, Ref2VA 480x864x73: 21 -> 16.8 s/step, output matches fp32.
Disable with MMH3_H3_FP16=0."""
import logging
import os

import torch

logger = logging.getLogger("MultiGPU")

S = 64.0


def _no_bf16_gpu():
    if not torch.cuda.is_available():
        return False
    return all(torch.cuda.get_device_capability(i)[0] < 8 for i in range(torch.cuda.device_count()))


def patch_minimax_h3_mixed_precision():
    if os.environ.get("MMH3_H3_FP16", "1") == "0" or not _no_bf16_gpu():
        return False
    try:
        import comfy.conds
        import comfy.model_base
        import comfy.model_management
        import comfy.ops
        import comfy.quant_ops
        import comfy.supported_models
        import comfy.ldm.minimax.model as h3
        from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    except ImportError:
        return False
    if getattr(h3.DiTBlock.forward, "_mmh3_mixed", False):
        return True

    comfy.supported_models.MiniMaxH3.supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    extra_conds = comfy.model_base.MiniMaxH3.extra_conds
    model_forward = h3.MiniMaxH3Model.forward

    def extra_conds_fp32_text(self, **kwargs):
        cross_attn = kwargs.get("cross_attn", None)
        out = extra_conds(self, **{**kwargs, "cross_attn": None})
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(self.diffusion_model.preprocess_text_embeds(
                cross_attn.to(device=kwargs["device"], dtype=torch.float32)))
        return out

    def model_forward_fp32_stream(self, x, timestep, context, *args, **kwargs):
        return model_forward(self, x, timestep, context.float(), *args, **kwargs)

    def attention_forward(self, x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps,
                                                   rot_dim=rope_freqs.shape[-3] * 2)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))
        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out.squeeze(0) * (1.0 / S)).float() * S

    def mlp_forward(self, x):
        a = self.fc1(x)
        a[..., a.shape[-1] // 2:].mul_(1.0 / S)  # swiglu = silu(gate) * up: scaling up scales the output
        return comfy.ops.linear_input_act(self.fc2, a, "swiglu").float() * S

    def block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        attention = self.attn if attention is None else attention
        x = x.float()
        rope = rope_freqs.to(torch.float16) if rope_freqs is not None else None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = h3._mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments).to(torch.float16)
        x = h3._mod_gate(x, gate_msa, attention(h, rope_freqs=rope, transformer_options=transformer_options), mod_segments)
        h = h3._mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments).to(torch.float16)
        return h3._mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

    block_forward._mmh3_mixed = True
    comfy.model_base.MiniMaxH3.extra_conds = extra_conds_fp32_text
    h3.MiniMaxH3Model.forward = model_forward_fp32_stream
    h3.Attention.forward = attention_forward
    h3.MLP.forward = mlp_forward
    h3.DiTBlock.forward = block_forward
    logger.info("[MultiGPU] MiniMax H3 fp16 mixed precision enabled (no bf16 GPU); MMH3_H3_FP16=0 disables")
    return True
