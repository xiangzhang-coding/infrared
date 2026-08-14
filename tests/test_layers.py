"""Unit tests for the Qwen2 building blocks (torch-only, no network/model).

Batch-first since T1. These pin the math the Seam-A parity gate checks end to
end: RMSNorm's fp32 path, RoPE, GQA head expansion, the additive-masked
attention (vs torch SDPA with the same mask), SwiGLU, and the sampler.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from infrared.model.config import Qwen2Config  # noqa: E402
from infrared.model.inputs import build_attention_mask  # noqa: E402
from infrared.model.layers import (  # noqa: E402
    MLP,
    Attention,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    masked_attention,
    repeat_kv,
    rotate_half,
)
from infrared.model.sampler import Sampler  # noqa: E402


def _tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_ids=(31,),
    )


def test_rmsnorm_matches_manual_formula() -> None:
    torch.manual_seed(0)
    norm = RMSNorm(16, eps=1e-6)
    norm.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(2, 5, 16)
    out = norm(x)
    var = x.pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (x * torch.rsqrt(var + 1e-6))
    assert torch.allclose(out, expected, atol=1e-6)


def test_rotate_half() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_is_identity_at_position_zero() -> None:
    rope = RotaryEmbedding(head_dim=8, theta=1e6)
    cos, sin = rope(torch.tensor([[0, 0]]))  # [B=1, S=2]
    q = torch.randn(1, 2, 3, 8)  # [B, S, H, D]
    k = torch.randn(1, 2, 2, 8)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_rope_preserves_norm() -> None:
    rope = RotaryEmbedding(head_dim=8, theta=1e6)
    cos, sin = rope(torch.arange(4)[None, :])  # [1, 4]
    q = torch.randn(1, 4, 3, 8)
    q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_repeat_kv_head_mapping() -> None:
    x = torch.randn(2, 3, 2, 4)  # [B, S, H_kv, D]
    out = repeat_kv(x, 3)
    assert out.shape == (2, 3, 6, 4)
    for j in range(6):
        assert torch.equal(out[:, :, j], x[:, :, j // 3])


def test_masked_attention_causal_matches_sdpa() -> None:
    torch.manual_seed(0)
    b, seq, heads, dim = 2, 6, 4, 8
    q, k, v = (torch.randn(b, seq, heads, dim) for _ in range(3))
    mask = build_attention_mask([0, 0], 0, seq, seq, torch.float32, "cpu")
    got = masked_attention(q, k, v, mask, scale=dim**-0.5)
    # SDPA with the same additive mask; it uses [B, H, S, D] layout.
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    ).transpose(1, 2)
    assert torch.allclose(got, ref, atol=1e-5)


def test_masked_attention_respects_left_padding() -> None:
    # Sequence 0 has 1 left-pad token; its query must ignore the pad key.
    torch.manual_seed(0)
    seq, heads, dim = 3, 2, 8
    q, k, v = (torch.randn(1, seq, heads, dim) for _ in range(3))
    mask = build_attention_mask([1], 0, seq, seq, torch.float32, "cpu")
    out_padded = masked_attention(q, k, v, mask, scale=dim**-0.5)
    # Dropping the pad column entirely must give the same real-token outputs.
    mask_real = build_attention_mask([0], 0, seq - 1, seq - 1, torch.float32, "cpu")
    out_real = masked_attention(
        q[:, 1:], k[:, 1:], v[:, 1:], mask_real, scale=dim**-0.5
    )
    assert torch.allclose(out_padded[:, 1:], out_real, atol=1e-5)


def test_mlp_is_swiglu() -> None:
    cfg = _tiny_config()
    mlp = MLP(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_attention_projection_biases_present() -> None:
    # Qwen2 gotcha: QKV have bias, O does not (R2 §3).
    attn = Attention(_tiny_config())
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None


def test_sampler_greedy_is_argmax() -> None:
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    assert Sampler().sample(logits, temperature=0.0) == 1


def test_sampler_temperature_is_seeded() -> None:
    logits = torch.randn(50)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = Sampler().sample(logits, temperature=0.8, generator=g1)
    b = Sampler().sample(logits, temperature=0.8, generator=g2)
    assert a == b
