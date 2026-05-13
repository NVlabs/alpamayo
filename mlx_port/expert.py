"""
MLX implementation of the Alpamayo action expert (diffusion denoising network).

This is a decoder-only transformer (Qwen3-style) that:
- Has no embed_tokens (actions are projected via action_in_proj)
- Uses RMSNorm on Q/K
- Uses SwiGLU MLP
- Supports non-causal attention when used with VLM KV cache

The structure is defined so that model.load_weights() succeeds on the
416 tensors from Alpamayo-R1-10B.
"""

from typing import Optional, List
import math

import mlx.core as mx
import mlx.nn as nn


# -----------------------------------------------------------------------------
# Helper modules (MLX ports of the original PyTorch modules)
# -----------------------------------------------------------------------------

class MLXFourierEncoderV2(nn.Module):
    """MLX port of FourierEncoderV2 (log-spaced frequencies)."""

    def __init__(self, dim: int, max_freq: float = 100.0):
        super().__init__()
        half = dim // 2
        # logspace frequencies (will be loaded from checkpoint)
        self.freqs = mx.zeros((1, half))   # placeholder; load_weights will overwrite

    def __call__(self, x):
        arg = x[..., None] * self.freqs * 2 * math.pi
        return mx.concatenate([mx.sin(arg), mx.cos(arg)], axis=-1) * math.sqrt(2)


class MLXMLPEncoder(nn.Module):
    """MLX port of MLPEncoder (used inside action_in_proj.encoder)."""

    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        self.outdim = outdim
        # Replicate the exact layer ordering from the original PyTorch code
        # so that parameter indices (0,2,3,5,6,...) match the checkpoint.
        enc_layers = [
            nn.Linear(num_input_feats, hidden_size, bias=True),  # 0
            nn.SiLU(),                                           # 1
        ]
        for i in range(num_enc_layers):
            if i < num_enc_layers - 1:
                enc_layers.extend([
                    nn.RMSNorm(hidden_size, eps=1e-5),           # 2
                    nn.Linear(hidden_size, hidden_size, bias=True),  # 3
                    nn.SiLU(),                                   # 4
                ])
            else:
                enc_layers.extend([
                    nn.RMSNorm(hidden_size, eps=1e-5),           # 5
                    nn.Linear(hidden_size, outdim, bias=True),   # 6
                ])

        self.trunk = enc_layers  # list so parameter names become trunk.0.weight etc.

    def __call__(self, x):
        for layer in self.trunk:
            x = layer(x)
        return x


class MLXPerWaypointActionInProjV2(nn.Module):
    """
    MLX port of PerWaypointActionInProjV2.
    We only need the submodule tree to match the checkpoint keys.
    """

    def __init__(self, num_fourier_feats: int = 20, max_freq: float = 100.0,
                 num_enc_layers: int = 2, hidden_size: int = 512, out_dim: int = 2048):
        super().__init__()
        # From the checkpoint we see sinus.0 and sinus.1 → 2 action dimensions
        self.sinus = [
            MLXFourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
            for _ in range(2)
        ]
        self.timestep_fourier_encoder = MLXFourierEncoderV2(
            dim=num_fourier_feats, max_freq=max_freq
        )
        self.encoder = MLXMLPEncoder(
            num_input_feats=60,   # 20*2 + 20 (empirical from checkpoint)
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)

    def __call__(self, x, timesteps):
        B, T, _ = x.shape
        x = x.astype(mx.float32)
        timesteps = timesteps.astype(mx.float32)

        # Fourier-encode each action dimension separately
        action_feats_list = [s(x[:, :, i]) for i, s in enumerate(self.sinus)]
        action_feats = mx.concatenate(action_feats_list, axis=-1)  # (B, T, 40)

        # Timestep encoding (use last element of timesteps tensor, repeat across waypoints)
        ts_last = timesteps[..., -1]  # (B,)
        timestep_feats = self.timestep_fourier_encoder(ts_last)  # (B, 20)
        timestep_feats = mx.repeat(timestep_feats[:, None, :], T, axis=1)  # (B, T, 20)

        combined = mx.concatenate((action_feats, timestep_feats), axis=-1)  # (B, T, 60)
        flat = combined.reshape(-1, combined.shape[-1])  # (B*T, 60)
        encoded = self.encoder(flat)  # (B*T, out_dim)
        encoded = encoded.reshape(B, T, -1)
        return self.norm(encoded)


# -----------------------------------------------------------------------------
# Expert Layer & Expert
# -----------------------------------------------------------------------------

class ExpertLayer(nn.Module):
    """Single decoder layer for the action expert."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 8256,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim

        # Attention (GQA support)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Q/K norms
        self.self_attn.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.self_attn.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        # MLP (SwiGLU)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Layer norms
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        past_key_values: tuple[mx.array, mx.array] | None = None,
        position_ids: mx.array | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        Forward pass for one Expert decoder layer with GQA + RoPE support.
        Returns (hidden_states, new_key_value) when use_cache=True.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # --- Self-Attention (GQA with prompt KV cache) ---
        B, T, _ = hidden_states.shape

        # 1. Linear projections (native KV head count from the checkpoint)
        q = self.self_attn.q_proj(hidden_states)
        k = self.self_attn.k_proj(hidden_states)
        v = self.self_attn.v_proj(hidden_states)

        # 2. Reshape to (B, T, num_heads, head_dim) / (B, T, num_kv_heads, head_dim)
        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        q = self.self_attn.q_norm(q)
        k = self.self_attn.k_norm(k)

        # 3. Apply rotary embeddings while still in (B, T, H, D) layout
        if position_ids is not None:
            q, k = self._apply_rope(q, k, position_ids)

        # ------------------------------------------------------------------
        # 4. GQA head expansion + KV-cache concatenation (canonical pattern)
        # ------------------------------------------------------------------
        # Both the prompt cache and the current tokens are stored with the
        # native number of KV heads (self.num_kv_heads).  For the attention
        # computation we must expand them to the number of query heads
        # (self.num_heads) using head repetition.
        #
        # The repetition MUST be applied to BOTH sides before they are
        # concatenated; otherwise the concatenate sees mismatched head counts.
        repeat_factor = self.num_heads // self.num_kv_heads   # 16 // 8 = 2

        # Transpose to (B, H, T, D) so the head dimension becomes axis 1.
        q = q.transpose(0, 2, 1, 3)   # (B, H_q, T, D)
        k = k.transpose(0, 2, 1, 3)   # (B, H_kv, T, D)
        v = v.transpose(0, 2, 1, 3)   # (B, H_kv, T, D)

        # 4b. Expand both the prompt cache and the current tokens to query-head count
        if past_key_values is not None:
            past_k, past_v = past_key_values
            past_k = mx.repeat(past_k, repeat_factor, axis=1)   # (B, H_q, T, D)
            past_v = mx.repeat(past_v, repeat_factor, axis=1)
            k = mx.repeat(k, repeat_factor, axis=1)             # current tokens also expanded
            v = mx.repeat(v, repeat_factor, axis=1)
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)
        else:
            k = mx.repeat(k, repeat_factor, axis=1)
            v = mx.repeat(v, repeat_factor, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn_output = attn_weights @ v

        # Merge heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, self.hidden_size)
        attn_output = self.self_attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # --- MLP (SwiGLU) ---
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        gate = self.mlp.gate_proj(hidden_states)
        up = self.mlp.up_proj(hidden_states)
        hidden_states = (gate * nn.silu(up)) @ self.mlp.down_proj.weight.T

        hidden_states = residual + hidden_states

        # Return updated KV cache if requested
        if use_cache:
            new_k = k[:, :, -T:, :]  # only the new tokens
            new_v = v[:, :, -T:, :]
            return hidden_states, (new_k, new_v)

        return hidden_states

    def _apply_rope(self, q, k, position_ids):
        """
        Apply rotary position embeddings (RoPE) to query and key tensors.

        Supports GQA: RoPE is applied to the original (smaller) k tensor before
        repeating KV heads to match query heads.
        """
        # q: (B, T, num_heads, head_dim)
        # k: (B, T, num_kv_heads, head_dim)
        # position_ids: (B, T) or (T,)

        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]

        B, T = position_ids.shape
        head_dim = q.shape[-1]

        # Standard RoPE inverse frequencies
        inv_freq = 1.0 / (10000 ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))

        # Compute angles
        t = position_ids.astype(mx.float32)  # (B, T)
        freqs = mx.outer(t, inv_freq)        # (B, T, head_dim//2)
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (B, T, head_dim)

        cos = mx.cos(emb)[..., None, :]  # (B, T, 1, head_dim)
        sin = mx.sin(emb)[..., None, :]

        def rotate_half(x):
            """Rotate the second half of the last dimension."""
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return mx.concatenate([-x2, x1], axis=-1)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        return q, k


class Expert(nn.Module):
    """
    Alpamayo action expert (diffusion denoising transformer).

    The submodule structure is defined to exactly match the 416 tensors
    stored in Alpamayo-R1-10B so that load_weights() succeeds cleanly.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 8256,
        num_layers: int = 36,
    ):
        super().__init__()

        self.layers = [
            ExpertLayer(hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size)
            for _ in range(num_layers)
        ]

        # Top-level norm present in checkpoint
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)

        # === Action projection modules (exact structure from Alpamayo) ===
        # We attach real modules so that create_expert_step_fn can use them directly.
        self.action_in_proj = MLXPerWaypointActionInProjV2(
            num_fourier_feats=20,
            max_freq=100.0,
            num_enc_layers=2,
            hidden_size=512,
            out_dim=hidden_size,
        )

        # Final projection from hidden_size → action_dim (usually 2 for accel/curvature)
        self.action_out_proj = nn.Linear(hidden_size, 2, bias=True)

        # Action space stats loaded from Alpamayo config.json (not hardcoded)
        import json
        from pathlib import Path
        cfg_path = Path("pre-trained/Alpamayo-R1-10B/config.json")
        if cfg_path.exists():
            with open(cfg_path) as f:
                alp_cfg = json.load(f)
            asc = alp_cfg.get("action_space_cfg", {})
            accel_mean = asc.get("accel_mean", 0.02902694707164455)
            accel_std = asc.get("accel_std", 0.6810426736454882)
            curv_mean = asc.get("curvature_mean", 0.0002692167976330542)
            curv_std = asc.get("curvature_std", 0.026148280660833106)
        else:
            accel_mean = 0.02902694707164455
            accel_std = 0.6810426736454882
            curv_mean = 0.0002692167976330542
            curv_std = 0.026148280660833106

        self.action_space = nn.Module()
        self.action_space.accel_mean = mx.array(accel_mean)
        self.action_space.accel_std = mx.array(accel_std)
        self.action_space.curvature_mean = mx.array(curv_mean)
        self.action_space.curvature_std = mx.array(curv_std)

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: mx.array | None = None,
        past_key_values: list | None = None,
        attention_mask: mx.array | None = None,
        use_cache: bool = False,
        is_causal: bool = True,
        **kwargs,
    ):
        """
        Forward pass for the full Expert model.

        Supports:
        - past_key_values: list of (k, v) tuples, one per layer
        - non-causal attention_mask (for diffusion tokens attending to VLM prompt)
        """
        hidden_states = inputs_embeds
        new_past_key_values = [] if use_cache else None

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values[i],
                position_ids=position_ids,
                use_cache=use_cache,
                is_causal=is_causal,
            )

            if use_cache:
                hidden_states, new_kv = layer_outputs
                new_past_key_values.append(new_kv)
            else:
                hidden_states = layer_outputs

        if use_cache:
            return hidden_states, new_past_key_values

        return hidden_states


# -----------------------------------------------------------------------------
# Step function factory (used by FlowMatching)
# -----------------------------------------------------------------------------

def create_expert_step_fn(
    expert: "Expert",
    prompt_cache,                 # past_key_values from VLM (list of (k, v) tuples)
    position_ids: mx.array,
    attention_mask: mx.array,
    n_diffusion_tokens: int,
    prefill_seq_len: int,
):
    """
    Create a step_fn compatible with FlowMatching.sample() using the real
    action_in_proj / action_out_proj attached to the expert.
    """
    action_in_proj = expert.action_in_proj
    action_out_proj = expert.action_out_proj

    def step_fn(x: mx.array, t: mx.array) -> mx.array:
        """
        x: (B*, n_waypoints, action_dim) or (B*, n_waypoints * action_dim)
        t: (B*, 1) or broadcastable
        """
        b_star = x.shape[0]

        # Handle flat input
        if x.ndim == 2:
            x = x.reshape(b_star, n_diffusion_tokens, -1)

        # Project noisy actions + timesteps into expert embeddings
        future_token_embeds = action_in_proj(x, t)  # (b*, n_diffusion_tokens, hidden_size)

        # Run expert with (adapted) cached VLM context (non-causal)
        expert_out = expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            is_causal=False,
        )

        last_hidden = expert_out[0] if isinstance(expert_out, tuple) else expert_out
        last_hidden = last_hidden[:, -n_diffusion_tokens:, :]

        # Project back to action space
        pred = action_out_proj(last_hidden)           # (b*, n_diffusion_tokens, 2)
        pred = pred.reshape(b_star, -1)               # flat
        return pred

    return step_fn


# -----------------------------------------------------------------------------
# Helper for building non-causal attention inputs (used when calling the expert
# on top of a VLM prompt cache)
# -----------------------------------------------------------------------------

def denormalize_trajectory(raw_action: mx.array, expert: "Expert") -> mx.array:
    """
    Convert the raw model output (normalized accel/curvature) back to physical units
    using the action_space statistics stored in the expert.

    This matches the logic in UnicycleAccelCurvatureActionSpace.action_to_traj.
    """
    accel_mean = expert.action_space.accel_mean
    accel_std = expert.action_space.accel_std
    curv_mean = expert.action_space.curvature_mean
    curv_std = expert.action_space.curvature_std

    accel = raw_action[..., 0] * accel_std + accel_mean
    curv = raw_action[..., 1] * curv_std + curv_mean
    return mx.stack([accel, curv], axis=-1)


def build_expert_attention_inputs(
    prompt_length: int,
    num_diffusion_tokens: int,
    device: str = "cpu",
    future_start_offset: int | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Build position_ids and attention_mask for the diffusion expert.

    This mirrors the logic in NVIDIA's AlpamayoR1.step_fn.

    Args:
        prompt_length: Number of tokens in the VLM prompt (including CoC + discrete tokens).
        num_diffusion_tokens: Number of future waypoints (usually 64).
        future_start_offset: Optional position of the <|traj_future_start|> token.
                             If None, we assume diffusion tokens start right after the prompt.

    Returns:
        position_ids: (1, num_diffusion_tokens)
        attention_mask: (1, 1, num_diffusion_tokens, prompt_length + num_diffusion_tokens)
                        with -inf for positions the diffusion tokens should not attend to.
    """
    total_len = prompt_length + num_diffusion_tokens

    # Position IDs for diffusion tokens (shifted by prompt length)
    if future_start_offset is not None:
        start_pos = future_start_offset + 1
    else:
        start_pos = prompt_length

    position_ids = mx.arange(start_pos, start_pos + num_diffusion_tokens)[None, :]

    # Build attention mask
    # Diffusion tokens can attend to the entire prompt + all other diffusion tokens (non-causal)
    attn_mask = mx.zeros((1, 1, num_diffusion_tokens, total_len), dtype=mx.float32)

    # For simplicity in the first version, we allow full attention.
    # In the real implementation we will mask out positions after the current diffusion token
    # if we want to keep some causality within the diffusion block.
    # For now we keep it fully non-causal (as NVIDIA does with is_causal=False).

    return position_ids, attn_mask
