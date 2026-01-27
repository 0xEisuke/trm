"""Implementation of the Generative Tiny Recursion Model (TRM)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TRMConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    latent_steps: int = 2
    deep_steps: int = 2
    supervision_steps: int = 2
    act_threshold: float = 0.95
    act_weight: float = 1.0
    pad_token_id: int = 0
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, data: Dict) -> "TRMConfig":
        return cls(**data)

    def to_dict(self) -> Dict:
        return asdict(self)


class TinyBlock(nn.Module):
    """Small Transformer block used across recursive updates."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = attn_mask == 0
        
        # Create causal mask to prevent attending to future positions
        seq_len = h.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device), diagonal=1).bool()
        
        attn_out, _ = self.attn(
            h, h, h, 
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
            need_weights=False
        )
        h = self.ln1(h + self.dropout(attn_out))
        ff_out = self.ffn(h)
        h = self.ln2(h + ff_out)
        return h


class TinyNet(nn.Module):
    """Shared tiny network that processes concatenated (x,y,z) inputs."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.xyz_proj = nn.Linear(d_model * 3, d_model)
        self.yz_proj = nn.Linear(d_model * 2, d_model)
        self.block = TinyBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward_xyz(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = torch.cat([x, y, z], dim=-1)
        h = self.xyz_proj(h)
        return self.block(h, attn_mask=attn_mask)

    def forward_yz(self, y: torch.Tensor, z: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = torch.cat([y, z], dim=-1)
        h = self.yz_proj(h)
        return self.block(h, attn_mask=attn_mask)


class TRMModel(nn.Module):
    """Main Generative TRM model."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.tiny_net = TinyNet(config.d_model, config.n_heads, dropout=config.dropout)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output_head.weight = self.token_embed.weight
        self.q_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))

    def _add_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tokens.size()
        self._ensure_positional_capacity(seq_len)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, seq_len)
        pos_embed = self.pos_embed(positions)
        return self.dropout(tokens + pos_embed)

    def _ensure_positional_capacity(self, seq_len: int) -> None:
        if seq_len <= self.pos_embed.num_embeddings:
            return
        old_embed = self.pos_embed
        new_embed = nn.Embedding(seq_len, self.config.d_model)
        new_embed.weight.data[: old_embed.num_embeddings] = old_embed.weight.data
        new_embed.weight.data[old_embed.num_embeddings :] = old_embed.weight.data[-1].unsqueeze(0)
        self.pos_embed = new_embed.to(old_embed.weight.device)
        self.config.max_seq_len = seq_len

    def _init_states(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = x.clone()
        z = torch.zeros_like(x)
        return y, z

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.config.latent_steps):
            z = self.tiny_net.forward_xyz(x, y, z, attention_mask)
            y = self.tiny_net.forward_yz(y, z, attention_mask)
        return y, z

    def _deep_recursion_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single deep recursion block with detach for the next supervision step."""
        y_state, z_state = y, z
        tracked_y = None
        tracked_z = None
        for depth in range(self.config.deep_steps):
            use_grad = depth == self.config.deep_steps - 1
            if use_grad:
                y_state, z_state = self.latent_recursion(x, y_state, z_state, attention_mask)
                tracked_y, tracked_z = y_state, z_state
            else:
                with torch.no_grad():
                    y_state, z_state = self.latent_recursion(x, y_state, z_state, attention_mask)
        assert tracked_y is not None and tracked_z is not None
        return tracked_y, tracked_z, tracked_y.detach(), tracked_z.detach()

    @staticmethod
    def _pool_for_q(y: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return y.mean(dim=1)
        mask = attention_mask.float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = torch.sum(y * mask.unsqueeze(-1), dim=1) / denom
        return pooled

    def _compute_act_labels(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets) | (attention_mask == 0)
            per_sequence = correct.all(dim=1).float()
            if threshold < 1.0:
                token_acc = ((predictions == targets).float() * attention_mask.float()).sum(dim=1)
                token_acc /= attention_mask.sum(dim=1).clamp(min=1)
                per_sequence = (token_acc >= threshold).float()
        return per_sequence.unsqueeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        supervision_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        attention_mask = attention_mask if attention_mask is not None else (input_ids != self.config.pad_token_id).long()
        token_embeddings = self.token_embed(input_ids)
        x = self._add_positional_encoding(token_embeddings)
        y, z = self._init_states(x)

        sup_steps = supervision_steps or self.config.supervision_steps
        logits_per_step: List[torch.Tensor] = []
        q_hats: List[torch.Tensor] = []
        ce_losses: List[torch.Tensor] = []
        act_losses: List[torch.Tensor] = []

        for _ in range(sup_steps):
            y_grad, _, y, z = self._deep_recursion_step(x, y, z, attention_mask)
            logits = self.output_head(y_grad)
            logits_per_step.append(logits)
            pooled = self._pool_for_q(y_grad, attention_mask)
            q_hat = torch.sigmoid(self.q_head(pooled))
            q_hats.append(q_hat)

            if target_ids is not None:
                ce = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=self.config.pad_token_id,
                )
                ce_losses.append(ce)
                labels = self._compute_act_labels(
                    logits,
                    target_ids,
                    attention_mask,
                    threshold=self.config.act_threshold,
                )
                bce = F.binary_cross_entropy(q_hat, labels)
                act_losses.append(bce)

        loss = None
        ce_loss = torch.stack(ce_losses).mean() if ce_losses else None
        act_loss = torch.stack(act_losses).mean() if act_losses else None
        if ce_loss is not None and act_loss is not None:
            loss = ce_loss + self.config.act_weight * act_loss
        elif ce_loss is not None:
            loss = ce_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "act_loss": act_loss,
            "logits": logits_per_step[-1] if logits_per_step else None,
            "all_logits": logits_per_step,
            "q_hats": q_hats,
        }

    def _inference_cycle(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        attention_mask: torch.Tensor,
        supervision_steps: Optional[int],
        q_threshold: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sup_steps = supervision_steps or self.config.supervision_steps
        q_hat = torch.zeros(x.size(0), device=x.device, dtype=torch.float)
        for _ in range(sup_steps):
            for _ in range(self.config.deep_steps):
                y, z = self.latent_recursion(x, y, z, attention_mask)
            pooled = self._pool_for_q(y, attention_mask)
            q_hat = torch.sigmoid(self.q_head(pooled)).squeeze(-1)
            if q_threshold is not None and torch.all(q_hat > q_threshold):
                break
        return y, z, q_hat

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        supervision_steps: Optional[int] = None,
        q_threshold: Optional[float] = 0.7,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        device = device or next(self.parameters()).device
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            attention_mask = (input_ids != self.config.pad_token_id).long()
            token_embeddings = self.token_embed(input_ids)
            x = self._add_positional_encoding(token_embeddings)
            y, z = self._init_states(x)
            y, z, _ = self._inference_cycle(x, y, z, attention_mask, supervision_steps, q_threshold)
            logits = self.output_head(y)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for input_id in set(input_ids.squeeze(0).tolist()):
                    next_token_logits[0, input_id] /= repetition_penalty
            
            next_token = _sample_logits(next_token_logits, top_k=top_k, top_p=top_p)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            if self.config.pad_token_id in next_token.tolist():
                break

        return input_ids.squeeze(0).tolist()


def _sample_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Sample tokens with guard rails against empty/invalid probability vectors."""
    squeeze_output = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_output = True

    vocab = logits.size(-1)

    if top_k > 0:
        k = min(top_k, vocab)
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    zero_rows = probs_sum.squeeze(-1) == 0

    if zero_rows.any():
        fallback_indices = torch.argmax(logits[zero_rows], dim=-1)
        fallback_one_hot = F.one_hot(fallback_indices, num_classes=vocab).float()
        probs[zero_rows] = fallback_one_hot
        probs_sum = probs.sum(dim=-1, keepdim=True)

    probs = probs / probs_sum.clamp(min=1e-12)
    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
    if squeeze_output:
        samples = samples.squeeze(0)
    return samples
