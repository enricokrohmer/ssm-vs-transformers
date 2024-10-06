from typing import Optional

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.networks.metaformer import (
    DecisionMetaformer,
    LayerNorm,
    PositionWiseFFN,
)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_head: int,
        dropout: float,
        use_bias: bool,
        block_size: int,
    ):
        super().__init__()
        assert model_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(model_dim, 3 * model_dim, bias=use_bias)
        # output projection
        self.c_proj = nn.Linear(model_dim, model_dim, bias=use_bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.model_dim = model_dim
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

        self.bias = self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.tensor, attention_mask: Optional[torch.tensor] = None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (model_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.model_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Construct causal mask
        mask = self.bias[:, :, :T, :T]

        if attention_mask is not None:
            assert attention_mask.size() == (
                B,
                T,
            ), f"Attention mask must be of size {(B, T)}"

            padding_mask = attention_mask.view(B, -1)
            padding_mask = padding_mask[:, None, None, :]
            mask = mask.bool() & padding_mask.bool()

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask.logical_not(), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_head: int,
        block_size: int,
        model_dim: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(model_dim, bias=bias)
        self.attn = CausalSelfAttention(model_dim, n_head, dropout, bias, block_size)
        self.ln_2 = LayerNorm(model_dim, bias=bias)
        self.mlp = PositionWiseFFN(model_dim, dropout, bias=bias)

    def forward(self, x: torch.tensor, attention_mask: Optional[torch.tensor] = None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class DecisionTransformer(DecisionMetaformer):
    def __init__(
        self,
        n_head: int,
        state_dim: int,
        act_dim: int,
        model_dim: int,
        num_layers: int,
        max_ep_len: int,
        max_length: int,
        action_tanh: bool = True,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        self.n_head = n_head
        self.max_ep_len = max_ep_len

        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            action_tanh=action_tanh,
            dropout=dropout,
            use_bias=use_bias,
            max_length=max_length,
        )

        self.embed_timestep = nn.Embedding(max_ep_len, model_dim)

    def _build_block(self) -> TransformerBlock:
        return TransformerBlock(
            n_head=self.n_head,
            block_size=self.max_length,
            model_dim=self.model_dim,
            dropout=self.dropout,
            bias=self.use_bias,
        )

    def init_weights(self):
        super().init_weights()
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )

    def forward(
        self,
        states: torch.tensor,
        actions: torch.tensor,
        rtgs: torch.tensor,
        timesteps: torch.tensor,
        masks: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        batch_size, seq_len = states.shape[0], states.shape[1]

        if masks is None:
            masks = torch.ones(
                batch_size, seq_len, dtype=torch.float, device=states.device
            )

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        rtg_embeddings = self.embed_return(rtgs) + time_embeddings

        stacked_masks = (
            torch.stack((masks, masks, masks), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_len)
        )
        stacked_inputs = (
            torch.stack([rtg_embeddings, state_embeddings, action_embeddings], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.model_dim)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        x = self.drop(stacked_inputs)
        for block in self.metaformer(x, attention_mask=stacked_masks):
            x = block(x, attention_mask=stacked_masks)
        x.reshape(batch_size, seq_len, 3, self.model_dim).permute(0, 2, 1, 3)

        action_preds = self.head(x[:, 1])
        return action_preds
