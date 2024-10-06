from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.networks.metaformer import (
    DecisionMetaformer,
    LayerNorm,
    PositionWiseFFN,
)
from einops import rearrange, repeat


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=x.deivce)
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


@dataclass
class MambaWeightInitParams:
    A_init_range: Tuple[float, float] = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: Tuple[float, float] = (0.0, float("inf"))


class InferenceCache:
    ssm_state: torch.Tensor  # Shape (batch, n_heads, head_dim, h_state_dim)

    @staticmethod
    def alloc(
        nheads: int, headdim: int, d_state: int, batch_size: int, device: torch.device
    ):
        return InferenceCache(
            torch.zeros(batch_size, nheads, headdim, d_state, device=device),
        )


class MambaSSM(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        h_state_dim: int,
        expand: int,
        use_bias: bool,
        init_params: MambaWeightInitParams,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.h_state_dim = h_state_dim
        self.expand = expand
        self.chunk_size = chunk_size

        self.inner_dim = expand * model_dim
        assert (
            self.inner_dim % self.head_dim == 0
        ), "expand * model_dim must be divisible by num_heads"
        self.n_heads = self.inner_dim // self.head_dim

        # Order: x, B, C, dt
        in_proj_dim = self.inner_dim + 2 * h_state_dim + self.n_heads
        self.in_proj = nn.Linear(model_dim, in_proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.inner_dim, model_dim, bias=use_bias)

        # initialize A
        assert (
            init_params.A_init_range[0] > 0
            and init_params.A_init_range[1] > init_params.A_init_range[0]
        )
        A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(
            *init_params.A_init_range
        )
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # initialize dt
        dt = torch.exp(
            torch.rand(self.nheads)
            * (math.log(init_params.dt_max) - math.log(init_params.dt_min))
            + math.log(init_params.dt_min)
        )
        dt = torch.clamp(dt, min=init_params.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

    def forward(
        self, u: torch.Tensor, h: Optional[InferenceCache] = None
    ) -> Tuple[torch.Tensor, InferenceCache]:
        if h:
            return self.step(u, h)

        xBCdt = self.in_proj(u)
        x, B, C, dt = torch.split(
            xBCdt,
            [self.inner_dim, self.h_state_dim, self.h_state_dim, self.n_heads],
            dim=-1,
        )

        A = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.chunk_size,
        )

        state = InferenceCache(ssm_state)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.out_proj(y)

        return y, state

    def step(
        self, u: torch.Tensor, h: InferenceCache
    ) -> tuple[torch.Tensor, InferenceCache]:
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        xBCdt = self.in_proj(u.squeeze(1))
        xBC, dt = torch.split(
            xBCdt, [self.inner_dim + 2 * self.h_state_dim, self.n_heads], dim=-1
        )

        x, B, C = torch.split(
            xBC, [self.inner_dim, self.h_state_dim, self.h_state_dim], dim=-1
        )

        A = -torch.exp(self.A_log)
        dt = F.softplus(dt + self.dt_bias)
        dA = torch.exp(A * dt)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


class MambaFormerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        h_state_dim: int,
        expand: int,
        use_bias: bool,
        dropout: float,
        init_params: MambaWeightInitParams,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(model_dim, bias=use_bias)
        self.ssm = MambaSSM(
            model_dim=model_dim,
            head_dim=head_dim,
            h_state_dim=h_state_dim,
            expand=expand,
            use_bias=use_bias,
            init_params=init_params,
            chunk_size=chunk_size,
        )
        self.ln_2 = LayerNorm(model_dim, bias=use_bias)
        self.mlp = PositionWiseFFN(model_dim, dropout, bias=use_bias)

    def forward(
        self, x: torch.Tensor, h: Optional[InferenceCache] = None
    ) -> Tuple[torch.Tensor, InferenceCache]:
        x, h = x + self.ssm(self.ln_1(x), h)
        x = x + self.mlp(self.ln_2(x))
        return x, h


class DecisionMambaFormer(DecisionMetaformer):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        num_layers: int,
        model_dim: int,
        head_dim: int,
        h_state_dim: int,
        expand: int,
        use_bias: bool,
        dropout: float,
        action_tanh: float,
        learnable_init_state: bool,
        init_params: MambaWeightInitParams,
        chunk_size: int = 16,
    ):
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.h_state_dim = h_state_dim
        self.expand = expand
        self.learnable_init_state = learnable_init_state
        self.init_params = init_params
        self.chunk_size = chunk_size

        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            action_tanh=action_tanh,
            dropout=dropout,
            use_bias=use_bias,
        )

        self.caches = [None for _ in range(num_layers)]

    def _build_block(self) -> nn.Module:
        return MambaFormerBlock(
            model_dim=self.model_dim,
            head_dim=self.head_dim,
            h_state_dim=self.h_state_dim,
            expand=self.expand,
            use_bias=self.use_bias,
            dropout=self.dropout,
            learnable_init_state=self.learnable_init_state,
            init_params=self.init_params,
            chunk_size=self.chunk_size,
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
    ) -> torch.tensor:
        states = input_dict["states"]
        batch_size, seq_len = states.size(0), states.size(1)

        state_embeddings, action_embeddings, rtg_embeddings = self._create_embeddings(
            input_dict
        )
        stacked_inputs = self._stack_embeddings(
            state_embeddings, action_embeddings, rtg_embeddings
        )

        x = self.embed_ln(stacked_inputs)
        x = self.drop(x)

        for block in self.metaformer:
            x, _ = block(x)
        x.reshape(batch_size, seq_len, 3, self.model_dim).permute(0, 2, 1, 3)

        action_preds = self.head(x[:, 1])
        return action_preds

    def predict_action(self, input_dict: Dict[str, torch.Tensor]) -> torch.tensor:
        states = input_dict["states"]
        assert states.shape[1] == 1, "Only one token can be decoded per inference step"

        state_embeddings, action_embeddings, rtg_embeddings = self._create_embeddings(
            input_dict
        )
        stacked_inputs = self._stack_embeddings(
            state_embeddings, action_embeddings, rtg_embeddings
        )

        x = self.embed_ln(stacked_inputs)
        x = self.drop(x)

        for i, block in enumerate(self.metaformer):
            x, self.caches[i] = block(x, self.caches[i])

        action_preds = self.head(x)
        return action_preds
