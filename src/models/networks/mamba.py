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


def ssd(x, A, B, C, chunk_size, initial_states: Optional[torch.Tensor] = None):
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
    if initial_states is None:
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
    pass


class MambaSSM(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        h_state_dim: int,
        expand: int,
        use_bias: bool,
        init_params: MambaWeightInitParams,
        learnable_init_state: bool = False,
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

        self.init_states = None
        if learnable_init_state:
            self.init_states = nn.Parameter(
                torch.zeros(
                    self.n_heads,
                    self.head_dim,
                    self.hi_state_dim,
                )
            )
            self.init_states._no_weight_decay = True

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
            self.init_states,
        )

        state = InferenceCache(ssm_state)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.out_proj(y)

        return y, state

    def step(
        self, u: torch.Tensor, h: InferenceCache
    ) -> tuple[torch.Tensor, InferenceCache]:
        pass


class MambaFormerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        h_state_dim: int,
        expand: int,
        use_bias: bool,
        dropout: float,
        learnable_init_state: bool,
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
            learnable_init_state=learnable_init_state,
            init_params=init_params,
            chunk_size=chunk_size,
        )
        self.ln_2 = LayerNorm(model_dim, bias=use_bias)
        self.mlp = PositionWiseFFN(model_dim, dropout, bias=use_bias)

    def forward(
        self, x: torch.Tensor, h: Optional[InferenceCache] = None
    ) -> tuple[torch.Tensor, InferenceCache]:
        x, h = x + self.ssm(self.ln_1(x), h)
        x = x + self.mlp(self.ln_2(x))
        return x, h


class DecisionMambaFormer(DecisionMetaformer):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        num_layers: int,
        max_length: int,
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
            max_length=max_length,
        )

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
        states = input_dict["seq"]
        actions = input_dict["act"]
        rtgs = input_dict["rtgs"]
        
        batch_size, seq_len = states.size(0), states.size(1)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        rtg_embeddings = self.embed_return(rtgs)

        stacked_inputs = (
            torch.stack([rtg_embeddings, state_embeddings, action_embeddings], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.model_dim)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)
        x = self.drop(stacked_inputs)

        for block in self.metaformer:
            x, _ = block(x)
        x.reshape(batch_size, seq_len, 3, self.model_dim).permute(0, 2, 1, 3)

        action_preds = self.head(x[:, 1])
        return action_preds
