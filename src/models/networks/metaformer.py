"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc = nn.Linear(model_dim, 4 * model_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * model_dim, model_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DecisionMetaformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        model_dim: int,
        num_layers: int,
        action_tanh: bool = True,
        dropout: float = 0.0,
        use_bias: bool = False,
        max_ep_len: Optional[int] = None,
    ):
        super().__init__()
        # Saving hyperparameters
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.action_tanh = action_tanh
        self.dropout = dropout
        self.use_bias = use_bias

        # Building the model
        self.drop = nn.Dropout(dropout)
        self.metaformer = nn.ModuleList(
            [self._build_block() for _ in range(num_layers)]
        )

        self.embed_return = nn.Embedding(1, model_dim)
        self.embed_state = torch.nn.Linear(state_dim, model_dim)
        self.embed_action = torch.nn.Linear(act_dim, model_dim)
        self.embed_timestep = (
            nn.Embedding(max_ep_len, model_dim) if max_ep_len else None
        )
        self.embed_ln = LayerNorm(model_dim, bias=use_bias)

        self.head = nn.Sequential(
            *([nn.Linear(model_dim, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_weights(self):
        self.apply(self._init_weights)

    def _build_block(self) -> nn.Module:
        raise NotImplementedError

    def _create_embeddings(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        states = self.embed_state(input_dict["states"])
        actions = self.embed_action(input_dict["action"])
        rtgs = self.embed_return(input_dict["rtg"])

        if self.embed_timestep is not None:
            timesteps = self.embed_timestep(input_dict["timesteps"])
            states += timesteps
            actions += timesteps
            rtgs += timesteps

        return states, actions, rtgs

    def _stack_embeddings(self, states, actions, rtgs):
        batch_size, seq_len = states.shape[0], states.shape[1]
        stacked_inputs = (
            torch.stack([rtgs, states, actions], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.model_dim)
        )

        return stacked_inputs

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> torch.tensor:
        raise NotImplementedError

    def predict_action(self, input_dict: Dict[str, torch.Tensor]) -> torch.tensor:
        state_embs, action_embs, rtg_embs = self._create_embeddings(input_dict)
        stacked_inputs = self._stack_embeddings(state_embs, action_embs, rtg_embs)

        x = self.embed_ln(stacked_inputs)
        x = self.drop(x)
        for block in self.metaformer:
            x = block(x)

        action_pred = self.head(x[:, -1]).squeeze(1)
        return action_pred
