from typing import Dict

import torch
from torch import nn


class RvS(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        model_dim: int,
        num_layers: int = 2,  # As in the RvS paper
        action_tanh: bool = True,
        dropout: float = 0.0,
        use_bias: bool = False,
        max_length: int = None,
    ):
        super().__init__()
        assert num_layers >= 0, "Number of layers must be a non-negative integer"

        hidden = []
        for _ in range(num_layers):
            hidden.append(nn.Linear(model_dim, model_dim, use_bias=use_bias))
            hidden.append(nn.GELU())
            hidden.append(nn.Dropout(dropout))

        self.in_fc = nn.Linear(state_dim + 1, model_dim, use_bias=use_bias)
        self.net = nn.Sequential(*hidden)
        self.head = nn.Sequential(
            *([nn.Linear(model_dim, act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> torch.tensor:
        states = input_dict["seq"]
        rtgs = input_dict["rtgs"]        
        
        stacked_inputs = torch.cat([states, rtgs], dim=-1)

        x = self.in_fc(stacked_inputs)
        x = self.net(x)
        x = self.head(x)

        return x
