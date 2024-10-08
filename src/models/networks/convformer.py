from typing import Dict

import torch
from torch import nn

from src.models.networks.metaformer import (
    LayerNorm,
    PositionWiseFFN,
    DecisionMetaformer,
)


class Convolution(nn.Module):
    def __init__(self, window_size: int, model_dim: int):
        super().__init__()
        self.window_size = window_size

        self.rtg_conv1d = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=self.window_size,
            groups=model_dim,
        )
        self.obs_conv1d = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=self.window_size,
            groups=model_dim,
        )
        self.act_conv1d = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=self.window_size,
            groups=model_dim,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        window_size = self.window_size

        padded_tensor = torch.nn.functional.pad(
            x, (0, 0, window_size - 1, 0)
        ).transpose(1, 2)

        rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
        obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
        act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

        conv_tensor = torch.cat(
            (
                rtg_conv_tensor.unsqueeze(3),
                obs_conv_tensor.unsqueeze(3),
                act_conv_tensor.unsqueeze(3),
            ),
            dim=3,
        )
        conv_tensor = conv_tensor.reshape(
            conv_tensor.shape[0], conv_tensor.shape[1], -1
        )
        conv_tensor = conv_tensor.transpose(1, 2).to(x.device)

        return conv_tensor


class ConvFormerBlock(nn.Module):
    def __init__(
        self,
        window_size: int,
        model_dim: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(model_dim, bias=bias)
        self.conv = Convolution(window_size, model_dim)
        self.ln_2 = LayerNorm(model_dim, bias=bias)
        self.mlp = PositionWiseFFN(model_dim, dropout, bias=bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.conv(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class DecisionConvFormer(DecisionMetaformer):
    def __init__(
        self,
        window_size: int,
        state_dim: int,
        act_dim: int,
        model_dim: int,
        num_layers: int,
        max_ep_len: int,
        action_tanh: bool = True,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        self.window_size = window_size

        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            action_tanh=action_tanh,
            dropout=dropout,
            use_bias=use_bias,
            max_ep_len=max_ep_len,
        )

    def _build_block(self) -> nn.Module:
        return ConvFormerBlock(
            window_size=self.window_size,
            model_dim=self.model_dim,
            dropout=self.dropout,
            bias=self.use_bias,
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
    ) -> torch.tensor:
        states = input_dict["states"]
        batch_size, seq_len = states.shape[0], states.shape[1]

        state_embeddings, action_embeddings, rtg_embeddings = self._create_embeddings(
            input_dict
        )
        stacked_inputs = self._stack_embeddings(
            state_embeddings, action_embeddings, rtg_embeddings
        )

        x = self.embed_ln(stacked_inputs)
        x = self.drop(x)
        for block in self.metaformer:
            x = block(x)
        x.reshape(batch_size, seq_len, 3, self.model_dim).permute(0, 2, 1, 3)

        action_preds = self.head(x[:, 1])  # predict next action given state
        return action_preds
