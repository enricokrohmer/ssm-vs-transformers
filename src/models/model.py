from typing import Tuple, Callable, Iterator, Union, Dict
import copy

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from src.models.networks.metaformer import DecisionMetaformer
from src.models.networks.rvs import RvS

TensorTuple = Tuple[Tensor, Tensor, Tensor]
PolicyNetwork = Union[DecisionMetaformer, RvS]


class SequenceModellingPolicy(LightningModule):
    def __init__(
        self,
        policy_network: PolicyNetwork,
        optimizer: Callable[[Iterator[Tensor]], Optimizer],
        scheduler: Callable[[torch.optim.Optimizer], LRScheduler],
        context_len: int = 1,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="policy_network")

        # Init network and loss
        self.net = policy_network
        self.criterion_task = torch.nn.MSELoss()

        self.register_buffer("max_RTG", torch.zeros(1))

    def setup(self, stage: str):
        if self.hparams.compile:
            torch.compile(self.net)
            torch.compile(self.criterion_task)

    def on_train_start(self) -> None:
        self.net.init_weights()

    def training_step(self, batch: Dict[str, torch.tensor]) -> torch.Tensor:
        masks = batch["pad_mask"]

        # Update max_RTG for inference
        max_rtgs = batch["full_demonstration_return"]
        max_rtgs = max_rtgs[~torch.isnan(max_rtgs)]
        self.max_RTG = torch.max(torch.cat([self.max_RTG.unsqueeze(0), max_rtgs]))

        # Compute action
        action_target = copy.deepcopy(batch["actions"])
        action_preds = self.net(batch)

        # Compute loss
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[masks.reshape(-1).bool()]
        action_target = action_target.reshape(-1, act_dim)[masks.reshape(-1).bool()]

        loss = self.criterion_task(action_target, action_preds)
        self.log(
            name="task_loss",
            value=loss,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

        return loss
    
    def _predict_action(self, batch: Tuple[torch.Tensor, torch.Tensor | None]) -> torch.Tensor:
        states, rewards = batch
        num_envs = states.shape[1]
        context_len = self.hparams.get("context_len", 1)

        start_index = 0 if self._current_timestep < context_len else 1 # Offset for fixed context length
        
        if self._current_timestep == 0:
            self.rtgs = self.rtgs = self.max_RTG * torch.ones(1, num_envs, device=states.device) # (1, num_envs)
            self.states = states.unsqueeze(1) # (num_envs, 1, state_dim)
            
            act_dim = self.net.action_dim
            self.actions = torch.empty(num_envs, 0, act_dim, device=states.device)            
        else:
            if rewards is None:
                raise ValueError("Rewards must be provided after the first timestep")
            
            self.states = torch.cat((self.states[:, start_index:-1], states.unsqueeze(1)), dim=1)
            self.rtgs = torch.cat([self.rtgs[:, start_index:-1], self.rtgs[:, -1] - rewards,])
        
        start_time = max(self._current_timestep - context_len, 0)
        timesteps = torch.tensor(
            list(range(start_time, self._current_timestep + 1)), device=states.device, dtype=states.dtype
        ).repeat(num_envs, 1)
        
        input_dict = {
            "states": self.states,
            "actions": self.actions,
            "rtgs": self.rtgs,
            "timesteps": timesteps,
        }
        
        action_pred = self.net(input_dict)
        self.actions = torch.cat((self.actions[:, start_index:-1], action_pred.unsqueeze(1)), dim=1)
        
        return action_pred
        
    def on_validation_batch_start(self, batch: copy.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.current_timestep = 0
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor | None], *args, **kwargs) -> Tensor:
        return self._predict_action(batch)
        
    def on_test_batch_start(self, batch: copy.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.current_timestep = 0
        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor | None], *args, **kwargs):
        return self._predict_action(batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.hparams.optimizer(self.net.parameters())
        scheduler = self.hparams.scheduler(optimizer)

        return [optimizer], [scheduler]
