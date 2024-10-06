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

TensorTuple = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
PolicyNetwork = Union[DecisionMetaformer, RvS]


class SequenceModellingPolicy(LightningModule):
    def __init__(
        self,
        policy_network: PolicyNetwork,
        optimizer: Callable[[Iterator[Tensor]], Optimizer],
        scheduler: Callable[[torch.optim.Optimizer], LRScheduler],
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
        self.log(
            max_rtgs,
            name="max_RTG",
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

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

    def validation_step(self, batch: TensorTuple) -> torch.Tensor:
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.hparams.optimizer(self.net.parameters())
        scheduler = self.hparams.scheduler(optimizer)

        return [optimizer], [scheduler]
