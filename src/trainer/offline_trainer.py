from typing import (
    Iterable,
    Union,
    Optional,
    List,
    Any,
    Mapping,
    Dict,
    Tuple,
    Literal,
    cast,
)
import os
from functools import partial

import numpy as np
import torch
import torch.utils
import lightning as L
from lightning.fabric import Fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.strategies.strategy import Strategy
import gymnasium as gym
import tqdm

from src.models.model import SequenceModellingPolicy


class OfflineRLTrainer:
    def __init__(
        self,
        num_epochs: int,
        max_steps: int,
        num_val_rollouts: int,
        num_test_rollouts: int,
        num_eval_steps: int,
        validation_frequency: int,
        num_parallel_envs: int = 1,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        ckpt_path: Optional[str] = None,
    ):
        self.fabric = Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )

        self.num_epochs = num_epochs  # Number of epochs to train
        self.max_steps = max_steps  # Maximum number of steps to train
        self.current_epoch = 0
        self.current_step = 0
        self.should_stop = False

        self.num_eval_steps = num_eval_steps  # Number of timesteps for each rollout
        self.num_val_rollouts = num_val_rollouts  # Number of rollouts for validation
        self.num_test_rollouts = num_test_rollouts  # Number of rollouts for testing
        self.num_parallel_envs = num_parallel_envs  # Number of parallel environments

        self.validation_frequency = validation_frequency  # Frequency of validation
        self._current_train_return: Optional[
            Union[torch.Tensor, Mapping[str, Any]]
        ] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        if not os.path.exist(ckpt_path):
            os.makedirs(ckpt_path)
        self.checkpoint_path = ckpt_path

    def fit(
        self,
        policy: SequenceModellingPolicy,
        train_dataloader: torch.utils.data.DataLoader,
        env: gym.Env,
        ckpt_path: Optional[str] = None,
    ):
        self.fabric.launch()
        if isinstance(self.fabric.strategy, L.fabric.strategies.fsdp.FSDPStrategy):
            # currently, there is no way to support fsdp with model.configure_optimizers in fabric
            # as it would require fabric to hold a reference to the model, which we don't want to.
            raise NotImplementedError("BYOT currently does not support FSDP")

        train_loader = self.fabric.setup_dataloaders(train_dataloader)
        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(
            policy.configure_optimizers()
        )
        model, optimizer = self.fabric.setup(policy, optimizer)
        env = gym.vector.SyncVectorEnv(env, self.num_parallel_envs)

        state = {"model": model, "optim": optimizer, "scheduler": scheduler_cfg}

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

                # check if we even need to train here
                if (
                    self.num_epochs is not None
                    and self.current_epoch >= self.num_epochs
                ):
                    self.should_stop = True

        while not self.should_stop:
            self.train_loop(
                policy=policy,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler_cfg=scheduler_cfg,
            )

            if self.current_epoch % self.validation_frequency == 0:
                self.env_rollout(policy, env, phase_is_test=(ckpt_path is not None))
                self.save(state)

            self.current_epoch += 1

            if self.num_epochs is not None and self.current_epoch >= self.num_epochs:
                self.should_stop = True

        self.should_stop = False

    def train_step(
        self,
        policy: SequenceModellingPolicy,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = policy.training_step(batch)

        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        self._current_train_return = loss
        return loss

    def train_loop(
        self,
        policy: SequenceModellingPolicy,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ] = None,
    ):
        self.fabric.call("on_train_epoch_start")

        iterable = self.progbar_wrapper(
            train_loader, total=len(train_loader), desc=f"Epoch {self.current_epoch}"
        )

        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)

            self.fabric.call("on_before_optimizer_step", optimizer)
            optimizer.step(partial(self.train_step, policy=policy, batch=batch))

            self.fabric.call("on_before_optimizer_zero_grad", optimizer)
            optimizer.zero_grad()

            self.fabric.call(
                "on_train_batch_end", self._current_train_return, batch, batch_idx
            )

            self.step_scheduler(
                policy, scheduler_cfg, level="step", current_value=self.current_step
            )

            self.current_step += 1
            if self.max_steps is not None and self.current_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end")

    def env_rollout(
        self,
        policy: SequenceModellingPolicy,
        env: gym.vector.SyncVectorEnv,
        phase_is_test: bool = False,
    ):
        policy.eval()
        torch.set_grad_enabled(False)
        self.fabric.call("on_validation_start")

        step_size = env.num_envs * self.fabric.world_size
        num_rollouts = (
            self.num_test_rollouts if phase_is_test else self.num_val_rollouts
        )
        assert num_rollouts % (env.num_envs * step_size) == 0

        total_rewards = np.zeros(env.num_envs)
        for i in self.probar_wrapper(
            range(num_rollouts, step=step_size), total=num_rollouts
        ):
            self.fabric.call("on_validation_batch_start")

            states, _ = torch.tensor(env.reset(), device=self.fabric.device)
            rewards = None
            for _ in range(self.num_eval_steps):
                actions = policy.validation_step((states, rewards))
                states_, rewards_, _, _ = env.step(actions.cpu().numpy())

                total_rewards[i : i + step_size] += rewards_

                states = torch.tensor(states_, device=self.fabric.device)
                rewards = torch.tensor(rewards_, device=self.fabric.device)

            self.fabric.call("on_validation_batch_end")

            mean, std = np.mean(total_rewards), np.std(total_rewards)
            result_dict = {"return": mean, "return_std": std}

            if not phase_is_test:
                self._current_val_return = result_dict
                prefix = "val_"
            else:
                prefix = "test_"

            self.fabric.log(name=prefix + "return", value=mean, sync_dist=True)
            self.fabric.log(name=prefix + "return_std", value=std, sync_dist=True)

        self.fabric.call("on_validation_end")
        policy.train()
        torch.set_grad_enabled(True)

    def probar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            model: The LightningModule to train
            scheduler_cfg: The learning rate scheduler configuration.
                Have a look at :meth:`lightning.pytorch.LightningModule.configure_optimizers` for supported values.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update(
                {"train_" + k: v for k, v in self._current_train_return.items()}
            )

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update(
                {"val_" + k: v for k, v in self._current_val_return.items()}
            )

        try:
            monitor = possible_monitor_vals[
                cast(Optional[str], scheduler_cfg["monitor"])
            ]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[
            Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]
        ],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(
                isinstance(_opt_cand, L.fabric.utilities.types.Optimizable)
                for _opt_cand in configure_optim_output
            ):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(
                        configure_optim_output[0]
                    )[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(
            os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"),
            state,
        )

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])
