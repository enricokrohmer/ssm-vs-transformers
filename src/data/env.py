from typing import List, Callable

import numpy as np
import torch
import gymnasium as gym


class RTGEnvironment(gym.Env):
    obs: torch.Tensor
    actions: torch.Tensor
    rtgs: torch.Tensor
    current_timestep: int
    current_reward: float

    def __init__(
        self,
        env: gym.Env,
        base_rtg: torch.Tensor,
        wrappers: List[Callable[[gym.Env], gym.Env]],
        max_episode_len: int,
        context_len: int,
        is_robomimic: bool = False,
    ):
        if max_episode_len <= 0:
            raise ValueError("max_episode_steps must be positive")
        if 0 <= context_len or context_len > max_episode_len:
            raise ValueError(f"""context_len must be positive and less than max_episode_steps. 
                             Got context_len={context_len} and max_episode_len={max_episode_len}""")

        self.max_episode_steps = max_episode_len
        self.context_len = context_len
        self.is_robomimic = is_robomimic

        if not base_rtg.shape == ():
            raise ValueError(
                f"base_rtg must be a scalar or a 1D tensor. Got shape {base_rtg.shape}"
            )
        self.base_rtg = base_rtg.unsqueeze(0)

        # Initialise environment
        for wrapper in wrappers:
            env = wrapper(env)
        self.env = env

        self.reset()

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        self.obs = torch.from_numpy(obs).float().unsqueeze(0)
        self.actions = torch.empty(
            (0, self.env.action_space.shape[0]), dtype=torch.float32
        )
        self.rtgs = self.base_rtg
        self.current_reward = 0.0

        self.current_timestep = 0
        self.timesteps = torch.tensor([self.current_timestep], dtype=torch.float32)

    def render(self):
        self.env.render()

    def _concat_to_context(
        self, new_obs: np.array, new_act: torch.tensor, new_reward: float
    ):
        self.obs = torch.cat(
            (self.obs, torch.from_numpy(new_obs).float().unsqueeze(0)), dim=0
        )
        self.actions = torch.cat((self.actions, new_act.unsqueeze(0)), dim=0)

        current_rtg = (
            self.rtgs[-1] - torch.tensor([new_reward], dtype=torch.float32)
        ).to(self.rtgs.device)
        self.rtgs = torch.cat((self.rtgs, current_rtg), dim=0)

        self.timesteps = torch.cat(
            (
                self.timesteps,
                torch.tensor([self.current_timestep], dtype=torch.float32),
            ),
            dim=0,
        )

        if self.obs.shape[0] > self.context_len:
            self.obs = self.obs[1:]
            self.actions = self.actions[1:]
            self.rtgs = self.rtgs[1:]
            self.timesteps = self.timesteps[1:]

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self.timestep += 1
        obs, reward, done, _ = self.env.step(action.numpy())

        self._concat_to_context(obs, action, reward)
        self.current_reward += reward

        done = done or self.current_timestep >= self.max_episode_steps
        if self.is_robomimic:
            done = done or reward == 1.0

        return {
            "states": self.obs,
            "actions": self.actions,
            "rtgs": self.rtgs,
            "timesteps": self.timesteps,
            "done": done,
        }

    def close(self):
        self.env.close()

    def get_current_reward(self) -> float:
        return self.current_reward
