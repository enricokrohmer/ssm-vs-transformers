from typing import List, Dict, Tuple

import torch
import numpy as np
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils


class RTGDataset(SequenceDataset):
    """Wrapper for RoboMimic datasets to support RTGs"""

    def __init__(
        self,
        hdf5_path: str,
        obs_keys: List[str],
        seq_length: int = 1,
        pad_seq: bool = True,
        use_hdf5_cache_mode: bool = False,
        hdf5_use_swmr: bool = True,
        hdf5_normalize_obs: bool = True,
    ):
        cache_mode = "low_dim" if use_hdf5_cache_mode else None
        self._initialize_obs_dict(obs_keys)

        super().__init__(
            hdf5_path=hdf5_path,
            obs_keys=obs_keys,
            dataset_keys=["actions", "rewards", "dones"],
            frame_stack=1,  # We do not support frame stapcking
            seq_length=seq_length,
            pad_frame_stack=False,  # We do not support frame stacking
            pad_seq_length=pad_seq,
            get_pad_mask=pad_seq,
            goal_mode=None,
            hdf5_cache_mode=cache_mode,
            hdf5_use_swmr=hdf5_use_swmr,
            hdf5_normalize_obs=hdf5_normalize_obs,
            filter_by_attribute=None,
            load_next_obs=False,
        )

    def _initialize_obs_dict(self, obs_keys):
        obs_dict = {
            "obs": {
                "low_dim": obs_keys,
            },
            "goal": {},
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_dict)

    def _calc_rtg_for_dataset(
        self,
        meta: dict,
        demo_id: int,
        index_in_demo: int,
        demo_length: int,
    ):
        # starting from this index, find all the future reward and dones
        future_info, _ = self.get_sequence_from_demo(
            demo_id, index_in_demo, keys=["rewards", "dones"], seq_length=demo_length
        )
        for t, d in enumerate(future_info["dones"]):
            if d:
                break
        # figure out which timestep of the demonstration leads to a success
        success_step = (
            index_in_demo + t + 1
        )  # extra +1 matches rew func in RobomimicRCGymWrapper
        meta["success_step"] = np.array([success_step], dtype=np.float32)
        future_rews = future_info["rewards"].copy()
        future_rews[t + 1 :] = 0.0
        # reverse cumsum
        rtgs = np.ascontiguousarray(np.flip(np.cumsum(np.flip(future_rews))))[
            :, np.newaxis
        ]
        meta["rtgs"] = rtgs[: self.seq_length]
        # include full demo return, which helps us pick the expert return at test-time
        if index_in_demo == 0:
            meta["full_demonstration_return"] = rtgs[0]
        else:
            # use nan as indicator to ignore this value in expert RTG calculation
            meta["success_step"] = np.full_like(meta["success_step"], np.nan)
            meta["full_demonstration_return"] = np.full_like(rtgs[0], np.nan)

    def _stack_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs_list = [obs_dict[key] for key in self.obs_keys]
        obs_list = np.concatenate(obs_list, axis=-1)
        return obs_list

    def get_sequence_from_demo(
        self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        assert num_frames_to_stack == 0  # RTGDataset does not support frame stacking
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo >= 0 and index_in_demo < demo_length

        # Beginning and end of a sequence
        seq_begin_idx = 0
        seq_end_idx = min(demo_length, index_in_demo + seq_length)

        seq_begin_pad = max(0, index_in_demo + seq_length - demo_length)

        if not self.pad_seq_length:
            seq_begin_pad == 0

        # Prepare sequence
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_idx:seq_end_idx]

        seq = TensorUtils.pad_sequence(
            seq, padding=(seq_begin_pad, 0), pad_same=True
        )  # We only pad at the beginning

        # Prepare mask
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_idx - seq_begin_idx))
        pad_mask = pad_mask[:, None].astype(bool)

        # Prepare timesteps
        end_i = min(index_in_demo + self.seq_length, demo_length)
        timesteps = np.arange(index_in_demo, end_i, dtype=np.float32)
        timesteps = TensorUtils.pad_sequence(
            timesteps, padding=(seq_begin_pad, 0), pad_same=True
        )  # We only pad at the beginning

        return seq, pad_mask, timesteps

    def get_item(self, index) -> Dict[str, torch.Tensor]:
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        index_in_demo = index - demo_start_index

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length,
        )

        # Calculate RTG
        self._calc_rtg_for_dataset(
            meta=meta,
            demo_id=demo_id,
            index_in_demo=index_in_demo,
            demo_length=demo_length,
        )

        # Get observations
        obs, masks, meta["timesteps"] = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            seq_length=self.seq_length,
            prefix="obs",
        )
        meta["obs"] = {k.split("/")[1]: obs[k] for k in obs}

        if self.hdf5_normalize_obs:
            meta["obs"] = ObsUtils.normalize_obs(
                meta["obs"], obs_normalization_stats=self.obs_normalization_stats
            )

        if self.get_pad_mask:
            meta["pad_mask"] = masks

        # Stack all oberservation modalities and convert to tensor
        meta["states"] = self._stack_obs(self, meta["obs"])
        for key in self.dataset_keys:
            meta[key] = torch.tensor(meta[key], dtype=torch.float32)

        return meta
