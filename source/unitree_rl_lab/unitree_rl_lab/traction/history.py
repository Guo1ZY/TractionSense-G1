"""Reset-safe time-major history used by Student and deployment."""

from __future__ import annotations

import torch


class TemporalHistoryBuffer:
    """Vectorized ``[environment, time, feature]`` history.

    Time is always oldest to newest. Reset clears every slot for the selected
    environments; an optional reset sample is written only to the newest slot,
    never duplicated across the entire history.
    """

    def __init__(
        self,
        num_envs: int,
        history_frames: int,
        feature_dim: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(num_envs, history_frames, feature_dim) <= 0:
            raise ValueError("history dimensions must be positive")
        self.num_envs = num_envs
        self.history_frames = history_frames
        self.feature_dim = feature_dim
        self.data = torch.zeros(
            (num_envs, history_frames, feature_dim),
            device=device,
            dtype=dtype,
        )
        self.filled = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        initial: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.data.device)
        else:
            env_ids = env_ids.to(device=self.data.device, dtype=torch.long)
        self.data[env_ids] = 0.0
        self.filled[env_ids] = 0
        if initial is not None:
            initial = initial.to(device=self.data.device, dtype=self.data.dtype)
            if initial.shape != (env_ids.numel(), self.feature_dim):
                raise ValueError(
                    f"initial shape {tuple(initial.shape)}, expected "
                    f"{(env_ids.numel(), self.feature_dim)}"
                )
            self.data[env_ids, -1] = torch.nan_to_num(initial)
            self.filled[env_ids] = 1

    def append(self, value: torch.Tensor) -> None:
        if value.shape != (self.num_envs, self.feature_dim):
            raise ValueError(
                f"value shape {tuple(value.shape)}, expected "
                f"{(self.num_envs, self.feature_dim)}"
            )
        self.data = torch.roll(self.data, shifts=-1, dims=1)
        self.data[:, -1] = torch.nan_to_num(
            value.to(device=self.data.device, dtype=self.data.dtype)
        )
        self.filled.clamp_max_(self.history_frames - 1).add_(1)

    def tensor(self) -> torch.Tensor:
        return self.data

    def flatten(self) -> torch.Tensor:
        """Return time-major oldest-to-newest flattened history."""
        return self.data.reshape(self.num_envs, self.history_frames * self.feature_dim)
