"""Time-major causal history shared by training, export, and deployment."""

from __future__ import annotations

import torch

from .schema import TORQUE_TRACTION_FRAME_SCHEMA, TorqueTractionHistorySchema


class TorqueTractionHistory:
    """Fixed-size ``oldest -> newest`` frame history with reset isolation."""

    def __init__(
        self,
        batch_size: int,
        *,
        schema: TorqueTractionHistorySchema = TORQUE_TRACTION_FRAME_SCHEMA,
        device: str | torch.device = "cpu",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.schema = schema
        self.device = torch.device(device)
        self.buffer = torch.zeros(
            batch_size,
            schema.history_frames,
            schema.frame_dimension,
            device=self.device,
        )
        self.valid_frames = torch.zeros(batch_size, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.buffer.zero_()
            self.valid_frames.zero_()
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self.buffer[ids] = 0.0
        self.valid_frames[ids] = 0

    def append(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.shape != (self.buffer.shape[0], self.schema.frame_dimension):
            raise ValueError("frame must have shape [batch,frame_dimension]")
        frame = torch.nan_to_num(frame.to(self.device))
        self.buffer[:, :-1].copy_(self.buffer[:, 1:].clone())
        self.buffer[:, -1].copy_(frame)
        self.valid_frames.add_(1).clamp_max_(self.schema.history_frames)
        return self.sequence()

    def sequence(self) -> torch.Tensor:
        return self.buffer.clone()

    def flattened(self) -> torch.Tensor:
        return self.buffer.reshape(self.buffer.shape[0], self.schema.flat_dimension).clone()

