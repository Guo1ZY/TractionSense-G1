"""Manager-based environment with causally complete HighEnd bank resets."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv

from unitree_rl_lab.tasks.locomotion.mdp.spatial_friction_state import SPATIAL_HIGH_END
from unitree_rl_lab.traction.high_end_state_bank import (
    ACTION_DIM,
    POLICY_DIM,
    policy_history_terms,
    seed_circular_buffer_logical,
)


class HighEndRecoveryRLEnv(ManagerBasedRLEnv):
    """Restore mechanics, temporal actor context and path references together.

    The reset event samples and writes the articulation state, then stages a
    bank row.  Isaac Lab subsequently resets all managers.  This subclass runs
    after that manager reset, reinstates episode/reference/action context and,
    after the first observation compute, replaces and seeds the exact 1864-D
    actor history.  Ordinary tasks continue to use upstream ManagerBasedRLEnv.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        pending = getattr(self, "_high_end_recovery_pending_sample_ids", None)
        bank = getattr(self, "_high_end_recovery_state_bank", None)
        if pending is None or bank is None:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        sample_ids = pending[ids]
        if bool((sample_ids < 0).any().item()):
            raise RuntimeError(
                "HighEnd recovery reset event did not stage one bank row per reset env"
            )
        observation = bank.arrays["observation"].index_select(0, sample_ids)
        if observation.shape != (ids.numel(), POLICY_DIM):
            raise RuntimeError("HighEnd state-bank actor observation ABI drift")

        def _ensure(name: str, shape: tuple[int, ...], *, dtype=torch.float32):
            value = getattr(self, name, None)
            expected = (self.num_envs, *shape)
            if value is None:
                value = torch.zeros(expected, device=self.device, dtype=dtype)
                setattr(self, name, value)
            elif tuple(value.shape) != expected or value.device != torch.device(self.device):
                raise RuntimeError(
                    f"{name} has incompatible shape/device: {tuple(value.shape)} / {value.device}"
                )
            return value

        _ensure("motion_feedback_initial_yaw", ())[ids] = bank.arrays[
            "motion_feedback_initial_yaw"
        ][sample_ids]
        _ensure("straight_heading_reference_xy", (2,))[ids] = bank.arrays[
            "straight_heading_reference_xy"
        ][sample_ids]
        _ensure("straight_heading_initialized", (), dtype=torch.bool)[ids] = True
        track_origin = bank.arrays["straight_track_origin_local_xy"][sample_ids].clone()
        track_origin += self.scene.env_origins[ids, :2]
        _ensure("straight_track_origin_xy", (2,))[ids] = track_origin
        _ensure("straight_track_lateral_axis", (2,))[ids] = bank.arrays[
            "straight_track_lateral_axis"
        ][sample_ids]
        _ensure("straight_track_initialized", (), dtype=torch.bool)[ids] = True

        # The privileged course state is restored for rewards/critic only.  It
        # is absent from the 1864-D actor and is never copied into observation.
        _ensure("spatial_course_stage_buf", (), dtype=torch.long)[ids] = SPATIAL_HIGH_END
        _ensure("spatial_low_contact_buf", (), dtype=torch.bool)[ids] = False
        _ensure("spatial_high_end_contact_buf", (), dtype=torch.bool)[ids] = True

        histories = policy_history_terms(observation)
        latest_action = histories["last_action"][:, -1]
        previous_action = histories["last_action"][:, -2]
        if latest_action.shape != (ids.numel(), ACTION_DIM):
            raise RuntimeError("HighEnd state-bank action history ABI drift")
        self.action_manager._action[ids] = latest_action
        self.action_manager._prev_action[ids] = previous_action

        command = histories["velocity_commands"][:, -1]
        term = self.command_manager.get_term("base_velocity")
        term.vel_command_b[ids, :3] = command.to(term.vel_command_b.dtype)
        term.is_standing_env[ids] = command.abs().amax(dim=1) <= 1.0e-7
        if hasattr(term, "is_spin_env"):
            term.is_spin_env[ids] = False
        if hasattr(term, "is_high_speed_env"):
            term.is_high_speed_env[ids] = command[:, 0] >= 0.65

        # Reward/observation functions use <=1 as their ordinary reset latch.
        # Two is the smallest value that preserves the bank references while
        # retaining a full fresh recovery episode horizon.
        self.episode_length_buf[ids] = 2
        last_sample_ids = getattr(self, "_high_end_recovery_last_sample_ids", None)
        if last_sample_ids is None:
            last_sample_ids = torch.full(
                (self.num_envs,), -1, device=self.device, dtype=torch.long
            )
            self._high_end_recovery_last_sample_ids = last_sample_ids
        last_sample_ids[ids] = sample_ids
        self._high_end_recovery_finalize_ids = ids.clone()
        self._high_end_recovery_finalize_sample_ids = sample_ids.clone()

    def reset(self, *args, **kwargs):
        observation, extras = super().reset(*args, **kwargs)
        self._finalize_high_end_recovery_context(observation)
        return observation, extras

    def step(self, action: torch.Tensor):
        result = super().step(action)
        self._finalize_high_end_recovery_context(result[0])
        return result

    def _finalize_high_end_recovery_context(self, observation) -> None:
        ids = getattr(self, "_high_end_recovery_finalize_ids", None)
        sample_ids = getattr(self, "_high_end_recovery_finalize_sample_ids", None)
        bank = getattr(self, "_high_end_recovery_state_bank", None)
        if ids is None or sample_ids is None or bank is None:
            return
        exact = bank.arrays["observation"].index_select(0, sample_ids)
        histories = policy_history_terms(exact)
        history_buffers = self.observation_manager._group_obs_term_history_buffer.get(
            "policy", {}
        )
        if set(histories) - set(history_buffers):
            raise RuntimeError(
                "HighEnd recovery policy history terms do not match ObservationManager: "
                f"missing={sorted(set(histories) - set(history_buffers))}"
            )
        for name, values in histories.items():
            seed_circular_buffer_logical(history_buffers[name], ids, values)
            restored = history_buffers[name].buffer.index_select(0, ids)
            if not torch.equal(restored, values.to(restored)):
                error = torch.max(torch.abs(restored - values.to(restored))).item()
                raise RuntimeError(
                    f"HighEnd recovery history {name} restore mismatch: max_abs={error}"
                )

        if torch.is_tensor(observation):
            policy = observation
        else:
            policy = observation["policy"]
        if policy.ndim != 2 or policy.shape[1] != POLICY_DIM:
            raise RuntimeError(
                f"HighEnd recovery expected policy[N,{POLICY_DIM}], got {tuple(policy.shape)}"
            )
        policy[ids] = exact.to(device=policy.device, dtype=policy.dtype)
        if self.observation_manager._obs_buffer is not None:
            cached = self.observation_manager._obs_buffer["policy"]
            if torch.is_tensor(cached):
                cached[ids] = exact.to(device=cached.device, dtype=cached.dtype)

        self._restore_hall_sensor_state(ids, sample_ids, exact)
        # The global-time spatial contact updater may run once between the
        # bank reset and this finalisation while PhysX contact buffers still
        # contain the previous rollout.  It intentionally treats pending bank
        # rows as reset (see mdp.spatial_friction), which prevents stale LOW
        # force from leaking into rewards.  Restore the bank's known
        # privileged stage only after all manager/event updates have finished.
        # This state is never copied into the 1864-D actor observation.
        self.spatial_course_stage_buf[ids] = SPATIAL_HIGH_END
        self.spatial_low_contact_buf[ids] = False
        self.spatial_high_end_contact_buf[ids] = False
        self._audit_high_end_recovery_context(ids, sample_ids, exact)
        self._high_end_recovery_pending_sample_ids[ids] = -1
        del self._high_end_recovery_finalize_ids
        del self._high_end_recovery_finalize_sample_ids

    def _restore_hall_sensor_state(
        self,
        ids: torch.Tensor,
        sample_ids: torch.Tensor,
        exact_observation: torch.Tensor,
    ) -> None:
        sensor = getattr(self, "_hall_foot_sensor", None)
        bank = self._high_end_recovery_state_bank
        if sensor is None:
            raise RuntimeError("HighEnd recovery requires initialized HallFootSensor")

        def _copy(target: torch.Tensor, name: str) -> None:
            source = bank.arrays[name].index_select(0, sample_ids)
            if tuple(target[ids].shape) != tuple(source.shape):
                raise RuntimeError(
                    f"Hall state {name} shape mismatch: target={tuple(target[ids].shape)}, "
                    f"bank={tuple(source.shape)}"
                )
            target[ids] = source.to(device=target.device, dtype=target.dtype)

        _copy(sensor.local_deformation, "hall_local_deformation")
        _copy(sensor.loading_history, "hall_loading_history")
        _copy(sensor.signal.filtered_absolute, "hall_signal_filtered_absolute")
        _copy(sensor.signal.processed, "hall_signal_processed")
        _copy(sensor.signal.baseline, "hall_signal_baseline")
        _copy(sensor.signal.drift, "hall_signal_drift")
        sensor.signal.raw[ids] = sensor.signal.filtered_absolute[ids]
        sensor.signal.ideal[ids] = sensor.signal.filtered_absolute[ids]
        sensor.ideal_field[ids] = sensor.signal.filtered_absolute[ids]
        auto_zero_count = max(int(sensor.cfg.auto_zero_samples), 1)
        sensor.signal._baseline_sum[ids] = sensor.signal.baseline[ids] * auto_zero_count
        sensor.signal._baseline_count[ids] = auto_zero_count
        sensor.signal.baseline_ready[ids] = True
        sensor.signal._filter_initialized[ids] = True

        _copy(sensor._policy_history, "hall_policy_history")
        _copy(sensor._policy_gain, "hall_policy_gain")
        _copy(sensor._policy_cross_axis, "hall_policy_cross_axis")
        _copy(sensor._policy_zero_residual, "hall_policy_zero_residual")
        _copy(sensor._policy_channel_keep, "hall_policy_channel_keep")
        _copy(sensor._policy_foot_keep, "hall_policy_foot_keep")
        _copy(sensor._policy_delay_steps, "hall_policy_delay_steps")
        _copy(sensor._reported_sample_period, "hall_reported_sample_period")
        latest_hall = exact_observation[:, 480:1830].reshape(
            ids.numel(), 15, 2, 15, 3
        )[:, -1]
        sensor._policy_observation[ids] = latest_hall
        sensor.valid_mask[ids] = True
        sensor.sample_age[ids] = 0.0

        debug = sensor.get_debug_data()
        self._hall_foot_packet_cache = {
            "raw": sensor.get_raw_data(),
            "filtered": sensor.get_filtered_data(),
            "normalized": sensor.get_policy_observation(),
            "norm": debug["magnetic_norm"],
            "delta": debug["magnetic_delta"],
            "deformation": debug["local_deformation"],
            "valid": sensor.get_policy_valid_mask(),
            "age": debug["sample_age"],
            "period": sensor.get_reported_sample_period(),
        }
        self._hall_foot_sensor_step = int(self.common_step_counter)
        if hasattr(self, "_hall_foot_prev_episode_length"):
            self._hall_foot_prev_episode_length[ids] = self.episode_length_buf[ids]

    def _audit_high_end_recovery_context(
        self,
        ids: torch.Tensor,
        sample_ids: torch.Tensor,
        exact_observation: torch.Tensor,
    ) -> None:
        """Fail closed if a supposedly complete reset silently lost context."""

        bank = self._high_end_recovery_state_bank
        robot = self.scene["robot"]
        local_root = torch.cat(
            (
                robot.data.root_pos_w.index_select(0, ids)
                - self.scene.env_origins.index_select(0, ids),
                robot.data.root_quat_w.index_select(0, ids),
            ),
            dim=-1,
        )
        checks = {
            "root_pose_local": (
                local_root,
                bank.arrays["root_pose_local"].index_select(0, sample_ids),
            ),
            "root_velocity": (
                robot.data.root_state_w.index_select(0, ids)[:, 7:13],
                bank.arrays["root_velocity"].index_select(0, sample_ids),
            ),
            "joint_pos": (
                robot.data.joint_pos.index_select(0, ids),
                bank.arrays["joint_pos"].index_select(0, sample_ids),
            ),
            "joint_vel": (
                robot.data.joint_vel.index_select(0, ids),
                bank.arrays["joint_vel"].index_select(0, sample_ids),
            ),
            "motion_feedback_initial_yaw": (
                self.motion_feedback_initial_yaw.index_select(0, ids),
                bank.arrays["motion_feedback_initial_yaw"].index_select(0, sample_ids),
            ),
            "straight_heading_reference_xy": (
                self.straight_heading_reference_xy.index_select(0, ids),
                bank.arrays["straight_heading_reference_xy"].index_select(0, sample_ids),
            ),
            "straight_track_lateral_axis": (
                self.straight_track_lateral_axis.index_select(0, ids),
                bank.arrays["straight_track_lateral_axis"].index_select(0, sample_ids),
            ),
            "action": (
                self.action_manager._action.index_select(0, ids),
                exact_observation[:, 335:480].reshape(ids.numel(), 5, ACTION_DIM)[:, -1],
            ),
            "previous_action": (
                self.action_manager._prev_action.index_select(0, ids),
                exact_observation[:, 335:480].reshape(ids.numel(), 5, ACTION_DIM)[:, -2],
            ),
        }
        expected_track_origin = bank.arrays["straight_track_origin_local_xy"].index_select(
            0, sample_ids
        ) + self.scene.env_origins.index_select(0, ids)[:, :2]
        checks["straight_track_origin_xy"] = (
            self.straight_track_origin_xy.index_select(0, ids),
            expected_track_origin,
        )
        command = self.command_manager.get_term("base_velocity").vel_command_b.index_select(
            0, ids
        )[:, :3]
        checks["command"] = (
            command,
            exact_observation[:, 30:45].reshape(ids.numel(), 5, 3)[:, -1],
        )
        sensor = self._hall_foot_sensor
        hall_checks = {
            "hall_local_deformation": sensor.local_deformation,
            "hall_loading_history": sensor.loading_history,
            "hall_signal_filtered_absolute": sensor.signal.filtered_absolute,
            "hall_signal_processed": sensor.signal.processed,
            "hall_signal_baseline": sensor.signal.baseline,
            "hall_signal_drift": sensor.signal.drift,
            "hall_policy_history": sensor._policy_history,
            "hall_policy_gain": sensor._policy_gain,
            "hall_policy_cross_axis": sensor._policy_cross_axis,
            "hall_policy_zero_residual": sensor._policy_zero_residual,
            "hall_policy_channel_keep": sensor._policy_channel_keep,
            "hall_policy_foot_keep": sensor._policy_foot_keep,
            "hall_policy_delay_steps": sensor._policy_delay_steps,
            "hall_reported_sample_period": sensor._reported_sample_period,
        }
        for name, target in hall_checks.items():
            checks[name] = (
                target.index_select(0, ids),
                bank.arrays[name].index_select(0, sample_ids),
            )

        errors: dict[str, float] = {}
        for name, (actual, expected) in checks.items():
            expected = expected.to(device=actual.device, dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise RuntimeError(
                    f"HighEnd recovery audit {name} invalid shape/data: "
                    f"actual={tuple(actual.shape)}, expected={tuple(expected.shape)}"
                )
            error = float(torch.max(torch.abs(actual - expected)).item())
            errors[name] = error
            tolerance = 2.0e-4 if name.startswith(("root_", "joint_")) else 1.0e-6
            if error > tolerance:
                raise RuntimeError(
                    f"HighEnd recovery audit {name} mismatch: "
                    f"max_abs={error:.9g} > {tolerance:.9g}"
                )
        if not torch.equal(
            self.spatial_course_stage_buf.index_select(0, ids),
            torch.full_like(ids, SPATIAL_HIGH_END),
        ):
            raise RuntimeError("HighEnd recovery audit lost privileged HIGH_END latch")
        self._high_end_recovery_last_audit = {
            "restored_env_count": int(ids.numel()),
            "sample_ids": sample_ids.detach().cpu().tolist(),
            "maximum_absolute_error": max(errors.values(), default=0.0),
            "field_errors": errors,
            "policy_dim": POLICY_DIM,
            "actor_uses_force_contact_mu_slip_or_stage": False,
        }


__all__ = ["HighEndRecoveryRLEnv"]
