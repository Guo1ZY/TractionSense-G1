// Copyright (c) 2026.
// Conservative two-surface speed governor for G1 deployment.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace isaaclab
{

enum class TractionState
{
    Unknown,
    Low,
    High,
};

enum class TractionGovernorMode
{
    ManualLow,
    ManualHigh,
    Auto,
};

struct TractionGovernorConfig
{
    bool enabled = false;
    bool lock_lateral_yaw = true;
    TractionGovernorMode mode = TractionGovernorMode::ManualLow;

    float low_speed_limit = 0.15f;
    float high_speed_limit = 0.35f;
    float accel_rate = 0.15f;
    float decel_rate = 0.80f;

    // AUTO with a 480-D proprioceptive classifier: LOW if p_low is large,
    // HIGH if small. Plantar/Hall channels are not part of this input.
    float probability_low_enter = 0.65f;
    float probability_high_enter = 0.35f;
    float low_hold_s = 0.20f;
    float high_hold_s = 1.00f;

    // AUTO without a Student: use an externally measured forward velocity.
    // UNKNOWN first walks slowly, then performs one guarded probe.
    float feedback_timeout_s = 0.25f;
    float min_detection_command = 0.20f;
    float warmup_s = 1.00f;
    float probe_s = 1.50f;
    float tracking_low_enter = 0.55f;
    float tracking_high_enter = 0.30f;
};

struct TractionFeedback
{
    bool valid = false;
    float measured_vx = 0.0f;
    // Negative means that no explicit slip score is available.
    float slip_score = -1.0f;
    // Negative means that no deployable traction classifier is connected.
    float low_probability = -1.0f;
};

struct TractionGovernorOutput
{
    std::array<float, 3> command = {0.0f, 0.0f, 0.0f};
    TractionState state = TractionState::Unknown;
    bool manual_override = false;
    bool probing = false;
    bool feedback_valid = false;
    float requested_vx = 0.0f;
    float measured_vx = 0.0f;
    float score = 0.0f;
};

class TractionSpeedGovernor
{
public:
    void configure(const TractionGovernorConfig& cfg)
    {
        cfg_ = cfg;
        cfg_.low_speed_limit = std::max(0.0f, cfg_.low_speed_limit);
        cfg_.high_speed_limit =
            std::max(cfg_.low_speed_limit, cfg_.high_speed_limit);
        cfg_.accel_rate = std::max(0.0f, cfg_.accel_rate);
        cfg_.decel_rate = std::max(cfg_.accel_rate, cfg_.decel_rate);
        cfg_.probability_high_enter =
            std::clamp(cfg_.probability_high_enter, 0.0f, 1.0f);
        cfg_.probability_low_enter = std::clamp(
            cfg_.probability_low_enter,
            cfg_.probability_high_enter,
            1.0f);
        cfg_.low_hold_s = std::max(0.0f, cfg_.low_hold_s);
        cfg_.high_hold_s = std::max(0.0f, cfg_.high_hold_s);
        cfg_.feedback_timeout_s = std::max(0.0f, cfg_.feedback_timeout_s);
        cfg_.min_detection_command =
            std::max(0.0f, cfg_.min_detection_command);
        cfg_.warmup_s = std::max(0.0f, cfg_.warmup_s);
        cfg_.probe_s = std::max(0.0f, cfg_.probe_s);
        cfg_.tracking_high_enter =
            std::clamp(cfg_.tracking_high_enter, 0.0f, 1.5f);
        cfg_.tracking_low_enter = std::clamp(
            cfg_.tracking_low_enter,
            cfg_.tracking_high_enter,
            1.5f);
        reset();
    }

    void reset()
    {
        output_command_ = {0.0f, 0.0f, 0.0f};
        low_evidence_s_ = 0.0f;
        high_evidence_s_ = 0.0f;
        unknown_time_s_ = 0.0f;
        probe_time_s_ = 0.0f;
        probe_score_sum_ = 0.0f;
        probe_score_count_ = 0;
        probing_ = false;
        feedback_missing_s_ = 0.0f;

        if (cfg_.mode == TractionGovernorMode::ManualHigh) {
            state_ = TractionState::High;
            manual_override_ = true;
        } else if (cfg_.mode == TractionGovernorMode::ManualLow) {
            state_ = TractionState::Low;
            manual_override_ = true;
        } else {
            state_ = TractionState::Unknown;
            manual_override_ = false;
        }
    }

    TractionGovernorOutput update(
        const std::array<float, 3>& requested,
        const TractionFeedback& feedback,
        bool select_low,
        bool select_high,
        bool select_auto,
        float dt)
    {
        dt = std::clamp(dt, 1.0e-4f, 0.1f);
        if (!cfg_.enabled) {
            output_command_ = requested;
            return make_output(requested, feedback, 0.0f);
        }

        if (select_low) {
            state_ = TractionState::Low;
            manual_override_ = true;
            clear_detection();
        } else if (select_high) {
            state_ = TractionState::High;
            manual_override_ = true;
            clear_detection();
        } else if (select_auto) {
            state_ = TractionState::Unknown;
            manual_override_ = false;
            clear_detection();
        }

        float score = 0.0f;
        if (!manual_override_) {
            score = update_auto(requested[0], feedback, dt);
        }

        const float speed_limit =
            (state_ == TractionState::High || probing_)
                ? cfg_.high_speed_limit
                : cfg_.low_speed_limit;
        std::array<float, 3> target = requested;
        target[0] = std::clamp(target[0], -speed_limit, speed_limit);
        if (cfg_.lock_lateral_yaw) {
            target[1] = 0.0f;
            target[2] = 0.0f;
        }
        apply_slew(target, dt);
        return make_output(requested, feedback, score);
    }

    static const char* state_name(TractionState state)
    {
        switch (state) {
            case TractionState::Low:
                return "LOW";
            case TractionState::High:
                return "HIGH";
            default:
                return "UNKNOWN";
        }
    }

    static const char* mode_name(TractionGovernorMode mode)
    {
        switch (mode) {
            case TractionGovernorMode::ManualLow:
                return "manual_low";
            case TractionGovernorMode::ManualHigh:
                return "manual_high";
            default:
                return "auto";
        }
    }

    const TractionGovernorConfig& config() const
    {
        return cfg_;
    }

private:
    void clear_detection()
    {
        low_evidence_s_ = 0.0f;
        high_evidence_s_ = 0.0f;
        unknown_time_s_ = 0.0f;
        probe_time_s_ = 0.0f;
        probe_score_sum_ = 0.0f;
        probe_score_count_ = 0;
        probing_ = false;
        feedback_missing_s_ = 0.0f;
    }

    float update_auto(
        float requested_vx,
        const TractionFeedback& feedback,
        float dt)
    {
        if (!feedback.valid) {
            feedback_missing_s_ += dt;
            if (feedback_missing_s_ >= cfg_.feedback_timeout_s) {
                state_ = TractionState::Unknown;
                probing_ = false;
                low_evidence_s_ = 0.0f;
                high_evidence_s_ = 0.0f;
            }
            return 1.0f;
        }
        feedback_missing_s_ = 0.0f;

        // Preferred path for the deployable proprioceptive estimator.
        if (feedback.low_probability >= 0.0f) {
            const float p_low =
                std::clamp(feedback.low_probability, 0.0f, 1.0f);
            // Terrain is not observable while standing still. Do not let a
            // static-pose shortcut promote UNKNOWN to HIGH before the robot
            // has accumulated walking response.
            if (std::abs(requested_vx) < cfg_.min_detection_command) {
                low_evidence_s_ = 0.0f;
                high_evidence_s_ = 0.0f;
                return p_low;
            }
            if (p_low >= cfg_.probability_low_enter) {
                low_evidence_s_ += dt;
                high_evidence_s_ = 0.0f;
            } else if (p_low <= cfg_.probability_high_enter) {
                high_evidence_s_ += dt;
                low_evidence_s_ = 0.0f;
            } else {
                low_evidence_s_ = 0.0f;
                high_evidence_s_ = 0.0f;
            }
            if (low_evidence_s_ >= cfg_.low_hold_s) {
                state_ = TractionState::Low;
                probing_ = false;
            } else if (high_evidence_s_ >= cfg_.high_hold_s) {
                state_ = TractionState::High;
                probing_ = false;
            }
            return p_low;
        }

        const float requested_abs = std::abs(requested_vx);
        if (requested_abs < cfg_.min_detection_command) {
            return 0.0f;
        }

        const float expected_abs =
            std::max(std::abs(output_command_[0]), cfg_.low_speed_limit);
        const float directed_measured =
            std::copysign(feedback.measured_vx, requested_vx);
        const float tracking_error = std::clamp(
            (expected_abs - directed_measured)
                / std::max(expected_abs, 0.05f),
            0.0f,
            1.5f);
        const float slip = feedback.slip_score >= 0.0f
            ? std::clamp(feedback.slip_score, 0.0f, 1.5f)
            : tracking_error;
        const float score =
            std::clamp(0.70f * tracking_error + 0.30f * slip, 0.0f, 1.5f);

        if (state_ == TractionState::Unknown) {
            unknown_time_s_ += dt;
            if (!probing_ && unknown_time_s_ >= cfg_.warmup_s) {
                probing_ = true;
                probe_time_s_ = 0.0f;
                probe_score_sum_ = 0.0f;
                probe_score_count_ = 0;
            }
            if (probing_) {
                probe_time_s_ += dt;
                if (probe_time_s_ >= 0.5f * cfg_.probe_s) {
                    probe_score_sum_ += score;
                    ++probe_score_count_;
                }
                if (probe_time_s_ >= cfg_.probe_s) {
                    const float mean_score = probe_score_count_ > 0
                        ? probe_score_sum_
                            / static_cast<float>(probe_score_count_)
                        : 1.0f;
                    state_ = mean_score >= cfg_.tracking_low_enter
                        ? TractionState::Low
                        : TractionState::High;
                    probing_ = false;
                }
            }
            return score;
        }

        if (state_ == TractionState::High) {
            if (score >= cfg_.tracking_low_enter) {
                low_evidence_s_ += dt;
            } else {
                low_evidence_s_ = 0.0f;
            }
            if (low_evidence_s_ >= cfg_.low_hold_s) {
                state_ = TractionState::Low;
                probing_ = false;
            }
        } else if (score <= cfg_.tracking_high_enter) {
            // LOW remains conservative. Promotion requires an explicit AUTO
            // reset/probe or a Student p_low signal; this avoids periodic
            // high-speed probes on a slippery real floor.
            high_evidence_s_ += dt;
        } else {
            high_evidence_s_ = 0.0f;
        }
        return score;
    }

    void apply_slew(const std::array<float, 3>& target, float dt)
    {
        for (size_t axis = 0; axis < output_command_.size(); ++axis) {
            const float current = output_command_[axis];
            const bool slowing =
                current * target[axis] < 0.0f
                || std::abs(target[axis]) < std::abs(current);
            const float rate = slowing ? cfg_.decel_rate : cfg_.accel_rate;
            if (rate <= 0.0f) {
                output_command_[axis] = target[axis];
                continue;
            }
            const float max_delta = rate * dt;
            output_command_[axis] += std::clamp(
                target[axis] - current,
                -max_delta,
                max_delta);
        }
    }

    TractionGovernorOutput make_output(
        const std::array<float, 3>& requested,
        const TractionFeedback& feedback,
        float score) const
    {
        TractionGovernorOutput output;
        output.command = output_command_;
        output.state = state_;
        output.manual_override = manual_override_;
        output.probing = probing_;
        output.feedback_valid = feedback.valid;
        output.requested_vx = requested[0];
        output.measured_vx = feedback.measured_vx;
        output.score = score;
        return output;
    }

    TractionGovernorConfig cfg_;
    TractionState state_ = TractionState::Unknown;
    bool manual_override_ = false;
    bool probing_ = false;
    std::array<float, 3> output_command_ = {0.0f, 0.0f, 0.0f};
    float low_evidence_s_ = 0.0f;
    float high_evidence_s_ = 0.0f;
    float unknown_time_s_ = 0.0f;
    float probe_time_s_ = 0.0f;
    float probe_score_sum_ = 0.0f;
    int probe_score_count_ = 0;
    float feedback_missing_s_ = 0.0f;
};

}  // namespace isaaclab
