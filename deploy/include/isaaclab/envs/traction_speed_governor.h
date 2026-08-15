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
    float critical_speed_limit = 0.0f;
    float low_lateral_limit = 0.05f;
    float high_lateral_limit = 0.25f;
    float low_yaw_limit = 0.15f;
    float high_yaw_limit = 0.60f;
    float accel_rate = 0.15f;
    float decel_rate = 0.80f;

    // AUTO with a causal Hall + proprioceptive-history risk estimator: LOW if
    // p_low is large, HIGH if small. The score is not a Hall-to-force inverse.
    float probability_low_enter = 0.65f;
    float probability_high_enter = 0.35f;
    float probability_critical_enter = 0.85f;
    float critical_hold_s = 0.04f;
    float probability_ema_alpha = 0.20f;
    float state_reference_ema_alpha = 0.01f;
    float relative_low_rise = 0.12f;
    float relative_high_drop = 0.12f;
    float low_hold_s = 0.20f;
    float high_hold_s = 1.00f;

    // AUTO without a Student: use an externally measured forward velocity.
    // UNKNOWN first walks slowly, then performs one guarded probe.
    float feedback_timeout_s = 0.25f;
    float min_detection_command = 0.20f;
    float startup_command_threshold = 0.02f;
    float warmup_s = 0.20f;
    float probe_s = 0.45f;
    float probe_speed_limit = 0.25f;
    float low_reprobe_s = 2.50f;
    float probe_relative_clear_drop = 0.08f;
    float crawl_pulse_s = 0.45f;
    float crawl_min_hold_s = 0.25f;
    float launch_accel_rate = 1.00f;
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
        cfg_.critical_speed_limit = std::clamp(
            cfg_.critical_speed_limit, 0.0f, cfg_.low_speed_limit);
        cfg_.low_lateral_limit = std::max(0.0f, cfg_.low_lateral_limit);
        cfg_.high_lateral_limit =
            std::max(cfg_.low_lateral_limit, cfg_.high_lateral_limit);
        cfg_.low_yaw_limit = std::max(0.0f, cfg_.low_yaw_limit);
        cfg_.high_yaw_limit =
            std::max(cfg_.low_yaw_limit, cfg_.high_yaw_limit);
        cfg_.accel_rate = std::max(0.0f, cfg_.accel_rate);
        cfg_.decel_rate = std::max(cfg_.accel_rate, cfg_.decel_rate);
        cfg_.probe_speed_limit = std::clamp(
            cfg_.probe_speed_limit,
            cfg_.low_speed_limit,
            cfg_.high_speed_limit);
        cfg_.low_reprobe_s = std::max(0.0f, cfg_.low_reprobe_s);
        cfg_.probe_relative_clear_drop = std::clamp(
            cfg_.probe_relative_clear_drop, 0.0f, 1.0f);
        cfg_.startup_command_threshold = std::max(
            0.0f, cfg_.startup_command_threshold);
        cfg_.crawl_pulse_s = std::max(0.0f, cfg_.crawl_pulse_s);
        cfg_.crawl_min_hold_s = std::max(0.0f, cfg_.crawl_min_hold_s);
        cfg_.launch_accel_rate = std::max(
            cfg_.accel_rate, cfg_.launch_accel_rate);
        cfg_.probability_high_enter =
            std::clamp(cfg_.probability_high_enter, 0.0f, 1.0f);
        cfg_.probability_low_enter = std::clamp(
            cfg_.probability_low_enter,
            cfg_.probability_high_enter,
            1.0f);
        cfg_.probability_critical_enter = std::clamp(
            cfg_.probability_critical_enter,
            cfg_.probability_low_enter,
            1.0f);
        cfg_.critical_hold_s = std::max(0.0f, cfg_.critical_hold_s);
        cfg_.probability_ema_alpha = std::clamp(
            cfg_.probability_ema_alpha, 1.0e-6f, 1.0f);
        cfg_.state_reference_ema_alpha = std::clamp(
            cfg_.state_reference_ema_alpha, 1.0e-6f, 1.0f);
        cfg_.relative_low_rise = std::clamp(
            cfg_.relative_low_rise, 0.0f, 1.0f);
        cfg_.relative_high_drop = std::clamp(
            cfg_.relative_high_drop, 0.0f, 1.0f);
        reset();
    }

    void reset()
    {
        output_command_ = {0.0f, 0.0f, 0.0f};
        low_evidence_s_ = 0.0f;
        high_evidence_s_ = 0.0f;
        critical_evidence_s_ = 0.0f;
        unknown_time_s_ = 0.0f;
        low_state_time_s_ = 0.0f;
        probe_time_s_ = 0.0f;
        probe_score_sum_ = 0.0f;
        probe_score_count_ = 0;
        probe_start_probability_ = 1.0f;
        probing_ = false;
        feedback_missing_s_ = 0.0f;
        crawl_cycle_time_s_ = 0.0f;
        probability_ema_ = 1.0f;
        probability_ema_initialized_ = false;
        state_probability_reference_ = 1.0f;
        state_reference_initialized_ = false;
        critical_stop_ = false;

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

        if (feedback.valid && feedback.low_probability >= 0.0f) {
            const float raw_probability = std::clamp(
                feedback.low_probability, 0.0f, 1.0f);
            critical_evidence_s_ =
                raw_probability >= cfg_.probability_critical_enter
                    ? critical_evidence_s_ + dt
                    : 0.0f;
            critical_stop_ = critical_evidence_s_ + 1.0e-6f
                >= cfg_.critical_hold_s;
        } else {
            critical_evidence_s_ = 0.0f;
            // Missing Hall data is uncertainty, and uncertainty has no
            // learned action authority in the deployed controller.
            critical_stop_ = !feedback.valid;
        }

        float score = 0.0f;
        if (!manual_override_) {
            score = update_auto(requested[0], feedback, dt);
        }

        float speed_limit = probing_
            ? cfg_.probe_speed_limit
            : state_ == TractionState::High
                ? cfg_.high_speed_limit
                : cfg_.low_speed_limit;
        const bool critical_risk = critical_stop_;
        if (critical_risk) {
            speed_limit = cfg_.critical_speed_limit;
        }
        std::array<float, 3> target = requested;
        target[0] = std::clamp(target[0], -speed_limit, speed_limit);
        const bool probe_forward = probing_
            && std::abs(requested[0]) >= cfg_.startup_command_threshold;
        if (probe_forward) {
            target[0] = std::copysign(
                cfg_.probe_speed_limit, requested[0]);
        }

        // After AUTO makes its first traction decision, commands below the
        // actor's launch dead zone are represented as bounded micro-step
        // pulses. LOW is included to avoid a weak-excitation deadlock while
        // duty cycle preserves the requested long-term mean speed. Manual LOW,
        // zero command, critical risk and invalid feedback remain conservative.
        const float requested_abs = std::abs(requested[0]);
        const bool crawl = !probing_
            && !critical_risk
            && !manual_override_
            && state_ != TractionState::Unknown
            && requested_abs >= cfg_.startup_command_threshold
            && requested_abs < cfg_.min_detection_command;
        bool crawl_active = false;
        if (crawl) {
            const float period = std::max(
                cfg_.crawl_pulse_s * cfg_.probe_speed_limit
                    / std::max(requested_abs, cfg_.startup_command_threshold),
                cfg_.crawl_pulse_s + cfg_.crawl_min_hold_s);
            crawl_cycle_time_s_ = std::fmod(
                crawl_cycle_time_s_ + dt, std::max(period, dt));
            crawl_active = crawl_cycle_time_s_ <= cfg_.crawl_pulse_s;
            target[0] = crawl_active
                ? std::copysign(cfg_.probe_speed_limit, requested[0])
                : 0.0f;
        } else {
            crawl_cycle_time_s_ = 0.0f;
        }
        if (critical_risk) {
            target = {0.0f, 0.0f, 0.0f};
        }
        if (cfg_.lock_lateral_yaw) {
            target[1] = 0.0f;
            target[2] = 0.0f;
        } else {
            if (critical_risk) {
                target[1] = 0.0f;
                target[2] = 0.0f;
                apply_slew(target, dt, false);
                return make_output(requested, feedback, score);
            }
            const bool high_state =
                state_ == TractionState::High || probing_;
            const float lateral_limit = high_state
                ? cfg_.high_lateral_limit
                : cfg_.low_lateral_limit;
            const float yaw_limit = high_state
                ? cfg_.high_yaw_limit
                : cfg_.low_yaw_limit;
            target[1] = std::clamp(
                target[1], -lateral_limit, lateral_limit);
            target[2] = std::clamp(target[2], -yaw_limit, yaw_limit);
        }
        apply_slew(target, dt, probing_ || crawl_active);
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
        critical_evidence_s_ = 0.0f;
        unknown_time_s_ = 0.0f;
        low_state_time_s_ = 0.0f;
        probe_time_s_ = 0.0f;
        probe_score_sum_ = 0.0f;
        probe_score_count_ = 0;
        probing_ = false;
        feedback_missing_s_ = 0.0f;
        crawl_cycle_time_s_ = 0.0f;
        probability_ema_ = 1.0f;
        probability_ema_initialized_ = false;
        state_probability_reference_ = 1.0f;
        state_reference_initialized_ = false;
        critical_stop_ = false;
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

        // Preferred path for a deployable causal risk estimator.  In the
        // Hall policy this value comes from Hall+proprio history and is not a
        // magnetic-to-force conversion.
        if (feedback.low_probability >= 0.0f) {
            const float raw_probability =
                std::clamp(feedback.low_probability, 0.0f, 1.0f);
            const float p_low = probability_ema_initialized_
                ? (1.0f - cfg_.probability_ema_alpha) * probability_ema_
                    + cfg_.probability_ema_alpha * raw_probability
                : raw_probability;
            probability_ema_ = p_low;
            probability_ema_initialized_ = true;
            // Terrain is not observable while standing still. Do not let a
            // static-pose shortcut promote UNKNOWN to HIGH before the robot
            // has accumulated walking response.
            if (std::abs(requested_vx) < cfg_.startup_command_threshold) {
                low_evidence_s_ = 0.0f;
                high_evidence_s_ = 0.0f;
                return p_low;
            }

            const TractionState prior_state = state_;
            const bool relative_low =
                state_ == TractionState::High
                && state_reference_initialized_
                && p_low - state_probability_reference_
                    >= cfg_.relative_low_rise
                && p_low > cfg_.probability_high_enter;
            const bool relative_high =
                state_ == TractionState::Low
                && state_reference_initialized_
                && state_probability_reference_ - p_low
                    >= cfg_.relative_high_drop
                && p_low < cfg_.probability_low_enter;

            // UNKNOWN completes a full bounded probe.  Once LOW/HIGH has
            // been established, use a change relative to that state's own
            // slowly moving reference instead of trusting one absolute Hall
            // value across soles, temperatures and power cycles.
            const bool low_evidence =
                !probing_ && (critical_stop_ || relative_low);
            const bool high_evidence = !probing_ && relative_high;
            if (low_evidence) {
                low_evidence_s_ += dt;
                high_evidence_s_ = 0.0f;
            } else if (high_evidence) {
                high_evidence_s_ += dt;
                low_evidence_s_ = 0.0f;
            } else if (!probing_) {
                low_evidence_s_ = 0.0f;
                high_evidence_s_ = 0.0f;
            }
            if (critical_stop_ || low_evidence_s_ >= cfg_.low_hold_s) {
                const bool entered_low = state_ != TractionState::Low;
                state_ = TractionState::Low;
                if (entered_low) {
                    low_state_time_s_ = 0.0f;
                }
            } else if (
                state_ == TractionState::Low
                && high_evidence_s_ >= cfg_.high_hold_s) {
                state_ = TractionState::High;
                low_state_time_s_ = 0.0f;
            }
            if (state_ != prior_state) {
                state_probability_reference_ = p_low;
                state_reference_initialized_ = true;
            }

            // High traction is accepted only after a bounded active probe.
            // A crawl/standstill observation need not contain enough
            // excitation to distinguish surface friction.
            if (!probing_) {
                if (state_ == TractionState::Unknown) {
                    unknown_time_s_ += dt;
                    if (
                        unknown_time_s_ >= cfg_.warmup_s
                        && !critical_stop_) {
                        probing_ = true;
                    }
                } else if (state_ == TractionState::Low) {
                    // Low-speed walking may not excite the embedded magnets
                    // enough for a biased risk estimate to clear by itself.
                    // Re-probe periodically under all non-critical, valid
                    // Hall evidence; a critical score still aborts the probe.
                    if (!critical_stop_) {
                        low_state_time_s_ += dt;
                    } else {
                        low_state_time_s_ = 0.0f;
                    }
                    if (
                        low_state_time_s_ >= cfg_.low_reprobe_s
                        && !critical_stop_) {
                        probing_ = true;
                    }
                }
                if (probing_) {
                    probe_start_probability_ = p_low;
                    probe_time_s_ = 0.0f;
                    probe_score_sum_ = 0.0f;
                    probe_score_count_ = 0;
                    low_evidence_s_ = 0.0f;
                    high_evidence_s_ = 0.0f;
                    unknown_time_s_ = 0.0f;
                    low_state_time_s_ = 0.0f;
                }
            }
            if (probing_ && critical_stop_) {
                probing_ = false;
                state_ = TractionState::Low;
                probe_time_s_ = 0.0f;
                state_probability_reference_ = p_low;
                state_reference_initialized_ = true;
            }
            if (probing_) {
                probe_time_s_ += dt;
                if (probe_time_s_ >= 0.5f * cfg_.probe_s) {
                    probe_score_sum_ += p_low;
                    ++probe_score_count_;
                }
                if (probe_time_s_ >= cfg_.probe_s) {
                    const float mean_probability = probe_score_count_ > 0
                        ? probe_score_sum_
                            / static_cast<float>(probe_score_count_)
                        : 1.0f;
                    const bool relative_clear =
                        probe_start_probability_ - mean_probability
                        >= cfg_.probe_relative_clear_drop;
                    state_ = (
                        mean_probability <= cfg_.probability_high_enter
                        || relative_clear)
                        ? TractionState::High
                        : TractionState::Low;
                    state_probability_reference_ = mean_probability;
                    state_reference_initialized_ = true;
                    probing_ = false;
                    probe_time_s_ = 0.0f;
                    low_state_time_s_ = 0.0f;
                    low_evidence_s_ = 0.0f;
                    high_evidence_s_ = 0.0f;
                }
            }
            if (
                state_reference_initialized_
                && state_ != TractionState::Unknown
                && !probing_) {
                state_probability_reference_ =
                    (1.0f - cfg_.state_reference_ema_alpha)
                        * state_probability_reference_
                    + cfg_.state_reference_ema_alpha * p_low;
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

    void apply_slew(
        const std::array<float, 3>& target,
        float dt,
        bool launch_forward)
    {
        for (size_t axis = 0; axis < output_command_.size(); ++axis) {
            const float current = output_command_[axis];
            const bool slowing =
                current * target[axis] < 0.0f
                || std::abs(target[axis]) < std::abs(current);
            const float rate = slowing
                ? cfg_.decel_rate
                : (launch_forward && axis == 0
                    ? cfg_.launch_accel_rate
                    : cfg_.accel_rate);
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
    float critical_evidence_s_ = 0.0f;
    float unknown_time_s_ = 0.0f;
    float low_state_time_s_ = 0.0f;
    float probe_time_s_ = 0.0f;
    float probe_score_sum_ = 0.0f;
    float probe_start_probability_ = 1.0f;
    int probe_score_count_ = 0;
    float feedback_missing_s_ = 0.0f;
    float crawl_cycle_time_s_ = 0.0f;
    float probability_ema_ = 1.0f;
    bool probability_ema_initialized_ = false;
    float state_probability_reference_ = 1.0f;
    bool state_reference_initialized_ = false;
    bool critical_stop_ = false;
};

}  // namespace isaaclab
