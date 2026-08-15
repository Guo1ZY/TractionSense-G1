#include <array>
#include <cassert>
#include <cmath>

#include "isaaclab/envs/traction_speed_governor.h"

using isaaclab::TractionFeedback;
using isaaclab::TractionGovernorConfig;
using isaaclab::TractionGovernorMode;
using isaaclab::TractionSpeedGovernor;
using isaaclab::TractionState;

int main()
{
    TractionGovernorConfig cfg;
    cfg.enabled = true;
    cfg.mode = TractionGovernorMode::Auto;
    cfg.low_speed_limit = 0.22f;
    cfg.high_speed_limit = 0.60f;
    cfg.probe_speed_limit = 0.50f;
    cfg.warmup_s = 0.02f;
    cfg.probe_s = 0.10f;
    cfg.low_reprobe_s = 10.0f;
    cfg.probability_low_enter = 0.65f;
    cfg.probability_high_enter = 0.55f;
    cfg.probability_critical_enter = 0.95f;
    cfg.critical_hold_s = 0.04f;
    cfg.low_hold_s = 0.10f;
    cfg.high_hold_s = 0.10f;
    cfg.probability_ema_alpha = 0.20f;
    cfg.state_reference_ema_alpha = 0.002f;
    cfg.relative_low_rise = 0.20f;
    cfg.relative_high_drop = 0.20f;
    cfg.probe_relative_clear_drop = 0.20f;
    cfg.accel_rate = 10.0f;
    cfg.decel_rate = 10.0f;
    cfg.launch_accel_rate = 10.0f;

    TractionSpeedGovernor governor;
    governor.configure(cfg);
    const std::array<float, 3> command = {0.60f, 0.0f, 0.0f};
    TractionFeedback feedback;
    feedback.valid = true;
    feedback.low_probability = 0.20f;

    isaaclab::TractionGovernorOutput output;
    for (int i = 0; i < 8; ++i) {
        output = governor.update(command, feedback, false, false, false, 0.02f);
    }
    assert(output.state == TractionState::High);
    assert(output.command[0] > cfg.low_speed_limit);

    // One critical sample is rejected by the 40 ms debounce.
    feedback.low_probability = 1.0f;
    output = governor.update(command, feedback, false, false, false, 0.02f);
    assert(output.state == TractionState::High);
    assert(output.command[0] > 0.0f);

    // Sustained relative risk rise transitions HIGH -> LOW.
    feedback.low_probability = 0.80f;
    for (int i = 0; i < 30; ++i) {
        output = governor.update(command, feedback, false, false, false, 0.02f);
    }
    assert(output.state == TractionState::Low);
    assert(output.command[0] <= cfg.low_speed_limit + 1.0e-6f);

    // Sustained risk drop relative to the LOW reference restores HIGH.
    feedback.low_probability = 0.15f;
    for (int i = 0; i < 40; ++i) {
        output = governor.update(command, feedback, false, false, false, 0.02f);
    }
    assert(output.state == TractionState::High);

    // Missing Hall feedback is conservative and never creates non-finite output.
    feedback.valid = false;
    output = governor.update(command, feedback, false, false, false, 0.02f);
    assert(std::isfinite(output.command[0]));
    assert(output.command[0] < cfg.high_speed_limit);
    return 0;
}
