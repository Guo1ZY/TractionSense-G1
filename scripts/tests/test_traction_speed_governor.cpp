#include "isaaclab/envs/traction_speed_governor.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace
{

bool near(float lhs, float rhs, float eps = 1.0e-5f)
{
    return std::abs(lhs - rhs) <= eps;
}

}

int main()
{
    using namespace isaaclab;

    TractionGovernorConfig cfg;
    cfg.enabled = true;
    cfg.mode = TractionGovernorMode::ManualLow;
    cfg.low_speed_limit = 0.15f;
    cfg.high_speed_limit = 0.35f;
    cfg.accel_rate = 100.0f;
    cfg.decel_rate = 100.0f;

    TractionSpeedGovernor governor;
    governor.configure(cfg);
    const TractionFeedback no_feedback;

    auto output = governor.update(
        {0.80f, 0.20f, 0.30f},
        no_feedback,
        false,
        false,
        false,
        0.02f);
    assert(output.state == TractionState::Low);
    assert(output.manual_override);
    assert(near(output.command[0], 0.15f));
    assert(near(output.command[1], 0.0f));
    assert(near(output.command[2], 0.0f));

    output = governor.update(
        {0.80f, 0.20f, 0.30f},
        no_feedback,
        false,
        true,
        false,
        0.02f);
    assert(output.state == TractionState::High);
    assert(output.manual_override);
    assert(near(output.command[0], 0.35f));

    output = governor.update(
        {-0.80f, 0.0f, 0.0f},
        no_feedback,
        true,
        false,
        false,
        0.02f);
    assert(output.state == TractionState::Low);
    assert(near(output.command[0], -0.15f));

    // The 480-D proprioceptive estimator can take over through AUTO.
    cfg.mode = TractionGovernorMode::Auto;
    cfg.low_hold_s = 0.04f;
    cfg.high_hold_s = 0.06f;
    governor.configure(cfg);
    TractionFeedback student_high;
    student_high.valid = true;
    student_high.low_probability = 0.10f;
    for (int i = 0; i < 10; ++i) {
        output = governor.update(
            {0.0f, 0.0f, 0.0f},
            student_high,
            false,
            false,
            false,
            0.02f);
    }
    assert(output.state == TractionState::Unknown);
    for (int i = 0; i < 3; ++i) {
        output = governor.update(
            {0.80f, 0.0f, 0.0f},
            student_high,
            false,
            false,
            false,
            0.02f);
    }
    assert(output.state == TractionState::High);
    assert(!output.manual_override);
    assert(near(output.command[0], 0.35f));

    TractionFeedback student_low;
    student_low.valid = true;
    student_low.low_probability = 0.90f;
    for (int i = 0; i < 2; ++i) {
        output = governor.update(
            {0.80f, 0.0f, 0.0f},
            student_low,
            false,
            false,
            false,
            0.02f);
    }
    assert(output.state == TractionState::Low);
    assert(near(output.command[0], 0.15f));

    // Missing AUTO feedback always falls back to conservative UNKNOWN/LOW cap.
    for (int i = 0; i < 20; ++i) {
        output = governor.update(
            {0.80f, 0.0f, 0.0f},
            no_feedback,
            false,
            false,
            false,
            0.02f);
    }
    assert(output.state == TractionState::Unknown);
    assert(near(output.command[0], 0.15f));

    std::cout << "traction_speed_governor: PASS\n";
    return 0;
}
