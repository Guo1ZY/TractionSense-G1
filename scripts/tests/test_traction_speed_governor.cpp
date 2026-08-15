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
    cfg.probability_critical_enter = 0.95f;
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

    // A deployable causal risk estimator can take over through AUTO.
    cfg.mode = TractionGovernorMode::Auto;
    cfg.low_hold_s = 0.04f;
    cfg.high_hold_s = 0.06f;
    cfg.warmup_s = 0.04f;
    cfg.probe_s = 0.04f;
    cfg.low_reprobe_s = 0.04f;
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

    // A persistent moderate Hall risk can be self-sustaining at the LOW gait
    // because the TPU is weakly excited.  Periodic bounded re-probing must
    // still occur below the critical threshold so recovered friction can be
    // observed and released.
    student_low.low_probability = 0.70f;
    bool saw_reprobe = false;
    for (int i = 0; i < 4; ++i) {
        output = governor.update(
            {0.80f, 0.0f, 0.0f},
            student_low,
            false,
            false,
            false,
            0.02f);
        saw_reprobe = saw_reprobe || output.probing;
    }
    assert(saw_reprobe);
    assert(output.command[0] <= cfg.probe_speed_limit + 1.0e-5f);

    student_low.low_probability = 1.0f;
    output = governor.update(
        {0.80f, 0.20f, 0.30f},
        student_low,
        false,
        false,
        false,
        0.02f);
    assert(near(output.command[0], 0.0f));

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

    // A non-zero command below the legacy gait dead zone receives one
    // bounded launch pulse; an exact zero command never moves.
    governor.configure(cfg);
    float small_max = 0.0f;
    for (int i = 0; i < 8; ++i) {
        output = governor.update(
            {0.05f, 0.0f, 0.0f},
            student_high,
            false,
            false,
            false,
            0.02f);
        small_max = std::max(small_max, output.command[0]);
    }
    assert(small_max > 0.05f);
    assert(small_max <= cfg.probe_speed_limit + 1.0e-5f);
    governor.configure(cfg);
    for (int i = 0; i < 8; ++i) {
        output = governor.update(
            {0.0f, 0.0f, 0.0f},
            student_high,
            false,
            false,
            false,
            0.02f);
    }
    assert(near(output.command[0], 0.0f));

    // Critical Hall risk aborts any launch and retains final authority.
    governor.configure(cfg);
    for (int i = 0; i < 4; ++i) {
        output = governor.update(
            {0.05f, 0.0f, 0.0f},
            student_high,
            false,
            false,
            false,
            0.02f);
    }
    TractionFeedback student_critical = student_high;
    student_critical.low_probability = 1.0f;
    output = governor.update(
        {0.05f, 0.0f, 0.0f},
        student_critical,
        false,
        false,
        false,
        0.02f);
    assert(!output.probing);
    assert(near(output.command[0], 0.0f));

    // A conservative AUTO LOW decision must not deadlock a requested
    // micro-step. Bounded pulses preserve the mean request while exact zero
    // and critical risk retain final authority.
    TractionGovernorConfig crawl_cfg = cfg;
    crawl_cfg.mode = TractionGovernorMode::Auto;
    crawl_cfg.probability_low_enter = 0.40f;
    crawl_cfg.probability_high_enter = 0.30f;
    crawl_cfg.probability_critical_enter = 0.90f;
    crawl_cfg.low_hold_s = 0.02f;
    crawl_cfg.warmup_s = 1.0f;
    crawl_cfg.low_reprobe_s = 10.0f;
    crawl_cfg.crawl_pulse_s = 0.10f;
    crawl_cfg.crawl_min_hold_s = 0.10f;
    crawl_cfg.launch_accel_rate = 100.0f;
    crawl_cfg.accel_rate = 100.0f;
    crawl_cfg.decel_rate = 100.0f;
    governor.configure(crawl_cfg);
    TractionFeedback moderate_low = student_high;
    moderate_low.low_probability = 0.60f;
    float low_crawl_max = 0.0f;
    float low_crawl_sum = 0.0f;
    for (int i = 0; i < 100; ++i) {
        output = governor.update(
            {0.05f, 0.0f, 0.0f},
            moderate_low,
            false,
            false,
            false,
            0.02f);
        low_crawl_max = std::max(low_crawl_max, output.command[0]);
        if (i >= 50) {
            low_crawl_sum += output.command[0];
        }
    }
    assert(output.state == TractionState::Low);
    assert(low_crawl_max > crawl_cfg.min_detection_command);
    assert(low_crawl_sum / 50.0f > 0.02f);
    assert(low_crawl_sum / 50.0f < 0.12f);
    output = governor.update(
        {0.0f, 0.0f, 0.0f},
        moderate_low,
        false,
        false,
        false,
        0.02f);
    assert(near(output.command[0], 0.0f));
    moderate_low.low_probability = 1.0f;
    output = governor.update(
        {0.05f, 0.0f, 0.0f},
        moderate_low,
        false,
        false,
        false,
        0.02f);
    assert(near(output.command[0], 0.0f));

    // When turning is enabled, LOW and HIGH apply separate lateral/yaw caps.
    cfg.mode = TractionGovernorMode::ManualLow;
    cfg.lock_lateral_yaw = false;
    cfg.low_lateral_limit = 0.05f;
    cfg.high_lateral_limit = 0.25f;
    cfg.low_yaw_limit = 0.15f;
    cfg.high_yaw_limit = 0.60f;
    governor.configure(cfg);
    output = governor.update(
        {0.8f, 0.2f, 0.5f},
        no_feedback,
        false,
        false,
        false,
        0.02f);
    assert(near(output.command[1], 0.05f));
    assert(near(output.command[2], 0.15f));
    output = governor.update(
        {0.8f, 0.2f, 0.5f},
        no_feedback,
        false,
        true,
        false,
        0.02f);
    assert(near(output.command[1], 0.20f));
    assert(near(output.command[2], 0.50f));

    std::cout << "traction_speed_governor: PASS\n";
    return 0;
}
