// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/envs/lateral_motion_feedback_preflight.h"
#include "isaaclab/envs/traction_speed_governor.h"
#include <iostream>
#include "isaaclab/utils/utils.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        try {
            const auto slew = cfg["commands"]["base_velocity"]["slew_rate"];
            if (slew) {
                command_slew_lin = std::max(0.0f, slew["lin"].as<float>(0.0f));
                command_slew_yaw = std::max(0.0f, slew["yaw"].as<float>(0.0f));
            }
        } catch (const std::exception&) {
            command_slew_lin = 0.0f;
            command_slew_yaw = 0.0f;
        }
        // Explicit environment overrides are useful for simulation A/B tests.
        // With no YAML field and no override the legacy behavior is unchanged.
        if (const char* value = std::getenv("G1_CMD_SLEW_LIN")) {
            command_slew_lin = std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_CMD_SLEW_YAW")) {
            command_slew_yaw = std::max(0.0f, std::strtof(value, nullptr));
        }
        configure_traction_governor();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
        const bool lateral_feedback_required =
            requires_lateral_motion_feedback(cfg["observations"]);
        const bool motion_sidecar_configured = nonempty_motion_feedback_path(
            std::getenv("G1_MOTION_FEEDBACK_PATH"));
        if (const char* estimator_path = std::getenv("G1_FRICTION_ESTIMATOR_ONNX")) {
            if (estimator_path[0] != '\0') {
                friction_estimator = std::make_unique<ScalarOrtRunner>(estimator_path);
                if (const char* alpha = std::getenv("G1_FRICTION_ESTIMATOR_ALPHA")) {
                    friction_estimator_alpha = std::clamp(std::strtof(alpha, nullptr), 0.01f, 1.0f);
                }
                std::cout << "[friction_estimator] ONNX=" << estimator_path
                          << " input=" << friction_estimator->input_size()
                          << " alpha=" << friction_estimator_alpha << std::endl;
            }
        }
        // The estimator overwrites policy_dim-2. Never install it for a
        // sensor-age schema merely because the environment variable is stale.
        const char* lateral_estimator_path =
            std::getenv("G1_LATERAL_VELOCITY_ESTIMATOR_ONNX");
        if (should_load_lateral_velocity_estimator(
                lateral_feedback_required, lateral_estimator_path)) {
            const char* estimator_path = lateral_estimator_path;
            try {
                lateral_velocity_estimator =
                    std::make_unique<ScalarOrtRunner>(estimator_path);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "Failed to load G1_LATERAL_VELOCITY_ESTIMATOR_ONNX '"
                    + std::string(estimator_path) + "': " + error.what());
            }
            if (const char* alpha =
                    std::getenv("G1_LATERAL_VELOCITY_ESTIMATOR_ALPHA")) {
                lateral_velocity_estimator_alpha =
                    std::clamp(std::strtof(alpha, nullptr), 0.01f, 1.0f);
            }
            std::cout
                << "[lateral_velocity_estimator] ONNX=" << estimator_path
                << " input=" << lateral_velocity_estimator->input_size()
                << " alpha=" << lateral_velocity_estimator_alpha
                << std::endl;
        }
        if (lateral_feedback_required) {
            validate_lateral_motion_feedback_preflight(
                true,
                motion_sidecar_configured,
                lateral_velocity_estimator != nullptr,
                observation_manager->policy_observation_size(),
                lateral_velocity_estimator
                    ? lateral_velocity_estimator->input_size()
                    : 0);
        }
        const char* classifier_path =
            std::getenv("G1_TRACTION_PROPRIO_CLASSIFIER_ONNX");
        const char* estimator_path =
            std::getenv("G1_TRACTION_PROPRIO_ESTIMATOR_ONNX");
        const char* traction_model_path =
            (classifier_path && classifier_path[0] != '\0')
                ? classifier_path
                : estimator_path;
        traction_proprio_output_is_probability =
            classifier_path && classifier_path[0] != '\0';
        if (traction_model_path && traction_model_path[0] != '\0') {
                traction_proprio_estimator =
                    std::make_unique<ScalarOrtRunner>(traction_model_path);
                if (const char* value =
                        std::getenv("G1_TRACTION_ESTIMATOR_ALPHA")) {
                    traction_proprio_estimator_alpha =
                        std::clamp(std::strtof(value, nullptr), 0.01f, 1.0f);
                }
                if (const char* value =
                        std::getenv("G1_TRACTION_MU_MIDPOINT")) {
                    traction_mu_midpoint =
                        std::clamp(std::strtof(value, nullptr), 0.20f, 0.90f);
                }
                if (const char* value =
                        std::getenv("G1_TRACTION_MU_TEMPERATURE")) {
                    traction_mu_temperature =
                        std::clamp(std::strtof(value, nullptr), 0.01f, 0.30f);
                }
                std::cout
                    << "[traction_proprio_estimator] ONNX="
                    << traction_model_path
                    << " input=" << traction_proprio_estimator->input_size()
                    << " output="
                    << (traction_proprio_output_is_probability
                            ? "p_low"
                            : "estimated_mu")
                    << " alpha=" << traction_proprio_estimator_alpha
                    << " midpoint=" << traction_mu_midpoint
                    << " temperature=" << traction_mu_temperature
                    << " foot_channels=IGNORED"
                    << std::endl;
        }
        if (const char* hall_risk_path =
                std::getenv("G1_TRACTION_HALL_RISK_ONNX")) {
            if (hall_risk_path[0] != '\0') {
                traction_hall_risk_estimator =
                    std::make_unique<ScalarOrtRunner>(hall_risk_path);
                if (const char* value =
                        std::getenv("G1_TRACTION_HALL_RISK_ALPHA")) {
                    traction_hall_risk_alpha = std::clamp(
                        std::strtof(value, nullptr), 0.01f, 1.0f);
                }
                std::cout
                    << "[traction_hall_risk] ONNX=" << hall_risk_path
                    << " input=" << traction_hall_risk_estimator->input_size()
                    << " alpha=" << traction_hall_risk_alpha
                    << " semantics=causal_risk_probability"
                    << " hall_to_force=DISABLED"
                    << std::endl;
                }
            }
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        filtered_velocity_command = {0.0f, 0.0f, 0.0f};
        observation_manager->reset();
        friction_estimate = 0.20f;
        friction_estimate_initialized = false;
        lateral_velocity_estimate = 0.0f;
        lateral_velocity_estimate_initialized = false;
        traction_governor.reset();
        traction_low_combo_previous = false;
        traction_high_combo_previous = false;
        traction_auto_combo_previous = false;
        traction_log_elapsed_s = 0.0f;
        traction_last_reported_state = TractionState::Unknown;
        traction_last_reported_manual = false;
        traction_proprio_mu = 0.20f;
        traction_proprio_estimator_initialized = false;
        traction_proprio_feedback = {};
        traction_hall_risk = 1.0f;
        traction_hall_risk_initialized = false;
        traction_hall_feedback = {};
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        apply_traction_hall_risk_estimator(obs);
        apply_traction_proprio_estimator(obs);
        apply_friction_estimator(obs);
        apply_lateral_velocity_estimator(obs);
        log_policy_observation(obs);
        auto action = alg->act(obs);
        action_manager->process_action(action);
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    std::unique_ptr<ScalarOrtRunner> friction_estimator;
    std::unique_ptr<ScalarOrtRunner> lateral_velocity_estimator;
    std::unique_ptr<ScalarOrtRunner> traction_proprio_estimator;
    std::unique_ptr<ScalarOrtRunner> traction_hall_risk_estimator;
    long episode_length = 0;
    float global_phase = 0.0f;

    std::vector<float> filter_velocity_command(const std::vector<float>& target)
    {
        if (target.size() != 3 || (command_slew_lin <= 0.0f && command_slew_yaw <= 0.0f)) {
            filtered_velocity_command = target;
            return target;
        }
        for (size_t axis = 0; axis < 3; ++axis) {
            const float rate = axis < 2 ? command_slew_lin : command_slew_yaw;
            if (rate <= 0.0f) {
                filtered_velocity_command[axis] = target[axis];
                continue;
            }
            const float max_delta = rate * step_dt;
            const float delta = std::clamp(
                target[axis] - filtered_velocity_command[axis], -max_delta, max_delta);
            filtered_velocity_command[axis] += delta;
        }
        return filtered_velocity_command;
    }

    std::vector<float> process_velocity_command(
        const std::vector<float>& requested)
    {
        if (requested.size() != 3 || !traction_governor.config().enabled) {
            return filter_velocity_command(requested);
        }

        bool low_combo = false;
        bool high_combo = false;
        bool auto_combo = false;
        if (robot->data.joystick) {
            auto* joystick = robot->data.joystick;
            low_combo = joystick->RB.pressed && joystick->down.pressed;
            high_combo = joystick->RB.pressed && joystick->up.pressed;
            auto_combo = joystick->RB.pressed && joystick->left.pressed;
        }
        const bool select_low = low_combo && !traction_low_combo_previous;
        const bool select_high =
            high_combo && !traction_high_combo_previous;
        const bool select_auto = auto_combo && !traction_auto_combo_previous;
        traction_low_combo_previous = low_combo;
        traction_high_combo_previous = high_combo;
        traction_auto_combo_previous = auto_combo;

        const auto feedback = read_traction_feedback();
        const std::array<float, 3> requested_array = {
            requested[0],
            requested[1],
            requested[2],
        };
        const auto output = traction_governor.update(
            requested_array,
            feedback,
            select_low,
            select_high,
            select_auto,
            step_dt);
        report_traction_state(output);
        log_traction_governor(output);
        return filter_velocity_command({
            output.command[0],
            output.command[1],
            output.command[2],
        });
    }

private:
    static bool parse_bool_string(const char* value, bool fallback)
    {
        if (!value) {
            return fallback;
        }
        const std::string text(value);
        if (text == "1" || text == "true" || text == "TRUE"
            || text == "on" || text == "ON") {
            return true;
        }
        if (text == "0" || text == "false" || text == "FALSE"
            || text == "off" || text == "OFF") {
            return false;
        }
        return fallback;
    }

    static TractionGovernorMode parse_traction_mode(
        const std::string& value,
        TractionGovernorMode fallback)
    {
        if (value == "manual_low" || value == "low") {
            return TractionGovernorMode::ManualLow;
        }
        if (value == "manual_high" || value == "high") {
            return TractionGovernorMode::ManualHigh;
        }
        if (value == "auto") {
            return TractionGovernorMode::Auto;
        }
        return fallback;
    }

    void configure_traction_governor()
    {
        TractionGovernorConfig governor_cfg;
        const YAML::Node node =
            cfg["commands"]["base_velocity"]["traction_governor"];
        if (node) {
            governor_cfg.enabled =
                node["enabled"].as<bool>(governor_cfg.enabled);
            governor_cfg.lock_lateral_yaw = node["lock_lateral_yaw"].as<bool>(
                governor_cfg.lock_lateral_yaw);
            governor_cfg.mode = parse_traction_mode(
                node["mode"].as<std::string>("manual_low"),
                governor_cfg.mode);
            governor_cfg.low_speed_limit = node["low_speed_limit"].as<float>(
                governor_cfg.low_speed_limit);
            governor_cfg.high_speed_limit = node["high_speed_limit"].as<float>(
                governor_cfg.high_speed_limit);
            governor_cfg.critical_speed_limit =
                node["critical_speed_limit"].as<float>(
                    governor_cfg.critical_speed_limit);
            governor_cfg.low_lateral_limit =
                node["low_lateral_limit"].as<float>(
                    governor_cfg.low_lateral_limit);
            governor_cfg.high_lateral_limit =
                node["high_lateral_limit"].as<float>(
                    governor_cfg.high_lateral_limit);
            governor_cfg.low_yaw_limit = node["low_yaw_limit"].as<float>(
                governor_cfg.low_yaw_limit);
            governor_cfg.high_yaw_limit = node["high_yaw_limit"].as<float>(
                governor_cfg.high_yaw_limit);
            governor_cfg.accel_rate =
                node["accel_rate"].as<float>(governor_cfg.accel_rate);
            governor_cfg.decel_rate =
                node["decel_rate"].as<float>(governor_cfg.decel_rate);
            governor_cfg.probability_low_enter =
                node["probability_low_enter"].as<float>(
                    governor_cfg.probability_low_enter);
            governor_cfg.probability_high_enter =
                node["probability_high_enter"].as<float>(
                    governor_cfg.probability_high_enter);
            governor_cfg.probability_critical_enter =
                node["probability_critical_enter"].as<float>(
                    governor_cfg.probability_critical_enter);
            governor_cfg.critical_hold_s =
                node["critical_hold_s"].as<float>(
                    governor_cfg.critical_hold_s);
            governor_cfg.probability_ema_alpha =
                node["probability_ema_alpha"].as<float>(
                    governor_cfg.probability_ema_alpha);
            governor_cfg.state_reference_ema_alpha =
                node["state_reference_ema_alpha"].as<float>(
                    governor_cfg.state_reference_ema_alpha);
            governor_cfg.relative_low_rise =
                node["relative_low_rise"].as<float>(
                    governor_cfg.relative_low_rise);
            governor_cfg.relative_high_drop =
                node["relative_high_drop"].as<float>(
                    governor_cfg.relative_high_drop);
            governor_cfg.low_hold_s =
                node["low_hold_s"].as<float>(governor_cfg.low_hold_s);
            governor_cfg.high_hold_s =
                node["high_hold_s"].as<float>(governor_cfg.high_hold_s);
            governor_cfg.feedback_timeout_s =
                node["feedback_timeout_s"].as<float>(
                    governor_cfg.feedback_timeout_s);
            governor_cfg.min_detection_command =
                node["min_detection_command"].as<float>(
                    governor_cfg.min_detection_command);
            governor_cfg.startup_command_threshold =
                node["startup_command_threshold"].as<float>(
                    governor_cfg.startup_command_threshold);
            governor_cfg.warmup_s =
                node["warmup_s"].as<float>(governor_cfg.warmup_s);
            governor_cfg.probe_s =
                node["probe_s"].as<float>(governor_cfg.probe_s);
            governor_cfg.probe_speed_limit =
                node["probe_speed_limit"].as<float>(
                    governor_cfg.probe_speed_limit);
            governor_cfg.low_reprobe_s = node["low_reprobe_s"].as<float>(
                governor_cfg.low_reprobe_s);
            governor_cfg.probe_relative_clear_drop =
                node["probe_relative_clear_drop"].as<float>(
                    governor_cfg.probe_relative_clear_drop);
            governor_cfg.crawl_pulse_s = node["crawl_pulse_s"].as<float>(
                governor_cfg.crawl_pulse_s);
            governor_cfg.crawl_min_hold_s =
                node["crawl_min_hold_s"].as<float>(
                    governor_cfg.crawl_min_hold_s);
            governor_cfg.launch_accel_rate =
                node["launch_accel_rate"].as<float>(
                    governor_cfg.launch_accel_rate);
            governor_cfg.tracking_low_enter =
                node["tracking_low_enter"].as<float>(
                    governor_cfg.tracking_low_enter);
            governor_cfg.tracking_high_enter =
                node["tracking_high_enter"].as<float>(
                    governor_cfg.tracking_high_enter);
        }

        if (const char* value = std::getenv("G1_TRACTION_GOVERNOR")) {
            governor_cfg.enabled =
                parse_bool_string(value, governor_cfg.enabled);
        }
        if (const char* value = std::getenv("G1_TRACTION_MODE")) {
            governor_cfg.mode = parse_traction_mode(
                value,
                governor_cfg.mode);
        }
        if (const char* value = std::getenv("G1_TRACTION_LOW_SPEED")) {
            governor_cfg.low_speed_limit =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_HIGH_SPEED")) {
            governor_cfg.high_speed_limit =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_LOW_LATERAL")) {
            governor_cfg.low_lateral_limit =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_HIGH_LATERAL")) {
            governor_cfg.high_lateral_limit =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_LOW_YAW")) {
            governor_cfg.low_yaw_limit =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_HIGH_YAW")) {
            governor_cfg.high_yaw_limit =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_ACCEL")) {
            governor_cfg.accel_rate =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_DECEL")) {
            governor_cfg.decel_rate =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_TRACTION_PROBE_SPEED")) {
            governor_cfg.probe_speed_limit =
                std::max(0.0f, std::strtof(value, nullptr));
        }
        traction_governor.configure(governor_cfg);

        if (governor_cfg.enabled) {
            std::cout
                << "[traction_governor] ENABLED mode="
                << TractionSpeedGovernor::mode_name(governor_cfg.mode)
                << " low=" << governor_cfg.low_speed_limit
                << " high=" << governor_cfg.high_speed_limit
                << " lateral=" << governor_cfg.low_lateral_limit
                << '/' << governor_cfg.high_lateral_limit
                << " yaw=" << governor_cfg.low_yaw_limit
                << '/' << governor_cfg.high_yaw_limit
                << " accel=" << governor_cfg.accel_rate
                << " decel=" << governor_cfg.decel_rate
                << " lock_vy_wz="
                << (governor_cfg.lock_lateral_yaw ? "true" : "false")
                << std::endl;
            std::cout
                << "[traction_governor] RB+DOWN=LOW, RB+UP=HIGH, "
                   "RB+LEFT=AUTO/RESET"
                << std::endl;
        }
    }

    static bool read_scalar_from_json(
        const std::string& payload,
        const std::string& key,
        float& value)
    {
        const std::string token = "\"" + key + "\"";
        const auto key_pos = payload.find(token);
        if (key_pos == std::string::npos) {
            return false;
        }
        const auto colon_pos = payload.find(':', key_pos + token.size());
        if (colon_pos == std::string::npos) {
            return false;
        }
        const char* begin = payload.c_str() + colon_pos + 1;
        char* end = nullptr;
        const float candidate = std::strtof(begin, &end);
        if (end == begin || !std::isfinite(candidate)) {
            return false;
        }
        value = candidate;
        return true;
    }

    TractionFeedback read_traction_feedback() const
    {
        const char* path = std::getenv("G1_TRACTION_FEEDBACK_PATH");
        if (!path || path[0] == '\0') {
            return traction_hall_feedback.valid
                ? traction_hall_feedback
                : traction_proprio_feedback;
        }

        try {
            const auto write_time = std::filesystem::last_write_time(path);
            const float age_s = std::chrono::duration<float>(
                decltype(write_time)::clock::now() - write_time).count();
            if (age_s < 0.0f
                || age_s > traction_governor.config().feedback_timeout_s) {
                return traction_hall_feedback.valid
                    ? traction_hall_feedback
                    : traction_proprio_feedback;
            }
        } catch (const std::exception&) {
            return traction_hall_feedback.valid
                ? traction_hall_feedback
                : traction_proprio_feedback;
        }

        std::ifstream input(path);
        if (!input) {
            return traction_hall_feedback.valid
                ? traction_hall_feedback
                : traction_proprio_feedback;
        }
        const std::string payload(
            (std::istreambuf_iterator<char>(input)),
            std::istreambuf_iterator<char>());
        TractionFeedback feedback;
        const bool has_vx =
            read_scalar_from_json(payload, "vx", feedback.measured_vx);
        const bool has_probability = read_scalar_from_json(
            payload,
            "p_low",
            feedback.low_probability);
        read_scalar_from_json(payload, "slip_score", feedback.slip_score);
        feedback.valid = has_vx || has_probability;
        if (feedback.valid) {
            return feedback;
        }
        return traction_hall_feedback.valid
            ? traction_hall_feedback
            : traction_proprio_feedback;
    }

    void apply_traction_hall_risk_estimator(
        const std::unordered_map<std::string, std::vector<float>>& obs)
    {
        if (!traction_hall_risk_estimator || obs.empty()) {
            return;
        }
        auto it = obs.find("obs");
        if (it == obs.end()) {
            it = obs.begin();
        }
        const auto& values = it->second;
        const size_t estimator_dim = traction_hall_risk_estimator->input_size();
        if (estimator_dim != 1864 || values.size() != estimator_dim) {
            if (!traction_hall_risk_mismatch_reported) {
                std::cerr
                    << "[traction_hall_risk] expected the exact 1864-D "
                       "Hall/proprio policy observation; model="
                    << estimator_dim << " policy=" << values.size()
                    << std::endl;
                traction_hall_risk_mismatch_reported = true;
            }
            traction_hall_feedback = {};
            return;
        }
        float prediction = traction_hall_risk_estimator->infer(values);
        if (!std::isfinite(prediction)) {
            traction_hall_feedback = {};
            return;
        }
        prediction = std::clamp(prediction, 0.0f, 1.0f);
        if (
            prediction
            >= traction_governor.config().probability_critical_enter) {
            // Critical Hall risk has immediate authority; EMA is used only
            // for ordinary hysteresis and must not delay an emergency stop.
            traction_hall_risk = prediction;
            traction_hall_risk_initialized = true;
        } else if (!traction_hall_risk_initialized) {
            traction_hall_risk = prediction;
            traction_hall_risk_initialized = true;
        } else {
            traction_hall_risk =
                (1.0f - traction_hall_risk_alpha) * traction_hall_risk
                + traction_hall_risk_alpha * prediction;
        }
        traction_hall_feedback.valid = true;
        traction_hall_feedback.measured_vx = 0.0f;
        traction_hall_feedback.slip_score = traction_hall_risk;
        traction_hall_feedback.low_probability = traction_hall_risk;
    }

    void apply_traction_proprio_estimator(
        const std::unordered_map<std::string, std::vector<float>>& obs)
    {
        if (!traction_proprio_estimator || obs.empty()) {
            return;
        }
        auto it = obs.find("obs");
        if (it == obs.end()) {
            it = obs.begin();
        }
        const auto& values = it->second;
        const size_t estimator_dim = traction_proprio_estimator->input_size();
        if (estimator_dim != 480 || values.size() < estimator_dim) {
            if (!traction_proprio_estimator_mismatch_reported) {
                std::cerr
                    << "[traction_proprio_estimator] expected a 480-D "
                       "model and policy observation with at least 480 "
                       "elements; model="
                    << estimator_dim << " policy=" << values.size()
                    << std::endl;
                traction_proprio_estimator_mismatch_reported = true;
            }
            traction_proprio_feedback = {};
            return;
        }

        // Only the common 480-D proprioceptive prefix is consumed. Any foot,
        // Hall, sensor-health or motion-feedback suffix is deliberately
        // ignored so invalid foot sensors cannot create a terrain decision.
        const std::vector<float> proprio(
            values.begin(),
            values.begin() + estimator_dim);
        float prediction = traction_proprio_estimator->infer(proprio);
        if (!std::isfinite(prediction)) {
            traction_proprio_feedback = {};
            return;
        }
        prediction = std::clamp(
            prediction,
            0.0f,
            traction_proprio_output_is_probability ? 1.0f : 1.30f);
        if (!traction_proprio_estimator_initialized) {
            traction_proprio_mu = prediction;
            traction_proprio_estimator_initialized = true;
        } else {
            traction_proprio_mu =
                (1.0f - traction_proprio_estimator_alpha)
                    * traction_proprio_mu
                + traction_proprio_estimator_alpha * prediction;
        }
        float p_low = traction_proprio_mu;
        if (!traction_proprio_output_is_probability) {
            const float normalized =
                (traction_proprio_mu - traction_mu_midpoint)
                / traction_mu_temperature;
            p_low = 1.0f / (1.0f + std::exp(normalized));
        }
        traction_proprio_feedback.valid = true;
        traction_proprio_feedback.measured_vx = 0.0f;
        traction_proprio_feedback.slip_score = -1.0f;
        traction_proprio_feedback.low_probability =
            std::clamp(p_low, 0.0f, 1.0f);
    }

    void report_traction_state(const TractionGovernorOutput& output)
    {
        if (output.state == traction_last_reported_state
            && output.manual_override == traction_last_reported_manual) {
            return;
        }
        std::cout
            << "[traction_governor] state="
            << TractionSpeedGovernor::state_name(output.state)
            << " source="
            << (output.manual_override ? "manual" : "auto")
            << " requested_vx=" << output.requested_vx
            << " limited_vx=" << output.command[0]
            << " score=" << output.score
            << std::endl;
        traction_last_reported_state = output.state;
        traction_last_reported_manual = output.manual_override;
    }

    void log_traction_governor(const TractionGovernorOutput& output)
    {
        const char* path = std::getenv("G1_TRACTION_GOVERNOR_LOG");
        if (!path || path[0] == '\0') {
            return;
        }
        traction_log_elapsed_s += step_dt;
        if (traction_log_elapsed_s < 0.10f) {
            return;
        }
        traction_log_elapsed_s = 0.0f;
        if (!traction_log_.is_open()) {
            traction_log_.open(path, std::ios::out | std::ios::app);
            if (traction_log_
                && traction_log_.tellp() == std::streampos(0)) {
                traction_log_
                    << "episode_time_s,state,source,feedback_valid,probing,"
                       "requested_vx,limited_vx,measured_vx,score\n";
            }
        }
        if (!traction_log_) {
            return;
        }
        traction_log_
            << episode_length * step_dt << ','
            << TractionSpeedGovernor::state_name(output.state) << ','
            << (output.manual_override ? "manual" : "auto") << ','
            << (output.feedback_valid ? 1 : 0) << ','
            << (output.probing ? 1 : 0) << ','
            << output.requested_vx << ','
            << output.command[0] << ','
            << output.measured_vx << ','
            << output.score << '\n';
        traction_log_.flush();
    }

    void apply_lateral_velocity_estimator(
        std::unordered_map<std::string, std::vector<float>>& obs)
    {
        if (!lateral_velocity_estimator || obs.empty()) {
            return;
        }
        auto it = obs.find("obs");
        if (it == obs.end()) {
            it = obs.begin();
        }
        auto& values = it->second;
        const size_t estimator_dim = lateral_velocity_estimator->input_size();
        if (values.size() != estimator_dim + 2) {
            if (!lateral_velocity_estimator_mismatch_reported) {
                std::cerr
                    << "[lateral_velocity_estimator] policy observation must "
                       "be estimator_dim+2; got "
                    << values.size() << " vs " << estimator_dim + 2
                    << std::endl;
                lateral_velocity_estimator_mismatch_reported = true;
            }
            return;
        }
        const std::vector<float> causal_prefix(
            values.begin(), values.begin() + estimator_dim);
        float prediction = lateral_velocity_estimator->infer(causal_prefix);
        if (!std::isfinite(prediction)) {
            prediction = 0.0f;
        }
        prediction = std::clamp(prediction, -1.5f, 1.5f);
        if (!lateral_velocity_estimate_initialized) {
            lateral_velocity_estimate = prediction;
            lateral_velocity_estimate_initialized = true;
        } else {
            lateral_velocity_estimate =
                (1.0f - lateral_velocity_estimator_alpha)
                    * lateral_velocity_estimate
                + lateral_velocity_estimator_alpha * prediction;
        }
        // Final two motion channels are [body_vy, relative_heading].
        values[estimator_dim] = lateral_velocity_estimate;
    }

    void apply_friction_estimator(
        std::unordered_map<std::string, std::vector<float>>& obs)
    {
        if (!friction_estimator || obs.empty()) {
            return;
        }
        auto it = obs.find("obs");
        if (it == obs.end()) {
            it = obs.begin();
        }
        auto& values = it->second;
        const size_t estimator_dim = friction_estimator->input_size();
        if (values.size() != estimator_dim + 1) {
            if (!friction_estimator_mismatch_reported) {
                std::cerr << "[friction_estimator] policy observation must be estimator_dim+1; got "
                          << values.size() << " vs " << estimator_dim + 1 << std::endl;
                friction_estimator_mismatch_reported = true;
            }
            return;
        }
        const std::vector<float> deploy_prefix(values.begin(), values.begin() + estimator_dim);
        float prediction = friction_estimator->infer(deploy_prefix);
        if (!std::isfinite(prediction)) {
            prediction = 0.20f;
        }
        prediction = std::clamp(prediction, 0.0f, 1.20f);
        if (!friction_estimate_initialized) {
            // Unknown/new surface starts conservatively and gains speed only
            // after several consistent observations.
            friction_estimate = std::min(prediction, 0.35f);
            friction_estimate_initialized = true;
        } else {
            friction_estimate =
                (1.0f - friction_estimator_alpha) * friction_estimate
                + friction_estimator_alpha * prediction;
        }
        values.back() = friction_estimate;
    }

    void log_policy_observation(
        const std::unordered_map<std::string, std::vector<float>>& obs)
    {
        const char* path = std::getenv("G1_POLICY_OBS_FILE");
        if (!path || path[0] == '\0' || obs.empty()) {
            return;
        }
        auto it = obs.find("obs");
        if (it == obs.end()) {
            it = obs.begin();
        }
        const auto& values = it->second;
        if (!policy_obs_log_.is_open()) {
            policy_obs_log_.open(path, std::ios::binary | std::ios::app);
        }
        if (!policy_obs_log_) {
            return;
        }
        // Repeated binary records: magic 'OBS1', dim, wall-clock ns, float32[dim].
        const uint32_t magic = 0x3153424fu;
        const uint32_t dim = static_cast<uint32_t>(values.size());
        const uint64_t stamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        policy_obs_log_.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        policy_obs_log_.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        policy_obs_log_.write(reinterpret_cast<const char*>(&stamp_ns), sizeof(stamp_ns));
        policy_obs_log_.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        policy_obs_log_.flush();
    }

    std::ofstream policy_obs_log_;
    float friction_estimator_alpha = 0.20f;
    float friction_estimate = 0.20f;
    bool friction_estimate_initialized = false;
    bool friction_estimator_mismatch_reported = false;
    float lateral_velocity_estimator_alpha = 0.35f;
    float lateral_velocity_estimate = 0.0f;
    bool lateral_velocity_estimate_initialized = false;
    bool lateral_velocity_estimator_mismatch_reported = false;
    float command_slew_lin = 0.0f;
    float command_slew_yaw = 0.0f;
    std::vector<float> filtered_velocity_command = {0.0f, 0.0f, 0.0f};
    TractionSpeedGovernor traction_governor;
    bool traction_low_combo_previous = false;
    bool traction_high_combo_previous = false;
    bool traction_auto_combo_previous = false;
    float traction_log_elapsed_s = 0.0f;
    TractionState traction_last_reported_state = TractionState::Unknown;
    bool traction_last_reported_manual = false;
    std::ofstream traction_log_;
    float traction_proprio_estimator_alpha = 0.20f;
    float traction_proprio_mu = 0.20f;
    float traction_mu_midpoint = 0.55f;
    float traction_mu_temperature = 0.08f;
    bool traction_proprio_estimator_initialized = false;
    bool traction_proprio_estimator_mismatch_reported = false;
    bool traction_proprio_output_is_probability = false;
    TractionFeedback traction_proprio_feedback;
    float traction_hall_risk_alpha = 0.20f;
    float traction_hall_risk = 1.0f;
    bool traction_hall_risk_initialized = false;
    bool traction_hall_risk_mismatch_reported = false;
    TractionFeedback traction_hall_feedback;
};

};
