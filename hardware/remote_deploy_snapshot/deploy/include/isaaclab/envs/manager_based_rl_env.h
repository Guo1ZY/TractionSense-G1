// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include "isaaclab/utils/utils.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <unordered_map>

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
        if (const char* value = std::getenv("G1_CMD_SLEW_LIN")) {
            command_slew_lin = std::max(0.0f, std::strtof(value, nullptr));
        }
        if (const char* value = std::getenv("G1_CMD_SLEW_YAW")) {
            command_slew_yaw = std::max(0.0f, std::strtof(value, nullptr));
        }
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());
        robot->data.joint_effort.resize(robot->data.joint_ids_map.size());

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
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        action_manager->reset();
        filtered_velocity_command = {0.0f, 0.0f, 0.0f};
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        record_policy_observation(obs);
        auto action = alg->act(obs);
        action_manager->process_action(action);
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;

    std::vector<float> filter_velocity_command(const std::vector<float>& target)
    {
        if (target.size() != 3) {
            throw std::runtime_error("base velocity command must contain exactly three values");
        }
        std::vector<float> bounded = target;
        // Independent of YAML and joystick gain, never permit a forward
        // command above the user-approved real-robot envelope.
        float hard_forward_limit = 1.0f;
        if (const char* value = std::getenv("G1_CMD_FORWARD_LIMIT")) {
            hard_forward_limit = std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f);
        }
        float hard_linear_limit = 1.0f;
        if (const char* value = std::getenv("G1_CMD_LINEAR_LIMIT")) {
            hard_linear_limit = std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f);
        }
        float hard_backward_limit = hard_linear_limit;
        if (const char* value = std::getenv("G1_CMD_BACKWARD_LIMIT")) {
            hard_backward_limit = std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f);
        }
        float hard_lateral_limit = hard_linear_limit;
        if (const char* value = std::getenv("G1_CMD_LATERAL_LIMIT")) {
            hard_lateral_limit = std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f);
        }
        float hard_yaw_limit = 1.0f;
        if (const char* value = std::getenv("G1_CMD_YAW_LIMIT")) {
            hard_yaw_limit = std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f);
        }
        bounded[0] = std::clamp(
            bounded[0], -hard_backward_limit, hard_forward_limit);
        bounded[1] = std::clamp(
            bounded[1], -hard_lateral_limit, hard_lateral_limit);
        bounded[2] = std::clamp(
            bounded[2], -hard_yaw_limit, hard_yaw_limit);

        for (size_t axis = 0; axis < bounded.size(); ++axis) {
            const float rate = axis < 2 ? command_slew_lin : command_slew_yaw;
            if (rate <= 0.0f) {
                filtered_velocity_command[axis] = bounded[axis];
                continue;
            }
            const float max_delta = rate * step_dt;
            filtered_velocity_command[axis] += std::clamp(
                bounded[axis] - filtered_velocity_command[axis],
                -max_delta,
                max_delta);
        }
        return filtered_velocity_command;
    }

private:
    void record_policy_observation(
        const std::unordered_map<std::string, std::vector<float>>& observations)
    {
        const char* path = std::getenv("G1_POLICY_OBS_FILE");
        if (!path || path[0] == '\0' || observations.empty()) {
            return;
        }
        const auto preferred = observations.find("obs");
        const auto& values = preferred != observations.end()
            ? preferred->second
            : observations.begin()->second;
        const uint32_t magic = 0x3153424fu;  // OBS1
        const uint32_t dimension = static_cast<uint32_t>(values.size());
        const uint64_t stamp_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        std::ofstream stream(path, std::ios::binary | std::ios::app);
        if (!stream) {
            return;
        }
        stream.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        stream.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        stream.write(reinterpret_cast<const char*>(&stamp_ns), sizeof(stamp_ns));
        stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    float command_slew_lin = 0.0f;
    float command_slew_yaw = 0.0f;
    std::vector<float> filtered_velocity_command = {0.0f, 0.0f, 0.0f};
};

};
