// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/foot_bridge.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_effort)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_effort;

    try {
        const std::vector<int> joint_ids =
            params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            Eigen::VectorXf selected(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                selected[i] = data[joint_ids[i]];
            }
            data = selected;
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    float vx = joystick->ly();
    float vy = -joystick->lx();
    float wz = -joystick->rx();
    if (const char* path = std::getenv("G1_CMD_FILE")) {
        std::ifstream input(path);
        input >> vx >> vy >> wz;
        if (!input) {
            vx = vy = wz = 0.0f;
        }
    }
    float gain_lin = 1.0f;
    if (const char* value = std::getenv("G1_CMD_GAIN_LIN")) {
        gain_lin = std::clamp(std::strtof(value, nullptr), 0.0f, 1.0f);
    }
    vx *= gain_lin;
    vy *= gain_lin;

    float deadband_lin = 0.0f;
    float deadband_yaw = 0.0f;
    try {
        const auto deadband = env->cfg["commands"]["base_velocity"]["deadband"];
        if (deadband) {
            deadband_lin = std::max(0.0f, deadband["lin"].as<float>(0.0f));
            deadband_yaw = std::max(0.0f, deadband["yaw"].as<float>(0.0f));
        }
    } catch (const std::exception&) {
    }
    if (const char* value = std::getenv("G1_CMD_DEADBAND_LIN")) {
        deadband_lin = std::clamp(std::strtof(value, nullptr), 0.0f, 0.20f);
    }
    if (const char* value = std::getenv("G1_CMD_DEADBAND_YAW")) {
        deadband_yaw = std::max(0.0f, std::strtof(value, nullptr));
    }
    if (std::abs(vx) <= deadband_lin) vx = 0.0f;
    if (std::abs(vy) <= deadband_lin) vy = 0.0f;
    if (std::abs(wz) <= deadband_yaw) wz = 0.0f;

    obs[0] = std::clamp(vx, cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(vy, cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(wz, cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    return env->filter_velocity_command(obs);
}

REGISTER_OBSERVATION(effective_friction_mu)
{
    float mu = 0.8f;
    const char* path = std::getenv("G1_FRICTION_ORACLE_PATH");
    if (path && path[0] != '\0') {
        std::ifstream input(path);
        float value = mu;
        input >> value;
        if (input && std::isfinite(value)) {
            mu = value;
        }
    }
    return {std::clamp(mu, 0.02f, 1.50f)};
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

REGISTER_OBSERVATION(foot_contact)
{
    return isaaclab::foot_bridge::contact_lr();
}

REGISTER_OBSERVATION(foot_normal_force)
{
    return isaaclab::foot_bridge::normal_lr();
}

REGISTER_OBSERVATION(foot_tangent_force)
{
    return isaaclab::foot_bridge::tangent_lr();
}

REGISTER_OBSERVATION(foot_friction_ratio)
{
    float eps_n = 5.0f;
    float clip_max = 2.0f;
    try {
        eps_n = params["eps"].as<float>();
    } catch (const std::exception&) {
    }
    try {
        clip_max = params["clip_max"].as<float>();
    } catch (const std::exception&) {
    }
    return isaaclab::foot_bridge::friction_ratio_lr(eps_n * 0.01f, clip_max);
}

REGISTER_OBSERVATION(foot_load_ratio)
{
    return isaaclab::foot_bridge::load_ratio_lr();
}

REGISTER_OBSERVATION(foot_sensor_valid)
{
    return isaaclab::foot_bridge::sensor_valid();
}

REGISTER_OBSERVATION(foot_sensor_age)
{
    return isaaclab::foot_bridge::sensor_age();
}

REGISTER_OBSERVATION(foot_magnetic_array)
{
    return isaaclab::foot_bridge::magnetic_lr();
}

REGISTER_OBSERVATION(foot_sensor_valid_lr)
{
    return isaaclab::foot_bridge::sensor_valid_lr();
}

REGISTER_OBSERVATION(foot_sensor_age_lr)
{
    return isaaclab::foot_bridge::sensor_age_lr();
}

REGISTER_OBSERVATION(foot_sample_period_lr)
{
    return isaaclab::foot_bridge::sample_period_lr();
}

}
}
