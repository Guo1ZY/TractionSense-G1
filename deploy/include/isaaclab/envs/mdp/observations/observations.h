// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>

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

    // Sticks: LY forward, LX strafe, RX yaw. Default gain=1 (safe for model_4000).
    // Only raise via env after a wide-cmd fine-tune.
    float gain_lin = 1.0f;
    float gain_yaw = 1.0f;
    if (const char * e = std::getenv("G1_CMD_GAIN_LIN")) {
        gain_lin = std::strtof(e, nullptr);
    }
    if (const char * e = std::getenv("G1_CMD_GAIN_YAW")) {
        gain_yaw = std::strtof(e, nullptr);
    }

    float vx = gain_lin * joystick->ly();
    float vy = gain_lin * (-joystick->lx());
    float wz = gain_yaw * (-joystick->rx());

    obs[0] = std::clamp(vx, cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(vy, cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(wz, cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return env->process_velocity_command(obs);
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

// ---------------------------------------------------------------------------
// Foot-sensor terms (align with unitree_rl_lab mdp/foot_sensor.py).
// Live values come from the zorn ROS → file bridge (see foot_bridge.h).
// If the bridge is down / stale, terms degrade to zeros (510-dim ONNX still runs).
// Order: left, right. observation_manager stacks history_length from deploy.yaml.
// ---------------------------------------------------------------------------

#include "isaaclab/envs/mdp/observations/foot_bridge.h"

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

REGISTER_OBSERVATION(foot_force_history)
{
    // Not used by foot policy.onnx (policy only has contact/normal/tangent).
    int history_steps = 3;
    try {
        history_steps = params["history_steps"].as<int>();
    } catch (const std::exception &) {
    }
    return std::vector<float>(static_cast<size_t>(history_steps * 2 * 3), 0.0f);
}

REGISTER_OBSERVATION(foot_force_vector)
{
    return std::vector<float>(6, 0.0f);
}

// Foot-Adaptive-V2 terms (schema foot_obs_v2). Zeros if bridge lacks fields.
REGISTER_OBSERVATION(foot_load_ratio)
{
    return isaaclab::foot_bridge::load_ratio_lr();
}

inline bool read_json_scalar(
    const std::string & path,
    const std::string & key,
    float & value)
{
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    const std::string payload(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    const std::string token = "\"" + key + "\"";
    const auto key_pos = payload.find(token);
    if (key_pos == std::string::npos) {
        return false;
    }
    const auto colon_pos = payload.find(':', key_pos + token.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    const char * begin = payload.c_str() + colon_pos + 1;
    char * end = nullptr;
    const float candidate = std::strtof(begin, &end);
    if (end == begin || !std::isfinite(candidate)) {
        return false;
    }
    value = candidate;
    return true;
}

inline std::vector<float> lateral_motion_feedback_impl(
    ManagerBasedRLEnv * env,
    YAML::Node params)
{
    // Heading is directly deployable: latch the IMU yaw when the policy state
    // resets, then report the wrapped relative error.  The lateral-velocity
    // source is deliberately explicit.  MuJoCo supplies exact world velocity
    // through its sidecar for Oracle Sim2Sim validation; hardware must replace
    // this with a contact-aided estimator before real deployment.
    const auto & quat = env->robot->data.root_quat_w;
    const float yaw = std::atan2(
        2.0f * (quat.w() * quat.z() + quat.x() * quat.y()),
        1.0f - 2.0f * (quat.y() * quat.y() + quat.z() * quat.z()));
    static std::unordered_map<ManagerBasedRLEnv *, float> initial_yaw;
    if (initial_yaw.find(env) == initial_yaw.end() || env->episode_length <= 1) {
        initial_yaw[env] = yaw;
    }
    const float heading_error = std::atan2(
        std::sin(yaw - initial_yaw[env]),
        std::cos(yaw - initial_yaw[env]));

    float lateral_velocity = 0.0f;
    if (const char * path = std::getenv("G1_MOTION_FEEDBACK_PATH")) {
        if (path[0] != '\0') {
            float world_vx = 0.0f;
            float world_vy = 0.0f;
            if (read_json_scalar(path, "vx", world_vx)
                && read_json_scalar(path, "vy", world_vy)) {
                lateral_velocity =
                    -std::sin(yaw) * world_vx + std::cos(yaw) * world_vy;
            }
        }
    }
    float lateral_gain = 1.0f;
    float heading_gain = 1.0f;
    if (const char * value = std::getenv("G1_MOTION_VY_GAIN")) {
        lateral_gain = std::clamp(std::strtof(value, nullptr), 0.0f, 4.0f);
    }
    if (const char * value = std::getenv("G1_MOTION_HEADING_GAIN")) {
        heading_gain = std::clamp(std::strtof(value, nullptr), 0.0f, 4.0f);
    }

    float lateral_clip = 1.5f;
    float heading_clip = 1.0f;
    try {
        lateral_clip = std::max(
            0.0f, params["lateral_velocity_clip"].as<float>());
    } catch (const std::exception &) {
    }
    try {
        heading_clip = std::max(
            0.0f, params["heading_error_clip"].as<float>());
    } catch (const std::exception &) {
    }
    return {
        std::clamp(
            lateral_gain * lateral_velocity, -lateral_clip, lateral_clip),
        std::clamp(heading_gain * heading_error, -heading_clip, heading_clip),
    };
}

REGISTER_OBSERVATION(foot_sensor_valid)
{
    // The 641-D Motion-Feedback Teacher intentionally reuses this field's
    // column position (630:640) to preserve checkpoint compatibility.  Its
    // generated deploy.yaml is identified by these motion-specific params.
    if (params["lateral_velocity_clip"] || params["heading_error_clip"]) {
        return lateral_motion_feedback_impl(env, params);
    }
    return isaaclab::foot_bridge::sensor_valid();
}

REGISTER_OBSERVATION(lateral_motion_feedback)
{
    return lateral_motion_feedback_impl(env, params);
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

REGISTER_OBSERVATION(foot_planar_vel)
{
    // Optional; zeros until FK body velocity is wired in deploy.
    return std::vector<float>(2, 0.0f);
}

REGISTER_OBSERVATION(foot_friction_ratio)
{
    // F0T1 carries Fn/Ft already scaled by the bridge.  Python's eps is in
    // newtons and training applies force scale=0.01, hence eps * 0.01 here.
    float eps_n = 5.0f;
    float clip_max = 2.0f;
    try {
        eps_n = params["eps"].as<float>();
    } catch (const std::exception &) {
    }
    try {
        clip_max = params["clip_max"].as<float>();
    } catch (const std::exception &) {
    }
    return isaaclab::foot_bridge::friction_ratio_lr(eps_n * 0.01f, clip_max);
}

REGISTER_OBSERVATION(foot_slip_proxy)
{
    return std::vector<float>(2, 0.0f);
}

inline std::vector<float> friction_oracle_scalar(YAML::Node params)
{
    float default_mu = 0.20f;
    float clip_max = 1.20f;
    try {
        default_mu = params["default_mu"].as<float>();
    } catch (const std::exception &) {
    }
    try {
        clip_max = params["clip_max"].as<float>();
    } catch (const std::exception &) {
    }

    const char *configured = std::getenv("G1_FRICTION_ORACLE_PATH");
    const std::string path =
        (configured && configured[0] != '\0') ? configured : "/tmp/g1_ground_mu";
    float mu = default_mu;
    std::ifstream input(path);
    float candidate = -1.0f;
    if (input >> candidate && std::isfinite(candidate) && candidate >= 0.0f) {
        mu = candidate;
    }
    return {std::clamp(mu, 0.0f, std::max(clip_max, 0.0f))};
}

REGISTER_OBSERVATION(ground_friction_mu)
{
    return friction_oracle_scalar(params);
}

// Python exports the privileged teacher term under its attribute name rather
// than the underlying function name.  Register both names so the generated
// 641-D deploy.yaml is accepted by g1_ctrl.
REGISTER_OBSERVATION(effective_friction_mu)
{
    return friction_oracle_scalar(params);
}

}
}
