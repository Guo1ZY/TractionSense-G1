#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace isaaclab
{
// keyboard velocity commands (no joystick).
// change "velocity_commands" → "keyboard_velocity_commands" in deploy.yaml
// Keys give unit direction; scaled by G1_CMD_GAIN_LIN / G1_CMD_GAIN_YAW then clamped
// to deploy ranges (same semantics as stick velocity_commands).
//   G1_CMD_GAIN_LIN=1.0  → hold w ≈ 1.0 m/s
//   G1_CMD_GAIN_LIN=1.5  → hold w ≈ 1.5 m/s (if lin_vel_x max ≥ 1.5)
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    // Automation-only exact velocity command. The file contains "vx vy wz"
    // and is enabled only when G1_CMD_FILE is explicitly set. This avoids
    // restarting g1_ctrl for every cell in a MuJoCo speed/friction matrix.
    if (const char * path = std::getenv("G1_CMD_FILE")) {
        std::ifstream command_file(path);
        float vx = 0.0f;
        float vy = 0.0f;
        float wz = 0.0f;
        if (command_file >> vx >> vy >> wz) {
            return env->process_velocity_command({
                std::clamp(vx, cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>()),
                std::clamp(vy, cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>()),
                std::clamp(wz, cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>())
            });
        }
    }

    float gain_lin = 1.0f;
    float gain_yaw = 1.0f;
    if (const char * e = std::getenv("G1_CMD_GAIN_LIN")) {
        gain_lin = std::strtof(e, nullptr);
    }
    if (const char * e = std::getenv("G1_CMD_GAIN_YAW")) {
        gain_yaw = std::strtof(e, nullptr);
    }

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        const auto & d = key_commands[key];
        float vx = gain_lin * d[0];
        float vy = gain_lin * d[1];
        float wz = gain_yaw * d[2];
        cmd[0] = std::clamp(vx, cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
        cmd[1] = std::clamp(vy, cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
        cmd[2] = std::clamp(wz, cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    }
    return env->process_velocity_command(cmd);
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    std::string policy_setting = cfg["policy_dir"].as<std::string>();
    if (const char* override_dir = std::getenv("G1_POLICY_DIR")) {
        if (override_dir[0] != '\0') {
            policy_setting = override_dir;
            spdlog::warn(
                "G1_POLICY_DIR override is active: {}",
                policy_setting);
        }
    }
    auto policy_dir = param::parser_policy_dir(policy_setting);

    // A Hall candidate carries its causal risk model beside policy.onnx.
    // Selecting the policy slot is the explicit activation step; no global
    // config is rewritten by the installer.  An explicit environment value
    // still takes precedence for controlled A/B tests.
    const auto bundled_hall_risk =
        policy_dir / "exported" / "hall_risk.onnx";
    if (
        std::filesystem::is_regular_file(bundled_hall_risk)
        && std::getenv("G1_TRACTION_HALL_RISK_ONNX") == nullptr) {
        const auto model_path = bundled_hall_risk.string();
        ::setenv("G1_TRACTION_HALL_RISK_ONNX", model_path.c_str(), 1);
        spdlog::info(
            "Bundled Hall-risk model enabled: {} (Hall-to-force disabled)",
            model_path);
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    auto policy = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx",
        static_cast<size_t>(env->action_manager->total_action_dim()));
    const size_t deploy_observation_dim =
        env->observation_manager->policy_observation_size();
    if (policy->input_count() != 1 || policy->input_name() != "obs"
        || policy->input_size() != deploy_observation_dim) {
        throw std::runtime_error(
            "Policy ONNX/deploy.yaml observation ABI mismatch: expected one "
            "input named 'obs' with " + std::to_string(deploy_observation_dim)
            + " values, got count=" + std::to_string(policy->input_count())
            + ", name='" + (policy->input_count() ? policy->input_name() : "")
            + "', size="
            + std::to_string(policy->input_count() ? policy->input_size() : 0));
    }
    env->alg = std::move(policy);

    // Default limit 1.0 rad. For MuJoCo hang-band bring-up, set e.g. G1_BAD_ORI_LIMIT=1.8
    // to avoid instant Velocity→Passive while settling. Real robot: leave unset.
    float bad_ori_limit = 1.0f;
    if (const char * e = std::getenv("G1_BAD_ORI_LIMIT")) {
        bad_ori_limit = std::strtof(e, nullptr);
        if (bad_ori_limit < 0.2f) {
            bad_ori_limit = 0.2f;
        }
        if (bad_ori_limit > 3.0f) {
            bad_ori_limit = 3.0f;
        }
        spdlog::info("G1_BAD_ORI_LIMIT={:.2f} rad (sim-friendly if larger)", bad_ori_limit);
    }
    auto * env_ptr = env.get();
    this->registered_checks.emplace_back(
        std::make_pair(
            [env_ptr, bad_ori_limit]()->bool {
                return isaaclab::mdp::bad_orientation(env_ptr, bad_ori_limit);
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
