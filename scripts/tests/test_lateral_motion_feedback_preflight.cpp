#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/envs/lateral_motion_feedback_preflight.h"

namespace
{

bool fails_with(
    bool sidecar,
    bool estimator,
    size_t policy_dim,
    size_t estimator_dim,
    const std::string& message)
{
    try {
        isaaclab::validate_lateral_motion_feedback_preflight(
            true, sidecar, estimator, policy_dim, estimator_dim);
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(message) != std::string::npos;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv)
{
    const YAML::Node motion = YAML::Load(R"(
base_ang_vel:
  params: {}
lateral_motion_feedback:
  params: {lateral_velocity_clip: 1.5}
)");
    assert(isaaclab::requires_lateral_motion_feedback(motion));

    const YAML::Node grouped_motion = YAML::Load(R"(
policy:
  base_ang_vel:
    params: {}
  lateral_motion_feedback:
    params: {}
)");
    assert(isaaclab::requires_lateral_motion_feedback(grouped_motion));

    const YAML::Node sensor_age = YAML::Load(R"(
base_ang_vel:
  params: {}
foot_sensor_age_lr:
  params: {}
)");
    assert(!isaaclab::requires_lateral_motion_feedback(sensor_age));

    // A non-motion policy receives no new source or dimension constraint.
    isaaclab::validate_lateral_motion_feedback_preflight(
        false, false, false, 0, 0);

    assert(fails_with(false, false, 1864, 0, "body_vy has no source"));

    // Either a non-empty sidecar or an exact causal-prefix estimator is valid.
    isaaclab::validate_lateral_motion_feedback_preflight(
        true, true, false, 1864, 0);
    isaaclab::validate_lateral_motion_feedback_preflight(
        true, false, true, 1864, 1862);

    assert(fails_with(
        false, true, 1864, 1864, "expected estimator input 1862"));
    // A configured but incompatible estimator is rejected even if a sidecar is
    // also present, instead of silently installing a broken overwrite path.
    assert(fails_with(
        true, true, 1864, 100, "expected estimator input 1862"));

    assert(!isaaclab::nonempty_motion_feedback_path(nullptr));
    assert(!isaaclab::nonempty_motion_feedback_path(""));
    assert(isaaclab::nonempty_motion_feedback_path("/tmp/motion.json"));
    assert(!isaaclab::should_load_lateral_velocity_estimator(
        false, "/tmp/estimator.onnx"));
    assert(!isaaclab::should_load_lateral_velocity_estimator(true, ""));
    assert(isaaclab::should_load_lateral_velocity_estimator(
        true, "/tmp/estimator.onnx"));

    // Optional integration check against repository deployment artifacts.
    if (argc >= 2) {
        isaaclab::ScalarOrtRunner estimator(argv[1]);
        assert(estimator.input_size() == 1862);
        isaaclab::validate_lateral_motion_feedback_preflight(
            true, false, true, 1864, estimator.input_size());
        const float prediction = estimator.infer(
            std::vector<float>(estimator.input_size(), 0.0f));
        assert(std::isfinite(prediction));
    }
    if (argc >= 4) {
        const YAML::Node motion_deploy = YAML::LoadFile(argv[2]);
        const YAML::Node age_deploy = YAML::LoadFile(argv[3]);
        assert(isaaclab::requires_lateral_motion_feedback(
            motion_deploy["observations"]));
        assert(!isaaclab::requires_lateral_motion_feedback(
            age_deploy["observations"]));
    }
    if (argc >= 5) {
        bool vector_output_rejected = false;
        try {
            isaaclab::ScalarOrtRunner not_an_estimator(argv[4]);
        } catch (const std::runtime_error& error) {
            vector_output_rejected = std::string(error.what()).find(
                "output must contain exactly one scalar") != std::string::npos;
        }
        assert(vector_output_rejected);
    }

    return 0;
}
