// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace isaaclab
{

/** Return true when an observation configuration explicitly contains the
 * lateral-motion term. Both the usual flat deploy schema and one level of
 * named observation groups are supported.
 */
inline bool requires_lateral_motion_feedback(const YAML::Node& observations)
{
    if (!observations || !observations.IsMap()) {
        return false;
    }
    for (auto it = observations.begin(); it != observations.end(); ++it) {
        if (it->first.as<std::string>() == "lateral_motion_feedback") {
            return true;
        }
        const YAML::Node child = it->second;
        if (child && child.IsMap() && child["lateral_motion_feedback"]) {
            return true;
        }
    }
    return false;
}

inline bool nonempty_motion_feedback_path(const char* path)
{
    return path != nullptr && path[0] != '\0';
}

inline bool should_load_lateral_velocity_estimator(
    bool feedback_required,
    const char* estimator_path)
{
    return feedback_required && nonempty_motion_feedback_path(estimator_path);
}

/** Fail closed before the control loop when body-vy has no deployable source,
 * or when an estimator cannot consume the exact causal prefix preceding the
 * final [body_vy, relative_heading] channels.
 */
inline void validate_lateral_motion_feedback_preflight(
    bool feedback_required,
    bool sidecar_configured,
    bool estimator_loaded,
    size_t policy_observation_size,
    size_t estimator_input_size = 0)
{
    // A sensor-age policy does not contain lateral_motion_feedback and must not
    // acquire new source or dimension requirements from this preflight.
    if (!feedback_required) {
        return;
    }
    if (policy_observation_size < 2) {
        throw std::runtime_error(
            "lateral_motion_feedback preflight: policy observation must end "
            "with [body_vy, relative_heading]");
    }
    if (estimator_loaded) {
        const size_t expected = policy_observation_size - 2;
        if (estimator_input_size != expected) {
            throw std::runtime_error(
                "lateral_motion_feedback preflight: estimator input mismatch; "
                "policy observation is "
                + std::to_string(policy_observation_size)
                + "-D, expected estimator input " + std::to_string(expected)
                + " (= policy_dim - 2), got "
                + std::to_string(estimator_input_size));
        }
    }
    if (!sidecar_configured && !estimator_loaded) {
        throw std::runtime_error(
            "lateral_motion_feedback preflight: body_vy has no source; set a "
            "non-empty G1_MOTION_FEEDBACK_PATH or load an estimator with "
            "G1_LATERAL_VELOCITY_ESTIMATOR_ONNX");
    }
}

}  // namespace isaaclab
