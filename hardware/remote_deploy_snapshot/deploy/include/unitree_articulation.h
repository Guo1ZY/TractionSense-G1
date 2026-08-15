// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // Joint positions, velocities and motor-estimated efforts.  Keep the
        // exact exported policy order rather than raw SDK motor order.
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            const auto sdk_id = data.joint_ids_map[i];
            data.joint_pos[i] = lowstate->msg_.motor_state()[sdk_id].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[sdk_id].dq();
            data.joint_effort[i] = lowstate->msg_.motor_state()[sdk_id].tau_est();
        }
    }

    LowStatePtr lowstate;
};

}
