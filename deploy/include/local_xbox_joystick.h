// Copyright (c) 2025 local — read Xbox-compatible pad in g1_ctrl directly.
// Avoids depending on MuJoCo use_joystick / wireless_remote forwarding.
#pragma once

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

#include <fcntl.h>
#include <linux/joystick.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <cerrno>

/**
 * Linux /dev/input/js* reader with Xbox 360-style button map (same as unitree_mujoco XBoxJoystick).
 */
class LocalXBoxJoystick : public unitree::common::UnitreeJoystick
{
public:
    explicit LocalXBoxJoystick(const std::string & device = "/dev/input/js0", int bits = 16)
    {
        max_value_ = 1 << (bits - 1);
        fd_ = ::open(device.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd_ < 0) {
            std::cerr << "[LocalXBoxJoystick] open failed: " << device
                      << " (errno=" << errno << "). Keyboard a/x/b still available.\n";
            return;
        }
        std::cout << "[LocalXBoxJoystick] opened " << device
                  << "  map=xbox  A/X/B for FSM, sticks for walk cmd\n";
        // drain init events
        js_event e;
        while (::read(fd_, &e, sizeof(e)) == static_cast<ssize_t>(sizeof(e))) {
        }
    }

    ~LocalXBoxJoystick()
    {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool ok() const { return fd_ >= 0; }

    void update() override
    {
        if (fd_ < 0) {
            return;
        }
        js_event e;
        while (::read(fd_, &e, sizeof(e)) == static_cast<ssize_t>(sizeof(e))) {
            const uint8_t type = e.type & ~JS_EVENT_INIT;
            if (type == JS_EVENT_BUTTON) {
                if (e.number < button_.size()) {
                    button_[e.number] = e.value ? 1 : 0;
                }
            } else if (type == JS_EVENT_AXIS) {
                if (e.number < axis_.size()) {
                    axis_[e.number] = e.value;
                }
            }
        }

        // Xbox 360 / xpad mapping (Linux)
        back(button_[6]);
        start(button_[7]);
        LB(button_[4]);
        RB(button_[5]);
        A(button_[0]);
        B(button_[1]);
        X(button_[2]);
        Y(button_[3]);
        up(axis_[7] < -1000);
        down(axis_[7] > 1000);
        left(axis_[6] < -1000);
        right(axis_[6] > 1000);
        // triggers often rest at -max and go to +max
        LT(axis_[2] > 0 ? float(axis_[2]) / float(max_value_) : 0.f);
        RT(axis_[5] > 0 ? float(axis_[5]) / float(max_value_) : 0.f);
        lx(float(axis_[0]) / float(max_value_));
        ly(-float(axis_[1]) / float(max_value_));
        rx(float(axis_[3]) / float(max_value_));
        ry(-float(axis_[4]) / float(max_value_));
    }

private:
    int fd_ = -1;
    int max_value_ = 32768;
    std::array<int, 16> button_{};
    std::array<int, 16> axis_{};
};
