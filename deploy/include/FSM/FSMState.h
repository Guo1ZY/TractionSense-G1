#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include "local_xbox_joystick.h"
#include <cctype>
#include <algorithm>
#include <memory>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                // Gamepad path (from MuJoCo wireless / real remote).
                // Also accept simple keyboard fallbacks when the g1_ctrl terminal has focus:
                //   A / a → same as A.on_pressed
                //   X / x → same as X.on_pressed
                //   B / b → same as B.on_pressed
                registered_checks.emplace_back(
                    std::make_pair(
                        [func, condition, target_fsm]()->bool{
                            if (func(FSMState::lowstate->joystick)) {
                                return true;
                            }
                            // Keyboard fallback for bare A/X/B (and .on_pressed variants)
                            if (FSMState::keyboard) {
                                auto k = FSMState::keyboard->key();
                                if (k.empty() || !FSMState::keyboard->on_pressed) {
                                    return false;
                                }
                                auto lower = k;
                                for (auto & ch : lower) {
                                    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                                }
                                // normalize condition: strip spaces, lower
                                std::string c = condition;
                                c.erase(std::remove_if(c.begin(), c.end(), ::isspace), c.end());
                                for (auto & ch : c) {
                                    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                                }
                                bool hit = false;
                                if ((c == "a.on_pressed" || c == "a") && lower == "a") {
                                    hit = true;
                                }
                                if ((c == "x.on_pressed" || c == "x") && lower == "x") {
                                    hit = true;
                                }
                                if ((c == "b.on_pressed" || c == "b") && lower == "b") {
                                    hit = true;
                                }
                                if (hit) {
                                    spdlog::info("FSM keyboard '{}' → {}", lower, target_fsm);
                                    return true;
                                }
                            }
                            return false;
                        },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        // Overlay local pad (g1_ctrl reads /dev/input/js0 itself).
        // Does not depend on MuJoCo use_joystick / wireless_remote.
        if (local_js && local_js->ok()) {
            local_js->update();
            auto key = local_js->combine();
            lowstate->joystick.extract(key);
        }
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
    /** Optional direct gamepad in g1_ctrl (Xbox map). */
    static std::shared_ptr<LocalXBoxJoystick> local_js;
};