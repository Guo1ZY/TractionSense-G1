#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();
std::shared_ptr<LocalXBoxJoystick> FSMState::local_js = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Direct gamepad in this process (does not need MuJoCo use_joystick)
    {
        std::string js_dev = "/dev/input/js0";
        if (const char * env = std::getenv("G1_JS_DEVICE")) {
            js_dev = env;
        }
        FSMState::local_js = std::make_shared<LocalXBoxJoystick>(js_dev, 16);
        if (FSMState::local_js->ok()) {
            spdlog::info("Local gamepad OK: {} (A=stand, X=velocity, B=passive)", js_dev);
        } else {
            spdlog::warn("Local gamepad unavailable; use keyboard a/x/b in THIS terminal");
        }
    }

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "\n=== Controls (this process) ===\n";
    std::cout << "  Gamepad (Xbox map):  A = FixStand,  X = Velocity,  B = Passive\n";
    std::cout << "  Keyboard (THIS terminal focus):  a / x / b  same as above\n";
    std::cout << "  Order: A (stand 2s) → X (walk)\n\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

