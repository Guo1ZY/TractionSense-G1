#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <unitree/robot/g1/loco/g1_loco_api.hpp>
#include <unitree/robot/g1/loco/g1_loco_client.hpp>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

namespace {
std::atomic_bool keep_running{true};

void signal_handler(int)
{
    keep_running.store(false);
}
}

void init_fsm_state(const std::string& lowcmd_topic)
{
    auto lowcmd_sub =
        std::make_shared<unitree::robot::g1::subscription::LowCmd>(lowcmd_topic);
    // Wait longer than SubscriptionBase's one-second timeout so a retained
    // historical DDS sample cannot be mistaken for an active command source.
    usleep(1.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical(
            "The other process is using {}, please close it first.",
            lowcmd_topic);
        unitree::robot::go2::shutdown();
        exit(EXIT_FAILURE);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>(lowcmd_topic);
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

    const char* user_ctrl_env = std::getenv("G1_USE_USER_CTRL");
    const bool use_user_ctrl =
        user_ctrl_env != nullptr && std::string(user_ctrl_env) == "1";
    const std::string lowcmd_topic =
        use_user_ctrl ? "rt/user_lowcmd" : "rt/lowcmd";

    std::unique_ptr<unitree::robot::g1::LocoClient> loco_client;
    if(use_user_ctrl)
    {
        loco_client = std::make_unique<unitree::robot::g1::LocoClient>();
        loco_client->Init();
        loco_client->SetTimeout(5.0f);
        int fsm_id = -1;
        const int32_t ret = loco_client->GetFsmId(fsm_id);
        if(ret != 0)
        {
            spdlog::critical("Cannot query G1 internal FSM (error {}).", ret);
            return EXIT_FAILURE;
        }
        if(fsm_id != 1)
        {
            spdlog::critical(
                "G1 EDU2 user control requires internal Passive (fsm_id=1); current fsm_id={}.",
                fsm_id);
            return EXIT_FAILURE;
        }
        spdlog::info("G1 EDU2 internal FSM is Passive; preparing user_lowcmd.");
    }

    init_fsm_state(lowcmd_topic);

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    if(use_user_ctrl)
    {
        // Give the EDU2 bridge a valid Passive user command before handing it
        // authority, matching Unitree's user_lowcmd switching sequence.
        usleep(0.1 * 1e6);
        const int32_t ret = loco_client->SwitchToUserCtrl();
        if(ret != 0)
        {
            spdlog::critical("SwitchToUserCtrl failed (error {}).", ret);
            return EXIT_FAILURE;
        }
        spdlog::info("G1 EDU2 switched to user control on {}.", lowcmd_topic);
    }

    std::cout << "\n=== Safe traction controls ===\n";
    std::cout << "  A / keyboard a: FixStand\n";
    std::cout << "  X / keyboard x: Velocity (forward command capped at 1.0 m/s)\n";
    std::cout << "  B / keyboard b: Passive / emergency software stop\n";
    std::cout << "  Harness bring-up starts at 0.20 m/s; keep the hardware E-stop ready.\n\n";

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    while (keep_running.load())
    {
        sleep(1);
    }

    fsm->stop();
    if(use_user_ctrl)
    {
        spdlog::info("Returning G1 EDU2 to internal Passive control...");
        const int32_t ret = loco_client->SwitchToInternalCtrl(
            unitree::robot::g1::InternalFsmMode::PASSIVE);
        if(ret != 0)
        {
            spdlog::critical("SwitchToInternalCtrl(PASSIVE) failed (error {}).", ret);
            return EXIT_FAILURE;
        }
        spdlog::info("G1 EDU2 internal Passive control restored.");
    }
    
    return 0;
}
