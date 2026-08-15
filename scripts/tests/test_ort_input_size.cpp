#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "isaaclab/algorithms/algorithms.h"

int main(int argc, char** argv)
{
    isaaclab::require_exact_ort_input_size("obs", 1864, 1864);

    bool short_failed_closed = false;
    try {
        isaaclab::require_exact_ort_input_size("obs", 1862, 1864);
    } catch (const std::runtime_error& error) {
        short_failed_closed = std::string(error.what()).find(
            "expected 1864, got 1862") != std::string::npos;
    }
    assert(short_failed_closed);

    bool long_failed_closed = false;
    try {
        isaaclab::require_exact_ort_input_size("obs", 1866, 1864);
    } catch (const std::runtime_error& error) {
        long_failed_closed = std::string(error.what()).find(
            "expected 1864, got 1866") != std::string::npos;
    }
    assert(long_failed_closed);

    if (argc == 2) {
        isaaclab::OrtRunner runner(argv[1], 29);
        assert(runner.output_size() == 29);
        assert(runner.input_count() == 1);
        assert(runner.input_name() == "obs");
        assert(runner.input_size() == 1864);
        std::unordered_map<std::string, std::vector<float>> valid_observation{
            {"obs", std::vector<float>(1864, 0.0f)},
        };
        const auto action = runner.act(std::move(valid_observation));
        assert(action.size() == 29);
        for (const float value : action) {
            assert(std::isfinite(value));
        }

        std::unordered_map<std::string, std::vector<float>> observation{
            {"obs", std::vector<float>(1862, 0.0f)},
        };
        bool runner_failed_before_inference = false;
        try {
            static_cast<void>(runner.act(std::move(observation)));
        } catch (const std::runtime_error& error) {
            runner_failed_before_inference = std::string(error.what()).find(
                "expected 1864, got 1862") != std::string::npos;
        }
        assert(runner_failed_before_inference);

        bool action_dim_failed_closed = false;
        try {
            isaaclab::OrtRunner wrong_action_contract(argv[1], 28);
        } catch (const std::runtime_error& error) {
            action_dim_failed_closed = std::string(error.what()).find(
                "expected 28, got 29") != std::string::npos;
        }
        assert(action_dim_failed_closed);
    }
    return 0;
}
