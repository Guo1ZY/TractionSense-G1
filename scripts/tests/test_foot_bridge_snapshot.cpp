#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "isaaclab/envs/mdp/observations/foot_bridge.h"

namespace
{

uint64_t now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

template <typename T>
void put(std::array<char, isaaclab::foot_bridge::kPacketBytesMagnetic>& bytes,
         size_t offset,
         const T& value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void write_packet(const std::filesystem::path& path,
                  uint32_t sequence,
                  float magnetic_value,
                  float period,
                  float valid)
{
    using namespace isaaclab::foot_bridge;
    std::array<char, kPacketBytesMagnetic> bytes{};
    put(bytes, 0, kMagicMagnetic);
    put(bytes, 4, sequence);
    const uint64_t stamp = now_ns();
    put(bytes, 8, stamp);
    put(bytes, 16, valid);
    put(bytes, 20, valid);
    const float age = 0.0f;
    put(bytes, 24, age);
    put(bytes, 28, age);
    put(bytes, 32, period);
    put(bytes, 36, period);
    for (size_t index = 0; index < kMagneticValues; ++index) {
        put(bytes, 40 + index * sizeof(float), magnetic_value);
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    assert(output.good());
}

}

int main()
{
    using namespace isaaclab::foot_bridge;
    const auto path = std::filesystem::temp_directory_path()
        / ("g1_foot_snapshot_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    const std::string path_text = path.string();
    assert(::setenv("G1_FOOT_BRIDGE_PATH", path_text.c_str(), 1) == 0);

    write_packet(path, 1, 1.0f, 0.02f, 1.0f);
    {
        ScopedObservationSnapshot snapshot;
        const auto magnetic = magnetic_lr();
        assert(magnetic.size() == kMagneticValues);
        assert(magnetic.front() == 1.0f);

        // Simulate a new packet arriving between observation-term calls.
        // Period and validity must remain paired with sequence 1.
        write_packet(path, 2, 2.0f, 0.04f, 0.25f);
        const auto period = sample_period_lr();
        const auto valid = sensor_valid_lr();
        assert(period[0] == 0.02f && period[1] == 0.02f);
        assert(valid[0] == 1.0f && valid[1] == 1.0f);
    }

    // A new control/observation scope reads sequence 2 immediately; there is
    // no wall-clock cache that can accidentally retain the preceding frame.
    {
        ScopedObservationSnapshot snapshot;
        const auto magnetic = magnetic_lr();
        const auto period = sample_period_lr();
        const auto valid = sensor_valid_lr();
        assert(magnetic.front() == 2.0f);
        assert(period[0] == 0.04f && period[1] == 0.04f);
        assert(valid[0] == 0.25f && valid[1] == 0.25f);
    }

    std::filesystem::remove(path);
    ::unsetenv("G1_FOOT_BRIDGE_PATH");
    return 0;
}
