#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace isaaclab
{
namespace foot_bridge
{

constexpr uint32_t kMagicF0T1 = 0x46305431u;
constexpr uint32_t kMagicF0T2 = 0x46305432u;
constexpr uint32_t kMagicF0M1 = 0x46304D31u;
constexpr float kStaleSeconds = 0.25f;
constexpr size_t kMagneticValues = 2 * 15 * 3;
constexpr size_t kMaxPacketBytes = 400;

struct Sample
{
    uint32_t sequence = 0;
    uint64_t stamp_ns = 0;
    std::array<float, 2> contact = {0.0f, 0.0f};
    std::array<float, 2> normal = {0.0f, 0.0f};
    std::array<float, 2> tangent = {0.0f, 0.0f};
    std::array<float, 2> load = {0.5f, 0.5f};
    std::array<float, kMagneticValues> magnetic{};
    std::array<float, 2> valid_lr = {0.0f, 0.0f};
    std::array<float, 2> age_lr = {1.0f, 1.0f};
    std::array<float, 2> sample_period = {0.02f, 0.02f};
    float valid = 0.0f;
    float age = 1.0f;
};

inline uint64_t wall_time_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

inline std::string bridge_path()
{
    const char* configured = std::getenv("G1_FOOT_BRIDGE_PATH");
    return configured && configured[0] != '\0'
        ? std::string(configured)
        : std::string("/tmp/g1_foot_rl_obs.bin");
}

template <typename T>
inline T read_scalar(const std::array<char, kMaxPacketBytes>& bytes, size_t offset)
{
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

inline Sample read_sample_uncached()
{
    Sample sample;
    std::ifstream input(bridge_path(), std::ios::binary);
    if (!input) {
        return sample;
    }
    std::array<char, kMaxPacketBytes> bytes{};
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const auto count = static_cast<size_t>(input.gcount());
    if (count < 40) {
        return sample;
    }

    const uint32_t magic = read_scalar<uint32_t>(bytes, 0);
    sample.sequence = read_scalar<uint32_t>(bytes, 4);
    sample.stamp_ns = read_scalar<uint64_t>(bytes, 8);
    sample.contact = {
        read_scalar<float>(bytes, 16),
        read_scalar<float>(bytes, 20),
    };
    sample.normal = {
        read_scalar<float>(bytes, 24),
        read_scalar<float>(bytes, 28),
    };

    if (magic == kMagicF0T1) {
        sample.tangent = {
            read_scalar<float>(bytes, 32),
            read_scalar<float>(bytes, 36),
        };
        const float total = std::abs(sample.normal[0]) + std::abs(sample.normal[1]);
        if (total > 1.0e-6f) {
            sample.load = {
                std::abs(sample.normal[0]) / total,
                std::abs(sample.normal[1]) / total,
            };
        }
        sample.valid_lr = {1.0f, 1.0f};
        sample.age_lr = {0.0f, 0.0f};
        sample.valid = 1.0f;
        sample.age = 0.0f;
    } else if (magic == kMagicF0T2 && count >= 48) {
        sample.load = {
            read_scalar<float>(bytes, 32),
            read_scalar<float>(bytes, 36),
        };
        sample.valid = read_scalar<float>(bytes, 40);
        sample.age = read_scalar<float>(bytes, 44);
        sample.valid_lr = {sample.valid, sample.valid};
        sample.age_lr = {sample.age, sample.age};
    } else if (magic == kMagicF0M1 && count >= kMaxPacketBytes) {
        sample.valid_lr = {
            read_scalar<float>(bytes, 16),
            read_scalar<float>(bytes, 20),
        };
        const std::array<float, 2> source_age_seconds = {
            read_scalar<float>(bytes, 24),
            read_scalar<float>(bytes, 28),
        };
        sample.sample_period = {
            read_scalar<float>(bytes, 32),
            read_scalar<float>(bytes, 36),
        };
        for (size_t index = 0; index < kMagneticValues; ++index) {
            sample.magnetic[index] = read_scalar<float>(
                bytes, 40 + index * sizeof(float));
        }
        sample.age_lr = {
            std::clamp(source_age_seconds[0] / kStaleSeconds, 0.0f, 1.0f),
            std::clamp(source_age_seconds[1] / kStaleSeconds, 0.0f, 1.0f),
        };
        sample.valid = std::min(sample.valid_lr[0], sample.valid_lr[1]);
        sample.age = std::max(sample.age_lr[0], sample.age_lr[1]);
    } else {
        return Sample{};
    }

    const uint64_t now = wall_time_ns();
    const float wall_age = sample.stamp_ns > 0 && now >= sample.stamp_ns
        ? static_cast<float>(now - sample.stamp_ns) * 1.0e-9f
        : kStaleSeconds + 1.0f;
    if (!std::isfinite(wall_age) || wall_age > kStaleSeconds) {
        sample.contact = {0.0f, 0.0f};
        sample.normal = {0.0f, 0.0f};
        sample.tangent = {0.0f, 0.0f};
        sample.load = {0.5f, 0.5f};
        sample.magnetic.fill(0.0f);
        sample.valid_lr = {0.0f, 0.0f};
        sample.age_lr = {1.0f, 1.0f};
        sample.valid = 0.0f;
        sample.age = 1.0f;
    } else {
        const float normalized_wall_age = std::clamp(
            wall_age / kStaleSeconds, 0.0f, 1.0f);
        for (float& value : sample.age_lr) {
            value = std::max(value, normalized_wall_age);
        }
        sample.age = std::clamp(
            std::max(sample.age, normalized_wall_age), 0.0f, 1.0f);
        sample.valid = std::clamp(sample.valid, 0.0f, 1.0f);
    }
    for (float& value : sample.contact) value = std::clamp(value, 0.0f, 1.0f);
    for (float& value : sample.normal) value = std::max(0.0f, value);
    for (float& value : sample.tangent) value = std::max(0.0f, value);
    for (float& value : sample.load) value = std::clamp(value, 0.0f, 1.0f);
    for (float& value : sample.magnetic) value = std::clamp(value, -6.0f, 6.0f);
    for (float& value : sample.valid_lr) value = std::clamp(value, 0.0f, 1.0f);
    for (float& value : sample.age_lr) value = std::clamp(value, 0.0f, 1.0f);
    for (float& value : sample.sample_period) {
        value = std::clamp(value, 0.001f, 0.25f);
    }
    return sample;
}

inline Sample sample()
{
    static std::mutex mutex;
    static Sample cached;
    static auto last_read = std::chrono::steady_clock::time_point::min();
    std::lock_guard<std::mutex> lock(mutex);
    const auto now = std::chrono::steady_clock::now();
    if (last_read == std::chrono::steady_clock::time_point::min()
        || now - last_read >= std::chrono::milliseconds(2)) {
        cached = read_sample_uncached();
        last_read = now;
    }
    return cached;
}

inline std::vector<float> contact_lr()
{
    const auto value = sample().contact;
    return {value[0], value[1]};
}

inline std::vector<float> normal_lr()
{
    const auto value = sample().normal;
    return {value[0], value[1]};
}

inline std::vector<float> tangent_lr()
{
    const auto value = sample().tangent;
    float scale = 1.0f;
    if (const char* configured = std::getenv("G1_FOOT_TANGENT_SCALE")) {
        scale = std::max(0.0f, std::strtof(configured, nullptr));
    }
    return {value[0] * scale, value[1] * scale};
}

inline std::vector<float> friction_ratio_lr(float epsilon, float clip_max)
{
    const auto value = sample();
    float scale = 1.0f;
    if (const char* configured = std::getenv("G1_FOOT_TANGENT_SCALE")) {
        scale = std::max(0.0f, std::strtof(configured, nullptr));
    }
    return {
        std::clamp(scale * value.tangent[0] / (value.normal[0] + epsilon), 0.0f, clip_max),
        std::clamp(scale * value.tangent[1] / (value.normal[1] + epsilon), 0.0f, clip_max),
    };
}

inline std::vector<float> load_ratio_lr()
{
    const auto value = sample().load;
    return {value[0], value[1]};
}

inline std::vector<float> sensor_valid()
{
    return {sample().valid};
}

inline std::vector<float> sensor_age()
{
    return {sample().age};
}

inline std::vector<float> magnetic_lr()
{
    const auto value = sample().magnetic;
    return std::vector<float>(value.begin(), value.end());
}

inline std::vector<float> sensor_valid_lr()
{
    const auto value = sample().valid_lr;
    return {value[0], value[1]};
}

inline std::vector<float> sensor_age_lr()
{
    const auto value = sample().age_lr;
    return {value[0], value[1]};
}

inline std::vector<float> sample_period_lr()
{
    const auto value = sample().sample_period;
    return {value[0], value[1]};
}

}  // namespace foot_bridge
}  // namespace isaaclab
