// Copyright (c) 2025 local foot-sensor bridge.
// SPDX-License-Identifier: BSD-3-Clause
//
// Shared-memory-style IPC between zorn ROS2 publisher (Python) and g1_ctrl (C++).
//
// Default path: /tmp/g1_foot_rl_obs.bin  (override with env G1_FOOT_BRIDGE_PATH)
//
// Binary layout F0T1 (little-endian, 40 bytes) — legacy Foot-Full:
//   uint32 magic     = 0x46305431  ('F','0','T','1')
//   uint32 seq
//   uint64 stamp_ns
//   float32 contact[2], normal[2], tangent[2]
//
// Binary layout F0T2 (48 bytes) — Adaptive-V2 (no shear on actor):
//   uint32 magic     = 0x46305432  ('F','0','T','2')
//   uint32 seq
//   uint64 stamp_ns
//   float32 contact[2], normal[2], load_ratio[2], valid, age_norm
//
// Binary layout F0M1 (400 bytes) — dual-foot 15xXYZ magnetic array:
//   uint32 magic     = 0x46304D31  ('F','0','M','1')
//   uint32 seq
//   uint64 stamp_ns
//   float32 valid_lr[2], source_age_s_lr[2], sample_period_s_lr[2]
//   float32 magnetic[2][15][3]
//
// Missing / stale → zeros + valid=0 (distinguish from true zero force when possible).

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>

namespace isaaclab
{
namespace foot_bridge
{

inline constexpr uint32_t kMagic = 0x46305431u; // F0T1
inline constexpr uint32_t kMagicV2 = 0x46305432u; // F0T2
inline constexpr uint32_t kMagicMagnetic = 0x46304D31u; // F0M1
inline constexpr size_t kPacketBytes = 40;
inline constexpr size_t kPacketBytesV2 = 48;
inline constexpr size_t kMagneticValues = 2 * 15 * 3;
inline constexpr size_t kPacketBytesMagnetic = 400;
// Drop readings older than this (ROS bridge died / paused).
inline constexpr double kStaleSec = 0.25;

struct Packet
{
    uint32_t magic = 0;
    uint32_t seq = 0;
    uint64_t stamp_ns = 0;
    float contact[2] = {0.f, 0.f};
    float normal[2] = {0.f, 0.f};
    float tangent[2] = {0.f, 0.f}; // F0T1 only
    float load[2] = {0.5f, 0.5f};  // F0T2
    float magnetic[kMagneticValues] = {};
    float valid_lr[2] = {0.f, 0.f};
    float age_lr[2] = {1.f, 1.f};
    float sample_period[2] = {0.02f, 0.02f};
    float valid = 0.f;
    float age = 1.f;
    bool ok = false;
};

inline std::string default_path()
{
    if (const char * env = std::getenv("G1_FOOT_BRIDGE_PATH")) {
        if (env[0] != '\0') {
            return std::string(env);
        }
    }
    return "/tmp/g1_foot_rl_obs.bin";
}

inline uint64_t now_ns()
{
    using clock = std::chrono::system_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count());
}

inline bool parse_packet(const char * buf, size_t n, Packet & out)
{
    if (n < 16) {
        return false;
    }
    std::memcpy(&out.magic, buf + 0, 4);
    std::memcpy(&out.seq, buf + 4, 4);
    std::memcpy(&out.stamp_ns, buf + 8, 8);

    if (out.magic == kMagic && n >= kPacketBytes) {
        std::memcpy(out.contact, buf + 16, 8);
        std::memcpy(out.normal, buf + 24, 8);
        std::memcpy(out.tangent, buf + 32, 8);
        // Derive load ratio from normals for F0T1 → V2 terms.
        const float s = std::fabs(out.normal[0]) + std::fabs(out.normal[1]) + 1e-3f;
        out.load[0] = std::fabs(out.normal[0]) / s;
        out.load[1] = std::fabs(out.normal[1]) / s;
        out.valid = 1.f;
        out.age = 0.f;
        out.ok = true;
        return true;
    }
    if (out.magic == kMagicV2 && n >= kPacketBytesV2) {
        std::memcpy(out.contact, buf + 16, 8);
        std::memcpy(out.normal, buf + 24, 8);
        std::memcpy(out.load, buf + 32, 8);
        std::memcpy(&out.valid, buf + 40, 4);
        std::memcpy(&out.age, buf + 44, 4);
        out.tangent[0] = out.tangent[1] = 0.f;
        out.ok = true;
        return true;
    }
    if (out.magic == kMagicMagnetic && n >= kPacketBytesMagnetic) {
        std::memcpy(out.valid_lr, buf + 16, 8);
        float source_age_s[2] = {0.f, 0.f};
        std::memcpy(source_age_s, buf + 24, 8);
        std::memcpy(out.sample_period, buf + 32, 8);
        std::memcpy(out.magnetic, buf + 40, kMagneticValues * sizeof(float));
        out.valid_lr[0] = std::clamp(out.valid_lr[0], 0.f, 1.f);
        out.valid_lr[1] = std::clamp(out.valid_lr[1], 0.f, 1.f);
        out.age_lr[0] = std::clamp(
            source_age_s[0] / static_cast<float>(kStaleSec), 0.f, 1.f);
        out.age_lr[1] = std::clamp(
            source_age_s[1] / static_cast<float>(kStaleSec), 0.f, 1.f);
        out.valid = std::min(out.valid_lr[0], out.valid_lr[1]);
        out.age = std::max(out.age_lr[0], out.age_lr[1]);
        for (float & value : out.magnetic) {
            value = std::clamp(value, -6.f, 6.f);
        }
        for (float & value : out.sample_period) {
            value = std::clamp(value, 0.001f, 0.25f);
        }
        out.ok = true;
        return true;
    }
    return false;
}

/** Read and validate one packet directly from the bridge file. */
inline Packet read_latest_uncached(bool * ok = nullptr)
{
    static std::mutex mu;

    std::lock_guard<std::mutex> lock(mu);
    const std::string path = default_path();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (ok) {
            *ok = false;
        }
        // No file: invalid sensor (not "standing with zero force" alone).
        Packet bad{};
        bad.valid = 0.f;
        bad.age = 1.f;
        return bad;
    }
    char buf[kPacketBytesMagnetic];
    ifs.read(buf, static_cast<std::streamsize>(kPacketBytesMagnetic));
    const auto got = static_cast<size_t>(ifs.gcount());
    // Accept F0T1 (40), F0T2 (48), or F0M1 (400).
    if (got < kPacketBytes) {
        if (ok) {
            *ok = false;
        }
        Packet bad{};
        bad.valid = 0.f;
        bad.age = 1.f;
        return bad;
    }
    Packet p;
    if (!parse_packet(buf, got, p)) {
        if (ok) {
            *ok = false;
        }
        Packet bad{};
        bad.valid = 0.f;
        bad.age = 1.f;
        return bad;
    }
    // Stale check — return zeros but mark valid=0 / age=1 so policy can degrade.
    const double age_s = static_cast<double>(now_ns() - p.stamp_ns) * 1e-9;
    if (age_s < 0.0 || age_s > kStaleSec) {
        if (ok) {
            *ok = false;
        }
        Packet stale{};
        stale.valid = 0.f;
        stale.age = 1.f;
        stale.valid_lr[0] = stale.valid_lr[1] = 0.f;
        stale.age_lr[0] = stale.age_lr[1] = 1.f;
        stale.seq = p.seq;
        stale.stamp_ns = p.stamp_ns;
        return stale;
    }
    // Fresh: fill age_norm if F0T1
    if (p.magic == kMagic) {
        p.age = static_cast<float>(std::min(1.0, age_s / kStaleSec));
        p.valid = 1.f;
    } else if (p.magic == kMagicMagnetic) {
        const float normalized_wall_age = static_cast<float>(
            std::clamp(age_s / kStaleSec, 0.0, 1.0));
        p.age_lr[0] = std::max(p.age_lr[0], normalized_wall_age);
        p.age_lr[1] = std::max(p.age_lr[1], normalized_wall_age);
        p.age = std::max(p.age_lr[0], p.age_lr[1]);
    }
    p.ok = true;
    if (ok) {
        *ok = true;
    }
    return p;
}

/**
 * Per-thread snapshot shared by every foot observation term in one observation
 * manager evaluation. An explicit scope is used instead of a wall-clock cache:
 * a slow control iteration must never inherit the previous iteration's packet,
 * while magnetic/period/valid terms in the same iteration remain coherent.
 */
struct ObservationSnapshotState
{
    Packet packet{};
    bool ok = false;
    bool captured = false;
    size_t depth = 0;
};

inline ObservationSnapshotState& observation_snapshot_state()
{
    thread_local ObservationSnapshotState state;
    return state;
}

inline void begin_observation_snapshot()
{
    auto& state = observation_snapshot_state();
    if (state.depth == 0) {
        // Read lazily on the first foot term. Non-foot policies must not incur
        // bridge file I/O merely because they share ObservationManager.
        state.captured = false;
        state.ok = false;
    }
    ++state.depth;
}

inline void end_observation_snapshot() noexcept
{
    auto& state = observation_snapshot_state();
    if (state.depth > 0) {
        --state.depth;
        if (state.depth == 0) {
            state.captured = false;
        }
    }
}

class ScopedObservationSnapshot
{
public:
    ScopedObservationSnapshot() { begin_observation_snapshot(); }
    ~ScopedObservationSnapshot() { end_observation_snapshot(); }

    ScopedObservationSnapshot(const ScopedObservationSnapshot&) = delete;
    ScopedObservationSnapshot& operator=(const ScopedObservationSnapshot&) = delete;
};

/** Return the current observation snapshot, or perform an uncached standalone read. */
inline Packet read_latest(bool * ok = nullptr)
{
    auto& state = observation_snapshot_state();
    if (state.depth > 0) {
        if (!state.captured) {
            state.packet = read_latest_uncached(&state.ok);
            state.captured = true;
        }
        if (ok) {
            *ok = state.ok;
        }
        return state.packet;
    }
    return read_latest_uncached(ok);
}

inline std::vector<float> contact_lr()
{
    const Packet p = read_latest();
    if (!p.ok || p.valid < 0.5f) {
        return {0.f, 0.f};
    }
    return {p.contact[0], p.contact[1]};
}

inline std::vector<float> normal_lr()
{
    const Packet p = read_latest();
    if (!p.ok || p.valid < 0.5f) {
        return {0.f, 0.f};
    }
    return {p.normal[0], p.normal[1]};
}

inline std::vector<float> tangent_lr()
{
    const Packet p = read_latest();
    if (!p.ok || p.valid < 0.5f) {
        return {0.f, 0.f};
    }
    // Some Isaac/PhysX contact-sensor configurations expose only normal
    // force, so policies trained there see tangent=0. Keep the raw bridge
    // packet intact for logging, but allow deploy/Sim2Sim to reproduce that
    // observation schema exactly. Default 1 preserves real shear sensors.
    float scale = 1.0f;
    if (const char * value = std::getenv("G1_FOOT_TANGENT_SCALE")) {
        scale = std::clamp(std::strtof(value, nullptr), 0.0f, 10.0f);
    }
    return {scale * p.tangent[0], scale * p.tangent[1]};
}

inline std::vector<float> friction_ratio_lr(float eps_scaled = 0.05f, float clip_max = 2.0f)
{
    const Packet p = read_latest();
    if (!p.ok || p.valid < 0.5f) {
        return {0.f, 0.f};
    }
    const float eps = std::max(eps_scaled, 1e-6f);
    float scale = 1.0f;
    if (const char * value = std::getenv("G1_FOOT_TANGENT_SCALE")) {
        scale = std::clamp(std::strtof(value, nullptr), 0.0f, 10.0f);
    }
    const float left = scale * std::fabs(p.tangent[0]) / (std::fabs(p.normal[0]) + eps);
    const float right = scale * std::fabs(p.tangent[1]) / (std::fabs(p.normal[1]) + eps);
    return {std::clamp(left, 0.f, clip_max), std::clamp(right, 0.f, clip_max)};
}

inline std::vector<float> load_ratio_lr()
{
    const Packet p = read_latest();
    if (!p.ok || p.valid < 0.5f) {
        return {0.5f, 0.5f};
    }
    return {p.load[0], p.load[1]};
}

inline std::vector<float> sensor_valid()
{
    const Packet p = read_latest();
    return {p.valid};
}

inline std::vector<float> sensor_age()
{
    const Packet p = read_latest();
    return {p.age};
}

inline std::vector<float> magnetic_lr()
{
    const Packet p = read_latest();
    if (!p.ok) {
        return std::vector<float>(kMagneticValues, 0.f);
    }
    return std::vector<float>(p.magnetic, p.magnetic + kMagneticValues);
}

inline std::vector<float> sensor_valid_lr()
{
    const Packet p = read_latest();
    return {p.valid_lr[0], p.valid_lr[1]};
}

inline std::vector<float> sensor_age_lr()
{
    const Packet p = read_latest();
    return {p.age_lr[0], p.age_lr[1]};
}

inline std::vector<float> sample_period_lr()
{
    const Packet p = read_latest();
    return {p.sample_period[0], p.sample_period[1]};
}

} // namespace foot_bridge
} // namespace isaaclab
