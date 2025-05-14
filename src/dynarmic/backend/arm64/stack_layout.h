/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstdint>
#include <bit>
#include <type_traits>

#include <mcl/stdint.hpp>

namespace Dynarmic::Backend::Arm64 {

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4324)  // Structure was padded due to alignment specifier
#endif

constexpr size_t SpillCount = 64;

// Use [[nodiscard]] to encourage correct usage and prevent accidental misuse.
struct alignas(16) [[nodiscard]] RSBEntry {
    u64 target;
    u64 code_ptr;

    constexpr RSBEntry() noexcept = default;
    constexpr RSBEntry(u64 target_, u64 code_ptr_) noexcept
        : target(target_), code_ptr(code_ptr_) {}
};

constexpr size_t RSBCount = 8;
constexpr u64 RSBIndexMask = (RSBCount - 1) * sizeof(RSBEntry);

// Use [[nodiscard]] and default member initializers for safety and clarity.
struct alignas(16) [[nodiscard]] StackLayout {
    std::array<RSBEntry, RSBCount> rsb{};

    // Use std::array for spill slots, but consider using std::span in the future for more flexibility.
    std::array<std::array<u64, 2>, SpillCount> spill{};

    u32 rsb_ptr = 0;

    s64 cycles_to_run = 0;

    u32 save_host_fpcr = 0;

    bool check_bit = false;

    // Provide constexpr default constructor for compile-time initialization.
    constexpr StackLayout() noexcept = default;

    // Provide explicit clear/reset method for performance and clarity.
    void Clear() noexcept {
        // Use std::fill for optimal zeroing.
        std::fill(rsb.begin(), rsb.end(), RSBEntry{});
        for (auto& s : spill) {
            s[0] = 0;
            s[1] = 0;
        }
        rsb_ptr = 0;
        cycles_to_run = 0;
        save_host_fpcr = 0;
        check_bit = false;
    }
};

#ifdef _MSC_VER
#    pragma warning(pop)
#endif

static_assert(sizeof(StackLayout) % 16 == 0, "StackLayout must be 16-byte aligned");
static_assert(std::is_trivially_copyable_v<StackLayout>, "StackLayout should be trivially copyable for performance");

}  // namespace Dynarmic::Backend::Arm64
