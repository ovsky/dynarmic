/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <type_traits>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/A64/a64_location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

struct alignas(16) A64JitState {
    // Use std::array for fixed-size registers, ensure cache alignment.
    std::array<u64, 31> reg{}; // General-purpose registers X0-X30
    u64 sp = 0;                // Stack pointer
    u64 pc = 0;                // Program counter

    u32 cpsr_nzcv = 0;         // Condition flags

    alignas(16) std::array<u64, 64> vec{}; // SIMD/floating-point registers Q0-Q31 (128-bit, stored as 2x u64 per Q)

    u32 exclusive_state = 0;   // Exclusive monitor state

    u32 fpsr = 0;              // Floating-point Status Register
    u32 fpcr = 0;              // Floating-point Control Register

    // Defaulted special member functions for optimal performance and clarity
    constexpr A64JitState() noexcept = default;
    constexpr A64JitState(const A64JitState&) noexcept = default;
    constexpr A64JitState(A64JitState&&) noexcept = default;
    constexpr A64JitState& operator=(const A64JitState&) noexcept = default;
    constexpr A64JitState& operator=(A64JitState&&) noexcept = default;
    ~A64JitState() = default;

    // Mark as [[nodiscard]] to encourage correct usage
    [[nodiscard]]
    IR::LocationDescriptor GetLocationDescriptor() const noexcept {
        // Use local constexpr for masks/shifts for better optimization
        constexpr u64 fpcr_mask = A64::LocationDescriptor::fpcr_mask;
        constexpr u64 fpcr_shift = A64::LocationDescriptor::fpcr_shift;
        constexpr u64 pc_mask = A64::LocationDescriptor::pc_mask;

        const u64 fpcr_u64 = static_cast<u64>(fpcr & fpcr_mask) << fpcr_shift;
        const u64 pc_u64 = pc & pc_mask;
        return IR::LocationDescriptor{pc_u64 | fpcr_u64};
    }
};

static_assert(std::is_trivially_copyable_v<A64JitState>, "A64JitState must be trivially copyable for optimal performance");
static_assert(alignof(A64JitState) == 16, "A64JitState must be 16-byte aligned for SIMD performance");

}  // namespace Dynarmic::Backend::Arm64
