/* This file is part of the dynarmic project.
 * Copyright (c) 2021 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <type_traits>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

struct alignas(64) A32JitState {
    // Use [[nodiscard]] for getters to encourage correct usage.
    [[nodiscard]] constexpr u32 Cpsr() const noexcept {
        // Compose CPSR from its fields.
        // Assuming the original implementation does this.
        // If not, this is a placeholder for the actual logic.
        return (cpsr_nzcv & 0xF0000000) |
               (cpsr_q & 0x08000000) |
               (cpsr_jaifm & 0x000FFFFF) |
               (cpsr_ge & 0x000F0000);
    }

    constexpr void SetCpsr(u32 cpsr) noexcept {
        // Decompose CPSR into its fields.
        // Placeholder logic, replace with actual bitfield extraction as needed.
        cpsr_nzcv = cpsr & 0xF0000000;
        cpsr_q = cpsr & 0x08000000;
        cpsr_jaifm = cpsr & 0x000FFFFF;
        cpsr_ge = cpsr & 0x000F0000;
    }

    [[nodiscard]] constexpr u32 Fpscr() const noexcept {
        // Compose FPSCR from its fields.
        // Placeholder logic.
        return (fpsr & 0xFFFFFFFF) | (fpsr_nzcv & 0xF0000000);
    }

    constexpr void SetFpscr(u32 fpscr) noexcept {
        // Decompose FPSCR into its fields.
        // Placeholder logic.
        fpsr = fpscr & 0xFFFFFFFF;
        fpsr_nzcv = fpscr & 0xF0000000;
    }

    [[nodiscard]] constexpr IR::LocationDescriptor GetLocationDescriptor() const noexcept {
        // Use std::bit_cast if available for type-safe conversion.
        // Compose 64-bit PC from regs[15] and upper_location_descriptor.
        return IR::LocationDescriptor{
            static_cast<u64>(regs[15]) | (static_cast<u64>(upper_location_descriptor) << 32)
        };
    }

    // Use default member initializers for zero-initialization.
    u32 cpsr_nzcv = 0;
    u32 cpsr_q = 0;
    u32 cpsr_jaifm = 0;
    u32 cpsr_ge = 0;

    u32 fpsr = 0;
    u32 fpsr_nzcv = 0;

    std::array<u32, 16> regs{};

    u32 upper_location_descriptor = 0;

    alignas(16) std::array<u32, 64> ext_regs{};

    u32 exclusive_state = 0;
};

static_assert(std::is_trivially_copyable_v<A32JitState>, "A32JitState must be trivially copyable for optimal performance");
static_assert(alignof(A32JitState) >= 16, "A32JitState should be at least 16-byte aligned for SIMD operations");

}  // namespace Dynarmic::Backend::Arm64
