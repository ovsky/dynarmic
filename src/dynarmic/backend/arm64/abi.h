/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <mcl/mp/metavalue/lift_value.hpp>
#include <mcl/stdint.hpp>
#include <oaknut/oaknut.hpp>

#include "dynarmic/common/always_false.h"

namespace Dynarmic::Backend::Arm64 {

// Use inline constexpr for better linkage and optimization
inline constexpr oaknut::XReg Xstate{28};
inline constexpr oaknut::XReg Xhalt{27};
inline constexpr oaknut::XReg Xticks{26};
inline constexpr oaknut::XReg Xfastmem{25};
inline constexpr oaknut::XReg Xpagetable{24};

inline constexpr oaknut::XReg Xscratch0{16}, Xscratch1{17}, Xscratch2{30};
inline constexpr oaknut::WReg Wscratch0{16}, Wscratch1{17}, Wscratch2{30};

// Use [[nodiscard]] to enforce correct usage
template<size_t bitsize>
[[nodiscard]] constexpr auto Rscratch0() {
    if constexpr (bitsize == 32) {
        return Wscratch0;
    } else if constexpr (bitsize == 64) {
        return Xscratch0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<bitsize>>, "Unsupported bitsize for Rscratch0");
    }
}

template<size_t bitsize>
[[nodiscard]] constexpr auto Rscratch1() {
    if constexpr (bitsize == 32) {
        return Wscratch1;
    } else if constexpr (bitsize == 64) {
        return Xscratch1;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<bitsize>>, "Unsupported bitsize for Rscratch1");
    }
}

// Use std::array for better type safety and performance
inline constexpr std::array<int, 21> GPR_ORDER{{
    19, 20, 21, 22, 23, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8
}};
inline constexpr std::array<int, 24> FPR_ORDER{{
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
}};

using RegisterList = u64;

// Use constexpr and noexcept where possible for compile-time evaluation and optimization
constexpr RegisterList ToRegList(const oaknut::Reg& reg) noexcept(false) {
    if (reg.is_vector()) {
        return RegisterList{1} << (reg.index() + 32);
    }

    if (reg.index() == 31) {
        throw std::out_of_range("ZR not allowed in reg list");
    }

    if (reg.index() == -1) {
        return RegisterList{1} << 31;
    }

    return RegisterList{1} << reg.index();
}

// Use [[maybe_unused]] to avoid warnings if not used
[[maybe_unused]] inline constexpr RegisterList ABI_CALLEE_SAVE = 0x0000ff00'7ff80000;
[[maybe_unused]] inline constexpr RegisterList ABI_CALLER_SAVE = 0xffffffff'4000ffff;

// Forward declarations for push/pop ABI helpers
void ABI_PushRegisters(oaknut::CodeGenerator& code, RegisterList rl, size_t stack_space);
void ABI_PopRegisters(oaknut::CodeGenerator& code, RegisterList rl, size_t stack_space);

}  // namespace Dynarmic::Backend::Arm64
