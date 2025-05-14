/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <mcl/bit_cast.hpp>
#include <mcl/stdint.hpp>
#include <mcl/type_traits/function_info.hpp>
#include <type_traits>
#include <utility>

namespace Dynarmic::Backend::Arm64 {

struct DevirtualizedCall {
    u64 fn_ptr;
    u64 this_ptr;
};

// Helper to ensure pointer alignment and avoid UB
[[nodiscard]]
constexpr u64 AdjustPointer(u64 ptr, u64 adj) noexcept {
    return ptr + (adj >> 1);
}

// MSVC/Windows implementation: pointer-to-member-function is just a pointer
template<auto mfp>
[[nodiscard]]
constexpr DevirtualizedCall DevirtualizeWindows(mcl::class_type<decltype(mfp)>* this_) noexcept {
    static_assert(sizeof(mfp) == 8, "Unexpected member function pointer size for MSVC/Windows.");
    return DevirtualizedCall{
        mcl::bit_cast<u64>(mfp),
        reinterpret_cast<u64>(this_)
    };
}

// ARM64 Itanium ABI implementation
template<auto mfp>
[[nodiscard]]
DevirtualizedCall DevirtualizeDefault(mcl::class_type<decltype(mfp)>* this_) noexcept {
    struct MemberFunctionPointer {
        u64 ptr;
        u64 adj;
    };

    static_assert(sizeof(MemberFunctionPointer) == 16, "Unexpected member function pointer size for Itanium ABI.");
    static_assert(sizeof(MemberFunctionPointer) == sizeof(mfp), "Member function pointer size mismatch.");

    // Use constexpr if possible for compile-time optimization
    const auto mfp_struct = mcl::bit_cast<MemberFunctionPointer>(mfp);

    u64 this_ptr = AdjustPointer(mcl::bit_cast<u64>(this_), mfp_struct.adj);
    u64 fn_ptr = mfp_struct.ptr;

    // If LSB of adj is set, it's a virtual function
    if (mfp_struct.adj & 1) [[unlikely]] {
        // Use std::launder to avoid UB with pointer aliasing
        auto* vtable_ptr = std::launder(reinterpret_cast<const u64*>(this_ptr));
        u64 vtable = *vtable_ptr;
        auto* fn_ptr_ptr = std::launder(reinterpret_cast<const u64*>(vtable + fn_ptr));
        fn_ptr = *fn_ptr_ptr;
    }

    return DevirtualizedCall{fn_ptr, this_ptr};
}

// Main entry point: selects implementation based on platform
template<auto mfp>
[[nodiscard]]
constexpr DevirtualizedCall Devirtualize(mcl::class_type<decltype(mfp)>* this_) noexcept {
#if defined(_WIN32) && defined(_MSC_VER)
    return DevirtualizeWindows<mfp>(this_);
#else
    return DevirtualizeDefault<mfp>(this_);
#endif
}

}  // namespace Dynarmic::Backend::Arm64
