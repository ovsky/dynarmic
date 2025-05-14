/* This file is part of the dynarmic project.
 * Copyright (c) 2023 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <mcl/stdint.hpp>

#include "dynarmic/backend/arm64/stack_layout.h"

namespace oaknut {
struct CodeGenerator;
struct Label;
}  // namespace oaknut

namespace Dynarmic::IR {
enum class Type;
}  // namespace Dynarmic::IR

namespace Dynarmic::Backend::Arm64 {

struct EmitContext;

// Use [[nodiscard]] for types that should not be ignored
using Vector = std::array<u64, 2>;

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4324)  // Structure was padded due to alignment specifier
#endif

enum class HostLocType : uint8_t {
    X,
    Q,
    Nzcv,
    Spill,
};

// Use alignas(16) for SIMD-friendly alignment, and default member initializers for safety
struct alignas(16) RegisterData {
    std::array<u64, 30> x{};                // General-purpose registers
    std::array<Vector, 32> q{};             // SIMD registers
    u32 nzcv = 0;                           // Condition flags
    decltype(StackLayout::spill) *spill = nullptr; // Pointer to spill area
    u32 fpsr = 0;                           // Floating-point status register

    RegisterData() = default;
    RegisterData(const RegisterData&) = default;
    RegisterData(RegisterData&&) noexcept = default;
    RegisterData& operator=(const RegisterData&) = default;
    RegisterData& operator=(RegisterData&&) noexcept = default;
    ~RegisterData() = default;
};

#ifdef _MSC_VER
#    pragma warning(pop)
#endif

// Use [[maybe_unused]] to avoid warnings if not used in all builds
[[maybe_unused]]
inline void EmitVerboseDebuggingOutput(oaknut::CodeGenerator& code, EmitContext& ctx);

[[maybe_unused]]
inline void PrintVerboseDebuggingOutputLine(RegisterData& reg_data, HostLocType reg_type, size_t reg_index, size_t inst_index, IR::Type inst_type);

}  // namespace Dynarmic::Backend::Arm64
