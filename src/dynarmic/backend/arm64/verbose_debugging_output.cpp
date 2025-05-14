/* This file is part of the dynarmic project.
 * Copyright (c) 2023 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/verbose_debugging_output.h"

#include <fmt/format.h>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/ir/type.h"

#include <array>
#include <type_traits>
#include <utility>
#include <cstddef>

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

// Use constexpr and inline for loop unrolling and inlining
template <typename F>
inline void ForEachXReg(F&& fn) noexcept {
    // X18 is skipped (platform register)
    // Unroll for better performance
    #pragma unroll
    for (int i = 0; i < 30; ++i) {
        if (i == 18) continue;
        fn(i);
    }
}

template <typename F>
inline void ForEachQReg(F&& fn) noexcept {
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        fn(i);
    }
}

// Use always_inline to encourage inlining for small helpers
#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ALWAYS_INLINE inline
#endif

ALWAYS_INLINE void SaveXRegs(oaknut::CodeGenerator& code) {
    ForEachXReg([&](int i) {
        code.STR(oaknut::XReg{i}, SP, offsetof(RegisterData, x) + i * sizeof(u64));
    });
}

ALWAYS_INLINE void SaveQRegs(oaknut::CodeGenerator& code) {
    ForEachQReg([&](int i) {
        code.STR(oaknut::QReg{i}, SP, offsetof(RegisterData, q) + i * sizeof(Vector));
    });
}

ALWAYS_INLINE void RestoreQRegs(oaknut::CodeGenerator& code) {
    ForEachQReg([&](int i) {
        code.LDR(oaknut::QReg{i}, SP, offsetof(RegisterData, q) + i * sizeof(Vector));
    });
}

ALWAYS_INLINE void RestoreXRegs(oaknut::CodeGenerator& code) {
    ForEachXReg([&](int i) {
        code.LDR(oaknut::XReg{i}, SP, offsetof(RegisterData, x) + i * sizeof(u64));
    });
}

void EmitVerboseDebuggingOutput(oaknut::CodeGenerator& code, EmitContext& ctx) {
    static_assert(std::is_standard_layout_v<RegisterData>, "RegisterData must be standard layout");

    code.SUB(SP, SP, sizeof(RegisterData));

    SaveXRegs(code);
    SaveQRegs(code);

    // Save NZCV
    code.MRS(X0, oaknut::SystemReg::NZCV);
    code.STR(X0, SP, offsetof(RegisterData, nzcv));

    // Save spill pointer
    code.ADD(X0, SP, sizeof(RegisterData) + offsetof(StackLayout, spill));
    code.STR(X0, SP, offsetof(RegisterData, spill));

    // Save FPSR
    code.MRS(X0, oaknut::SystemReg::FPSR);
    code.STR(X0, SP, offsetof(RegisterData, fpsr));

    ctx.reg_alloc.EmitVerboseDebuggingOutput();

    // Restore FPSR
    code.LDR(X0, SP, offsetof(RegisterData, fpsr));
    code.MSR(oaknut::SystemReg::FPSR, X0);

    // Restore NZCV
    code.LDR(X0, SP, offsetof(RegisterData, nzcv));
    code.MSR(oaknut::SystemReg::NZCV, X0);

    RestoreQRegs(code);
    RestoreXRegs(code);

    code.ADD(SP, SP, sizeof(RegisterData));
}

constexpr const char* GetTypeFormat(IR::Type type) noexcept {
    // Use a lookup table for faster mapping
    switch (type) {
    case IR::Type::U1:
    case IR::Type::U8:      return "{:02x}";
    case IR::Type::U16:     return "{:04x}";
    case IR::Type::U32:
    case IR::Type::NZCVFlags: return "{:08x}";
    case IR::Type::U64:     return "{:016x}";
    case IR::Type::U128:    return "{:016x}{:016x}";
    default:                return nullptr;
    }
}

// Use [[likely]]/[[unlikely]] for branch prediction hints (C++20)
#if defined(__has_cpp_attribute)
#  if __has_cpp_attribute(likely)
#    define LIKELY [[likely]]
#    define UNLIKELY [[unlikely]]
#  else
#    define LIKELY
#    define UNLIKELY
#  endif
#else
#  define LIKELY
#  define UNLIKELY
#endif

void PrintVerboseDebuggingOutputLine(RegisterData& reg_data, HostLocType reg_type, size_t reg_index, size_t inst_index, IR::Type inst_type) {
    fmt::print("dynarmic debug: %{:05} = ", inst_index);

    // Use switch with fallthrough for performance, avoid lambda
    Vector value;
    switch (reg_type) {
    case HostLocType::X:
        value = {reg_data.x[reg_index], 0};
        break;
    case HostLocType::Q:
        value = reg_data.q[reg_index];
        break;
    case HostLocType::Nzcv:
        value = {reg_data.nzcv, 0};
        break;
    case HostLocType::Spill:
        value = (*reg_data.spill)[reg_index];
        break;
    default:
        fmt::print("invalid reg_type! ");
        value = {0, 0};
        break;
    }

    if (const char* fmt_str = GetTypeFormat(inst_type)) LIKELY {
        switch (inst_type) {
        case IR::Type::U1:
        case IR::Type::U8:
            fmt::print(fmt_str, value[0] & 0xffu);
            break;
        case IR::Type::U16:
            fmt::print(fmt_str, value[0] & 0xffffu);
            break;
        case IR::Type::U32:
        case IR::Type::NZCVFlags:
            fmt::print(fmt_str, value[0] & 0xffffffffu);
            break;
        case IR::Type::U64:
            fmt::print(fmt_str, value[0]);
            break;
        case IR::Type::U128:
            fmt::print(fmt_str, value[1], value[0]);
            break;
        default:
            fmt::print("invalid inst_type!");
            break;
        }
    } else UNLIKELY {
        fmt::print("invalid inst_type!");
    }

    fmt::print("\n");
}

#undef ALWAYS_INLINE
#undef LIKELY
#undef UNLIKELY

}  // namespace Dynarmic::Backend::Arm64
