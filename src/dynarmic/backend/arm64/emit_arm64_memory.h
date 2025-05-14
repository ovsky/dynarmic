/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <mcl/stdint.hpp>
#include <type_traits>

namespace oaknut {
struct CodeGenerator;
struct Label;
}  // namespace oaknut

namespace Dynarmic::IR {
enum class AccType;
class Inst;
}  // namespace Dynarmic::IR

namespace Dynarmic::Backend::Arm64 {

struct EmitContext;
enum class LinkTarget;

// Use [[nodiscard]] to encourage error checking if these functions are extended to return values in the future.
// Use constexpr if possible for template instantiations.
// Use std::enable_if_t to restrict bitsize to valid values at compile time for better type safety and optimization.

template<size_t bitsize, typename = std::enable_if_t<bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64>>
inline void EmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

template<size_t bitsize, typename = std::enable_if_t<bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64>>
inline void EmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

template<size_t bitsize, typename = std::enable_if_t<bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64>>
inline void EmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

template<size_t bitsize, typename = std::enable_if_t<bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64>>
inline void EmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

}  // namespace Dynarmic::Backend::Arm64
