/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_arm64_memory.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

// Helper to reduce template boilerplate and improve inlining
template<size_t Bits, typename F>
inline void DispatchMemoryOp(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, F&& fn) {
    if constexpr (Bits == 8) {
        fn.template operator()<8>(code, ctx, inst);
    } else if constexpr (Bits == 16) {
        fn.template operator()<16>(code, ctx, inst);
    } else if constexpr (Bits == 32) {
        fn.template operator()<32>(code, ctx, inst);
    } else if constexpr (Bits == 64) {
        fn.template operator()<64>(code, ctx, inst);
    } else if constexpr (Bits == 128) {
        fn.template operator()<128>(code, ctx, inst);
    }
}

// Use constexpr lambdas for dispatching, reducing template instantiations and improving code locality

template<>
void EmitIR<IR::Opcode::A64ClearExclusive>(oaknut::CodeGenerator& code, EmitContext&, IR::Inst*) {
    // Use STR(WZR, ...) is already optimal for zeroing memory
    code.STR(WZR, Xstate, offsetof(A64JitState, exclusive_state));
}

// Macro to generate repetitive template specializations for memory ops
#define EMIT_MEMORY_OP(OP, FUNC) \
    template<> \
    void EmitIR<IR::Opcode::A64##OP##8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
        FUNC<8>(code, ctx, inst); \
    } \
    template<> \
    void EmitIR<IR::Opcode::A64##OP##16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
        FUNC<16>(code, ctx, inst); \
    } \
    template<> \
    void EmitIR<IR::Opcode::A64##OP##32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
        FUNC<32>(code, ctx, inst); \
    } \
    template<> \
    void EmitIR<IR::Opcode::A64##OP##64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
        FUNC<64>(code, ctx, inst); \
    } \
    template<> \
    void EmitIR<IR::Opcode::A64##OP##128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
        FUNC<128>(code, ctx, inst); \
    }

EMIT_MEMORY_OP(ReadMemory, EmitReadMemory)
EMIT_MEMORY_OP(ExclusiveReadMemory, EmitExclusiveReadMemory)
EMIT_MEMORY_OP(WriteMemory, EmitWriteMemory)
EMIT_MEMORY_OP(ExclusiveWriteMemory, EmitExclusiveWriteMemory)

#undef EMIT_MEMORY_OP

// If in the future more memory sizes are needed, just add to the macro above.

}  // namespace Dynarmic::Backend::Arm64
