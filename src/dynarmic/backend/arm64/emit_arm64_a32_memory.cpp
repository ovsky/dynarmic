/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_arm64_memory.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

// Helper to reduce template boilerplate and improve inlining
template<size_t Bits, typename F>
inline void DispatchMemoryOp(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, F&& func) {
    if constexpr (Bits == 8) {
        func.template operator()<8>(code, ctx, inst);
    } else if constexpr (Bits == 16) {
        func.template operator()<16>(code, ctx, inst);
    } else if constexpr (Bits == 32) {
        func.template operator()<32>(code, ctx, inst);
    } else if constexpr (Bits == 64) {
        func.template operator()<64>(code, ctx, inst);
    }
}

// Use always_inline to encourage inlining for these tiny wrappers
template<>
__attribute__((always_inline))
void EmitIR<IR::Opcode::A32ClearExclusive>(oaknut::CodeGenerator& code, EmitContext&, IR::Inst*) {
    // Use STR WZR, [Xstate, #offsetof(A32JitState, exclusive_state)]
    // This is already optimal, but let's ensure offset is constexpr
    constexpr auto offset = offsetof(A32JitState, exclusive_state);
    code.STR(WZR, Xstate, offset);
}

// Macro to generate the repetitive template specializations for memory ops
#define EMIT_MEMORY_OP(OPCODE, FUNC) \
    template<> \
    __attribute__((always_inline)) \
    void EmitIR<IR::Opcode::OPCODE>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
        FUNC(code, ctx, inst); \
    }

#define EMIT_MEMORY_OP_BITS(OPCODE, FUNC, BITS) \
    template<> \
    __attribute__((always_inline)) \
    void EmitIR<IR::Opcode::OPCODE>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
        FUNC<BITS>(code, ctx, inst); \
    }

// ReadMemory
EMIT_MEMORY_OP_BITS(A32ReadMemory8, EmitReadMemory, 8)
EMIT_MEMORY_OP_BITS(A32ReadMemory16, EmitReadMemory, 16)
EMIT_MEMORY_OP_BITS(A32ReadMemory32, EmitReadMemory, 32)
EMIT_MEMORY_OP_BITS(A32ReadMemory64, EmitReadMemory, 64)

// ExclusiveReadMemory
EMIT_MEMORY_OP_BITS(A32ExclusiveReadMemory8, EmitExclusiveReadMemory, 8)
EMIT_MEMORY_OP_BITS(A32ExclusiveReadMemory16, EmitExclusiveReadMemory, 16)
EMIT_MEMORY_OP_BITS(A32ExclusiveReadMemory32, EmitExclusiveReadMemory, 32)
EMIT_MEMORY_OP_BITS(A32ExclusiveReadMemory64, EmitExclusiveReadMemory, 64)

// WriteMemory
EMIT_MEMORY_OP_BITS(A32WriteMemory8, EmitWriteMemory, 8)
EMIT_MEMORY_OP_BITS(A32WriteMemory16, EmitWriteMemory, 16)
EMIT_MEMORY_OP_BITS(A32WriteMemory32, EmitWriteMemory, 32)
EMIT_MEMORY_OP_BITS(A32WriteMemory64, EmitWriteMemory, 64)

// ExclusiveWriteMemory
EMIT_MEMORY_OP_BITS(A32ExclusiveWriteMemory8, EmitExclusiveWriteMemory, 8)
EMIT_MEMORY_OP_BITS(A32ExclusiveWriteMemory16, EmitExclusiveWriteMemory, 16)
EMIT_MEMORY_OP_BITS(A32ExclusiveWriteMemory32, EmitExclusiveWriteMemory, 32)
EMIT_MEMORY_OP_BITS(A32ExclusiveWriteMemory64, EmitExclusiveWriteMemory, 64)

#undef EMIT_MEMORY_OP
#undef EMIT_MEMORY_OP_BITS

}  // namespace Dynarmic::Backend::Arm64
