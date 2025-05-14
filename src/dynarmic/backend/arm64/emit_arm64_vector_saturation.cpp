/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/mp/metavalue/lift_value.hpp>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/common/always_false.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

// Helper to map element size to vector type member function
template<size_t size>
constexpr auto GetVectorType(auto* reg) {
    if constexpr (size == 8) {
        return reg->B16();
    } else if constexpr (size == 16) {
        return reg->H8();
    } else if constexpr (size == 32) {
        return reg->S4();
    } else if constexpr (size == 64) {
        return reg->D2();
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<size>>, "Invalid vector element size");
    }
}

// Generalized Emit function for vector saturated ops
template<size_t size, typename EmitFn>
inline void Emit(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto* Qresult = ctx.reg_alloc.WriteQ(inst);
    auto* Qa = ctx.reg_alloc.ReadQ(args[0]);
    auto* Qb = ctx.reg_alloc.ReadQ(args[1]);
    RegAlloc::Realize(Qresult, Qa, Qb);
    ctx.fpsr.Load();

    emit(GetVectorType<size>(Qresult), GetVectorType<size>(Qa), GetVectorType<size>(Qb));
}

// Macro to reduce boilerplate for similar opcodes
#define DEFINE_VECTOR_SATURATED_OP(OPCODE, MNEMONIC, SIZE) \
template<> \
void EmitIR<IR::Opcode::OPCODE##8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    Emit<8>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.MNEMONIC(Vresult, Va, Vb); }); \
} \
template<> \
void EmitIR<IR::Opcode::OPCODE##16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    Emit<16>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.MNEMONIC(Vresult, Va, Vb); }); \
} \
template<> \
void EmitIR<IR::Opcode::OPCODE##32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    Emit<32>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.MNEMONIC(Vresult, Va, Vb); }); \
} \
template<> \
void EmitIR<IR::Opcode::OPCODE##64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    Emit<64>(code, ctx, inst, [&](auto Vresult, auto Va, auto Vb) { code.MNEMONIC(Vresult, Va, Vb); }); \
}

// Use macro for all saturated vector ops
DEFINE_VECTOR_SATURATED_OP(VectorSignedSaturatedAdd, SQADD, 8)
DEFINE_VECTOR_SATURATED_OP(VectorSignedSaturatedSub, SQSUB, 8)
DEFINE_VECTOR_SATURATED_OP(VectorUnsignedSaturatedAdd, UQADD, 8)
DEFINE_VECTOR_SATURATED_OP(VectorUnsignedSaturatedSub, UQSUB, 8)

#undef DEFINE_VECTOR_SATURATED_OP

}  // namespace Dynarmic::Backend::Arm64
