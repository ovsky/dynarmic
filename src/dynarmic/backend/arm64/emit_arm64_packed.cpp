/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

// Helper to realize multiple registers at once (variadic)
template<typename... Args>
inline void RealizeAll(Args&&... args) {
    (RegAlloc::Realize(std::forward<Args>(args)), ...);
}

// Generalized packed op emitter with perfect forwarding
template<typename EmitFn>
inline void EmitPackedOp(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RealizeAll(Vresult, Va, Vb);
    std::forward<EmitFn>(emit)(Vresult, Va, Vb);
}

// For saturated ops, also spill FPSR
template<typename EmitFn>
inline void EmitSaturatedPackedOp(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RealizeAll(Vresult, Va, Vb);
    ctx.fpsr.Spill();
    std::forward<EmitFn>(emit)(Vresult, Va, Vb);
}

// Macro to reduce boilerplate for simple packed ops
#define EMIT_PACKED_OP(OPCODE, INSTR, TYPE) \
template<> \
void EmitIR<IR::Opcode::OPCODE>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.INSTR(Vresult->TYPE(), Va->TYPE(), Vb->TYPE()); }); \
}

// Macro for saturated packed ops
#define EMIT_SATURATED_PACKED_OP(OPCODE, INSTR, TYPE) \
template<> \
void EmitIR<IR::Opcode::OPCODE>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    EmitSaturatedPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.INSTR(Vresult->TYPE(), Va->TYPE(), Vb->TYPE()); }); \
}

// Macro for halving packed ops
#define EMIT_HALVING_PACKED_OP(OPCODE, INSTR, TYPE) \
template<> \
void EmitIR<IR::Opcode::OPCODE>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) { code.INSTR(Vresult->TYPE(), Va->TYPE(), Vb->TYPE()); }); \
}

// --- U8/S8/U16/S16 Add/Sub with GE handling ---

template<typename T, typename T2>
inline void EmitPackedAddWithGE(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, auto&& add_instr, auto&& ge_instr) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);
    const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RealizeAll(Vresult, Va, Vb);

    add_instr(Vresult, Va, Vb);

    if (ge_inst) {
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);
        ge_instr(Vge, Va, Vb, Vresult);
    }
}

template<>
void EmitIR<IR::Opcode::PackedAddU8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<uint8_t, uint8_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.ADD(Vresult->B8(), Va->B8(), Vb->B8()); },
        [](auto& Vge, auto& Va, auto&, auto& Vresult) { code.CMHI(Vge->B8(), Va->B8(), Vresult->B8()); }
    );
}

template<>
void EmitIR<IR::Opcode::PackedAddS8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<int8_t, int8_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.ADD(Vresult->B8(), Va->B8(), Vb->B8()); },
        [](auto& Vge, auto& Va, auto& Vb, auto&) {
            code.SHADD(Vge->B8(), Va->B8(), Vb->B8());
            code.CMGE(Vge->B8(), Vge->B8(), 0);
        }
    );
}

template<>
void EmitIR<IR::Opcode::PackedSubU8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<uint8_t, uint8_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.SUB(Vresult->B8(), Va->B8(), Vb->B8()); },
        [](auto& Vge, auto& Va, auto& Vb, auto&) {
            code.UHSUB(Vge->B8(), Va->B8(), Vb->B8());
            code.CMGE(Vge->B8(), Vge->B8(), 0);
        }
    );
}

template<>
void EmitIR<IR::Opcode::PackedSubS8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<int8_t, int8_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.SUB(Vresult->B8(), Va->B8(), Vb->B8()); },
        [](auto& Vge, auto& Va, auto& Vb, auto&) {
            code.SHSUB(Vge->B8(), Va->B8(), Vb->B8());
            code.CMGE(Vge->B8(), Vge->B8(), 0);
        }
    );
}

template<>
void EmitIR<IR::Opcode::PackedAddU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<uint16_t, uint16_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.ADD(Vresult->H4(), Va->H4(), Vb->H4()); },
        [](auto& Vge, auto& Va, auto&, auto& Vresult) { code.CMHI(Vge->H4(), Va->H4(), Vresult->H4()); }
    );
}

template<>
void EmitIR<IR::Opcode::PackedAddS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<int16_t, int16_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.ADD(Vresult->H4(), Va->H4(), Vb->H4()); },
        [](auto& Vge, auto& Va, auto& Vb, auto&) {
            code.SHADD(Vge->H4(), Va->H4(), Vb->H4());
            code.CMGE(Vge->H4(), Vge->H4(), 0);
        }
    );
}

template<>
void EmitIR<IR::Opcode::PackedSubU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<uint16_t, uint16_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.SUB(Vresult->H4(), Va->H4(), Vb->H4()); },
        [](auto& Vge, auto& Va, auto& Vb, auto&) {
            code.UHSUB(Vge->H4(), Va->H4(), Vb->H4());
            code.CMGE(Vge->H4(), Vge->H4(), 0);
        }
    );
}

template<>
void EmitIR<IR::Opcode::PackedSubS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddWithGE<int16_t, int16_t>(
        code, ctx, inst,
        [](auto& Vresult, auto& Va, auto& Vb) { code.SUB(Vresult->H4(), Va->H4(), Vb->H4()); },
        [](auto& Vge, auto& Va, auto& Vb, auto&) {
            code.SHSUB(Vge->H4(), Va->H4(), Vb->H4());
            code.CMGE(Vge->H4(), Vge->H4(), 0);
        }
    );
}

// --- AddSub/SubAdd/halving variants ---

template<bool add_is_hi, bool is_signed, bool is_halving>
void EmitPackedAddSub(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto ge_inst = inst->GetAssociatedPseudoOperation(IR::Opcode::GetGEFromOp);
    const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Va = ctx.reg_alloc.ReadD(args[0]);
    auto Vb = ctx.reg_alloc.ReadD(args[1]);
    RealizeAll(Vresult, Va, Vb);

    // Use local temporaries for V0, V1, V2 to avoid global register pressure
    auto V0 = ctx.reg_alloc.ScratchD();
    auto V1 = ctx.reg_alloc.ScratchD();
    auto V2 = ctx.reg_alloc.ScratchD();

    if constexpr (is_signed) {
        code.SXTL(V0->S4(), Va->H4());
        code.SXTL(V1->S4(), Vb->H4());
    } else {
        code.UXTL(V0->S4(), Va->H4());
        code.UXTL(V1->S4(), Vb->H4());
    }
    code.EXT(V1->B8(), V1->B8(), V1->B8(), 4);

    code.MOVI(V2, oaknut::RepImm{add_is_hi ? 0b11110000 : 0b00001111});
    code.EOR(V1->B8(), V1->B8(), V2->B8());
    code.SUB(V1->S2(), V1->S2(), V2->S2());
    code.SUB(Vresult->S2(), V0->S2(), V1->S2());

    if constexpr (is_halving) {
        if constexpr (is_signed)
            code.SSHR(Vresult->S2(), Vresult->S2(), 1);
        else
            code.USHR(Vresult->S2(), Vresult->S2(), 1);
    }

    if (ge_inst) {
        ASSERT(!is_halving);
        auto Vge = ctx.reg_alloc.WriteD(ge_inst);
        RegAlloc::Realize(Vge);

        if constexpr (is_signed) {
            code.CMGE(Vge->S2(), Vresult->S2(), 0);
            code.XTN(Vge->H4(), Vge->toQ().S4());
        } else {
            code.CMEQ(Vge->H4(), Vresult->H4(), 0);
            code.EOR(Vge->B8(), Vge->B8(), V2->B8());
            code.SHRN(Vge->H4(), Vge->toQ().S4(), 16);
        }
    }

    code.XTN(Vresult->H4(), Vresult->toQ().S4());
}

template<>
void EmitIR<IR::Opcode::PackedAddSubU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, false, false>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::PackedAddSubS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, true, false>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::PackedSubAddU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, false, false>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::PackedSubAddS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, true, false>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::PackedHalvingAddSubU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, false, true>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::PackedHalvingAddSubS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<true, true, true>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::PackedHalvingSubAddU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, false, true>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::PackedHalvingSubAddS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedAddSub<false, true, true>(code, ctx, inst);
}

// --- Halving ops ---
EMIT_HALVING_PACKED_OP(PackedHalvingAddU8, UHADD, B8)
EMIT_HALVING_PACKED_OP(PackedHalvingAddS8, SHADD, B8)
EMIT_HALVING_PACKED_OP(PackedHalvingSubU8, UHSUB, B8)
EMIT_HALVING_PACKED_OP(PackedHalvingSubS8, SHSUB, B8)
EMIT_HALVING_PACKED_OP(PackedHalvingAddU16, UHADD, H4)
EMIT_HALVING_PACKED_OP(PackedHalvingAddS16, SHADD, H4)
EMIT_HALVING_PACKED_OP(PackedHalvingSubU16, UHSUB, H4)
EMIT_HALVING_PACKED_OP(PackedHalvingSubS16, SHSUB, H4)

// --- Saturated ops ---
EMIT_SATURATED_PACKED_OP(PackedSaturatedAddU8, UQADD, B8)
EMIT_SATURATED_PACKED_OP(PackedSaturatedAddS8, SQADD, B8)
EMIT_SATURATED_PACKED_OP(PackedSaturatedSubU8, UQSUB, B8)
EMIT_SATURATED_PACKED_OP(PackedSaturatedSubS8, SQSUB, B8)
EMIT_SATURATED_PACKED_OP(PackedSaturatedAddU16, UQADD, H4)
EMIT_SATURATED_PACKED_OP(PackedSaturatedAddS16, SQADD, H4)
EMIT_SATURATED_PACKED_OP(PackedSaturatedSubU16, UQSUB, H4)
EMIT_SATURATED_PACKED_OP(PackedSaturatedSubS16, SQSUB, H4)

// --- AbsDiffSumU8 ---
template<>
void EmitIR<IR::Opcode::PackedAbsDiffSumU8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitPackedOp(code, ctx, inst, [&](auto& Vresult, auto& Va, auto& Vb) {
        auto V2 = ctx.reg_alloc.ScratchD();
        code.MOVI(V2, oaknut::RepImm{0b00001111});
        code.UABD(Vresult->B8(), Va->B8(), Vb->B8());
        code.AND(Vresult->B8(), Vresult->B8(), V2->B8());  // TODO: Zext tracking
        code.UADDLV(Vresult->toH(), Vresult->B8());
    });
}

// --- PackedSelect ---
template<>
void EmitIR<IR::Opcode::PackedSelect>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteD(inst);
    auto Vge = ctx.reg_alloc.ReadD(args[0]);
    auto Va = ctx.reg_alloc.ReadD(args[1]);
    auto Vb = ctx.reg_alloc.ReadD(args[2]);
    RealizeAll(Vresult, Vge, Va, Vb);

    code.FMOV(Vresult, Vge);  // TODO: Move elimination
    code.BSL(Vresult->B8(), Vb->B8(), Va->B8());
}

#undef EMIT_PACKED_OP
#undef EMIT_SATURATED_PACKED_OP
#undef EMIT_HALVING_PACKED_OP

}  // namespace Dynarmic::Backend::Arm64
