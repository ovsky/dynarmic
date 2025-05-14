/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <oaknut/oaknut.hpp>
#include <type_traits>
#include <utility>

#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

// Helper to force inline for small templates
#if defined(__GNUC__) || defined(__clang__)
#define DYNARMIC_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define DYNARMIC_FORCE_INLINE __forceinline
#else
#define DYNARMIC_FORCE_INLINE inline
#endif

// Use perfect forwarding for emit lambdas, and constexpr if for compile-time branching

template<size_t bitsize, typename EmitFn>
DYNARMIC_FORCE_INLINE void EmitTwoOp(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteVec<bitsize>(inst);
    auto Voperand = ctx.reg_alloc.ReadVec<bitsize>(args[0]);
    RegAlloc::Realize(Vresult, Voperand);
    ctx.fpsr.Load();

    std::forward<EmitFn>(emit)(Vresult, Voperand);
}

template<size_t bitsize, typename EmitFn>
DYNARMIC_FORCE_INLINE void EmitThreeOp(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteVec<bitsize>(inst);
    auto Va = ctx.reg_alloc.ReadVec<bitsize>(args[0]);
    auto Vb = ctx.reg_alloc.ReadVec<bitsize>(args[1]);
    RegAlloc::Realize(Vresult, Va, Vb);
    ctx.fpsr.Load();

    std::forward<EmitFn>(emit)(Vresult, Va, Vb);
}

template<size_t bitsize, typename EmitFn>
DYNARMIC_FORCE_INLINE void EmitFourOp(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vresult = ctx.reg_alloc.WriteVec<bitsize>(inst);
    auto Va = ctx.reg_alloc.ReadVec<bitsize>(args[0]);
    auto Vb = ctx.reg_alloc.ReadVec<bitsize>(args[1]);
    auto Vc = ctx.reg_alloc.ReadVec<bitsize>(args[2]);
    RegAlloc::Realize(Vresult, Va, Vb, Vc);
    ctx.fpsr.Load();

    std::forward<EmitFn>(emit)(Vresult, Va, Vb, Vc);
}

template<size_t bitsize_from, size_t bitsize_to, typename EmitFn>
DYNARMIC_FORCE_INLINE void EmitConvert(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vto = ctx.reg_alloc.WriteVec<bitsize_to>(inst);
    auto Vfrom = ctx.reg_alloc.ReadVec<bitsize_from>(args[0]);
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[1].GetImmediateU8());
    RegAlloc::Realize(Vto, Vfrom);
    ctx.fpsr.Load();

    ASSERT(rounding_mode == ctx.FPCR().RMode());

    std::forward<EmitFn>(emit)(Vto, Vfrom);
}

template<size_t bitsize_from, size_t bitsize_to, bool is_signed>
DYNARMIC_FORCE_INLINE void EmitToFixed(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Rto = ctx.reg_alloc.WriteReg<std::max<size_t>(bitsize_to, 32)>(inst);
    auto Vfrom = ctx.reg_alloc.ReadVec<bitsize_from>(args[0]);
    const size_t fbits = args[1].GetImmediateU8();
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
    RegAlloc::Realize(Rto, Vfrom);
    ctx.fpsr.Load();

    if (rounding_mode == FP::RoundingMode::TowardsZero) {
        if constexpr (is_signed) {
            if constexpr (bitsize_to == 16) {
                code.FCVTZS(Rto, Vfrom, fbits + 16);
                code.ASR(Wscratch0, Rto, 31);
                code.ADD(Rto, Rto, Wscratch0, LSR, 16);
                code.LSR(Rto, Rto, 16);
            } else if (fbits) {
                code.FCVTZS(Rto, Vfrom, fbits);
            } else {
                code.FCVTZS(Rto, Vfrom);
            }
        } else {
            if constexpr (bitsize_to == 16) {
                code.FCVTZU(Rto, Vfrom, fbits + 16);
                code.LSR(Rto, Rto, 16);
            } else if (fbits) {
                code.FCVTZU(Rto, Vfrom, fbits);
            } else {
                code.FCVTZU(Rto, Vfrom);
            }
        }
    } else {
        ASSERT(fbits == 0);
        ASSERT(bitsize_to != 16);
        if constexpr (is_signed) {
            switch (rounding_mode) {
            case FP::RoundingMode::ToNearest_TieEven:
                code.FCVTNS(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsPlusInfinity:
                code.FCVTPS(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsMinusInfinity:
                code.FCVTMS(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsZero:
                code.FCVTZS(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToNearest_TieAwayFromZero:
                code.FCVTAS(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToOdd:
                ASSERT_FALSE("Unimplemented");
                break;
            default:
                ASSERT_FALSE("Invalid RoundingMode");
                break;
            }
        } else {
            switch (rounding_mode) {
            case FP::RoundingMode::ToNearest_TieEven:
                code.FCVTNU(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsPlusInfinity:
                code.FCVTPU(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsMinusInfinity:
                code.FCVTMU(Rto, Vfrom);
                break;
            case FP::RoundingMode::TowardsZero:
                code.FCVTZU(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToNearest_TieAwayFromZero:
                code.FCVTAU(Rto, Vfrom);
                break;
            case FP::RoundingMode::ToOdd:
                ASSERT_FALSE("Unimplemented");
                break;
            default:
                ASSERT_FALSE("Invalid RoundingMode");
                break;
            }
        }
    }
}

template<size_t bitsize_from, size_t bitsize_to, typename EmitFn>
DYNARMIC_FORCE_INLINE void EmitFromFixed(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, EmitFn&& emit) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Vto = ctx.reg_alloc.WriteVec<bitsize_to>(inst);
    auto Rfrom = ctx.reg_alloc.ReadReg<std::max<size_t>(bitsize_from, 32)>(args[0]);
    const size_t fbits = args[1].GetImmediateU8();
    const auto rounding_mode = static_cast<FP::RoundingMode>(args[2].GetImmediateU8());
    RegAlloc::Realize(Vto, Rfrom);
    ctx.fpsr.Load();

    if (rounding_mode == ctx.FPCR().RMode()) {
        std::forward<EmitFn>(emit)(Vto, Rfrom, static_cast<u8>(fbits));
    } else {
        FP::FPCR new_fpcr = ctx.FPCR();
        new_fpcr.RMode(rounding_mode);

        code.MOV(Wscratch0, new_fpcr.Value());
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);

        std::forward<EmitFn>(emit)(Vto, Rfrom, static_cast<u8>(fbits));

        code.MOV(Wscratch0, ctx.FPCR().Value());
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);
    }
}

template<size_t size>
DYNARMIC_FORCE_INLINE void EmitCompare(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto flags = ctx.reg_alloc.WriteFlags(inst);
    auto Va = ctx.reg_alloc.ReadVec<size>(args[0]);
    const bool exc_on_qnan = args[2].GetImmediateU1();

    if (args[1].IsImmediate() && args[1].GetImmediateU64() == 0) {
        RegAlloc::Realize(flags, Va);
        ctx.fpsr.Load();

        if (exc_on_qnan) {
            code.FCMPE(Va, 0);
        } else {
            code.FCMP(Va, 0);
        }
    } else {
        auto Vb = ctx.reg_alloc.ReadVec<size>(args[1]);
        RegAlloc::Realize(flags, Va, Vb);
        ctx.fpsr.Load();

        if (exc_on_qnan) {
            code.FCMPE(Va, Vb);
        } else {
            code.FCMP(Va, Vb);
        }
    }
}

// Macro to reduce boilerplate for simple two/three/four opcodes
#define DYNARMIC_EMIT_TWO_OP(OP, SIZE, INSTR) \
template<> \
void EmitIR<IR::Opcode::OP>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    EmitTwoOp<SIZE>(code, ctx, inst, [&](auto& result, auto& operand) { code.INSTR(result, operand); }); \
}

#define DYNARMIC_EMIT_THREE_OP(OP, SIZE, INSTR) \
template<> \
void EmitIR<IR::Opcode::OP>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    EmitThreeOp<SIZE>(code, ctx, inst, [&](auto& result, auto& a, auto& b) { code.INSTR(result, a, b); }); \
}

#define DYNARMIC_EMIT_FOUR_OP(OP, SIZE, INSTR) \
template<> \
void EmitIR<IR::Opcode::OP>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { \
    EmitFourOp<SIZE>(code, ctx, inst, [&](auto& result, auto& a, auto& b, auto& c) { code.INSTR(result, b, c, a); }); \
}

// Unimplemented stubs
#define DYNARMIC_EMIT_UNIMPLEMENTED(OP) \
template<> \
void EmitIR<IR::Opcode::OP>(oaknut::CodeGenerator&, EmitContext&, IR::Inst*) { \
    ASSERT_FALSE("Unimplemented"); \
}

// Two operand
DYNARMIC_EMIT_UNIMPLEMENTED(FPAbs16)
DYNARMIC_EMIT_TWO_OP(FPAbs32, 32, FABS)
DYNARMIC_EMIT_TWO_OP(FPAbs64, 64, FABS)

DYNARMIC_EMIT_THREE_OP(FPAdd32, 32, FADD)
DYNARMIC_EMIT_THREE_OP(FPAdd64, 64, FADD)

template<>
void EmitIR<IR::Opcode::FPCompare32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitCompare<32>(code, ctx, inst);
}
template<>
void EmitIR<IR::Opcode::FPCompare64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitCompare<64>(code, ctx, inst);
}

DYNARMIC_EMIT_THREE_OP(FPDiv32, 32, FDIV)
DYNARMIC_EMIT_THREE_OP(FPDiv64, 64, FDIV)

DYNARMIC_EMIT_THREE_OP(FPMax32, 32, FMAX)
DYNARMIC_EMIT_THREE_OP(FPMax64, 64, FMAX)
DYNARMIC_EMIT_THREE_OP(FPMaxNumeric32, 32, FMAXNM)
DYNARMIC_EMIT_THREE_OP(FPMaxNumeric64, 64, FMAXNM)
DYNARMIC_EMIT_THREE_OP(FPMin32, 32, FMIN)
DYNARMIC_EMIT_THREE_OP(FPMin64, 64, FMIN)
DYNARMIC_EMIT_THREE_OP(FPMinNumeric32, 32, FMINNM)
DYNARMIC_EMIT_THREE_OP(FPMinNumeric64, 64, FMINNM)

DYNARMIC_EMIT_THREE_OP(FPMul32, 32, FMUL)
DYNARMIC_EMIT_THREE_OP(FPMul64, 64, FMUL)

DYNARMIC_EMIT_UNIMPLEMENTED(FPMulAdd16)
DYNARMIC_EMIT_FOUR_OP(FPMulAdd32, 32, FMADD)
DYNARMIC_EMIT_FOUR_OP(FPMulAdd64, 64, FMADD)

DYNARMIC_EMIT_UNIMPLEMENTED(FPMulSub16)
DYNARMIC_EMIT_FOUR_OP(FPMulSub32, 32, FMSUB)
DYNARMIC_EMIT_FOUR_OP(FPMulSub64, 64, FMSUB)

DYNARMIC_EMIT_THREE_OP(FPMulX32, 32, FMULX)
DYNARMIC_EMIT_THREE_OP(FPMulX64, 64, FMULX)

DYNARMIC_EMIT_UNIMPLEMENTED(FPNeg16)
DYNARMIC_EMIT_TWO_OP(FPNeg32, 32, FNEG)
DYNARMIC_EMIT_TWO_OP(FPNeg64, 64, FNEG)

DYNARMIC_EMIT_UNIMPLEMENTED(FPRecipEstimate16)
DYNARMIC_EMIT_TWO_OP(FPRecipEstimate32, 32, FRECPE)
DYNARMIC_EMIT_TWO_OP(FPRecipEstimate64, 64, FRECPE)

DYNARMIC_EMIT_UNIMPLEMENTED(FPRecipExponent16)
DYNARMIC_EMIT_TWO_OP(FPRecipExponent32, 32, FRECPX)
DYNARMIC_EMIT_TWO_OP(FPRecipExponent64, 64, FRECPX)

DYNARMIC_EMIT_UNIMPLEMENTED(FPRecipStepFused16)
DYNARMIC_EMIT_THREE_OP(FPRecipStepFused32, 32, FRECPS)
DYNARMIC_EMIT_THREE_OP(FPRecipStepFused64, 64, FRECPS)

DYNARMIC_EMIT_UNIMPLEMENTED(FPRoundInt16)

template<>
void EmitIR<IR::Opcode::FPRoundInt32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    const bool exact = inst->GetArg(2).GetU1();

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Sresult = ctx.reg_alloc.WriteS(inst);
    auto Soperand = ctx.reg_alloc.ReadS(args[0]);
    RegAlloc::Realize(Sresult, Soperand);
    ctx.fpsr.Load();

    if (exact) {
        ASSERT(ctx.FPCR().RMode() == rounding_mode);
        code.FRINTX(Sresult, Soperand);
    } else {
        switch (rounding_mode) {
        case FP::RoundingMode::ToNearest_TieEven:
            code.FRINTN(Sresult, Soperand);
            break;
        case FP::RoundingMode::TowardsPlusInfinity:
            code.FRINTP(Sresult, Soperand);
            break;
        case FP::RoundingMode::TowardsMinusInfinity:
            code.FRINTM(Sresult, Soperand);
            break;
        case FP::RoundingMode::TowardsZero:
            code.FRINTZ(Sresult, Soperand);
            break;
        case FP::RoundingMode::ToNearest_TieAwayFromZero:
            code.FRINTA(Sresult, Soperand);
            break;
        default:
            ASSERT_FALSE("Invalid RoundingMode");
        }
    }
}

template<>
void EmitIR<IR::Opcode::FPRoundInt64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());
    const bool exact = inst->GetArg(2).GetU1();

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Dresult = ctx.reg_alloc.WriteD(inst);
    auto Doperand = ctx.reg_alloc.ReadD(args[0]);
    RegAlloc::Realize(Dresult, Doperand);
    ctx.fpsr.Load();

    if (exact) {
        ASSERT(ctx.FPCR().RMode() == rounding_mode);
        code.FRINTX(Dresult, Doperand);
    } else {
        switch (rounding_mode) {
        case FP::RoundingMode::ToNearest_TieEven:
            code.FRINTN(Dresult, Doperand);
            break;
        case FP::RoundingMode::TowardsPlusInfinity:
            code.FRINTP(Dresult, Doperand);
            break;
        case FP::RoundingMode::TowardsMinusInfinity:
            code.FRINTM(Dresult, Doperand);
            break;
        case FP::RoundingMode::TowardsZero:
            code.FRINTZ(Dresult, Doperand);
            break;
        case FP::RoundingMode::ToNearest_TieAwayFromZero:
            code.FRINTA(Dresult, Doperand);
            break;
        default:
            ASSERT_FALSE("Invalid RoundingMode");
        }
    }
}

DYNARMIC_EMIT_UNIMPLEMENTED(FPRSqrtEstimate16)
DYNARMIC_EMIT_TWO_OP(FPRSqrtEstimate32, 32, FRSQRTE)
DYNARMIC_EMIT_TWO_OP(FPRSqrtEstimate64, 64, FRSQRTE)

DYNARMIC_EMIT_UNIMPLEMENTED(FPRSqrtStepFused16)
DYNARMIC_EMIT_THREE_OP(FPRSqrtStepFused32, 32, FRSQRTS)
DYNARMIC_EMIT_THREE_OP(FPRSqrtStepFused64, 64, FRSQRTS)

DYNARMIC_EMIT_TWO_OP(FPSqrt32, 32, FSQRT)
DYNARMIC_EMIT_TWO_OP(FPSqrt64, 64, FSQRT)

DYNARMIC_EMIT_THREE_OP(FPSub32, 32, FSUB)
DYNARMIC_EMIT_THREE_OP(FPSub64, 64, FSUB)

template<>
void EmitIR<IR::Opcode::FPHalfToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Hfrom) { code.FCVT(Dto, Hfrom); });
}
template<>
void EmitIR<IR::Opcode::FPHalfToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Hfrom) { code.FCVT(Sto, Hfrom); });
}
template<>
void EmitIR<IR::Opcode::FPSingleToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Sfrom) { code.FCVT(Dto, Sfrom); });
}
template<>
void EmitIR<IR::Opcode::FPSingleToHalf>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<32, 16>(code, ctx, inst, [&](auto& Hto, auto& Sfrom) { code.FCVT(Hto, Sfrom); });
}
template<>
void EmitIR<IR::Opcode::FPDoubleToHalf>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitConvert<64, 16>(code, ctx, inst, [&](auto& Hto, auto& Dfrom) { code.FCVT(Hto, Dfrom); });
}
template<>
void EmitIR<IR::Opcode::FPDoubleToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    const auto rounding_mode = static_cast<FP::RoundingMode>(inst->GetArg(1).GetU8());

    if (rounding_mode == FP::RoundingMode::ToOdd) {
        auto args = ctx.reg_alloc.GetArgumentInfo(inst);
        auto Sto = ctx.reg_alloc.WriteS(inst);
        auto Dfrom = ctx.reg_alloc.ReadD(args[0]);
        RegAlloc::Realize(Sto, Dfrom);
        ctx.fpsr.Load();

        code.FCVTXN(Sto, Dfrom);
        return;
    }

    EmitConvert<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Dfrom) { code.FCVT(Sto, Dfrom); });
}

template<> void EmitIR<IR::Opcode::FPDoubleToFixedS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<64, 16, true>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPDoubleToFixedS32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<64, 32, true>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPDoubleToFixedS64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<64, 64, true>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPDoubleToFixedU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<64, 16, false>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPDoubleToFixedU32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<64, 32, false>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPDoubleToFixedU64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<64, 64, false>(code, ctx, inst); }

DYNARMIC_EMIT_UNIMPLEMENTED(FPHalfToFixedS16)
DYNARMIC_EMIT_UNIMPLEMENTED(FPHalfToFixedS32)
DYNARMIC_EMIT_UNIMPLEMENTED(FPHalfToFixedS64)
DYNARMIC_EMIT_UNIMPLEMENTED(FPHalfToFixedU16)
DYNARMIC_EMIT_UNIMPLEMENTED(FPHalfToFixedU32)
DYNARMIC_EMIT_UNIMPLEMENTED(FPHalfToFixedU64)

template<> void EmitIR<IR::Opcode::FPSingleToFixedS16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<32, 16, true>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPSingleToFixedS32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<32, 32, true>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPSingleToFixedS64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<32, 64, true>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPSingleToFixedU16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<32, 16, false>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPSingleToFixedU32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<32, 32, false>(code, ctx, inst); }
template<> void EmitIR<IR::Opcode::FPSingleToFixedU64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) { EmitToFixed<32, 64, false>(code, ctx, inst); }

template<>
void EmitIR<IR::Opcode::FPFixedU16ToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.LSL(Wscratch0, Wfrom, 16);
        code.UCVTF(Sto, Wscratch0, fbits + 16);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedS16ToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        code.LSL(Wscratch0, Wfrom, 16);
        code.SCVTF(Sto, Wscratch0, fbits + 16);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedU16ToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.LSL(Wscratch0, Wfrom, 16);
        code.UCVTF(Dto, Wscratch0, fbits + 16);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedS16ToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<16, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        code.LSL(Wscratch0, Wfrom, 16);
        code.SCVTF(Dto, Wscratch0, fbits + 16);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedU32ToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        if (fbits)
            code.UCVTF(Sto, Wfrom, fbits);
        else
            code.UCVTF(Sto, Wfrom);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedS32ToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 32>(code, ctx, inst, [&](auto& Sto, auto& Wfrom, u8 fbits) {
        if (fbits)
            code.SCVTF(Sto, Wfrom, fbits);
        else
            code.SCVTF(Sto, Wfrom);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedU32ToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        if (fbits)
            code.UCVTF(Dto, Wfrom, fbits);
        else
            code.UCVTF(Dto, Wfrom);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedS32ToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<32, 64>(code, ctx, inst, [&](auto& Dto, auto& Wfrom, u8 fbits) {
        if (fbits)
            code.SCVTF(Dto, Wfrom, fbits);
        else
            code.SCVTF(Dto, Wfrom);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedU64ToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 64>(code, ctx, inst, [&](auto& Dto, auto& Xfrom, u8 fbits) {
        if (fbits)
            code.UCVTF(Dto, Xfrom, fbits);
        else
            code.UCVTF(Dto, Xfrom);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedU64ToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Xfrom, u8 fbits) {
        if (fbits)
            code.UCVTF(Sto, Xfrom, fbits);
        else
            code.UCVTF(Sto, Xfrom);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedS64ToDouble>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 64>(code, ctx, inst, [&](auto& Dto, auto& Xfrom, u8 fbits) {
        if (fbits)
            code.SCVTF(Dto, Xfrom, fbits);
        else
            code.SCVTF(Dto, Xfrom);
    });
}
template<>
void EmitIR<IR::Opcode::FPFixedS64ToSingle>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    EmitFromFixed<64, 32>(code, ctx, inst, [&](auto& Sto, auto& Xfrom, u8 fbits) {
        if (fbits)
            code.SCVTF(Sto, Xfrom, fbits);
        else
            code.SCVTF(Sto, Xfrom);
    });
}

#undef DYNARMIC_EMIT_TWO_OP
#undef DYNARMIC_EMIT_THREE_OP
#undef DYNARMIC_EMIT_FOUR_OP
#undef DYNARMIC_EMIT_UNIMPLEMENTED
#undef DYNARMIC_FORCE_INLINE

}  // namespace Dynarmic::Backend::Arm64
