/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/emit_arm64_memory.h"

#include <optional>
#include <utility>
#include <array>
#include <mcl/bit_cast.hpp>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fastmem.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

namespace {

constexpr std::array<LinkTarget, 5> ReadMemoryTargets{
    LinkTarget::ReadMemory8,
    LinkTarget::ReadMemory16,
    LinkTarget::ReadMemory32,
    LinkTarget::ReadMemory64,
    LinkTarget::ReadMemory128
};

constexpr std::array<LinkTarget, 5> WriteMemoryTargets{
    LinkTarget::WriteMemory8,
    LinkTarget::WriteMemory16,
    LinkTarget::WriteMemory32,
    LinkTarget::WriteMemory64,
    LinkTarget::WriteMemory128
};

constexpr std::array<LinkTarget, 5> WrappedReadMemoryTargets{
    LinkTarget::WrappedReadMemory8,
    LinkTarget::WrappedReadMemory16,
    LinkTarget::WrappedReadMemory32,
    LinkTarget::WrappedReadMemory64,
    LinkTarget::WrappedReadMemory128
};

constexpr std::array<LinkTarget, 5> WrappedWriteMemoryTargets{
    LinkTarget::WrappedWriteMemory8,
    LinkTarget::WrappedWriteMemory16,
    LinkTarget::WrappedWriteMemory32,
    LinkTarget::WrappedWriteMemory64,
    LinkTarget::WrappedWriteMemory128
};

constexpr std::array<LinkTarget, 5> ExclusiveReadMemoryTargets{
    LinkTarget::ExclusiveReadMemory8,
    LinkTarget::ExclusiveReadMemory16,
    LinkTarget::ExclusiveReadMemory32,
    LinkTarget::ExclusiveReadMemory64,
    LinkTarget::ExclusiveReadMemory128
};

constexpr std::array<LinkTarget, 5> ExclusiveWriteMemoryTargets{
    LinkTarget::ExclusiveWriteMemory8,
    LinkTarget::ExclusiveWriteMemory16,
    LinkTarget::ExclusiveWriteMemory32,
    LinkTarget::ExclusiveWriteMemory64,
    LinkTarget::ExclusiveWriteMemory128
};

constexpr size_t BitsizeToIndex(size_t bitsize) {
    switch (bitsize) {
        case 8:   return 0;
        case 16:  return 1;
        case 32:  return 2;
        case 64:  return 3;
        case 128: return 4;
        default:  return 5; // Invalid
    }
}

constexpr bool IsValidBitsize(size_t bitsize) {
    return bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64 || bitsize == 128;
}

inline bool IsOrdered(IR::AccType acctype) noexcept {
    using enum IR::AccType;
    return acctype == ORDERED || acctype == ORDEREDRW || acctype == LIMITEDORDERED;
}

inline LinkTarget ReadMemoryLinkTarget(size_t bitsize) {
    const auto idx = BitsizeToIndex(bitsize);
    ASSERT(idx < ReadMemoryTargets.size());
    return ReadMemoryTargets[idx];
}

inline LinkTarget WriteMemoryLinkTarget(size_t bitsize) {
    const auto idx = BitsizeToIndex(bitsize);
    ASSERT(idx < WriteMemoryTargets.size());
    return WriteMemoryTargets[idx];
}

inline LinkTarget WrappedReadMemoryLinkTarget(size_t bitsize) {
    const auto idx = BitsizeToIndex(bitsize);
    ASSERT(idx < WrappedReadMemoryTargets.size());
    return WrappedReadMemoryTargets[idx];
}

inline LinkTarget WrappedWriteMemoryLinkTarget(size_t bitsize) {
    const auto idx = BitsizeToIndex(bitsize);
    ASSERT(idx < WrappedWriteMemoryTargets.size());
    return WrappedWriteMemoryTargets[idx];
}

inline LinkTarget ExclusiveReadMemoryLinkTarget(size_t bitsize) {
    const auto idx = BitsizeToIndex(bitsize);
    ASSERT(idx < ExclusiveReadMemoryTargets.size());
    return ExclusiveReadMemoryTargets[idx];
}

inline LinkTarget ExclusiveWriteMemoryLinkTarget(size_t bitsize) {
    const auto idx = BitsizeToIndex(bitsize);
    ASSERT(idx < ExclusiveWriteMemoryTargets.size());
    return ExclusiveWriteMemoryTargets[idx];
}

template<size_t bitsize>
void CallbackOnlyEmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    EmitRelocation(code, ctx, ReadMemoryLinkTarget(bitsize));
    if (ordered) code.DMB(oaknut::BarrierOp::ISH);

    if constexpr (bitsize == 128) {
        code.MOV(Q8.B16(), Q0.B16());
        ctx.reg_alloc.DefineAsRegister(inst, Q8);
    } else {
        ctx.reg_alloc.DefineAsRegister(inst, X0);
    }
}

template<size_t bitsize>
void CallbackOnlyEmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1]);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    code.MOV(Wscratch0, 1);
    code.STRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    EmitRelocation(code, ctx, ExclusiveReadMemoryLinkTarget(bitsize));
    if (ordered) code.DMB(oaknut::BarrierOp::ISH);

    if constexpr (bitsize == 128) {
        code.MOV(Q8.B16(), Q0.B16());
        ctx.reg_alloc.DefineAsRegister(inst, Q8);
    } else {
        ctx.reg_alloc.DefineAsRegister(inst, X0);
    }
}

template<size_t bitsize>
void CallbackOnlyEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    if (ordered) code.DMB(oaknut::BarrierOp::ISH);
    EmitRelocation(code, ctx, WriteMemoryLinkTarget(bitsize));
    if (ordered) code.DMB(oaknut::BarrierOp::ISH);
}

template<size_t bitsize>
void CallbackOnlyEmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.PrepareForCall({}, args[1], args[2]);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    oaknut::Label end;

    if (ordered) code.DMB(oaknut::BarrierOp::ISH);
    code.MOV(W0, 1);
    code.LDRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    code.CBZ(Wscratch0, end);
    code.STRB(WZR, Xstate, ctx.conf.state_exclusive_state_offset);
    EmitRelocation(code, ctx, ExclusiveWriteMemoryLinkTarget(bitsize));
    if (ordered) code.DMB(oaknut::BarrierOp::ISH);
    code.l(end);
    ctx.reg_alloc.DefineAsRegister(inst, X0);
}

constexpr size_t page_bits = 12;
constexpr size_t page_size = 1 << page_bits;
constexpr size_t page_mask = (1 << page_bits) - 1;

template<size_t bitsize>
void EmitDetectMisalignedVAddr(oaknut::CodeGenerator& code, EmitContext& ctx, oaknut::XReg Xaddr, const SharedLabel& fallback) {
    static_assert(IsValidBitsize(bitsize));
    if (bitsize == 8 || (ctx.conf.detect_misaligned_access_via_page_table & bitsize) == 0) return;

    if (!ctx.conf.only_detect_misalignment_via_page_table_on_page_boundary) {
        constexpr std::array<u64, 5> align_masks{0, 0b1, 0b11, 0b111, 0b1111};
        const u64 align_mask = align_masks[BitsizeToIndex(bitsize)];
        code.TST(Xaddr, align_mask);
        code.B(NE, *fallback);
    } else {
        code.AND(Xscratch0, Xaddr, page_mask);
        code.CMP(Xscratch0, page_size - bitsize / 8);
        code.B(HI, *fallback);
    }
}

template<size_t bitsize>
std::pair<oaknut::XReg, oaknut::XReg> InlinePageTableEmitVAddrLookup(oaknut::CodeGenerator& code, EmitContext& ctx, oaknut::XReg Xaddr, const SharedLabel& fallback) {
    static_assert(IsValidBitsize(bitsize));
    const size_t valid_page_index_bits = ctx.conf.page_table_address_space_bits - page_bits;
    const size_t unused_top_bits = 64 - ctx.conf.page_table_address_space_bits;

    EmitDetectMisalignedVAddr<bitsize>(code, ctx, Xaddr, fallback);

    if (ctx.conf.silently_mirror_page_table || unused_top_bits == 0) {
        code.UBFX(Xscratch0, Xaddr, page_bits, valid_page_index_bits);
    } else {
        code.LSR(Xscratch0, Xaddr, page_bits);
        code.TST(Xscratch0, u64(~u64(0)) << valid_page_index_bits);
        code.B(NE, *fallback);
    }

    code.LDR(Xscratch0, Xpagetable, Xscratch0, LSL, 3);

    if (ctx.conf.page_table_pointer_mask_bits != 0) {
        const u64 mask = u64(~u64(0)) << ctx.conf.page_table_pointer_mask_bits;
        code.AND(Xscratch0, Xscratch0, mask);
    }

    code.CBZ(Xscratch0, *fallback);

    if (ctx.conf.absolute_offset_page_table) {
        return {Xscratch0, Xaddr};
    }
    code.AND(Xscratch1, Xaddr, page_mask);
    return {Xscratch0, Xscratch1};
}

template<std::size_t bitsize>
CodePtr EmitMemoryLdr(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xbase, oaknut::XReg Xoffset, bool ordered, bool extend32 = false) {
    static_assert(IsValidBitsize(bitsize));
    const auto index_ext = extend32 ? oaknut::IndexExt::UXTW : oaknut::IndexExt::LSL;
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};

    CodePtr fastmem_location = code.xptr<CodePtr>();

    if (ordered) {
        code.ADD(Xscratch0, Xbase, Roffset, add_ext);
        fastmem_location = code.xptr<CodePtr>();
        if constexpr (bitsize == 8) {
            code.LDARB(oaknut::WReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 16) {
            code.LDARH(oaknut::WReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 32) {
            code.LDAR(oaknut::WReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 64) {
            code.LDAR(oaknut::XReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 128) {
            code.LDR(oaknut::QReg{value_idx}, Xscratch0);
            code.DMB(oaknut::BarrierOp::ISH);
        }
    } else {
        fastmem_location = code.xptr<CodePtr>();
        if constexpr (bitsize == 8) {
            code.LDRB(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 16) {
            code.LDRH(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 32) {
            code.LDR(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 64) {
            code.LDR(oaknut::XReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 128) {
            code.LDR(oaknut::QReg{value_idx}, Xbase, Roffset, index_ext);
        }
    }

    return fastmem_location;
}

template<std::size_t bitsize>
CodePtr EmitMemoryStr(oaknut::CodeGenerator& code, int value_idx, oaknut::XReg Xbase, oaknut::XReg Xoffset, bool ordered, bool extend32 = false) {
    static_assert(IsValidBitsize(bitsize));
    const auto index_ext = extend32 ? oaknut::IndexExt::UXTW : oaknut::IndexExt::LSL;
    const auto add_ext = extend32 ? oaknut::AddSubExt::UXTW : oaknut::AddSubExt::LSL;
    const auto Roffset = extend32 ? oaknut::RReg{Xoffset.toW()} : oaknut::RReg{Xoffset};

    CodePtr fastmem_location = code.xptr<CodePtr>();

    if (ordered) {
        code.ADD(Xscratch0, Xbase, Roffset, add_ext);
        fastmem_location = code.xptr<CodePtr>();
        if constexpr (bitsize == 8) {
            code.STLRB(oaknut::WReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 16) {
            code.STLRH(oaknut::WReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 32) {
            code.STLR(oaknut::WReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 64) {
            code.STLR(oaknut::XReg{value_idx}, Xscratch0);
        } else if constexpr (bitsize == 128) {
            code.DMB(oaknut::BarrierOp::ISH);
            code.STR(oaknut::QReg{value_idx}, Xscratch0);
            code.DMB(oaknut::BarrierOp::ISH);
        }
    } else {
        fastmem_location = code.xptr<CodePtr>();
        if constexpr (bitsize == 8) {
            code.STRB(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 16) {
            code.STRH(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 32) {
            code.STR(oaknut::WReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 64) {
            code.STR(oaknut::XReg{value_idx}, Xbase, Roffset, index_ext);
        } else if constexpr (bitsize == 128) {
            code.STR(oaknut::QReg{value_idx}, Xbase, Roffset, index_ext);
        }
    }

    return fastmem_location;
}

template<size_t bitsize>
void InlinePageTableEmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.WriteQ(inst);
        } else {
            return ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        }
    }();
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    EmitMemoryLdr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end] {
        code.l(*fallback);
        code.MOV(Xscratch0, Xaddr);
        EmitRelocation(code, ctx, WrappedReadMemoryLinkTarget(bitsize));
        if (ordered) code.DMB(oaknut::BarrierOp::ISH);
        if constexpr (bitsize == 128) {
            code.MOV(Rvalue.B16(), Q0.B16());
        } else {
            code.MOV(Rvalue.toX(), Xscratch0);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

template<size_t bitsize>
void InlinePageTableEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.ReadQ(args[2]);
        } else {
            return ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        }
    }();
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = InlinePageTableEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    EmitMemoryStr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered);

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end] {
        code.l(*fallback);
        if constexpr (bitsize == 128) {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Q0.B16(), Rvalue.B16());
        } else {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Rvalue.toX());
        }
        if (ordered) code.DMB(oaknut::BarrierOp::ISH);
        EmitRelocation(code, ctx, WrappedWriteMemoryLinkTarget(bitsize));
        if (ordered) code.DMB(oaknut::BarrierOp::ISH);
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

std::optional<DoNotFastmemMarker> ShouldFastmem(EmitContext& ctx, IR::Inst* inst) {
    if (!ctx.conf.fastmem_pointer || !ctx.fastmem.SupportsFastmem()) return std::nullopt;
    const auto marker = std::make_tuple(ctx.block.Location(), inst->GetName());
    if (ctx.fastmem.ShouldFastmem(marker)) return marker;
    return std::nullopt;
}

inline bool ShouldExt32(const EmitContext& ctx) noexcept {
    return ctx.conf.fastmem_address_space_bits == 32 && ctx.conf.silently_mirror_fastmem;
}

template<size_t bitsize>
std::pair<oaknut::XReg, oaknut::XReg> FastmemEmitVAddrLookup(oaknut::CodeGenerator& code, EmitContext& ctx, oaknut::XReg Xaddr, const SharedLabel& fallback) {
    static_assert(IsValidBitsize(bitsize));
    if (ctx.conf.fastmem_address_space_bits == 64 || ShouldExt32(ctx)) {
        return {Xfastmem, Xaddr};
    }

    if (ctx.conf.silently_mirror_fastmem) {
        code.UBFX(Xscratch0, Xaddr, 0, ctx.conf.fastmem_address_space_bits);
        return {Xfastmem, Xscratch0};
    }

    code.LSR(Xscratch0, Xaddr, ctx.conf.fastmem_address_space_bits);
    code.CBNZ(Xscratch0, *fallback);
    return {Xfastmem, Xaddr};
}

template<size_t bitsize>
void FastmemEmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.WriteQ(inst);
        } else {
            return ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
        }
    }();
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    const auto fastmem_location = EmitMemoryLdr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered, ShouldExt32(ctx));

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end, fastmem_location] {
        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = mcl::bit_cast<u64>(code.xptr<void*>()),
                },
                .recompile = ctx.conf.recompile_on_fastmem_failure,
            });

        code.l(*fallback);
        code.MOV(Xscratch0, Xaddr);
        EmitRelocation(code, ctx, WrappedReadMemoryLinkTarget(bitsize));
        if (ordered) code.DMB(oaknut::BarrierOp::ISH);
        if constexpr (bitsize == 128) {
            code.MOV(Rvalue.B16(), Q0.B16());
        } else {
            code.MOV(Rvalue.toX(), Xscratch0);
        }
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

template<size_t bitsize>
void FastmemEmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, DoNotFastmemMarker marker) {
    static_assert(IsValidBitsize(bitsize));
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = [&] {
        if constexpr (bitsize == 128) {
            return ctx.reg_alloc.ReadQ(args[2]);
        } else {
            return ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
        }
    }();
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    SharedLabel fallback = GenSharedLabel(), end = GenSharedLabel();

    const auto [Xbase, Xoffset] = FastmemEmitVAddrLookup<bitsize>(code, ctx, Xaddr, fallback);
    const auto fastmem_location = EmitMemoryStr<bitsize>(code, Rvalue->index(), Xbase, Xoffset, ordered, ShouldExt32(ctx));

    ctx.deferred_emits.emplace_back([&code, &ctx, inst, marker, Xaddr = *Xaddr, Rvalue = *Rvalue, ordered, fallback, end, fastmem_location] {
        ctx.ebi.fastmem_patch_info.emplace(
            fastmem_location - ctx.ebi.entry_point,
            FastmemPatchInfo{
                .marker = marker,
                .fc = FakeCall{
                    .call_pc = mcl::bit_cast<u64>(code.xptr<void*>()),
                },
                .recompile = ctx.conf.recompile_on_fastmem_failure,
            });

        code.l(*fallback);
        if constexpr (bitsize == 128) {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Q0.B16(), Rvalue.B16());
        } else {
            code.MOV(Xscratch0, Xaddr);
            code.MOV(Xscratch1, Rvalue.toX());
        }
        if (ordered) code.DMB(oaknut::BarrierOp::ISH);
        EmitRelocation(code, ctx, WrappedWriteMemoryLinkTarget(bitsize));
        if (ordered) code.DMB(oaknut::BarrierOp::ISH);
        ctx.conf.emit_check_memory_abort(code, ctx, inst, *end);
        code.B(*end);
    });

    code.l(*end);
}

}  // namespace

template<size_t bitsize>
void EmitReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    if (const auto marker = ShouldFastmem(ctx, inst)) {
        FastmemEmitReadMemory<bitsize>(code, ctx, inst, *marker);
    } else if (ctx.conf.page_table_pointer != 0) {
        InlinePageTableEmitReadMemory<bitsize>(code, ctx, inst);
    } else {
        CallbackOnlyEmitReadMemory<bitsize>(code, ctx, inst);
    }
}

template<size_t bitsize>
void EmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    CallbackOnlyEmitExclusiveReadMemory<bitsize>(code, ctx, inst);
}

template<size_t bitsize>
void EmitWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    if (const auto marker = ShouldFastmem(ctx, inst)) {
        FastmemEmitWriteMemory<bitsize>(code, ctx, inst, *marker);
    } else if (ctx.conf.page_table_pointer != 0) {
        InlinePageTableEmitWriteMemory<bitsize>(code, ctx, inst);
    } else {
        CallbackOnlyEmitWriteMemory<bitsize>(code, ctx, inst);
    }
}

template<size_t bitsize>
void EmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(IsValidBitsize(bitsize));
    CallbackOnlyEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
}

template void EmitReadMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitReadMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveReadMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitWriteMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<8>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<16>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<32>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<64>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template void EmitExclusiveWriteMemory<128>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

}  // namespace Dynarmic::Backend::Arm64
