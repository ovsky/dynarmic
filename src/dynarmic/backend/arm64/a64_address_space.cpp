/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/a64_address_space.h"

#include <array>
#include <utility>
#include <type_traits>
#include <cstdint>

#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::Backend::Arm64 {

// Helper to reduce code duplication for trampolines
namespace {
    template<auto mfp, typename T>
    inline void* EmitCallTrampolineImpl(oaknut::CodeGenerator& code, T* this_ptr) {
        using namespace oaknut::util;
        const auto info = Devirtualize<mfp>(this_ptr);
        oaknut::Label l_addr, l_this;
        void* target = code.xptr<void*>();
        code.LDR(X0, l_this);
        code.LDR(Xscratch0, l_addr);
        code.BR(Xscratch0);
        code.align(8);
        code.l(l_this); code.dx(info.this_ptr);
        code.l(l_addr); code.dx(info.fn_ptr);
        return target;
    }

    template<auto mfp, typename T>
    inline void* EmitWrappedReadCallTrampolineImpl(oaknut::CodeGenerator& code, T* this_ptr) {
        using namespace oaknut::util;
        const auto info = Devirtualize<mfp>(this_ptr);
        oaknut::Label l_addr, l_this;
        constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Xscratch0);
        void* target = code.xptr<void*>();
        ABI_PushRegisters(code, save_regs, 0);
        code.LDR(X0, l_this);
        code.MOV(X1, Xscratch0);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        code.MOV(Xscratch0, X0);
        ABI_PopRegisters(code, save_regs, 0);
        code.RET();
        code.align(8);
        code.l(l_this); code.dx(info.this_ptr);
        code.l(l_addr); code.dx(info.fn_ptr);
        return target;
    }

    template<auto callback, typename T>
    inline void* EmitExclusiveReadCallTrampolineImpl(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
        using namespace oaknut::util;
        oaknut::Label l_addr, l_this;
        auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr) -> T {
            return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
                return (conf.callbacks->*callback)(vaddr);
            });
        };
        void* target = code.xptr<void*>();
        code.LDR(X0, l_this);
        code.LDR(Xscratch0, l_addr);
        code.BR(Xscratch0);
        code.align(8);
        code.l(l_this); code.dx(mcl::bit_cast<u64>(&conf));
        code.l(l_addr); code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));
        return target;
    }

    template<auto mfp, typename T>
    inline void* EmitWrappedWriteCallTrampolineImpl(oaknut::CodeGenerator& code, T* this_ptr) {
        using namespace oaknut::util;
        const auto info = Devirtualize<mfp>(this_ptr);
        oaknut::Label l_addr, l_this;
        constexpr u64 save_regs = ABI_CALLER_SAVE;
        void* target = code.xptr<void*>();
        ABI_PushRegisters(code, save_regs, 0);
        code.LDR(X0, l_this);
        code.MOV(X1, Xscratch0);
        code.MOV(X2, Xscratch1);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        ABI_PopRegisters(code, save_regs, 0);
        code.RET();
        code.align(8);
        code.l(l_this); code.dx(info.this_ptr);
        code.l(l_addr); code.dx(info.fn_ptr);
        return target;
    }

    template<auto callback, typename T>
    inline void* EmitExclusiveWriteCallTrampolineImpl(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
        using namespace oaknut::util;
        oaknut::Label l_addr, l_this;
        auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr, T value) -> u32 {
            return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr,
                [&](T expected) -> bool {
                    return (conf.callbacks->*callback)(vaddr, value, expected);
                }) ? 0 : 1;
        };
        void* target = code.xptr<void*>();
        code.LDR(X0, l_this);
        code.LDR(Xscratch0, l_addr);
        code.BR(Xscratch0);
        code.align(8);
        code.l(l_this); code.dx(mcl::bit_cast<u64>(&conf));
        code.l(l_addr); code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));
        return target;
    }

    inline void* EmitRead128CallTrampolineImpl(oaknut::CodeGenerator& code, A64::UserCallbacks* this_ptr) {
        using namespace oaknut::util;
        const auto info = Devirtualize<&A64::UserCallbacks::MemoryRead128>(this_ptr);
        oaknut::Label l_addr, l_this;
        void* target = code.xptr<void*>();
        ABI_PushRegisters(code, (1ull << 29) | (1ull << 30), 0);
        code.LDR(X0, l_this);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        code.FMOV(D0, X0);
        code.FMOV(V0.D()[1], X1);
        ABI_PopRegisters(code, (1ull << 29) | (1ull << 30), 0);
        code.RET();
        code.align(8);
        code.l(l_this); code.dx(info.this_ptr);
        code.l(l_addr); code.dx(info.fn_ptr);
        return target;
    }

    inline void* EmitWrappedRead128CallTrampolineImpl(oaknut::CodeGenerator& code, A64::UserCallbacks* this_ptr) {
        using namespace oaknut::util;
        const auto info = Devirtualize<&A64::UserCallbacks::MemoryRead128>(this_ptr);
        oaknut::Label l_addr, l_this;
        constexpr u64 save_regs = ABI_CALLER_SAVE & ~ToRegList(Q0);
        void* target = code.xptr<void*>();
        ABI_PushRegisters(code, save_regs, 0);
        code.LDR(X0, l_this);
        code.MOV(X1, Xscratch0);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        code.FMOV(D0, X0);
        code.FMOV(V0.D()[1], X1);
        ABI_PopRegisters(code, save_regs, 0);
        code.RET();
        code.align(8);
        code.l(l_this); code.dx(info.this_ptr);
        code.l(l_addr); code.dx(info.fn_ptr);
        return target;
    }

    inline void* EmitExclusiveRead128CallTrampolineImpl(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
        using namespace oaknut::util;
        oaknut::Label l_addr, l_this;
        auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr) -> Vector {
            return conf.global_monitor->ReadAndMark<Vector>(conf.processor_id, vaddr, [&]() -> Vector {
                return conf.callbacks->MemoryRead128(vaddr);
            });
        };
        void* target = code.xptr<void*>();
        ABI_PushRegisters(code, (1ull << 29) | (1ull << 30), 0);
        code.LDR(X0, l_this);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        code.FMOV(D0, X0);
        code.FMOV(V0.D()[1], X1);
        ABI_PopRegisters(code, (1ull << 29) | (1ull << 30), 0);
        code.RET();
        code.align(8);
        code.l(l_this); code.dx(mcl::bit_cast<u64>(&conf));
        code.l(l_addr); code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));
        return target;
    }

    inline void* EmitWrite128CallTrampolineImpl(oaknut::CodeGenerator& code, A64::UserCallbacks* this_ptr) {
        using namespace oaknut::util;
        const auto info = Devirtualize<&A64::UserCallbacks::MemoryWrite128>(this_ptr);
        oaknut::Label l_addr, l_this;
        void* target = code.xptr<void*>();
        code.LDR(X0, l_this);
        code.FMOV(X2, D0);
        code.FMOV(X3, V0.D()[1]);
        code.LDR(Xscratch0, l_addr);
        code.BR(Xscratch0);
        code.align(8);
        code.l(l_this); code.dx(info.this_ptr);
        code.l(l_addr); code.dx(info.fn_ptr);
        return target;
    }

    inline void* EmitWrappedWrite128CallTrampolineImpl(oaknut::CodeGenerator& code, A64::UserCallbacks* this_ptr) {
        using namespace oaknut::util;
        const auto info = Devirtualize<&A64::UserCallbacks::MemoryWrite128>(this_ptr);
        oaknut::Label l_addr, l_this;
        constexpr u64 save_regs = ABI_CALLER_SAVE;
        void* target = code.xptr<void*>();
        ABI_PushRegisters(code, save_regs, 0);
        code.LDR(X0, l_this);
        code.MOV(X1, Xscratch0);
        code.FMOV(X2, D0);
        code.FMOV(X3, V0.D()[1]);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        ABI_PopRegisters(code, save_regs, 0);
        code.RET();
        code.align(8);
        code.l(l_this); code.dx(info.this_ptr);
        code.l(l_addr); code.dx(info.fn_ptr);
        return target;
    }

    inline void* EmitExclusiveWrite128CallTrampolineImpl(oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
        using namespace oaknut::util;
        oaknut::Label l_addr, l_this;
        auto fn = [](const A64::UserConfig& conf, A64::VAddr vaddr, Vector value) -> u32 {
            return conf.global_monitor->DoExclusiveOperation<Vector>(conf.processor_id, vaddr,
                [&](Vector expected) -> bool {
                    return conf.callbacks->MemoryWriteExclusive128(vaddr, value, expected);
                }) ? 0 : 1;
        };
        void* target = code.xptr<void*>();
        code.LDR(X0, l_this);
        code.FMOV(X2, D0);
        code.FMOV(X3, V0.D()[1]);
        code.LDR(Xscratch0, l_addr);
        code.BR(Xscratch0);
        code.align(8);
        code.l(l_this); code.dx(mcl::bit_cast<u64>(&conf));
        code.l(l_addr); code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));
        return target;
    }
} // namespace

A64AddressSpace::A64AddressSpace(const A64::UserConfig& conf)
    : AddressSpace(conf.code_cache_size)
    , conf(conf)
{
    EmitPrelude();
}

IR::Block A64AddressSpace::GenerateIR(IR::LocationDescriptor descriptor) const {
    const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
    IR::Block ir_block = A64::Translate(A64::LocationDescriptor{descriptor}, get_code,
        {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});

    Optimization::A64CallbackConfigPass(ir_block, conf);
    Optimization::NamingPass(ir_block);
    if (conf.HasOptimization(OptimizationFlag::GetSetElimination) && !conf.check_halt_on_memory_access) {
        Optimization::A64GetSetElimination(ir_block);
        Optimization::DeadCodeElimination(ir_block);
    }
    if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
        Optimization::ConstantPropagation(ir_block);
        Optimization::DeadCodeElimination(ir_block);
    }
    if (conf.HasOptimization(OptimizationFlag::MiscIROpt)) {
        Optimization::A64MergeInterpretBlocksPass(ir_block, conf.callbacks);
    }
    Optimization::VerificationPass(ir_block);

    return ir_block;
}

void A64AddressSpace::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
}

void A64AddressSpace::EmitPrelude() {
    using namespace oaknut::util;

    UnprotectCodeMemory();

    // Use lambdas to reduce code repetition for similar assignments
    auto assign_trampolines = [this](auto& info, oaknut::CodeGenerator& code, const A64::UserConfig& conf) {
        info.read_memory_8   = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryRead8>(code, conf.callbacks);
        info.read_memory_16  = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryRead16>(code, conf.callbacks);
        info.read_memory_32  = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryRead32>(code, conf.callbacks);
        info.read_memory_64  = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryRead64>(code, conf.callbacks);
        info.read_memory_128 = EmitRead128CallTrampolineImpl(code, conf.callbacks);

        info.wrapped_read_memory_8   = EmitWrappedReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead8>(code, conf.callbacks);
        info.wrapped_read_memory_16  = EmitWrappedReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead16>(code, conf.callbacks);
        info.wrapped_read_memory_32  = EmitWrappedReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead32>(code, conf.callbacks);
        info.wrapped_read_memory_64  = EmitWrappedReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead64>(code, conf.callbacks);
        info.wrapped_read_memory_128 = EmitWrappedRead128CallTrampolineImpl(code, conf.callbacks);

        info.exclusive_read_memory_8   = EmitExclusiveReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead8, u8>(code, conf);
        info.exclusive_read_memory_16  = EmitExclusiveReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead16, u16>(code, conf);
        info.exclusive_read_memory_32  = EmitExclusiveReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead32, u32>(code, conf);
        info.exclusive_read_memory_64  = EmitExclusiveReadCallTrampolineImpl<&A64::UserCallbacks::MemoryRead64, u64>(code, conf);
        info.exclusive_read_memory_128 = EmitExclusiveRead128CallTrampolineImpl(code, conf);

        info.write_memory_8   = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite8>(code, conf.callbacks);
        info.write_memory_16  = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite16>(code, conf.callbacks);
        info.write_memory_32  = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite32>(code, conf.callbacks);
        info.write_memory_64  = EmitCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite64>(code, conf.callbacks);
        info.write_memory_128 = EmitWrite128CallTrampolineImpl(code, conf.callbacks);

        info.wrapped_write_memory_8   = EmitWrappedWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite8>(code, conf.callbacks);
        info.wrapped_write_memory_16  = EmitWrappedWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite16>(code, conf.callbacks);
        info.wrapped_write_memory_32  = EmitWrappedWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite32>(code, conf.callbacks);
        info.wrapped_write_memory_64  = EmitWrappedWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWrite64>(code, conf.callbacks);
        info.wrapped_write_memory_128 = EmitWrappedWrite128CallTrampolineImpl(code, conf.callbacks);

        info.exclusive_write_memory_8   = EmitExclusiveWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWriteExclusive8, u8>(code, conf);
        info.exclusive_write_memory_16  = EmitExclusiveWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWriteExclusive16, u16>(code, conf);
        info.exclusive_write_memory_32  = EmitExclusiveWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWriteExclusive32, u32>(code, conf);
        info.exclusive_write_memory_64  = EmitExclusiveWriteCallTrampolineImpl<&A64::UserCallbacks::MemoryWriteExclusive64, u64>(code, conf);
        info.exclusive_write_memory_128 = EmitExclusiveWrite128CallTrampolineImpl(code, conf);

        info.call_svc = EmitCallTrampolineImpl<&A64::UserCallbacks::CallSVC>(code, conf.callbacks);
        info.exception_raised = EmitCallTrampolineImpl<&A64::UserCallbacks::ExceptionRaised>(code, conf.callbacks);
        info.isb_raised = EmitCallTrampolineImpl<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>(code, conf.callbacks);
        info.ic_raised = EmitCallTrampolineImpl<&A64::UserCallbacks::InstructionCacheOperationRaised>(code, conf.callbacks);
        info.dc_raised = EmitCallTrampolineImpl<&A64::UserCallbacks::DataCacheOperationRaised>(code, conf.callbacks);
        info.get_cntpct = EmitCallTrampolineImpl<&A64::UserCallbacks::GetCNTPCT>(code, conf.callbacks);
        info.add_ticks = EmitCallTrampolineImpl<&A64::UserCallbacks::AddTicks>(code, conf.callbacks);
        info.get_ticks_remaining = EmitCallTrampolineImpl<&A64::UserCallbacks::GetTicksRemaining>(code, conf.callbacks);
    };

    assign_trampolines(prelude_info, code, conf);

    oaknut::Label return_from_run_code, l_return_to_dispatcher;

    prelude_info.run_code = code.xptr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));
        code.MOV(X19, X0);
        code.MOV(Xstate, X1);
        code.MOV(Xhalt, X2);
        if (conf.page_table) code.MOV(Xpagetable, mcl::bit_cast<u64>(conf.page_table));
        if (conf.fastmem_pointer) code.MOV(Xfastmem, *conf.fastmem_pointer);

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.LDR(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; ++i) {
                code.STR(Xscratch0, SP, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.BL(prelude_info.get_ticks_remaining);
            code.MOV(Xticks, X0);
            code.STR(Xticks, SP, offsetof(StackLayout, cycles_to_run));
        }

        code.MRS(Xscratch1, oaknut::SystemReg::FPCR);
        code.STR(Wscratch1, SP, offsetof(StackLayout, save_host_fpcr));
        code.LDR(Wscratch0, Xstate, offsetof(A64JitState, fpcr));
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);

        code.LDAR(Wscratch0, Xhalt);
        code.CBNZ(Wscratch0, return_from_run_code);

        code.BR(X19);
    }

    prelude_info.step_code = code.xptr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));
        code.MOV(X19, X0);
        code.MOV(Xstate, X1);
        code.MOV(Xhalt, X2);
        if (conf.page_table) code.MOV(Xpagetable, mcl::bit_cast<u64>(conf.page_table));
        if (conf.fastmem_pointer) code.MOV(Xfastmem, *conf.fastmem_pointer);

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.LDR(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; ++i) {
                code.STR(Xscratch0, SP, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.MOV(Xticks, 1);
            code.STR(Xticks, SP, offsetof(StackLayout, cycles_to_run));
        }

        code.MRS(Xscratch1, oaknut::SystemReg::FPCR);
        code.STR(Wscratch1, SP, offsetof(StackLayout, save_host_fpcr));
        code.LDR(Wscratch0, Xstate, offsetof(A64JitState, fpcr));
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);

        oaknut::Label step_hr_loop;
        code.l(step_hr_loop);
        code.LDAXR(Wscratch0, Xhalt);
        code.CBNZ(Wscratch0, return_from_run_code);
        code.ORR(Wscratch0, Wscratch0, static_cast<u32>(HaltReason::Step));
        code.STLXR(Wscratch1, Wscratch0, Xhalt);
        code.CBNZ(Wscratch1, step_hr_loop);

        code.BR(X19);
    }

    prelude_info.return_to_dispatcher = code.xptr<void*>();
    {
        oaknut::Label l_this, l_addr;
        code.LDAR(Wscratch0, Xhalt);
        code.CBNZ(Wscratch0, return_from_run_code);

        if (conf.enable_cycle_counting) {
            code.CMP(Xticks, 0);
            code.B(LE, return_from_run_code);
        }

        code.LDR(X0, l_this);
        code.MOV(X1, Xstate);
        code.LDR(Xscratch0, l_addr);
        code.BLR(Xscratch0);
        code.BR(X0);

        const auto fn = [](A64AddressSpace& self, A64JitState& context) -> CodePtr {
            return self.GetOrEmit(context.GetLocationDescriptor());
        };

        code.align(8);
        code.l(l_this); code.dx(mcl::bit_cast<u64>(this));
        code.l(l_addr); code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));
    }

    prelude_info.return_from_run_code = code.xptr<void*>();
    {
        code.l(return_from_run_code);

        if (conf.enable_cycle_counting) {
            code.LDR(X1, SP, offsetof(StackLayout, cycles_to_run));
            code.SUB(X1, X1, Xticks);
            code.BL(prelude_info.add_ticks);
        }

        code.LDR(Wscratch0, SP, offsetof(StackLayout, save_host_fpcr));
        code.MSR(oaknut::SystemReg::FPCR, Xscratch0);

        oaknut::Label exit_hr_loop;
        code.l(exit_hr_loop);
        code.LDAXR(W0, Xhalt);
        code.STLXR(Wscratch0, WZR, Xhalt);
        code.CBNZ(Wscratch0, exit_hr_loop);

        ABI_PopRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));
        code.RET();
    }

    code.align(8);
    code.l(l_return_to_dispatcher);
    code.dx(mcl::bit_cast<u64>(prelude_info.return_to_dispatcher));

    prelude_info.end_of_prelude = code.offset();

    mem.invalidate_all();
    ProtectCodeMemory();
}

EmitConfig A64AddressSpace::GetEmitConfig() {
    return EmitConfig{
        .optimizations = conf.unsafe_optimizations ? conf.optimizations : conf.optimizations & all_safe_optimizations,
        .hook_isb = conf.hook_isb,
        .cntfreq_el0 = conf.cntfrq_el0,
        .ctr_el0 = conf.ctr_el0,
        .dczid_el0 = conf.dczid_el0,
        .tpidrro_el0 = conf.tpidrro_el0,
        .tpidr_el0 = conf.tpidr_el0,
        .check_halt_on_memory_access = conf.check_halt_on_memory_access,
        .page_table_pointer = mcl::bit_cast<u64>(conf.page_table),
        .page_table_address_space_bits = conf.page_table_address_space_bits,
        .page_table_pointer_mask_bits = conf.page_table_pointer_mask_bits,
        .silently_mirror_page_table = conf.silently_mirror_page_table,
        .absolute_offset_page_table = conf.absolute_offset_page_table,
        .detect_misaligned_access_via_page_table = conf.detect_misaligned_access_via_page_table,
        .only_detect_misalignment_via_page_table_on_page_boundary = conf.only_detect_misalignment_via_page_table_on_page_boundary,
        .fastmem_pointer = conf.fastmem_pointer,
        .recompile_on_fastmem_failure = conf.recompile_on_fastmem_failure,
        .fastmem_address_space_bits = conf.fastmem_address_space_bits,
        .silently_mirror_fastmem = conf.silently_mirror_fastmem,
        .wall_clock_cntpct = conf.wall_clock_cntpct,
        .enable_cycle_counting = conf.enable_cycle_counting,
        .always_little_endian = true,
        .descriptor_to_fpcr = [](const IR::LocationDescriptor& location) { return A64::LocationDescriptor{location}.FPCR(); },
        .emit_cond = EmitA64Cond,
        .emit_condition_failed_terminal = EmitA64ConditionFailedTerminal,
        .emit_terminal = EmitA64Terminal,
        .emit_check_memory_abort = EmitA64CheckMemoryAbort,
        .state_nzcv_offset = offsetof(A64JitState, cpsr_nzcv),
        .state_fpsr_offset = offsetof(A64JitState, fpsr),
        .state_exclusive_state_offset = offsetof(A64JitState, exclusive_state),
        .coprocessors{},
        .very_verbose_debugging_output = conf.very_verbose_debugging_output,
    };
}

void A64AddressSpace::RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo&) {
    const A64::LocationDescriptor descriptor{block.Location()};
    const A64::LocationDescriptor end_location{block.EndLocation()};
    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);
}

}  // namespace Dynarmic::Backend::Arm64
