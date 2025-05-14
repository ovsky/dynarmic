/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/a32_address_space.h"

#include <array>
#include <utility>
#include <type_traits>
#include <boost/container/static_vector.hpp>

#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/frontend/A32/a32_location_descriptor.h"
#include "dynarmic/frontend/A32/translate/a32_translate.h"
#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::Backend::Arm64 {

// Helper to force inlining for small trampoline functions
#if defined(__GNUC__) || defined(__clang__)
#define DYN_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define DYN_FORCE_INLINE __forceinline
#else
#define DYN_FORCE_INLINE inline
#endif

namespace {

template<auto mfp, typename T>
DYN_FORCE_INLINE void* EmitCallTrampoline(oaknut::CodeGenerator& code, T* this_) {
    using namespace oaknut::util;

    const auto info = Devirtualize<mfp>(this_);

    oaknut::Label l_addr, l_this;

    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);

    code.align(8);
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto mfp, typename T>
DYN_FORCE_INLINE void* EmitWrappedReadCallTrampoline(oaknut::CodeGenerator& code, T* this_) {
    using namespace oaknut::util;

    const auto info = Devirtualize<mfp>(this_);

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
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto callback, typename T>
DYN_FORCE_INLINE void* EmitExclusiveReadCallTrampoline(oaknut::CodeGenerator& code, const A32::UserConfig& conf) {
    using namespace oaknut::util;

    oaknut::Label l_addr, l_this;

    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr) -> T {
        return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
            return (conf.callbacks->*callback)(vaddr);
        });
    };

    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);

    code.align(8);
    code.l(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.l(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

template<auto mfp, typename T>
DYN_FORCE_INLINE void* EmitWrappedWriteCallTrampoline(oaknut::CodeGenerator& code, T* this_) {
    using namespace oaknut::util;

    const auto info = Devirtualize<mfp>(this_);

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
    code.l(l_this);
    code.dx(info.this_ptr);
    code.l(l_addr);
    code.dx(info.fn_ptr);

    return target;
}

template<auto callback, typename T>
DYN_FORCE_INLINE void* EmitExclusiveWriteCallTrampoline(oaknut::CodeGenerator& code, const A32::UserConfig& conf) {
    using namespace oaknut::util;

    oaknut::Label l_addr, l_this;

    auto fn = [](const A32::UserConfig& conf, A32::VAddr vaddr, T value) -> u32 {
        return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr,
                                                            [&](T expected) -> bool {
                                                                return (conf.callbacks->*callback)(vaddr, value, expected);
                                                            })
                 ? 0
                 : 1;
    };

    void* target = code.xptr<void*>();
    code.LDR(X0, l_this);
    code.LDR(Xscratch0, l_addr);
    code.BR(Xscratch0);

    code.align(8);
    code.l(l_this);
    code.dx(mcl::bit_cast<u64>(&conf));
    code.l(l_addr);
    code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));

    return target;
}

} // anonymous namespace

A32AddressSpace::A32AddressSpace(const A32::UserConfig& conf)
        : AddressSpace(conf.code_cache_size)
        , conf(conf) {
    EmitPrelude();
}

IR::Block A32AddressSpace::GenerateIR(IR::LocationDescriptor descriptor) const {
    // Use static thread_local for temporary IR block to avoid heap allocations
    thread_local IR::Block ir_block;
    ir_block = A32::Translate(A32::LocationDescriptor{descriptor}, conf.callbacks, {conf.arch_version, conf.define_unpredictable_behaviour, conf.hook_hint_instructions});

    Optimization::PolyfillPass(ir_block, {});
    Optimization::NamingPass(ir_block);
    if (conf.HasOptimization(OptimizationFlag::GetSetElimination)) {
        Optimization::A32GetSetElimination(ir_block, {.convert_nzc_to_nz = true});
        Optimization::DeadCodeElimination(ir_block);
    }
    if (conf.HasOptimization(OptimizationFlag::ConstProp)) {
        Optimization::A32ConstantMemoryReads(ir_block, conf.callbacks);
        Optimization::ConstantPropagation(ir_block);
        Optimization::DeadCodeElimination(ir_block);
    }
    Optimization::IdentityRemovalPass(ir_block);
    Optimization::VerificationPass(ir_block);

    return ir_block;
}

void A32AddressSpace::InvalidateCacheRanges(const boost::icl::interval_set<u32>& ranges) {
    InvalidateBasicBlocks(block_ranges.InvalidateRanges(ranges));
}

void A32AddressSpace::EmitPrelude() {
    using namespace oaknut::util;

    UnprotectCodeMemory();

    // Use a static array to batch trampoline emission for better cache locality
    struct TrampolineEntry {
        void** out_ptr;
        void* (*emit_fn)(oaknut::CodeGenerator&, const A32::UserConfig&, void*);
        const void* arg;
    };

    // Lambda wrappers for template emission
    auto emit_read8 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryRead8>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_read16 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryRead16>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_read32 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryRead32>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_read64 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryRead64>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_read8 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead8>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_read16 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead16>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_read32 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead32>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_read64 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedReadCallTrampoline<&A32::UserCallbacks::MemoryRead64>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_exclusive_read8 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead8, u8>(code, conf);
    };
    auto emit_exclusive_read16 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead16, u16>(code, conf);
    };
    auto emit_exclusive_read32 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead32, u32>(code, conf);
    };
    auto emit_exclusive_read64 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveReadCallTrampoline<&A32::UserCallbacks::MemoryRead64, u64>(code, conf);
    };
    auto emit_write8 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite8>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_write16 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite16>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_write32 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite32>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_write64 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::MemoryWrite64>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_write8 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite8>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_write16 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite16>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_write32 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite32>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_wrapped_write64 = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitWrappedWriteCallTrampoline<&A32::UserCallbacks::MemoryWrite64>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_exclusive_write8 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive8, u8>(code, conf);
    };
    auto emit_exclusive_write16 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive16, u16>(code, conf);
    };
    auto emit_exclusive_write32 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive32, u32>(code, conf);
    };
    auto emit_exclusive_write64 = [](oaknut::CodeGenerator& code, const A32::UserConfig& conf, void*) {
        return EmitExclusiveWriteCallTrampoline<&A32::UserCallbacks::MemoryWriteExclusive64, u64>(code, conf);
    };
    auto emit_call_svc = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::CallSVC>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_exception_raised = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::ExceptionRaised>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_isb_raised = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::InstructionSynchronizationBarrierRaised>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_add_ticks = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::AddTicks>(code, static_cast<A32::UserCallbacks*>(cb));
    };
    auto emit_get_ticks_remaining = [](oaknut::CodeGenerator& code, const A32::UserConfig&, void* cb) {
        return EmitCallTrampoline<&A32::UserCallbacks::GetTicksRemaining>(code, static_cast<A32::UserCallbacks*>(cb));
    };

    // Trampoline emission table
    const TrampolineEntry trampolines[] = {
        {reinterpret_cast<void**>(&prelude_info.read_memory_8), emit_read8, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.read_memory_16), emit_read16, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.read_memory_32), emit_read32, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.read_memory_64), emit_read64, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_read_memory_8), emit_wrapped_read8, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_read_memory_16), emit_wrapped_read16, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_read_memory_32), emit_wrapped_read32, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_read_memory_64), emit_wrapped_read64, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.exclusive_read_memory_8), emit_exclusive_read8, nullptr},
        {reinterpret_cast<void**>(&prelude_info.exclusive_read_memory_16), emit_exclusive_read16, nullptr},
        {reinterpret_cast<void**>(&prelude_info.exclusive_read_memory_32), emit_exclusive_read32, nullptr},
        {reinterpret_cast<void**>(&prelude_info.exclusive_read_memory_64), emit_exclusive_read64, nullptr},
        {reinterpret_cast<void**>(&prelude_info.write_memory_8), emit_write8, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.write_memory_16), emit_write16, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.write_memory_32), emit_write32, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.write_memory_64), emit_write64, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_write_memory_8), emit_wrapped_write8, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_write_memory_16), emit_wrapped_write16, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_write_memory_32), emit_wrapped_write32, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.wrapped_write_memory_64), emit_wrapped_write64, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.exclusive_write_memory_8), emit_exclusive_write8, nullptr},
        {reinterpret_cast<void**>(&prelude_info.exclusive_write_memory_16), emit_exclusive_write16, nullptr},
        {reinterpret_cast<void**>(&prelude_info.exclusive_write_memory_32), emit_exclusive_write32, nullptr},
        {reinterpret_cast<void**>(&prelude_info.exclusive_write_memory_64), emit_exclusive_write64, nullptr},
        {reinterpret_cast<void**>(&prelude_info.call_svc), emit_call_svc, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.exception_raised), emit_exception_raised, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.isb_raised), emit_isb_raised, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.add_ticks), emit_add_ticks, conf.callbacks},
        {reinterpret_cast<void**>(&prelude_info.get_ticks_remaining), emit_get_ticks_remaining, conf.callbacks}
    };

    for (const auto& entry : trampolines) {
        *entry.out_ptr = entry.emit_fn(code, conf, const_cast<void*>(entry.arg));
    }

    oaknut::Label return_from_run_code, l_return_to_dispatcher;

    prelude_info.run_code = code.xptr<PreludeInfo::RunCodeFuncType>();
    {
        ABI_PushRegisters(code, ABI_CALLEE_SAVE | (1 << 30), sizeof(StackLayout));

        code.MOV(X19, X0);
        code.MOV(Xstate, X1);
        code.MOV(Xhalt, X2);
        if (conf.page_table) {
            code.MOV(Xpagetable, mcl::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.MOV(Xfastmem, *conf.fastmem_pointer);
        }

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.LDR(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; i++) {
                code.STR(Xscratch0, SP, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.BL(prelude_info.get_ticks_remaining);
            code.MOV(Xticks, X0);
            code.STR(Xticks, SP, offsetof(StackLayout, cycles_to_run));
        }

        code.LDR(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.AND(Wscratch0, Wscratch0, 0xffff0000);
        code.MRS(Xscratch1, oaknut::SystemReg::FPCR);
        code.STR(Wscratch1, SP, offsetof(StackLayout, save_host_fpcr));
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
        if (conf.page_table) {
            code.MOV(Xpagetable, mcl::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.MOV(Xfastmem, *conf.fastmem_pointer);
        }

        if (conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
            code.LDR(Xscratch0, l_return_to_dispatcher);
            for (size_t i = 0; i < RSBCount; i++) {
                code.STR(Xscratch0, SP, offsetof(StackLayout, rsb) + offsetof(RSBEntry, code_ptr) + i * sizeof(RSBEntry));
            }
        }

        if (conf.enable_cycle_counting) {
            code.MOV(Xticks, 1);
            code.STR(Xticks, SP, offsetof(StackLayout, cycles_to_run));
        }

        code.LDR(Wscratch0, Xstate, offsetof(A32JitState, upper_location_descriptor));
        code.AND(Wscratch0, Wscratch0, 0xffff0000);
        code.MRS(Xscratch1, oaknut::SystemReg::FPCR);
        code.STR(Wscratch1, SP, offsetof(StackLayout, save_host_fpcr));
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

        const auto fn = [](A32AddressSpace& self, A32JitState& context) -> CodePtr {
            return self.GetOrEmit(context.GetLocationDescriptor());
        };

        code.align(8);
        code.l(l_this);
        code.dx(mcl::bit_cast<u64>(this));
        code.l(l_addr);
        code.dx(mcl::bit_cast<u64>(Common::FptrCast(fn)));
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

EmitConfig A32AddressSpace::GetEmitConfig() {
    return EmitConfig{
        .optimizations = conf.unsafe_optimizations ? conf.optimizations : conf.optimizations & all_safe_optimizations,

        .hook_isb = conf.hook_isb,

        .cntfreq_el0{},
        .ctr_el0{},
        .dczid_el0{},
        .tpidrro_el0{},
        .tpidr_el0{},

        .check_halt_on_memory_access = conf.check_halt_on_memory_access,

        .page_table_pointer = mcl::bit_cast<u64>(conf.page_table),
        .page_table_address_space_bits = 32,
        .page_table_pointer_mask_bits = conf.page_table_pointer_mask_bits,
        .silently_mirror_page_table = true,
        .absolute_offset_page_table = conf.absolute_offset_page_table,
        .detect_misaligned_access_via_page_table = conf.detect_misaligned_access_via_page_table,
        .only_detect_misalignment_via_page_table_on_page_boundary = conf.only_detect_misalignment_via_page_table_on_page_boundary,

        .fastmem_pointer = conf.fastmem_pointer,
        .recompile_on_fastmem_failure = conf.recompile_on_fastmem_failure,
        .fastmem_address_space_bits = 32,
        .silently_mirror_fastmem = true,

        .wall_clock_cntpct = conf.wall_clock_cntpct,
        .enable_cycle_counting = conf.enable_cycle_counting,

        .always_little_endian = conf.always_little_endian,

        .descriptor_to_fpcr = [](const IR::LocationDescriptor& location) { return FP::FPCR{A32::LocationDescriptor{location}.FPSCR().Value()}; },
        .emit_cond = EmitA32Cond,
        .emit_condition_failed_terminal = EmitA32ConditionFailedTerminal,
        .emit_terminal = EmitA32Terminal,
        .emit_check_memory_abort = EmitA32CheckMemoryAbort,

        .state_nzcv_offset = offsetof(A32JitState, cpsr_nzcv),
        .state_fpsr_offset = offsetof(A32JitState, fpsr),
        .state_exclusive_state_offset = offsetof(A32JitState, exclusive_state),

        .coprocessors = conf.coprocessors,

        .very_verbose_debugging_output = conf.very_verbose_debugging_output,
    };
}

void A32AddressSpace::RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo&) {
    const A32::LocationDescriptor descriptor{block.Location()};
    const A32::LocationDescriptor end_location{block.EndLocation()};
    const auto range = boost::icl::discrete_interval<u32>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);
}

#undef DYN_FORCE_INLINE

}  // namespace Dynarmic::Backend::Arm64
