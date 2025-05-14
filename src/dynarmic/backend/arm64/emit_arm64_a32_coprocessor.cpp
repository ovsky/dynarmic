/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <oaknut/oaknut.hpp>
#include <utility>
#include <optional>
#include <variant>
#include <memory>
#include <array>

#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/interface/A32/coprocessor.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

// Helper: Always_inline for small helpers
#if defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline
#endif

// Emit a coprocessor exception (should never return)
[[noreturn]] static void EmitCoprocessorException() {
    ASSERT_FALSE("Should raise coproc exception here");
    __builtin_unreachable();
}

// Helper to call a coprocessor callback, with optional arguments
FORCE_INLINE static void CallCoprocCallback(
    oaknut::CodeGenerator& code,
    EmitContext& ctx,
    const A32::Coprocessor::Callback& callback,
    IR::Inst* inst = nullptr,
    std::optional<Argument::copyable_reference> arg0 = std::nullopt,
    std::optional<Argument::copyable_reference> arg1 = std::nullopt
) {
    ctx.reg_alloc.PrepareForCall({}, arg0, arg1);

    if (callback.user_arg) {
        code.MOV(X0, reinterpret_cast<u64>(*callback.user_arg));
    }

    code.MOV(Xscratch0, reinterpret_cast<u64>(callback.function));
    code.BLR(Xscratch0);

    if (inst) {
        ctx.reg_alloc.DefineAsRegister(inst, X0);
    }
}

// Helper: Extract coproc_info with bounds checking and type safety
template<size_t N>
FORCE_INLINE static auto GetCoprocInfo(const IR::Inst* inst) {
    const auto& coproc_info = inst->GetArg(0).GetCoprocInfo();
    ASSERT_MSG(coproc_info.size() >= N, "coproc_info too small");
    return coproc_info;
}

// Helper: Get coprocessor pointer, or emit exception and return
FORCE_INLINE static std::shared_ptr<A32::Coprocessor> GetCoprocOrException(
    EmitContext& ctx, size_t coproc_num
) {
    auto coproc = ctx.conf.coprocessors[coproc_num];
    if (!coproc) {
        EmitCoprocessorException();
    }
    return coproc;
}

// Helper: Handle action variant for one-word send/get
template<typename ActionVariant, typename CallbackHandler, typename PtrHandler>
FORCE_INLINE static void HandleCoprocAction(
    const ActionVariant& action,
    CallbackHandler&& cb_handler,
    PtrHandler&& ptr_handler
) {
    if (std::holds_alternative<std::monostate>(action)) {
        EmitCoprocessorException();
    } else if (const auto cb = std::get_if<A32::Coprocessor::Callback>(&action)) {
        cb_handler(*cb);
    } else if (const auto ptr = std::get_if<u32*>(&action)) {
        ptr_handler(*ptr);
    } else {
        UNREACHABLE();
    }
}

// Helper: Handle action variant for two-word send/get
template<typename ActionVariant, typename CallbackHandler, typename PtrsHandler>
FORCE_INLINE static void HandleCoprocAction2(
    const ActionVariant& action,
    CallbackHandler&& cb_handler,
    PtrsHandler&& ptrs_handler
) {
    if (std::holds_alternative<std::monostate>(action)) {
        EmitCoprocessorException();
    } else if (const auto cb = std::get_if<A32::Coprocessor::Callback>(&action)) {
        cb_handler(*cb);
    } else if (const auto ptrs = std::get_if<std::array<u32*, 2>>(&action)) {
        ptrs_handler(*ptrs);
    } else {
        UNREACHABLE();
    }
}

// EmitIR for A32CoprocInternalOperation
template<>
void EmitIR<IR::Opcode::A32CoprocInternalOperation>(
    oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst
) {
    const auto coproc_info = GetCoprocInfo<7>(inst);
    const size_t coproc_num = coproc_info[0];
    const bool two = coproc_info[1] != 0;
    const auto opc1 = static_cast<unsigned>(coproc_info[2]);
    const auto CRd = static_cast<A32::CoprocReg>(coproc_info[3]);
    const auto CRn = static_cast<A32::CoprocReg>(coproc_info[4]);
    const auto CRm = static_cast<A32::CoprocReg>(coproc_info[5]);
    const auto opc2 = static_cast<unsigned>(coproc_info[6]);

    auto coproc = GetCoprocOrException(ctx, coproc_num);
    const auto action = coproc->CompileInternalOperation(two, opc1, CRd, CRn, CRm, opc2);
    if (!action) {
        EmitCoprocessorException();
    }
    CallCoprocCallback(code, ctx, *action);
}

// EmitIR for A32CoprocSendOneWord
template<>
void EmitIR<IR::Opcode::A32CoprocSendOneWord>(
    oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst
) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto coproc_info = GetCoprocInfo<6>(inst);
    const size_t coproc_num = coproc_info[0];
    const bool two = coproc_info[1] != 0;
    const auto opc1 = static_cast<unsigned>(coproc_info[2]);
    const auto CRn = static_cast<A32::CoprocReg>(coproc_info[3]);
    const auto CRm = static_cast<A32::CoprocReg>(coproc_info[4]);
    const auto opc2 = static_cast<unsigned>(coproc_info[5]);

    auto coproc = GetCoprocOrException(ctx, coproc_num);
    const auto action = coproc->CompileSendOneWord(two, opc1, CRn, CRm, opc2);

    HandleCoprocAction(action,
        [&](const A32::Coprocessor::Callback& cb) {
            CallCoprocCallback(code, ctx, cb, nullptr, args[1]);
        },
        [&](u32* destination_ptr) {
            auto Wvalue = ctx.reg_alloc.ReadW(args[1]);
            RegAlloc::Realize(Wvalue);
            code.MOV(Xscratch0, reinterpret_cast<u64>(destination_ptr));
            code.STR(Wvalue, Xscratch0);
        }
    );
}

// EmitIR for A32CoprocSendTwoWords
template<>
void EmitIR<IR::Opcode::A32CoprocSendTwoWords>(
    oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst
) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto coproc_info = GetCoprocInfo<4>(inst);
    const size_t coproc_num = coproc_info[0];
    const bool two = coproc_info[1] != 0;
    const auto opc = static_cast<unsigned>(coproc_info[2]);
    const auto CRm = static_cast<A32::CoprocReg>(coproc_info[3]);

    auto coproc = GetCoprocOrException(ctx, coproc_num);
    const auto action = coproc->CompileSendTwoWords(two, opc, CRm);

    HandleCoprocAction2(action,
        [&](const A32::Coprocessor::Callback& cb) {
            CallCoprocCallback(code, ctx, cb, nullptr, args[1], args[2]);
        },
        [&](const std::array<u32*, 2>& destination_ptrs) {
            auto Wvalue1 = ctx.reg_alloc.ReadW(args[1]);
            auto Wvalue2 = ctx.reg_alloc.ReadW(args[2]);
            RegAlloc::Realize(Wvalue1, Wvalue2);
            code.MOV(Xscratch0, reinterpret_cast<u64>(destination_ptrs[0]));
            code.MOV(Xscratch1, reinterpret_cast<u64>(destination_ptrs[1]));
            code.STR(Wvalue1, Xscratch0);
            code.STR(Wvalue2, Xscratch1);
        }
    );
}

// EmitIR for A32CoprocGetOneWord
template<>
void EmitIR<IR::Opcode::A32CoprocGetOneWord>(
    oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst
) {
    const auto coproc_info = GetCoprocInfo<6>(inst);
    const size_t coproc_num = coproc_info[0];
    const bool two = coproc_info[1] != 0;
    const auto opc1 = static_cast<unsigned>(coproc_info[2]);
    const auto CRn = static_cast<A32::CoprocReg>(coproc_info[3]);
    const auto CRm = static_cast<A32::CoprocReg>(coproc_info[4]);
    const auto opc2 = static_cast<unsigned>(coproc_info[5]);

    auto coproc = GetCoprocOrException(ctx, coproc_num);
    const auto action = coproc->CompileGetOneWord(two, opc1, CRn, CRm, opc2);

    HandleCoprocAction(action,
        [&](const A32::Coprocessor::Callback& cb) {
            CallCoprocCallback(code, ctx, cb, inst);
        },
        [&](u32* source_ptr) {
            auto Wvalue = ctx.reg_alloc.WriteW(inst);
            RegAlloc::Realize(Wvalue);
            code.MOV(Xscratch0, reinterpret_cast<u64>(source_ptr));
            code.LDR(Wvalue, Xscratch0);
        }
    );
}

// EmitIR for A32CoprocGetTwoWords
template<>
void EmitIR<IR::Opcode::A32CoprocGetTwoWords>(
    oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst
) {
    const auto coproc_info = GetCoprocInfo<4>(inst);
    const size_t coproc_num = coproc_info[0];
    const bool two = coproc_info[1] != 0;
    const unsigned opc = coproc_info[2];
    const auto CRm = static_cast<A32::CoprocReg>(coproc_info[3]);

    auto coproc = GetCoprocOrException(ctx, coproc_num);
    const auto action = coproc->CompileGetTwoWords(two, opc, CRm);

    HandleCoprocAction2(action,
        [&](const A32::Coprocessor::Callback& cb) {
            CallCoprocCallback(code, ctx, cb, inst);
        },
        [&](const std::array<u32*, 2>& source_ptrs) {
            auto Xvalue = ctx.reg_alloc.WriteX(inst);
            RegAlloc::Realize(Xvalue);
            code.MOV(Xscratch0, reinterpret_cast<u64>(source_ptrs[0]));
            code.MOV(Xscratch1, reinterpret_cast<u64>(source_ptrs[1]));
            code.LDR(Xvalue, Xscratch0);
            code.LDR(Wscratch1, Xscratch1);
            code.BFI(Xvalue, Xscratch1, 32, 32);
        }
    );
}

// EmitIR for A32CoprocLoadWords
template<>
void EmitIR<IR::Opcode::A32CoprocLoadWords>(
    oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst
) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto coproc_info = GetCoprocInfo<6>(inst);
    const size_t coproc_num = coproc_info[0];
    const bool two = coproc_info[1] != 0;
    const bool long_transfer = coproc_info[2] != 0;
    const auto CRd = static_cast<A32::CoprocReg>(coproc_info[3]);
    const bool has_option = coproc_info[4] != 0;

    std::optional<u8> option = std::nullopt;
    if (has_option) {
        option = coproc_info[5];
    }

    auto coproc = GetCoprocOrException(ctx, coproc_num);
    const auto action = coproc->CompileLoadWords(two, long_transfer, CRd, option);
    if (!action) {
        EmitCoprocessorException();
    }
    CallCoprocCallback(code, ctx, *action, nullptr, args[1]);
}

// EmitIR for A32CoprocStoreWords
template<>
void EmitIR<IR::Opcode::A32CoprocStoreWords>(
    oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst
) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto coproc_info = GetCoprocInfo<6>(inst);
    const size_t coproc_num = coproc_info[0];
    const bool two = coproc_info[1] != 0;
    const bool long_transfer = coproc_info[2] != 0;
    const auto CRd = static_cast<A32::CoprocReg>(coproc_info[3]);
    const bool has_option = coproc_info[4] != 0;

    std::optional<u8> option = std::nullopt;
    if (has_option) {
        option = coproc_info[5];
    }

    auto coproc = GetCoprocOrException(ctx, coproc_num);
    const auto action = coproc->CompileStoreWords(two, long_transfer, CRd, option);
    if (!action) {
        EmitCoprocessorException();
    }
    CallCoprocCallback(code, ctx, *action, nullptr, args[1]);
}

#undef FORCE_INLINE

}  // namespace Dynarmic::Backend::Arm64
