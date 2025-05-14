/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/abi.h"

#include <vector>
#include <array>
#include <bit>
#include <algorithm>
#include <mcl/bit/bit_field.hpp>
#include <mcl/stdint.hpp>
#include <oaknut/oaknut.hpp>

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

static constexpr size_t gpr_size = 8;
static constexpr size_t fpr_size = 16;

// Use std::span for non-owning views, and reserve vectors for performance.
struct FrameInfo {
    std::vector<int> gprs;
    std::vector<int> fprs;
    size_t frame_size;
    size_t gprs_size;
    size_t fprs_size;
};

// Use std::popcount for fast bit counting, and std::countr_zero for fast index finding.
static std::vector<int> ListToIndexes(u32 list) {
    std::vector<int> indexes;
    indexes.reserve(std::popcount(list));
    while (list) {
        int idx = std::countr_zero(list);
        indexes.push_back(idx);
        list &= ~(1u << idx);
    }
    return indexes;
}

static FrameInfo CalculateFrameInfo(RegisterList rl, size_t frame_size) {
    const auto gprs = ListToIndexes(static_cast<u32>(rl));
    const auto fprs = ListToIndexes(static_cast<u32>(rl >> 32));

    const size_t num_gprs = gprs.size();
    const size_t num_fprs = fprs.size();

    // Align gprs_size to 16 bytes for paired STP/LDP
    const size_t gprs_size = ((num_gprs + 1) / 2) * 16;
    const size_t fprs_size = num_fprs * 16;

    return {
        gprs,
        fprs,
        frame_size,
        gprs_size,
        fprs_size,
    };
}

// Unroll the loop for small register sets for better performance.
// Use restrict-like hints and const refs where possible.
#define DO_IT(TYPE, REG_TYPE, PAIR_OP, SINGLE_OP, OFFSET)                                                                                       \
    if (!frame_info.TYPE##s.empty()) {                                                                                                          \
        const auto& regs = frame_info.TYPE##s;                                                                                                 \
        const size_t n = regs.size();                                                                                                          \
        size_t i = 0;                                                                                                                          \
        for (; i + 1 < n; i += 2) {                                                                                                            \
            code.PAIR_OP(oaknut::REG_TYPE{regs[i]}, oaknut::REG_TYPE{regs[i + 1]}, SP, (OFFSET) + i * TYPE##_size);                            \
        }                                                                                                                                      \
        if (i < n) {                                                                                                                           \
            code.SINGLE_OP(oaknut::REG_TYPE{regs[i]}, SP, (OFFSET) + i * TYPE##_size);                                                         \
        }                                                                                                                                      \
    }

void ABI_PushRegisters(oaknut::CodeGenerator& code, RegisterList rl, size_t frame_size) {
    const FrameInfo frame_info = CalculateFrameInfo(rl, frame_size);

    // Combine SUBs for better scheduling and less codegen
    const size_t total_save_size = frame_info.gprs_size + frame_info.fprs_size + frame_info.frame_size;
    code.SUB(SP, SP, total_save_size);

    // Save GPRs and FPRs
    DO_IT(gpr, XReg, STP, STR, 0)
    DO_IT(fpr, QReg, STP, STR, frame_info.gprs_size)
}

void ABI_PopRegisters(oaknut::CodeGenerator& code, RegisterList rl, size_t frame_size) {
    const FrameInfo frame_info = CalculateFrameInfo(rl, frame_size);

    // Restore FPRs and GPRs
    DO_IT(gpr, XReg, LDP, LDR, 0)
    DO_IT(fpr, QReg, LDP, LDR, frame_info.gprs_size)

    // Combine ADDs for better scheduling and less codegen
    const size_t total_restore_size = frame_info.gprs_size + frame_info.fprs_size + frame_info.frame_size;
    code.ADD(SP, SP, total_restore_size);
}

}  // namespace Dynarmic::Backend::Arm64
