/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/fpsr_manager.h"

#include <oaknut/oaknut.hpp>
#include <utility> // for std::exchange

#include "dynarmic/backend/arm64/abi.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

FpsrManager::FpsrManager(oaknut::CodeGenerator& code, size_t state_fpsr_offset) noexcept
    : code{code}, state_fpsr_offset{state_fpsr_offset}, fpsr_loaded{false} {}

void FpsrManager::Spill() {
    if (!std::exchange(fpsr_loaded, false))
        return;

    // Load the saved FPSR value from memory
    code.LDR(Wscratch0, Xstate, state_fpsr_offset);
    // Read the current FPSR register
    code.MRS(Wscratch1, oaknut::SystemReg::FPSR);
    // Merge the two (bitwise OR)
    code.ORR(Wscratch0, Wscratch0, Wscratch1);
    // Store the result back to memory
    code.STR(Wscratch0, Xstate, state_fpsr_offset);
}

void FpsrManager::Load() {
    if (std::exchange(fpsr_loaded, true))
        return;

    // Clear FPSR by writing zero
    code.MSR(oaknut::SystemReg::FPSR, XZR);
}

void FpsrManager::GetFpsr(oaknut::WReg dest) {
    // Load the saved FPSR value from memory
    code.LDR(dest, Xstate, state_fpsr_offset);

    if (fpsr_loaded) {
        // Read the current FPSR register
        code.MRS(Wscratch1, oaknut::SystemReg::FPSR);
        // Merge with the loaded value
        code.ORR(dest, dest, Wscratch1);
    }
}

}  // namespace Dynarmic::Backend::Arm64
