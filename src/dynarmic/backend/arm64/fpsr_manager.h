/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <utility>
#include <mcl/stdint.hpp>

namespace oaknut {
struct CodeGenerator;
struct WReg;
}  // namespace oaknut

namespace Dynarmic::Backend::Arm64 {

class FpsrManager {
public:
    // Use [[nodiscard]] to encourage correct usage.
    [[nodiscard]]
    explicit FpsrManager(oaknut::CodeGenerator& code, size_t state_fpsr_offset) noexcept
        : code(code), state_fpsr_offset(state_fpsr_offset) {}

    // Mark as noexcept for performance and clarity.
    void Spill() noexcept;
    void Load() noexcept;
    void Overwrite() noexcept { fpsr_loaded = false; }

    // Pass register by const reference to avoid unnecessary copies.
    void GetFpsr(const oaknut::WReg& reg) noexcept;

    // Delete copy operations, allow move if needed.
    FpsrManager(const FpsrManager&) = delete;
    FpsrManager& operator=(const FpsrManager&) = delete;
    FpsrManager(FpsrManager&&) = default;
    FpsrManager& operator=(FpsrManager&&) = default;

private:
    oaknut::CodeGenerator& code;
    size_t state_fpsr_offset;
    bool fpsr_loaded = false;
};

}  // namespace Dynarmic::Backend::Arm64
