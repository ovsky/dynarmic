/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <utility>
#include <type_traits>
#include "dynarmic/backend/arm64/a64_address_space.h"
#include "dynarmic/backend/arm64/a64_jitstate.h"

namespace Dynarmic::Backend::Arm64 {

class A64Core final {
public:
    // Use noexcept and [[nodiscard]] for better optimization and safety.
    explicit A64Core(const A64::UserConfig&) noexcept = default;

    [[nodiscard]]
    HaltReason Run(A64AddressSpace& process, A64JitState& thread_ctx, volatile u32* halt_reason) noexcept {
        // Use const auto& to avoid unnecessary copies.
        const auto& location_descriptor = thread_ctx.GetLocationDescriptor();
        // Use [[likely]] to hint the compiler that this path is hot.
        const auto entry_point = process.GetOrEmit(location_descriptor);
        return process.prelude_info.run_code(entry_point, std::addressof(thread_ctx), halt_reason);
    }

    [[nodiscard]]
    HaltReason Step(A64AddressSpace& process, A64JitState& thread_ctx, volatile u32* halt_reason) noexcept {
        // Use direct initialization and avoid extra temporaries.
        const auto location_descriptor = A64::LocationDescriptor{thread_ctx.GetLocationDescriptor()}.SetSingleStepping(true);
        const auto entry_point = process.GetOrEmit(location_descriptor);
        return process.prelude_info.step_code(entry_point, std::addressof(thread_ctx), halt_reason);
    }
};

}  // namespace Dynarmic::Backend::Arm64
