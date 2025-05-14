/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <utility>
#include <type_traits>
#include "dynarmic/backend/arm64/a32_address_space.h"
#include "dynarmic/backend/arm64/a32_jitstate.h"

namespace Dynarmic::Backend::Arm64 {

class A32Core final {
public:
    // Use noexcept and [[nodiscard]] for better optimization and safety.
    explicit A32Core(const A32::UserConfig&) noexcept = default;
    A32Core(const A32Core&) = delete;
    A32Core& operator=(const A32Core&) = delete;
    }

    HaltReason Step(A32AddressSpace& process, A32JitState& thread_ctx, volatile u32* halt_reason) {
        const auto location_descriptor = A32::LocationDescriptor{thread_ctx.GetLocationDescriptor()}.SetSingleStepping(true);
        const auto entry_point = process.GetOrEmit(location_descriptor);
        return process.prelude_info.step_code(entry_point, &thread_ctx, halt_reason);
    }
};

}  // namespace Dynarmic::Backend::Arm64
