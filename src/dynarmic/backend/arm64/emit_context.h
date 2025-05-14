/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>
#include <utility>
#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/reg_alloc.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/basic_block.h"

namespace Dynarmic::IR {
class Block;
}  // namespace Dynarmic::IR

namespace Dynarmic::Backend::Arm64 {

struct EmitConfig;
class FastmemManager;
class FpsrManager;

// Use alias for shared label, prefer make_shared for efficiency
using SharedLabel = std::shared_ptr<oaknut::Label>;

// Use [[nodiscard]] to encourage correct usage
[[nodiscard]]
inline SharedLabel GenSharedLabel() {
    // Use std::make_shared for exception safety and performance
    return std::make_shared<oaknut::Label>();
}

// Use [[nodiscard]] for functions returning values
struct EmitContext {
    IR::Block& block;
    RegAlloc& reg_alloc;
    const EmitConfig& conf;
    EmittedBlockInfo& ebi;
    FpsrManager& fpsr;
    FastmemManager& fastmem;

    // Use reserve to avoid reallocations if possible
    std::vector<std::function<void()>> deferred_emits;

    EmitContext(IR::Block& block_,
                RegAlloc& reg_alloc_,
                const EmitConfig& conf_,
                EmittedBlockInfo& ebi_,
                FpsrManager& fpsr_,
                FastmemManager& fastmem_)
        : block(block_)
        , reg_alloc(reg_alloc_)
        , conf(conf_)
        , ebi(ebi_)
        , fpsr(fpsr_)
        , fastmem(fastmem_)
    {
        // Optionally reserve space if a typical size is known
        // deferred_emits.reserve(8);
    }

    // Mark as [[nodiscard]] to prevent accidental discards
    [[nodiscard]]
    FP::FPCR FPCR(bool fpcr_controlled = true) const noexcept {
        // Use auto for type deduction and clarity
        const auto fpcr = conf.descriptor_to_fpcr(block.Location());
        return fpcr_controlled ? fpcr : fpcr.ASIMDStandardValue();
    }

    // Optionally, provide move-only lambda support for deferred_emits
    template <typename F>
    void AddDeferredEmit(F&& f) {
        deferred_emits.emplace_back(std::forward<F>(f));
    }
};

}  // namespace Dynarmic::Backend::Arm64
