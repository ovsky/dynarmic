/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <utility>
#include <type_traits>
#include <boost/icl/interval_set.hpp>

#include "dynarmic/backend/arm64/address_space.h"
#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/interface/A64/config.h"

namespace Dynarmic::Backend::Arm64 {

struct EmittedBlockInfo;

class A64AddressSpace final : public AddressSpace {
public:
    explicit A64AddressSpace(const A64::UserConfig& conf);

    [[nodiscard]]
    IR::Block GenerateIR(IR::LocationDescriptor) const override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);

protected:
    friend class A64Core;

    void EmitPrelude();
    [[nodiscard]]
    EmitConfig GetEmitConfig() override;
    void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) override;

    // Use const where possible, prefer initialization list, and mark as noexcept if possible
    const A64::UserConfig conf;
    BlockRangeInformation<u64> block_ranges;
};

// Implementation

inline A64AddressSpace::A64AddressSpace(const A64::UserConfig& conf_)
    : AddressSpace{}, conf(conf_), block_ranges{} {}

inline IR::Block A64AddressSpace::GenerateIR(IR::LocationDescriptor descriptor) const {
    // Use modern C++ idioms, e.g., auto, noexcept, [[nodiscard]]
    return AddressSpace::GenerateIR(std::move(descriptor));
}

inline void A64AddressSpace::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    // Use efficient iteration and avoid unnecessary copies
    for (const auto& interval : ranges) {
        block_ranges.InvalidateRange(interval.lower(), interval.upper());
    }
}

inline void A64AddressSpace::EmitPrelude() {
    // If base class has a prelude, call it
    AddressSpace::EmitPrelude();
    // Add any Arm64-specific prelude code here if needed
}

inline EmitConfig A64AddressSpace::GetEmitConfig() {
    // Use modern return value optimization
    return AddressSpace::GetEmitConfig();
}

inline void A64AddressSpace::RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) {
    // Use perfect forwarding if possible, and avoid unnecessary copies
    block_ranges.RegisterBlock(block, block_info);
}

}  // namespace Dynarmic::Backend::Arm64
