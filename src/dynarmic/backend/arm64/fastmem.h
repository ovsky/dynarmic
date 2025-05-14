/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>
#include <type_traits>

#include <mcl/hash/xmrx.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

#include "dynarmic/backend/exception_handler.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

// Use [[nodiscard]] to encourage correct usage.
using DoNotFastmemMarker = std::tuple<IR::LocationDescriptor, unsigned>;

struct DoNotFastmemMarkerHash {
    size_t operator()(const DoNotFastmemMarker& value) const noexcept {
        // Use std::get for tuple access, combine values for better distribution.
        // Use std::hash for unsigned for better genericity.
        const auto loc_val = std::get<0>(value).Value();
        const auto idx_val = static_cast<u64>(std::get<1>(value));
        // Mix the two values for better hash distribution.
        return mcl::hash::xmrx(loc_val ^ (idx_val + 0x9e3779b97f4a7c15ull + (loc_val << 6) + (loc_val >> 2)));
    }
};

struct FastmemPatchInfo {
    DoNotFastmemMarker marker;
    FakeCall fc;
    bool recompile;
    // Consider adding constructors for efficiency if needed.
};

class FastmemManager {
public:
    explicit FastmemManager(ExceptionHandler& eh) noexcept
        : exception_handler(eh) {}

    [[nodiscard]]
    bool SupportsFastmem() const noexcept {
        return exception_handler.SupportsFastmem();
    }

    [[nodiscard]]
    bool ShouldFastmem(const DoNotFastmemMarker& marker) const noexcept {
        // Use find instead of count for better performance (avoids double hashing).
        return do_not_fastmem.find(marker) == do_not_fastmem.end();
    }

    void MarkDoNotFastmem(DoNotFastmemMarker marker) {
        // Use emplace for in-place construction, avoids unnecessary copies.
        do_not_fastmem.emplace(std::move(marker));
    }

    // Optionally, provide a way to clear or query the set for advanced use cases.
    // void ClearDoNotFastmem() { do_not_fastmem.clear(); }
    // size_t DoNotFastmemCount() const noexcept { return do_not_fastmem.size(); }

private:
    ExceptionHandler& exception_handler;
    tsl::robin_set<DoNotFastmemMarker, DoNotFastmemMarkerHash> do_not_fastmem;
    // Optionally, reserve space if the expected number of markers is known for better performance.
    // static constexpr size_t kExpectedMarkers = 128;
    // FastmemManager(ExceptionHandler& eh) : exception_handler(eh), do_not_fastmem(kExpectedMarkers) {}
};

}  // namespace Dynarmic::Backend::Arm64
