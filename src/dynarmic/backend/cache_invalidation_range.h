/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include <boost/icl/discrete_interval.hpp>

namespace Dynarmic::Backend {

/**
 * Converts the (start_address, length) pair accepted by the public
 * `Jit::InvalidateCacheRange` entry points into the closed interval of guest
 * addresses that the pair denotes.
 *
 * The end address is computed in a type wide enough to hold the whole sum and
 * is then saturated at the largest representable program counter. Computing it
 * in `ProgramCounterType` instead would silently wrap: a range that runs off
 * the top of the address space would end up describing a short range just
 * above address zero, and a zero length would end up describing the entire
 * address space.
 *
 * @param start_address The first address of the range.
 * @param length The length of the range in bytes.
 * @returns The interval covering [start_address, start_address + length), or
 *          `std::nullopt` when the range is empty, which is exactly when
 *          `length` is zero. An empty range must be treated as a no-op by
 *          callers rather than invalidating a single address.
 */
template<typename ProgramCounterType>
[[nodiscard]] std::optional<boost::icl::discrete_interval<ProgramCounterType>> MakeInvalidationRange(ProgramCounterType start_address, std::size_t length) {
    if (length == 0) {
        return std::nullopt;
    }

    // `length` may be wider than ProgramCounterType (a 32-bit guest with a
    // 64-bit host), and the sum of the two may overflow both, so widen to
    // uintmax_t and clamp explicitly instead of relying on wraparound.
    constexpr auto domain_max = static_cast<std::uintmax_t>(std::numeric_limits<ProgramCounterType>::max());

    const auto start = static_cast<std::uintmax_t>(start_address);
    const auto span = static_cast<std::uintmax_t>(length) - 1;
    const auto end_address = span >= domain_max - start ? domain_max : start + span;

    return boost::icl::discrete_interval<ProgramCounterType>::closed(start_address, static_cast<ProgramCounterType>(end_address));
}

}  // namespace Dynarmic::Backend
