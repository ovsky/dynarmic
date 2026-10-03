/* This file is part of the dynarmic project.
 * Copyright (c) 2024 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <set>
#include <utility>
#include <vector>

#include <boost/icl/discrete_interval.hpp>
#include <boost/icl/interval_set.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/ir/location_descriptor.h"

using namespace Dynarmic;

namespace {

IR::LocationDescriptor Descriptor(u64 value) {
    return IR::LocationDescriptor{value};
}

template<typename ProgramCounterType, typename BoundType>
tsl::robin_set<IR::LocationDescriptor> InvalidateRange(Backend::BlockRangeInformation<ProgramCounterType>& ranges, BoundType low, BoundType high) {
    boost::icl::interval_set<ProgramCounterType> to_invalidate;
    to_invalidate.add(boost::icl::discrete_interval<ProgramCounterType>::closed(static_cast<ProgramCounterType>(low), static_cast<ProgramCounterType>(high)));
    return ranges.InvalidateRanges(to_invalidate);
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> InvalidateDisjointRanges(Backend::BlockRangeInformation<ProgramCounterType>& ranges, const std::vector<std::pair<ProgramCounterType, ProgramCounterType>>& bounds) {
    boost::icl::interval_set<ProgramCounterType> to_invalidate;
    for (const auto& [low, high] : bounds) {
        to_invalidate.add(boost::icl::discrete_interval<ProgramCounterType>::closed(low, high));
    }
    return ranges.InvalidateRanges(to_invalidate);
}

bool Contains(const tsl::robin_set<IR::LocationDescriptor>& set, u64 value) {
    return set.count(Descriptor(value)) != 0;
}

}  // namespace

TEST_CASE("BlockRangeInformation: invalidating a range reports the blocks it covers", "[backend]") {
    Backend::BlockRangeInformation<u64> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1003), Descriptor(1));
    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x2000, 0x2003), Descriptor(2));

    const auto invalidated = InvalidateRange(ranges, 0x1000u, 0x1003u);
    REQUIRE(invalidated.size() == 1);
    CHECK(Contains(invalidated, 1));
    CHECK_FALSE(Contains(invalidated, 2));
}

TEST_CASE("BlockRangeInformation: invalidating a range that covers nothing reports nothing", "[backend]") {
    Backend::BlockRangeInformation<u64> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1003), Descriptor(1));

    CHECK(InvalidateRange(ranges, 0x3000u, 0x3003u).empty());
}

TEST_CASE("BlockRangeInformation: a partially overlapping range still evicts the whole block", "[backend]") {
    Backend::BlockRangeInformation<u64> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x100F), Descriptor(1));

    const auto invalidated = InvalidateRange(ranges, 0x1008u, 0x1009u);
    REQUIRE(invalidated.size() == 1);
    CHECK(Contains(invalidated, 1));
}

TEST_CASE("BlockRangeInformation: invalidated ranges are dropped rather than kept", "[backend]") {
    /* The point of this test: reporting a block as invalidated is only half
     * the job. If the range stays in the map, a later invalidation of the same
     * addresses reports the block a second time even though it was never
     * emitted again, and the map grows for the lifetime of the process. */
    Backend::BlockRangeInformation<u64> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1003), Descriptor(1));

    REQUIRE(InvalidateRange(ranges, 0x1000u, 0x1003u).size() == 1);

    // Re-emitting the same block and invalidating it again must report it
    // again, so the second invalidation only finds it because AddRange put it
    // back.
    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1003), Descriptor(1));
    CHECK(InvalidateRange(ranges, 0x1000u, 0x1003u).size() == 1);

    // Nothing has been emitted since, so there is nothing left to report.
    CHECK(InvalidateRange(ranges, 0x1000u, 0x1003u).empty());
}

TEST_CASE("BlockRangeInformation: a re-emitted block does not inherit the extent of the block it replaced", "[backend]") {
    /* AddRange merges overlapping intervals, so an extent left over from an
     * invalidated block would be unioned onto the extent of the block that
     * replaces it. Repeated recompiles would then widen the recorded range a
     * little further every time, eventually invalidating blocks that are not
     * remotely concerned. */
    Backend::BlockRangeInformation<u64> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1FFF), Descriptor(1));
    REQUIRE(InvalidateRange(ranges, 0x1000u, 0x1FFFu).size() == 1);

    // The guest code changed, so the replacement block is much shorter.
    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1003), Descriptor(1));

    // Only the replacement block's own extent may be reported.
    CHECK(InvalidateRange(ranges, 0x1004u, 0x1005u).empty());
    CHECK(InvalidateRange(ranges, 0x1000u, 0x1003u).size() == 1);
}

TEST_CASE("BlockRangeInformation: several disjoint invalidation ranges are all applied", "[backend]") {
    Backend::BlockRangeInformation<u64> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1003), Descriptor(1));
    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x2000, 0x2003), Descriptor(2));
    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x3000, 0x3003), Descriptor(3));

    const std::vector<std::pair<u64, u64>> disjoint{{0x1000, 0x1003}, {0x3000, 0x3003}};
    const auto invalidated = InvalidateDisjointRanges(ranges, disjoint);
    REQUIRE(invalidated.size() == 2);
    CHECK(Contains(invalidated, 1));
    CHECK(Contains(invalidated, 3));
    CHECK_FALSE(Contains(invalidated, 2));

    // Block 2 survives, and only block 2.
    CHECK(InvalidateRange(ranges, 0x2000u, 0x2003u).size() == 1);
    CHECK(InvalidateRange(ranges, 0x1000u, 0x1003u).empty());
}

TEST_CASE("BlockRangeInformation: blocks sharing an address are all reported", "[backend]") {
    /* Two blocks can cover the same addresses when they differ in context, for
     * example in Thumb versus ARM state. Both have to be reported. */
    Backend::BlockRangeInformation<u32> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u32>::closed(0x1000, 0x1003), Descriptor(1));
    ranges.AddRange(boost::icl::discrete_interval<u32>::closed(0x1000, 0x1003), Descriptor(2));

    const auto invalidated = InvalidateRange(ranges, 0x1000u, 0x1003u);
    REQUIRE(invalidated.size() == 2);
    CHECK(Contains(invalidated, 1));
    CHECK(Contains(invalidated, 2));

    CHECK(InvalidateRange(ranges, 0x1000u, 0x1003u).empty());
}

TEST_CASE("BlockRangeInformation: ClearCache forgets every range", "[backend]") {
    Backend::BlockRangeInformation<u64> ranges;

    ranges.AddRange(boost::icl::discrete_interval<u64>::closed(0x1000, 0x1003), Descriptor(1));
    ranges.ClearCache();

    CHECK(InvalidateRange(ranges, 0x1000u, 0x1003u).empty());
}