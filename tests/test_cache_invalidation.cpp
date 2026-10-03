/* This file is part of the dynarmic project.
 * Copyright (c) 2024 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstdint>
#include <cstddef>
#include <limits>

#include <boost/icl/discrete_interval.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/cache_invalidation_range.h"

using namespace Dynarmic;

namespace {

template<typename ProgramCounterType>
boost::icl::discrete_interval<ProgramCounterType> MakeRange(ProgramCounterType start_address, size_t length) {
    const auto range = Backend::MakeInvalidationRange<ProgramCounterType>(start_address, length);
    REQUIRE(range.has_value());
    return *range;
}

constexpr u32 u32_max = std::numeric_limits<u32>::max();
constexpr u64 u64_max = std::numeric_limits<u64>::max();

}  // namespace

TEST_CASE("MakeInvalidationRange: a zero length range is empty", "[backend]") {
    SECTION("A32") {
        CHECK_FALSE(Backend::MakeInvalidationRange<u32>(0, 0).has_value());
        CHECK_FALSE(Backend::MakeInvalidationRange<u32>(0x1000, 0).has_value());
        CHECK_FALSE(Backend::MakeInvalidationRange<u32>(u32_max, 0).has_value());
    }

    SECTION("A64") {
        CHECK_FALSE(Backend::MakeInvalidationRange<u64>(0, 0).has_value());
        CHECK_FALSE(Backend::MakeInvalidationRange<u64>(0x1000, 0).has_value());
        CHECK_FALSE(Backend::MakeInvalidationRange<u64>(u64_max, 0).has_value());
    }
}

TEST_CASE("MakeInvalidationRange: ordinary ranges are closed intervals", "[backend]") {
    SECTION("A32") {
        const auto range = MakeRange<u32>(0x1000, 4);
        CHECK(range.lower() == 0x1000);
        CHECK(range.upper() == 0x1003);
    }

    SECTION("A64") {
        const auto range = MakeRange<u64>(0x1000, 4);
        CHECK(range.lower() == 0x1000);
        CHECK(range.upper() == 0x1003);
    }
}

TEST_CASE("MakeInvalidationRange: single byte ranges cover one address", "[backend]") {
    SECTION("A32") {
        const auto range = MakeRange<u32>(0x1000, 1);
        CHECK(range.lower() == 0x1000);
        CHECK(range.upper() == 0x1000);
    }

    SECTION("A64") {
        const auto range = MakeRange<u64>(0x1000, 1);
        CHECK(range.lower() == 0x1000);
        CHECK(range.upper() == 0x1000);
    }
}

TEST_CASE("MakeInvalidationRange: a range ending exactly at the top of the address space is not clamped away", "[backend]") {
    SECTION("A32") {
        const auto range = MakeRange<u32>(u32_max - 3, 4);
        CHECK(range.lower() == u32_max - 3);
        CHECK(range.upper() == u32_max);
    }

    SECTION("A64") {
        const auto range = MakeRange<u64>(u64_max - 3, 4);
        CHECK(range.lower() == u64_max - 3);
        CHECK(range.upper() == u64_max);
    }
}

TEST_CASE("MakeInvalidationRange: a range running off the top of the address space saturates", "[backend]") {
    /* This is the case that used to wrap: computing start_address + length - 1
     * in the program counter type produces 0x0F here, which describes a short
     * range just above address zero rather than the range the caller asked for.
     * boost::icl silently discards the resulting inverted interval, so the
     * invalidation used to be dropped and stale code kept running. */
    SECTION("A32") {
        const auto range = MakeRange<u32>(0xFFFFFFF0, 0x20);
        CHECK(range.lower() == 0xFFFFFFF0);
        CHECK(range.upper() == u32_max);
    }

    SECTION("A64") {
        const auto range = MakeRange<u64>(0xFFFFFFFFFFFFFFF0, 0x20);
        CHECK(range.lower() == 0xFFFFFFFFFFFFFFF0);
        CHECK(range.upper() == u64_max);
    }
}

TEST_CASE("MakeInvalidationRange: a length wider than the address space saturates", "[backend]") {
    SECTION("A32") {
        const auto range = MakeRange<u32>(0x1000, size_t{1} << 32);
        CHECK(range.lower() == 0x1000);
        CHECK(range.upper() == u32_max);
    }

    SECTION("A64") {
        const auto range = MakeRange<u64>(0x1000, std::numeric_limits<size_t>::max());
        CHECK(range.lower() == 0x1000);
        CHECK(range.upper() == u64_max);
    }
}

TEST_CASE("MakeInvalidationRange: a length of exactly the address space size saturates", "[backend]") {
    /* 2^32 bytes covers every address of the A32 address space exactly, so the
     * saturated end address is the last addressable one rather than one below
     * it. */
    const auto range = MakeRange<u32>(0, size_t{1} << 32);
    CHECK(range.lower() == 0);
    CHECK(range.upper() == u32_max);
}

TEST_CASE("MakeInvalidationRange: the largest representable length does not over-saturate", "[backend]") {
    /* A length one byte short of wrapping must still end one byte short of the
     * top of the address space; saturating here would over-invalidate. */
    SECTION("A32") {
        const auto range = MakeRange<u32>(0, u32_max);
        CHECK(range.lower() == 0);
        CHECK(range.upper() == u32_max - 1);
    }

    SECTION("A64") {
        const auto range = MakeRange<u64>(0, static_cast<size_t>(u64_max));
        CHECK(range.lower() == 0);
        CHECK(range.upper() == u64_max - 1);
    }
}
