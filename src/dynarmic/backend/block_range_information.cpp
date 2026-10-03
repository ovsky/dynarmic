/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
    block_ranges.add(std::make_pair(range, std::set<IR::LocationDescriptor>{location}));
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    block_ranges.clear();
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    for (auto invalidate_interval : ranges) {
        auto pair = block_ranges.equal_range(invalidate_interval);

        // A block that overlaps the invalidated range is evicted as a whole, so
        // its entry has to be removed in full rather than trimmed to the part
        // outside the range. Trimming would leave the block's stale extent
        // behind, and because AddRange merges overlapping intervals, that
        // extent would then be unioned back in the next time the block is
        // emitted and would grow a little further on every recompile.
        //
        // The entries are collected before any is erased because erasing from
        // the map invalidates the iterators equal_range handed us.
        std::vector<decltype(block_ranges)::iterator> entries;
        entries.reserve(static_cast<size_t>(std::distance(pair.first, pair.second)));
        for (auto it = pair.first; it != pair.second; ++it) {
            entries.push_back(it);
            erase_locations.insert(it->second.begin(), it->second.end());
        }

        for (auto it : entries) {
            block_ranges.erase(it);
        }
    }
    return erase_locations;
}

template class BlockRangeInformation<u32>;
template class BlockRangeInformation<u64>;

}  // namespace Dynarmic::Backend
