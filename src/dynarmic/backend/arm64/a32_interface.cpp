/* This file is part of the dynarmic project.
 * Copyright (c) 2021 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <boost/icl/interval_set.hpp>
#include <mcl/assert.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/arm64/a32_address_space.h"
#include "dynarmic/backend/arm64/a32_core.h"
#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/interface/A32/a32.h"

namespace Dynarmic::A32 {

using namespace Backend::Arm64;

class Jit::Impl final {
public:
    Impl(Jit* jit_interface, A32::UserConfig conf)
        : jit_interface(jit_interface)
        , conf(std::move(conf))
        , current_address_space(this->conf)
        , core(this->conf)
        , halt_reason(0)
        , invalidate_entire_cache(false)
    {}

    HaltReason Run() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(halt_reason.load(std::memory_order_acquire)));

        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
        };

        HaltReason hr = core.Run(current_address_space, current_state, &halt_reason);

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    HaltReason Step() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(halt_reason.load(std::memory_order_acquire)));

        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
        };

        HaltReason hr = core.Step(current_address_space, current_state, &halt_reason);

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    void ClearCache() {
        {
            std::unique_lock lock(invalidation_mutex);
            invalidate_entire_cache = true;
        }
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
        {
            std::unique_lock lock(invalidation_mutex);
            invalid_cache_ranges.add(boost::icl::discrete_interval<u32>::closed(
                start_address, static_cast<u32>(start_address + length - 1)));
        }
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        current_state = {};
    }

    void HaltExecution(HaltReason hr) {
        halt_reason.fetch_or(static_cast<u32>(hr), std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void ClearHalt(HaltReason hr) {
        halt_reason.fetch_and(~static_cast<u32>(hr), std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    std::array<std::uint32_t, 16>& Regs() noexcept {
        return current_state.regs;
    }

    const std::array<std::uint32_t, 16>& Regs() const noexcept {
        return current_state.regs;
    }

    std::array<std::uint32_t, 64>& ExtRegs() noexcept {
        return current_state.ext_regs;
    }

    const std::array<std::uint32_t, 64>& ExtRegs() const noexcept {
        return current_state.ext_regs;
    }

    std::uint32_t Cpsr() const noexcept {
        return current_state.Cpsr();
    }

    void SetCpsr(std::uint32_t value) noexcept {
        current_state.SetCpsr(value);
    }

    std::uint32_t Fpscr() const noexcept {
        return current_state.Fpscr();
    }

    void SetFpscr(std::uint32_t value) noexcept {
        current_state.SetFpscr(value);
    }

    void ClearExclusiveState() noexcept {
        current_state.exclusive_state = false;
    }

    void DumpDisassembly() const {
        ASSERT_FALSE("Unimplemented");
    }

private:
    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock(invalidation_mutex);

            ClearHalt(HaltReason::CacheInvalidation);

            if (invalidate_entire_cache) {
                current_address_space.ClearCache();
                invalidate_entire_cache = false;
                invalid_cache_ranges.clear();
                return;
            }

            if (!invalid_cache_ranges.empty()) {
                current_address_space.InvalidateCacheRanges(invalid_cache_ranges);
                invalid_cache_ranges.clear();
                return;
            }
        }
    }

    Jit* jit_interface;
    const A32::UserConfig conf;
    A32JitState current_state{};
    A32AddressSpace current_address_space;
    A32Core core;

    std::atomic<u32> halt_reason;

    mutable std::mutex invalidation_mutex;
    boost::icl::interval_set<u32> invalid_cache_ranges;
    bool invalidate_entire_cache;
};

Jit::Jit(UserConfig conf)
    : impl(std::make_unique<Impl>(this, std::move(conf))) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

void Jit::InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

std::array<std::uint32_t, 16>& Jit::Regs() {
    return impl->Regs();
}

const std::array<std::uint32_t, 16>& Jit::Regs() const {
    return impl->Regs();
}

std::array<std::uint32_t, 64>& Jit::ExtRegs() {
    return impl->ExtRegs();
}

const std::array<std::uint32_t, 64>& Jit::ExtRegs() const {
    return impl->ExtRegs();
}

std::uint32_t Jit::Cpsr() const {
    return impl->Cpsr();
}

void Jit::SetCpsr(std::uint32_t value) {
    impl->SetCpsr(value);
}

std::uint32_t Jit::Fpscr() const {
    return impl->Fpscr();
}

void Jit::SetFpscr(std::uint32_t value) {
    impl->SetFpscr(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

void Jit::DumpDisassembly() const {
    impl->DumpDisassembly();
}

}  // namespace Dynarmic::A32
