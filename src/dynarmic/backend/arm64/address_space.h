/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <map>
#include <optional>
#include <memory>
#include <shared_mutex>
#include <utility>

#include <mcl/stdint.hpp>
#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/fastmem.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

// Use alias for improved readability and flexibility
using LocationDescriptor = IR::LocationDescriptor;
using BlockSet = tsl::robin_set<LocationDescriptor>;
using BlockMap = tsl::robin_map<LocationDescriptor, CodePtr>;
using ReverseBlockMap = std::map<CodePtr, LocationDescriptor>;
using BlockInfoMap = tsl::robin_map<CodePtr, EmittedBlockInfo>;
using BlockRefMap = tsl::robin_map<LocationDescriptor, tsl::robin_set<CodePtr>>;

class AddressSpace {
public:
    explicit AddressSpace(size_t code_cache_size);
    virtual ~AddressSpace();

    // Generate IR for a given location descriptor
    virtual IR::Block GenerateIR(LocationDescriptor) const = 0;

    // Get the code pointer for a location descriptor, emitting if necessary
    CodePtr Get(LocationDescriptor descriptor);

    // Returns "most likely" LocationDescriptor associated with the emitted code at that location
    std::optional<LocationDescriptor> ReverseGetLocation(CodePtr host_pc);

    // Returns "most likely" entry_point associated with the emitted code at that location
    CodePtr ReverseGetEntryPoint(CodePtr host_pc);

    // Get or emit code for a location descriptor
    CodePtr GetOrEmit(LocationDescriptor descriptor);

    // Invalidate a set of basic blocks
    void InvalidateBasicBlocks(const BlockSet& descriptors);

    // Clear the code cache
    void ClearCache();

    // Dump disassembly for debugging
    void DumpDisassembly() const;

protected:
    // Emit configuration for code generation
    virtual EmitConfig GetEmitConfig() = 0;
    // Register a new basic block
    virtual void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) = 0;

    // Protect code memory (for platforms with NX support)
    void ProtectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.protect();
#endif
    }

    // Unprotect code memory (for platforms with NX support)
    void UnprotectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.unprotect();
#endif
    }

    // Get the remaining size in the code cache
    size_t GetRemainingSize();

    // Emit a block of code from IR
    EmittedBlockInfo Emit(IR::Block ir_block);

    // Link a block after emission
    void Link(EmittedBlockInfo& block);

    // Link block links for relocation
    void LinkBlockLinks(CodePtr entry_point, CodePtr target_ptr, const std::vector<BlockRelocation>& block_relocations_list);

    // Relink for a given descriptor
    void RelinkForDescriptor(LocationDescriptor target_descriptor, CodePtr target_ptr);

    // Fastmem callback
    FakeCall FastmemCallback(u64 host_pc);

    // --- Data members ---

    const size_t code_cache_size;
    oaknut::CodeBlock mem;
    oaknut::CodeGenerator code;

    // Use mutable and shared_mutex for thread-safe access to block_entries and related maps
    mutable std::shared_mutex block_mutex;

    // A LocationDescriptor will have one current CodePtr.
    // There can be multiple other CodePtrs which are older, previously invalidated blocks.
    BlockMap block_entries;
    ReverseBlockMap reverse_block_entries;
    BlockInfoMap block_infos;
    BlockRefMap block_references;

    ExceptionHandler exception_handler;
    FastmemManager fastmem_manager;

    struct PreludeInfo {
        std::ptrdiff_t end_of_prelude;

        using RunCodeFuncType = HaltReason (*)(CodePtr entry_point, void* jit_state, volatile u32* halt_reason);
        RunCodeFuncType run_code;
        RunCodeFuncType step_code;
        void* return_to_dispatcher;
        void* return_from_run_code;

        void* read_memory_8;
        void* read_memory_16;
        void* read_memory_32;
        void* read_memory_64;
        void* read_memory_128;
        void* wrapped_read_memory_8;
        void* wrapped_read_memory_16;
        void* wrapped_read_memory_32;
        void* wrapped_read_memory_64;
        void* wrapped_read_memory_128;
        void* exclusive_read_memory_8;
        void* exclusive_read_memory_16;
        void* exclusive_read_memory_32;
        void* exclusive_read_memory_64;
        void* exclusive_read_memory_128;
        void* write_memory_8;
        void* write_memory_16;
        void* write_memory_32;
        void* write_memory_64;
        void* write_memory_128;
        void* wrapped_write_memory_8;
        void* wrapped_write_memory_16;
        void* wrapped_write_memory_32;
        void* wrapped_write_memory_64;
        void* wrapped_write_memory_128;
        void* exclusive_write_memory_8;
        void* exclusive_write_memory_16;
        void* exclusive_write_memory_32;
        void* exclusive_write_memory_64;
        void* exclusive_write_memory_128;

        void* call_svc;
        void* exception_raised;
        void* dc_raised;
        void* ic_raised;
        void* isb_raised;

        void* get_cntpct;
        void* add_ticks;
        void* get_ticks_remaining;
    } prelude_info;

    // --- Helper methods for thread safety and performance ---

    // Thread-safe read access to block_entries
    std::optional<CodePtr> FindBlockEntry(const LocationDescriptor& descriptor) const {
        std::shared_lock lock(block_mutex);
        auto it = block_entries.find(descriptor);
        if (it != block_entries.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Thread-safe write access to block_entries
    void SetBlockEntry(const LocationDescriptor& descriptor, CodePtr code_ptr) {
        std::unique_lock lock(block_mutex);
        block_entries[descriptor] = code_ptr;
    }

    // Thread-safe erase from block_entries
    void EraseBlockEntry(const LocationDescriptor& descriptor) {
        std::unique_lock lock(block_mutex);
        block_entries.erase(descriptor);
    }

    // Thread-safe clear of all block maps
    void ClearAllBlockMaps() {
        std::unique_lock lock(block_mutex);
        block_entries.clear();
        reverse_block_entries.clear();
        block_infos.clear();
        block_references.clear();
    }
};

}  // namespace Dynarmic::Backend::Arm64
