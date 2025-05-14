/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>
#include <optional>
#include <utility>
#include <type_traits>
#include <span>
#include <bit>
#include <concepts>

#include <mcl/stdint.hpp>
#include <tsl/robin_map.h>

#include "dynarmic/backend/arm64/fastmem.h"
#include "dynarmic/interface/A32/coprocessor.h"
#include "dynarmic/interface/optimization_flags.h"
#include "dynarmic/ir/location_descriptor.h"

namespace oaknut {
struct CodeGenerator;
struct Label;
}  // namespace oaknut

namespace Dynarmic::FP {
class FPCR;
}  // namespace Dynarmic::FP

namespace Dynarmic::IR {
class Block;
class Inst;
enum class Cond : u8;
enum class Opcode : u16;
}  // namespace Dynarmic::IR

namespace Dynarmic::Backend::Arm64 {

struct EmitContext;

using CodePtr = std::byte*;

enum class LinkTarget : u8 {
    ReturnToDispatcher,
    ReturnFromRunCode,
    ReadMemory8,
    ReadMemory16,
    ReadMemory32,
    ReadMemory64,
    ReadMemory128,
    WrappedReadMemory8,
    WrappedReadMemory16,
    WrappedReadMemory32,
    WrappedReadMemory64,
    WrappedReadMemory128,
    ExclusiveReadMemory8,
    ExclusiveReadMemory16,
    ExclusiveReadMemory32,
    ExclusiveReadMemory64,
    ExclusiveReadMemory128,
    WriteMemory8,
    WriteMemory16,
    WriteMemory32,
    WriteMemory64,
    WriteMemory128,
    WrappedWriteMemory8,
    WrappedWriteMemory16,
    WrappedWriteMemory32,
    WrappedWriteMemory64,
    WrappedWriteMemory128,
    ExclusiveWriteMemory8,
    ExclusiveWriteMemory16,
    ExclusiveWriteMemory32,
    ExclusiveWriteMemory64,
    ExclusiveWriteMemory128,
    CallSVC,
    ExceptionRaised,
    InstructionSynchronizationBarrierRaised,
    InstructionCacheOperationRaised,
    DataCacheOperationRaised,
    GetCNTPCT,
    AddTicks,
    GetTicksRemaining,
};

struct Relocation {
    std::ptrdiff_t code_offset;
    LinkTarget target;

    constexpr Relocation(std::ptrdiff_t offset, LinkTarget tgt) noexcept
        : code_offset(offset), target(tgt) {}
};

enum class BlockRelocationType : u8 {
    Branch,
    MoveToScratch1,
};

struct BlockRelocation {
    std::ptrdiff_t code_offset;
    BlockRelocationType type;

    constexpr BlockRelocation(std::ptrdiff_t offset, BlockRelocationType t) noexcept
        : code_offset(offset), type(t) {}
};

struct EmittedBlockInfo {
    CodePtr entry_point = nullptr;
    size_t size = 0;
    std::vector<Relocation> relocations;
    tsl::robin_map<IR::LocationDescriptor, std::vector<BlockRelocation>> block_relocations;
    tsl::robin_map<std::ptrdiff_t, FastmemPatchInfo> fastmem_patch_info;

    constexpr EmittedBlockInfo() = default;
    constexpr EmittedBlockInfo(CodePtr ep, size_t sz) : entry_point(ep), size(sz) {}
    EmittedBlockInfo(const EmittedBlockInfo&) = default;
    EmittedBlockInfo& operator=(const EmittedBlockInfo&) = default;
    EmittedBlockInfo(EmittedBlockInfo&&) noexcept = default;
    EmittedBlockInfo& operator=(EmittedBlockInfo&&) noexcept = default;
    ~EmittedBlockInfo() = default;
};

struct EmitConfig {
    OptimizationFlag optimizations = no_optimizations;

    [[nodiscard]]
    constexpr bool HasOptimization(OptimizationFlag f) const noexcept {
        return (f & optimizations) != no_optimizations;
    }

    bool hook_isb = false;

    // System registers
    u64 cntfreq_el0 = 0;
    u32 ctr_el0 = 0;
    u32 dczid_el0 = 0;
    const u64* tpidrro_el0 = nullptr;
    u64* tpidr_el0 = nullptr;

    // Memory
    bool check_halt_on_memory_access = false;

    // Page table
    u64 page_table_pointer = 0;
    size_t page_table_address_space_bits = 0;
    int page_table_pointer_mask_bits = 0;
    bool silently_mirror_page_table = false;
    bool absolute_offset_page_table = false;
    u8 detect_misaligned_access_via_page_table = 0;
    bool only_detect_misalignment_via_page_table_on_page_boundary = false;

    // Fastmem
    std::optional<u64> fastmem_pointer;
    bool recompile_on_fastmem_failure = false;
    size_t fastmem_address_space_bits = 0;
    bool silently_mirror_fastmem = false;

    // Timing
    bool wall_clock_cntpct = false;
    bool enable_cycle_counting = false;

    // Endianness
    bool always_little_endian = false;

    // Frontend specific callbacks
    FP::FPCR (*descriptor_to_fpcr)(const IR::LocationDescriptor& descriptor) = nullptr;
    oaknut::Label (*emit_cond)(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Cond cond) = nullptr;
    void (*emit_condition_failed_terminal)(oaknut::CodeGenerator& code, EmitContext& ctx) = nullptr;
    void (*emit_terminal)(oaknut::CodeGenerator& code, EmitContext& ctx) = nullptr;
    void (*emit_check_memory_abort)(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, oaknut::Label& end) = nullptr;

    // State offsets
    size_t state_nzcv_offset = 0;
    size_t state_fpsr_offset = 0;
    size_t state_exclusive_state_offset = 0;

    // A32 specific
    std::array<std::shared_ptr<A32::Coprocessor>, 16> coprocessors{};

    // Debugging
    bool very_verbose_debugging_output = false;

    constexpr EmitConfig() = default;
    EmitConfig(const EmitConfig&) = default;
    EmitConfig& operator=(const EmitConfig&) = default;
    EmitConfig(EmitConfig&&) noexcept = default;
    EmitConfig& operator=(EmitConfig&&) noexcept = default;
    ~EmitConfig() = default;
};

// Use [[nodiscard]] and [[gnu::always_inline]] for performance hints
[[nodiscard]]
inline EmittedBlockInfo EmitArm64(
    oaknut::CodeGenerator& code,
    IR::Block block,
    const EmitConfig& emit_conf,
    FastmemManager& fastmem_manager
);

template<IR::Opcode op>
    requires std::is_enum_v<IR::Opcode>
[[gnu::always_inline]]
inline void EmitIR(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

[[gnu::always_inline]]
inline void EmitRelocation(oaknut::CodeGenerator& code, EmitContext& ctx, LinkTarget link_target);

[[gnu::always_inline]]
inline void EmitBlockLinkRelocation(oaknut::CodeGenerator& code, EmitContext& ctx, const IR::LocationDescriptor& descriptor, BlockRelocationType type);

[[nodiscard]]
[[gnu::always_inline]]
inline oaknut::Label EmitA32Cond(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Cond cond);

[[nodiscard]]
[[gnu::always_inline]]
inline oaknut::Label EmitA64Cond(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Cond cond);

[[gnu::always_inline]]
inline void EmitA32Terminal(oaknut::CodeGenerator& code, EmitContext& ctx);

[[gnu::always_inline]]
inline void EmitA64Terminal(oaknut::CodeGenerator& code, EmitContext& ctx);

[[gnu::always_inline]]
inline void EmitA32ConditionFailedTerminal(oaknut::CodeGenerator& code, EmitContext& ctx);

[[gnu::always_inline]]
inline void EmitA64ConditionFailedTerminal(oaknut::CodeGenerator& code, EmitContext& ctx);

[[gnu::always_inline]]
inline void EmitA32CheckMemoryAbort(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, oaknut::Label& end);

[[gnu::always_inline]]
inline void EmitA64CheckMemoryAbort(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst, oaknut::Label& end);

}  // namespace Dynarmic::Backend::Arm64
