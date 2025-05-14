/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/a32_jitstate.h"

#include <mcl/bit/bit_field.hpp>
#include <mcl/stdint.hpp>
#include <bit> // For std::rotl, std::rotr if needed (C++20), but not used here

namespace Dynarmic::Backend::Arm64 {

namespace {

constexpr u32 NZCV_MASK = 0xF0000000;
constexpr u32 Q_MASK    = 1u << 27;
constexpr u32 GE_MASKS[4] = { 1u << 19, 1u << 18, 1u << 17, 1u << 16 };
constexpr u32 GE_BITS[4]  = { 31, 23, 15, 7 };
constexpr u32 E_MASK   = 1u << 9;
constexpr u32 T_MASK   = 1u << 5;
constexpr u32 IT_LOW_MASK  = 0b11111100'00000000u;
constexpr u32 IT_HIGH_MASK = 0b00000011'00000000u;
constexpr u32 JAIFM_MASK = 0x010001DF;

constexpr u32 FPCR_MASK = A32::LocationDescriptor::FPSCR_MODE_MASK;
constexpr u32 FPSR_MASK = 0x0800'009f;

[[nodiscard]] constexpr u32 ExtractBits(u32 value, u32 mask, u32 shift = 0) noexcept {
    return (value & mask) >> shift;
}

[[nodiscard]] constexpr u32 SetBitIf(u32 cond, u32 bit) noexcept {
    return cond ? bit : 0;
}

} // anonymous namespace

u32 A32JitState::Cpsr() const {
    u32 cpsr = 0;

    // NZCV flags
    cpsr |= cpsr_nzcv;
    // Q flag
    cpsr |= cpsr_q;

    // GE flags (bits 19,18,17,16)
    for (size_t i = 0; i < 4; ++i) {
        cpsr |= mcl::bit::get_bit<GE_BITS[i]>(cpsr_ge) ? GE_MASKS[i] : 0;
    }

    // E flag (bit 9), T flag (bit 5)
    cpsr |= SetBitIf(mcl::bit::get_bit<1>(upper_location_descriptor), E_MASK);
    cpsr |= SetBitIf(mcl::bit::get_bit<0>(upper_location_descriptor), T_MASK);

    // IT state
    cpsr |= static_cast<u32>(upper_location_descriptor & IT_LOW_MASK);
    cpsr |= static_cast<u32>((upper_location_descriptor & IT_HIGH_MASK) << 17);

    // Other flags
    cpsr |= cpsr_jaifm;

    return cpsr;
}

void A32JitState::SetCpsr(u32 cpsr) {
    // NZCV flags
    cpsr_nzcv = cpsr & NZCV_MASK;
    // Q flag
    cpsr_q = cpsr & Q_MASK;

    // GE flags
    cpsr_ge = 0;
    cpsr_ge |= SetBitIf(mcl::bit::get_bit<19>(cpsr), 0xFF000000);
    cpsr_ge |= SetBitIf(mcl::bit::get_bit<18>(cpsr), 0x00FF0000);
    cpsr_ge |= SetBitIf(mcl::bit::get_bit<17>(cpsr), 0x0000FF00);
    cpsr_ge |= SetBitIf(mcl::bit::get_bit<16>(cpsr), 0x000000FF);

    // E flag, T flag, IT state
    upper_location_descriptor &= 0xFFFF0000;
    upper_location_descriptor |= SetBitIf(mcl::bit::get_bit<9>(cpsr), 2);
    upper_location_descriptor |= SetBitIf(mcl::bit::get_bit<5>(cpsr), 1);
    upper_location_descriptor |= (cpsr & IT_LOW_MASK);
    upper_location_descriptor |= ((cpsr >> 17) & IT_HIGH_MASK);

    // Other flags
    cpsr_jaifm = cpsr & JAIFM_MASK;
}

u32 A32JitState::Fpscr() const {
    // Compose FPSCR from upper_location_descriptor, fpsr, and fpsr_nzcv
    return (upper_location_descriptor & 0xFFFF0000) | (fpsr & FPSR_MASK) | (fpsr_nzcv & 0xF0000000);
}

void A32JitState::SetFpscr(u32 fpscr) {
    fpsr_nzcv = fpscr & 0xF0000000;
    fpsr = fpscr & FPSR_MASK;
    upper_location_descriptor = (upper_location_descriptor & 0x0000FFFF) | (fpscr & FPCR_MASK);
}

}  // namespace Dynarmic::Backend::Arm64
