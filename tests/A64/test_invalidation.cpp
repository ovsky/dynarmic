/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <catch2/catch_test_macros.hpp>

#include "./testenv.h"
#include "dynarmic/interface/A64/a64.h"

using namespace Dynarmic;

TEST_CASE("ensure fast dispatch entry is cleared even when a block does not have any patching requirements", "[a64]") {
    A64TestEnv env;

    A64::UserConfig conf{&env};
    A64::Jit jit{conf};

    REQUIRE(conf.HasOptimization(OptimizationFlag::FastDispatch));

    env.code_mem_start_address = 100;
    env.code_mem.clear();
    env.code_mem.emplace_back(0xd2800d80);  // MOV X0, 108
    env.code_mem.emplace_back(0xd61f0000);  // BR X0
    env.code_mem.emplace_back(0xd2800540);  // MOV X0, 42
    env.code_mem.emplace_back(0x14000000);  // B .

    jit.SetPC(100);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    jit.SetPC(100);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    jit.InvalidateCacheRange(108, 4);

    jit.SetPC(100);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    env.code_mem[2] = 0xd28008a0;  // MOV X0, 69

    jit.SetPC(100);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    jit.InvalidateCacheRange(108, 4);

    jit.SetPC(100);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 69);

    jit.SetPC(100);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 69);
}

TEST_CASE("ensure fast dispatch entry is cleared even when a block does not have any patching requirements 2", "[a64]") {
    A64TestEnv env;

    A64::UserConfig conf{&env};
    A64::Jit jit{conf};

    REQUIRE(conf.HasOptimization(OptimizationFlag::FastDispatch));

    env.code_mem.emplace_back(0xd2800100);  // MOV X0, 8
    env.code_mem.emplace_back(0xd61f0000);  // BR X0
    env.code_mem.emplace_back(0xd2800540);  // MOV X0, 42
    env.code_mem.emplace_back(0x14000000);  // B .

    jit.SetPC(0);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    jit.SetPC(0);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    jit.InvalidateCacheRange(8, 4);

    jit.SetPC(0);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    env.code_mem[2] = 0xd28008a0;  // MOV X0, 69

    jit.SetPC(0);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 42);

    jit.InvalidateCacheRange(8, 4);

    jit.SetPC(0);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 69);

    jit.SetPC(0);
    env.ticks_left = 4;
    jit.Run();
    REQUIRE(jit.GetRegister(0) == 69);
}

TEST_CASE("a zero length invalidation is a no-op", "[a64]") {
    A64TestEnv env;

    A64::UserConfig conf{&env};
    A64::Jit jit{conf};

    env.code_mem.emplace_back(0xd2800540);  // MOV X0, 42
    env.code_mem.emplace_back(0x14000000);  // B .

    jit.SetPC(0);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());

    // Invalidate the block that was just compiled, then change the code it
    // came from. A no-op invalidation must leave the compiled block in place,
    // so the old instruction is still what runs.
    jit.InvalidateCacheRange(0, 0);

    env.code_mem[0] = 0xd28008a0;  // MOV X0, 69

    jit.SetPC(0);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());
    REQUIRE(jit.GetRegister(0) == 42);

    // The block is still live, so a real invalidation takes effect.
    jit.InvalidateCacheRange(0, 4);

    jit.SetPC(0);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());
    REQUIRE(jit.GetRegister(0) == 69);
}

TEST_CASE("invalidation of code at the top of the address space is not dropped", "[a64]") {
    /* A range running off the end of the address space used to wrap around and
     * describe a short range just above address zero, which boost::icl then
     * discarded entirely. The invalidation was silently lost and the stale
     * block kept running. */
    constexpr u64 code_address = 0xFFFFFFFFFFFFF000;

    A64TestEnv env;
    env.code_mem_start_address = code_address;
    env.code_mem.emplace_back(0xd2800540);  // MOV X0, 42
    env.code_mem.emplace_back(0x14000000);  // B .

    A64::UserConfig conf{&env};
    A64::Jit jit{conf};

    jit.SetPC(code_address);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());
    REQUIRE(jit.GetRegister(0) == 42);

    env.code_mem[0] = 0xd28008a0;  // MOV X0, 69

    // Without this invalidation the stale block is expected to still run.
    jit.SetPC(code_address);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());
    REQUIRE(jit.GetRegister(0) == 42);

    // A range that starts on the block but runs off the end of the address
    // space: 0xFFFFFFFFFFFFF000 + 0x2000 - 1 wraps to 0x1FFF when computed in
    // the address space width, which describes a range nowhere near the block
    // and is discarded by boost::icl. Saturating at the top of the address
    // space has to invalidate the block instead.
    jit.InvalidateCacheRange(code_address, 0x2000);

    jit.SetPC(code_address);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());
    REQUIRE(jit.GetRegister(0) == 69);
}

TEST_CASE("invalidation at the top of the address space that does not wrap still works", "[a64]") {
    /* Guards the fix against over-reaching: a range that fits inside the
     * address space must still be treated as exactly that range. */
    constexpr u64 code_address = 0xFFFFFFFFFFFFF000;

    A64TestEnv env;
    env.code_mem_start_address = code_address;
    env.code_mem.emplace_back(0xd2800540);  // MOV X0, 42
    env.code_mem.emplace_back(0x14000000);  // B .

    A64::UserConfig conf{&env};
    A64::Jit jit{conf};

    jit.SetPC(code_address);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());
    REQUIRE(jit.GetRegister(0) == 42);

    env.code_mem[0] = 0xd28008a0;  // MOV X0, 69

    jit.InvalidateCacheRange(code_address, 4);

    jit.SetPC(code_address);
    env.ticks_left = 2;
    REQUIRE(!jit.Run());
    REQUIRE(jit.GetRegister(0) == 69);
}
