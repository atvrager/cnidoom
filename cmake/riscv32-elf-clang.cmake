# CMake toolchain file for bare-metal RISC-V 32-bit.
#
# Target: RV32IMF_Zve32x_Zve32f (integer + single-float + vector).
# Uses the Kelvin toolchain (GCC 15 + newlib), auto-fetched at configure time.
#
# The toolchain tarball is downloaded once into the build tree and cached.
# This makes the build fully reproducible on any machine.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv32)

# ── Auto-fetch Kelvin toolchain ──────────────────────────────────────
set(KELVIN_TC_URL
    "https://storage.googleapis.com/shodan-public-artifacts/toolchain_kelvin_tar_files/toolchain_kelvin_v2-2025-09-11.tar.gz")
set(KELVIN_TC_SHA256
    "" CACHE STRING "SHA256 of toolchain tarball (empty = skip verification)")

# Cache the download in the project-level .toolchain/ directory so it
# survives build-directory wipes.
set(KELVIN_TC_CACHE "${CMAKE_CURRENT_LIST_DIR}/../.toolchain")
set(KELVIN_TC_DIR "${KELVIN_TC_CACHE}/toolchain_kelvin_v2")
set(KELVIN_TC_TARBALL "${KELVIN_TC_CACHE}/toolchain_kelvin_v2.tar.gz")

if(NOT EXISTS "${KELVIN_TC_DIR}/bin/riscv32-unknown-elf-gcc")
    file(MAKE_DIRECTORY "${KELVIN_TC_CACHE}")
    if(NOT EXISTS "${KELVIN_TC_TARBALL}")
        message(STATUS "Downloading Kelvin RV32 toolchain (~1.4 GB)...")
        file(DOWNLOAD "${KELVIN_TC_URL}" "${KELVIN_TC_TARBALL}"
            SHOW_PROGRESS
            STATUS DL_STATUS)
        list(GET DL_STATUS 0 DL_RC)
        if(NOT DL_RC EQUAL 0)
            file(REMOVE "${KELVIN_TC_TARBALL}")
            message(FATAL_ERROR "Toolchain download failed: ${DL_STATUS}")
        endif()
    endif()
    message(STATUS "Extracting Kelvin RV32 toolchain...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${KELVIN_TC_TARBALL}"
        WORKING_DIRECTORY "${KELVIN_TC_CACHE}"
        RESULT_VARIABLE TAR_RC)
    if(NOT TAR_RC EQUAL 0)
        message(FATAL_ERROR "Toolchain extraction failed")
    endif()
    if(NOT EXISTS "${KELVIN_TC_DIR}/bin/riscv32-unknown-elf-gcc")
        message(FATAL_ERROR "Toolchain extraction produced unexpected layout")
    endif()
endif()

# ── Compiler setup (GCC from Kelvin toolchain) ──────────────────────
set(TC "${KELVIN_TC_DIR}")

set(CMAKE_C_COMPILER   "${TC}/bin/riscv32-unknown-elf-gcc")
set(CMAKE_CXX_COMPILER "${TC}/bin/riscv32-unknown-elf-g++")
set(CMAKE_ASM_COMPILER "${TC}/bin/riscv32-unknown-elf-gcc")
set(CMAKE_AR           "${TC}/bin/riscv32-unknown-elf-ar")
set(CMAKE_RANLIB       "${TC}/bin/riscv32-unknown-elf-ranlib")
set(CMAKE_OBJCOPY      "${TC}/bin/riscv32-unknown-elf-objcopy")
set(CMAKE_OBJDUMP      "${TC}/bin/riscv32-unknown-elf-objdump")

# Architecture must match the toolchain's newlib/libgcc build:
#   --with-arch=rv32im_zve32x --with-abi=ilp32
# We add F + Zve32f so the compiler can emit hw float & vector instructions,
# but the ABI stays ilp32 (soft-float calling convention) to link with the
# pre-built libraries.
set(RV32_ARCH_FLAGS "-march=rv32imf_zve32x_zve32f -mabi=ilp32")

set(CMAKE_C_FLAGS_INIT
    "${RV32_ARCH_FLAGS} -fno-exceptions -fno-unwind-tables -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS_INIT
    "${RV32_ARCH_FLAGS} -fno-exceptions -fno-rtti -fno-unwind-tables -ffunction-sections -fdata-sections")
set(CMAKE_ASM_FLAGS_INIT
    "${RV32_ARCH_FLAGS}")

# Link with newlib + libgcc, use our own linker script (passed per-target).
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-nostartfiles -Wl,--gc-sections")

# Don't try to compile a test program (it would fail without our startup code).
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

# Find programs on the host, libraries/headers in the sysroot.
set(CMAKE_FIND_ROOT_PATH "${TC}/riscv32-unknown-elf")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
