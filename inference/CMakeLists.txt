cmake_minimum_required(VERSION 3.16)
project(doom_agent C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(DOOM_AGENT_BUILD_TESTS "Build unit tests for inference code" ON)
option(DOOM_AGENT_HOST "Build host inference engine (TFLM + SDL2)" OFF)
option(DOOM_AGENT_STATIC "Build static code-generated inference backend" OFF)
option(DOOM_AGENT_RV32 "Build bare-metal RISC-V 32-bit target for QEMU" OFF)

set(DOOM_AGENT_KERNEL_TARGET "generic" CACHE STRING
    "Kernel target: generic, x86, riscv")
set(DOOM_AGENT_MODEL "${CMAKE_SOURCE_DIR}/../models/doom_agent_int8.tflite"
    CACHE FILEPATH "Path to INT8 TFLite model for codegen")
set(DOOM_AGENT_EMBED_WAD "${CMAKE_SOURCE_DIR}/../wads/DOOM1.WAD"
    CACHE FILEPATH "WAD file to embed in the binary (empty = use semihosting)")

set(DOOMGENERIC_DIR "${CMAKE_SOURCE_DIR}/../doomgeneric/doomgeneric"
    CACHE PATH "Path to doomgeneric source directory")
set(TFLM_DIR "${CMAKE_SOURCE_DIR}/../tflite-micro"
    CACHE PATH "Path to TFLite Micro source directory")
set(MODEL_DATA "${CMAKE_SOURCE_DIR}/../models/doom_agent_model.c"
    CACHE FILEPATH "Path to xxd-generated model data C file")

# --------------------------------------------------------------------------
# Build TFLM static library from source (if submodule is present)
# --------------------------------------------------------------------------
set(TFLM_LIB "")
set(TFLM_INCLUDE_DIRS "")

if(EXISTS "${TFLM_DIR}/tensorflow/lite/micro" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Generic")
    # Build via TFLM's own Makefile — it handles dependency downloads,
    # kernel selection, and host CPU optimization automatically.
    set(TFLM_GEN_DIR "${TFLM_DIR}/gen")
    set(TFLM_MAKEFILE "${TFLM_DIR}/tensorflow/lite/micro/tools/make/Makefile")

    # Detect the output directory TFLM uses (e.g. linux_x86_64_default_gcc).
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(TFLM_TARGET_DIR "${TFLM_GEN_DIR}/linux_${HOST_ARCH}_default_gcc")
    set(TFLM_LIB_PATH "${TFLM_TARGET_DIR}/lib/libtensorflow-microlite.a")

    if(NOT EXISTS "${TFLM_LIB_PATH}")
        message(STATUS "Building TFLM static library (this may take a minute)...")
        execute_process(
            COMMAND make -f ${TFLM_MAKEFILE} microlite -j
            WORKING_DIRECTORY ${TFLM_DIR}
            RESULT_VARIABLE TFLM_BUILD_RESULT
        )
        if(NOT TFLM_BUILD_RESULT EQUAL 0)
            message(WARNING "TFLM build failed — TFLM targets will be skipped")
        endif()
    endif()

    if(EXISTS "${TFLM_LIB_PATH}")
        set(TFLM_LIB "${TFLM_LIB_PATH}")
        set(TFLM_INCLUDE_DIRS
            ${TFLM_DIR}
            ${TFLM_DIR}/third_party/flatbuffers/include
            ${TFLM_DIR}/third_party/gemmlowp
            ${TFLM_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
            ${TFLM_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp
        )
        message(STATUS "TFLM library: ${TFLM_LIB}")
    endif()
else()
    message(STATUS "tflite-micro not found at ${TFLM_DIR} — "
            "run: git submodule update --init --recursive")
endif()

# --------------------------------------------------------------------------
# Preprocessing library (pure C, no external deps — testable standalone)
# --------------------------------------------------------------------------
add_library(doom_agent_preprocess STATIC
    doom_agent_preprocess.c
    doom_agent.h
)
target_include_directories(doom_agent_preprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --------------------------------------------------------------------------
# Golden log library (pure C, no external deps)
# --------------------------------------------------------------------------
add_library(doom_agent_golden_log STATIC
    golden_log.c
    golden_log.h
)
target_include_directories(doom_agent_golden_log PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --------------------------------------------------------------------------
# Agent orchestration (needs a backend at link time)
# --------------------------------------------------------------------------
add_library(doom_agent_core STATIC
    doom_agent.c
    doom_agent.h
)
target_include_directories(doom_agent_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(doom_agent_core PUBLIC doom_agent_preprocess)

# --------------------------------------------------------------------------
# TFLM bridge — bare metal backend (compiled-in model)
# --------------------------------------------------------------------------
if(TFLM_LIB)
    add_library(doom_agent_tflm STATIC
        doom_agent_tflm.cc
    )
    target_include_directories(doom_agent_tflm PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TFLM_INCLUDE_DIRS}
    )
    target_compile_definitions(doom_agent_tflm PUBLIC TF_LITE_STATIC_MEMORY)
    target_link_libraries(doom_agent_tflm PUBLIC
        doom_agent_core
        ${TFLM_LIB}
    )
endif()

# --------------------------------------------------------------------------
# Static code-generated inference backend
# --------------------------------------------------------------------------
if(DOOM_AGENT_STATIC)
    set(GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")

    # Codegen: generate C source from TFLite model.
    set(CODEGEN_SCRIPT "${CMAKE_SOURCE_DIR}/../tools/codegen_graph.py")
    set(GEN_GRAPH_C "${GENERATED_DIR}/doom_agent_graph.c")
    set(GEN_GRAPH_H "${GENERATED_DIR}/doom_agent_graph.h")
    set(GEN_WEIGHTS_C "${GENERATED_DIR}/doom_agent_weights.c")
    set(GEN_WEIGHTS_H "${GENERATED_DIR}/doom_agent_weights.h")

    add_custom_command(
        OUTPUT ${GEN_GRAPH_C} ${GEN_GRAPH_H} ${GEN_WEIGHTS_C} ${GEN_WEIGHTS_H}
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}/../tflite-micro
            uv run python ${CODEGEN_SCRIPT}
                --model ${DOOM_AGENT_MODEL}
                --output-dir ${GENERATED_DIR}
        DEPENDS ${CODEGEN_SCRIPT} ${DOOM_AGENT_MODEL}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
        COMMENT "Generating static inference engine from ${DOOM_AGENT_MODEL}"
    )

    # Collect kernel sources.  Target-specific files override generic ones.
    file(GLOB GENERIC_KERNEL_SRCS "kernels/generic/*.c")
    set(KERNEL_SRCS ${GENERIC_KERNEL_SRCS})

    if(NOT "${DOOM_AGENT_KERNEL_TARGET}" STREQUAL "generic")
        file(GLOB TARGET_KERNEL_SRCS
            "kernels/${DOOM_AGENT_KERNEL_TARGET}/*.c")
        foreach(TARGET_SRC ${TARGET_KERNEL_SRCS})
            get_filename_component(TARGET_NAME ${TARGET_SRC} NAME)
            # Remove the generic file with the same name.
            list(FILTER KERNEL_SRCS EXCLUDE REGEX "/${TARGET_NAME}$")
            list(APPEND KERNEL_SRCS ${TARGET_SRC})
        endforeach()
        message(STATUS "Kernel target: ${DOOM_AGENT_KERNEL_TARGET} "
                "(${TARGET_KERNEL_SRCS})")
    else()
        message(STATUS "Kernel target: generic (reference C)")
    endif()

    # Static inference library.
    add_library(doom_agent_static STATIC
        doom_agent_static.c
        ${GEN_GRAPH_C}
        ${GEN_WEIGHTS_C}
        ${KERNEL_SRCS}
    )
    target_include_directories(doom_agent_static PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GENERATED_DIR}
    )
    target_link_libraries(doom_agent_static PUBLIC doom_agent_core m)

    # Enable AVX2 for x86 kernels.
    if("${DOOM_AGENT_KERNEL_TARGET}" STREQUAL "x86")
        target_compile_options(doom_agent_static PRIVATE -mavx2)
    endif()
endif()

# --------------------------------------------------------------------------
# Full agent binary — bare metal / headless (doomgeneric + TFLM + compiled-in model)
# --------------------------------------------------------------------------
if(EXISTS "${DOOMGENERIC_DIR}/doomgeneric.h" AND TARGET doom_agent_tflm)
    file(GLOB DOOM_SRCS "${DOOMGENERIC_DIR}/*.c")
    # Exclude ALL platform backends and allegro-specific files.
    list(FILTER DOOM_SRCS EXCLUDE REGEX "doomgeneric_[a-z]+\\.c$")
    list(FILTER DOOM_SRCS EXCLUDE REGEX "i_allegro")
    list(FILTER DOOM_SRCS EXCLUDE REGEX "i_sdlmusic\\.c$")
    list(FILTER DOOM_SRCS EXCLUDE REGEX "i_sdlsound\\.c$")

    add_executable(doom_agent_bin
        doomgeneric_agent.c
        ${DOOM_SRCS}
    )
    if(EXISTS "${MODEL_DATA}")
        target_sources(doom_agent_bin PRIVATE ${MODEL_DATA})
    else()
        message(WARNING "Model data not found at ${MODEL_DATA} — "
                "run: xxd -i -n doom_agent_model models/doom_agent_int8.tflite > models/doom_agent_model.c")
    endif()

    target_include_directories(doom_agent_bin PRIVATE ${DOOMGENERIC_DIR})
    target_link_libraries(doom_agent_bin PRIVATE
        doom_agent_core doom_agent_tflm m pthread
    )
    set_target_properties(doom_agent_bin PROPERTIES OUTPUT_NAME doom_agent)
else()
    message(STATUS "Skipping headless agent binary — "
            "doomgeneric or TFLM not available")
endif()

# --------------------------------------------------------------------------
# Host inference engine (TFLM from file + SDL2 + doomgeneric)
# --------------------------------------------------------------------------
if(DOOM_AGENT_HOST)
    if(NOT EXISTS "${DOOMGENERIC_DIR}/doomgeneric.h")
        message(FATAL_ERROR
            "DOOM_AGENT_HOST requires doomgeneric. "
            "Run: git submodule update --init --recursive")
    endif()

    if(NOT TFLM_LIB)
        message(FATAL_ERROR
            "DOOM_AGENT_HOST requires TFLM. "
            "Run: git submodule update --init --recursive")
    endif()

    find_package(SDL2 REQUIRED)

    # Collect doomgeneric sources (exclude all platform backends).
    file(GLOB HOST_DOOM_SRCS "${DOOMGENERIC_DIR}/*.c")
    list(FILTER HOST_DOOM_SRCS EXCLUDE REGEX "doomgeneric_[a-z]+\\.c$")
    list(FILTER HOST_DOOM_SRCS EXCLUDE REGEX "i_allegro")
    list(FILTER HOST_DOOM_SRCS EXCLUDE REGEX "i_sdlmusic\\.c$")
    list(FILTER HOST_DOOM_SRCS EXCLUDE REGEX "i_sdlsound\\.c$")

    add_executable(doom_agent_host_bin
        doomgeneric_agent_sdl.c
        doom_agent_host.cc
        doom_agent_preprocess.c
        golden_log.c
        ${HOST_DOOM_SRCS}
    )

    target_compile_definitions(doom_agent_host_bin PRIVATE
        DOOM_AGENT_HOST=1
        TF_LITE_STATIC_MEMORY
    )

    target_include_directories(doom_agent_host_bin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DOOMGENERIC_DIR}
        ${TFLM_INCLUDE_DIRS}
        ${SDL2_INCLUDE_DIRS}
    )

    target_link_libraries(doom_agent_host_bin PRIVATE
        ${TFLM_LIB}
        ${SDL2_LIBRARIES}
        m pthread
    )

    set_target_properties(doom_agent_host_bin PROPERTIES
        OUTPUT_NAME doom_agent_host
    )
endif()

# --------------------------------------------------------------------------
# Static host binary (SDL2 + static code-generated backend — no TFLM)
# --------------------------------------------------------------------------
if(DOOM_AGENT_HOST AND TARGET doom_agent_static)
    # Reuse the doomgeneric sources collected for the TFLM host above.
    add_executable(doom_agent_static_host_bin
        doomgeneric_agent_sdl.c
        doom_agent_static_host.c
        doom_agent_preprocess.c
        golden_log.c
        ${HOST_DOOM_SRCS}
    )

    target_compile_definitions(doom_agent_static_host_bin PRIVATE
        DOOM_AGENT_HOST=1
    )

    target_include_directories(doom_agent_static_host_bin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GENERATED_DIR}
        ${DOOMGENERIC_DIR}
        ${SDL2_INCLUDE_DIRS}
    )

    target_link_libraries(doom_agent_static_host_bin PRIVATE
        doom_agent_static
        ${SDL2_LIBRARIES}
        m pthread
    )

    if("${DOOM_AGENT_KERNEL_TARGET}" STREQUAL "x86")
        target_compile_options(doom_agent_static_host_bin PRIVATE -mavx2)
    endif()

    set_target_properties(doom_agent_static_host_bin PROPERTIES
        OUTPUT_NAME doom_agent_static_host
    )
endif()

# --------------------------------------------------------------------------
# Bare-metal RISC-V 32-bit target (QEMU virt + ramfb display)
# --------------------------------------------------------------------------
if(DOOM_AGENT_RV32 AND TARGET doom_agent_static)
    if(NOT EXISTS "${DOOMGENERIC_DIR}/doomgeneric.h")
        message(FATAL_ERROR
            "DOOM_AGENT_RV32 requires doomgeneric. "
            "Run: git submodule update --init --recursive")
    endif()

    # Collect doomgeneric sources (same filtering as other targets).
    file(GLOB RV32_DOOM_SRCS "${DOOMGENERIC_DIR}/*.c")
    list(FILTER RV32_DOOM_SRCS EXCLUDE REGEX "doomgeneric_[a-z]+\\.c$")
    list(FILTER RV32_DOOM_SRCS EXCLUDE REGEX "i_allegro")
    list(FILTER RV32_DOOM_SRCS EXCLUDE REGEX "i_sdlmusic\\.c$")
    list(FILTER RV32_DOOM_SRCS EXCLUDE REGEX "i_sdlsound\\.c$")

    # Embedded WAD: convert the WAD binary into a linkable .o with a
    # .wad section (placed in DDR by the linker script).  Replaces
    # semihosting file I/O with a const-blob memory-mapped backend.
    set(RV32_WAD_SRCS "")
    if(DOOM_AGENT_EMBED_WAD AND EXISTS "${DOOM_AGENT_EMBED_WAD}")
        set(EMBEDDED_WAD_OBJ "${CMAKE_CURRENT_BINARY_DIR}/embedded_wad.o")

        add_custom_command(
            OUTPUT ${EMBEDDED_WAD_OBJ}
            COMMAND ${CMAKE_COMMAND} -E copy
                "${DOOM_AGENT_EMBED_WAD}"
                "${CMAKE_CURRENT_BINARY_DIR}/embedded_wad.bin"
            COMMAND ${CMAKE_OBJCOPY}
                -I binary -O elf32-littleriscv -B riscv
                --rename-section .data=.wad,alloc,load,readonly,data,contents
                --redefine-sym _binary_embedded_wad_bin_start=_embedded_wad_start
                --redefine-sym _binary_embedded_wad_bin_end=_embedded_wad_end
                --redefine-sym _binary_embedded_wad_bin_size=_embedded_wad_size
                embedded_wad.bin "${EMBEDDED_WAD_OBJ}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS "${DOOM_AGENT_EMBED_WAD}"
            COMMENT "Embedding WAD: ${DOOM_AGENT_EMBED_WAD}"
        )

        # Replace stdc file I/O with our memory-mapped embedded backend.
        list(FILTER RV32_DOOM_SRCS EXCLUDE REGEX "w_file_stdc\\.c$")
        set(RV32_WAD_SRCS
            "${CMAKE_CURRENT_SOURCE_DIR}/w_file_embedded.c"
            "${EMBEDDED_WAD_OBJ}"
        )

        message(STATUS "Embedded WAD: ${DOOM_AGENT_EMBED_WAD}")
    else()
        message(STATUS "WAD: semihosting (no embedded WAD)")
    endif()

    # Platform support sources (startup, UART, syscalls, ramfb).
    set(RV32_PLATFORM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/platform/rv32")
    set(RV32_PLATFORM_SRCS
        ${RV32_PLATFORM_DIR}/startup.S
        ${RV32_PLATFORM_DIR}/syscalls.c
        ${RV32_PLATFORM_DIR}/uart.c
        ${RV32_PLATFORM_DIR}/ramfb.c
    )

    # Linker script.
    set(RV32_LINKER_SCRIPT "${RV32_PLATFORM_DIR}/linker.ld")

    add_executable(doom_agent_rv32_bin
        doomgeneric_agent_rv32.c
        ${RV32_DOOM_SRCS}
        ${RV32_WAD_SRCS}
        ${RV32_PLATFORM_SRCS}
    )

    target_include_directories(doom_agent_rv32_bin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DOOMGENERIC_DIR}
        ${GENERATED_DIR}
    )

    # doom_agent_core references tflm_* symbols defined in doom_agent_static.
    # Wrap all our static libs in --start-group / --end-group so the linker
    # rescans them to resolve the circular dependency.
    target_link_libraries(doom_agent_rv32_bin PRIVATE
        -Wl,--start-group
        doom_agent_static doom_agent_core doom_agent_preprocess
        -Wl,--end-group
        -lc -lm -lgcc
    )

    # Bare-metal link flags: use our linker script, link newlib + libgcc.
    target_link_options(doom_agent_rv32_bin PRIVATE
        -T${RV32_LINKER_SCRIPT}
        -Wl,--no-warn-rwx-segments
        -Wl,--Map=$<TARGET_FILE_DIR:doom_agent_rv32_bin>/doom_agent_rv32.map
    )
    set_target_properties(doom_agent_rv32_bin PROPERTIES
        OUTPUT_NAME doom_agent_rv32
        SUFFIX ".elf"
        LINK_DEPENDS ${RV32_LINKER_SCRIPT}
    )

    message(STATUS "RISC-V 32-bit target enabled (QEMU virt, ramfb display)")

    # RVV vs generic bit-accuracy test (bare-metal, runs under QEMU).
    if("${DOOM_AGENT_KERNEL_TARGET}" STREQUAL "riscv")
        set(RV32_TEST_PLATFORM_SRCS
            ${RV32_PLATFORM_DIR}/startup.S
            ${RV32_PLATFORM_DIR}/syscalls.c
            ${RV32_PLATFORM_DIR}/uart.c
        )

        add_executable(test_rvv_bitexact
            test_rvv_bitexact.c
            test_rvv_generic_kernels.c
            ${RV32_TEST_PLATFORM_SRCS}
        )
        target_include_directories(test_rvv_bitexact PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        target_link_libraries(test_rvv_bitexact PRIVATE
            doom_agent_static
            -lc -lm -lgcc
        )
        target_link_options(test_rvv_bitexact PRIVATE
            -T${RV32_LINKER_SCRIPT}
            -Wl,--no-warn-rwx-segments
        )
        set_target_properties(test_rvv_bitexact PROPERTIES
            OUTPUT_NAME test_rvv_bitexact
            SUFFIX ".elf"
            LINK_DEPENDS ${RV32_LINKER_SCRIPT}
        )

        # CTest integration: run under QEMU with a 60-second timeout.
        find_program(QEMU_RV32 qemu-system-riscv32)
        if(QEMU_RV32)
            add_test(
                NAME rvv_bitexact
                COMMAND ${QEMU_RV32}
                    -machine virt
                    -cpu rv32,v=true,vlen=128,zve32f=true
                    -m 128M
                    -nographic
                    -bios none
                    -semihosting-config enable=on,target=native
                    -kernel $<TARGET_FILE:test_rvv_bitexact>
            )
            set_tests_properties(rvv_bitexact PROPERTIES TIMEOUT 60)
        endif()
    endif()
endif()

# --------------------------------------------------------------------------
# Tests (CTest)
# --------------------------------------------------------------------------
if(DOOM_AGENT_BUILD_TESTS)
    enable_testing()

    add_executable(test_preprocess test_preprocess.c)
    target_link_libraries(test_preprocess PRIVATE doom_agent_preprocess m)
    add_test(NAME preprocess COMMAND test_preprocess)

    add_executable(test_quantize test_quantize.c)
    target_link_libraries(test_quantize PRIVATE doom_agent_preprocess m)
    add_test(NAME quantize COMMAND test_quantize)

    # Static vs TFLM byte-identical comparison test.
    if(TARGET doom_agent_static AND TFLM_LIB)
        add_executable(test_static_vs_tflm test_static_vs_tflm.cc)
        target_include_directories(test_static_vs_tflm PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${GENERATED_DIR}
            ${TFLM_INCLUDE_DIRS}
        )
        target_compile_definitions(test_static_vs_tflm PRIVATE
            TF_LITE_STATIC_MEMORY
        )
        target_link_libraries(test_static_vs_tflm PRIVATE
            doom_agent_static
            ${TFLM_LIB}
            m pthread
        )
        if("${DOOM_AGENT_KERNEL_TARGET}" STREQUAL "x86")
            target_compile_options(test_static_vs_tflm PRIVATE -mavx2)
        endif()
        set_target_properties(test_static_vs_tflm PROPERTIES
            OUTPUT_NAME test_static_vs_tflm
        )
        add_test(
            NAME static_vs_tflm
            COMMAND test_static_vs_tflm
                --model ${DOOM_AGENT_MODEL} --trials 100
        )
    endif()
endif()
