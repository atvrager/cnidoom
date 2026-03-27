cmake_minimum_required(VERSION 3.16)
project(doom_agent C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(DOOM_AGENT_BUILD_TESTS "Build unit tests for inference code" ON)
option(DOOM_AGENT_HOST "Build host inference engine (TFLM + SDL2)" OFF)

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

if(EXISTS "${TFLM_DIR}/tensorflow/lite/micro")
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
                "run: xxd -i models/doom_agent_int8.tflite > models/doom_agent_model.c")
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
endif()
