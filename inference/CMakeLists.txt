cmake_minimum_required(VERSION 3.16)
project(doom_agent C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(DOOM_AGENT_BUILD_TESTS "Build unit tests for inference code" ON)

set(DOOMGENERIC_DIR "${CMAKE_SOURCE_DIR}/../doomgeneric/doomgeneric"
    CACHE PATH "Path to doomgeneric source directory")
set(TFLM_DIR "${CMAKE_SOURCE_DIR}/../tflite-micro"
    CACHE PATH "Path to TFLite Micro source/install directory")
set(MODEL_DATA "${CMAKE_SOURCE_DIR}/../models/doom_agent_model.c"
    CACHE FILEPATH "Path to xxd-generated model data C file")

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
# Agent orchestration (needs TFLM bridge at link time)
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
# TFLM bridge (C++ — needs tflite-micro headers and static lib)
# --------------------------------------------------------------------------
if(EXISTS "${TFLM_DIR}/tensorflow")
    add_library(doom_agent_tflm STATIC
        doom_agent_tflm.cc
    )
    target_include_directories(doom_agent_tflm PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TFLM_DIR}
        ${TFLM_DIR}/third_party/flatbuffers/include
        ${TFLM_DIR}/third_party/gemmlowp
    )
    target_link_libraries(doom_agent_tflm PUBLIC doom_agent_core)

    # Link the pre-built TFLM static lib if available.
    find_library(TFLM_LIB
        NAMES tensorflow-microlite
        HINTS ${TFLM_DIR}/lib ${TFLM_DIR}/gen/lib
    )
    if(TFLM_LIB)
        target_link_libraries(doom_agent_tflm PUBLIC ${TFLM_LIB})
    endif()
else()
    message(STATUS "tflite-micro not found at ${TFLM_DIR} — skipping TFLM build")
endif()

# --------------------------------------------------------------------------
# Full agent binary (links doomgeneric + tflm + model data)
# --------------------------------------------------------------------------
if(EXISTS "${DOOMGENERIC_DIR}/doomgeneric.h" AND TARGET doom_agent_tflm)
    # Collect doomgeneric sources (exclude other platform backends).
    file(GLOB DOOM_SRCS "${DOOMGENERIC_DIR}/*.c")
    list(FILTER DOOM_SRCS EXCLUDE REGEX "doomgeneric_sdl\\.c$")
    list(FILTER DOOM_SRCS EXCLUDE REGEX "doomgeneric_win\\.c$")
    list(FILTER DOOM_SRCS EXCLUDE REGEX "doomgeneric_xlib\\.c$")
    list(FILTER DOOM_SRCS EXCLUDE REGEX "doomgeneric_sosol\\.c$")

    add_executable(doom_agent_bin
        doomgeneric_agent.c
        ${DOOM_SRCS}
    )
    # Model data (xxd -i output).
    if(EXISTS "${MODEL_DATA}")
        target_sources(doom_agent_bin PRIVATE ${MODEL_DATA})
    else()
        message(WARNING "Model data not found at ${MODEL_DATA} — "
                "run: xxd -i models/doom_agent_int8.tflite > models/doom_agent_model.c")
    endif()

    target_include_directories(doom_agent_bin PRIVATE
        ${DOOMGENERIC_DIR}
    )
    target_link_libraries(doom_agent_bin PRIVATE
        doom_agent_core
        doom_agent_tflm
        m
        pthread
    )
    set_target_properties(doom_agent_bin PROPERTIES
        OUTPUT_NAME doom_agent
    )
else()
    message(STATUS "Skipping full agent binary — "
            "doomgeneric or tflite-micro not available")
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
