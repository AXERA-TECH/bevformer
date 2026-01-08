cmake_minimum_required(VERSION 3.13)

# Target chip type (default to ax650)
if(NOT DEFINED AXERA_TARGET_CHIP)
    set(AXERA_TARGET_CHIP "ax650" CACHE STRING "AXERA target chip (ax650, ax637)")
endif()

project(BEVFormer CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building BEVFormer for chip: ${AXERA_TARGET_CHIP}")

# Set OpenCV directory (default to bundled prebuilt path, allow override)
if(NOT DEFINED OpenCV_DIR)
    if(DEFINED ENV{OpenCV_DIR})
        set(OpenCV_DIR "${ENV{OpenCV_DIR}}" CACHE PATH "OpenCV directory")
    else()
        set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv-aarch64-linux/lib/cmake/opencv4" CACHE PATH "OpenCV directory")
    endif()
endif()

message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

find_package(Threads REQUIRED)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
endif()

set(CHIP_CMAKE_FILE "${CMAKE_SOURCE_DIR}/cmake/${AXERA_TARGET_CHIP}.cmake")
if(EXISTS "${CHIP_CMAKE_FILE}")
    include("${CHIP_CMAKE_FILE}")
    message(STATUS "Using chip-specific CMake file: ${CHIP_CMAKE_FILE}")
else()
    message(STATUS "Chip-specific CMake file not found: ${CHIP_CMAKE_FILE}, using default configuration")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
if(JSONCPP_FOUND)
    include_directories(${JSONCPP_INCLUDE_DIRS})
    link_directories(${JSONCPP_LIBRARY_DIRS})
else()
    message(WARNING "jsoncpp not found via pkg-config, assuming system installation")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

set(SOURCES
    src/main.cpp
    src/bevformer_common.cpp
    src/data_loader.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/visualization.cpp
    src/utils.cpp
)

set(HEADERS
    include/bevformer_common.hpp
    include/data_loader.hpp
    include/preprocess.hpp
    include/postprocess.hpp
    include/visualization.hpp
    include/utils.hpp
    include/timer.hpp
)

add_executable(bevformer_inference ${SOURCES} ${HEADERS})

target_link_libraries(bevformer_inference
    ${OpenCV_LIBS}
    Threads::Threads
    ax_engine
    ax_interpreter
    ax_sys
    ax_ivps
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(bevformer_inference OpenMP::OpenMP_CXX)
endif()

# jsoncpp is optional - only link if found
if(JSONCPP_FOUND)
    target_link_libraries(bevformer_inference ${JSONCPP_LIBRARIES})
endif()

if(DEFINED BSP_MSP_DIR)
    target_include_directories(bevformer_inference PRIVATE ${BSP_MSP_DIR}/include)
    target_link_directories(bevformer_inference PRIVATE ${BSP_MSP_DIR}/lib)
else()
    message(WARNING "BSP_MSP_DIR not set, using default path")
    target_include_directories(bevformer_inference PRIVATE /opt/axera/msp/out/include)
    target_link_directories(bevformer_inference PRIVATE /opt/axera/msp/out/lib)
endif()

target_compile_options(bevformer_inference PRIVATE -O3 -Wall)

install(TARGETS bevformer_inference DESTINATION bin)

