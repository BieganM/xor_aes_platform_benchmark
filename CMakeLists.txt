cmake_minimum_required(VERSION 3.18)
project(HPC_Encryption_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_OPENMP "Build OpenMP version" ON)
option(BUILD_CUDA "Build CUDA version" OFF)
option(BUILD_METAL "Build Metal version" OFF)

if(APPLE)
    set(BUILD_METAL ON)
    set(BUILD_CUDA OFF)
    message(STATUS "Detected macOS - enabling Metal, disabling CUDA")
endif()

if(UNIX AND NOT APPLE)
    set(BUILD_CUDA ON)
    set(BUILD_METAL OFF)
    message(STATUS "Detected Linux - enabling CUDA, disabling Metal")
endif()

# Auto-detect OpenSSL on macOS (Homebrew)
if(APPLE AND NOT OPENSSL_ROOT_DIR)
    execute_process(
        COMMAND brew --prefix openssl
        OUTPUT_VARIABLE BREW_OPENSSL_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE BREW_OPENSSL_RESULT
    )
    if(BREW_OPENSSL_RESULT EQUAL 0)
        set(OPENSSL_ROOT_DIR "${BREW_OPENSSL_PREFIX}")
        message(STATUS "Auto-detected OpenSSL: ${OPENSSL_ROOT_DIR}")
    endif()
endif()

find_package(OpenSSL REQUIRED)

if(BUILD_OPENMP)
    if(APPLE)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE LIBOMP_RESULT
        )
        if(LIBOMP_RESULT EQUAL 0)
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        endif()
    endif()
    
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found - install with: brew install libomp")
        set(BUILD_OPENMP OFF)
    endif()
endif()

if(BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")
    else()
        message(WARNING "CUDA not found - disabling CUDA support")
        set(BUILD_CUDA OFF)
    endif()
endif()

set(COMMON_SOURCES
    src/common/timer.cpp
    src/common/csv_logger.cpp
    src/common/verification.cpp
    src/common/file_utils.cpp
    src/common/power_monitor.cpp
)

set(XOR_CPU_SOURCES
    src/engines/xor/xor_sequential.cpp
)

set(AES_CPU_SOURCES
    src/engines/aes/aes_sequential.cpp
)

if(BUILD_OPENMP)
    list(APPEND XOR_CPU_SOURCES src/engines/xor/xor_openmp.cpp)
    list(APPEND AES_CPU_SOURCES src/engines/aes/aes_openmp.cpp)
endif()

set(ALL_SOURCES
    src/main.cpp
    ${COMMON_SOURCES}
    ${XOR_CPU_SOURCES}
    ${AES_CPU_SOURCES}
)

if(BUILD_METAL)
    list(APPEND ALL_SOURCES
        src/engines/xor/xor_metal.mm
        src/engines/aes/aes_metal.mm
    )
endif()

if(BUILD_CUDA)
    list(APPEND ALL_SOURCES
        src/engines/xor/xor_cuda.cu
        src/engines/aes/aes_cuda.cu
    )
endif()

add_executable(hpc_benchmark ${ALL_SOURCES})

target_include_directories(hpc_benchmark PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OPENSSL_INCLUDE_DIR}
)

target_link_libraries(hpc_benchmark PRIVATE
    ${OPENSSL_LIBRARIES}
)

if(BUILD_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(hpc_benchmark PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(hpc_benchmark PRIVATE HAS_OPENMP)
endif()

if(BUILD_CUDA)
    set_target_properties(hpc_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "60;70;75;80;86"
    )
    target_compile_definitions(hpc_benchmark PRIVATE HAS_CUDA)
endif()

if(BUILD_METAL)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    target_link_libraries(hpc_benchmark PRIVATE
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
    )
    target_compile_definitions(hpc_benchmark PRIVATE HAS_METAL)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(hpc_benchmark PRIVATE -O3 -march=native)
    endif()
endif()



message(STATUS "")
message(STATUS "=== HPC Encryption Benchmark Configuration ===")
message(STATUS "Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP:         ${BUILD_OPENMP}")
message(STATUS "CUDA:           ${BUILD_CUDA}")
message(STATUS "Metal:          ${BUILD_METAL}")
message(STATUS "OpenSSL:        ${OPENSSL_VERSION}")
message(STATUS "===============================================")
message(STATUS "")
