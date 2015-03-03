project(mallocMC)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")


###############################################################################
# CUDA
###############################################################################
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-use_fast_math;")
set(CUDA_ARCH sm_35 CACHE STRING "Set GPU architecture")

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mallocMC/src/include)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/scatteralloc)

OPTION(CUDA_OUTPUT_INTERMEDIATE_CODE "Output ptx code" OFF)
if(CUDA_OUTPUT_INTERMEDIATE_CODE)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas;-v;--keep;--keep-dir ${PROJECT_BINARY_DIR}/nvcctemp;")
endif(CUDA_OUTPUT_INTERMEDIATE_CODE)

SET(CUDA_OPTIMIZATION_TYPE "-O3" CACHE STRING "CUDA Optimization")
set_property(CACHE CUDA_OPTIMIZATION_TYPE PROPERTY STRINGS "unset;-G0;-O0;-O1;-O2;-O3")
if(NOT ${CUDA_OPTIMIZATION_TYPE} STREQUAL  "unset")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_OPTIMIZATION_TYPE}")
endif()


###############################################################################
# Boost
###############################################################################
find_package(Boost 1.48.0 REQUIRED filesystem system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)


################################################################################
# Halloc
################################################################################
set(HALLOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mallocMC/halloc")
set(HALLOC_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mallocMC/halloc")

set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-rdc=true")

cuda_include_directories(${HALLOC_INSTALL_DIR}/include)
list(APPEND CUDA_NVCC_FLAGS "-lhalloc")
list(APPEND CUDA_NVCC_FLAGS "-L${HALLOC_INSTALL_DIR}/lib")
list(APPEND LIBS "${HALLOC_INSTALL_DIR}/lib/libhalloc.a")

add_custom_target(halloc
  COMMAND mkdir -p "${HALLOC_INSTALL_DIR}/include"
  COMMAND mkdir -p "${HALLOC_INSTALL_DIR}/lib"
  COMMAND make --directory=${HALLOC_DIR}
  COMMAND make install --directory=${HALLOC_DIR} PREFIX=${HALLOC_INSTALL_DIR}
  )


###############################################################################
# Executables
###############################################################################

#set(WHICH_BENCHMARK "benchmark_1")
#set(WHICH_BENCHMARK "benchmark_2")
set(WHICH_BENCHMARK "benchmark_3")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/${WHICH_BENCHMARK}.c*")

add_executable(result_parser "${CMAKE_CURRENT_SOURCE_DIR}/src/parse_results.cpp")
target_link_libraries(result_parser ${LIBS})


###############################################################################
# Benchmarks with MALLOCMC
###############################################################################
foreach(ALLOCATOR_LOOP MALLOCMC)
  foreach(TYPEDEF_LOOP ScatterAllocator ScatterAllocatorCoalesced HallocAllocator)
    foreach(ALLOCATION_SIZE_LOOP 16 32 64 128 LOG LIN)
      add_definitions(-DBENCHMARK_ALLOCATOR=${ALLOCATOR_LOOP})
      add_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOCATION_SIZE_LOOP})
      add_definitions(-DMMC_TYPEDEF=${TYPEDEF_LOOP})

      set(MY_NAME "${WHICH_BENCHMARK}_${ALLOCATOR_LOOP}_${TYPEDEF_LOOP}_${ALLOCATION_SIZE_LOOP}")
      cuda_add_executable(${MY_NAME} ${SRCFILES})
      target_link_libraries(${MY_NAME} ${LIBS})

      remove_definitions(-DMMC_TYPEDEF=${TYPEDEF_LOOP})
      remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOCATION_SIZE_LOOP})
      remove_definitions(-DBENCHMARK_ALLOCATOR=${ALLOCATOR_LOOP})
    endforeach(ALLOCATION_SIZE_LOOP)

  endforeach(TYPEDEF_LOOP)
endforeach(ALLOCATOR_LOOP)


add_custom_target(benchmarks DEPENDS
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocator_LOG
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocator_LIN
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocator_16
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocator_32
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocator_64
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocator_128
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocatorCoalesced_LOG
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocatorCoalesced_LIN
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocatorCoalesced_16
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocatorCoalesced_32
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocatorCoalesced_64
  ${WHICH_BENCHMARK}_MALLOCMC_ScatterAllocatorCoalesced_128
  ${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_LOG
  ${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_LIN
  ${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_16
  ${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_32
  ${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_64
  ${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_128
  )

add_dependencies(${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_LOG halloc)
add_dependencies(${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_LIN halloc)
add_dependencies(${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_16 halloc)
add_dependencies(${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_32 halloc)
add_dependencies(${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_64 halloc)
add_dependencies(${WHICH_BENCHMARK}_MALLOCMC_HallocAllocator_128 halloc)



