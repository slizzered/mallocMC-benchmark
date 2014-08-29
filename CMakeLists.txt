project(mallocMC)
cmake_minimum_required(VERSION 2.8.5)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")


###############################################################################
# CUDA
###############################################################################
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-arch=sm_35;-use_fast_math;")
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







###############################################################################
# Executables
###############################################################################

set(WHICH_BENCHMARK "benchmark_1")
#set(WHICH_BENCHMARK "benchmark_2")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/${WHICH_BENCHMARK}.c*")

option(ALLOC_SIZE OFF)

if(ALLOC_SIZE)
  set(ALLOC_SIZE_FIXED ${ALLOC_SIZE})
else(ALLOC_SIZE)
  set(ALLOC_SIZE_FIXED 16) 
endif(ALLOC_SIZE)

add_executable(result_parser "${CMAKE_CURRENT_SOURCE_DIR}/src/parse_results.cpp")
target_link_libraries(result_parser ${LIBS})


###############################################################################
# Benchmarks with MALLOCMC
###############################################################################
add_definitions(-DBENCHMARK_ALLOCATOR=MALLOCMC)

# logarithmic distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)
cuda_add_executable(${WHICH_BENCHMARK}_mallocMC_log ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_mallocMC_log ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)

# linear distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)
cuda_add_executable(${WHICH_BENCHMARK}_mallocMC_lin EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_mallocMC_lin ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)

# fixed distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})
cuda_add_executable(${WHICH_BENCHMARK}_mallocMC_${ALLOC_SIZE_FIXED} EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_mallocMC_${ALLOC_SIZE_FIXED} ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})

remove_definitions(-DBENCHMARK_ALLOCATOR=MALLOCMC)



###############################################################################
# Benchmarks with CUDA Device Malloc
###############################################################################
add_definitions(-DBENCHMARK_ALLOCATOR=CUDAMALLOC)

# logarithmic distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)
cuda_add_executable(${WHICH_BENCHMARK}_deviceMalloc_log EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_deviceMalloc_log ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)

# linear distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)
cuda_add_executable(${WHICH_BENCHMARK}_deviceMalloc_lin EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_deviceMalloc_lin ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)

# fixed distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})
cuda_add_executable(${WHICH_BENCHMARK}_deviceMalloc_${ALLOC_SIZE_FIXED} EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_deviceMalloc_${ALLOC_SIZE_FIXED} ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})

remove_definitions(-DBENCHMARK_ALLOCATOR=CUDAMALLOC)


###############################################################################
# Benchmarks with CUDA ScatterAlloc
###############################################################################
add_definitions(-DBENCHMARK_ALLOCATOR=SCATTERALLOC)

# logarithmic distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)
cuda_add_executable(${WHICH_BENCHMARK}_scatterAlloc_log EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_scatterAlloc_log ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)

# linear distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)
cuda_add_executable(${WHICH_BENCHMARK}_scatterAlloc_lin EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_scatterAlloc_lin ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)

# fixed distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})
cuda_add_executable(${WHICH_BENCHMARK}_scatterAlloc_${ALLOC_SIZE_FIXED} EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(${WHICH_BENCHMARK}_scatterAlloc_${ALLOC_SIZE_FIXED} ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})

remove_definitions(-DBENCHMARK_ALLOCATOR=SCATTERALLOC)


add_custom_target(benchmarks DEPENDS
  ${WHICH_BENCHMARK}_mallocMC_log
  ${WHICH_BENCHMARK}_mallocMC_lin
  ${WHICH_BENCHMARK}_mallocMC_${ALLOC_SIZE_FIXED}
  ${WHICH_BENCHMARK}_deviceMalloc_log
  ${WHICH_BENCHMARK}_deviceMalloc_lin
  ${WHICH_BENCHMARK}_deviceMalloc_${ALLOC_SIZE_FIXED}
  ${WHICH_BENCHMARK}_scatterAlloc_log
  ${WHICH_BENCHMARK}_scatterAlloc_lin
  ${WHICH_BENCHMARK}_scatterAlloc_${ALLOC_SIZE_FIXED}
  )

add_custom_target(compare DEPENDS
  ${WHICH_BENCHMARK}_mallocMC_${ALLOC_SIZE_FIXED}
  ${WHICH_BENCHMARK}_scatterAlloc_${ALLOC_SIZE_FIXED}
  )

