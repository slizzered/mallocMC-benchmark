project(mallocMC)
cmake_minimum_required(VERSION 2.8.5)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")


###############################################################################
# CUDA
###############################################################################
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-arch=sm_20;-use_fast_math;")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mallocMC/src/include)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/scatteralloc)

OPTION(CUDA_OUTPUT_INTERMEDIATE_CODE "Output ptx code" OFF)
if(CUDA_OUTPUT_INTERMEDIATE_CODE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas;-v;--keep")
endif(CUDA_OUTPUT_INTERMEDIATE_CODE)

SET(CUDA_OPTIMIZATION_TYPE "unset" CACHE STRING "CUDA Optimization")
set_property(CACHE CUDA_OPTIMIZATION_TYPE PROPERTY STRINGS "unset;-G0;-O0;-O1;-O2;-O3")
if(NOT ${CUDA_OPTIMIZATION_TYPE} STREQUAL  "unset")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_OPTIMIZATION_TYPE}")
endif()


###############################################################################
# Boost
###############################################################################
find_package(Boost 1.48.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)







###############################################################################
# Executables
###############################################################################

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*") 

option(ALLOC_SIZE OFF)

if(ALLOC_SIZE)
  set(ALLOC_SIZE_FIXED ${ALLOC_SIZE})
else(ALLOC_SIZE)
  set(ALLOC_SIZE_FIXED 16) 
endif(ALLOC_SIZE)


###############################################################################
# Benchmarks with MALLOCMC
###############################################################################
add_definitions(-DBENCHMARK_ALLOCATOR=MALLOCMC)

# logarithmic distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)
cuda_add_executable(benchmark_1_mallocMC_log ${SRCFILES})
target_link_libraries(benchmark_1_mallocMC_log ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)

# linear distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)
cuda_add_executable(benchmark_1_mallocMC_lin EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_mallocMC_lin ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)

# fixed distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})
cuda_add_executable(benchmark_1_mallocMC_${ALLOC_SIZE_FIXED} EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_mallocMC_${ALLOC_SIZE_FIXED} ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})

remove_definitions(-DBENCHMARK_ALLOCATOR=MALLOCMC)



###############################################################################
# Benchmarks with CUDA Device Malloc
###############################################################################
add_definitions(-DBENCHMARK_ALLOCATOR=CUDAMALLOC)

# logarithmic distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)
cuda_add_executable(benchmark_1_deviceMalloc_log EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_deviceMalloc_log ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)

# linear distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)
cuda_add_executable(benchmark_1_deviceMalloc_lin EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_deviceMalloc_lin ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)

# fixed distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})
cuda_add_executable(benchmark_1_deviceMalloc_${ALLOC_SIZE_FIXED} EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_deviceMalloc_${ALLOC_SIZE_FIXED} ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})

remove_definitions(-DBENCHMARK_ALLOCATOR=CUDAMALLOC)


###############################################################################
# Benchmarks with CUDA ScatterAlloc
###############################################################################
add_definitions(-DBENCHMARK_ALLOCATOR=SCATTERALLOC)

# logarithmic distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)
cuda_add_executable(benchmark_1_scatterAlloc_log EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_scatterAlloc_log ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LOG)

# linear distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)
cuda_add_executable(benchmark_1_scatterAlloc_lin EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_scatterAlloc_lin ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=ALLOC_LIN)

# fixed distribution
add_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})
cuda_add_executable(benchmark_1_scatterAlloc_${ALLOC_SIZE_FIXED} EXCLUDE_FROM_ALL ${SRCFILES})
target_link_libraries(benchmark_1_scatterAlloc_${ALLOC_SIZE_FIXED} ${LIBS})
remove_definitions(-DBENCHMARK_ALLOCATION_SIZE=${ALLOC_SIZE_FIXED})

remove_definitions(-DBENCHMARK_ALLOCATOR=SCATTERALLOC)


add_custom_target(benchmarks DEPENDS
  benchmark_1_mallocMC_log
  benchmark_1_mallocMC_lin
  benchmark_1_mallocMC_${ALLOC_SIZE_FIXED}
  benchmark_1_deviceMalloc_log
  benchmark_1_deviceMalloc_lin
  benchmark_1_deviceMalloc_${ALLOC_SIZE_FIXED}
  benchmark_1_scatterAlloc_log
  benchmark_1_scatterAlloc_lin
  benchmark_1_scatterAlloc_${ALLOC_SIZE_FIXED}
  )
