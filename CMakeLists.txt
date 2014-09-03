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


################################################################################
# VampirTrace
################################################################################
option(VAMPIR_ENABLE "Create PIConGPU with VampirTrace support" OFF)
# set filters: please do NOT use line breaks WITHIN the string!
set(VT_INST_FILE_FILTER
  "stl,usr/include,libgpugrid,vector_types.h,Vector.hpp,DeviceBuffer.hpp,DeviceBufferIntern.hpp,Buffer.hpp,StrideMapping.hpp,StrideMappingMethods.hpp,MappingDescription.hpp,AreaMapping.hpp,AreaMappingMethods.hpp,ExchangeMapping.hpp,ExchangeMappingMethods.hpp,DataSpace.hpp,Manager.hpp,Manager.tpp,Transaction.hpp,Transaction.tpp,TransactionManager.hpp,TransactionManager.tpp,Vector.tpp,Mask.hpp,ITask.hpp,EventTask.hpp,EventTask.tpp,StandartAccessor.hpp,StandartNavigator.hpp,HostBuffer.hpp,HostBufferIntern.hpp"
  CACHE STRING "VampirTrace: Files to exclude from instrumentation")
set(VT_INST_FUNC_FILTER
  "vector,Vector,dim3,GPUGrid,execute,allocator,Task,Manager,Transaction,Mask,operator,DataSpace,PitchedBox,Event,new,getGridDim,GetCurrentDataSpaces,MappingDescription,getOffset,getParticlesBuffer,getDataSpace,getInstance"
  CACHE STRING "VampirTrace: Functions to exclude from instrumentation")
if(VAMPIR_ENABLE)
  message(STATUS "Building with VampirTrace support")
  set(VAMPIR_ROOT "$ENV{VT_ROOT}")
  if(NOT VAMPIR_ROOT)
    message(FATAL_ERROR "Environment variable VT_ROOT not set!")
  endif(NOT VAMPIR_ROOT)
  # compile flags
  execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-compile
    OUTPUT_VARIABLE VT_COMPILEFLAGS
    RESULT_VARIABLE VT_CONFIG_RETURN
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT VT_CONFIG_RETURN EQUAL 0)
    message(FATAL_ERROR "Can NOT execute 'vtc++' at $ENV{VT_ROOT}/bin/vtc++ - check file permissions")
  endif()
  # link flags
  execute_process(COMMAND $ENV{VT_ROOT}/bin/vtc++ -vt:hyb -vt:showme-link
    OUTPUT_VARIABLE VT_LINKFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # bugfix showme
  string(REPLACE "--as-needed" "--no-as-needed" VT_LINKFLAGS "${VT_LINKFLAGS}")
  # modify our flags
  set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${VT_LINKFLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VT_COMPILEFLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-file-list=${VT_INST_FILE_FILTER}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions-exclude-function-list=${VT_INST_FUNC_FILTER}")
  # nvcc flags (rly necessary?)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -Xcompiler=-finstrument-functions,-finstrument-functions-exclude-file-list=\\\"${VT_INST_FILE_FILTER}\\\"
    -Xcompiler=-finstrument-functions-exclude-function-list=\\\"${VT_INST_FUNC_FILTER}\\\"
    -Xcompiler=-DVTRACE -Xcompiler=-I\\\"${VT_ROOT}/include/vampirtrace\\\"
    -v)
  # for manual instrumentation and hints that vampir is enabled in our code
  add_definitions(-DVTRACE)
  # titan work around: currently (5.14.4) the -D defines are not provided by -vt:showme-compile
  add_definitions(-DMPICH_IGNORE_CXX_SEEK)
endif(VAMPIR_ENABLE)


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

#set(WHICH_BENCHMARK "benchmark_1")
set(WHICH_BENCHMARK "benchmark_2")
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

