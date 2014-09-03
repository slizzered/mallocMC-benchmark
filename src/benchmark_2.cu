/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#define MALLOCMC 5
#define CUDAMALLOC 7
#define SCATTERALLOC 13
#define ALLOC_LOG 0
#define ALLOC_LIN 1

#define BENCHMARK_VERIFY 0

#include "print_machine_readable.hpp"
#include "dout.hpp"
#include "cmd_line.hpp"
#include "macros.hpp"

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <typeinfo>
#include <vector>
#include <string>
#include <utility>
#include <curand_kernel.h>
#include <map>
#include <algorithm>

#if BENCHMARK_ALLOCATOR == MALLOCMC
// basic files for mallocMC
#include <mallocMC/mallocMC_overwrites.hpp>
#include <mallocMC/mallocMC_utils.hpp>
#include "benchmark_1.config.hpp"
MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)
#endif


#if BENCHMARK_ALLOCATOR == SCATTERALLOC
typedef unsigned uint32;
//set the template arguments using HEAPARGS
// pagesize ... byter per page
// accessblocks ... number of superblocks
// regionsize ... number of regions for meta data structur
// wastefactor ... how much memory can be wasted per alloc (multiplicative factor)
// use_coalescing ... combine memory requests of within each warp
// resetfreedpages ... allow pages to be reused with a different size
#define HEAPARGS 4096, 8, 16, 2, true, false
#include "tools/heap_impl.cuh"
#include "tools/utils.h"
#endif






typedef std::map<int,std::map<int,std::vector<unsigned long long> > > benchmarkMap;

bool run_benchmark_2(const size_t, const unsigned, const bool, const unsigned);
std::string writeBenchmarkData();
std::string writeAveragedValues(benchmarkMap &);

std::vector<std::pair<std::string,std::string> > machine_output;


int main(int argc, char** argv){
  bool correct          = false;
  bool machine_readable = false;
  size_t heapInMB       = heapInMB_default;
  unsigned threads      = threads_default;
  unsigned blocks       = blocks_default;
  unsigned device       = 0;


  parse_cmdline(argc, argv, &heapInMB, &threads, &blocks, &machine_readable, &device);

  cudaSetDevice(device);
  cudaDeviceReset();
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  if( deviceProp.major < 2 ) {
    std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
    std::cerr << deviceProp.major << "."<< deviceProp.minor << ")" << std::endl;
    return 1;
  }

  correct = run_benchmark_2(heapInMB, threads, machine_readable, device);
  std::cerr << threads << "    " << writeBenchmarkData() << std::endl;

  cudaDeviceReset();

  if(!machine_readable || verbose){
    if(correct){
      std::cout << "\033[0;32mverification successful ✔\033[0m" << std::endl;
      return 0;
    }else{
      std::cerr << "\033[0;31mverification failed\033[0m" << std::endl;
      return 1;
    }
  }
}

__device__ int globalSuccess = 1;
__device__ int globalAllocationsInit = 0;
__device__ long long unsigned globalAllocationsContinued = 0;
__device__ long long unsigned globalFreeContinued = 0;
__device__ int globalFreeTeardown = 0;
__device__ int globalFailsInit = 0;
__device__ int globalFailsContinued = 0;
__device__ long long unsigned globalAllocClocks = 0;
__device__ long long unsigned globalFreeClocks = 0;


__global__ void cleanup_kernel(){
  printf("cleanup kernel ran successfully\n");
  globalSuccess = 1;
  globalAllocationsInit = 0;
  globalAllocationsContinued = 0;
  globalFreeContinued = 0;
  globalFreeTeardown = 0;
  globalFailsInit = 0;
  globalFailsContinued = 0;
  globalAllocClocks = 0;
  globalFreeClocks = 0;
}


void init_kernel(){
  CUDA_CHECK_KERNEL_SYNC(cleanup_kernel<<<1,1>>>());
  cudaDeviceSynchronize();
}

__global__ void getBenchmarkData(
        //int *devAllocationsInit,
        long long unsigned *devAllocationsContinued,
        long long unsigned *devFreeContinued,
        //int *devFreeTeardown,
        //int *devFailsInit,
        //int *devFailsContinued,
        long long unsigned *devAllocClocks,
        long long unsigned *devFreeClocks
        ){

        //*devAllocationsInit = globalAllocationsInit;
        *devAllocationsContinued = globalAllocationsContinued;
        *devFreeContinued = globalFreeContinued;
        //*devFreeTeardown = globalFreeTeardown;
        //*devFailsInit = globalFailsInit;
        //*devFailsContinued = globalFailsContinued;
        *devAllocClocks = globalAllocClocks;
        *devFreeClocks = globalFreeClocks;
}

__global__ void getTeardown(){
  printf("Free-operations during Teardown: %d\n",globalFreeTeardown);
  printf("Total allocations: %d\n",globalAllocationsInit+globalAllocationsContinued);
  printf("Total free:        %d\n",globalFreeContinued+globalFreeTeardown);
  printf("Average clocks per alloc: %llu\n",(long long unsigned)(globalAllocClocks/globalAllocationsContinued));
  printf("Average clocks per free : %llu\n",(long long unsigned)(globalFreeClocks/globalFreeContinued));
}
__global__ void getWarmupStats(){
  printf("Alloc-operations during Warmup: %d (%d fails)\n",globalAllocationsContinued,globalFailsContinued);
  printf("Free-operations during Warmup: %d\n",globalFreeContinued);
  printf("Average clocks per alloc Warmup: %llu\n",(long long unsigned)(globalAllocClocks/globalAllocationsContinued));
  printf("Average clocks per free Warmup: %llu\n",(long long unsigned)(globalFreeClocks/globalFreeContinued));
}
__global__ void getContinuedStats(){
  printf("Alloc-operations during run: %d (%d fails)\n",globalAllocationsContinued,globalFailsContinued);
  printf("Free-operations during run: %d\n",globalFreeContinued);
  printf("Average clocks per alloc run: %llu\n",(long long unsigned)(globalAllocClocks/globalAllocationsContinued));
  printf("Average clocks per free run: %llu\n",(long long unsigned)(globalFreeClocks/globalFreeContinued));
}

/**
 * produces a uniform distribution of values from {16,32,64,128}
 */
__device__ int getAllocSizeLinScale(const int id, curandState_t* randomState){
  //pick a number from {0,1,2,3} (uniform distribution)
  int multiplier = ceil(curand_uniform(&randomState[id])*3)-1;
  return 16 << multiplier;
}

/**
 * produces a uniform distribution of values from {16,32,64,128}
 */
//__host__ int getAllocSizeLinScale(std::default_random_engine generator, std::uniform_real_distribution<float> distribution){
__host__ int getAllocSizeLinScale(){
  //pick a number from {0,1,2,3} (uniform distribution)
  //int multiplier = distribution(generator); 
  float probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  int multiplier = probability*32;
  return 16 << multiplier%4;
}

/**
 * produces a logarithmic distribution of values from {16,32,64,128}
 * 64 is twice as likely as 128
 * 32 is twice as likely as 64
 * 16 is twice as likely as 32
 */
__device__ int getAllocSizeLogScale(const int id, curandState_t* randomState){
  //pick a number from (1,16] (uniformly distributed)
  float x = curand_uniform(&randomState[id])*15 + 1; 
  //get a number from {1,2,3,4}
  //picking 2 is 2 times more probable than picking 1
  //picking 3 is 4 times more probable than picking 1
  //picking 4 is 8 times more probable than picking 1
  int shift = ceil(log2(x));
#if BENCHMARK_VERIFY == 1
  assert(shift > 0);
  assert(shift <= 4);
#endif
  return 256 >> shift;
}

/**
 * produces a logarithmic distribution of values from {16,32,64,128}
 * 64 is twice as likely as 128
 * 32 is twice as likely as 64
 * 16 is twice as likely as 32
 */
//__host__ int getAllocSizeLogScale(std::default_random_engine generator, std::uniform_real_distribution<float> distribution){
__host__ int getAllocSizeLogScale(){
  //pick a number from [2,32) (uniformly distributed)
  //float x = distribution(generator); 
  float probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  float x = probability * 30 + 2;
  //get a number from {1,2,3,4}
  //picking 2 is 2 times more probable than picking 1
  //picking 3 is 4 times more probable than picking 1
  //picking 4 is 8 times more probable than picking 1
  int shift = floor(log2(x));
#if BENCHMARK_VERIFY == 1
  assert(shift > 0);
  assert(shift <= 4);
#endif
  return 256 >> shift;
}

//__host__ int getAllocSize(std::default_random_engine generator, std::uniform_real_distribution<float> distribution){
__host__ int getAllocSize(){
#if BENCHMARK_ALLOCATION_SIZE == ALLOC_LOG
      //return getAllocSizeLogScale(generator, distribution);
      return getAllocSizeLogScale();
#endif
#if BENCHMARK_ALLOCATION_SIZE == ALLOC_LIN
      //return getAllocSizeLinScale(generator, distribution);
      return getAllocSizeLinScale();
#endif
#if BENCHMARK_ALLOCATION_SIZE > 7
      return BENCHMARK_ALLOCATION_SIZE;
#endif
}

__device__ int getAllocSize(const int id, curandState_t* randomState){
#if BENCHMARK_ALLOCATION_SIZE == ALLOC_LOG
    return getAllocSizeLogScale(id, randomState);
#endif
#if BENCHMARK_ALLOCATION_SIZE == ALLOC_LIN
    return getAllocSizeLinScale(id, randomState);
#endif
#if BENCHMARK_ALLOCATION_SIZE > 7
  return BENCHMARK_ALLOCATION_SIZE;
#endif
}

__device__ void* allocUnderTest(size_t size,long long unsigned* duration){
  long long unsigned start_time = clock64();
#if BENCHMARK_ALLOCATOR == MALLOCMC
    void* p = mallocMC::malloc(size);
#endif
#if BENCHMARK_ALLOCATOR == CUDAMALLOC
    void* p = malloc(size);
#endif
#if BENCHMARK_ALLOCATOR == SCATTERALLOC
    void* p = theHeap.alloc(size);
#endif
  long long unsigned stop_time = clock64();
  *duration = stop_time-start_time;
  return p;
}

__device__ void freeUnderTest(void* p,long long unsigned* duration){
  long long unsigned start_time = clock64();
#if BENCHMARK_ALLOCATOR == MALLOCMC
    mallocMC::free(p);
#endif
#if BENCHMARK_ALLOCATOR == CUDAMALLOC
    free(p);
#endif
#if BENCHMARK_ALLOCATOR == SCATTERALLOC
    theHeap.dealloc(p);
#endif
  long long unsigned stop_time = clock64();
  *duration = stop_time-start_time;
}



__device__ int* testRequest(int id, int alloc_size, int* p){
#if BENCHMARK_ALLOCATOR == MALLOCMC
#if BENCHMARK_VERIFY == 1
  if(p==NULL){
    int slotsRemaining = mallocMC::getAvailableSlots(alloc_size);
    if(slotsRemaining>10){
      printf("thread %d wants to allocate %d bytes (%u slots remaining), but did NOT get anything!\n",
          id, alloc_size, slotsRemaining);
      atomicAnd(&globalSuccess,0);
    }
  }
#endif
#endif
  return p;
}

__global__ void createPointerStorageInThreads(
    int*** pointerStore,
    size_t maxPointersPerThread,
    int desiredThreads,
    int* fillLevelsPerThread,
    int* pointersPerThread
    ){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;
  int** p = (int**) malloc(sizeof(int*) * maxPointersPerThread);
  if(p == NULL) atomicAnd(&globalSuccess, 0);
  pointerStore[id] = p;
  fillLevelsPerThread[id] = 0;
  pointersPerThread[id] = 0;
  //curand_init(seed, id, 0, &randomState[id]);
}

__global__ void allocKernel(
    int*** pointerStore,
    int maxBytesPerThread,
    int desiredThreads,
    int* fillLevelsPerThread,
    int* pointersPerThread,
    const int alloc_size
    ){
    
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;
  if(fillLevelsPerThread[id]+128 >= maxBytesPerThread) return;

  long long unsigned duration=0llu;
  int * p = (int*) allocUnderTest(alloc_size, &duration);

  if(p == NULL){
    atomicAdd(&globalFailsInit,1);
  }
  else{
    p[0] = alloc_size;
    fillLevelsPerThread[id] += alloc_size;
    pointerStore[id][pointersPerThread[id]++] = p;
    atomicAdd(&globalAllocationsContinued,1);
    atomicAdd(&globalAllocClocks, duration);
  }
}


__global__ void freeKernel(
    int*** pointerStore,
    int desiredThreads,
    int* fillLevelsPerThread,
    int* pointersPerThread,
    const int alloc_size
    ){
    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;
  if(pointersPerThread[id] == 0) return;

  int free_size = pointerStore[id][--pointersPerThread[id]][0];
  //int free_size = alloc_size;
  long long unsigned duration=0llu;

  freeUnderTest(pointerStore[id][pointersPerThread[id]],&duration);

  fillLevelsPerThread[id] -= free_size;
  atomicAdd(&globalFreeContinued,1llu);
  atomicAdd(&globalFreeClocks, duration);
}



__global__ void deallocAll(
    int*** pointerStore,
    int desiredThreads,
    int* fillLevelsPerThread,
    int* pointersPerThread
    ){
    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;

  int fillLevel = fillLevelsPerThread[id];
  int** pointerStoreReg = pointerStore[id];
  int pointersPerThreadReg = pointersPerThread[id];


  while(pointersPerThreadReg > 0) { 
    int free_size = pointerStoreReg[--pointersPerThreadReg][0];
    long long unsigned duration=0llu;
    freeUnderTest(pointerStoreReg[pointersPerThreadReg],&duration);
    fillLevel -= free_size;
    atomicAdd(&globalFreeTeardown,1);
  }  

  free(pointerStore[id]);
}

__global__ void getSuccessState(int* success){
  printf("Allocations done during initialization: %d (%d times, no memory was available)\n",
      globalAllocationsInit,globalFailsInit);
  printf("Allocations done while running: %llu (%d times, no memory was available)\n",
      globalAllocationsContinued,globalFailsContinued);
  printf("Free-operations done while running: %llu\n",
      globalFreeContinued);
  success[0] = globalSuccess;
}

std::string writeBenchmarkData(){
  //int hostAllocationsInit = 0;
  long long unsigned hostAllocationsContinued = 0;
  long long unsigned hostFreeContinued = 0;
  //int hostFreeTeardown = 0;
  //int hostFailsInit = 0;
  //int hostFailsContinued = 0;
  long long unsigned hostAllocClocks = 0;
  long long unsigned hostFreeClocks = 0;
  //int *devAllocationsInit;
  long long unsigned *devAllocationsContinued;
  long long unsigned *devFreeContinued;
  //int *devFreeTeardown;
  //int *devFailsInit;
  //int *devFailsContinued;
  long long unsigned *devAllocClocks;
  long long unsigned *devFreeClocks;

  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devAllocationsContinued,sizeof(long long unsigned)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devFreeContinued,sizeof(long long unsigned)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devAllocClocks,sizeof(long long unsigned)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devFreeClocks,sizeof(long long unsigned)));
  CUDA_CHECK_KERNEL_SYNC(getBenchmarkData<<<1,1>>>(
        //devAllocationsInit,
        devAllocationsContinued,
        devFreeContinued,
        //devFreeTeardown,
        //devFailsInit,
        //devFailsContinued,
        devAllocClocks,
        devFreeClocks
        ));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostAllocationsContinued,devAllocationsContinued,sizeof(long long unsigned),cudaMemcpyDeviceToHost));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostFreeContinued,devFreeContinued,sizeof(long long unsigned),cudaMemcpyDeviceToHost));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostAllocClocks,devAllocClocks,sizeof(long long unsigned),cudaMemcpyDeviceToHost));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostFreeClocks,devFreeClocks,sizeof(long long unsigned),cudaMemcpyDeviceToHost));

  std::stringstream ss;
  ss << hostAllocClocks/hostAllocationsContinued << "    ";
  ss << hostFreeClocks/hostFreeContinued;

  return ss.str();
}


/**
 * Verify the heap allocation of mallocMC
 *
 * Allocates as much memory as the heap allows. Make sure that allocated
 * memory actually holds the correct values without corrupting them. Will
 * fill the datastructure with values that are relative to the index and
 * later evalute, if the values inside stayed the same after allocating all
 * memory.
 * Datastructure: Array that holds up to nPointers pointers to arrays of size
 * ELEMS_PER_SLOT, each being of type allocElem_t.
 *
 * @return true if the verification was successful,
 *         false otherwise
 */
bool run_benchmark_2(
    const size_t heapMB,
    const unsigned desiredThreads,
    const bool machine_readable,
    const unsigned device
    ){

  int h_globalSuccess=0;
  //std::default_random_engine generator;
  //std::uniform_real_distribution<float> distribution(2,32);

  init_kernel();
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  unsigned maxBlocksPerSM = 8;
  if(deviceProp.major > 2) maxBlocksPerSM *= 2; //set to 16 for 3.0 and higher
  if(deviceProp.major >= 5) maxBlocksPerSM *= 2; //increase again to 32 for 5.0 and higher

  //use the smallest possible blocksize that is still able to fill the multiprocessor
  const size_t threadsUsedInBlock = deviceProp.maxThreadsPerMultiProcessor / maxBlocksPerSM;
  const size_t maxUsefulBlocks = maxBlocksPerSM * deviceProp.multiProcessorCount;
  dout() << "threadsUsedInBlock: " << threadsUsedInBlock << std::endl;
  dout() << "maxUsefulBlocks:    " << maxUsefulBlocks << std::endl;
  dout() << "Clock Frequency:    " << deviceProp.clockRate/1000.0 << "MHz" << std::endl;
  
  const unsigned threads = threadsUsedInBlock;
  const unsigned blocks  = maxUsefulBlocks;

  const size_t usableMemoryMB   = deviceProp.totalGlobalMem / size_t(1024U * 1024U);
  if(heapMB > usableMemoryMB/2) dout() << "Warning: heapSize fills more than 50% of global Memory" << std::endl;

  const size_t heapSize         = size_t(1024U*1024U) * heapMB;
  machine_output.push_back(MK_STRINGPAIR(heapSize));

  //if a single thread allocates only the minimal chunksize, it can not exceed this number
  size_t maxStoredChunks = heapMB * size_t(1024U * 1024U) / size_t(16U);
  size_t maxMemPerThread = heapMB * size_t(1024U * 1024U) / desiredThreads;
  int maxChunksPerThread = maxMemPerThread / 16;
  int maxChunksTotal = maxChunksPerThread * desiredThreads;

  int*** pointerStoreForThreads;
  int* fillLevelsPerThread;
  int* pointersPerThread;
  curandState_t* randomState;
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &pointerStoreForThreads, desiredThreads*sizeof(int**)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &fillLevelsPerThread, desiredThreads*sizeof(int)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &pointersPerThread, desiredThreads*sizeof(int)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &randomState, desiredThreads * sizeof(curandState_t)));



  //dout() << "maxStoredChunks: " << maxStoredChunks << std::endl;
  size_t pointerSize = maxChunksTotal*sizeof(int**)*4;
  dout() << "necessary memory for pointers: " << pointerSize << std::endl;
  dout() << "reserved Heapsize:             " << heapSize << std::endl;

#if BENCHMARK_ALLOCATOR == MALLOCMC
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, pointerSize);
    mallocMC::initHeap(heapSize);
#endif
#if BENCHMARK_ALLOCATOR == CUDAMALLOC
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, pointerSize + heapSize);
#endif
#if BENCHMARK_ALLOCATOR == SCATTERALLOC
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, pointerSize);
    initHeap(heapSize);
#endif

  size_t maxPointersPerThread = ceil(float(maxStoredChunks)/desiredThreads);
  CUDA_CHECK_KERNEL_SYNC(createPointerStorageInThreads<<<blocks,threads>>>(
      pointerStoreForThreads,
      maxChunksPerThread,
      desiredThreads,
      fillLevelsPerThread,
      pointersPerThread
      ));

  
#if BENCHMARK_ALLOCATOR != CUDAMALLOC
    for(int i=16;i<256;i = i << 1){
      if(BENCHMARK_ALLOCATION_SIZE > 1) i = BENCHMARK_ALLOCATION_SIZE;
#if BENCHMARK_ALLOCATOR == MALLOCMC
      dout() << "before warmup: free slots of size " << i << ": " << mallocMC::getAvailableSlots(i) << std::endl;
#else 
      dout() << "before warmup: free slots of size " << i << ": " << getAvailableSlotsHost(i) << std::endl;
#endif
      if(BENCHMARK_ALLOCATION_SIZE > 1) break;
    }
#endif

    srand(31415);
    for(int warm_i=0; warm_i < 50000; ++warm_i){
      float probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      //int alloc_size = getAllocSize(generator,distribution);
      int alloc_size = getAllocSize();
      if(probability <= 0.75){
        CUDA_CHECK_KERNEL_SYNC(allocKernel<<<blocks,threads>>>(
              pointerStoreForThreads,
              maxMemPerThread,
              desiredThreads,
              fillLevelsPerThread,
              pointersPerThread,
              alloc_size
              ));
      }

      probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      if(probability <= 0.75){
        CUDA_CHECK_KERNEL_SYNC(freeKernel<<<blocks,threads>>>(
              pointerStoreForThreads,
              desiredThreads,
              fillLevelsPerThread,
              pointersPerThread,
              alloc_size
              ));

      }
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_KERNEL_SYNC(getWarmupStats<<<1,1>>>());
    CUDA_CHECK_KERNEL_SYNC(cleanup_kernel<<<1,1>>>());
    dout() << "WARMUP COMPLETE" << std::endl;

    for(int cont_i=0; cont_i < 50000; ++cont_i){
      float probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      //int alloc_size = getAllocSize(generator,distribution);
      int alloc_size = getAllocSize();
      if(probability <= 0.75){
        CUDA_CHECK_KERNEL_SYNC(allocKernel<<<blocks,threads>>>(
              pointerStoreForThreads,
              maxMemPerThread,
              desiredThreads,
              fillLevelsPerThread,
              pointersPerThread,
              alloc_size
              ));
      }

      probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      if(probability <= 0.75){
        CUDA_CHECK_KERNEL_SYNC(freeKernel<<<blocks,threads>>>(
              pointerStoreForThreads,
              desiredThreads,
              fillLevelsPerThread,
              pointersPerThread,
              alloc_size
              ));
      }
    }


  int* d_success;
  cudaMalloc((void**) &d_success,sizeof(int));
  getSuccessState<<<1,1>>>(d_success);
  BENCHMARK_CHECKED_CALL(cudaMemcpy((void*) &h_globalSuccess,d_success, sizeof(int), cudaMemcpyDeviceToHost));
  machine_output.push_back(MK_STRINGPAIR(h_globalSuccess));
//  print_machine_readable(machine_output);

  // release all memory
  CUDA_CHECK_KERNEL_SYNC(deallocAll<<<blocks,threads>>>(
      pointerStoreForThreads,
      desiredThreads,
      fillLevelsPerThread,
      pointersPerThread
      ));
  CUDA_CHECK_KERNEL_SYNC(getTeardown<<<1,1>>>());
  cudaDeviceSynchronize();

#if BENCHMARK_ALLOCATOR != CUDAMALLOC
    for(int i=16;i<256;i = i << 1){
      if(BENCHMARK_ALLOCATION_SIZE > 1) i = BENCHMARK_ALLOCATION_SIZE;
#if BENCHMARK_ALLOCATOR == MALLOCMC
      dout() << "after filling: free slots of size " << i << ": " << mallocMC::getAvailableSlots(i) << std::endl;
#else 
      dout() << "after filling: free slots of size " << i << ": " << getAvailableSlotsHost(i) << std::endl;
#endif
      if(BENCHMARK_ALLOCATION_SIZE > 1) break;
    }
#endif



#if BENCHMARK_ALLOCATOR == MALLOCMC
    h_globalSuccess = h_globalSuccess && (mallocMC::getAvailableSlots(16)==1036320);
    mallocMC::finalizeHeap();
#endif
#if BENCHMARK_ALLOCATOR == SCATTERALLOC
    h_globalSuccess = h_globalSuccess && (getAvailableSlotsHost(16)==1036320);
#endif
  cudaFree(d_success);
  cudaFree(pointerStoreForThreads);
  cudaFree(fillLevelsPerThread);
  cudaFree(pointersPerThread);
  cudaFree(randomState);
  dout() << "done "<< std::endl;

  return h_globalSuccess;
}
