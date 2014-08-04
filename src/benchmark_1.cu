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

#include "print_machine_readable.hpp"
#include "dout.hpp"
#include "benchmark_1.config.hpp"
#include "cmd_line.hpp"
#include "macros.hpp"

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <typeinfo>
#include <vector>
#include <string>
#include <utility>
#include <curand_kernel.h>

// basic files for mallocMC
#include <mallocMC/mallocMC_overwrites.hpp>
#include <mallocMC/mallocMC_utils.hpp>


#define MALLOCMC 5
#define CUDAMALLOC 7

#define ALLOC_LOG 0
#define ALLOC_LIN 1

#define BENCHMARK_VERIFY 1

MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)



bool run_benchmark_1(const size_t, const unsigned, const bool);


std::vector<std::pair<std::string,std::string> > machine_output;

int main(int argc, char** argv){
  bool correct          = false;
  bool machine_readable = false;
  size_t heapInMB       = heapInMB_default;
  unsigned threads      = threads_default;
  unsigned blocks       = blocks_default;


  parse_cmdline(argc, argv, &heapInMB, &threads, &blocks, &machine_readable);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  if( deviceProp.major < 2 ) {
    std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
    std::cerr << deviceProp.major << "."<< deviceProp.minor << ")" << std::endl;
    return 1;
  }

  cudaSetDevice(0);

  machine_output.push_back(MK_STRINGPAIR(deviceProp.major));
  machine_output.push_back(MK_STRINGPAIR(deviceProp.minor));
  machine_output.push_back(MK_STRINGPAIR(deviceProp.name));
  machine_output.push_back(MK_STRINGPAIR(deviceProp.totalGlobalMem));



  correct = run_benchmark_1(heapInMB, threads, machine_readable);

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
__device__ int globalAllocationsContinued = 0;
__device__ int globalFreeContinued = 0;
__device__ int globalFreeTeardown = 0;
__device__ int globalFailsInit = 0;
__device__ int globalFailsContinued = 0;

__global__ void dummy_kernel(){
  printf("dummy kernel ran successfully\n");
}


void init_kernel(unsigned threads, unsigned blocks){
  CUDA_CHECK_KERNEL_SYNC(dummy_kernel<<<1,1>>>());
  cudaDeviceSynchronize();
}

__global__ void getTeardown(){
  printf("Free-operations during Teardown: %d\n",globalFreeTeardown);
  printf("Total allocations: %d\n",globalAllocationsInit+globalAllocationsContinued);
  printf("Total free:        %d\n",globalFreeContinued+globalFreeTeardown);
}

/**
 * produces a uniform distribution of values from {16,32,64,128}
 */
__device__ int getAllocSizeLinScale(const int id, curandState_t* randomState){
  //pick a number from {0,1,2,3} (uniform distribution)
  int multiplier = ceil(curand_uniform(&randomState[id])*4)-1;
  return 16 << multiplier;
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
#ifdef BENCHMARK_VERIFY
  assert(shift > 0);
  assert(shift <= 4);
#endif
  return 256 >> shift;
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

__device__ void* allocUnderTest(size_t size){
#if BENCHMARK_ALLOCATOR == MALLOCMC
    return mallocMC::malloc(size);
#endif
#if BENCHMARK_ALLOCATOR == CUDAMALLOC
    return malloc(size);
#endif
}

__device__ void freeUnderTest(void* p){
#if BENCHMARK_ALLOCATOR == MALLOCMC
    mallocMC::free(p);
#endif
#if BENCHMARK_ALLOCATOR == CUDAMALLOC
    free(p);
#endif
}

__device__ int* testRequestInit(int id, int alloc_size, int* p){
  //if(p==NULL){
  //  int slotsRemaining = mallocMC::getAvailableSlots(alloc_size);
  //  if(slotsRemaining>0){
  //    printf("Init: thread %d wants to allocate %d bytes (%d slots remaining), but did NOT get anything!\n",
  //        id, alloc_size, slotsRemaining);
  //    atomicAnd(&globalSuccess,0);
  //  }
  //}
  return p;
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
    int* pointersPerThread,
    curandState_t* randomState,
    int seed
    ){
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;
  int** p = (int**) malloc(sizeof(int*) * maxPointersPerThread);
  if(p == NULL) atomicAnd(&globalSuccess, 0);
  pointerStore[id] = p;
  fillLevelsPerThread[id] = 0;
  pointersPerThread[id] = 0;
  curand_init(seed, id, 0, &randomState[id]);
  
}


__global__ void initialFillingOfPointerStorage(
    int*** pointerStore,
    int maxBytesPerThread,
    int desiredThreads,
    int* fillLevelsPerThread,
    int* pointersPerThread,
    curandState_t* randomState
    ){
    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;

  int fillLevel = fillLevelsPerThread[id];
  int** pointerStoreReg = pointerStore[id];
  int pointersPerThreadReg = pointersPerThread[id];

  while(true){

    const float fillLevelPercent = float(fillLevel) / maxBytesPerThread;
    const int alloc_size = getAllocSize(id, randomState);

    if(fillLevel+128 >= maxBytesPerThread) break;
    if(fillLevelPercent >= 0.75) break;
    //if(mallocMC::getAvailableSlots(alloc_size) < 1) break;

    int * p = (int*) allocUnderTest(alloc_size);
    if(testRequestInit(id,alloc_size,p) == NULL){
      atomicAdd(&globalFailsInit,1);
      break;
    }
    else{
      p[0] = alloc_size;
      fillLevel += alloc_size;
      pointerStoreReg[pointersPerThreadReg++] = p;
      atomicAdd(&globalAllocationsInit,1);

    }
  }

  //printf("Thread %d  fillLevel: %d byte (%.0f%%) maxBytesPerThread: %d allocatedElements: %d\n",id, fillLevel, 100*float(fillLevel)/maxBytesPerThread,maxBytesPerThread, pointersPerThreadReg);

  fillLevelsPerThread[id] = fillLevel;
  pointerStore[id] = pointerStoreReg;
  pointersPerThread[id] = pointersPerThreadReg;

}

__global__ void continuedFillingOfPointerStorage(
    int*** pointerStore,
    int maxBytesPerThread,
    int desiredThreads,
    int* fillLevelsPerThread,
    int* pointersPerThread,
    curandState_t* randomState
    ){
    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;

  int fillLevel = fillLevelsPerThread[id];
  int** pointerStoreReg = pointerStore[id];
  int pointersPerThreadReg = pointersPerThread[id];
  int counter = 0;


  while(counter < 100){
    ++counter;
    float probability = min(curand_uniform(&randomState[id])+0.25,1.);
    //float probability = max(curand_uniform(&randomState[id]), curand_uniform(&randomState[id]));
    float fillLevelPercent = float(fillLevel) / maxBytesPerThread;
    //probably, the fill level is higher than 75% -> deallocate
    
    if(pointersPerThreadReg > 0 && (probability <= fillLevelPercent)) {
        //printf("thread %d wants to free %d bytes of memory\n",id, free_size);
        int free_size = pointerStoreReg[--pointersPerThreadReg][0];
        freeUnderTest(pointerStoreReg[pointersPerThreadReg]);
        fillLevel -= free_size;
#ifdef BENCHMARK_VERIFY
        atomicAdd(&globalFreeContinued,1);
#endif
    }else{
      const int alloc_size = getAllocSize(id, randomState);

      if(fillLevel+128 <= maxBytesPerThread){ 
        //printf("thread %d wants to allocate %d bytes of memory\n",id, alloc_size);
        int* p = (int*) allocUnderTest(alloc_size);
        if(testRequestInit(id, alloc_size, p) == NULL){
          atomicAdd(&globalFailsContinued,1);
        }
        else{
          p[0] = alloc_size;
          fillLevel += alloc_size;
          pointerStoreReg[pointersPerThreadReg++] = p;
#ifdef BENCHMARK_VERIFY
          atomicAdd(&globalAllocationsContinued,1);
#endif
        }
      }
    }
  }

  fillLevelsPerThread[id] = fillLevel;
  pointerStore[id] = pointerStoreReg;
  pointersPerThread[id] = pointersPerThreadReg;
  //printf("Thread %d  fillLevel: %d byte (%.0f%%) maxBytesPerThread: %d allocatedElements: %d\n",id, fillLevel, 100*float(fillLevel)/maxBytesPerThread,maxBytesPerThread, pointersPerThreadReg);
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
    freeUnderTest(pointerStoreReg[pointersPerThreadReg]);
    fillLevel -= free_size;
    atomicAdd(&globalFreeTeardown,1);
  }  

  free(pointerStore[id]);
}

__global__ void getSuccessState(int* success){
  printf("Allocations done during initialization: %d (%d times, no memory was available)\n",
      globalAllocationsInit,globalFailsInit);
  printf("Allocations done while running: %d (%d times, no memory was available)\n",
      globalAllocationsContinued,globalFailsContinued);
  printf("Free-operations done while running: %d\n",
      globalFreeContinued);
  success[0] = globalSuccess;
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
bool run_benchmark_1(
    const size_t heapMB,
    const unsigned desiredThreads,
    const bool machine_readable
    ){

  int h_globalSuccess=0;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  unsigned maxBlocksPerSM = 8;
  if(deviceProp.major > 2) maxBlocksPerSM *= 2; //set to 16 for 3.0 and higher
  if(deviceProp.major >= 5) maxBlocksPerSM *= 2; //increase again to 32 for 5.0 and higher

  //use the smallest possible blocksize that is still able to fill the multiprocessor
  const size_t threadsUsedInBlock = deviceProp.maxThreadsPerMultiProcessor / maxBlocksPerSM;
  const size_t maxUsefulBlocks = maxBlocksPerSM * deviceProp.multiProcessorCount;
  dout() << "threadsUsedInBlock: " << threadsUsedInBlock << std::endl;
  dout() << "maxUsefulBlocks: " << maxUsefulBlocks << std::endl;
  
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
  size_t maxMemTotal = maxMemPerThread * desiredThreads;

  int*** pointerStoreForThreads;
  int* fillLevelsPerThread;
  int* pointersPerThread;
  curandState_t* randomState;
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &pointerStoreForThreads, desiredThreads*sizeof(int**)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &fillLevelsPerThread, desiredThreads*sizeof(int)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &pointersPerThread, desiredThreads*sizeof(int)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &randomState, desiredThreads * sizeof(curandState_t)));

  // initializing the heap
  if(BENCHMARK_ALLOCATOR == MALLOCMC){
    mallocMC::initHeap(heapSize);
  }


  //dout() << "maxStoredChunks: " << maxStoredChunks << std::endl;
  size_t completeStorage = maxMemTotal + maxChunksTotal*sizeof(int*);
  dout() << "necessary memory for malloc on device: " << completeStorage << std::endl;
  dout() << "mallocMC Heapsize:                     " << heapSize << std::endl;

  cudaDeviceSetLimit(cudaLimitMallocHeapSize,completeStorage*2);

  size_t maxPointersPerThread = ceil(float(maxStoredChunks)/desiredThreads);
  CUDA_CHECK_KERNEL_SYNC(createPointerStorageInThreads<<<blocks,threads>>>(
      pointerStoreForThreads,
      maxChunksPerThread,
      desiredThreads,
      fillLevelsPerThread,
      pointersPerThread,
      randomState,
      42
      ));

  
  if(BENCHMARK_ALLOCATOR == MALLOCMC){
    for(int i=16;i<256;i = i << 1){
      if(BENCHMARK_ALLOCATION_SIZE > 1) i = BENCHMARK_ALLOCATION_SIZE;
      dout() << "before filling: free slots of size " << i << ": " << mallocMC::getAvailableSlots(i) << std::endl;
      if(BENCHMARK_ALLOCATION_SIZE > 1) break;
    }
  }
  //each thread can handle up to ceil(float(maxStoredChunks)/desiredThreads)
  //pointers. 
  //However, if the chunks are bigger than 16byte, the heap-size is far more limiting.
  // It is therefore preferable to supply the maximum allowed memory for each
  // thread and assume, that the thread will not exceed it's pointer-space
  CUDA_CHECK_KERNEL_SYNC(initialFillingOfPointerStorage<<<blocks,threads>>>(
      pointerStoreForThreads,
      maxMemPerThread,
      desiredThreads,
      fillLevelsPerThread,
      pointersPerThread,
      randomState
      ));
  cudaDeviceSynchronize();

  if(BENCHMARK_ALLOCATOR == MALLOCMC){
    for(int i=16;i<256;i = i << 1){
      if(BENCHMARK_ALLOCATION_SIZE > 1) i = BENCHMARK_ALLOCATION_SIZE;
      //dout() << "after filling: free slots of size " << i << ": " << mallocMC::getAvailableSlots(i) << std::endl;
      if(BENCHMARK_ALLOCATION_SIZE > 1) break;
    }
  }

  CUDA_CHECK_KERNEL_SYNC(continuedFillingOfPointerStorage<<<blocks,threads>>>(
      pointerStoreForThreads,
      maxMemPerThread,
      desiredThreads,
      fillLevelsPerThread,
      pointersPerThread,
      randomState
      ));
  cudaDeviceSynchronize();


  dout() << "FILLING COMPLETE" << std::endl;


  if(BENCHMARK_ALLOCATOR == MALLOCMC){
    for(int i=16;i<256;i = i << 1){
      if(BENCHMARK_ALLOCATION_SIZE > 1) i = BENCHMARK_ALLOCATION_SIZE;
      //dout() << "after continous run: free slots of size " << i << ": " << mallocMC::getAvailableSlots(i) << std::endl;
      if(BENCHMARK_ALLOCATION_SIZE > 1) break;
    }
  }


  machine_output.push_back(MK_STRINGPAIR(desiredThreads));
  machine_output.push_back(MK_STRINGPAIR(ScatterConfig::pagesize::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterConfig::accessblocks::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterConfig::regionsize::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterConfig::wastefactor::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterConfig::resetfreedpages::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterHashParams::hashingK::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterHashParams::hashingDistMP::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterHashParams::hashingDistWP::value));
  machine_output.push_back(MK_STRINGPAIR(ScatterHashParams::hashingDistWPRel::value));

  int* d_success;
  cudaMalloc((void**) &d_success,sizeof(int));
  getSuccessState<<<1,1>>>(d_success);
  MALLOCMC_CUDA_CHECKED_CALL(cudaMemcpy((void*) &h_globalSuccess,d_success, sizeof(int), cudaMemcpyDeviceToHost));
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
  CUDA_CHECK_KERNEL_SYNC(dummy_kernel<<<1,1>>>());
  cudaDeviceSynchronize();

  if(BENCHMARK_ALLOCATOR == MALLOCMC){
    for(int i=16;i<256;i = i << 1){
      if(BENCHMARK_ALLOCATION_SIZE > 1) i = BENCHMARK_ALLOCATION_SIZE;
      dout() << "after freeing everything: free slots of size " << i << ": " << mallocMC::getAvailableSlots(i) << std::endl;
      if(BENCHMARK_ALLOCATION_SIZE > 1) break;
    }
  }

  if(BENCHMARK_ALLOCATOR == MALLOCMC){
    mallocMC::finalizeHeap();
  }
  cudaFree(d_success);
  cudaFree(pointerStoreForThreads);
  cudaFree(fillLevelsPerThread);
  cudaFree(pointersPerThread);
  cudaFree(randomState);
  dout() << "done "<< std::endl;

  return h_globalSuccess;
}
