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

// each pointer in the datastructure will point to this many
// elements of type allocElem_t
#define ELEMS_PER_SLOT 750


MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)


// the type of the elements to allocate
typedef unsigned long long allocElem_t;

bool run_benchmark_1(const size_t, const unsigned, const bool);

__global__ void dummy_kernel(){
  printf("dummy kernel ran successfully\n");
}

void init_kernel(unsigned threads, unsigned blocks){
  CUDA_CHECK_KERNEL_SYNC(dummy_kernel<<<1,1>>>());
  cudaDeviceSynchronize();
}

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


  init_kernel(threads, blocks);

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
  int counter = 0;


  float fillLevelPercent = float(fillLevel) / maxBytesPerThread;
  while(fillLevelPercent < 0.75){
    fillLevelPercent = float(fillLevel) / maxBytesPerThread;
      //const int multiplier = ceil(curand_uniform(&randomState[id])*4)-1;
      //const int alloc_size = 128 << multiplier;
      const int alloc_size = 128;
      //printf("thread %d wants to allocate %d bytes (%d slots remaining)\n",id, alloc_size, mallocMC::getAvailableSlots(alloc_size));
      pointerStoreReg[++pointersPerThreadReg] = (int*) malloc(alloc_size);
      if(pointerStoreReg[pointersPerThreadReg] == NULL){
        printf("thread %d wants to allocate %d bytes (%d slots remaining), but did NOT get anything!\n",id, alloc_size, mallocMC::getAvailableSlots(alloc_size));
        break;
      }
      pointerStoreReg[pointersPerThreadReg][0] = alloc_size;
      fillLevel += alloc_size;
    ++counter;
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


  while(fillLevel+128 < maxBytesPerThread && counter < 1000){
    const float probability = min(curand_uniform(&randomState[id])+0.25,1.);
    const float fillLevelPercent = float(fillLevel) / maxBytesPerThread;
    if(pointersPerThreadReg > 0 && probability <= fillLevelPercent) { //probably, the fill level is higher than 75% -> deallocate
        int free_size = pointerStoreReg[pointersPerThreadReg][0];
        printf("thread %d wants to free %d bytes of memory\n",id, free_size);
        free(pointerStoreReg[pointersPerThreadReg--]);
        fillLevel -= free_size;
    }else{ 
     // const int multiplier = ceil(curand_uniform(&randomState[id])*4)-1;
     // const int alloc_size = 16 << multiplier;
      const int alloc_size = 128;
      //printf("thread %d wants to allocate %d bytes (%d slots remaining)\n",id, alloc_size, mallocMC::getAvailableSlots(alloc_size));
      printf("thread %d wants to allocate %d bytes of memory\n",id, alloc_size);
      pointerStoreReg[++pointersPerThreadReg] = (int*) malloc(alloc_size);
      if(pointerStoreReg[pointersPerThreadReg] == NULL){
        printf("thread %d wants to allocate %d bytes (%d slots remaining), but did NOT get anything!\n",id, alloc_size, mallocMC::getAvailableSlots(alloc_size));
        break;
      }
      pointerStoreReg[pointersPerThreadReg][0] = alloc_size;
      fillLevel += alloc_size;
    }
    ++counter;
  }

  printf("Thread %d  fillLevel: %d byte (%.0f%%) maxBytesPerThread: %d allocatedElements: %d\n",id, fillLevel, 100*float(fillLevel)/maxBytesPerThread,maxBytesPerThread, pointersPerThreadReg);

  fillLevelsPerThread[id] = fillLevel;
  pointerStore[id] = pointerStoreReg;
  pointersPerThread[id] = pointersPerThreadReg;

}

__global__ void getSuccessState(int* success){
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

  int h_globalSuccess=-1;

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
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &pointerStoreForThreads, desiredThreads*sizeof(allocElem_t*)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &fillLevelsPerThread, sizeof(int)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &pointersPerThread, sizeof(int)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &randomState, desiredThreads * sizeof(curandState_t)));

  // initializing the heap
  mallocMC::initHeap(heapSize*2);


  //dout() << "maxStoredChunks: " << maxStoredChunks << std::endl;
  size_t completeStorage = maxMemTotal + maxChunksTotal*sizeof(int*);
  dout() << "necessary memory for malloc on device: " << completeStorage << std::endl;
  dout() << "mallocMC Heapsize:                     " << heapSize*100 << std::endl;

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

  CUDA_CHECK_KERNEL_SYNC(continuedFillingOfPointerStorage<<<blocks,threads>>>(
      pointerStoreForThreads,
      maxMemPerThread,
      desiredThreads,
      fillLevelsPerThread,
      pointersPerThread,
      randomState
      ));
  cudaDeviceSynchronize();




  // release all memory
  dout() << "deallocation...        ";
  mallocMC::finalizeHeap();

  dout() << "done "<< std::endl;

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
  print_machine_readable(machine_output);
  cudaFree(d_success);
  cudaFree(pointerStoreForThreads);
  cudaFree(fillLevelsPerThread);

  machine_output.push_back(MK_STRINGPAIR(h_globalSuccess));

  return h_globalSuccess;
}
