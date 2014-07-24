/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
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


__global__ void check_content(
    allocElem_t** data,
    unsigned long long *counter,
    unsigned long long* globalSum,
    const size_t nSlots,
    int* correct
    ){

  unsigned long long sum=0;
  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= nSlots){break;}
    const size_t offset = pos*ELEMS_PER_SLOT;
    for(size_t i=0;i<ELEMS_PER_SLOT;++i){
      if (static_cast<allocElem_t>(data[pos][i]) != static_cast<allocElem_t>(offset+i)){
        atomicAnd(correct,0);
      }
      sum += static_cast<unsigned long long>(data[pos][i]);
    }
  }
  atomicAdd(globalSum,sum);
}



/**
 * allocate a lot of small arrays and fill them
 *
 * Each array has the size ELEMS_PER_SLOT and the type allocElem_t.
 * Each element will be filled with a number that is related to its
 * position in the datastructure.
 *
 * @param data the datastructure to allocate
 * @param counter should be initialized with 0 and will
 *        hold, how many allocations were done
 * @param globalSum will hold the sum of all values over all
 *        allocated structures (for verification purposes)
 */
__global__ void allocAll(
    allocElem_t** data,
    unsigned long long* counter,
    unsigned long long* globalSum
    ){

  unsigned long long sum=0;
  while(true){
    allocElem_t* p = (allocElem_t*) mallocMC::malloc(sizeof(allocElem_t) * ELEMS_PER_SLOT);
    if(p == NULL) break;

    size_t pos = atomicAdd(counter,1);
    const size_t offset = pos*ELEMS_PER_SLOT;
    for(size_t i=0;i<ELEMS_PER_SLOT;++i){
      p[i] = static_cast<allocElem_t>(offset + i);
      sum += static_cast<unsigned long long>(p[i]);
    }
    data[pos] = p;
  }

  atomicAdd(globalSum,sum);
}

__device__ int globalSuccess = 1;

__global__ void createPointerStorageInThreads(
    size_t** pointerStore,
    size_t maxStorageSize,
    int desiredThreads,
    int* fillLevelsPerThread
    ){
  

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id > desiredThreads) return;
  size_t* p = (size_t*) malloc(sizeof(size_t) * maxStorageSize);
//if(p == NULL) atomicAnd(&globalSuccess, 0);
  pointerStore[id] = p;
  fillLevelsPerThread[id] = 0;
}

__global__ void initialFillingOfPointerStorage(
    size_t** pointerStore,
    size_t maxStorageSize,
    int desiredThreads,
    int* fillLevelsPerThread
    ){

}

/**
 * free all the values again
 *
 * @param data the datastructure to free
 * @param counter should be an empty space on device memory,
 *        counts how many elements were freed
 * @param max the maximum number of elements to free
 */
__global__ void deallocAll(
    allocElem_t** data,
    unsigned long long* counter,
    const size_t nSlots
    ){

  while(true){
    size_t pos = atomicAdd(counter,1);
    if(pos >= nSlots) break;
    mallocMC::free(data[pos]);
  }
}



/**
 * wrapper function to allocate memory on device
 *
 * allocates memory with mallocMC. Returns the number of
 * created elements as well as the sum of these elements
 *
 * @param d_testData the datastructure which will hold
 *        pointers to the created elements
 * @param h_nSlots will be filled with the number of elements
 *        that were allocated
 * @param h_sum will be filled with the sum of all elements created
 * @param blocks the size of the CUDA grid
 * @param threads the number of CUDA threads per block
 */
void allocate(
    allocElem_t** d_testData,
    unsigned long long* h_nSlots,
    unsigned long long* h_sum,
    const unsigned blocks,
    const unsigned threads
    ){

  dout() << "allocating on device...";

  unsigned long long zero = 0;
  unsigned long long *d_sum;
  unsigned long long *d_nSlots;

  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_sum,sizeof(unsigned long long)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &d_nSlots, sizeof(unsigned long long)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMemcpy(d_sum,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMemcpy(d_nSlots,&zero,sizeof(unsigned long long),cudaMemcpyHostToDevice));

  CUDA_CHECK_KERNEL_SYNC(allocAll<<<blocks,threads>>>(d_testData,d_nSlots,d_sum));

  MALLOCMC_CUDA_CHECKED_CALL(cudaMemcpy(h_sum,d_sum,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMemcpy(h_nSlots,d_nSlots,sizeof(unsigned long long),cudaMemcpyDeviceToHost));
  cudaFree(d_sum);
  cudaFree(d_nSlots);
  dout() << "done" << std::endl;
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

  int h_globalSuccess=4;

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
  
  unsigned threads = threadsUsedInBlock;
  unsigned blocks  = maxUsefulBlocks;

  const size_t usableMemoryMB   = deviceProp.totalGlobalMem / size_t(1024U * 1024U);
  if(heapMB > usableMemoryMB/2) dout() << "Warning: heapSize fills more than 50% of global Memory" << std::endl;

  const size_t heapSize         = size_t(1024U*1024U) * heapMB;
  machine_output.push_back(MK_STRINGPAIR(heapSize));

  size_t** pointerStoreForThreads;
  int* fillLevelsPerThread;
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &pointerStoreForThreads, desiredThreads*sizeof(allocElem_t*)));
  MALLOCMC_CUDA_CHECKED_CALL(cudaMalloc((void**) &fillLevelsPerThread, sizeof(int)));

  //if a single thread allocates only the minimal chunksize, it can not exceed this number
  size_t maxStorageSize = heapMB * size_t(1024U * 1024U) / size_t(16U);

  createPointerStorageInThreads<<<blocks,threads>>>(pointerStoreForThreads,maxStorageSize,desiredThreads,fillLevelsPerThread);

  initialFillingOfPointerStorage<<<blocks,threads,sizeof(int)*threads>>>(
      pointerStoreForThreads,
      maxStorageSize,
      desiredThreads,
      fillLevelsPerThread
      );


  bool correct                  = true;

  // initializing the heap
  mallocMC::initHeap(heapSize);



  // verifying on device
  //correct = correct && verify(d_testData,usedSlots,blocks,threads);

  // damaging one cell
  dout() << "damaging of element... ";
  //CUDA_CHECK_KERNEL_SYNC(damageElement<<<1,1>>>(d_testData));
  dout() << "done" << std::endl;

  // verifying on device
  // THIS SHOULD FAIL (damage was done before!). Therefore, we must inverse the logic
  //correct = correct && !verify(d_testData,usedSlots,blocks,threads);


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

  MALLOCMC_CUDA_CHECKED_CALL(cudaMemcpy(&h_globalSuccess,(int*) &globalSuccess, sizeof(int), cudaMemcpyDeviceToHost));
  machine_output.push_back(MK_STRINGPAIR(h_globalSuccess));

  print_machine_readable(machine_output);

  return correct;
}
