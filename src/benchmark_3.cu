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

#define LOG 0
#define LIN 1

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

// basic files for mallocMC
#include <mallocMC/mallocMC_overwrites.hpp>
#include <mallocMC/mallocMC_utils.hpp>
#include "benchmark_3.config.hpp"
//MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)
//MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocatorCoalesced)
//MALLOCMC_SET_ALLOCATOR_TYPE(HallocAllocator)
MALLOCMC_SET_ALLOCATOR_TYPE(MMC_TYPEDEF)

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
__device__ long long unsigned globalAllocationsContinued = 0;
__device__ long long unsigned globalFreeContinued = 0;
__device__ long long unsigned globalFreeTeardown = 0;
__device__ long long unsigned globalFailsContinued = 0;
__device__ long long unsigned globalAllocClocks = 0;
__device__ long long unsigned globalFreeClocks = 0;


__global__ void cleanup_kernel(){
  printf("cleanup kernel ran successfully\n");
  globalSuccess = 1;
  globalAllocationsContinued = 0llu;
  globalFreeContinued = 0llu;
  globalFreeTeardown = 0llu;
  globalFailsContinued = 0llu;
  globalAllocClocks = 0llu;
  globalFreeClocks = 0llu;
}


void init_kernel(){
  CUDA_CHECK_KERNEL_SYNC(cleanup_kernel<<<1,1>>>());
  cudaDeviceSynchronize();
}

__global__ void init_srand_kernel(int seed, curandState_t* randomState, int desiredThreads){
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id>desiredThreads) return;
	curand_init(seed, id, 0, &randomState[id]);
}

__global__ void getBenchmarkData(
        long long unsigned *devAllocationsContinued,
        long long unsigned *devFreeContinued,
        long long unsigned *devAllocClocks,
        long long unsigned *devFreeClocks
        ){

        *devAllocationsContinued = globalAllocationsContinued;
        *devFreeContinued = globalFreeContinued;
        *devAllocClocks = globalAllocClocks;
        *devFreeClocks = globalFreeClocks;
}

__global__ void getTeardown(){
  printf("Free-operations during Teardown: %llu\n",globalFreeTeardown);
  printf("Total allocations: %llu\n",globalAllocationsContinued);
  printf("Total free:        %llu\n",globalFreeContinued+globalFreeTeardown);
  printf("Average clocks per alloc: %llu\n",(long long unsigned)(globalAllocClocks/globalAllocationsContinued));
  printf("Average clocks per free : %llu\n",(long long unsigned)(globalFreeClocks/globalFreeContinued));
}
__global__ void getWarmupStats(){
  printf("Alloc-operations during Warmup: %llu (%d fails)\n",globalAllocationsContinued,globalFailsContinued);
  printf("Free-operations during Warmup: %llu\n",globalFreeContinued);
  printf("Average clocks per alloc Warmup: %llu\n",(long long unsigned)(globalAllocClocks/globalAllocationsContinued));
  printf("Average clocks per free Warmup: %llu\n",(long long unsigned)(globalFreeClocks/globalFreeContinued));
}
__global__ void getContinuedStats(){
  printf("Alloc-operations during run: %llu (%llu fails)\n",globalAllocationsContinued,globalFailsContinued);
  printf("Free-operations during run: %llu\n",globalFreeContinued);
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

__host__ int getAllocSize(){
#if BENCHMARK_ALLOCATION_SIZE == LOG
      //return getAllocSizeLogScale(generator, distribution);
      return getAllocSizeLogScale();
#endif
#if BENCHMARK_ALLOCATION_SIZE == LIN
      //return getAllocSizeLinScale(generator, distribution);
      return getAllocSizeLinScale();
#endif
#if BENCHMARK_ALLOCATION_SIZE > 7
      return BENCHMARK_ALLOCATION_SIZE;
#endif
}

__device__ int getAllocSize(const int id, curandState_t* randomState){
#if BENCHMARK_ALLOCATION_SIZE == LOG
    return getAllocSizeLogScale(id, randomState);
#endif
#if BENCHMARK_ALLOCATION_SIZE == LIN
    return getAllocSizeLinScale(id, randomState);
#endif
#if BENCHMARK_ALLOCATION_SIZE > 7
  return BENCHMARK_ALLOCATION_SIZE;
#endif
}

__device__ void* allocUnderTest(size_t size,long long unsigned* duration){
  long long unsigned start_time = clock64();
  void* p = mallocMC::malloc(size);
  long long unsigned stop_time = clock64();
  *duration = stop_time-start_time;
  return p;
}

__device__ void freeUnderTest(void* p,long long unsigned* duration){
  long long unsigned start_time = clock64();
  mallocMC::free(p);
  long long unsigned stop_time = clock64();
  *duration = stop_time-start_time;
}


__global__ void allocKernel(
    int** pointerStore,
	int pointerStoreSize,
    int desiredThreads,
    const int alloc_size,
	curandState_t* randomState
    ){
    
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;
  int maxPointersPerThread = pointerStoreSize/desiredThreads;
  if(maxPointersPerThread < 1) return; 
  int r = ceil(curand_uniform(&randomState[id])*(float)maxPointersPerThread)-1;
  int position = id*maxPointersPerThread + r;

  if(pointerStore[position] == NULL){
	  long long unsigned duration=0llu;
	  int * p = (int*) allocUnderTest(alloc_size, &duration);
	  if(p == NULL){
		  atomicAdd(&globalFailsContinued,1llu);
	  }else{
		  pointerStore[position] = p;
		  atomicAdd(&globalAllocationsContinued,1llu);
		  atomicAdd(&globalAllocClocks, duration);
	  }
  }
}


__global__ void freeKernel(
    int** pointerStore,
	int pointerStoreSize,
    int desiredThreads,
	curandState_t* randomState
    ){
    
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= desiredThreads) return;
  int maxPointersPerThread = pointerStoreSize/desiredThreads;
  if(maxPointersPerThread < 1) return; 
  int r = ceil(curand_uniform(&randomState[id])*maxPointersPerThread)-1;
  int position = id*maxPointersPerThread + r;

  if(pointerStore[position] != NULL){
	  long long unsigned duration=0llu;
	  freeUnderTest(pointerStore[position],&duration);
	  pointerStore[position] = NULL;
	  atomicAdd(&globalFreeContinued,1llu);
	  atomicAdd(&globalFreeClocks, duration);
  }
}


__global__ void getSuccessState(int* success){
  printf("Allocations done while running: %llu (%d times, no memory was available)\n",
      globalAllocationsContinued,globalFailsContinued);
  printf("Free-operations done while running: %llu\n",
      globalFreeContinued);
  success[0] = globalSuccess;
}

std::string writeBenchmarkData(){
  long long unsigned hostAllocationsContinued = 0;
  long long unsigned hostFreeContinued = 0;
  long long unsigned hostAllocClocks = 0;
  long long unsigned hostFreeClocks = 0;
  long long unsigned *devAllocationsContinued;
  long long unsigned *devFreeContinued;
  long long unsigned *devAllocClocks;
  long long unsigned *devFreeClocks;

  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devAllocationsContinued,sizeof(long long unsigned)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devFreeContinued,sizeof(long long unsigned)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devAllocClocks,sizeof(long long unsigned)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &devFreeClocks,sizeof(long long unsigned)));
  CUDA_CHECK_KERNEL_SYNC(getBenchmarkData<<<1,1>>>(
        devAllocationsContinued,
        devFreeContinued,
        devAllocClocks,
        devFreeClocks
        ));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostAllocationsContinued,devAllocationsContinued,sizeof(long long unsigned),cudaMemcpyDeviceToHost));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostFreeContinued,devFreeContinued,sizeof(long long unsigned),cudaMemcpyDeviceToHost));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostAllocClocks,devAllocClocks,sizeof(long long unsigned),cudaMemcpyDeviceToHost));
  BENCHMARK_CHECKED_CALL(cudaMemcpy(&hostFreeClocks,devFreeClocks,sizeof(long long unsigned),cudaMemcpyDeviceToHost));

  std::stringstream ss;
  if(hostAllocationsContinued == 0) hostAllocationsContinued = 1;
  ss << hostAllocClocks/hostAllocationsContinued << "    ";
  if(hostFreeContinued == 0) hostFreeContinued = 1;
  ss << hostFreeClocks/hostFreeContinued;

  return ss.str();
}


void allocation_cycle(int** pointerStore, int pointerStoreSize, int desiredThreads, unsigned blocks, unsigned threads, curandState_t* randomState){

	float probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	int alloc_size = getAllocSize();
	if(probability <= 0.75){
		CUDA_CHECK_KERNEL_SYNC(allocKernel<<<blocks,threads>>>(
					pointerStore,
					pointerStoreSize,
					desiredThreads,
					alloc_size,
					randomState
					));
	}

	probability = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	if(probability <= 0.75){
		CUDA_CHECK_KERNEL_SYNC(freeKernel<<<blocks,threads>>>(
					pointerStore,
					pointerStoreSize,
					desiredThreads,
					randomState
					));
	}
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

  // a higher pointerStoreSize will increase variability of the accesses
  int pointerStoreSize = desiredThreads*100;

  const size_t usableMemoryMB   = deviceProp.totalGlobalMem / size_t(1024U * 1024U);
  if(heapMB*10 > usableMemoryMB*5) dout() << "Warning: heapSize fills more than 50% of global Memory" << std::endl;
  if(heapMB*10 > usableMemoryMB*8) dout() << "Warning: heapSize fills more than 80% of global Memory" << std::endl;

  size_t heapSize         = size_t(1024U*1024U) * heapMB;
  // keep an extra 100MB for internal structures the allocators might need
  const size_t maxPossibleHeapSize = deviceProp.totalGlobalMem - pointerStoreSize*sizeof(int*) - desiredThreads*sizeof(curandState_t) - 100U*1024U*1024U;
  if(heapSize > maxPossibleHeapSize){
	  dout() << "chosen HeapSize of " << heapMB << " MB is too big, reducing to " << maxPossibleHeapSize / size_t(1024U*1024U) << " MB" << std::endl;
	  heapSize = maxPossibleHeapSize;
  }
  dout() << "necessary memory for pointers: " << pointerStoreSize*sizeof(int*)*4 << " bytes" << std::endl;
  dout() << "reserved Heapsize:             " << heapSize << std::endl;
  machine_output.push_back(MK_STRINGPAIR(heapSize));

  int** pointerStore;
  curandState_t* randomState;
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &pointerStore, pointerStoreSize*sizeof(int*)));
  BENCHMARK_CHECKED_CALL(cudaMalloc((void**) &randomState, desiredThreads * sizeof(curandState_t)));

  mallocMC::initHeap(heapSize);
  srand(31415);
  init_srand_kernel<<<threads,blocks>>>(31415, randomState, desiredThreads);

  for(int warm_i=0; warm_i < 5000; ++warm_i){
	  allocation_cycle(pointerStore, pointerStoreSize, desiredThreads, blocks, threads, randomState);
  }

  cudaDeviceSynchronize();
  CUDA_CHECK_KERNEL_SYNC(getWarmupStats<<<1,1>>>());
  CUDA_CHECK_KERNEL_SYNC(cleanup_kernel<<<1,1>>>());
  dout() << "WARMUP COMPLETE" << std::endl;

  for(int cont_i=0; cont_i < 50000; ++cont_i){
	  allocation_cycle(pointerStore, pointerStoreSize, desiredThreads, blocks, threads, randomState);
  }

  int* d_success;
  cudaMalloc((void**) &d_success,sizeof(int));
  getSuccessState<<<1,1>>>(d_success);
  BENCHMARK_CHECKED_CALL(cudaMemcpy((void*) &h_globalSuccess,d_success, sizeof(int), cudaMemcpyDeviceToHost));
  machine_output.push_back(MK_STRINGPAIR(h_globalSuccess));
  //print_machine_readable(machine_output);

  CUDA_CHECK_KERNEL_SYNC(getTeardown<<<1,1>>>());
  cudaDeviceSynchronize();

  mallocMC::finalizeHeap();
  cudaFree(d_success);
  cudaFree(pointerStore);
  cudaFree(randomState);
  dout() << "done "<< std::endl;

  return h_globalSuccess;
}
