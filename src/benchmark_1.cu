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
#include <mallocMC/mallocMC_hostclass.hpp>
#include <mallocMC/mallocMC_utils.hpp>

// Load all available policies for mallocMC
#include <mallocMC/CreationPolicies.hpp>
#include <mallocMC/DistributionPolicies.hpp>
#include <mallocMC/OOMPolicies.hpp>
#include <mallocMC/ReservePoolPolicies.hpp>
#include <mallocMC/AlignmentPolicies.hpp>
    
// get a CUDA error and print it nicely
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; \
  if(error!=cudaSuccess){\
    printf("<%s>:%i ",__FILE__,__LINE__);\
    printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

// start kernel, wait for finish and check errors
#define CUDA_CHECK_KERNEL_SYNC(...) __VA_ARGS__;CUDA_CHECK(cudaDeviceSynchronize())

// each pointer in the datastructure will point to this many
// elements of type allocElem_t
#define ELEMS_PER_SLOT 750


// configurate the CreationPolicy "Scatter"
struct ScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef mallocMC::Allocator< 
  mallocMC::CreationPolicies::Scatter<ScatterConfig,ScatterHashParams>,
  mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
  mallocMC::OOMPolicies::ReturnNull,
  mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
  mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;

MALLOCMC_SET_ALLOCATOR_TYPE(ScatterAllocator)


// the type of the elements to allocate
typedef unsigned long long allocElem_t;

bool run_benchmark_1(const size_t, const unsigned, const bool);
void parse_cmdline(const int, char**, size_t*, unsigned*, unsigned*, bool*);
void print_help(char**);



// define some defaults
static const unsigned threads_default = 128;
static const unsigned blocks_default  = 64; 
static const size_t heapInMB_default  = 1024; // 1GB


__global__ void dummy_kernel(){
  printf("dummy kernel ran successfully\n");
}

void init_kernel(unsigned threads, unsigned blocks){
  CUDA_CHECK_KERNEL_SYNC(dummy_kernel<<<1,1>>>());
  cudaDeviceSynchronize();
}

int main(int argc, char** argv){
  bool correct          = false;
  bool machine_readable = false;
  size_t heapInMB       = heapInMB_default;
  unsigned threads      = threads_default;
  unsigned blocks       = blocks_default;

  std::vector<std::pair<std::string,std::string> > machine_output;
  machine_output.push_back(MK_STRINGPAIR(ScatterConfig::pagesize::value));
  print_machine_readable(machine_output);

  parse_cmdline(argc, argv, &heapInMB, &threads, &blocks, &machine_readable);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);


  cudaSetDevice(0);

  init_kernel(threads, blocks);

  if( deviceProp.major < 2 ) {
    std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
    std::cerr << deviceProp.major << "."<< deviceProp.minor << ")" << std::endl;
    //return 1;
  }

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


/**
 * will parse command line arguments
 *
 * for more details, see print_help()
 *
 * @param argc argc from main()
 * @param argv argv from main()
 * @param heapInMP will be filled with the heapsize, if given as a parameter
 * @param threads will be filled with number of threads, if given as a parameter
 * @param blocks will be filled with number of blocks, if given as a parameter
 */
void parse_cmdline(
    const int argc,
    char**argv,
    size_t *heapInMB,
    unsigned *threads,
    unsigned *blocks,
    bool *machine_readable
    ){

  std::vector<std::pair<std::string, std::string> > parameters;

  // Parse Commandline, tokens are shaped like ARG=PARAM or ARG
  // This requires to use '=', if you want to supply a value with a parameter
  for (int i = 1; i < argc; ++i) {
    char* pos = strtok(argv[i], "=");
    std::pair < std::string, std::string > p(std::string(pos), std::string(""));
    pos = strtok(NULL, "=");
    if (pos != NULL) {
      p.second = std::string(pos);
    }
    parameters.push_back(p);
  }

  // go through all parameters that were found
  for (unsigned i = 0; i < parameters.size(); ++i) {
    std::pair < std::string, std::string > p = parameters.at(i);

    if (p.first == "-v" || p.first == "--verbose") {
      verbose = true;
    }

    if (p.first == "--threads") {
      *threads = atoi(p.second.c_str());
    }

    if (p.first == "--blocks") {
      *blocks = atoi(p.second.c_str());
    }

    if(p.first == "--heapsize") {
      *heapInMB = size_t(atoi(p.second.c_str()));
    }

    if(p.first == "-h" || p.first == "--help"){
      print_help(argv);
      exit(0);
    }

    if(p.first == "-m" || p.first == "--machine_readable"){
      *machine_readable = true;
    }
  }
}


/**
 * prints a helpful message about program use
 *
 * @param argv the argv-parameter from main, used to find the program name
 */
void print_help(char** argv){
  std::stringstream s;

  s << "SYNOPSIS:"                                              << std::endl;
  s << argv[0] << " [OPTIONS]"                                  << std::endl;
  s << ""                                                       << std::endl;
  s << "OPTIONS:"                                               << std::endl;
  s << "  -h, --help"                                           << std::endl;
  s << "    Print this help message and exit"                   << std::endl;
  s << ""                                                       << std::endl;
  s << "  -v, --verbose"                                        << std::endl;
  s << "    Print information about parameters and progress"    << std::endl;
  s << ""                                                       << std::endl;
  s << "  -m, --machine_readable"                               << std::endl;
  s << "    Print all relevant parameters as CSV. This will"    << std::endl;
  s << "    suppress all other output unless explicitly"        << std::endl;
  s << "    requested with --verbose or -v"                     << std::endl;
  s << ""                                                       << std::endl;
  s << "  --threads=N"                                          << std::endl;
  s << "    Set the number of threads per block (default "                  ;
  s <<                               threads_default << "128)"  << std::endl;
  s << ""                                                       << std::endl;
  s << "  --blocks=N"                                           << std::endl;
  s << "    Set the number of blocks in the grid (default "                 ;
  s <<                                   blocks_default << ")"  << std::endl;
  s << ""                                                       << std::endl;
  s << "  --heapsize=N"                                         << std::endl;
  s << "    Set the heapsize to N Megabyte (default "                       ;
  s <<                         heapInMB_default << "1024)"      << std::endl;

  std::cout << s.str();
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
  if(p == NULL) atomicAnd(&globalSuccess, 0);
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


  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  unsigned maxBlocksPerSM = 8;
  if(deviceProp.major > 2) maxBlocksPerSM *= 2; //set to 16 for 3.0 and higher
  if(deviceProp.major > 3) maxBlocksPerSM *= 2; //increase again to 32 for 5.0 and higher

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

  std::vector<std::pair<std::string,std::string> > machine_output;
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

  print_machine_readable(machine_output);

  return correct;
}
