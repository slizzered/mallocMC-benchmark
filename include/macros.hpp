// get a CUDA error and print it nicely
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; \
  if(error!=cudaSuccess){\
    printf("<%s>:%i ",__FILE__,__LINE__);\
    printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

// start kernel, wait for finish and check errors
#define CUDA_CHECK_KERNEL_SYNC(...) __VA_ARGS__;CUDA_CHECK(cudaDeviceSynchronize())

