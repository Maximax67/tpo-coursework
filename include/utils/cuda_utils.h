#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                             \
  do                                                                 \
  {                                                                  \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
      char errorMsg[256];                                            \
      snprintf(errorMsg, sizeof(errorMsg), "CUDA error %s:%d: '%s'", \
               __FILE__, __LINE__, cudaGetErrorString(err));         \
      throw std::runtime_error(errorMsg);                            \
    }                                                                \
  } while (0)

#define BLOCK_SIZE 16
#define BLOCK_DIM 256

inline void initializeCuda()
{
  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0)
  {
    throw std::runtime_error("No CUDA devices found");
  }

  CUDA_CHECK(cudaSetDevice(0));

  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  printf("Using CUDA device: %s\n", deviceProp.name);
  printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
}

inline bool initCUDA()
{
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0)
  {
    return false;
  }

  int dev = 0;
  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  CUDA_CHECK(cudaSetDevice(dev));

  return true;
}
