#include "layers/pool.h"

__global__ void maxPoolingForwardKernel(
    const float *input,
    float *output,
    int *maxIndices,
    int inputHeight,
    int inputWidth,
    int channels,
    int poolSize,
    int stride,
    int outputHeight,
    int outputWidth)
{
  int ox = blockIdx.x * blockDim.x + threadIdx.x;
  int oy = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (ox < outputWidth && oy < outputHeight && c < channels)
  {
    int ixStart = ox * stride;
    int iyStart = oy * stride;

    float maxVal = -INFINITY;
    int maxIdx = -1;

    for (int ky = 0; ky < poolSize; ++ky)
    {
      for (int kx = 0; kx < poolSize; ++kx)
      {
        int ix = ixStart + kx;
        int iy = iyStart + ky;

        if (ix < inputWidth && iy < inputHeight)
        {
          int inputIdx = (iy * inputWidth + ix) * channels + c;
          float val = input[inputIdx];

          if (val > maxVal)
          {
            maxVal = val;
            maxIdx = inputIdx;
          }
        }
      }
    }

    int outputIdx = (oy * outputWidth + ox) * channels + c;
    output[outputIdx] = maxVal;
    maxIndices[outputIdx] = maxIdx;
  }
}

__global__ void maxPoolingForwardKernelSequential(
    const float *input,
    float *output,
    int *maxIndices,
    int inputHeight,
    int inputWidth,
    int channels,
    int poolSize,
    int stride,
    int outputHeight,
    int outputWidth)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int c = 0; c < channels; ++c)
    {
      for (int oy = 0; oy < outputHeight; ++oy)
      {
        for (int ox = 0; ox < outputWidth; ++ox)
        {
          int ixStart = ox * stride;
          int iyStart = oy * stride;

          float maxVal = -INFINITY;
          int maxIdx = -1;

          for (int ky = 0; ky < poolSize; ++ky)
          {
            for (int kx = 0; kx < poolSize; ++kx)
            {
              int ix = ixStart + kx;
              int iy = iyStart + ky;

              if (ix < inputWidth && iy < inputHeight)
              {
                int inputIdx = (iy * inputWidth + ix) * channels + c;
                float val = input[inputIdx];

                if (val > maxVal)
                {
                  maxVal = val;
                  maxIdx = inputIdx;
                }
              }
            }
          }

          int outputIdx = (oy * outputWidth + ox) * channels + c;
          output[outputIdx] = maxVal;
          maxIndices[outputIdx] = maxIdx;
        }
      }
    }
  }
}

__global__ void max_pooling_backward_kernel(
    const float *outputGradient, float *inputGradient, const int *maxIndices,
    int outputHeight, int outputWidth, int channels)
{
  int outputX = blockIdx.x * blockDim.x + threadIdx.x;
  int outputY = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (outputX < outputWidth && outputY < outputHeight && c < channels)
  {
    int outputIdx = (c * outputHeight + outputY) * outputWidth + outputX;
    int maxIdx = maxIndices[outputIdx];

    if (maxIdx >= 0)
    {
      atomicAdd(&inputGradient[maxIdx], outputGradient[outputIdx]);
    }
  }
}

__global__ void max_pooling_backward_kernel_sequential(
    const float *outputGradient,
    float *inputGradient,
    const int *maxIndices,
    int outputHeight,
    int outputWidth,
    int channels)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int c = 0; c < channels; ++c)
    {
      for (int oy = 0; oy < outputHeight; ++oy)
      {
        for (int ox = 0; ox < outputWidth; ++ox)
        {
          int outputIdx = (c * outputHeight + oy) * outputWidth + ox;
          int maxIdx = maxIndices[outputIdx];

          if (maxIdx >= 0)
          {
            inputGradient[maxIdx] += outputGradient[outputIdx];
          }
        }
      }
    }
  }
}

PoolingLayer::PoolingLayer(
    const std::vector<size_t> &inputShape,
    size_t poolSize,
    size_t stride) : inputShape_(inputShape),
                     poolSize_(poolSize)
{
  stride_ = (stride == 0) ? poolSize : stride;

  size_t inputHeight = inputShape_[0];
  size_t inputWidth = inputShape_[1];
  size_t channels = inputShape_[2];

  size_t outputHeight = (inputHeight - poolSize_) / stride_ + 1;
  size_t outputWidth = (inputWidth - poolSize_) / stride_ + 1;

  outputShape_ = {outputHeight, outputWidth, channels};

  size_t outputSize = outputHeight * outputWidth * channels;
  cudaMalloc(&d_maxIndices_, outputSize * sizeof(int));
}

PoolingLayer::~PoolingLayer()
{
  if (d_maxIndices_)
    cudaFree(d_maxIndices_);
}

void PoolingLayer::forward(const float *input, float *output, cudaStream_t stream, bool seq)
{
  size_t inputHeight = inputShape_[0];
  size_t inputWidth = inputShape_[1];
  size_t channels = inputShape_[2];

  size_t outputHeight = outputShape_[0];
  size_t outputWidth = outputShape_[1];

  if (seq)
  {
    maxPoolingForwardKernel<<<1, 1, 0, stream>>>(
        input, output, d_maxIndices_,
        inputHeight, inputWidth, channels,
        poolSize_, stride_,
        outputHeight, outputWidth);
  }
  else
  {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(
        (outputWidth + blockDim.x - 1) / blockDim.x,
        (outputHeight + blockDim.y - 1) / blockDim.y,
        channels);

    maxPoolingForwardKernel<<<gridDim, blockDim, 0, stream>>>(
        input, output, d_maxIndices_,
        inputHeight, inputWidth, channels,
        poolSize_, stride_,
        outputHeight, outputWidth);
  }

  CUDA_CHECK(cudaGetLastError());
}

void PoolingLayer::backward(const float *input, const float *output,
                            const float *outputGradient, float *inputGradient, cudaStream_t stream, bool seq)
{
  size_t inputHeight = inputShape_[0];
  size_t inputWidth = inputShape_[1];
  size_t channels = inputShape_[2];

  size_t outputHeight = outputShape_[0];
  size_t outputWidth = outputShape_[1];

  size_t inputSize = inputHeight * inputWidth * channels;
  cudaMemsetAsync(inputGradient, 0, inputSize * sizeof(float), stream);

  if (seq)
  {
    max_pooling_backward_kernel_sequential<<<1, 1, 0, stream>>>(
        outputGradient, inputGradient, d_maxIndices_,
        outputHeight, outputWidth, channels);
  }
  else
  {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim(
        (outputWidth + blockDim.x - 1) / blockDim.x,
        (outputHeight + blockDim.y - 1) / blockDim.y,
        channels);

    max_pooling_backward_kernel<<<gridDim, blockDim, 0, stream>>>(
        outputGradient, inputGradient, d_maxIndices_,
        outputHeight, outputWidth, channels);
  }

  CUDA_CHECK(cudaGetLastError());
}

std::vector<size_t> PoolingLayer::getOutputShape() const
{
  return outputShape_;
}

std::vector<size_t> PoolingLayer::getInputShape() const
{
  return inputShape_;
}
