#include "layers/fc.h"
#include "utils/cuda_utils.h"
#include <cmath>
#include <iostream>
#include <random>

__global__ void fullyConnectedForwardKernel(const float *input, const float *weights,
                                            const float *bias, float *output,
                                            int inputSize, int outputSize)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < outputSize)
  {
    float sum = 0.0f;
    for (int i = 0; i < inputSize; i++)
    {
      sum += input[i] * weights[idx * inputSize + i];
    }
    sum += bias[idx];
    output[idx] = sum;
  }
}

__global__ void fullyConnectedForwardKernelSequential(const float *input, const float *weights,
                                                      const float *bias, float *output,
                                                      int inputSize, int outputSize)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int idx = 0; idx < outputSize; ++idx)
    {
      float sum = 0.0f;
      for (int i = 0; i < inputSize; ++i)
      {
        sum += input[i] * weights[idx * inputSize + i];
      }
      sum += bias[idx];
      output[idx] = sum;
    }
  }
}

__global__ void applyReLUKernel(float *data, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    data[idx] = fmaxf(0.0f, data[idx]);
  }
}

__global__ void applyReLUKernelSequential(float *data, int size)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int idx = 0; idx < size; ++idx)
    {
      data[idx] = fmaxf(0.0f, data[idx]);
    }
  }
}

__global__ void applyReLUDerivativeKernel(const float *output, float *gradient, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    gradient[idx] *= (output[idx] > 0.0f) ? 1.0f : 0.0f;
  }
}

__global__ void applyReLUDerivativeKernelSequential(const float *output, float *gradient, int size)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int idx = 0; idx < size; ++idx)
    {
      gradient[idx] *= (output[idx] > 0.0f) ? 1.0f : 0.0f;
    }
  }
}

__global__ void fullyConnectedBackwardKernel(const float *input, const float *outputGradient,
                                             float *inputGradient, const float *weights,
                                             int inputSize, int outputSize)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < inputSize)
  {
    float sum = 0.0f;
    for (int i = 0; i < outputSize; i++)
    {
      sum += outputGradient[i] * weights[i * inputSize + idx];
    }
    inputGradient[idx] = sum;
  }
}

__global__ void fullyConnectedBackwardKernelSequential(const float *input, const float *outputGradient,
                                                       float *inputGradient, const float *weights,
                                                       int inputSize, int outputSize)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int idx = 0; idx < inputSize; ++idx)
    {
      float sum = 0.0f;
      for (int i = 0; i < outputSize; ++i)
      {
        sum += outputGradient[i] * weights[i * inputSize + idx];
      }
      inputGradient[idx] = sum;
    }
  }
}

__global__ void updateWeightsKernel(float *weights, float *biases,
                                    const float *outputGradient,
                                    int inputSize, int outputSize, float learningRate)
{
  int outIdx = blockIdx.y;
  int inIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (outIdx < outputSize && inIdx < inputSize)
  {
    int idx = outIdx * inputSize + inIdx;
    weights[idx] -= learningRate * outputGradient[outIdx];
  }

  if (inIdx == 0 && outIdx < outputSize)
  {
    biases[outIdx] -= learningRate * outputGradient[outIdx];
  }
}

__global__ void updateWeightsKernelSequential(float *weights, float *biases,
                                              const float *outputGradient,
                                              int inputSize, int outputSize, float learningRate)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int outIdx = 0; outIdx < outputSize; ++outIdx)
    {
      for (int inIdx = 0; inIdx < inputSize; ++inIdx)
      {
        int idx = outIdx * inputSize + inIdx;
        weights[idx] -= learningRate * outputGradient[outIdx];
      }
      biases[outIdx] -= learningRate * outputGradient[outIdx];
    }
  }
}

FullyConnectedLayer::FullyConnectedLayer(
    size_t inputSize,
    size_t outputSize,
    bool useReLU) : inputSize_(inputSize),
                    outputSize_(outputSize),
                    useReLU_(useReLU)
{
  weights_.resize(inputSize_ * outputSize_);
  biases_.resize(outputSize_);

  initializeWeights();

  CUDA_CHECK(cudaMalloc(&d_weights_, inputSize_ * outputSize_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_biases_, outputSize_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output_, outputSize_ * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_weights_, weights_.data(),
                        inputSize_ * outputSize_ * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biases_, biases_.data(),
                        outputSize_ * sizeof(float),
                        cudaMemcpyHostToDevice));
}

FullyConnectedLayer::~FullyConnectedLayer()
{
  if (d_weights_)
    cudaFree(d_weights_);
  if (d_biases_)
    cudaFree(d_biases_);
  if (d_output_)
    cudaFree(d_output_);
}

void FullyConnectedLayer::forward(const float *input, float *output, cudaStream_t stream, bool seq)
{

  if (seq)
  {
    fullyConnectedForwardKernelSequential<<<1, 1, 0, stream>>>(
        input, d_weights_, d_biases_, output,
        inputSize_, outputSize_);

    if (useReLU_)
    {
      applyReLUKernelSequential<<<1, 1, 0, stream>>>(output, outputSize_);
    }
  }
  else
  {
    int gridSize = (outputSize_ + BLOCK_DIM - 1) / BLOCK_DIM;

    fullyConnectedForwardKernel<<<gridSize, BLOCK_DIM, 0, stream>>>(
        input, d_weights_, d_biases_, output,
        inputSize_, outputSize_);

    if (useReLU_)
    {
      applyReLUKernel<<<gridSize, BLOCK_DIM, 0, stream>>>(output, outputSize_);
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(d_output_, output,
                             outputSize_ * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
}

void FullyConnectedLayer::backward(const float *input, const float *output,
                                   const float *outputGradient, float *inputGradient, cudaStream_t stream, bool seq)
{
  float *d_gradient;

  if (seq)
  {
    if (useReLU_)
    {
      CUDA_CHECK(cudaMallocAsync(&d_gradient, outputSize_ * sizeof(float), stream));
      CUDA_CHECK(cudaMemcpyAsync(d_gradient, outputGradient,
                                 outputSize_ * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));

      applyReLUDerivativeKernelSequential<<<1, 1, 0, stream>>>(
          d_output_, d_gradient, outputSize_);

      CUDA_CHECK(cudaGetLastError());

      fullyConnectedBackwardKernelSequential<<<1, 1, 0, stream>>>(
          input, d_gradient, inputGradient, d_weights_,
          inputSize_, outputSize_);
    }
    else
    {
      fullyConnectedBackwardKernel<<<1, 1, 0, stream>>>(
          input, outputGradient, inputGradient, d_weights_,
          inputSize_, outputSize_);
    }
  }
  else
  {
    int inputGridSize = (inputSize_ + BLOCK_DIM - 1) / BLOCK_DIM;

    if (useReLU_)
    {
      int outputGridSize = (outputSize_ + BLOCK_DIM - 1) / BLOCK_DIM;
      CUDA_CHECK(cudaMallocAsync(&d_gradient, outputSize_ * sizeof(float), stream));
      CUDA_CHECK(cudaMemcpyAsync(d_gradient, outputGradient,
                                 outputSize_ * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));

      applyReLUDerivativeKernel<<<outputGridSize, BLOCK_DIM, 0, stream>>>(
          d_output_, d_gradient, outputSize_);

      CUDA_CHECK(cudaGetLastError());

      fullyConnectedBackwardKernel<<<inputGridSize, BLOCK_DIM, 0, stream>>>(
          input, d_gradient, inputGradient, d_weights_,
          inputSize_, outputSize_);
    }
    else
    {
      fullyConnectedBackwardKernel<<<inputGridSize, BLOCK_DIM, 0, stream>>>(
          input, outputGradient, inputGradient, d_weights_,
          inputSize_, outputSize_);
    }
  }

  cudaFreeAsync(d_gradient, stream);

  CUDA_CHECK(cudaGetLastError());
}

void FullyConnectedLayer::updateWeights(float learningRate, const float *d_outputGradient, cudaStream_t stream, bool seq)
{
  if (seq)
  {
    updateWeightsKernelSequential<<<1, 1, 0, stream>>>(
        d_weights_, d_biases_, d_outputGradient,
        inputSize_, outputSize_, learningRate);
  }
  else
  {
    dim3 gridSize(
        (inputSize_ + BLOCK_DIM - 1) / BLOCK_DIM,
        outputSize_);

    updateWeightsKernel<<<gridSize, BLOCK_DIM, 0, stream>>>(
        d_weights_, d_biases_, d_outputGradient,
        inputSize_, outputSize_, learningRate);
  }

  CUDA_CHECK(cudaMemcpyAsync(weights_.data(), d_weights_,
                             inputSize_ * outputSize_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(biases_.data(), d_biases_,
                             outputSize_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaGetLastError());
}

std::vector<size_t> FullyConnectedLayer::getOutputShape() const
{
  return {outputSize_};
}

std::vector<size_t> FullyConnectedLayer::getInputShape() const
{
  return {inputSize_};
}

std::string FullyConnectedLayer::getName() const
{
  return "FullyConnected";
}

void FullyConnectedLayer::loadWeights(const std::vector<float> &weights)
{
  if (weights.size() != inputSize_ * outputSize_ + outputSize_)
  {
    throw std::runtime_error("Invalid weights size for fully connected layer");
  }

  std::copy(weights.begin(), weights.begin() + inputSize_ * outputSize_, weights_.begin());

  std::copy(weights.begin() + inputSize_ * outputSize_, weights.end(), biases_.begin());

  CUDA_CHECK(cudaMemcpy(d_weights_, weights_.data(),
                        inputSize_ * outputSize_ * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biases_, biases_.data(),
                        outputSize_ * sizeof(float),
                        cudaMemcpyHostToDevice));
}

std::vector<float> FullyConnectedLayer::getWeights() const
{
  std::vector<float> allWeights(inputSize_ * outputSize_ + outputSize_);

  std::copy(weights_.begin(), weights_.end(), allWeights.begin());

  std::copy(biases_.begin(), biases_.end(),
            allWeights.begin() + inputSize_ * outputSize_);

  return allWeights;
}

void FullyConnectedLayer::initializeWeights()
{
  std::random_device rd;
  std::mt19937 gen(rd());

  float limit = std::sqrt(6.0f / (inputSize_ + outputSize_));
  std::uniform_real_distribution<float> dis(-limit, limit);

  for (size_t i = 0; i < weights_.size(); i++)
  {
    weights_[i] = dis(gen);
  }

  std::fill(biases_.begin(), biases_.end(), 0.0f);
}

void FullyConnectedLayer::applyReLU(float *data, size_t size)
{
  int gridSize = (size + BLOCK_DIM - 1) / BLOCK_DIM;

  applyReLUKernel<<<gridSize, BLOCK_DIM>>>(data, size);
}

void FullyConnectedLayer::applyReLUDerivative(const float *output, float *gradient, size_t size)
{
  int gridSize = (size + BLOCK_DIM - 1) / BLOCK_DIM;

  applyReLUDerivativeKernel<<<gridSize, BLOCK_DIM>>>(output, gradient, size);
}
