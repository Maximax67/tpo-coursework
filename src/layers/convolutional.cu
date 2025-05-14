#include "layers/convolutional.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>

__global__ void conv_forward_kernel(
    const float *input, float *output,
    const float *weights, const float *biases,
    int inputHeight, int inputWidth, int inputChannels,
    int outputHeight, int outputWidth, int numFilters,
    int filterSize, int stride, int padding)
{
  int outputX = blockIdx.x * blockDim.x + threadIdx.x;
  int outputY = blockIdx.y * blockDim.y + threadIdx.y;
  int filterIdx = blockIdx.z;

  if (outputX < outputWidth && outputY < outputHeight && filterIdx < numFilters)
  {
    float sum = biases[filterIdx];

    for (int c = 0; c < inputChannels; ++c)
    {
      for (int fy = 0; fy < filterSize; ++fy)
      {
        for (int fx = 0; fx < filterSize; ++fx)
        {
          int inputY = outputY * stride - padding + fy;
          int inputX = outputX * stride - padding + fx;

          if (inputY >= 0 && inputY < inputHeight && inputX >= 0 && inputX < inputWidth)
          {
            int inputIdx = (c * inputHeight + inputY) * inputWidth + inputX;
            int weightIdx = ((filterIdx * inputChannels + c) * filterSize + fy) * filterSize + fx;

            sum += input[inputIdx] * weights[weightIdx];
          }
        }
      }
    }

    int outputIdx = (filterIdx * outputHeight + outputY) * outputWidth + outputX;
    output[outputIdx] = sum > 0 ? sum : 0;
  }
}

__global__ void conv_forward_kernel_sequential(
    const float *input, float *output,
    const float *weights, const float *biases,
    int inputHeight, int inputWidth, int inputChannels,
    int outputHeight, int outputWidth, int numFilters,
    int filterSize, int stride, int padding)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int filterIdx = 0; filterIdx < numFilters; ++filterIdx)
    {
      for (int outputY = 0; outputY < outputHeight; ++outputY)
      {
        for (int outputX = 0; outputX < outputWidth; ++outputX)
        {
          float sum = biases[filterIdx];

          for (int c = 0; c < inputChannels; ++c)
          {
            for (int fy = 0; fy < filterSize; ++fy)
            {
              for (int fx = 0; fx < filterSize; ++fx)
              {
                int inputY = outputY * stride - padding + fy;
                int inputX = outputX * stride - padding + fx;

                if (inputY >= 0 && inputY < inputHeight &&
                    inputX >= 0 && inputX < inputWidth)
                {
                  int inputIdx = (c * inputHeight + inputY) * inputWidth + inputX;
                  int weightIdx = ((filterIdx * inputChannels + c) * filterSize + fy) * filterSize + fx;

                  sum += input[inputIdx] * weights[weightIdx];
                }
              }
            }
          }

          int outputIdx = (filterIdx * outputHeight + outputY) * outputWidth + outputX;
          output[outputIdx] = sum > 0 ? sum : 0;
        }
      }
    }
  }
}

__global__ void conv_backward_kernel(
    const float *input, const float *output, const float *outputGradient, float *inputGradient,
    const float *weights, float *weightGradients, float *biasGradients,
    int inputHeight, int inputWidth, int inputChannels,
    int outputHeight, int outputWidth, int numFilters,
    int filterSize, int stride, int padding)
{
  int fx = blockIdx.x * blockDim.x + threadIdx.x;
  int fy = blockIdx.y * blockDim.y + threadIdx.y;
  int filterIdx = blockIdx.z;

  if (fx < filterSize && fy < filterSize && filterIdx < numFilters)
  {
    for (int c = 0; c < inputChannels; ++c)
    {
      float grad = 0.0f;

      for (int oy = 0; oy < outputHeight; ++oy)
      {
        for (int ox = 0; ox < outputWidth; ++ox)
        {
          int inputY = oy * stride - padding + fy;
          int inputX = ox * stride - padding + fx;

          if (inputY >= 0 && inputY < inputHeight && inputX >= 0 && inputX < inputWidth)
          {
            int inputIdx = (c * inputHeight + inputY) * inputWidth + inputX;
            int outputIdx = (filterIdx * outputHeight + oy) * outputWidth + ox;

            if (output[outputIdx] > 0)
            {
              grad += outputGradient[outputIdx] * input[inputIdx];
            }
          }
        }
      }

      int weightIdx = ((filterIdx * inputChannels + c) * filterSize + fy) * filterSize + fx;
      weightGradients[weightIdx] += grad;
    }
  }

  if (fx == 0 && fy == 0 && filterIdx < numFilters)
  {
    float grad = 0.0f;

    for (int oy = 0; oy < outputHeight; ++oy)
    {
      for (int ox = 0; ox < outputWidth; ++ox)
      {
        int outputIdx = (filterIdx * outputHeight + oy) * outputWidth + ox;

        if (output[outputIdx] > 0)
        {
          grad += outputGradient[outputIdx];
        }
      }
    }

    biasGradients[filterIdx] += grad;
  }

  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (ix < inputWidth && iy < inputHeight && c < inputChannels)
  {
    float grad = 0.0f;

    for (int f = 0; f < numFilters; ++f)
    {
      for (int fy = 0; fy < filterSize; ++fy)
      {
        for (int fx = 0; fx < filterSize; ++fx)
        {
          int ox = (ix + padding - fx);
          int oy = (iy + padding - fy);

          if (ox % stride == 0 && oy % stride == 0)
          {
            ox /= stride;
            oy /= stride;

            if (ox >= 0 && ox < outputWidth && oy >= 0 && oy < outputHeight)
            {
              int outputIdx = (f * outputHeight + oy) * outputWidth + ox;
              int weightIdx = ((f * inputChannels + c) * filterSize + fy) * filterSize + fx;

              if (output[outputIdx] > 0)
              {
                grad += outputGradient[outputIdx] * weights[weightIdx];
              }
            }
          }
        }
      }
    }

    int inputGradIdx = (c * inputHeight + iy) * inputWidth + ix;
    atomicAdd(&inputGradient[inputGradIdx], grad);
  }
}

__global__ void conv_backward_kernel_sequential(
    const float *input, const float *output, const float *outputGradient, float *inputGradient,
    const float *weights, float *weightGradients, float *biasGradients,
    int inputHeight, int inputWidth, int inputChannels,
    int outputHeight, int outputWidth, int numFilters,
    int filterSize, int stride, int padding)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int f = 0; f < numFilters; ++f)
    {
      for (int c = 0; c < inputChannels; ++c)
      {
        for (int fy = 0; fy < filterSize; ++fy)
        {
          for (int fx = 0; fx < filterSize; ++fx)
          {
            float grad = 0.0f;

            for (int oy = 0; oy < outputHeight; ++oy)
            {
              for (int ox = 0; ox < outputWidth; ++ox)
              {
                int inputY = oy * stride - padding + fy;
                int inputX = ox * stride - padding + fx;

                if (inputY >= 0 && inputY < inputHeight &&
                    inputX >= 0 && inputX < inputWidth)
                {
                  int inputIdx = (c * inputHeight + inputY) * inputWidth + inputX;
                  int outputIdx = (f * outputHeight + oy) * outputWidth + ox;

                  if (output[outputIdx] > 0)
                  {
                    grad += outputGradient[outputIdx] * input[inputIdx];
                  }
                }
              }
            }

            int weightIdx = ((f * inputChannels + c) * filterSize + fy) * filterSize + fx;
            weightGradients[weightIdx] += grad;
          }
        }
      }
    }

    for (int f = 0; f < numFilters; ++f)
    {
      float grad = 0.0f;
      for (int oy = 0; oy < outputHeight; ++oy)
      {
        for (int ox = 0; ox < outputWidth; ++ox)
        {
          int outputIdx = (f * outputHeight + oy) * outputWidth + ox;
          if (output[outputIdx] > 0)
          {
            grad += outputGradient[outputIdx];
          }
        }
      }
      biasGradients[f] += grad;
    }

    for (int c = 0; c < inputChannels; ++c)
    {
      for (int iy = 0; iy < inputHeight; ++iy)
      {
        for (int ix = 0; ix < inputWidth; ++ix)
        {
          float grad = 0.0f;

          for (int f = 0; f < numFilters; ++f)
          {
            for (int fy = 0; fy < filterSize; ++fy)
            {
              for (int fx = 0; fx < filterSize; ++fx)
              {
                int ox = (ix + padding - fx);
                int oy = (iy + padding - fy);

                if (ox % stride == 0 && oy % stride == 0)
                {
                  ox /= stride;
                  oy /= stride;

                  if (ox >= 0 && ox < outputWidth &&
                      oy >= 0 && oy < outputHeight)
                  {
                    int outputIdx = (f * outputHeight + oy) * outputWidth + ox;
                    int weightIdx = ((f * inputChannels + c) * filterSize + fy) * filterSize + fx;

                    if (output[outputIdx] > 0)
                    {
                      grad += outputGradient[outputIdx] * weights[weightIdx];
                    }
                  }
                }
              }
            }
          }

          int inputGradIdx = (c * inputHeight + iy) * inputWidth + ix;
          inputGradient[inputGradIdx] += grad;
        }
      }
    }
  }
}

ConvolutionalLayer::ConvolutionalLayer(
    const std::vector<size_t> &inputShape,
    size_t numFilters,
    size_t filterSize,
    size_t stride,
    size_t padding) : inputShape_(inputShape),
                      numFilters_(numFilters),
                      filterSize_(filterSize),
                      stride_(stride),
                      padding_(padding),
                      rng_(std::random_device{}())
{
  size_t inputHeight = inputShape_[0];
  size_t inputWidth = inputShape_[1];
  size_t inputChannels = inputShape_[2];

  size_t outputHeight = (inputHeight + 2 * padding_ - filterSize_) / stride_ + 1;
  size_t outputWidth = (inputWidth + 2 * padding_ - filterSize_) / stride_ + 1;

  outputShape_ = {outputHeight, outputWidth, numFilters_};

  filterVolume_ = filterSize_ * filterSize_ * inputChannels;
  weightsSize_ = filterVolume_ * numFilters_;

  CUDA_CHECK(cudaMalloc(&d_weights_, sizeof(float) * weightsSize_));
  CUDA_CHECK(cudaMalloc(&d_biases_, sizeof(float) * numFilters_));
  CUDA_CHECK(cudaMalloc(&d_weightGradients_, sizeof(float) * weightsSize_));
  CUDA_CHECK(cudaMalloc(&d_biasGradients_, sizeof(float) * numFilters_));

  initializeWeights();

  CUDA_CHECK(cudaMemset(d_weightGradients_, 0, sizeof(float) * weightsSize_));
  CUDA_CHECK(cudaMemset(d_biasGradients_, 0, sizeof(float) * numFilters_));
}

ConvolutionalLayer::~ConvolutionalLayer()
{
  if (d_weights_)
    cudaFree(d_weights_);
  if (d_biases_)
    cudaFree(d_biases_);
  if (d_weightGradients_)
    cudaFree(d_weightGradients_);
  if (d_biasGradients_)
    cudaFree(d_biasGradients_);
}

void ConvolutionalLayer::forward(const float *input, float *output, cudaStream_t stream, bool seq)
{
  int inputHeight = inputShape_[0];
  int inputWidth = inputShape_[1];
  int inputChannels = inputShape_[2];

  int outputHeight = outputShape_[0];
  int outputWidth = outputShape_[1];

  cudaMemsetAsync(output, 0, sizeof(float) * outputHeight * outputWidth * numFilters_, stream);

  if (seq)
  {
    conv_forward_kernel_sequential<<<1, 1, 0, stream>>>(
        input, output,
        d_weights_, d_biases_,
        inputHeight, inputWidth, inputChannels,
        outputHeight, outputWidth, numFilters_,
        filterSize_, stride_, padding_);
  }
  else
  {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (outputWidth + blockDim.x - 1) / blockDim.x,
        (outputHeight + blockDim.y - 1) / blockDim.y,
        numFilters_);

    conv_forward_kernel<<<gridDim, blockDim, 0, stream>>>(
        input, output,
        d_weights_, d_biases_,
        inputHeight, inputWidth, inputChannels,
        outputHeight, outputWidth, numFilters_,
        filterSize_, stride_, padding_);
  }

  CUDA_CHECK(cudaGetLastError());
}

void ConvolutionalLayer::backward(const float *input, const float *output,
                                  const float *outputGradient, float *inputGradient, cudaStream_t stream, bool seq)
{
  int inputHeight = inputShape_[0];
  int inputWidth = inputShape_[1];
  int inputChannels = inputShape_[2];

  int outputHeight = outputShape_[0];
  int outputWidth = outputShape_[1];

  cudaMemsetAsync(inputGradient, 0, sizeof(float) * inputHeight * inputWidth * inputChannels, stream);

  dim3 blockDim(filterSize_, filterSize_);
  dim3 gridDim(BLOCK_SIZE, BLOCK_SIZE, numFilters_);

  if (seq)
  {
    conv_backward_kernel_sequential<<<1, 1, 0, stream>>>(
        input, output, outputGradient, inputGradient,
        d_weights_, d_weightGradients_, d_biasGradients_,
        inputHeight, inputWidth, inputChannels,
        outputHeight, outputWidth, numFilters_,
        filterSize_, stride_, padding_);
  }
  else
  {
    dim3 blockDim(filterSize_, filterSize_);
    dim3 gridDim(
        (inputWidth + blockDim.x - 1) / blockDim.x,
        (inputHeight + blockDim.y - 1) / blockDim.y,
        inputChannels);

    conv_backward_kernel<<<gridDim, blockDim, 0, stream>>>(
        input, output, outputGradient, inputGradient,
        d_weights_, d_weightGradients_, d_biasGradients_,
        inputHeight, inputWidth, inputChannels,
        outputHeight, outputWidth, numFilters_,
        filterSize_, stride_, padding_);
  }

  CUDA_CHECK(cudaGetLastError());
}

__global__ void updateKernel(float *weights, float *gradients, int size, float lr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    weights[idx] -= lr * gradients[idx];
  }
}

__global__ void updateKernel_sequential(float *weights, float *gradients, int size, float lr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    for (int i = 0; i < size; ++i)
    {
      weights[i] -= lr * gradients[i];
    }
  }
}

void ConvolutionalLayer::updateWeights(float learningRate, const float *d_outputGradient, cudaStream_t stream, bool seq)
{
  int weightsSize = weightsSize_;
  int numFilters = numFilters_;

  if (seq)
  {
    updateKernel_sequential<<<1, 1, 0, stream>>>(d_weights_, d_weightGradients_, weightsSize, learningRate);
    updateKernel_sequential<<<1, 1, 0, stream>>>(d_biases_, d_biasGradients_, numFilters, learningRate);
  }
  else
  {
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim((weightsSize + blockDim.x - 1) / blockDim.x);
    updateKernel<<<gridDim, blockDim, 0, stream>>>(d_weights_, d_weightGradients_, weightsSize, learningRate);

    gridDim = dim3((numFilters + blockDim.x - 1) / blockDim.x);
    updateKernel<<<gridDim, blockDim, 0, stream>>>(d_biases_, d_biasGradients_, numFilters, learningRate);
  }

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemsetAsync(d_weightGradients_, 0, sizeof(float) * weightsSize, stream));
  CUDA_CHECK(cudaMemsetAsync(d_biasGradients_, 0, sizeof(float) * numFilters, stream));

  CUDA_CHECK(cudaGetLastError());
}

std::vector<size_t> ConvolutionalLayer::getOutputShape() const
{
  return outputShape_;
}

std::vector<size_t> ConvolutionalLayer::getInputShape() const
{
  return inputShape_;
}

void ConvolutionalLayer::loadWeights(const std::vector<float> &weights)
{
  if (weights.size() != weightsSize_ + numFilters_)
  {
    std::cerr << "Error: Weight vector size does not match layer parameters" << std::endl;
    return;
  }

  CUDA_CHECK(cudaMemcpy(d_weights_, weights.data(), sizeof(float) * weightsSize_, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biases_, weights.data() + weightsSize_, sizeof(float) * numFilters_, cudaMemcpyHostToDevice));
}

std::vector<float> ConvolutionalLayer::getWeights() const
{
  std::vector<float> weights(weightsSize_ + numFilters_);

  CUDA_CHECK(cudaMemcpy(weights.data(), d_weights_, sizeof(float) * weightsSize_, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(weights.data() + weightsSize_, d_biases_, sizeof(float) * numFilters_, cudaMemcpyDeviceToHost));

  return weights;
}

void ConvolutionalLayer::initializeWeights()
{
  std::vector<float> weights(weightsSize_);
  std::vector<float> biases(numFilters_, 0.0f);

  float scale = sqrtf(2.0f / (filterSize_ * filterSize_ * inputShape_[2]));
  std::normal_distribution<float> dist(0.0f, scale);

  for (size_t i = 0; i < weightsSize_; ++i)
  {
    weights[i] = dist(rng_);
  }

  CUDA_CHECK(cudaMemcpy(d_weights_, weights.data(), sizeof(float) * weightsSize_, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biases_, biases.data(), sizeof(float) * numFilters_, cudaMemcpyHostToDevice));
}
