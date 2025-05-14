#pragma once

#include "layer.h"
#include "utils/cuda_utils.h"

#include <iostream>

class PoolingLayer : public Layer
{
public:
  PoolingLayer(
      const std::vector<size_t> &inputShape,
      size_t poolSize,
      size_t stride = 0);

  ~PoolingLayer();

  void forward(const float *input, float *output, cudaStream_t stream, bool seq = false) override;
  void backward(const float *input, const float *output,
                const float *outputGradient, float *inputGradient, cudaStream_t stream, bool seq = false) override;

  void updateWeights(float learningRate, const float *d_outputGradient, cudaStream_t stream, bool seq = false) override {}

  std::vector<size_t> getOutputShape() const override;
  std::vector<size_t> getInputShape() const override;

  std::string getName() const override { return "MaxPooling"; }

  void loadWeights(const std::vector<float> &weights) override {}
  std::vector<float> getWeights() const override { return {}; }

private:
  std::vector<size_t> inputShape_;
  std::vector<size_t> outputShape_;

  size_t poolSize_;
  size_t stride_;

  int *d_maxIndices_ = nullptr;
};