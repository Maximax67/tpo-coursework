#pragma once

#include "layer.h"
#include "utils/cuda_utils.h"

#include <random>

class ConvolutionalLayer : public Layer
{
public:
  ConvolutionalLayer(
      const std::vector<size_t> &inputShape,
      size_t numFilters,
      size_t filterSize,
      size_t stride = 1,
      size_t padding = 0);

  ~ConvolutionalLayer();

  void forward(const float *input, float *output, cudaStream_t stream, bool seq = false) override;
  void backward(const float *input, const float *output,
                const float *outputGradient, float *inputGradient, cudaStream_t stream, bool seq = false) override;
  void updateWeights(float learningRate, const float *d_outputGradient, cudaStream_t stream, bool seq = false) override;

  std::vector<size_t> getOutputShape() const override;
  std::vector<size_t> getInputShape() const override;

  std::string getName() const override { return "Convolutional"; }

  void loadWeights(const std::vector<float> &weights) override;
  std::vector<float> getWeights() const override;

private:
  std::vector<size_t> inputShape_;
  std::vector<size_t> outputShape_;

  size_t numFilters_;
  size_t filterSize_;
  size_t stride_;
  size_t padding_;

  float *d_weights_ = nullptr;
  float *d_biases_ = nullptr;

  float *d_weightGradients_ = nullptr;
  float *d_biasGradients_ = nullptr;

  size_t filterVolume_;
  size_t weightsSize_;
  std::mt19937 rng_;

  void initializeWeights();
};
