#pragma once

#include "layer.h"

class FullyConnectedLayer : public Layer
{
public:
  FullyConnectedLayer(size_t inputSize, size_t outputSize, bool useReLU = true);
  ~FullyConnectedLayer();

  void forward(const float *input, float *output, cudaStream_t stream, bool seq = false) override;
  void backward(const float *input, const float *output,
                const float *outputGradient, float *inputGradient, cudaStream_t stream, bool seq = false) override;
  void updateWeights(float learningRate, const float *d_outputGradient, cudaStream_t stream, bool seq = false) override;

  std::vector<size_t> getOutputShape() const override;
  std::vector<size_t> getInputShape() const override;
  std::string getName() const override;

  void loadWeights(const std::vector<float> &weights) override;
  std::vector<float> getWeights() const override;

private:
  size_t inputSize_;
  size_t outputSize_;
  bool useReLU_;

  std::vector<float> weights_;
  std::vector<float> biases_;

  float *d_weights_ = nullptr;
  float *d_biases_ = nullptr;
  float *d_output_ = nullptr;

  void initializeWeights();
  void applyReLU(float *data, size_t size);
  void applyReLUDerivative(const float *output, float *gradient, size_t size);
};
