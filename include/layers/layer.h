#pragma once

#include <vector>
#include <string>
#include <memory>

class Layer
{
public:
  Layer() = default;
  virtual ~Layer() = default;

  virtual void forward(const float *input, float *output, cudaStream_t stream, bool seq = false) = 0;
  virtual void backward(const float *input, const float *output,
                        const float *outputGradient, float *inputGradient, cudaStream_t stream, bool seq = false) = 0;
  virtual void updateWeights(float learningRate, const float *gradient, cudaStream_t stream, bool seq = false) = 0;

  virtual std::vector<size_t> getOutputShape() const = 0;
  virtual std::vector<size_t> getInputShape() const = 0;

  virtual std::string getName() const = 0;

  virtual void loadWeights(const std::vector<float> &weights) = 0;
  virtual std::vector<float> getWeights() const = 0;
};

using LayerPtr = std::shared_ptr<Layer>;
