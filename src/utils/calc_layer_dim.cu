#include "utils/calc_layer_dim.h"

CalcLayerDim::LayerDimensions CalcLayerDim::calcConvOutputs(size_t inputWidth, size_t inputHeight, size_t filterSize, size_t stride, size_t padding)
{
  return {
      (inputWidth - filterSize + 2 * padding) / stride + 1,
      (inputHeight - filterSize + 2 * padding) / stride + 1};
}

CalcLayerDim::LayerDimensions CalcLayerDim::calcPoolOutputs(size_t inputWidth, size_t inputHeight, size_t poolSize, size_t stride)
{
  return {
      (inputWidth - poolSize) / stride + 1,
      (inputHeight - poolSize) / stride + 1};
}
