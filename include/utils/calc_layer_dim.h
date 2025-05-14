#pragma once

namespace CalcLayerDim
{
  struct LayerDimensions
  {
    size_t w;
    size_t h;
  };

  LayerDimensions calcConvOutputs(size_t inputWidth, size_t inputHeight, size_t filterSize, size_t stride, size_t padding);

  LayerDimensions calcPoolOutputs(size_t inputWidth, size_t inputHeight, size_t poolSize, size_t stride);
}
