#pragma once

#include "core/cnn.h"

#include "layers/layer.h"
#include "layers/convolutional.h"
#include "layers/pool.h"
#include "layers/fc.h"

#include "utils/timer.h"
#include "utils/calc_layer_dim.h"

namespace PrepareModel
{
  constexpr int CONV1_FILTERS = 32;
  constexpr int CONV1_SIZE = 5;
  constexpr int POOL1_SIZE = 2;
  constexpr int POOL1_STRIDE = 2;

  constexpr int CONV2_FILTERS = 64;
  constexpr int CONV2_SIZE = 5;
  constexpr int POOL2_SIZE = 2;
  constexpr int POOL2_STRIDE = 2;

  constexpr int FC1_SIZE = 2048;

  LayerPtr createConvLayer(const std::vector<size_t> &inputSize, int numFilters, int filterSize);
  LayerPtr createPoolLayer(const std::vector<size_t> &inputSize, int poolSize, int poolStride);
  LayerPtr createFullyConnectedLayer(int inputSize, int outputSize, bool useReLU = true);

  CNN prepare(size_t imageSize, size_t numClasses, size_t numChannels, bool quiet = true);
}
