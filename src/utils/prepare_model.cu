#include "utils/prepare_model.h"

LayerPtr PrepareModel::createConvLayer(const std::vector<size_t> &inputSize, int numFilters, int filterSize)
{
  return std::make_shared<ConvolutionalLayer>(inputSize, numFilters, filterSize);
}

LayerPtr PrepareModel::createPoolLayer(const std::vector<size_t> &inputSize, int poolSize, int poolStride)
{
  return std::make_shared<PoolingLayer>(inputSize, poolSize, poolStride);
}

LayerPtr PrepareModel::createFullyConnectedLayer(int inputSize, int outputSize, bool useReLU)
{
  return std::make_shared<FullyConnectedLayer>(inputSize, outputSize);
}

CNN PrepareModel::prepare(size_t imageSize, size_t numClasses, size_t numChannels, bool quiet)
{
  auto conv1 = CalcLayerDim::calcConvOutputs(imageSize, imageSize, CONV1_SIZE, 1, 0);
  auto pool1 = CalcLayerDim::calcPoolOutputs(conv1.w, conv1.h, POOL1_SIZE, POOL1_STRIDE);
  auto conv2 = CalcLayerDim::calcConvOutputs(pool1.w, pool1.h, CONV2_SIZE, 1, 0);
  auto pool2 = CalcLayerDim::calcPoolOutputs(conv2.w, conv2.h, POOL2_SIZE, POOL2_STRIDE);

  size_t pool2Size = CONV2_FILTERS * pool2.w * pool2.h;

  if (!quiet)
  {
    std::cout << "Setting up model... ";
  }

  Timer timer;
  timer.start();

  LayerPtr cnn1_layer = createConvLayer({imageSize, imageSize, numChannels}, CONV1_FILTERS, CONV1_SIZE);
  LayerPtr pool1_layer = createPoolLayer({conv1.w, conv1.h, CONV1_FILTERS}, POOL1_SIZE, POOL1_STRIDE);
  LayerPtr cnn2_layer = createConvLayer({pool1.w, pool1.h, CONV1_FILTERS}, CONV2_FILTERS, CONV2_SIZE);
  LayerPtr pool2_layer = createPoolLayer({conv2.w, conv2.h, CONV2_FILTERS}, POOL2_SIZE, POOL2_STRIDE);
  LayerPtr fc1_layer = createFullyConnectedLayer(pool2Size, FC1_SIZE, true);
  LayerPtr fc2_layer = createFullyConnectedLayer(FC1_SIZE, numClasses, false);

  CNN cnn;
  cnn.addLayer(cnn1_layer);
  cnn.addLayer(pool1_layer);
  cnn.addLayer(cnn2_layer);
  cnn.addLayer(pool2_layer);
  cnn.addLayer(fc1_layer);
  cnn.addLayer(fc2_layer);

  if (!quiet)
  {
    std::cout << timer.elapsed() << std::endl
              << std::endl;
  }

  return cnn;
}
