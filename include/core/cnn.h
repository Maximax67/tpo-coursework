#pragma once

#include "layers/layer.h"
#include "utils/cuda_utils.h"

#include <fstream>
#include <iostream>
#include <numeric>
#include <future>

class CNN
{
public:
  CNN();
  ~CNN();

  struct TrainResult
  {
    float loss;
    float accuracy;
  };

  void addLayer(LayerPtr layer);

  const std::vector<LayerPtr> &getLayers() const { return layers_; }

  TrainResult train(const std::vector<float> &input, const std::vector<float> &target, float learningRate);
  TrainResult trainBatch(const std::vector<std::vector<float>> &batchInputs,
                         const std::vector<std::vector<float>> &batchTargets,
                         float learningRate);

  int predict(const std::vector<float> &input);
  std::vector<int> predictBatch(const std::vector<std::vector<float>> &batchInputs);

  float evaluateTestDataset(const std::vector<std::vector<float>> &inputs,
                            const std::vector<int> &labels);
  float evaluateBatch(const std::vector<std::vector<float>> &batchInputs,
                      const std::vector<int> &batchLabels);

  void saveModel(const std::string &filename);

  bool loadModel(const std::string &filename);

  void printArchitecture() const;

private:
  std::vector<LayerPtr> layers_;

  std::vector<float *> d_layerOutputs_;
  std::vector<float *> d_layerGradients_;

  std::vector<std::vector<float *>> d_batch_layerOutputs_;
  std::vector<std::vector<float *>> d_batch_layerGradients_;
  std::vector<size_t> d_batch_layerGradients_sizes_;

  std::vector<cudaStream_t> streams;
  std::vector<float *> d_batch_accumulatedGradients;

  std::vector<float>
  forward(const std::vector<float> &input, cudaStream_t stream);
  std::vector<std::vector<float>> CNN::forwardBatch(const std::vector<std::vector<float>> &batchInputs, std::vector<cudaStream_t> streams);

  void allocateMemory();
  void allocateMemoryBatch(size_t batchSize);

  void freeMemory();
  void freeMemoryBatch();

  TrainResult computeLossAndAccuracy(const float *output, const float *target, float *gradient, size_t size);

  bool initializeCuda_;
  int currentBatchSize_;

  void setBatchSize(int batchSize);
};
