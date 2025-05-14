#pragma once

#include "core/cnn.h"
#include "utils/image_utils.h"

#include <iostream>
#include <chrono>
#include <iomanip>

class Trainer
{
public:
  Trainer(CNN &model,
          const std::vector<Image> &trainImages,
          const std::vector<int> &trainLabels,
          const std::vector<Image> &testImages,
          const std::vector<int> &testLabels,
          const std::vector<Image> &validationImages,
          const std::vector<int> &validationLabels,
          int numClasses,
          int batchSize = 100,
          float learningRate = 0.0001f,
          int epochs = 5,
          bool useBatchMode = true);

  void train();
  float evaluateTestDataset(bool quiet = false);
  float evaluateDataset(const std::vector<Image> &images, const std::vector<int> &labels, bool quiet = false);
  void setBatchMode(bool isBatchMode);
  bool isBatchMode();
  void printTimingInfo();

private:
  struct TimingInfo {
    std::vector<double> epochTimes;
    std::vector<std::vector<double>> batchTimes;
    std::vector<std::vector<double>> sampleTimes;
  };

  void resetTiming();
  std::vector<float> labelToOneHot(int label);
  void printConfusionMatrix(const std::vector<std::vector<int>> &confusionMatrix);

  CNN &cnn;

  const std::vector<Image> &trainImages;
  const std::vector<int> &trainLabels;
  const std::vector<Image> &testImages;
  const std::vector<int> &testLabels;
  const std::vector<Image> &validationImages;
  const std::vector<int> &validationLabels;

  int numClasses;
  int batchSize;
  float learningRate;
  int epochs;
  bool useBatchMode;

  TimingInfo timing;
};
