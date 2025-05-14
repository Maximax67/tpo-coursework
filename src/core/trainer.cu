#include "core/trainer.h"

Trainer::Trainer(CNN &model,
                 const std::vector<Image> &trainImages,
                 const std::vector<int> &trainLabels,
                 const std::vector<Image> &testImages,
                 const std::vector<int> &testLabels,
                 const std::vector<Image> &validationImages,
                 const std::vector<int> &validationLabels,
                 int numClasses,
                 int batchSize,
                 float learningRate,
                 int epochs,
                 bool useBatchMode)
    : cnn(model), trainImages(trainImages), trainLabels(trainLabels),
      testImages(testImages), testLabels(testLabels),
      validationImages(validationImages), validationLabels(validationLabels),
      numClasses(numClasses), batchSize(batchSize),
      learningRate(learningRate), epochs(epochs),
      useBatchMode(useBatchMode) {}

std::vector<float> Trainer::labelToOneHot(int label)
{
  std::vector<float> oneHot(numClasses, 0.0f);
  if (label >= 0 && label < numClasses)
    oneHot[label] = 1.0f;

  return oneHot;
}

void Trainer::train()
{
  resetTiming();

  int numSamples = trainImages.size();

  std::cout << "Starting training on " << numSamples << " samples for " << epochs << " epochs" << std::endl;
  std::cout << std::fixed << std::setprecision(4);

  auto startTime = std::chrono::high_resolution_clock::now();

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    std::cout << "\nEpoch " << (epoch + 1) << "/" << epochs << std::endl;

    auto epochStart = std::chrono::high_resolution_clock::now();
    std::vector<double> currentBatchTimes;
    std::vector<double> currentSampleTimes;

    float totalLoss = 0.0f;
    float totalAccuracy = 0.0f;
    float epochAccuracy = 0.0f;

    if (useBatchMode)
    {
      int numBatches = (numSamples + batchSize - 1) / batchSize;
      for (int batch = 0; batch < numBatches; ++batch)
      {
        auto batchStart = std::chrono::high_resolution_clock::now();

        int start = batch * batchSize;
        int end = std::min(start + batchSize, numSamples);
        int size = end - start;

        std::vector<std::vector<float>> images(size);
        std::vector<std::vector<float>> labels(size);

        for (int i = 0; i < size; ++i)
        {
          images[i] = trainImages[start + i].data;
          labels[i] = labelToOneHot(trainLabels[start + i]);
        }

        CNN::TrainResult result = cnn.trainBatch(images, labels, learningRate);

        totalLoss += result.loss;
        totalAccuracy += result.accuracy * size;

        auto batchEnd = std::chrono::high_resolution_clock::now();
        currentBatchTimes.push_back(std::chrono::duration<double>(batchEnd - batchStart).count());
      }
    }
    else
    {
      for (int i = 0; i < numSamples; ++i)
      {
        auto sampleStart = std::chrono::high_resolution_clock::now();

        std::vector<float> image = trainImages[i].data;
        std::vector<float> label = labelToOneHot(trainLabels[i]);

        CNN::TrainResult result = cnn.train(image, label, learningRate);

        totalLoss += result.loss;
        totalAccuracy += result.accuracy;

        auto sampleEnd = std::chrono::high_resolution_clock::now();
        currentSampleTimes.push_back(std::chrono::duration<double>(sampleEnd - sampleStart).count());
      }
    }

    float epochLoss = totalLoss / numSamples;

    auto epochEnd = std::chrono::high_resolution_clock::now();
    timing.epochTimes.push_back(std::chrono::duration<double>(epochEnd - epochStart).count());
    timing.batchTimes.push_back(currentBatchTimes);
    timing.sampleTimes.push_back(currentSampleTimes);

    float valAccuracy = evaluateDataset(validationImages, validationLabels, true);

    std::cout << "Epoch " << (epoch + 1) << " training loss: " << epochLoss << ", Train accuracy: " << epochAccuracy << ", Validation accuracy: " << valAccuracy << std::endl;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
  std::cout << "\nTraining completed in " << duration << " seconds" << std::endl;
}

float Trainer::evaluateTestDataset(bool quiet)
{
  return evaluateDataset(testImages, testLabels, quiet);
}

float Trainer::evaluateDataset(const std::vector<Image> &images, const std::vector<int> &labels, bool quiet)
{
  int correct = 0;
  int total = images.size();
  std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));

  if (useBatchMode)
  {
    int numBatches = (total + batchSize - 1) / batchSize;
    for (int b = 0; b < numBatches; ++b)
    {
      int start = b * batchSize;
      int end = std::min(start + batchSize, total);
      int size = end - start;

      std::vector<std::vector<float>> inputs(size);
      std::vector<int> targets(size);

      for (int i = 0; i < size; ++i)
      {
        inputs[i] = images[start + i].data;
        targets[i] = labels[start + i];
      }

      std::vector<int> preds = cnn.predictBatch(inputs);

      for (int i = 0; i < size; ++i)
      {
        int pred = preds[i];
        int target = targets[i];
        confusionMatrix[target][pred]++;
        if (pred == target)
        {
          correct++;
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < total; ++i)
    {
      std::vector<float> input = images[i].data;
      int target = labels[i];
      int pred = cnn.predict(input);
      confusionMatrix[target][pred]++;
      if (pred == target)
      {
        correct++;
      }
    }
  }

  float accuracy = static_cast<float>(correct) / total;

  if (quiet)
  {
    return accuracy;
  }

  std::cout << "Final evaluation: " << correct << " correct out of " << total
            << " (" << accuracy * 100.0f << "%)" << std::endl;

  std::vector<float> precision(numClasses, 0.0f);
  std::vector<float> recall(numClasses, 0.0f);
  std::vector<float> f1(numClasses, 0.0f);

  for (int i = 0; i < numClasses; ++i)
  {
    int TP = confusionMatrix[i][i];
    int FP = 0;
    int FN = 0;

    for (int j = 0; j < numClasses; ++j)
    {
      if (j != i)
      {
        FP += confusionMatrix[j][i];
        FN += confusionMatrix[i][j];
      }
    }

    if (TP + FP > 0)
      precision[i] = static_cast<float>(TP) / (TP + FP);
    if (TP + FN > 0)
      recall[i] = static_cast<float>(TP) / (TP + FN);
    if (precision[i] + recall[i] > 0)
      f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]);
  }

  std::cout << "\nClass-wise Evaluation:\n";
  for (int i = 0; i < numClasses; ++i)
  {
    std::cout << "Class " << i << " - Precision: " << precision[i]
              << ", Recall: " << recall[i] << ", F1 Score: " << f1[i] << std::endl;
  }

  float totalPrecision = 0.0f;
  float totalRecall = 0.0f;
  float totalF1 = 0.0f;

  for (int i = 0; i < numClasses; ++i)
  {
    totalPrecision += precision[i];
    totalRecall += recall[i];
    totalF1 += f1[i];
  }

  totalPrecision /= numClasses;
  totalRecall /= numClasses;
  totalF1 /= numClasses;

  std::cout << "\nOverall Metrics:\n";
  std::cout << "Total Precision: " << totalPrecision
            << ", Total Recall: " << totalRecall
            << ", Total F1 Score: " << totalF1 << std::endl;

  std::cout << std::endl
            << "Confusion Matrix (Heatmap):" << std::endl;
  printConfusionMatrix(confusionMatrix);

  return accuracy;
}

void Trainer::printConfusionMatrix(const std::vector<std::vector<int>> &confusionMatrix)
{
  int numClasses = confusionMatrix.size();

  std::cout << std::setw(12) << "Pred/True";
  for (int i = 0; i < numClasses; ++i)
    std::cout << std::setw(6) << i;
  std::cout << std::endl;

  std::cout << std::setw(12) << "  ";
  for (int i = 0; i < numClasses; ++i)
    std::cout << "------";
  std::cout << std::endl;

  for (int i = 0; i < numClasses; ++i)
  {
    std::cout << std::setw(10) << i << " |";

    for (int j = 0; j < numClasses; ++j)
    {
      std::cout << std::setw(6) << confusionMatrix[i][j];
    }
    std::cout << std::endl;
  }
}

void Trainer::setBatchMode(bool isBatchMode)
{
  useBatchMode = isBatchMode;
}

bool Trainer::isBatchMode()
{
  return useBatchMode;
}

void Trainer::printTimingInfo()
{
  std::cout << "\n==== Timing Summary ====" << std::endl;
  for (size_t epoch = 0; epoch < timing.epochTimes.size(); ++epoch)
  {
    std::cout << "Epoch " << (epoch + 1)
              << " took " << std::fixed << std::setprecision(4)
              << timing.epochTimes[epoch] << " seconds\n";

    if (useBatchMode)
    {
      const auto &batchTimes = timing.batchTimes[epoch];
      if (!batchTimes.empty())
      {
        double sum = std::accumulate(batchTimes.begin(), batchTimes.end(), 0.0);
        double avg = sum / batchTimes.size();
        double min = *std::min_element(batchTimes.begin(), batchTimes.end());
        double max = *std::max_element(batchTimes.begin(), batchTimes.end());

        std::cout << "Batch timings (avg/min/max): "
                  << avg << " / " << min << " / " << max << " s" << std::endl;
      }
    }
    else
    {
      const auto &sampleTimes = timing.sampleTimes[epoch];
      if (!sampleTimes.empty())
      {
        double sum = std::accumulate(sampleTimes.begin(), sampleTimes.end(), 0.0);
        double avg = sum / sampleTimes.size();
        double min = *std::min_element(sampleTimes.begin(), sampleTimes.end());
        double max = *std::max_element(sampleTimes.begin(), sampleTimes.end());

        std::cout << "Sample timings (avg/min/max): "
                  << avg << " / " << min << " / " << max << " s" << std::endl;
      }
    }
  }

  std::cout << std::endl;
}

void Trainer::resetTiming()
{
  timing.epochTimes.clear();
  timing.batchTimes.clear();
  timing.sampleTimes.clear();
}
