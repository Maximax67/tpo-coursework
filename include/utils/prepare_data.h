#pragma once

#include "dataset/euro_sat_dataset.h"
#include "utils/timer.h"

namespace PrepareData
{
  const std::string trainData = "D:/CUDA-CNN-Coursework/EuroSAT/train.csv";
  const std::string testData = "D:/CUDA-CNN-Coursework/EuroSAT/test.csv";
  const std::string validationData = "D:/CUDA-CNN-Coursework/EuroSAT/validation.csv";

  bool prepare(EuroSatDataset &train_dataset, EuroSatDataset &test_dataset, EuroSatDataset &validation_dataset, bool quiet = true);
}
