#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include "dataset/images_dataset.h"
#include "utils/image_utils.h"

class EuroSatDataset : public ImagesDataset
{
public:
  EuroSatDataset();
  bool loadData(const std::string &dataDir) override;

private:
  struct EuroSatData
  {
    std::string filename;
    int label;
  };

  bool loadEuroSatCSV(const std::string &csvFile, std::vector<EuroSatData> &data);
};
