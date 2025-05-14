#include "dataset/euro_sat_dataset.h"

EuroSatDataset::EuroSatDataset() : ImagesDataset()
{
  imageSize = 64;
  channels = 3;
  numClasses = 10;
}

bool EuroSatDataset::loadData(const std::string &dataPath)
{
  std::vector<EuroSatData> data;

  if (!loadEuroSatCSV(dataPath, data))
  {
    return false;
  }

  for (const auto &entry : data)
  {
    std::string filePath = (std::filesystem::path(dataPath).parent_path() / entry.filename).string();
    Image img = ImageUtils::loadImage(filePath, false);
    images.push_back(img);
    labels.push_back(entry.label);
  }

  return true;
}

bool EuroSatDataset::loadEuroSatCSV(const std::string &csvFile, std::vector<EuroSatData> &data)
{
  std::ifstream file(csvFile);
  if (!file.is_open())
  {
    return false;
  }

  std::string line;
  std::getline(file, line);

  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string idx, filename, label, className;
    std::getline(ss, idx, ',');
    std::getline(ss, filename, ',');
    std::getline(ss, label, ',');
    std::getline(ss, className, ',');

    EuroSatData entry;
    entry.filename = filename;
    entry.label = std::stoi(label);
    data.push_back(entry);
  }

  return true;
}
