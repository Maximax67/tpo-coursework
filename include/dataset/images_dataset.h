#pragma once

#include "utils/image_utils.h"

#include <vector>
#include <string>
#include <iostream>
#include <random>

class ImagesDataset
{
public:
  ImagesDataset() : imageSize(0), channels(0), numClasses(0) {}

  virtual ~ImagesDataset() {}
  virtual bool loadData(const std::string &dataDir) = 0;

  void normalizeData();
  void shuffle();

  size_t getImageCount() const;
  size_t getImageSize() const;
  size_t getChannels() const;
  size_t getNumClasses() const;

  const std::vector<Image> &getImages() const;
  const std::vector<int> &getLabels() const;

protected:
  size_t imageSize;
  size_t channels;
  size_t numClasses;

  std::vector<Image> images;
  std::vector<int> labels;
};
