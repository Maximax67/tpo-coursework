#include "dataset/images_dataset.h"

void ImagesDataset::shuffle()
{
  std::random_device rd;
  std::mt19937 g(rd());

  std::vector<size_t> indices(images.size());
  for (size_t i = 0; i < indices.size(); i++)
  {
    indices[i] = i;
  }

  std::shuffle(indices.begin(), indices.end(), g);

  std::vector<Image> shuffledImages(images.size());
  std::vector<int> shuffledLabels(labels.size());

  for (size_t i = 0; i < indices.size(); i++)
  {
    shuffledImages[i] = images[indices[i]];
    shuffledLabels[i] = labels[indices[i]];
  }

  images = std::move(shuffledImages);
  labels = std::move(shuffledLabels);
}

void ImagesDataset::normalizeData()
{
  for (Image &img : images)
  {
    ImageUtils::normalizeImage(img);
  }
}

size_t ImagesDataset::getImageCount() const { return images.size(); }
size_t ImagesDataset::getImageSize() const { return imageSize; }
size_t ImagesDataset::getChannels() const { return channels; }
size_t ImagesDataset::getNumClasses() const { return numClasses; }

const std::vector<Image> &ImagesDataset::getImages() const { return images; }
const std::vector<int> &ImagesDataset::getLabels() const { return labels; }
