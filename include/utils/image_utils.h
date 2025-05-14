#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <vector>
#include <string>
#include <filesystem>
#include <iostream>

struct Image
{
  std::vector<float> data;
  int width;
  int height;
  int channels;

  size_t size() const { return width * height * channels; }
};

namespace ImageUtils
{
  Image loadImage(const std::string &filename, bool grayscale = false);
  void normalizeImage(Image &image);
}

#endif // IMAGE_UTILS_H
