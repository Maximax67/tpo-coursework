#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#include "utils/image_utils.h"

namespace ImageUtils
{
  Image loadImage(const std::string &filename, bool grayscale)
  {
    Image image;

    int width, height, channels;
    unsigned char *data = stbi_load(
        filename.c_str(),
        &width, &height, &channels,
        grayscale ? STBI_grey : STBI_rgb);

    if (!data)
    {
      std::cerr << "Error loading image: " << filename << std::endl;
      std::cerr << "STB Error: " << stbi_failure_reason() << std::endl;
      return image;
    }

    image.width = width;
    image.height = height;
    image.channels = grayscale ? 1 : 3;

    size_t dataSize = width * height * image.channels;
    image.data.resize(dataSize);

    for (size_t i = 0; i < dataSize; ++i)
    {
      image.data[i] = static_cast<float>(data[i]) / 255.0f;
    }

    stbi_image_free(data);

    return image;
  }

  void normalizeImage(Image &image)
  {
    if (image.data.empty())
    {
      return;
    }

    float minVal = image.data[0];
    float maxVal = image.data[0];

    for (float val : image.data)
    {
      minVal = std::min(minVal, val);
      maxVal = std::max(maxVal, val);
    }

    if (maxVal > minVal)
    {
      for (float &val : image.data)
      {
        val = (val - minVal) / (maxVal - minVal);
      }
    }
  }
}
