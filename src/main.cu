#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "core/cnn.h"
#include "core/trainer.h"

#include "utils/prepare_data.h"
#include "utils/prepare_model.h"
#include "utils/benchmark.h"

#include "dataset/euro_sat_dataset.h"

constexpr int NUM_EPOCHS = 20;
constexpr int BATCH_SIZE = 128;
constexpr float LEARNING_RATE = 0.001f;

int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    int choice;

    std::cout << "Select mode:" << std::endl;
    std::cout << "1. Benchmark" << std::endl;
    std::cout << "2. Train" << std::endl;
    std::cout << "3. Evaluate" << std::endl;
    std::cout << "Enter value: ";
    std::cin >> choice;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << std::endl
              << "Using GPU: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl
              << std::endl;

    if (choice == 1)
    {
        Benchmark::runAll();
        return 0;
    }

    if (choice != 2 && choice != 3)
    {
        std::cout << "Invalid option!" << std::endl;
        return 1;
    }

    EuroSatDataset train_dataset, test_dataset, validation_dataset;
    if (!PrepareData::prepare(train_dataset, test_dataset, validation_dataset, false))
    {
        return 1;
    }

    const std::vector<Image> &trainImages = train_dataset.getImages();
    const std::vector<int> &trainLabels = train_dataset.getLabels();
    const std::vector<Image> &testImages = test_dataset.getImages();
    const std::vector<int> &testLabels = test_dataset.getLabels();
    const std::vector<Image> &validationImages = validation_dataset.getImages();
    const std::vector<int> &validationLabels = validation_dataset.getLabels();

    const size_t imageSize = train_dataset.getImageSize();
    const size_t numClasses = train_dataset.getNumClasses();
    const size_t numChannels = train_dataset.getChannels();

    CNN cnn = PrepareModel::prepare(imageSize, numClasses, numChannels, false);

    std::cout << "--- Model Architecture ---" << std::endl;
    cnn.printArchitecture();
    std::cout << std::endl;

    Trainer trainer(cnn, trainImages, trainLabels, testImages, testLabels, validationImages, validationLabels,
                    numClasses, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS);

    if (choice == 2)
    {
        trainer.setBatchMode(true);
        trainer.train();
        trainer.printTimingInfo();

        trainer.evaluateTestDataset();

        cnn.saveModel("model.txt");

        return 0;
    }

    cnn.loadModel("model.txt");

    trainer.evaluateTestDataset();

    return 0;
}
