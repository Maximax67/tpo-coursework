#include "utils/prepare_data.h"

bool PrepareData::prepare(EuroSatDataset &train_dataset, EuroSatDataset &test_dataset, EuroSatDataset &validation_dataset, bool quiet)
{
  if (!quiet)
  {
    std::cout << "Loading EuroSat dataset... ";
  }

  Timer timer;
  timer.start();

  if (!train_dataset.loadData(trainData))
  {
    std::cerr << "\nFailed to load train EuroSat dataset!" << std::endl;
    return false;
  }

  if (!validation_dataset.loadData(validationData))
  {
    std::cerr << "\nFailed to load validation EuroSat dataset!" << std::endl;
    return false;
  }

  if (!test_dataset.loadData(testData))
  {
    std::cerr << "\nFailed to load test EuroSat dataset!" << std::endl;
    return false;
  }

  if (!quiet)
  {
    std::cout << timer.elapsed() << std::endl;
    std::cout << "Train dataset: " << train_dataset.getImageCount() << " images" << std::endl;
    std::cout << "Validation dataset: " << validation_dataset.getImageCount() << " images" << std::endl;
    std::cout << "Test dataset: " << test_dataset.getImageCount() << " images" << std::endl;
    std::cout << "Normalizing data... ";
  }

  timer.start();
  train_dataset.normalizeData();
  test_dataset.normalizeData();

  if (!quiet)
  {
    std::cout << timer.elapsed() << std::endl
              << "Shuffling train data... ";
  }

  timer.start();
  train_dataset.shuffle();

  if (!quiet)
  {
    std::cout << timer.elapsed() << std::endl;
  }

  return true;
};
