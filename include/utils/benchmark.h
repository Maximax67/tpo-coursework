#pragma once

#include "layers/layer.h"
#include "layers/layer.h"
#include "layers/convolutional.h"
#include "layers/pool.h"
#include "layers/fc.h"

#include "utils/timer.h"
#include "utils/cuda_utils.h"

#include <numeric>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>

struct BenchmarkStats
{
  float minTime;
  float maxTime;
  float avgTime;

  void print(const std::string &title) const;
};

class Benchmark
{
public:
  static BenchmarkStats run(std::shared_ptr<Layer> layer,
                            const std::vector<float> &input,
                            const std::vector<float> &gradOutput,
                            bool batchMode,
                            size_t inputSize,
                            size_t outputSize,
                            int batchSize = 32,
                            int iterations = 5,
                            int numSamples = 2048);

  static void benchmarkFullyConnected();
  static void benchmarkConvolutional();
  static void benchmarkPooling();
  static void runAll();
  static void warmup(int iterations = 5, int inSize = 128, int outSize = 128);

private:
  static double runSingle(std::shared_ptr<Layer> layer,
                         const std::vector<float> &input,
                         const std::vector<float> &gradOutput,
                         size_t inputSize,
                         size_t outputSize,
                         int numSamples);

  static double runBatch(std::shared_ptr<Layer> layer,
                         const std::vector<float> &input,
                         const std::vector<float> &gradOutput,
                         size_t inputSize,
                         size_t outputSize,
                         int batchSize,
                         int numSamples);

  static void printTable(const std::map<std::string, std::map<int, BenchmarkStats>> &results, const std::string &metric);
};
