#include "utils/benchmark.h"

void BenchmarkStats::print(const std::string &title) const
{
  std::cout << title << ":\n";
  std::cout << "  Min Time: " << minTime << " ms\n";
  std::cout << "  Max Time: " << maxTime << " ms\n";
  std::cout << "  Avg Time: " << avgTime << " ms\n\n";
}

BenchmarkStats Benchmark::run(std::shared_ptr<Layer> layer,
                              const std::vector<float> &input,
                              const std::vector<float> &gradOutput,
                              bool batchMode,
                              size_t inputSize,
                              size_t outputSize,
                              int batchSize,
                              int iterations,
                              int numSamples)
{
  std::vector<float> timings;
  timings.reserve(iterations);

  for (int i = 0; i < iterations; ++i)
  {
    float elapsed = batchMode
                        ? runBatch(layer, input, gradOutput, inputSize, outputSize, batchSize, numSamples)
                        : runSingle(layer, input, gradOutput, inputSize, outputSize, numSamples);
    timings.push_back(elapsed);
  }

  float minTime = *std::min_element(timings.begin(), timings.end());
  float maxTime = *std::max_element(timings.begin(), timings.end());
  float avgTime = std::accumulate(timings.begin(), timings.end(), 0.0f) / iterations;

  return {minTime, maxTime, avgTime};
}

double Benchmark::runSingle(std::shared_ptr<Layer> layer,
                            const std::vector<float> &input,
                            const std::vector<float> &gradOutput,
                            size_t inputSize,
                            size_t outputSize,
                            int numSamples)
{
  float *d_input = nullptr, *d_output = nullptr;
  float *d_gradOut = nullptr, *d_gradIn = nullptr;
  cudaStream_t stream;

  cudaStreamCreate(&stream);
  CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gradOut, outputSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gradIn, inputSize * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gradOut, gradOutput.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice));

  Timer timer;
  timer.start();

  for (int i = 0; i < numSamples; ++i)
  {
    layer->forward(d_input, d_output, stream);
    cudaStreamSynchronize(stream);
    layer->backward(d_input, d_output, d_gradOut, d_gradIn, stream);
    cudaStreamSynchronize(stream);
  }

  double elapsed = timer.ms();

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_gradOut);
  cudaFree(d_gradIn);
  cudaStreamDestroy(stream);

  return elapsed;
}

double Benchmark::runBatch(std::shared_ptr<Layer> layer,
                           const std::vector<float> &input,
                           const std::vector<float> &gradOutput,
                           size_t inputSize,
                           size_t outputSize,
                           int batchSize,
                           int numSamples)
{
  std::vector<float *> d_inputs(batchSize), d_outputs(batchSize);
  std::vector<float *> d_gradOuts(batchSize), d_gradIns(batchSize);
  std::vector<cudaStream_t> streams(batchSize);

  for (int b = 0; b < batchSize; ++b)
  {
    CUDA_CHECK(cudaMalloc(&d_inputs[b], inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs[b], outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradOuts[b], outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradIns[b], inputSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inputs[b], input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gradOuts[b], gradOutput.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaStreamCreate(&streams[b]);
  }

  int numBatches = numSamples / batchSize;
  int lastBatchSize = numSamples % batchSize;

  if (lastBatchSize == 0)
  {
    lastBatchSize = batchSize;
  }

  Timer timer;
  timer.start();

  for (int i = 0; i < numBatches; ++i)
  {
    int currentBatchSize = (i == numBatches - 1) ? lastBatchSize : batchSize;

    for (int b = 0; b < currentBatchSize; ++b)
    {
      layer->forward(d_inputs[b], d_outputs[b], streams[b], true);
      layer->backward(d_inputs[b], d_outputs[b], d_gradOuts[b], d_gradIns[b], streams[b], true);
    }

    cudaDeviceSynchronize();
  }

  double elapsed = timer.ms();

  for (int b = 0; b < batchSize; ++b)
  {
    cudaFree(d_inputs[b]);
    cudaFree(d_outputs[b]);
    cudaFree(d_gradOuts[b]);
    cudaFree(d_gradIns[b]);
    cudaStreamDestroy(streams[b]);
  }

  return elapsed;
}

void Benchmark::benchmarkFullyConnected()
{
  std::vector<std::pair<int, int>> configs = {
      {512, 512}, {1024, 1024}, {4096, 4096}, {8192, 8192}};

  std::vector<int> batchSizes = {8, 16, 32, 64, 128, 256};

  for (auto [inSize, outSize] : configs)
  {
    auto layer = std::make_shared<FullyConnectedLayer>(inSize, outSize);
    std::vector<float> input(inSize, 1.0f);
    std::vector<float> gradOut(outSize, 0.5f);

    std::string desc = "FC Layer " + std::to_string(inSize) + "x" + std::to_string(outSize);

    auto statsSingle = run(layer, input, gradOut, false, inSize, outSize);
    statsSingle.print(desc);

    for (int batchSize : batchSizes)
    {
      std::vector<float> batchInput(inSize * batchSize, 1.0f);
      std::vector<float> batchGrad(outSize * batchSize, 0.5f);

      auto statsBatch = run(layer, batchInput, batchGrad, true, inSize, outSize, batchSize);
      statsBatch.print(desc + ", Batch " + std::to_string(batchSize));
    }
  }
}

void Benchmark::benchmarkConvolutional()
{
  std::vector<std::tuple<int, int, int, int>> configs = {
      {32, 3, 3, 16}, {64, 3, 3, 16}, {128, 3, 3, 16}, {256, 3, 3, 16}};

  std::vector<int> batchSizes = {8, 16, 32, 64, 128, 256};

  for (auto &[size, kernel, stride, filters] : configs)
  {
    std::vector<size_t> inputConvSize = {(size_t)size, (size_t)size, 3};
    auto layer = std::make_shared<ConvolutionalLayer>(inputConvSize, filters, kernel, stride, 1);
    size_t inputSize = size * size * 3;
    size_t outputSize = layer->getOutputShape()[0] *
                        layer->getOutputShape()[1] *
                        layer->getOutputShape()[2];

    std::vector<float> input(inputSize, 1.0f);
    std::vector<float> gradOut(outputSize, 0.5f);

    std::string desc = "Conv Layer " + std::to_string(size) + "x" + std::to_string(size) + "x3";

    auto statsSingle = run(layer, input, gradOut, false, inputSize, outputSize);
    statsSingle.print(desc);

    for (int batchSize : batchSizes)
    {
      std::vector<float> batchInput(inputSize * batchSize, 1.0f);
      std::vector<float> batchGrad(outputSize * batchSize, 0.5f);

      auto statsBatch = run(layer, batchInput, batchGrad, true, inputSize, outputSize, batchSize);
      statsBatch.print(desc + ", Batch " + std::to_string(batchSize));
    }
  }
}

void Benchmark::benchmarkPooling()
{
  std::vector<std::tuple<size_t, size_t, size_t>> configs = {
      {32, 2, 2}, {64, 2, 2}, {128, 2, 2}, {256, 2, 2}};

  std::vector<int> batchSizes = {8, 16, 32, 64, 128, 256};

  for (const auto &[size, kernel, stride] : configs)
  {
    std::vector<size_t> inputShape = {size, size, 3};
    auto layer = std::make_shared<PoolingLayer>(inputShape, kernel, stride);
    size_t inputSize = size * size * 3;
    size_t outputSize = layer->getOutputShape()[0] *
                        layer->getOutputShape()[1] *
                        layer->getOutputShape()[2];

    std::vector<float> input(inputSize, 1.0f);
    std::vector<float> gradOut(outputSize, 0.5f);

    std::string desc = "Pooling Layer " + std::to_string(size) + "x" + std::to_string(size) + "x3";

    auto statsSingle = run(layer, input, gradOut, false, inputSize, outputSize);
    statsSingle.print(desc);

    for (int batchSize : batchSizes)
    {
      std::vector<float> batchInput(inputSize * batchSize, 1.0f);
      std::vector<float> batchGrad(outputSize * batchSize, 0.5f);

      auto statsBatch = run(layer, batchInput, batchGrad, true, inputSize, outputSize, batchSize);
      statsBatch.print(desc + ", Batch " + std::to_string(batchSize));
    }
  }
}

void Benchmark::warmup(int iterations, int inSize, int outSize)
{
  auto layer = std::make_shared<FullyConnectedLayer>(inSize, outSize);
  std::vector<float> input(inSize, 1.0f);
  std::vector<float> gradOut(outSize, 0.5f);

  for (int i = 0; i < iterations; ++i)
  {
    run(layer, input, gradOut, false, inSize, outSize);
  }
}

void Benchmark::runAll()
{
  std::cout << "--- Warming up GPU ---\n";
  warmup();
  std::cout << "--- Warm up complete ---\n";

  std::cout << "--- Benchmarking Fully Connected Layers ---\n";
  benchmarkFullyConnected();

  std::cout << "--- Benchmarking Convolutional Layers ---\n";
  benchmarkConvolutional();

  std::cout << "--- Benchmarking Pooling Layers ---\n";
  benchmarkPooling();
}
