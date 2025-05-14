#include "core/cnn.h"

__global__ void accumulateGradientsKernel(float *accumulated, const float *grad, size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    atomicAdd(&accumulated[idx], grad[idx]);
  }
}

CNN::CNN() : initializeCuda_(false)
{
  initializeCuda_ = initCUDA();
  if (!initializeCuda_)
  {
    std::cerr << "Failed to initialize CUDA" << std::endl;
  }

  currentBatchSize_ = 0;
}

CNN::~CNN()
{
  freeMemory();
  freeMemoryBatch();

  for (cudaStream_t &stream : streams)
  {
    cudaStreamDestroy(stream);
  }
}

void CNN::addLayer(LayerPtr layer)
{
  layers_.push_back(layer);
}

void CNN::allocateMemory()
{
  if (layers_.empty())
  {
    return;
  }

  freeMemory();
  freeMemoryBatch();

  for (size_t i = 0; i < layers_.size(); ++i)
  {
    float *d_output = nullptr;
    float *d_gradient = nullptr;

    size_t outputSize = 1;
    for (auto dim : layers_[i]->getOutputShape())
    {
      outputSize *= dim;
    }

    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradient, outputSize * sizeof(float)));

    d_layerOutputs_.push_back(d_output);
    d_layerGradients_.push_back(d_gradient);
  }

  float *d_inputGradients = nullptr;
  size_t inputSize = 1;
  for (auto dim : layers_[0]->getInputShape())
  {
    inputSize *= dim;
  }

  CUDA_CHECK(cudaMalloc(&d_inputGradients, inputSize * sizeof(float)));
  d_layerGradients_.insert(d_layerGradients_.begin(), d_inputGradients);
}

void CNN::allocateMemoryBatch(size_t batchSize)
{
  if (layers_.empty())
    return;

  freeMemory();
  freeMemoryBatch();

  std::vector<float *> inputGradients(batchSize);
  size_t inputSize = 1;
  for (auto dim : layers_[0]->getInputShape())
    inputSize *= dim;

  for (size_t j = 0; j < batchSize; ++j)
  {
    float *d_inputGrad = nullptr;
    CUDA_CHECK(cudaMalloc(&d_inputGrad, inputSize * sizeof(float)));
    inputGradients[j] = d_inputGrad;
  }

  d_batch_layerGradients_.push_back(inputGradients);
  d_batch_layerGradients_sizes_.push_back(inputSize);

  for (size_t i = 0; i < layers_.size(); ++i)
  {
    std::vector<float *> batchOutputs(batchSize);
    std::vector<float *> batchGradients(batchSize);

    size_t outputSize = 1;
    for (auto dim : layers_[i]->getOutputShape())
      outputSize *= dim;

    for (size_t j = 0; j < batchSize; ++j)
    {
      float *d_output = nullptr;
      float *d_gradient = nullptr;

      CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_gradient, outputSize * sizeof(float)));

      batchOutputs[j] = d_output;
      batchGradients[j] = d_gradient;
    }

    d_batch_layerOutputs_.push_back(batchOutputs);
    d_batch_layerGradients_.push_back(batchGradients);
    d_batch_layerGradients_sizes_.push_back(outputSize);
  }

  d_batch_accumulatedGradients.resize(layers_.size());
  for (size_t l = 0; l < layers_.size(); ++l)
  {
    size_t gradSize = d_batch_layerGradients_sizes_[l + 1];
    CUDA_CHECK(cudaMalloc(&d_batch_accumulatedGradients[l], gradSize * sizeof(float)));
  }
}

void CNN::freeMemory()
{
  for (auto d_output : d_layerOutputs_)
  {
    if (d_output)
    {
      cudaFree(d_output);
    }
  }

  for (auto d_gradient : d_layerGradients_)
  {
    if (d_gradient)
    {
      cudaFree(d_gradient);
    }
  }

  d_layerOutputs_.clear();
  d_layerGradients_.clear();
}

void CNN::freeMemoryBatch()
{
  for (auto d_batch_output : d_batch_layerOutputs_)
  {
    for (auto d_output : d_batch_output)
    {
      if (d_output)
      {
        cudaFree(d_output);
      }
    }
  }

  for (auto d_batch_gradient : d_batch_layerGradients_)
  {
    for (auto d_gradient : d_batch_gradient)
    {
      if (d_gradient)
      {
        cudaFree(d_gradient);
      }
    }
  }

  for (auto d_accumulated : d_batch_accumulatedGradients)
  {
    if (d_accumulated)
    {
      cudaFree(d_accumulated);
    }
  }

  d_batch_layerOutputs_.clear();
  d_batch_layerGradients_.clear();
  d_batch_layerGradients_sizes_.clear();
  d_batch_accumulatedGradients.clear();
}

void CNN::setBatchSize(int batchSize)
{
  if (currentBatchSize_ == batchSize)
  {
    return;
  }

  currentBatchSize_ = batchSize;
  allocateMemoryBatch(batchSize);

  if (streams.size() == batchSize)
  {
    return;
  }

  streams.resize(batchSize);
  for (size_t i = 0; i < batchSize; ++i)
  {
    if (streams[i] == nullptr)
    {
      cudaStreamCreate(&streams[i]);
    }
  }
}

std::vector<float> CNN::forward(const std::vector<float> &input, cudaStream_t stream)
{
  if (layers_.empty())
  {
    return input;
  }

  if (d_layerOutputs_.empty())
  {
    allocateMemory();
  }

  size_t expectedSize = 1;
  for (auto dim : layers_[0]->getInputShape())
    expectedSize *= dim;

  size_t inputSize = input.size() * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(d_layerOutputs_[0], input.data(), inputSize, cudaMemcpyHostToDevice, stream));

  for (size_t i = 0; i < layers_.size(); ++i)
  {
    const float *layerInput = (i == 0) ? d_layerOutputs_[0] : d_layerOutputs_[i - 1];
    float *layerOutput = d_layerOutputs_[i];
    layers_[i]->forward(layerInput, layerOutput, stream, true);
  }

  size_t outputSize = 1;
  for (auto dim : layers_.back()->getOutputShape())
  {
    outputSize *= dim;
  }

  std::vector<float> output(outputSize);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(output.data(), d_layerOutputs_.back(), outputSize * sizeof(float), cudaMemcpyDeviceToHost));

  return output;
}

std::vector<std::vector<float>> CNN::forwardBatch(const std::vector<std::vector<float>> &batchInputs, std::vector<cudaStream_t> streams)
{
  size_t batchSize = batchInputs.size();
  if (batchSize == 0 || layers_.empty())
    return {};

  setBatchSize(batchSize);

  for (size_t b = 0; b < batchSize; ++b)
  {
    float *inputDevice = d_batch_layerOutputs_[0][b];
    CUDA_CHECK(cudaMemcpyAsync(inputDevice, batchInputs[b].data(),
                               batchInputs[b].size() * sizeof(float),
                               cudaMemcpyHostToDevice, streams[b]));
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    for (size_t l = 0; l < layers_.size(); ++l)
    {
      float *input = (l == 0) ? d_batch_layerOutputs_[0][b] : d_batch_layerOutputs_[l - 1][b];
      float *output = d_batch_layerOutputs_[l][b];
      layers_[l]->forward(input, output, streams[b]);
    }
  }

  size_t outputSize = 1;
  for (auto dim : layers_.back()->getOutputShape())
    outputSize *= dim;

  std::vector<std::vector<float>> outputs(batchSize, std::vector<float>(outputSize));

  for (size_t b = 0; b < batchSize; ++b)
  {
    CUDA_CHECK(cudaStreamSynchronize(streams[b]));
    CUDA_CHECK(cudaMemcpyAsync(outputs[b].data(), d_batch_layerOutputs_.back()[b],
                               outputSize * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[b]));
  }

  return outputs;
}

CNN::TrainResult CNN::computeLossAndAccuracy(const float *output, const float *target, float *gradient, size_t size)
{
  std::vector<float> outputHost(size);
  std::vector<float> targetHost(size);

  cudaMemcpy(outputHost.data(), output, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(targetHost.data(), target, size * sizeof(float), cudaMemcpyDeviceToHost);

  float maxVal = *std::max_element(outputHost.begin(), outputHost.end());

  std::vector<float> expOutput(size);
  float sumExp = 0.0f;

  for (size_t i = 0; i < size; ++i)
  {
    expOutput[i] = std::exp(outputHost[i] - maxVal);
    sumExp += expOutput[i];
  }

  std::vector<float> softmax(size);
  for (size_t i = 0; i < size; ++i)
  {
    softmax[i] = expOutput[i] / sumExp;
  }

  float loss = 0.0f;
  for (size_t i = 0; i < size; ++i)
  {
    if (targetHost[i] > 0.0f)
    {
      loss -= std::log(std::max(softmax[i], 1e-10f)) * targetHost[i];
    }
  }

  std::vector<float> gradientHost(size);
  for (size_t i = 0; i < size; ++i)
  {
    gradientHost[i] = softmax[i] - targetHost[i];
  }

  cudaMemcpy(gradient, gradientHost.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  size_t predictedClass = std::distance(softmax.begin(), std::max_element(softmax.begin(), softmax.end()));
  size_t targetClass = std::distance(targetHost.begin(), std::max_element(targetHost.begin(), targetHost.end()));

  float isCorrect = (predictedClass == targetClass) ? 1.0f : 0.0f;

  return {loss, isCorrect};
}

CNN::TrainResult CNN::train(const std::vector<float> &input, const std::vector<float> &target, float learningRate)
{
  if (d_layerOutputs_.empty())
  {
    allocateMemory();
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  forward(input, stream);

  size_t outputSize = 1;
  for (auto dim : layers_.back()->getOutputShape())
  {
    outputSize *= dim;
  }

  float *d_target = nullptr;
  CUDA_CHECK(cudaMalloc(&d_target, target.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_target, target.data(), target.size() * sizeof(float), cudaMemcpyHostToDevice));

  TrainResult res = computeLossAndAccuracy(d_layerOutputs_.back(), d_target, d_layerGradients_.back(), outputSize);

  for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
  {
    const float *layerInput = (i == 0) ? d_layerOutputs_[0] : d_layerOutputs_[i - 1];
    const float *layerOutput = d_layerOutputs_[i];
    const float *outputGradient = d_layerGradients_[i + 1];
    float *inputGradient = d_layerGradients_[i];

    layers_[i]->backward(layerInput, layerOutput, outputGradient, inputGradient, stream, true);

    cudaStreamSynchronize(stream);

    layers_[i]->updateWeights(learningRate, outputGradient, stream, true);
  }

  cudaFreeAsync(d_target, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaStreamDestroy(stream);

  return res;
}

CNN::TrainResult CNN::trainBatch(const std::vector<std::vector<float>> &batchInputs,
                                 const std::vector<std::vector<float>> &batchTargets,
                                 float learningRate)
{
  size_t batchSize = batchInputs.size();
  if (batchSize == 0 || layers_.empty())
  {
    return {0.0f, 0.0f};
  }

  setBatchSize(batchSize);

  for (size_t l = 0; l < layers_.size(); ++l)
  {
    cudaStream_t stream = streams[l % batchSize];
    size_t gradSize = d_batch_layerGradients_sizes_[l + 1];
    CUDA_CHECK(cudaMemsetAsync(d_batch_accumulatedGradients[l], 0, gradSize * sizeof(float), stream));
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    float *inputDevice = d_batch_layerOutputs_[0][b];
    CUDA_CHECK(cudaMemcpyAsync(inputDevice, batchInputs[b].data(),
                               batchInputs[b].size() * sizeof(float),
                               cudaMemcpyHostToDevice, streams[b]));
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    for (size_t l = 0; l < layers_.size(); ++l)
    {
      float *input = (l == 0) ? d_batch_layerOutputs_[0][b] : d_batch_layerOutputs_[l - 1][b];
      float *output = d_batch_layerOutputs_[l][b];
      layers_[l]->forward(input, output, streams[b]);
    }
  }

  size_t outputSize = 1;
  for (auto dim : layers_.back()->getOutputShape())
    outputSize *= dim;

  std::vector<float *> d_targets(batchSize), d_outputGradients(batchSize);
  for (size_t b = 0; b < batchSize; ++b)
  {
    CUDA_CHECK(cudaMallocAsync(&d_targets[b], outputSize * sizeof(float), streams[b]));
    CUDA_CHECK(cudaMemcpyAsync(d_targets[b], batchTargets[b].data(),
                               outputSize * sizeof(float), cudaMemcpyHostToDevice, streams[b]));

    d_outputGradients[b] = d_batch_layerGradients_.back()[b];
  }

  std::vector<float> losses(batchSize);
  std::vector<std::future<TrainResult>> futures(batchSize);

  for (size_t b = 0; b < batchSize; ++b)
  {
    cudaStreamSynchronize(streams[b]);
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    futures[b] = std::async(std::launch::async, [&, b]()
                            { return computeLossAndAccuracy(d_batch_layerOutputs_.back()[b],
                                                            d_targets[b],
                                                            d_outputGradients[b],
                                                            outputSize); });
  }

  float totalTrainAccuracy = 0.0f;
  for (size_t b = 0; b < batchSize; ++b)
  {
    TrainResult res = futures[b].get();
    losses[b] = res.loss;
    totalTrainAccuracy += res.accuracy;
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l)
    {
      float *input = (l == 0) ? d_batch_layerOutputs_[0][b] : d_batch_layerOutputs_[l - 1][b];
      float *output = d_batch_layerOutputs_[l][b];
      float *gradOutput = (l == static_cast<int>(layers_.size()) - 1)
                              ? d_outputGradients[b]
                              : d_batch_layerGradients_[l + 1][b];
      float *gradInput = d_batch_layerGradients_[l][b];

      layers_[l]->backward(input, output, gradOutput, gradInput, streams[b]);

      size_t gradOutputSize = d_batch_layerGradients_sizes_[l + 1];
      int gridSize = (gradOutputSize + BLOCK_DIM - 1) / BLOCK_DIM;

      accumulateGradientsKernel<<<gridSize, BLOCK_DIM, 0, streams[b]>>>(
          d_batch_accumulatedGradients[l],
          gradOutput,
          gradOutputSize);

      CUDA_CHECK(cudaGetLastError());
    }
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    cudaStreamSynchronize(streams[b]);
  }

  for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l)
  {
    cudaStream_t stream = streams[l % batchSize];
    layers_[l]->updateWeights(learningRate, d_batch_accumulatedGradients[l], stream);
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    cudaStreamSynchronize(streams[b]);
    cudaFreeAsync(d_targets[b], streams[b]);
  }

  float loss = std::accumulate(losses.begin(), losses.end(), 0.0f);
  float trainAccuracy = totalTrainAccuracy / batchSize;

  return {loss, trainAccuracy};
}

int CNN::predict(const std::vector<float> &input)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<float> output = forward(input, stream);

  cudaStreamDestroy(stream);

  return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}

float CNN::evaluateTestDataset(const std::vector<std::vector<float>> &inputs, const std::vector<int> &labels)
{
  if (inputs.size() != labels.size() || inputs.empty())
  {
    return 0.0f;
  }

  int correctPredictions = 0;

  for (size_t i = 0; i < inputs.size(); ++i)
  {
    int prediction = predict(inputs[i]);
    if (prediction == labels[i])
    {
      correctPredictions++;
    }
  }

  return static_cast<float>(correctPredictions) / inputs.size();
}

std::vector<int> CNN::predictBatch(const std::vector<std::vector<float>> &batchInputs)
{
  size_t batchSize = batchInputs.size();
  if (batchSize == 0 || layers_.empty())
    return {};

  setBatchSize(batchSize);

  for (size_t b = 0; b < batchSize; ++b)
  {
    CUDA_CHECK(cudaMemcpyAsync(d_batch_layerOutputs_[0][b], batchInputs[b].data(),
                               batchInputs[b].size() * sizeof(float), cudaMemcpyHostToDevice, streams[b]));
  }

  for (size_t b = 0; b < batchSize; ++b)
  {
    for (size_t l = 0; l < layers_.size(); ++l)
    {
      float *input = (l == 0) ? d_batch_layerOutputs_[0][b] : d_batch_layerOutputs_[l - 1][b];
      float *output = d_batch_layerOutputs_[l][b];
      layers_[l]->forward(input, output, streams[b]);
    }
  }

  std::vector<int> predictions(batchSize);
  size_t outputSize = 1;
  for (auto dim : layers_.back()->getOutputShape())
    outputSize *= dim;

  for (size_t b = 0; b < batchSize; ++b)
  {
    std::vector<float> output(outputSize);
    CUDA_CHECK(cudaMemcpyAsync(output.data(), d_batch_layerOutputs_.back()[b],
                               outputSize * sizeof(float), cudaMemcpyDeviceToHost, streams[b]));

    cudaStreamSynchronize(streams[b]);
    predictions[b] = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
  }

  return predictions;
}

float CNN::evaluateBatch(const std::vector<std::vector<float>> &batchInputs, const std::vector<int> &batchLabels)
{
  if (batchInputs.size() != batchLabels.size() || batchInputs.empty())
  {
    return 0.0f;
  }

  std::vector<int> predictions = predictBatch(batchInputs);

  int correct = 0;
  for (size_t i = 0; i < batchLabels.size(); ++i)
  {
    if (predictions[i] == batchLabels[i])
      ++correct;
  }

  return static_cast<float>(correct) / batchLabels.size();
}

void CNN::saveModel(const std::string &filename)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file)
  {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  size_t numLayers = layers_.size();
  file.write(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));

  for (const auto &layer : layers_)
  {
    std::string layerName = layer->getName();
    size_t nameLength = layerName.size();
    file.write(reinterpret_cast<char *>(&nameLength), sizeof(nameLength));
    file.write(layerName.c_str(), nameLength);

    std::vector<float> weights = layer->getWeights();
    size_t weightsSize = weights.size();
    file.write(reinterpret_cast<char *>(&weightsSize), sizeof(weightsSize));
    file.write(reinterpret_cast<char *>(weights.data()), weightsSize * sizeof(float));
  }

  file.close();
}

bool CNN::loadModel(const std::string &filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    std::cerr << "Failed to open file for loading model: " << filename << std::endl;
    return false;
  }

  size_t numLayers;
  file.read(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));

  if (numLayers != layers_.size())
  {
    std::cerr << "Model architecture mismatch. Expected " << layers_.size()
              << " layers, but found " << numLayers << " in file." << std::endl;
    file.close();
    return false;
  }

  for (size_t i = 0; i < numLayers; ++i)
  {
    size_t nameLength;
    file.read(reinterpret_cast<char *>(&nameLength), sizeof(nameLength));
    std::string layerName(nameLength, ' ');
    file.read(&layerName[0], nameLength);

    if (layerName != layers_[i]->getName())
    {
      std::cerr << "Layer type mismatch at index " << i << ". Expected "
                << layers_[i]->getName() << " but found " << layerName << std::endl;
      file.close();
      return false;
    }

    size_t weightsSize;
    file.read(reinterpret_cast<char *>(&weightsSize), sizeof(weightsSize));
    std::vector<float> weights(weightsSize);
    file.read(reinterpret_cast<char *>(weights.data()), weightsSize * sizeof(float));

    layers_[i]->loadWeights(weights);
  }

  file.close();
  return true;
}

void CNN::printArchitecture() const
{
  for (const auto &layer : layers_)
  {
    std::vector<size_t> outputShape = layer->getOutputShape();
    std::vector<size_t> inputShape = layer->getInputShape();

    std::string layerName = layer->getName();

    std::cout << layerName << ":\n";

    std::cout << "  Input Shape: (";
    for (size_t i = 0; i < inputShape.size(); ++i)
    {
      std::cout << inputShape[i];
      if (i < inputShape.size() - 1)
        std::cout << ", ";
    }
    std::cout << ")\n";

    std::cout << "  Output Shape: (";
    for (size_t i = 0; i < outputShape.size(); ++i)
    {
      std::cout << outputShape[i];
      if (i < outputShape.size() - 1)
        std::cout << ", ";
    }
    std::cout << ")\n";
  }
}
