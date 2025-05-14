cmake_minimum_required(VERSION 3.20)
project(cuda_cnn_coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75;80;86)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(cuda_cnn_coursework
        src/main.cu
        src/utils/benchmark.cu
        src/utils/image_utils.cu
        src/utils/timer.cu
        src/utils/calc_layer_dim.cu
        src/utils/prepare_data.cu
        src/utils/prepare_model.cu
        src/core/cnn.cu
        src/core/trainer.cu
        src/layers/convolutional.cu
        src/layers/fc.cu
        src/layers/pool.cu
        src/dataset/images_dataset.cu
        src/dataset/euro_sat_dataset.cu
)

target_compile_options(cuda_cnn_coursework PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

set_target_properties(cuda_cnn_coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
