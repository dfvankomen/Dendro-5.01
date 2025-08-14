#include "compression.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "asyncExchangeContex.h"
#include "blosc.h"
#include "mpi.h"
#ifdef DENDRO_ENABLE_ML_LIBRARIES
#include "onnxruntime_cxx_api.h"
#endif
#include "zfp.h"
#include "zfp/bitstream.h"

#define BATCH_SIZE 16

// #define __DENDRO_ZFP_USE_TRUE_4D__

namespace MachineLearningAlgorithms {

ONNXCompression onnxcomp(6, 2);

template <typename T>
size_t ONNXCompression::do_3d_compression(const T* originalMatrix,
                                          unsigned char* outputArray,
                                          size_t batchSize) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ONNX Compression only accepts doubles or floats as inputs!");

#ifdef DENDRO_ENABLE_ML_LIBRARIES
    if (!m_3d_encoder) {
        throw std::runtime_error(
            "ERROR: The 3D encoder is not initialized! Make sure set_models() "
            "is called.");
    }

    // update batch size
    m_input_shape_3d[0] = batchSize;

    // only resize if we're going to use this buffer
    if constexpr (std::is_same_v<T, double>) {
        if (m_doubleToFloatBuffer_3d.size() < m_total3DPts * batchSize) {
            m_doubleToFloatBuffer_3d.resize(m_total3DPts * batchSize);
        }
    }

    Ort::Value tensor_data = createOnnxTensorFromData(
        originalMatrix, m_total3DPts, batchSize, m_doubleToFloatBuffer_3d,
        m_memory_info, m_input_shape_3d);

    const float* tensor_values = tensor_data.GetTensorData<float>();

    const char* input_names[]  = {m_input_name_3d.c_str()};
    const char* output_names[] = {m_output_name_3d.c_str()};

    auto output = m_3d_encoder->Run(Ort::RunOptions{nullptr}, input_names,
                                    &tensor_data, 1, output_names, 1);

    const float* output_data = output[0].GetTensorData<float>();

    std::memcpy(outputArray, output_data,
                m_nOuts3dEncoder * sizeof(float) * batchSize);
    return batchSize * m_nOuts3dEncoder * sizeof(float);

#else
    std::memcpy(outputArray, originalMatrix,
                m_total3DPts * sizeof(T) * batchSize);
    return m_total3DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t ONNXCompression::do_3d_decompression(
    const unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ONNX Compression only accepts doubles or floats as inputs!");

#ifdef DENDRO_ENABLE_ML_LIBRARIES
    if (!m_3d_decoder) {
        throw std::runtime_error(
            "ERROR: The 3D decoder is not initialized! Make sure set_models() "
            "is called.");
    }

    const float* floatInputArray =
        reinterpret_cast<const float*>(compressedBuffer);

    m_decoder_shape_3d[0] = batchSize;

    if (m_doubleToFloatBuffer_3d.size() < m_total3DPts * batchSize) {
        m_doubleToFloatBuffer_3d.resize(m_total3DPts * batchSize);
    }

    Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
        m_memory_info, const_cast<float*>(floatInputArray),
        m_nOuts3dEncoder * batchSize, m_decoder_shape_3d.data(),
        m_decoder_shape_3d.size());
    const float* tensor_values = tensor_data.GetTensorData<float>();

    const char* input_names[]  = {m_decoder_input_name_3d.c_str()};
    const char* output_names[] = {m_decoder_output_name_3d.c_str()};

#if 1
    if constexpr (std::is_same_v<T, double>) {
        // output to the buffer array so we're not deleting it
        m_input_shape_3d[0]      = batchSize;

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, m_doubleToFloatBuffer_3d.data(),
            m_total3DPts * batchSize, m_input_shape_3d.data(),
            m_input_shape_3d.size());

        m_3d_decoder->Run(Ort::RunOptions{nullptr}, input_names, &tensor_data,
                          1, output_names, &output_tensor, 1);

        // now the data has been written to output tensor so we don't have to
        // worry about *more copies*, we just convert back to doubles
        std::transform(
            m_doubleToFloatBuffer_3d.data(),
            m_doubleToFloatBuffer_3d.data() + (m_total3DPts * batchSize),
            outputArray, [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        m_input_shape_3d[0]      = batchSize;

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, outputArray, m_total3DPts * batchSize,
            m_input_shape_3d.data(), m_input_shape_3d.size());

        m_3d_decoder->Run(Ort::RunOptions{nullptr}, input_names, &tensor_data,
                          1, output_names, &output_tensor, 1);

        // here, we have the data just within output_tensor, which mapped
        // directly to our output array, so we are done
    }

#else
    auto output = m_3d_decoder->Run(Ort::RunOptions{nullptr}, input_names,
                                    &tensor_data, 1, output_names, 1);

    const float* output_data = output[0].GetTensorData<float>();

    if constexpr (std::is_same_v<T, double>) {
        std::transform(output_data, output_data + (m_total3DPts * batchSize),
                       outputArray,
                       [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        std::memcpy(outputArray, output_data,
                    m_total3DPts * sizeof(float) * batchSize);
    }
#endif
    return m_nOuts3dEncoder * sizeof(float) * batchSize;
#else
    std::memcpy(outputArray, compressedBuffer,
                m_total3DPts * sizeof(T) * batchSize);
    return m_total3DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t ONNXCompression::do_2d_compression(const T* originalMatrix,
                                          unsigned char* outputArray,
                                          size_t batchSize) {
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ONNX Compression only accepts doubles or floats as inputs!");

    if (!m_2d_encoder) {
        throw std::runtime_error(
            "ERROR: The 2D encoder is not initialized! Make sure set_models() "
            "is called.");
    }

    m_input_shape_2d[0] = batchSize;

    if (m_doubleToFloatBuffer_2d.size() < m_total2DPts * batchSize) {
        m_doubleToFloatBuffer_2d.resize(m_total2DPts * batchSize);
    }

    // tensor data can now come from this function
    Ort::Value tensor_data = createOnnxTensorFromData(
        originalMatrix, m_total2DPts, batchSize, m_doubleToFloatBuffer_2d,
        m_memory_info, m_input_shape_2d);

    // const float* tensor_values = tensor_data.GetTensorData<float>();

    const char* input_names[]  = {m_input_name_2d.c_str()};
    const char* output_names[] = {m_output_name_2d.c_str()};

    auto output = m_2d_encoder->Run(Ort::RunOptions{nullptr}, input_names,
                                    &tensor_data, 1, output_names, 1);
    const float* output_data = output[0].GetTensorData<float>();

    std::memcpy(outputArray, output_data,
                m_nOuts2dEncoder * sizeof(float) * batchSize);
    return m_nOuts2dEncoder * sizeof(float) * batchSize;
#else
    std::memcpy(outputArray, originalMatrix,
                m_total2DPts * sizeof(T) * batchSize);
    return m_total2DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t ONNXCompression::do_2d_decompression(
    const unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ONNX Compression only accepts doubles or floats as inputs!");

#ifdef DENDRO_ENABLE_ML_LIBRARIES
    if (!m_2d_decoder) {
        throw std::runtime_error(
            "ERROR: The 2D decoder is not initialized! Make sure set_models() "
            "is called.");
    }

    const float* floatInputArray =
        reinterpret_cast<const float*>(compressedBuffer);

    m_decoder_shape_2d[0] = batchSize;

    if (m_doubleToFloatBuffer_2d.size() < m_total2DPts * batchSize) {
        m_doubleToFloatBuffer_2d.resize(m_total2DPts * batchSize);
    }

    Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
        m_memory_info, const_cast<float*>(floatInputArray),
        m_nOuts2dEncoder * batchSize, m_decoder_shape_2d.data(),
        m_decoder_shape_2d.size());
    const float* tensor_values = tensor_data.GetTensorData<float>();

    const char* input_names[]  = {m_decoder_input_name_2d.c_str()};
    const char* output_names[] = {m_decoder_output_name_2d.c_str()};

#if 1
    if constexpr (std::is_same_v<T, double>) {
        // output to the buffer array so we're not deleting it
        m_input_shape_2d[0]      = batchSize;

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, m_doubleToFloatBuffer_2d.data(),
            m_total2DPts * batchSize, m_input_shape_2d.data(),
            m_input_shape_2d.size());

        m_2d_decoder->Run(Ort::RunOptions{nullptr}, input_names, &tensor_data,
                          1, output_names, &output_tensor, 1);

        // now the data has been written to output tensor so we don't have to
        // worry about *more copies*, we just convert back to doubles
        std::transform(
            m_doubleToFloatBuffer_2d.data(),
            m_doubleToFloatBuffer_2d.data() + (m_total2DPts * batchSize),
            outputArray, [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        m_input_shape_2d[0]      = batchSize;

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, outputArray, m_total2DPts * batchSize,
            m_input_shape_2d.data(), m_input_shape_2d.size());

        m_2d_decoder->Run(Ort::RunOptions{nullptr}, input_names, &tensor_data,
                          1, output_names, &output_tensor, 1);

        // here, we have the data just within output_tensor, which mapped
        // directly to our output array, so we are done
    }

#else
    auto output = m_2d_decoder->Run(Ort::RunOptions{nullptr}, input_names,
                                    &tensor_data, 1, output_names, 1);

    const float* output_data = output[0].GetTensorData<float>();

    // convert to doubles
    if constexpr (std::is_same_v<T, double>) {
        std::transform(output_data, output_data + (m_total2DPts * batchSize),
                       outputArray,
                       [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        std::memcpy(outputArray, output_data,
                    m_total2DPts * sizeof(float) * batchSize);
    }
#endif
    return m_nOuts2dEncoder * sizeof(float) * batchSize;
#else
    std::memcpy(outputArray, compressedBuffer,
                m_total2DPts * sizeof(T) * batchSize);
    return m_total2DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t ONNXCompression::do_1d_compression(const T* originalMatrix,
                                          unsigned char* outputArray,
                                          size_t batchSize) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ONNX Compression only accepts doubles or floats as inputs!");

#ifdef DENDRO_ENABLE_ML_LIBRARIES
    if (!m_1d_encoder) {
        throw std::runtime_error(
            "ERROR: The 1D encoder is not initialized! Make sure set_models() "
            "is called.");
    }

    m_input_shape_1d[0] = batchSize;

    if (m_doubleToFloatBuffer_1d.size() < m_total1DPts * batchSize) {
        m_doubleToFloatBuffer_1d.resize(m_total1DPts * batchSize);
    }

    // convert to tensor data
    Ort::Value tensor_data = createOnnxTensorFromData(
        originalMatrix, m_total1DPts, batchSize, m_doubleToFloatBuffer_1d,
        m_memory_info, m_input_shape_1d);

    // const float* tensor_values = tensor_data.GetTensorData<float>();

    const char* input_names[]  = {m_input_name_1d.c_str()};
    const char* output_names[] = {m_output_name_1d.c_str()};

    auto output = m_1d_encoder->Run(Ort::RunOptions{nullptr}, input_names,
                                    &tensor_data, 1, output_names, 1);
    const float* output_data = output[0].GetTensorData<float>();

    std::memcpy(outputArray, output_data,
                m_nOuts1dEncoder * sizeof(float) * batchSize);
    return m_nOuts1dEncoder * sizeof(float) * batchSize;
#else
    std::memcpy(outputArray, originalMatrix,
                m_total1DPts * sizeof(T) * batchSize);
    return m_total1DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t ONNXCompression::do_1d_decompression(
    const unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ONNX Compression only accepts doubles or floats as inputs!");

#ifdef DENDRO_ENABLE_ML_LIBRARIES
    if (!m_1d_decoder) {
        throw std::runtime_error(
            "ERROR: The 1D decoder is not initialized! Make sure set_models() "
            "is called.");
    }

    const float* floatInputArray =
        reinterpret_cast<const float*>(compressedBuffer);

    m_decoder_shape_1d[0] = batchSize;

    if (m_doubleToFloatBuffer_1d.size() < m_total1DPts * batchSize) {
        m_doubleToFloatBuffer_1d.resize(m_total1DPts * batchSize);
    }

    Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
        m_memory_info, const_cast<float*>(floatInputArray),
        m_nOuts1dEncoder * batchSize, m_decoder_shape_1d.data(),
        m_decoder_shape_1d.size());
    const float* tensor_values = tensor_data.GetTensorData<float>();

    const char* input_names[]  = {m_decoder_input_name_1d.c_str()};
    const char* output_names[] = {m_decoder_output_name_1d.c_str()};

#if 1
    if constexpr (std::is_same_v<T, double>) {
        // output to the buffer array so we're not deleting it
        m_input_shape_1d[0]      = batchSize;

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, m_doubleToFloatBuffer_1d.data(),
            m_total1DPts * batchSize, m_input_shape_1d.data(),
            m_input_shape_1d.size());

        m_1d_decoder->Run(Ort::RunOptions{nullptr}, input_names, &tensor_data,
                          1, output_names, &output_tensor, 1);

        // now the data has been written to output tensor so we don't have to
        // worry about *more copies*, we just convert back to doubles
        std::transform(
            m_doubleToFloatBuffer_1d.data(),
            m_doubleToFloatBuffer_1d.data() + (m_total1DPts * batchSize),
            outputArray, [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        m_input_shape_1d[0]      = batchSize;

        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
            m_memory_info, outputArray, m_total1DPts * batchSize,
            m_input_shape_1d.data(), m_input_shape_1d.size());

        m_1d_decoder->Run(Ort::RunOptions{nullptr}, input_names, &tensor_data,
                          1, output_names, &output_tensor, 1);

        // here, we have the data just within output_tensor, which mapped
        // directly to our output array, so we are done
    }

#else
    auto output = m_1d_decoder->Run(Ort::RunOptions{nullptr}, input_names,
                                    &tensor_data, 1, output_names, 1);

    const float* output_data = output[0].GetTensorData<float>();

    // convert to doubles
    if constexpr (std::is_same_v<T, double>) {
        std::transform(output_data, output_data + (m_total1DPts * batchSize),
                       outputArray,
                       [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        std::memcpy(outputArray, output_data,
                    m_total1DPts * sizeof(float) * batchSize);
    }
#endif
    return m_nOuts1dEncoder * sizeof(float) * batchSize;
#else
    std::memcpy(outputArray, compressedBuffer,
                m_total1DPts * sizeof(T) * batchSize);
    return m_total1DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t ONNXCompression::do_0d_compression(const T* originalMatrix,
                                          unsigned char* outputArray,
                                          size_t batchSize) {
    // TODO: 0D compression beyond just copying!
    std::memcpy(outputArray, originalMatrix,
                m_total0DPts * sizeof(T) * batchSize);
    return m_total0DPts * sizeof(T) * batchSize;
}

template <typename T>
size_t ONNXCompression::do_0d_decompression(
    const unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
    std::memcpy(outputArray, compressedBuffer,
                m_total0DPts * sizeof(T) * batchSize);
    return m_total0DPts * sizeof(T) * batchSize;
}

TorchScriptCompression mlcomp(6);

template <typename T>
size_t TorchScriptCompression::do_3d_compression(T* originalMatrix,
                                                 unsigned char* outputArray,
                                                 size_t batchSize) {
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    // start by creating the vector size
    torch::Tensor input_data = convertDataToModelType(
        originalMatrix, m_total3DPts, batchSize, "float");

    input_data =
        reshapeTensor3DBlock(input_data, m_numVars, m_pointsPerDim, batchSize);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_data);

    torch::Tensor output;
    try {
        output = m_3d_encoder.forward(inputs).toTensor();
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass on 3d encoder: " << e.what()
                  << std::endl;
        exit(-1);
    }

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }

    auto size = output.numel();

    if (size != m_nOuts3dEncoder * batchSize) {
        std::cerr << "ERROR: Mismatch on 3d AutoEncoder output sizes: expected "
                  << m_nOuts3dEncoder << ", got " << size << std::endl;
        exit(-1);
    }

    float* floatOutputArray = reinterpret_cast<float*>(outputArray);
    std::memcpy(floatOutputArray, output.data_ptr<float>(),
                size * sizeof(float));

    return sizeof(float) * m_nOuts3dEncoder * batchSize;
#else
    std::memcpy(outputArray, originalMatrix,
                m_total3DPts * sizeof(T) * batchSize);
    return m_total3DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t TorchScriptCompression::do_3d_decompression(
    unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    // simple reinterpret cast
    float* floatInputArray   = reinterpret_cast<float*>(compressedBuffer);

    torch::Tensor input_data = convertDataToModelType(
        floatInputArray, m_nOuts3dEncoder, batchSize, "float");

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_data);

    torch::Tensor output;
    try {
        output = m_3d_decoder.forward(inputs).toTensor();
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass in 3d decompression: "
                  << e.what() << std::endl;
        exit(-1);
    }

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }

    auto size = output.numel();

    if (size != m_total3DPts * batchSize) {
        std::cerr << "ERROR: Mismatch on 3D Decoder output sizes: expected "
                  << m_total3DPts << ", got " << size << std::endl;
        exit(-1);
    }

    if constexpr (std::is_same_v<T, double>) {
        // then do a transform to our double outputs
        std::transform(output.data_ptr<float>(),
                       output.data_ptr<float>() + m_total3DPts * batchSize,
                       outputArray,
                       [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        // or just copy if the output array expects floats
        std::memcpy(outputArray, output.data_ptr<float>(),
                    m_total3DPts * sizeof(float) * batchSize);
    }

    return sizeof(float) * m_nOuts3dEncoder * batchSize;
#else
    std::memcpy(outputArray, compressedBuffer,
                m_total3DPts * sizeof(T) * batchSize);
    return m_total3DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t TorchScriptCompression::do_2d_compression(T* originalMatrix,
                                                 unsigned char* outputArray,
                                                 size_t batchSize) {
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    // start by creating the vector size
    torch::Tensor input_data = convertDataToModelType(
        originalMatrix, m_total2DPts, batchSize, "float");

    input_data =
        reshapeTensor2DBlock(input_data, m_numVars, m_pointsPerDim, batchSize);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_data);

    torch::Tensor output;
    try {
        output = m_2d_encoder.forward(inputs).toTensor();
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass on 2d encoder: " << e.what()
                  << std::endl;
        exit(-1);
    }

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }

    auto size = output.numel();

    if (size != m_nOuts2dEncoder * batchSize) {
        std::cerr << "ERROR: Mismatch on 2d AutoEncoder output sizes: expected "
                  << m_nOuts2dEncoder << ", got " << size << std::endl;
        exit(-1);
    }

    float* floatOutputArray = reinterpret_cast<float*>(outputArray);
    std::memcpy(floatOutputArray, output.data_ptr<float>(),
                size * sizeof(float));

    return sizeof(float) * m_nOuts2dEncoder * batchSize;
#else
    std::memcpy(outputArray, originalMatrix,
                m_total2DPts * sizeof(T) * batchSize);
    return m_total2DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t TorchScriptCompression::do_2d_decompression(
    unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    // simple reinterpret cast
    float* floatInputArray   = reinterpret_cast<float*>(compressedBuffer);

    torch::Tensor input_data = convertDataToModelType(
        floatInputArray, m_nOuts2dEncoder, batchSize, "float");

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_data);

    torch::Tensor output;
    try {
        output = m_2d_decoder.forward(inputs).toTensor();
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass in 2d decompression: "
                  << e.what() << std::endl;
        exit(-1);
    }

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }

    auto size = output.numel();

    if (size != m_total2DPts * batchSize) {
        std::cerr << "ERROR: Mismatch on 2D Decoder output sizes: expected "
                  << m_total2DPts << ", got " << size << std::endl;
        exit(-1);
    }

    if constexpr (std::is_same_v<T, double>) {
        // then do a transform to our double outputs
        std::transform(output.data_ptr<float>(),
                       output.data_ptr<float>() + m_total2DPts * batchSize,
                       outputArray,
                       [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        // or just copy if the output array expects floats
        std::memcpy(outputArray, output.data_ptr<float>(),
                    m_total2DPts * sizeof(float) * batchSize);
    }

    return sizeof(float) * m_nOuts2dEncoder * batchSize;
#else
    std::memcpy(outputArray, compressedBuffer,
                m_total2DPts * sizeof(T) * batchSize);
    return m_total2DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t TorchScriptCompression::do_1d_compression(T* originalMatrix,
                                                 unsigned char* outputArray,
                                                 size_t batchSize) {
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    // start by creating the vector size
    torch::Tensor input_data = convertDataToModelType(
        originalMatrix, m_total1DPts, batchSize, "float");

    input_data =
        reshapeTensor1DBlock(input_data, m_numVars, m_pointsPerDim, batchSize);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_data);

    torch::Tensor output;
    try {
        output = m_1d_encoder.forward(inputs).toTensor();
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass on 1d encoder: " << e.what()
                  << std::endl;
        exit(-1);
    }

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }

    auto size = output.numel();

    if (size != m_nOuts1dEncoder * batchSize) {
        std::cerr << "ERROR: Mismatch on 1d encoder output sizes: expected "
                  << m_nOuts1dEncoder << ", got " << size << std::endl;
        exit(-1);
    }

    float* floatOutputArray = reinterpret_cast<float*>(outputArray);
    std::memcpy(floatOutputArray, output.data_ptr<float>(),
                size * sizeof(float));

    return sizeof(float) * m_nOuts1dEncoder * batchSize;
#else
    std::memcpy(outputArray, originalMatrix,
                m_total1DPts * sizeof(T) * batchSize);
    return m_total1DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t TorchScriptCompression::do_1d_decompression(
    unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    // simple reinterpret cast
    float* floatInputArray   = reinterpret_cast<float*>(compressedBuffer);

    torch::Tensor input_data = convertDataToModelType(
        floatInputArray, m_nOuts1dEncoder, batchSize, "float");

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_data);

    torch::Tensor output;
    try {
        output = m_1d_decoder.forward(inputs).toTensor();
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass in 1d decompression: "
                  << e.what() << std::endl;
        exit(-1);
    }

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }

    auto size = output.numel();

    if (size != m_total1DPts * batchSize) {
        std::cerr << "ERROR: Mismatch on 1D Decoder output sizes: expected "
                  << m_total1DPts << ", got " << size << std::endl;
        exit(-1);
    }

    if constexpr (std::is_same_v<T, double>) {
        // then do a transform to our double outputs
        std::transform(output.data_ptr<float>(),
                       output.data_ptr<float>() + m_total1DPts * batchSize,
                       outputArray,
                       [](float d) { return static_cast<double>(d); });
    } else if constexpr (std::is_same_v<T, float>) {
        // or just copy if the output array expects floats
        std::memcpy(outputArray, output.data_ptr<float>(),
                    m_total1DPts * sizeof(float) * batchSize);
    }

    return sizeof(float) * m_nOuts1dEncoder * batchSize;
#else
    std::memcpy(outputArray, compressedBuffer,
                m_total1DPts * sizeof(T) * batchSize);
    return m_total1DPts * sizeof(T) * batchSize;
#endif
}

template <typename T>
size_t TorchScriptCompression::do_0d_compression(T* originalMatrix,
                                                 unsigned char* outputArray,
                                                 size_t batchSize) {
    // TODO: 0D compression beyond just copying!
    std::memcpy(outputArray, originalMatrix,
                m_total0DPts * sizeof(T) * batchSize);
    return m_total0DPts * sizeof(T) * batchSize;
}

template <typename T>
size_t TorchScriptCompression::do_0d_decompression(
    unsigned char* compressedBuffer, T* outputArray, size_t batchSize) {
    std::memcpy(outputArray, compressedBuffer,
                m_total0DPts * sizeof(T) * batchSize);
    return m_total0DPts * sizeof(T) * batchSize;
}

}  // namespace MachineLearningAlgorithms

namespace ChebyshevAlgorithms {

// a "global" object that we can call from pretty much anywhere
ChebyshevCompression cheby{6, 3};

void ChebyshevCompression::set_compression_type(const size_t& eleOrder,
                                                const size_t& nReduced) {
    // TODO: autogenerate this
    if (eleOrder == 6) {
        if (nReduced == 1) {
            set_chebyshev_mat_ele6_out1_dim1();
            set_chebyshev_mat_ele6_out1_dim2();
            set_chebyshev_mat_ele6_out1_dim3();
        } else if (nReduced == 2) {
            set_chebyshev_mat_ele6_out2_dim1();
            set_chebyshev_mat_ele6_out2_dim2();
            set_chebyshev_mat_ele6_out2_dim3();
        } else if (nReduced == 3) {
            set_chebyshev_mat_ele6_out3_dim1();
            set_chebyshev_mat_ele6_out3_dim2();
            set_chebyshev_mat_ele6_out3_dim3();
        } else if (nReduced == 4) {
            set_chebyshev_mat_ele6_out4_dim1();
            set_chebyshev_mat_ele6_out4_dim2();
            set_chebyshev_mat_ele6_out4_dim3();
        } else if (eleOrder == 8) {
            if (nReduced == 1) {
                set_chebyshev_mat_ele8_out1_dim1();
                set_chebyshev_mat_ele8_out1_dim2();
                set_chebyshev_mat_ele8_out1_dim3();
            } else if (nReduced == 2) {
                set_chebyshev_mat_ele8_out2_dim1();
                set_chebyshev_mat_ele8_out2_dim2();
                set_chebyshev_mat_ele8_out2_dim3();
            } else if (nReduced == 3) {
                set_chebyshev_mat_ele8_out3_dim1();
                set_chebyshev_mat_ele8_out3_dim2();
                set_chebyshev_mat_ele8_out3_dim3();
            } else if (nReduced == 4) {
                set_chebyshev_mat_ele8_out4_dim1();
                set_chebyshev_mat_ele8_out4_dim2();
                set_chebyshev_mat_ele8_out4_dim3();
            }
        }
    }

    recalculate_byte_sizes();
}

void ChebyshevCompression::do_array_norm(double* array, const size_t count,
                                         double& minVal, double& maxVal) {
    minVal       = *std::min_element(array, array + count);
    maxVal       = *std::max_element(array, array + count);
    double range = maxVal - minVal;

    if (minVal < -1.0 || maxVal > 1.0) {
        // if we're outside -1 and or 1, then we need to do full normalization
        for (size_t i = 0; i < count; i++) {
            array[i] = 2.0 * ((array[i] - minVal) / range) - 1.0;
        }
    } else if (range > 1e-8) {
        // apply a shift if we're within the vals
        double shift = -(minVal + maxVal) / 2.0;
        for (size_t i = 0; i < count; i++) {
            array[i] += shift;
        }
    } else {
        // otherwise do nothing, we're close to zero or something
    }
}

void ChebyshevCompression::undo_array_norm(double* array, const size_t count,
                                           const double minVal,
                                           const double maxVal) {
    double range = maxVal - minVal;

    if (minVal < -1.0 || maxVal > 1.0) {
        // if we're outside -1 and or 1, then we need to do full
        // denormalization
        for (size_t i = 0; i < count; i++) {
            array[i] = ((array[i] + 1.0) / 2.0) * range + minVal;
        }
    } else if (range > 1e-8) {
        // apply a shift if we're within the vals
        double shift = -(minVal + maxVal) / 2.0;
        for (size_t i = 0; i < count; i++) {
            array[i] -= shift;
        }
    } else {
        // otherwise do nothing, we didn't do anything above
    }
}

size_t ChebyshevCompression::do_3d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    // the number will always be (eleorder - 1) ^3 because it's a 3d block,
    // turns out this number coencides with cheb_dim3_decomp

    double maxVal, minVal;
    do_array_norm(originalMatrix, cheb_dim3_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    // that *I* control
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    char TRANS = 'T';

    dgemv_(&TRANS, &cheb_dim3_decomp, &cheb_dim3_comp, &alpha, A_cheb_dim3,
           &cheb_dim3_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    // now we're "compressed", and we return the total number of bytes we
    // wrote:
    return bytes_3d;
}

size_t ChebyshevCompression::do_3d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim3_decomp, &cheb_dim3_comp, &alpha, A_cheb_dim3,
           &cheb_dim3_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim3_decomp, minVal, maxVal);

    // total number of doubles coming out
    return bytes_3d;
}

size_t ChebyshevCompression::do_2d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    double maxVal, minVal;
    do_array_norm(originalMatrix, cheb_dim2_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    char TRANS = 'T';
    // now do the matrix-vector multiplication
    dgemv_(&TRANS, &cheb_dim2_decomp, &cheb_dim2_comp, &alpha, A_cheb_dim2,
           &cheb_dim2_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    // total number of bytes that we wrote
    return bytes_2d;
}

size_t ChebyshevCompression::do_2d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim2_decomp, &cheb_dim2_comp, &alpha, A_cheb_dim2,
           &cheb_dim2_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim2_decomp, minVal, maxVal);

    // total number of compress values coming out, to advance. The doubles we
    // know are easy
    return bytes_2d;
}

size_t ChebyshevCompression::do_1d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    double maxVal, minVal;

    do_array_norm(originalMatrix, cheb_dim1_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    // do the matrix math
    char TRANSA         = 'T';
    char TRANSB         = 'N';

    double* temp_output = new double[3];

    // matrix multiplication because it works
    dgemv_(&TRANSA, &cheb_dim1_decomp, &cheb_dim1_comp, &alpha, A_cheb_dim1,
           &cheb_dim1_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    return bytes_1d;
}

size_t ChebyshevCompression::do_1d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    unsigned char* ptr = compressedBuffer;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    // if (minVal > maxVal) {
    //     std::cout << "Detected minval larger than maxval in 1d decompression!
    //     "
    //               << maxVal << " " << minVal << std::endl;
    // }

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim1_decomp, &cheb_dim1_comp, &alpha, A_cheb_dim1,
           &cheb_dim1_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim1_decomp, minVal, maxVal);

    // total number of doubles coming out
    return bytes_1d;
}

}  // namespace ChebyshevAlgorithms

namespace ZFPAlgorithms {

// "global" object for ZFP algorithm that can be called by dendro
ZFPCompression zfpblockwise(6, 5.0);

template <typename T>
size_t ZFPCompression::do_4d_compression(T* originalMatrix,
                                         unsigned char* outputArray,
                                         size_t batchSize) {
    // std::memcpy(outputArray, originalMatrix,
    //             batchSize * zfp_dim4_decomp * sizeof(T));
    // return batchSize * zfp_dim4_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim4_decomp * sizeof(T) * batchSize;

#ifdef __DENDRO_ZFP_USE_TRUE_4D__
    // combine based on batch size itself
    if (field_4d != nullptr) {
        zfp_field_free(field_4d);
        field_4d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_4d =
            zfp_field_4d(NULL, zfp_type_double, numVars * batchSize,
                         zfp_num_per_dim, zfp_num_per_dim, zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_4d =
            zfp_field_4d(NULL, zfp_type_float, numVars * batchSize,
                         zfp_num_per_dim, zfp_num_per_dim, zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_4d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 4D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }
#else
    // combine based on batch size itself
    if (field_3d != nullptr) {
        zfp_field_free(field_3d);
        field_3d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_3d = zfp_field_3d(NULL, zfp_type_double,
                                numVars * batchSize * zfp_num_per_dim,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_3d = zfp_field_3d(NULL, zfp_type_float,
                                numVars * batchSize * zfp_num_per_dim,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_3d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 4D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }
#endif

// create a field
#ifdef __DENDRO_ZFP_USE_TRUE_4D__
    zfp_field_set_pointer(field_4d, originalMatrix);
#else
    zfp_field_set_pointer(field_3d, originalMatrix);
#endif

// need to calculate the maximum size
#ifdef __DENDRO_ZFP_USE_TRUE_4D__
    size_t bufsize = zfp_stream_maximum_size(zfp4d, field_4d);
#else
    size_t bufsize = zfp_stream_maximum_size(zfp3d, field_3d);
#endif
    if (bufsize == 0) {
        std::cerr << "CRITICAL ERROR CALCULATING MAXIMUM SIZE!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

// associate the bitstream with ZFP stream
#ifdef __DENDRO_ZFP_USE_TRUE_4D__
    zfp_stream_set_bit_stream(zfp4d, stream);
#else
    zfp_stream_set_bit_stream(zfp3d, stream);
#endif

#ifdef __DENDRO_ZFP_USE_TRUE_4D__
    size_t outsize = zfp_compress(zfp4d, field_4d);
#else
    size_t outsize = zfp_compress(zfp3d, field_3d);
#endif
    if (outsize == 0) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 4D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 4D ZFP STREAM! The "
                     "compressed buffer is larger than the original!"
                  << std::endl;
        std::cerr << "Number of points to compress: "
                  << zfp_dim4_decomp * batchSize << " ("
                  << zfp_dim4_decomp * sizeof(T) * batchSize
                  << " bytes), number of bytes in compressed stream: "
                  << outsize << std::endl;
#endif

        // just copy the raw data
        std::memcpy(outputArray + sizeof(size_t), originalMatrix,
                    uncompressed_size);
        outsize = uncompressed_size;

        // exit(EXIT_FAILURE);
    }

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_4d_decompression(unsigned char* compressedBuffer,
                                           T* outputArray, size_t batchSize) {
    // std::memcpy(outputArray, compressedBuffer,
    //             batchSize * zfp_dim4_decomp * sizeof(T));
    // return batchSize * zfp_dim4_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim4_decomp * sizeof(T) * batchSize;

    // first extract out the buffer size
    size_t bufsize;
    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    if (bufsize == uncompressed_size) {
        std::memcpy(outputArray, compressedBuffer + sizeof(size_t),
                    uncompressed_size);
        return bufsize + sizeof(size_t);
        // exit(EXIT_FAILURE);
    }

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

#ifdef __DENDRO_ZFP_USE_TRUE_4D__
    // combine based on batch size itself
    if (field_4d != nullptr) {
        zfp_field_free(field_4d);
        field_4d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_4d =
            zfp_field_4d(NULL, zfp_type_double, numVars * batchSize,
                         zfp_num_per_dim, zfp_num_per_dim, zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_4d =
            zfp_field_4d(NULL, zfp_type_float, numVars * batchSize,
                         zfp_num_per_dim, zfp_num_per_dim, zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_4d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 4D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    zfp_stream_set_bit_stream(zfp4d, stream);
    zfp_field_set_pointer(field_4d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp4d, field_4d);
#else
    // combine based on batch size itself
    if (field_3d != nullptr) {
        zfp_field_free(field_3d);
        field_3d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_3d = zfp_field_3d(NULL, zfp_type_double,
                                numVars * batchSize * zfp_num_per_dim,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_3d = zfp_field_3d(NULL, zfp_type_float,
                                numVars * batchSize * zfp_num_per_dim,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_3d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 4D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    zfp_stream_set_bit_stream(zfp3d, stream);
    zfp_field_set_pointer(field_3d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp3d, field_3d);
#endif

    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 4D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_3d_compression(T* originalMatrix,
                                         unsigned char* outputArray,
                                         size_t batchSize) {
    // std::memcpy(outputArray, originalMatrix,
    //             batchSize * zfp_dim3_decomp * sizeof(T));
    // return batchSize * zfp_dim3_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim3_decomp * sizeof(T) * batchSize;

    // combine based on batch size itself
    if (field_3d != nullptr) {
        zfp_field_free(field_3d);
        field_3d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_3d = zfp_field_3d(NULL, zfp_type_double, numVars * batchSize,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_3d = zfp_field_3d(NULL, zfp_type_float, numVars * batchSize,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_3d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 3D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // create a field
    zfp_field_set_pointer(field_3d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize = zfp_stream_maximum_size(zfp3d, field_3d);
    if (bufsize == 0) {
        std::cerr << "CRITICAL ERROR CALCULATING MAXIMUM SIZE!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp3d, stream);

    size_t outsize = zfp_compress(zfp3d, field_3d);
    if (outsize == 0) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 3D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 3D ZFP STREAM! The "
                     "compressed buffer is larger than the original!"
                  << std::endl;
        std::cerr << "Number of points to compress: "
                  << zfp_dim3_decomp * batchSize << " ("
                  << zfp_dim3_decomp * sizeof(T) * batchSize
                  << " bytes), number of bytes in compressed stream: "
                  << outsize << std::endl;
#endif

        // just copy the raw data
        std::memcpy(outputArray + sizeof(size_t), originalMatrix,
                    uncompressed_size);
        outsize = uncompressed_size;

        // exit(EXIT_FAILURE);
    }

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_3d_decompression(unsigned char* compressedBuffer,
                                           T* outputArray, size_t batchSize) {
    // std::memcpy(outputArray, compressedBuffer,
    //             batchSize * zfp_dim3_decomp * sizeof(T));
    // return batchSize * zfp_dim3_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim3_decomp * sizeof(T) * batchSize;

    // first extract out the buffer size
    size_t bufsize;
    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    if (bufsize == uncompressed_size) {
        std::memcpy(outputArray, compressedBuffer + sizeof(size_t),
                    uncompressed_size);
        return bufsize + sizeof(size_t);
        // exit(EXIT_FAILURE);
    }

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // combine based on batch size itself
    if (field_3d != nullptr) {
        zfp_field_free(field_3d);
        field_3d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_3d = zfp_field_3d(NULL, zfp_type_double, numVars * batchSize,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_3d = zfp_field_3d(NULL, zfp_type_float, numVars * batchSize,
                                zfp_num_per_dim, zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_3d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 3D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    zfp_stream_set_bit_stream(zfp3d, stream);
    zfp_field_set_pointer(field_3d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp3d, field_3d);
    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 3D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_2d_compression(T* originalMatrix,
                                         unsigned char* outputArray,
                                         size_t batchSize) {
    // std::memcpy(outputArray, originalMatrix,
    //             batchSize * zfp_dim2_decomp * sizeof(T));
    // return batchSize * zfp_dim2_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim2_decomp * sizeof(T) * batchSize;

    // combine based on batch size itself
    if (field_2d != nullptr) {
        zfp_field_free(field_2d);
        field_2d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_2d = zfp_field_2d(NULL, zfp_type_double, numVars * batchSize,
                                zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_2d = zfp_field_2d(NULL, zfp_type_float, numVars * batchSize,
                                zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_2d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 2D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // create a field
    zfp_field_set_pointer(field_2d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize = zfp_stream_maximum_size(zfp2d, field_2d);
    if (bufsize == 0) {
        std::cerr << "CRITICAL ERROR CALCULATING MAXIMUM SIZE!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp2d, stream);

    size_t outsize = zfp_compress(zfp2d, field_2d);
    if (outsize == 0) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 2D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 2D ZFP STREAM! The "
                     "compressed buffer is larger than the original!"
                  << std::endl;
        std::cerr << "Number of points to compress: "
                  << zfp_dim2_decomp * batchSize << " ("
                  << zfp_dim2_decomp * sizeof(T) * batchSize
                  << " bytes), number of bytes in compressed stream: "
                  << outsize << std::endl;
#endif

        // just copy the raw data
        std::memcpy(outputArray + sizeof(size_t), originalMatrix,
                    uncompressed_size);
        outsize = uncompressed_size;

        // exit(EXIT_FAILURE);
    }

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_2d_decompression(unsigned char* compressedBuffer,
                                           T* outputArray, size_t batchSize) {
    // std::memcpy(outputArray, compressedBuffer,
    //             batchSize * zfp_dim2_decomp * sizeof(T));
    // return batchSize * zfp_dim2_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim2_decomp * sizeof(T) * batchSize;

    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    if (bufsize == uncompressed_size) {
        std::memcpy(outputArray, compressedBuffer + sizeof(size_t),
                    uncompressed_size);
        return bufsize + sizeof(size_t);
        // exit(EXIT_FAILURE);
    }

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // combine based on batch size itself
    if (field_2d != nullptr) {
        zfp_field_free(field_2d);
        field_2d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_2d = zfp_field_2d(NULL, zfp_type_double, numVars * batchSize,
                                zfp_num_per_dim);
    } else if constexpr (std::is_same_v<T, float>) {
        field_2d = zfp_field_2d(NULL, zfp_type_float, numVars * batchSize,
                                zfp_num_per_dim);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_2d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 2D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    zfp_stream_set_bit_stream(zfp2d, stream);
    zfp_field_set_pointer(field_2d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp2d, field_2d);
    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 2D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_1d_compression(T* originalMatrix,
                                         unsigned char* outputArray,
                                         size_t batchSize) {
    // std::memcpy(outputArray, originalMatrix,
    //             batchSize * zfp_dim1_decomp * sizeof(T));
    // return batchSize * zfp_dim1_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim1_decomp * sizeof(T) * batchSize;

    // combine based on batch size itself
    if (field_1d != nullptr) {
        zfp_field_free(field_1d);
        field_1d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_1d = zfp_field_1d(NULL, zfp_type_double, numVars * batchSize);
    } else if constexpr (std::is_same_v<T, float>) {
        field_1d = zfp_field_1d(NULL, zfp_type_float, numVars * batchSize);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_1d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 1D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // create a field
    zfp_field_set_pointer(field_1d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize = zfp_stream_maximum_size(zfp1d, field_1d);
    if (bufsize == 0) {
        std::cerr << "CRITICAL ERROR CALCULATING MAXIMUM SIZE!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp1d, stream);

    size_t outsize = zfp_compress(zfp1d, field_1d);
    if (outsize == 0) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 1D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 1D ZFP STREAM! The "
                     "compressed buffer is larger than the original!"
                  << std::endl;
        std::cerr << "Number of points to compress: "
                  << zfp_dim1_decomp * batchSize << " ("
                  << zfp_dim1_decomp * sizeof(T) * batchSize
                  << " bytes), number of bytes in compressed stream: "
                  << outsize << std::endl;
#endif

        // just copy the raw data
        std::memcpy(outputArray + sizeof(size_t), originalMatrix,
                    uncompressed_size);
        outsize = uncompressed_size;

        // exit(EXIT_FAILURE);
    }

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_1d_decompression(unsigned char* compressedBuffer,
                                           T* outputArray, size_t batchSize) {
    // std::memcpy(outputArray, compressedBuffer,
    //             batchSize * zfp_dim1_decomp * sizeof(T));
    return batchSize * zfp_dim1_decomp * sizeof(T);
    const size_t uncompressed_size = zfp_dim1_decomp * sizeof(T) * batchSize;

    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    if (bufsize == uncompressed_size) {
        std::memcpy(outputArray, compressedBuffer + sizeof(size_t),
                    uncompressed_size);
        return bufsize + sizeof(size_t);
        // exit(EXIT_FAILURE);
    }

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);
    if (stream == nullptr) {
        std::cerr << "CRITICAL ERROR OPENING BITSTREAM!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // combine based on batch size itself
    if (field_1d != nullptr) {
        zfp_field_free(field_1d);
        field_1d = nullptr;
    }
    if constexpr (std::is_same_v<T, double>) {
        field_1d = zfp_field_1d(NULL, zfp_type_double, numVars * batchSize);
    } else if constexpr (std::is_same_v<T, float>) {
        field_1d = zfp_field_1d(NULL, zfp_type_float, numVars * batchSize);
    } else {
        throw std::runtime_error(
            "ZFP should not be called with something other than double/float");
    }
    if (field_1d == nullptr) {
        std::cerr << "CRITICAL ERROR CREATING 1D FIELD!" << std::endl;
        exit(EXIT_FAILURE);
    }

    zfp_stream_set_bit_stream(zfp1d, stream);
    zfp_field_set_pointer(field_1d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp1d, field_1d);
    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 1D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

template <typename T>
size_t ZFPCompression::do_0d_compression(T* originalMatrix,
                                         unsigned char* outputArray,
                                         size_t batchSize) {
    // TODO: 0D compression beyond just copying!
    std::memcpy(outputArray, originalMatrix,
                batchSize * zfp_dim1_decomp * sizeof(T));
    return batchSize * zfp_dim1_decomp * sizeof(T);
}

template <typename T>
size_t ZFPCompression::do_0d_decompression(unsigned char* compressedBuffer,
                                           T* outputArray, size_t batchSize) {
    std::memcpy(outputArray, compressedBuffer,
                batchSize * zfp_dim1_decomp * sizeof(T));
    return batchSize * zfp_dim1_decomp * sizeof(T);
}

}  // namespace ZFPAlgorithms

namespace BLOSCAlgorithms {

BloscCompression bloscblockwise(6, "lz4", 4, 1);

template <typename T>
size_t BloscCompression::do_4d_compression(T* originalMatrix,
                                           unsigned char* outputArray,
                                           size_t batchSize) {
    size_t input_size = blosc_original_bytes_4d * batchSize;
    size_t output_size_buffer =
        blosc_original_bytes_4d * batchSize + BLOSC_MAX_OVERHEAD;

    // make sure the output array includes our header
    // std::cout << "attempting to compress " << blosc_original_bytes_3d
    //           << std::endl;
    int compressedSize =
        blosc_compress(clevel, doShuffle, sizeof(T), input_size, originalMatrix,
                       outputArray + sizeof(size_t), output_size_buffer);
    // TODO: original bytes overhead should be likely be adjusted since this
    // will *multiply*

    // TODO: if compressed size is 0, we have to disregard the buffer
    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 4d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_4d_decompression(unsigned char* compressedBuffer,
                                             T* outputArray, size_t batchSize) {
    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    size_t expected_out_size = blosc_original_bytes_4d * batchSize;

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData = blosc_decompress(compressedBuffer + sizeof(size_t),
                                            outputArray, expected_out_size);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 4d!" << std::endl;
        std::cout << "number of bytes expected: " << expected_out_size
                  << " - output decompressedData " << decompressedData
                  << " read compressed size: " << outSize
                  << " batch size is: " << batchSize << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_3d_compression(T* originalMatrix,
                                           unsigned char* outputArray,
                                           size_t batchSize) {
    size_t input_size = blosc_original_bytes_3d * batchSize;
    size_t output_size_buffer =
        blosc_original_bytes_3d * batchSize + BLOSC_MAX_OVERHEAD;

    // make sure the output array includes our header
    // std::cout << "attempting to compress " << blosc_original_bytes_3d
    //           << std::endl;
    int compressedSize = blosc_compress(
        clevel, doShuffle, sizeof(T), input_size, originalMatrix,
        outputArray + sizeof(size_t), output_size_buffer * batchSize);

    // TODO: if compressed size is 0, we have to disregard the buffer
    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 3d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_3d_decompression(unsigned char* compressedBuffer,
                                             T* outputArray, size_t batchSize) {
    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    size_t expected_out_size = blosc_original_bytes_3d * batchSize;

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData = blosc_decompress(compressedBuffer + sizeof(size_t),
                                            outputArray, expected_out_size);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 3d!" << std::endl;
        std::cout << "number of bytes expected: "
                  << blosc_original_bytes_3d * batchSize
                  << " - output decompressedData " << decompressedData
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_2d_compression(T* originalMatrix,
                                           unsigned char* outputArray,
                                           size_t batchSize) {
    size_t input_size = blosc_original_bytes_2d * batchSize;
    size_t output_size_buffer =
        blosc_original_bytes_2d * batchSize + BLOSC_MAX_OVERHEAD;

    // TODO: need some kind of better metric or way to know if we can compress
    // or not. Current idea is if it fails, we still do a copy. We attempt to
    // copy it back out into 32 bits and see if it's garbage? idk
    if (input_size < 36 * sizeof(T)) {
        std::memcpy(outputArray, originalMatrix, input_size);
        return input_size;
    }

    // make sure the output array includes our header
    int compressedSize =
        blosc_compress(clevel, doShuffle, sizeof(T), input_size, originalMatrix,
                       outputArray + sizeof(size_t), output_size_buffer);

    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 2d!" << std::endl;
        exit(EXIT_FAILURE);
    } else if (compressedSize == output_size_buffer) {
        // std::cerr << "ERROR: found a block that can't be compressed in 2D!"
        //           << std::endl;
        // exit(EXIT_FAILURE);
        // need some method of handling it a bit better
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_2d_decompression(unsigned char* compressedBuffer,
                                             T* outputArray, size_t batchSize) {
    size_t expected_out_size = blosc_original_bytes_2d * batchSize;
    // TODO: see 2d_compression above, this needs to be handled better
    if (expected_out_size < 36 * sizeof(T)) {
        std::memcpy(outputArray, compressedBuffer, expected_out_size);
        return expected_out_size;
    }
    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData = blosc_decompress(compressedBuffer + sizeof(size_t),
                                            outputArray, expected_out_size);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 2d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_1d_compression(T* originalMatrix,
                                           unsigned char* outputArray,
                                           size_t batchSize) {
    size_t input_size = blosc_original_bytes_1d * batchSize;
    size_t output_size_buffer =
        blosc_original_bytes_1d * batchSize + BLOSC_MAX_OVERHEAD;

    // TODO: see 1d_compression above, this needs to be handled better
    if (input_size < 36 * sizeof(T)) {
        std::memcpy(outputArray, originalMatrix, input_size);
        return input_size;
    }

    // make sure the output array includes our header
    int compressedSize =
        blosc_compress(clevel, doShuffle, sizeof(T), input_size, originalMatrix,
                       outputArray + sizeof(size_t), output_size_buffer);

    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 1d!" << std::endl;
        exit(EXIT_FAILURE);
    } else if (compressedSize == 0) {
        // it failed to compress if we're at 0, which means garbage, so we want
        // to copy in the data
        std::cout << "FAILED in 1d Case" << std::endl;
    } else if (compressedSize == output_size_buffer) {
        // we weren't able to get any compression!
    } else {
        // success
        // std::cout << "SUCCCESS! Got a compressed 1d! Hooray!" << std::endl;
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_1d_decompression(unsigned char* compressedBuffer,
                                             T* outputArray, size_t batchSize) {
    size_t expected_out_size = blosc_original_bytes_1d * batchSize;
    // TODO: see 1d_compression above, this needs to be handled better
    if (expected_out_size < 36 * sizeof(T)) {
        std::memcpy(outputArray, compressedBuffer, expected_out_size);
        return expected_out_size;
    }

    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData = blosc_decompress(compressedBuffer + sizeof(size_t),
                                            outputArray, expected_out_size);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 1d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

template <typename T>
size_t BloscCompression::do_0d_compression(T* originalMatrix,
                                           unsigned char* outputArray,
                                           size_t batchSize) {
    // TODO: 0D compression beyond just copying!
    std::memcpy(outputArray, originalMatrix,
                blosc_original_bytes_1d * batchSize);
    return blosc_original_bytes_1d * batchSize;
}

template <typename T>
size_t BloscCompression::do_0d_decompression(unsigned char* compressedBuffer,
                                             T* outputArray, size_t batchSize) {
    std::memcpy(outputArray, compressedBuffer,
                blosc_original_bytes_1d * batchSize);
    return blosc_original_bytes_1d * batchSize;
}

unsigned char* compressData(const char* blosc_compressor, int clevel, int n,
                            double* originalData, int& byteStreamSize) {
    blosc_set_compressor(blosc_compressor);
    int originalDataBytes         = n * sizeof(double);

    // Calculate the maximum possible size for the compressed data
    // This value is suggested to not be modified and does not affect the size
    // of the final compressed form
    int maxCompressedSize         = originalDataBytes + BLOSC_MAX_OVERHEAD;
    unsigned char* compressedData = new unsigned char[maxCompressedSize];

    // Blosc compression function: int blosc_ocmpress(int clevel, int shuffle,
    // size_t typesize, size_t nbytes, const void* src, void* dest, size_t
    // destsize); Parameters: clevel: Compression level (0-9, 0 being no
    // compression, 9 being maximum compression). shuffle: Bitshuffle filter for
    // data rearrangement.
    //          - Pass 0 for no shuffling, which can be used for data without a
    //          specific pattern or when compression speed is a priority.
    //          - Pass 1 for byte shuffle, which is effective for numerical data
    //          where each element is larger than a byte, as it aligns the least
    //          significant bits of the data types. Good for data with repeating
    //          patterns at the byte level.
    //          - Pass 2 for bit shuffle, which is more aggressive than byte
    //          shuffle and aligns the bits across data types. Useful for
    //          numerical data with repeating patterns at the bit level, often
    //          leading to better compression.
    // typesize: The size of the datatype in the array (in bytes).
    // nbytes: The number of bytes to compress from the source buffer.
    // src: Pointer to the data buffer to compress.
    // dest: Pointer to the buffer where the compressed data will be stored.
    // destsize: Maximum size of the destination buffer.
    int compressedSize =
        blosc_compress(clevel, 1, sizeof(double), originalDataBytes,
                       originalData, compressedData, maxCompressedSize);
    if (compressedSize < 0) {
        throw std::runtime_error("blosc could not compress data.");
    }

    // Allocate memory for the bytestream, including space for the size of the
    // original data
    byteStreamSize            = compressedSize + sizeof(originalDataBytes);
    unsigned char* bytestream = new unsigned char[byteStreamSize];
    // Copy compressed data to bytestream
    std::memcpy(bytestream, compressedData, compressedSize);
    // Pack originalDataBytes at the end of the bytestream
    std::memcpy(bytestream + compressedSize, &originalDataBytes,
                sizeof(originalDataBytes));
    delete[] compressedData;
    return bytestream;
}

double* decompressData(unsigned char* byteStream, int byteStreamSize) {
    // Check if byteStream is valid
    if (!byteStream || byteStreamSize <= 0) {
        return nullptr;
    }
    // Unpack originalDataBytes from the end of the byteStream
    int originalDataBytes;
    std::memcpy(&originalDataBytes,
                byteStream + (byteStreamSize - sizeof(originalDataBytes)),
                sizeof(originalDataBytes));
    double* decompressedData = new double[originalDataBytes / sizeof(double)];

    int decompressedSize =
        blosc_decompress(byteStream, decompressedData, originalDataBytes);

    // Check for decompression error
    if (decompressedSize < 0) {
        // Handle decompression error (e.g., return null or throw an exception)
        throw std::runtime_error("blosc could not decompress data.");
    }

    return decompressedData;
}

void decompressData(unsigned char* byteStream, int byteStreamSize,
                    double* outBuff) {
    // Check if byteStream is valid
    // if (!byteStream || byteStreamSize <= 0) {
    //     return nullptr;
    // }
    // Unpack originalDataBytes from the end of the byteStream

    int originalDataBytes;
    std::memcpy(&originalDataBytes,
                byteStream + (byteStreamSize - sizeof(originalDataBytes)),
                sizeof(originalDataBytes));
    // double* decompressedData = new double[originalDataBytes /
    // sizeof(double)];

    int decompressedSize =
        blosc_decompress(byteStream, outBuff, originalDataBytes);

    // Check for decompression error
    if (decompressedSize < 0) {
        // Handle decompression error (e.g., return null or throw an exception)
        throw std::runtime_error("blosc could not decompress data.");
    }
}
}  // namespace BLOSCAlgorithms

namespace dendro_compress {

GaussianFiltering gaussfilter;
ChebyshevFiltering chebyfilter;

CompressionType COMPRESSION_OPTION   = CompressionType::ZFP;
FilterType COMPRESSION_FILTER_OPTION = FilterType::F_NONE;

void set_compression_options(CompressionType compT,
                             const CompressionOptions& compOpt,
                             const ot::CTXSendType sendType,
                             const FilterType filterT) {
    dendro_compress::COMPRESSION_OPTION        = compT;
    dendro_compress::COMPRESSION_FILTER_OPTION = filterT;

    std::cout << "Set compression option to: "
              << dendro_compress::COMPRESSION_OPTION << std::endl;
    std::cout << "Set filtering option to: "
              << dendro_compress::COMPRESSION_FILTER_OPTION << std::endl;

    ZFPAlgorithms::zfpblockwise.setCtxSendType(sendType);

    // then set up the options for all types
    ZFPAlgorithms::zfpblockwise.setEleOrder(compOpt.eleOrder);

    // TEMP: this needs to be an option
    ZFPAlgorithms::zfpblockwise.setUpForMultiVariable(compOpt.eleOrder,
                                                      compOpt.numVars);

    if (compOpt.zfpMode == "accuracy") {
        ZFPAlgorithms::zfpblockwise.setAccuracy(compOpt.zfpAccuracyTolerance);
    } else if (compOpt.zfpMode == "rate") {
        ZFPAlgorithms::zfpblockwise.setRate(compOpt.zfpRate);
    } else if (compOpt.zfpMode == "precision") {
        ZFPAlgorithms::zfpblockwise.setPrecision(compOpt.zfpPrecision);
    }

    // set up for BLOSC
    BLOSCAlgorithms::bloscblockwise.setCtxSendType(sendType);
    BLOSCAlgorithms::bloscblockwise.setEleOrder(compOpt.eleOrder);
    BLOSCAlgorithms::bloscblockwise.setCompressor(compOpt.bloscCompressor);
    BLOSCAlgorithms::bloscblockwise.setUpForMultiVariable(compOpt.eleOrder,
                                                          compOpt.numVars);

    // set up for Chebyshev
    ChebyshevAlgorithms::cheby.set_compression_type(compOpt.eleOrder,
                                                    compOpt.chebyNReduced);

    dendro_compress::gaussfilter.set_ctx_send_type(sendType);
    // dendro_compress::gaussfilter.set_sigma(0.68);
    // dendro_compress::gaussfilter.set_sigma(0.5);
    dendro_compress::gaussfilter.set_sigma(0.4);
    dendro_compress::gaussfilter.set_radius(2);
    dendro_compress::gaussfilter.set_sizes(compOpt.eleOrder);

    dendro_compress::chebyfilter.set_ctx_send_type(sendType);
    dendro_compress::chebyfilter.set_sizes(compOpt.eleOrder,
                                           compOpt.chebyNReduced);

    // set up for ML Algorithms
    if (dendro_compress::COMPRESSION_OPTION == CompressionType::ONNX_MODEL) {
        std::cout << "CONFIGURING FOR ONNX" << std::endl;
        MachineLearningAlgorithms::onnxcomp.set_sizes(compOpt.eleOrder,
                                                      compOpt.numVars);
        MachineLearningAlgorithms::onnxcomp.set_models(
            compOpt.encoder_3d_path, compOpt.decoder_3d_path,
            compOpt.encoder_2d_path, compOpt.decoder_2d_path,
            compOpt.encoder_1d_path, compOpt.decoder_1d_path,
            compOpt.encoder_0d_path, compOpt.decoder_0d_path);
    } else if (dendro_compress::COMPRESSION_OPTION ==
               CompressionType::TORCH_SCRIPT) {
        MachineLearningAlgorithms::mlcomp.set_sizes(compOpt.eleOrder,
                                                    compOpt.numVars);
        MachineLearningAlgorithms::mlcomp.set_models(
            compOpt.encoder_3d_path, compOpt.decoder_3d_path,
            compOpt.encoder_2d_path, compOpt.decoder_2d_path,
            compOpt.encoder_1d_path, compOpt.decoder_1d_path,
            compOpt.encoder_0d_path, compOpt.decoder_0d_path);
    }

#ifndef DENDRO_ENABLE_ML_LIBRARIES
#pragma message("Dendro will *not* compile with ML library support!")
    if (dendro_compress::COMPRESSION_OPTION == CompressionType::ONNX_MODEL ||
        dendro_compress::COMPRESSION_OPTION == CompressionType::TORCH_SCRIPT) {
        std::cout << "WARNING: DENDRO WAS COMPILED WIHOUT ML LIBRARY SUPPORT "
                     "AND THE DETECTED COMPRESSION OPTION ASKED FOR ML. THE "
                     "PROGRAM WILL CONTINUE BUT WILL NOT ACTUALLY LOAD OR USE "
                     "MACHINE LEARNING MODELS IN COMMUNICATION, IT WILL FORCE "
                     "AN UNOPTIMIZED 'COPY-ONLY' COMPRESSION."
                  << std::endl;
    }
#endif
}

template <typename T>
std::size_t single_block_compress_3d(T* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
// TODO: need to allow these buffers to be templated!
#if 0
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_3d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_3d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_3d_compression(buffer,
                                                                     bufferOut);
            break;
#endif
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_decompress_3d(unsigned char* buffer, T* bufferOut) {
    switch (COMPRESSION_OPTION) {
// TODO: need to allow these buffers to be templated!
#if 0
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_3d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_3d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // chebyshev decompression
            return BLOSCAlgorithms::bloscblockwise.do_3d_decompression(
                buffer, bufferOut);
            break;
#endif
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_compress_3d(T* buffer,
                                             unsigned char* bufferOut,
                                             const size_t points_per_dim,
                                             const size_t dof,
                                             const size_t batchSize = 1) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_4d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc compression
            return BLOSCAlgorithms::bloscblockwise.do_4d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_3d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_3d_compression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_decompress_3d(unsigned char* buffer,
                                               T* bufferOut,
                                               const size_t batchSize = 1) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_4d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc compression
            return BLOSCAlgorithms::bloscblockwise.do_4d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_3d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_3d_decompression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_compress_2d(T* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
#if 0
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_2d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_2d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_2d_compression(buffer,
                                                                     bufferOut);
            break;
#endif
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_decompress_2d(unsigned char* buffer, T* bufferOut) {
    switch (COMPRESSION_OPTION) {
// TODO: need to allow these buffers to be templated!
#if 0
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_2d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_2d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_2d_decompression(
                buffer, bufferOut);
            break;
#endif
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_compress_2d(T* buffer,
                                             unsigned char* bufferOut,
                                             const size_t points_per_dim,
                                             const size_t dof,
                                             const size_t batchSize = 1) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_3d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc compression
            return BLOSCAlgorithms::bloscblockwise.do_3d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_2d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_2d_compression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_decompress_2d(unsigned char* buffer,
                                               T* bufferOut,
                                               const size_t batchSize = 1) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_3d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc decompression
            return BLOSCAlgorithms::bloscblockwise.do_3d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_2d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_2d_decompression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_compress_1d(T* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
// TODO: need to allow these buffers to be templated!
#if 0
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_1d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_1d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_1d_compression(buffer,
                                                                     bufferOut);
            break;
#endif
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_decompress_1d(unsigned char* buffer, T* bufferOut) {
    switch (COMPRESSION_OPTION) {
// TODO: need to allow these buffers to be templated!
#if 0
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_1d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_1d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // chebyshev decompression
            return BLOSCAlgorithms::bloscblockwise.do_1d_decompression(
                buffer, bufferOut);
            break;
#endif
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_compress_1d(T* buffer,
                                             unsigned char* bufferOut,
                                             const size_t points_per_dim,
                                             const size_t dof,
                                             const size_t batchSize = 1) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_2d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc compression
            return BLOSCAlgorithms::bloscblockwise.do_2d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_1d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_1d_compression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_decompress_1d(unsigned char* buffer,
                                               T* bufferOut,
                                               const size_t batchSize = 1) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_2d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc decompression
            return BLOSCAlgorithms::bloscblockwise.do_2d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_1d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_1d_decompression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_compress_0d(T* buffer,
                                             unsigned char* bufferOut,
                                             const size_t points_per_dim,
                                             const size_t dof,
                                             const size_t batchSize = 1) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_0d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc compression
            return BLOSCAlgorithms::bloscblockwise.do_0d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_0d_compression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_0d_compression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t single_block_all_dof_decompress_0d(unsigned char* buffer,
                                               T* bufferOut,
                                               const size_t batchSize = 1) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_0d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // blosc decompression
            return BLOSCAlgorithms::bloscblockwise.do_0d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::TORCH_SCRIPT:
            return MachineLearningAlgorithms::mlcomp.do_0d_decompression(
                buffer, bufferOut, batchSize);
            break;
        case dendro_compress::CompressionType::ONNX_MODEL:
            return MachineLearningAlgorithms::onnxcomp.do_0d_decompression(
                buffer, bufferOut, batchSize);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

template <typename T>
std::size_t blockwise_compression(
    T* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder) {
    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1;
    const size_t total_points_1d = points_per_dim;
    const size_t total_points_2d = total_points_1d * points_per_dim;
    const size_t total_points_3d = total_points_2d * points_per_dim;

    // TODO: set the compression type elsewhere
    // ChebyshevAlgorithms::cheby.set_compression_type(eleorder, 2);
    // ChebyshevAlgorithms::cheby.print();

    std::size_t comp_offset      = 0;
    std::size_t orig_offset      = 0;

    for (size_t ib = 0; ib < numBlocks; ib++) {
        // decode the value
        const auto& config = blockConfiguration[blockConfigOffset + ib];

        // xdim         = config.getX();
        // ydim         = config.getY();
        // zdim         = config.getZ();

        // get the "dimensionality" of the block
        ndim               = config.getNDim();

        // now based on the ndim, we will set up our compression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                std::memcpy(compressBuffer + comp_offset, &buffer[orig_offset],
                            sizeof(T));
                comp_offset += sizeof(T);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_compress_1d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_compress_2d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_compress_3d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise compression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }

    return comp_offset;
}

template <typename T>
std::size_t blockwise_decompression(
    T* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder) {
    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;

    // these values are used to define the output side, since we're
    // decompressing back to our values. All of the decompression methods should
    // return how many bytes to advance the compression offset.
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1;
    const size_t total_points_1d = points_per_dim;
    const size_t total_points_2d = total_points_1d * points_per_dim;
    const size_t total_points_3d = total_points_2d * points_per_dim;

    std::size_t comp_offset      = 0;
    std::size_t orig_offset      = 0;

    for (std::size_t ib = 0; ib < numBlocks; ib++) {
        const auto& config = blockConfiguration[blockConfigOffset + ib];

        // xdim         = config.getX();
        // ydim         = config.getY();
        // zdim         = config.getZ();

        // get the "dimensionality" of the block
        ndim               = config.getNDim();

        // now based on the ndim, we will use our decompression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                std::memcpy(&buffer[orig_offset], compressBuffer + comp_offset,
                            sizeof(T));
                comp_offset += sizeof(T);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_decompress_1d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_decompress_2d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_decompress_3d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise decompression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }
    return comp_offset;
}

bool debugAssertAllBlocksSameDim(
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset, const size_t numTest, const uint32_t ndim) {
    bool all_equal = true;

    for (unsigned int i = 0; i < numTest; ++i) {
        const auto& config = blockConfiguration[blockConfigOffset + i];

        // xdim         = config.getX();
        // ydim         = config.getY();
        // zdim         = config.getZ();

        // get the "dimensionality" of the block
        uint32_t ndim_curr = config.getNDim();

        if (ndim_curr != ndim) {
            std::cerr << "ERROR: THIS BLOCK DOES NOT HAVE " << ndim
                      << " DIMENSIONS: BLOCK " << blockConfigOffset + i
                      << " has dim " << ndim_curr << std::endl;
            all_equal = false;
            break;
        }
    }

    return all_equal;
}

template <typename T>
std::size_t blockwise_all_dof_compression(
    T* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // std::cout << rank << ": INSIDE COMPRESSION - numblocks: " << numBlocks
    //           << " - counts/offsets: ";
    // for (auto& iii : blockDimCounts) {
    //     std::cout << iii << " ";
    // }
    // std::cout << " - ";
    // for (auto& iii : blockDimOffsets) {
    //     std::cout << iii << " ";
    // }
    // std::cout << std::endl;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1 * dof;
    const size_t total_points_1d = points_per_dim * dof;
    const size_t total_points_2d = points_per_dim * points_per_dim * dof;
    const size_t total_points_3d =
        points_per_dim * points_per_dim * points_per_dim * dof;

    // TODO: set the compression type elsewhere
    // ChebyshevAlgorithms::cheby.set_compression_type(eleorder, 2);
    // ChebyshevAlgorithms::cheby.print();

    std::size_t comp_offset         = 0;
    std::size_t orig_offset         = 0;

    std::size_t curr_block_no       = 0;
    std::size_t curr_inner_block_no = 0;
    for (unsigned int currNdim : {3, 2, 1, 0}) {
        // see if we can get our batch size with what's left after currBlockNo

        // reset the inner block number
        curr_inner_block_no = 0;

        while (curr_inner_block_no < blockDimCounts[currNdim]) {
            unsigned int remaining_blocks =
                blockDimCounts[currNdim] - curr_inner_block_no;

            unsigned int items_to_process =
                std::min(remaining_blocks, batchSize);

            if (items_to_process == 0) {
                // break out of the loop if there are no more items to process!
                // this could go infinite...
                break;
            }

            // std::cout << rank << ": " << currNdim
            //           << ": ITEMS TO PROCESS: " << items_to_process
            //           << " CURRENTLY ON: " << curr_block_no << std::endl;
            assert((debugAssertAllBlocksSameDim(
                blockConfiguration, blockConfigOffset + curr_block_no,
                items_to_process, currNdim)));

            // then we process this amount

            // now based on the ndim, we will set up our compression methods
            switch (currNdim) {
                case 0:
                    // no compression on a single point
                    comp_offset += single_block_all_dof_compress_0d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        points_per_dim, dof, items_to_process);
                    orig_offset += total_points_0d * items_to_process;
                    break;
                case 1:
                    comp_offset += single_block_all_dof_compress_1d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        points_per_dim, dof, items_to_process);
                    orig_offset += total_points_1d * items_to_process;
                    break;
                case 2:
                    comp_offset += single_block_all_dof_compress_2d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        points_per_dim, dof, items_to_process);
                    orig_offset += total_points_2d * items_to_process;
                    break;
                case 3:
                    comp_offset += single_block_all_dof_compress_3d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        points_per_dim, dof, items_to_process);
                    orig_offset += total_points_3d * items_to_process;
                    break;
                default:
                    std::cerr
                        << "Invalid number of dimensions found when doing "
                           "blockwise compression. Exiting!"
                        << std::endl;
                    exit(0);
                    break;
            }

            curr_block_no += items_to_process;
            curr_inner_block_no += items_to_process;
        }
    }

    // make sure we got through all of them!
    assert((curr_block_no == numBlocks));

#if 0
    for (size_t ib = 0; ib < numBlocks; ib++) {
        // decode the value
        const auto& config = blockConfiguration[blockConfigOffset + ib];

        // xdim         = config.getX();
        // ydim         = config.getY();
        // zdim         = config.getZ();

        // get the "dimensionality" of the block
        ndim               = config.getNDim();

        // now based on the ndim, we will set up our compression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                comp_offset += single_block_all_dof_compress_0d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    points_per_dim, dof);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_all_dof_compress_1d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    points_per_dim, dof);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_all_dof_compress_2d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    points_per_dim, dof);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_all_dof_compress_3d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    points_per_dim, dof);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise compression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }
#endif

    return comp_offset;
}

template <typename T>
std::size_t blockwise_all_dof_decompression(
    T* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // std::cout << rank << ": INSIDE COMPRESSION - numblocks: " << numBlocks
    //           << " - counts/offsets: ";
    // for (auto& iii : blockDimCounts) {
    //     std::cout << iii << " ";
    // }
    // std::cout << " - ";
    // for (auto& iii : blockDimOffsets) {
    //     std::cout << iii << " ";
    // }
    // std::cout << std::endl;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;

    // these values are used to define the output side, since we're
    // decompressing back to our values. All of the decompression methods should
    // return how many bytes to advance the compression offset.
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1 * dof;
    const size_t total_points_1d = points_per_dim * dof;
    const size_t total_points_2d = points_per_dim * points_per_dim * dof;
    const size_t total_points_3d =
        points_per_dim * points_per_dim * points_per_dim * dof;

    std::size_t comp_offset         = 0;
    std::size_t orig_offset         = 0;

    std::size_t curr_block_no       = 0;
    std::size_t curr_inner_block_no = 0;
    for (unsigned int currNdim : {3, 2, 1, 0}) {
        // see if we can get our batch size with what's left after currBlockNo

        while (curr_inner_block_no < blockDimCounts[currNdim]) {
            unsigned int remaining_blocks =
                blockDimCounts[currNdim] - curr_inner_block_no;

            unsigned int items_to_process =
                std::min(remaining_blocks, batchSize);

            if (items_to_process == 0) {
                // break out of the loop if there are no more items to process!
                // this could go infinite...
                break;
            }

            assert((debugAssertAllBlocksSameDim(
                blockConfiguration, blockConfigOffset + curr_block_no,
                items_to_process, currNdim)));

            // then we process this amount

            // now based on the ndim, we will set up our compression methods
            switch (currNdim) {
                case 0:
                    // no compression on a single point
                    comp_offset += single_block_all_dof_decompress_0d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_0d * items_to_process;
                    break;
                case 1:
                    comp_offset += single_block_all_dof_decompress_1d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_1d * items_to_process;
                    break;
                case 2:
                    comp_offset += single_block_all_dof_decompress_2d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_2d * items_to_process;
                    break;
                case 3:
                    comp_offset += single_block_all_dof_decompress_3d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_3d * items_to_process;
                    break;
                default:
                    std::cerr
                        << "Invalid number of dimensions found when doing "
                           "blockwise decompression. Exiting!"
                        << std::endl;
                    exit(0);
                    break;
            }

            curr_block_no += items_to_process;
            curr_inner_block_no += items_to_process;
        }

        // reset the inner block number
        curr_inner_block_no = 0;
    }

    // make sure we got through all of them!
    assert((curr_block_no == numBlocks));

    if (COMPRESSION_FILTER_OPTION != FilterType::F_NONE) {
        // if (true) {
        orig_offset   = 0;

        curr_block_no = 0;
        for (unsigned int currNdim : {3, 2, 1, 0}) {
            // see if we can get our batch size with what's left after
            // currBlockNo

            unsigned int items_to_process = blockDimCounts[currNdim];

            if (items_to_process == 0) {
                // std::cout << "NO ITEMS TO PROCESS FOR DIM: " << currNdim
                //           << std::endl;
                continue;
            }

            assert((debugAssertAllBlocksSameDim(
                blockConfiguration, blockConfigOffset + curr_block_no,
                items_to_process, currNdim)));

            // then we process this amount

            // now based on the ndim, we will set up our compression methods
            switch (currNdim) {
                case 0:
                    // no filtering on a single point
                    // remember total_points_0d is equal to items to process
                    orig_offset += total_points_0d * items_to_process;
                    break;
                case 1:
                    if (COMPRESSION_FILTER_OPTION == FilterType::F_CHEBYSHEV) {
                        chebyfilter.do_1d_filtering(&buffer[orig_offset],
                                                    items_to_process * dof);
                    } else if (COMPRESSION_FILTER_OPTION ==
                               FilterType::F_GAUSSIAN) {
                        gaussfilter.do_1d_filtering(&buffer[orig_offset],
                                                    items_to_process * dof);
                    }
                    orig_offset += total_points_1d * items_to_process;
                    break;
                case 2:
                    if (COMPRESSION_FILTER_OPTION == FilterType::F_CHEBYSHEV) {
                        chebyfilter.do_2d_filtering(&buffer[orig_offset],
                                                    items_to_process * dof);
                    } else if (COMPRESSION_FILTER_OPTION ==
                               FilterType::F_GAUSSIAN) {
                        gaussfilter.do_2d_filtering(&buffer[orig_offset],
                                                    items_to_process * dof);
                    }
                    orig_offset += total_points_2d * items_to_process;
                    break;
                case 3:
                    if (COMPRESSION_FILTER_OPTION == FilterType::F_CHEBYSHEV) {
                        chebyfilter.do_3d_filtering(&buffer[orig_offset],
                                                    items_to_process * dof);
                    } else if (COMPRESSION_FILTER_OPTION ==
                               FilterType::F_GAUSSIAN) {
                        gaussfilter.do_3d_filtering(&buffer[orig_offset],
                                                    items_to_process * dof);
                    }
                    orig_offset += total_points_3d * items_to_process;
                    break;
                default:
                    std::cerr
                        << "Invalid number of dimensions found when doing "
                           "blockwise decompression. Exiting!"
                        << std::endl;
                    exit(0);
                    break;
            }

            curr_block_no += items_to_process;
        }
    }

#if 0
    for (std::size_t ib = 0; ib < numBlocks; ib++) {
        // std::cout << "ib: " << ib + 1 << "/" << numBlocks << std::endl;
        const auto& config = blockConfiguration[blockConfigOffset + ib];

        // xdim         = config.getX();
        // ydim         = config.getY();
        // zdim         = config.getZ();

        // get the "dimensionality" of the block
        ndim               = config.getNDim();

        // now based on the ndim, we will use our decompression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                comp_offset += single_block_all_dof_decompress_0d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_all_dof_decompress_1d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_all_dof_decompress_2d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_all_dof_decompress_3d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise decompression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }
#endif

    return comp_offset;
}

std::ostream& operator<<(std::ostream& out, const CompressionOptions opts) {
    return out << "<Compression Options: eleorder " << opts.eleOrder
               << ", bloscCompressor " << opts.bloscCompressor
               << ", bloscCLevel " << opts.bloscClevel << ", bloscDoShuffle "
               << opts.bloscDoShuffle << ", zfpMode " << opts.zfpMode
               << ", zfpRate " << opts.zfpRate << ", zfpAccuracy "
               << opts.zfpAccuracyTolerance << ", chebyNReduced "
               << opts.chebyNReduced << ", encoder_3d_path|decoder_3d_path "
               << opts.encoder_3d_path << "|" << opts.decoder_3d_path
               << ", encoder_2d_path|decoder_2d_path " << opts.encoder_2d_path
               << "|" << opts.decoder_2d_path
               << ", encoder_1d_path|decoder_1d_path " << opts.encoder_1d_path
               << "|" << opts.decoder_1d_path
               << ", encoder_0d_path|decoder_0d_path " << opts.encoder_0d_path
               << "|" << opts.decoder_0d_path << ">";
}

std::ostream& operator<<(std::ostream& out, const CompressionType t) {
    return out << "<CompressionType: " << COMPRESSION_TYPE_NAMES[t] << ">";
}

std::ostream& operator<<(std::ostream& out, const FilterType t) {
    return out << "<FilterType: " << FILTER_TYPE_NAMES[t] << ">";
}

// BEGIN EXPLICIT INSTANTIATIONS
template std::size_t blockwise_compression<double>(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);
template std::size_t blockwise_compression<float>(
    float* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);

template std::size_t blockwise_decompression<double>(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);
template std::size_t blockwise_decompression<float>(
    float* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);

template std::size_t blockwise_all_dof_compression<double>(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);
template std::size_t blockwise_all_dof_compression<float>(
    float* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

template std::size_t blockwise_all_dof_decompression<double>(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);
template std::size_t blockwise_all_dof_decompression<float>(
    float* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

template <typename T>
std::size_t blockwise_all_dof_compression_class(
    dendrocompression::Compression<T>* compressor, T* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // std::cout << rank << ": INSIDE COMPRESSION - numblocks: " << numBlocks
    //           << " - counts/offsets: ";
    // for (auto& iii : blockDimCounts) {
    //     std::cout << iii << " ";
    // }
    // std::cout << " - ";
    // for (auto& iii : blockDimOffsets) {
    //     std::cout << iii << " ";
    // }
    // std::cout << std::endl;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1 * dof;
    const size_t total_points_1d = points_per_dim * dof;
    const size_t total_points_2d = points_per_dim * points_per_dim * dof;
    const size_t total_points_3d =
        points_per_dim * points_per_dim * points_per_dim * dof;

    // TODO: set the compression type elsewhere
    // ChebyshevAlgorithms::cheby.set_compression_type(eleorder, 2);
    // ChebyshevAlgorithms::cheby.print();

    std::size_t comp_offset         = 0;
    std::size_t orig_offset         = 0;

    std::size_t curr_block_no       = 0;
    std::size_t curr_inner_block_no = 0;
    for (unsigned int currNdim : {3, 2, 1, 0}) {
        // see if we can get our batch size with what's left after currBlockNo

        // reset the inner block number
        curr_inner_block_no = 0;

        while (curr_inner_block_no < blockDimCounts[currNdim]) {
            unsigned int remaining_blocks =
                blockDimCounts[currNdim] - curr_inner_block_no;

            unsigned int items_to_process =
                std::min(remaining_blocks, batchSize);

            if (items_to_process == 0) {
                // break out of the loop if there are no more items to process!
                // this could go infinite...
                break;
            }

            // std::cout << rank << ": " << currNdim
            //           << ": ITEMS TO PROCESS: " << items_to_process
            //           << " CURRENTLY ON: " << curr_block_no << std::endl;
            assert((debugAssertAllBlocksSameDim(
                blockConfiguration, blockConfigOffset + curr_block_no,
                items_to_process, currNdim)));

            // then we process this amount

            // now based on the ndim, we will set up our compression methods
            switch (currNdim) {
                case 0:
                    // no compression on a single point
                    comp_offset += compressor->do_compress_0d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        items_to_process);
                    orig_offset += total_points_0d * items_to_process;
                    break;
                case 1:
                    comp_offset += compressor->do_compress_1d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        items_to_process);
                    orig_offset += total_points_1d * items_to_process;
                    break;
                case 2:
                    comp_offset += compressor->do_compress_2d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        items_to_process);
                    orig_offset += total_points_2d * items_to_process;
                    break;
                case 3:
                    comp_offset += compressor->do_compress_3d(
                        &buffer[orig_offset], compressBuffer + comp_offset,
                        items_to_process);
                    orig_offset += total_points_3d * items_to_process;
                    break;
                default:
                    std::cerr
                        << "Invalid number of dimensions found when doing "
                           "blockwise compression. Exiting!"
                        << std::endl;
                    exit(0);
                    break;
            }

            curr_block_no += items_to_process;
            curr_inner_block_no += items_to_process;
        }
    }

    // make sure we got through all of them!
    assert((curr_block_no == numBlocks));

    return comp_offset;
}

template <typename T>
std::size_t blockwise_all_dof_decompression_class(
    dendrocompression::Compression<T>* compressor, T* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // std::cout << rank << ": INSIDE COMPRESSION - numblocks: " << numBlocks
    //           << " - counts/offsets: ";
    // for (auto& iii : blockDimCounts) {
    //     std::cout << iii << " ";
    // }
    // std::cout << " - ";
    // for (auto& iii : blockDimOffsets) {
    //     std::cout << iii << " ";
    // }
    // std::cout << std::endl;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;

    // these values are used to define the output side, since we're
    // decompressing back to our values. All of the decompression methods should
    // return how many bytes to advance the compression offset.
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1 * dof;
    const size_t total_points_1d = points_per_dim * dof;
    const size_t total_points_2d = points_per_dim * points_per_dim * dof;
    const size_t total_points_3d =
        points_per_dim * points_per_dim * points_per_dim * dof;

    std::size_t comp_offset         = 0;
    std::size_t orig_offset         = 0;

    std::size_t curr_block_no       = 0;
    std::size_t curr_inner_block_no = 0;
    for (unsigned int currNdim : {3, 2, 1, 0}) {
        // see if we can get our batch size with what's left after currBlockNo

        while (curr_inner_block_no < blockDimCounts[currNdim]) {
            unsigned int remaining_blocks =
                blockDimCounts[currNdim] - curr_inner_block_no;

            unsigned int items_to_process =
                std::min(remaining_blocks, batchSize);

            if (items_to_process == 0) {
                // break out of the loop if there are no more items to process!
                // this could go infinite...
                break;
            }

            assert((debugAssertAllBlocksSameDim(
                blockConfiguration, blockConfigOffset + curr_block_no,
                items_to_process, currNdim)));

            // then we process this amount

            // now based on the ndim, we will set up our compression methods
            switch (currNdim) {
                case 0:
                    // no compression on a single point
                    comp_offset += compressor->do_decompress_0d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_0d * items_to_process;
                    break;
                case 1:
                    comp_offset += compressor->do_decompress_1d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_1d * items_to_process;
                    break;
                case 2:
                    comp_offset += compressor->do_decompress_2d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_2d * items_to_process;
                    break;
                case 3:
                    comp_offset += compressor->do_decompress_3d(
                        compressBuffer + comp_offset, &buffer[orig_offset],
                        items_to_process);
                    orig_offset += total_points_3d * items_to_process;
                    break;
                default:
                    std::cerr
                        << "Invalid number of dimensions found when doing "
                           "blockwise decompression. Exiting!"
                        << std::endl;
                    exit(0);
                    break;
            }

            curr_block_no += items_to_process;
            curr_inner_block_no += items_to_process;
        }

        // reset the inner block number
        curr_inner_block_no = 0;
    }

    // make sure we got through all of them!
    assert((curr_block_no == numBlocks));

    // TODO: copy or modify the compression filtering option

    return comp_offset;
}

template std::size_t blockwise_all_dof_compression_class<float>(
    dendrocompression::Compression<float>* compressor, float* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

template std::size_t blockwise_all_dof_compression_class<double>(
    dendrocompression::Compression<double>* compressor, double* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

#if 0
template std::size_t blockwise_all_dof_compression_class<float, float>(
    dendrocompression::Compression<float>* compressor, float* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

template std::size_t blockwise_all_dof_compression_class<double, double>(
    dendrocompression::Compression<double>* compressor, double* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

// template std::size_t blockwise_all_dof_compression_class<double, float>(
//     dendrocompression::Compression<float>* compressor, double* buffer,
//     unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
//     const std::vector<sm_config::SMConfig>& blockConfiguration,
//     const size_t blockConfigOffset,
//     const std::array<unsigned int, 4>& blockDimCounts,
//     const std::array<unsigned int, 4>& blockDimOffsets, const size_t
//     eleorder, const unsigned int batchSize);
//
// template std::size_t blockwise_all_dof_compression_class<float, double>(
//     const dendrocompression::Compression<double>* compressor, float* buffer,
//     unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
//     const std::vector<sm_config::SMConfig>& blockConfiguration,
//     const size_t blockConfigOffset,
//     const std::array<unsigned int, 4>& blockDimCounts,
//     const std::array<unsigned int, 4>& blockDimOffsets, const size_t
//     eleorder, const unsigned int batchSize);
#endif

template std::size_t blockwise_all_dof_decompression_class<float>(
    dendrocompression::Compression<float>* compressor, float* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

template std::size_t blockwise_all_dof_decompression_class<double>(
    dendrocompression::Compression<double>* compressor, double* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

#if 0
template std::size_t blockwise_all_dof_decompression_class<float, float>(
    dendrocompression::Compression<float>* compressor, float* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

template std::size_t blockwise_all_dof_decompression_class<double, double>(
    dendrocompression::Compression<double>* compressor, double* buffer,
    unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
    const std::vector<sm_config::SMConfig>& blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4>& blockDimCounts,
    const std::array<unsigned int, 4>& blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize);

// template std::size_t blockwise_all_dof_decompression_class<double, float>(
//     dendrocompression::Compression<float>* compressor, double* buffer,
//     unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
//     const std::vector<sm_config::SMConfig>& blockConfiguration,
//     const size_t blockConfigOffset,
//     const std::array<unsigned int, 4>& blockDimCounts,
//     const std::array<unsigned int, 4>& blockDimOffsets, const size_t
//     eleorder, const unsigned int batchSize);
//
// template std::size_t blockwise_all_dof_decompression_class<float, double>(
//     const dendrocompression::Compression<double>* compressor, float* buffer,
//     unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
//     const std::vector<sm_config::SMConfig>& blockConfiguration,
//     const size_t blockConfigOffset,
//     const std::array<unsigned int, 4>& blockDimCounts,
//     const std::array<unsigned int, 4>& blockDimOffsets, const size_t
//     eleorder, const unsigned int batchSize);
//
// template std::size_t blockwise_all_dof_decompression_class<double, float>(
//     const dendrocompression::Compression<float>* compressor, double* buffer,
//     unsigned char* compressBuffer, const size_t numBlocks, const size_t dof,
//     const std::vector<sm_config::SMConfig>& blockConfiguration,
//     const size_t blockConfigOffset,
//     const std::array<unsigned int, 4>& blockDimCounts,
//     const std::array<unsigned int, 4>& blockDimOffsets, const size_t
//     eleorder, const unsigned int batchSize);
//
#endif

std::unique_ptr<dendrocompression::Compression<double>> compressor_double;
std::unique_ptr<dendrocompression::Compression<float>> compressor_float;

void setUpCompressor(dendrocompression::CompressionType compressor_type,
                     std::vector<std::any> compressor_parameters) {
    // simple as just creating the compressor through the factory
    // TODO: should adjust based on float/double compressor (and other
    // potential types!)
    std::cout << "COMPRESSOR TYPE: " << compressor_type << std::endl;
    compressor_double = dendrocompression::doubleCompressor.create(
        compressor_type, compressor_parameters);

    std::cout << "Now building float: " << compressor_type << std::endl;
    // TODO: this can be "smarter" based on the parameter that we want to use
    // for setup
    compressor_float = dendrocompression::floatCompressor.create(
        compressor_type, compressor_parameters);
}

}  // namespace dendro_compress
