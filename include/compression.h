/**
 * @file SVDAlgorithms.h
 * @brief This header file provides SVD-based data compression and decompression
 * algorithms.
 *
 * No padding.
 *
 */

#pragma once

#ifdef DENDRO_ENABLE_ML_LIBRARIES
#include <onnxruntime_cxx_api.h>
#include <torch/script.h>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#endif

#include <omp.h>

#include "asyncExchangeContex.h"
#include "lapac.h"
#include "scattermapConfig.h"

// Disables Eigen's memory alignment which could lead to extra memory padding.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#define EIGEN_DONT_ALIGN
// #include <Eigen/Dense>
// #include <Eigen/SVD>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#ifdef _DEBUG_ZFP_
class RunningRMSE {
   private:
    double sum_squared_errors;
    size_t total_points;

   public:
    RunningRMSE() : sum_squared_errors(0.0), total_points(0) {}

    void update(const double *original, const double *decompressed,
                size_t pts) {
        if (original == nullptr || decompressed == nullptr || pts == 0) {
            return;
        }

        for (size_t i = 0; i < pts; ++i) {
            double error = original[i] - decompressed[i];
            sum_squared_errors += error * error;
        }

        total_points += pts;
    }

    double get_rmse() const {
        if (total_points == 0) {
            return 0.0;
        }

        double mean_sq_error = sum_squared_errors / total_points;
        return std::sqrt(mean_sq_error);
    }
    void reset() {
        sum_squared_errors = 0.0;
        total_points       = 0;
    }
};
#endif

namespace MachineLearningAlgorithms {

#ifdef DENDRO_ENABLE_ML_LIBRARIES
template <typename T>
Ort::Value createOnnxTensorFromData(const T *originalMatrix,
                                    const unsigned int nPoints,
                                    const unsigned int nBatches,
                                    std::vector<float> &floatBuffer,
                                    const Ort::MemoryInfo &meminfo,
                                    std::vector<long> &inputShape) {
    if constexpr (std::is_same_v<T, double>) {
        // convert to floats
        std::transform(originalMatrix, originalMatrix + (nPoints * nBatches),
                       floatBuffer.begin(),
                       [](double d) { return static_cast<float>(d); });

        // then return the tensor data
        return Ort::Value::CreateTensor<float>(
            meminfo, const_cast<float *>(floatBuffer.data()),
            floatBuffer.size(), inputShape.data(), inputShape.size());
    } else if constexpr (std::is_same_v<T, float>) {
        // no need to convert if we need the floats as output
        return Ort::Value::CreateTensor<float>(
            meminfo, const_cast<float *>(originalMatrix), nPoints * nBatches,
            inputShape.data(), inputShape.size());
    } else {
        std::cerr << "Somehow ONNX Tensor Creation FAILED in templating!"
                  << std::endl;
        exit(-1);
        return Ort::Value(nullptr);
    }
}
#endif

class ONNXCompression {
   public:
#ifdef DENDRO_ENABLE_ML_LIBRARIES
    ONNXCompression(const size_t &eleOrder = 6, const size_t &numVars = 1)
        : m_eleOrder(eleOrder),
          m_numVars(numVars),
          m_memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                   OrtMemTypeDefault)) {
#else
    ONNXCompression(const size_t &eleOrder = 6, const size_t &numVars = 1)
        : m_eleOrder(eleOrder), m_numVars(numVars) {
#endif
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts = pointsPerDim * pointsPerDim * pointsPerDim * m_numVars;
        m_total2DPts = pointsPerDim * pointsPerDim * m_numVars;
        m_total1DPts = pointsPerDim * m_numVars;
        m_total0DPts = m_numVars;

        m_input_shape_3d = {1, static_cast<long>(numVars), pointsPerDim,
                            pointsPerDim, pointsPerDim};
        m_input_shape_2d = {1, static_cast<long>(numVars), pointsPerDim,
                            pointsPerDim};
        m_input_shape_1d = {1, static_cast<long>(numVars), pointsPerDim};
        m_input_shape_0d = {1, static_cast<long>(numVars)};

        m_doubleToFloatBuffer_3d.resize(m_total3DPts);
        m_doubleToFloatBuffer_2d.resize(m_total2DPts);
        m_doubleToFloatBuffer_1d.resize(m_total1DPts);
        m_doubleToFloatBuffer_0d.resize(m_total0DPts);

#ifdef DENDRO_ENABLE_ML_LIBRARIES
        // m_memory_info =
        //     Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        m_session_options.SetIntraOpNumThreads(1);
        m_session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#endif
    }

    void set_sizes(const size_t &eleOrder = 6, const size_t &numVars = 1) {
        m_eleOrder                = eleOrder;
        m_numVars                 = numVars;
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts = pointsPerDim * pointsPerDim * pointsPerDim * m_numVars;
        m_total2DPts = pointsPerDim * pointsPerDim * m_numVars;
        m_total1DPts = pointsPerDim * m_numVars;
        m_total0DPts = m_numVars;

        m_input_shape_3d = {1, static_cast<long>(numVars), pointsPerDim,
                            pointsPerDim, pointsPerDim};
        m_input_shape_2d = {1, static_cast<long>(numVars), pointsPerDim,
                            pointsPerDim};
        m_input_shape_1d = {1, static_cast<long>(numVars), pointsPerDim};
        m_input_shape_0d = {1, static_cast<long>(numVars)};

        m_doubleToFloatBuffer_3d = std::vector<float>(m_total3DPts);
        m_doubleToFloatBuffer_2d = std::vector<float>(m_total2DPts);
        m_doubleToFloatBuffer_1d = std::vector<float>(m_total1DPts);
        m_doubleToFloatBuffer_0d = std::vector<float>(m_total0DPts);
    }

    void set_models(const std::string &encoder_3d_path,
                    const std::string &decoder_3d_path,
                    const std::string &encoder_2d_path,
                    const std::string &decoder_2d_path,
                    const std::string &encoder_1d_path,
                    const std::string &decoder_1d_path,
                    const std::string &encoder_0d_path,
                    const std::string &decoder_0d_path) {
        m_encoder_3d_path = encoder_3d_path;
        m_decoder_3d_path = decoder_3d_path;
        m_encoder_2d_path = encoder_2d_path;
        m_decoder_2d_path = decoder_2d_path;
        m_encoder_1d_path = encoder_1d_path;
        m_decoder_1d_path = decoder_1d_path;
        m_encoder_0d_path = encoder_0d_path;
        m_decoder_0d_path = decoder_0d_path;

#ifdef DENDRO_ENABLE_ML_LIBRARIES
        // then attempt to load 3D
        m_3d_encoder = std::make_unique<Ort::Session>(
            m_env, m_encoder_3d_path.c_str(), m_session_options);
        m_3d_decoder = std::make_unique<Ort::Session>(
            m_env, m_decoder_3d_path.c_str(), m_session_options);

        // 2D
        m_2d_encoder = std::make_unique<Ort::Session>(
            m_env, m_encoder_2d_path.c_str(), m_session_options);
        m_2d_decoder = std::make_unique<Ort::Session>(
            m_env, m_decoder_2d_path.c_str(), m_session_options);

        // 1D
        m_1d_encoder = std::make_unique<Ort::Session>(
            m_env, m_encoder_1d_path.c_str(), m_session_options);
        m_1d_decoder = std::make_unique<Ort::Session>(
            m_env, m_decoder_1d_path.c_str(), m_session_options);

        // TODO: 0D, will need checks for it

        // allocator that helps us get the input and output names
        Ort::AllocatorWithDefaultOptions allocator;

        // --------
        // 3d Checks

        // CALCULATE THE OUTPUT SIZE OF THE ENCODER TO STORE INTERNALLY
        std::vector<float> test_data(m_total3DPts, 1.0);

        Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
            m_memory_info, const_cast<float *>(test_data.data()),
            test_data.size(), m_input_shape_3d.data(), m_input_shape_3d.size());

        // this returns a smart pointer, which we'll just clear at the end of
        // the function anyway
        auto output_name_ptr =
            m_3d_encoder->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        m_output_name_3d    = output_name_ptr.get();

        auto input_name_ptr = m_3d_encoder->GetInputNameAllocated(0, allocator);
        m_input_name_3d     = input_name_ptr.get();

        const char *input_names_3d[]  = {m_input_name_3d.c_str()};
        const char *output_names_3d[] = {m_output_name_3d.c_str()};
        auto output =
            m_3d_encoder->Run(Ort::RunOptions{nullptr}, input_names_3d,
                              &tensor_data, 1, output_names_3d, 1);

        m_nOuts3dEncoder =
            output[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // decoder names
        output_name_ptr = m_3d_decoder->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        m_decoder_output_name_3d = output_name_ptr.get();

        input_name_ptr = m_3d_decoder->GetInputNameAllocated(0, allocator);
        m_decoder_input_name_3d = input_name_ptr.get();

        // now we can do the same for other dimensionalities

        // --------
        // 2d Checks
        test_data.resize(m_total2DPts);
        // override tensor data
        tensor_data = Ort::Value::CreateTensor<float>(
            m_memory_info, const_cast<float *>(test_data.data()),
            test_data.size(), m_input_shape_2d.data(), m_input_shape_2d.size());
        output_name_ptr  = m_2d_encoder->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        m_output_name_2d = output_name_ptr.get();

        input_name_ptr   = m_2d_encoder->GetInputNameAllocated(0, allocator);
        m_input_name_2d  = input_name_ptr.get();

        const char *input_names_2d[]  = {m_input_name_2d.c_str()};
        const char *output_names_2d[] = {m_output_name_2d.c_str()};
        output = m_2d_encoder->Run(Ort::RunOptions{nullptr}, input_names_2d,
                                   &tensor_data, 1, output_names_2d, 1);

        m_nOuts2dEncoder =
            output[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // decoder names
        output_name_ptr = m_2d_decoder->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        m_decoder_output_name_2d = output_name_ptr.get();

        input_name_ptr = m_2d_decoder->GetInputNameAllocated(0, allocator);
        m_decoder_input_name_2d = input_name_ptr.get();

        // --------
        // 1d Checks
        test_data.resize(m_total1DPts);
        // override tensor data
        tensor_data = Ort::Value::CreateTensor<float>(
            m_memory_info, const_cast<float *>(test_data.data()),
            test_data.size(), m_input_shape_1d.data(), m_input_shape_1d.size());
        output_name_ptr  = m_1d_encoder->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        m_output_name_1d = output_name_ptr.get();

        input_name_ptr   = m_1d_encoder->GetInputNameAllocated(0, allocator);
        m_input_name_1d  = input_name_ptr.get();

        const char *input_names_1d[]  = {m_input_name_1d.c_str()};
        const char *output_names_1d[] = {m_output_name_1d.c_str()};
        output = m_1d_encoder->Run(Ort::RunOptions{nullptr}, input_names_1d,
                                   &tensor_data, 1, output_names_1d, 1);

        m_nOuts1dEncoder =
            output[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // decoder names
        output_name_ptr = m_1d_decoder->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        m_decoder_output_name_1d = output_name_ptr.get();

        input_name_ptr = m_1d_decoder->GetInputNameAllocated(0, allocator);
        m_decoder_input_name_1d = input_name_ptr.get();

        m_decoder_shape_3d      = {1, m_nOuts3dEncoder};
        m_decoder_shape_2d      = {1, m_nOuts2dEncoder};
        m_decoder_shape_1d      = {1, m_nOuts1dEncoder};
#endif

        // TODO: input names for decoder, though the export script should have
        // them be the same
    }

    template <typename T>
    size_t do_3d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1);

    template <typename T>
    size_t do_3d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1);

    template <typename T>
    size_t do_2d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1);

    template <typename T>
    size_t do_2d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1);

    template <typename T>
    size_t do_1d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1);

    template <typename T>
    size_t do_1d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1);

    template <typename T>
    size_t do_0d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1);

    template <typename T>
    size_t do_0d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1);

   private:
    unsigned int m_eleOrder;
    unsigned int m_pointsPerDim;
    unsigned int m_numVars;
    unsigned int m_total3DPts;
    unsigned int m_total2DPts;
    unsigned int m_total1DPts;
    unsigned int m_total0DPts;

    std::vector<int64_t> m_input_shape_3d;
    std::vector<int64_t> m_input_shape_2d;
    std::vector<int64_t> m_input_shape_1d;
    std::vector<int64_t> m_input_shape_0d;

    std::vector<int64_t> m_decoder_shape_3d;
    std::vector<int64_t> m_decoder_shape_2d;
    std::vector<int64_t> m_decoder_shape_1d;
    std::vector<int64_t> m_decoder_shape_0d;

    std::string m_encoder_3d_path;
    std::string m_decoder_3d_path;
    std::string m_encoder_2d_path;
    std::string m_decoder_2d_path;
    std::string m_encoder_1d_path;
    std::string m_decoder_1d_path;
    std::string m_encoder_0d_path;
    std::string m_decoder_0d_path;

    std::string m_input_name_3d;
    std::string m_output_name_3d;
    std::string m_input_name_2d;
    std::string m_output_name_2d;
    std::string m_input_name_1d;
    std::string m_output_name_1d;
    std::string m_input_name_0d;
    std::string m_output_name_0d;

    std::string m_decoder_input_name_3d;
    std::string m_decoder_output_name_3d;
    std::string m_decoder_input_name_2d;
    std::string m_decoder_output_name_2d;
    std::string m_decoder_input_name_1d;
    std::string m_decoder_output_name_1d;
    std::string m_decoder_input_name_0d;
    std::string m_decoder_output_name_0d;

#ifdef DENDRO_ENABLE_ML_LIBRARIES
    // Ort::Env m_env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
    //                "AutoencodersHandler");
    Ort::Env m_env;
    Ort::SessionOptions m_session_options;
    Ort::MemoryInfo m_memory_info;

    std::unique_ptr<Ort::Session> m_3d_encoder;
    std::unique_ptr<Ort::Session> m_3d_decoder;
    std::unique_ptr<Ort::Session> m_2d_encoder;
    std::unique_ptr<Ort::Session> m_2d_decoder;
    std::unique_ptr<Ort::Session> m_1d_encoder;
    std::unique_ptr<Ort::Session> m_1d_decoder;
    std::unique_ptr<Ort::Session> m_0d_encoder;
    std::unique_ptr<Ort::Session> m_0d_decoder;
#endif

    unsigned int m_nOuts3dEncoder = 0;
    unsigned int m_nOuts2dEncoder = 0;
    unsigned int m_nOuts1dEncoder = 0;
    unsigned int m_nOuts0dEncoder = 0;

    std::vector<float> m_doubleToFloatBuffer_3d;
    std::vector<float> m_doubleToFloatBuffer_2d;
    std::vector<float> m_doubleToFloatBuffer_1d;
    std::vector<float> m_doubleToFloatBuffer_0d;

    // unused, but considered for other options
    ot::CTXSendType m_ctxSendType;
};

#ifdef DENDRO_ENABLE_ML_LIBRARIES
template <typename T>
torch::Tensor convertDataToModelType(std::vector<T> &in,
                                     const char *module_type) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "T must be float or double for conversion!");

    if constexpr (std::is_same_v<T, float>) {
        if (strcmp(module_type, "float") == 0) {
            return torch::from_blob(const_cast<float *>(in.data()),
                                    {1, static_cast<long>(in.size())},
                                    torch::kFloat);
        } else if (strcmp(module_type, "double") == 0) {
            std::vector<double> double_output(in.size());
            std::transform(in.begin(), in.end(), double_output.begin(),
                           [](float d) { return static_cast<double>(d); });
            return torch::from_blob(double_output.data(),
                                    {1, static_cast<long>(in.size())},
                                    torch::kDouble);
        } else {
            std::cerr << "Model data type not currently supported!"
                      << std::endl;
            exit(0);
            return torch::Tensor();
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (strcmp(module_type, "float") == 0) {
            std::vector<float> float_output(in.size());
            std::transform(in.begin(), in.end(), float_output.begin(),
                           [](double d) { return static_cast<float>(d); });
            return torch::from_blob(float_output.data(),
                                    {1, static_cast<long>(in.size())},
                                    torch::kFloat);
        } else if (strcmp(module_type, "double") == 0) {
            return torch::from_blob(
                in.data(), {1, static_cast<long>(in.size())}, torch::kDouble);
        } else {
            std::cerr << "Model data type not currently supported!"
                      << std::endl;
            exit(0);
            return torch::Tensor();
        }
    } else {
        std::cerr << "Internal error: T managed to not be a float or a double "
                     "in data conversion wrapper."
                  << std::endl;
        exit(0);
        return torch::Tensor();
    }
}

template <typename T>
torch::Tensor convertDataToModelType(const T *in, size_t n_pts,
                                     size_t batch_size,
                                     const char *module_type) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "T must be float or double for conversion!");

    auto create_tensor = [n_pts, batch_size](auto *data_ptr, torch::Dtype dtype,
                                             auto data_holder) {
        return torch::from_blob(
            data_ptr, {static_cast<long>(batch_size), static_cast<long>(n_pts)},
            // NOTE: in order to actually keep the tensor data alive, the data
            // holder needs to be set up properly
            [data_holder](void *) { /* deleter keeps data_holder alive */ },
            torch::TensorOptions().dtype(dtype));
    };

    if constexpr (std::is_same_v<T, float>) {
        if (strcmp(module_type, "float") == 0) {
            return create_tensor(const_cast<float *>(in), torch::kFloat,
                                 nullptr);
        } else if (strcmp(module_type, "double") == 0) {
            auto double_data =
                std::make_shared<std::vector<double>>(n_pts * batch_size);
            std::transform(in, in + (n_pts * batch_size), double_data->begin(),
                           [](float f) { return static_cast<double>(f); });
            return create_tensor(double_data->data(), torch::kDouble,
                                 double_data);
        }
    } else {
        if (strcmp(module_type, "float") == 0) {
            auto float_data =
                std::make_shared<std::vector<float>>(n_pts * batch_size);
            std::transform(in, in + (n_pts * batch_size), float_data->begin(),
                           [](double d) { return static_cast<float>(d); });
            return create_tensor(float_data->data(), torch::kFloat, float_data);
        } else if (strcmp(module_type, "double") == 0) {
            return create_tensor(const_cast<double *>(in), torch::kDouble,
                                 nullptr);
        }
    }

    throw std::invalid_argument("Unsupported data type: " +
                                std::string(module_type));
}

inline torch::Tensor reshapeTensor3DBlock(torch::Tensor input_tensor,
                                          int64_t numVars, int64_t nx,
                                          int64_t batch_size) {
    std::vector<int64_t> new_shape = {batch_size, numVars, nx, nx, nx};

    try {
        return input_tensor.reshape(torch::IntArrayRef(new_shape));
    } catch (const std::exception &e) {
        std::cerr << "Error during reshape: " << e.what() << std::endl;
        exit(-1);
    }
}

inline torch::Tensor reshapeTensor2DBlock(torch::Tensor input_tensor,
                                          int64_t numVars, int64_t nx,
                                          int64_t batch_size) {
    std::vector<int64_t> new_shape = {batch_size, numVars, nx, nx};

    try {
        return input_tensor.reshape(torch::IntArrayRef(new_shape));
    } catch (const std::exception &e) {
        std::cerr << "Error during reshape: " << e.what() << std::endl;
        exit(-1);
    }
}

inline torch::Tensor reshapeTensor1DBlock(torch::Tensor input_tensor,
                                          int64_t numVars, int64_t nx,
                                          int64_t batch_size) {
    std::vector<int64_t> new_shape = {batch_size, numVars, nx};

    try {
        return input_tensor.reshape(torch::IntArrayRef(new_shape));
    } catch (const std::exception &e) {
        std::cerr << "Error during reshape: " << e.what() << std::endl;
        exit(-1);
    }
}
#endif

class TorchScriptCompression {
   public:
    TorchScriptCompression(const size_t &eleOrder = 6,
                           const size_t &numVars  = 1)
        : m_eleOrder(eleOrder), m_numVars(numVars) {
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts = pointsPerDim * pointsPerDim * pointsPerDim * m_numVars;
        m_total2DPts = pointsPerDim * pointsPerDim * m_numVars;
        m_total1DPts = pointsPerDim * m_numVars;
        m_total0DPts = m_numVars;
    }

    void set_sizes(const size_t &eleOrder = 6, const size_t &numVars = 1) {
        m_eleOrder                = eleOrder;
        m_numVars                 = numVars;
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts = pointsPerDim * pointsPerDim * pointsPerDim * m_numVars;
        m_total2DPts = pointsPerDim * pointsPerDim * m_numVars;
        m_total1DPts = pointsPerDim * m_numVars;
        m_total0DPts = m_numVars;
    }

    void set_models(const std::string &encoder_3d_path,
                    const std::string &decoder_3d_path,
                    const std::string &encoder_2d_path,
                    const std::string &decoder_2d_path,
                    const std::string &encoder_1d_path,
                    const std::string &decoder_1d_path,
                    const std::string &encoder_0d_path,
                    const std::string &decoder_0d_path) {
        m_encoder_3d_path = encoder_3d_path;
        m_decoder_3d_path = decoder_3d_path;
        m_encoder_2d_path = encoder_2d_path;
        m_decoder_2d_path = decoder_2d_path;
        m_encoder_1d_path = encoder_1d_path;
        m_decoder_1d_path = decoder_1d_path;
        m_encoder_0d_path = encoder_0d_path;
        m_decoder_0d_path = decoder_0d_path;

#ifdef DENDRO_ENABLE_ML_LIBRARIES
        // then attempt to load 3D
        try {
            m_3d_encoder = torch::jit::load(m_encoder_3d_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 3d encoder model! - attempted "
                         "to load: "
                      << m_encoder_3d_path << std::endl;
            exit(-1);
        }
        try {
            m_3d_decoder = torch::jit::load(m_decoder_3d_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 3d decoder model! - attempted "
                         "to load: "
                      << m_decoder_3d_path << std::endl;
            exit(-1);
        }

        // 2D
        try {
            m_2d_encoder = torch::jit::load(m_encoder_2d_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 2d encoder model! - attempted "
                         "to load: "
                      << m_encoder_2d_path << std::endl;
            exit(-1);
        }
        try {
            m_2d_decoder = torch::jit::load(m_decoder_2d_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 2d decoder model! - attempted "
                         "to load: "
                      << m_decoder_2d_path << std::endl;
            exit(-1);
        }

        // 1D
        try {
            m_1d_encoder = torch::jit::load(m_encoder_1d_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 1d encoder model! - attempted "
                         "to load: "
                      << m_encoder_1d_path << std::endl;
            exit(-1);
        }
        try {
            m_1d_decoder = torch::jit::load(m_decoder_1d_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 1d encoder model! - attempted "
                         "to load: "
                      << m_decoder_1d_path << std::endl;
            exit(-1);
        }

        // TODO: 0D, will need checks for it

        // now we can do quick checks on the output of the
        // encoder/decoder pairs
        std::vector<float> test_data(m_total3DPts, 1.0);
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor output;
        torch::Tensor input_tensor;

        input_tensor = convertDataToModelType(test_data, "float");
        inputs.push_back(input_tensor);

        try {
            output = m_3d_encoder.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            // NOTE: torch throws a runtime_error if there's ever a
            // mismatch, anything else we don't want to handle
            std::cerr << "Error when attempting to run the 3d encoder, it's "
                         "possible the input size is incorrect!\n";
            exit(-1);
        }

        // index 0 is batch size, so we'll store this!
        m_nOuts3dEncoder = output.sizes()[1];

        inputs.clear();
        inputs.push_back(output);

        try {
            output = m_3d_decoder.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            std::cerr << "Error when attempting to run the 3d decoder on "
                         "initialization, it's "
                         "possible the input size is incorrect or it doesn't "
                         "match with the encoder!\n";
            exit(-1);
        }
        // if we were successful, we've at least got matching data

        // NOW CHECK 2D
        test_data.resize(m_total2DPts);
        input_tensor = convertDataToModelType(test_data, "float");
        inputs.clear();
        inputs.push_back(input_tensor);
        try {
            output = m_2d_encoder.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            // NOTE: torch throws a runtime_error if there's ever a
            // mismatch, anything else we don't want to handle
            std::cerr << "Error when attempting to run the 2d encoder, it's "
                         "possible the input size is incorrect!\n";
            exit(-1);
        }

        // index 0 is batch size, so we'll store this!
        m_nOuts2dEncoder = output.sizes()[1];

        inputs.clear();
        inputs.push_back(output);

        try {
            output = m_2d_decoder.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            std::cerr << "Error when attempting to run the 2d decoder on "
                         "initialization, it's "
                         "possible the input size is incorrect or it doesn't "
                         "match with the encoder!\n";
            exit(-1);
        }

        // NOW CHECK 1D
        test_data.resize(m_total1DPts);
        input_tensor = convertDataToModelType(test_data, "float");
        inputs.clear();
        inputs.push_back(input_tensor);
        try {
            output = m_1d_encoder.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            // NOTE: torch throws a runtime_error if there's ever a
            // mismatch, anything else we don't want to handle
            std::cerr << "Error when attempting to run the 1d encoder, it's "
                         "possible the input size is incorrect!\n";
            exit(-1);
        }

        // index 0 is batch size, so we'll store this!
        m_nOuts1dEncoder = output.sizes()[1];

        inputs.clear();
        inputs.push_back(output);

        try {
            output = m_1d_decoder.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            std::cerr << "Error when attempting to run the 2d decoder on "
                         "initialization, it's "
                         "possible the input size is incorrect or it doesn't "
                         "match with the encoder!\n";
            exit(-1);
        }

        // and we've finished loading the models and getting things we
        // needed
        // TODO: 0D model information

#endif
    }

    template <typename T>
    size_t do_3d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);

    template <typename T>
    size_t do_3d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_2d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);

    template <typename T>
    size_t do_2d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_1d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);

    template <typename T>
    size_t do_1d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_0d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);

    template <typename T>
    size_t do_0d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

   private:
    unsigned int m_eleOrder;
    unsigned int m_pointsPerDim;
    unsigned int m_numVars;
    unsigned int m_total3DPts;
    unsigned int m_total2DPts;
    unsigned int m_total1DPts;
    unsigned int m_total0DPts;
    std::string m_encoder_3d_path;
    std::string m_decoder_3d_path;
    std::string m_encoder_2d_path;
    std::string m_decoder_2d_path;
    std::string m_encoder_1d_path;
    std::string m_decoder_1d_path;
    std::string m_encoder_0d_path;
    std::string m_decoder_0d_path;

#ifdef DENDRO_ENABLE_ML_LIBRARIES
    torch::jit::Module m_3d_encoder;
    torch::jit::Module m_3d_decoder;
    torch::jit::Module m_2d_encoder;
    torch::jit::Module m_2d_decoder;
    torch::jit::Module m_1d_encoder;
    torch::jit::Module m_1d_decoder;
    torch::jit::Module m_0d_encoder;
    torch::jit::Module m_0d_decoder;
#endif

    unsigned int m_nOuts3dEncoder;
    unsigned int m_nOuts2dEncoder;
    unsigned int m_nOuts1dEncoder;
    unsigned int m_nOuts0dEncoder;

    ot::CTXSendType m_ctxSendType;
};

}  // namespace MachineLearningAlgorithms

namespace ChebyshevAlgorithms {

class ChebyshevCompression {
   public:
    ChebyshevCompression() {
        // set default to ele6, out3
        set_chebyshev_mat_ele6_out3_dim1();
        set_chebyshev_mat_ele6_out3_dim2();
        set_chebyshev_mat_ele6_out3_dim3();

        recalculate_byte_sizes();
    }

    ChebyshevCompression(const size_t &eleOrder = 6,
                         const size_t &nReduced = 3) {
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
            }
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

        recalculate_byte_sizes();
    }

    ~ChebyshevCompression() {
        if (A_cheb_dim1 != nullptr) {
            delete[] A_cheb_dim1;
            A_cheb_dim1 = nullptr;
        }
        if (A_cheb_dim2 != nullptr) {
            delete[] A_cheb_dim2;
            A_cheb_dim2 = nullptr;
        }
        if (A_cheb_dim3 != nullptr) {
            delete[] A_cheb_dim3;
            A_cheb_dim3 = nullptr;
        }
    }

    void set_compression_type(const size_t &eleOrder = 6,
                              const size_t &nReduced = 3);

    void print() {
        std::cout << "ChebyShev Info: mat3d Dims: " << cheb_dim3_decomp << ", "
                  << cheb_dim3_comp << " | mat2d Dims: " << cheb_dim2_decomp
                  << ", " << cheb_dim2_comp
                  << " | mat1d Dims: " << cheb_dim1_decomp << ", "
                  << cheb_dim1_comp
                  << " with compressed byte sizes (1, 2, 3): " << bytes_1d
                  << ", " << bytes_2d << ", " << bytes_3d << std::endl;
    }

    void do_array_norm(double *array, const size_t count, double &minVal,
                       double &maxVal);
    void undo_array_norm(double *array, const size_t count, const double minVal,
                         const double maxVal);

    size_t do_3d_compression(double *originalMatrix,
                             unsigned char *outputArray);
    size_t do_3d_decompression(unsigned char *compressedBuffer,
                               double *outputArray);

    size_t do_2d_compression(double *originalMatrix,
                             unsigned char *outputArray);
    size_t do_2d_decompression(unsigned char *compressedBuffer,
                               double *outputArray);

    size_t do_1d_compression(double *originalMatrix,
                             unsigned char *outputArray);
    size_t do_1d_decompression(unsigned char *compressedBuffer,
                               double *outputArray);

   private:
    double *A_cheb_dim1  = nullptr;
    double *A_cheb_dim2  = nullptr;
    double *A_cheb_dim3  = nullptr;

    // using ints to avoid casting for dgemm_
    int cheb_dim1_decomp = 0;
    int cheb_dim1_comp   = 0;
    int cheb_dim2_decomp = 0;
    int cheb_dim2_comp   = 0;
    int cheb_dim3_decomp = 0;
    int cheb_dim3_comp   = 0;
    int single_dim       = 1;
    double alpha         = 1.0;
    double beta          = 0.0;

    // then know how many total bytes we're going to use
    unsigned int doubles_1d;
    unsigned int doubles_2d;
    unsigned int doubles_3d;
    unsigned int bytes_1d;
    unsigned int bytes_2d;
    unsigned int bytes_3d;

    void recalculate_byte_sizes() {
        doubles_1d = 2 + cheb_dim1_comp;
        doubles_2d = 2 + cheb_dim2_comp;
        doubles_3d = 2 + cheb_dim3_comp;
        bytes_1d   = doubles_1d * sizeof(double);
        bytes_2d   = doubles_2d * sizeof(double);
        bytes_3d   = doubles_3d * sizeof(double);
    }

#include "generated/cheb_transform_ele6.inc.h"
#include "generated/cheb_transform_ele8.inc.h"
};

// build up an object that we can just use
extern ChebyshevCompression cheby;

}  // namespace ChebyshevAlgorithms

#include <zfp.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

namespace ZFPAlgorithms {

class ZFPCompression {
   public:
    ZFPCompression(const size_t &eleOrder = 6, const double rate = 5.0,
                   const ot::CTXSendType sendtype = ot::CTXSendType::CTX_DOUBLE)
        : eleOrder(eleOrder), rate(rate), m_ctxSendType(sendtype) {
        zfp_num_per_dim = eleOrder - 1;

        zfp_dim0_decomp = 1;
        zfp_dim1_decomp = zfp_num_per_dim;
        zfp_dim2_decomp = zfp_dim1_decomp * zfp_num_per_dim;
        zfp_dim3_decomp = zfp_dim2_decomp * zfp_num_per_dim;

        // TODO: calculate if the rate is too large

        // streams, by default are in set rate mode, this is good for
        // knowing our size
        zfp3d           = zfp_stream_open(NULL);
        zfp2d           = zfp_stream_open(NULL);
        zfp1d           = zfp_stream_open(NULL);

        if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
            fieldType = zfp_type_double;

            zfp_stream_set_rate(zfp3d, rate, zfp_type_double, 3, 0);
            field_3d = zfp_field_3d(NULL, zfp_type_double, zfp_num_per_dim,
                                    zfp_num_per_dim, zfp_num_per_dim);

            zfp_stream_set_rate(zfp2d, rate, zfp_type_double, 2, 0);
            field_2d = zfp_field_2d(NULL, zfp_type_double, zfp_num_per_dim,
                                    zfp_num_per_dim);

            zfp_stream_set_rate(zfp1d, rate, zfp_type_double, 1, 0);
            field_1d = zfp_field_1d(NULL, zfp_type_double, zfp_num_per_dim);
        } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
            fieldType = zfp_type_float;

            zfp_stream_set_rate(zfp3d, rate, zfp_type_float, 3, 0);
            field_3d = zfp_field_3d(NULL, zfp_type_float, zfp_num_per_dim,
                                    zfp_num_per_dim, zfp_num_per_dim);

            zfp_stream_set_rate(zfp2d, rate, zfp_type_float, 2, 0);
            field_2d = zfp_field_2d(NULL, zfp_type_float, zfp_num_per_dim,
                                    zfp_num_per_dim);

            zfp_stream_set_rate(zfp1d, rate, zfp_type_float, 1, 0);
            field_1d = zfp_field_1d(NULL, zfp_type_float, zfp_num_per_dim);
        } else {
            throw std::invalid_argument(
                "Invalid input type for sendtype in ZFPCompression");
        }

        mode_set = "rate";
    }

    ~ZFPCompression() { close_and_free_all(); }

    void setEleOrder(const size_t &eleOrder_in) {
        close_and_free_all();

        // then set the new values
        eleOrder        = eleOrder_in;
        zfp_num_per_dim = eleOrder - 1;
        zfp_dim0_decomp = 1;
        zfp_dim1_decomp = zfp_num_per_dim;
        zfp_dim2_decomp = zfp_dim1_decomp * zfp_num_per_dim;
        zfp_dim3_decomp = zfp_dim2_decomp * zfp_num_per_dim;

        // finally open the new streams
        zfp3d           = zfp_stream_open(NULL);
        zfp2d           = zfp_stream_open(NULL);
        zfp1d           = zfp_stream_open(NULL);

        // then the fields
        if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
            fieldType = zfp_type_double;
            field_3d  = zfp_field_3d(NULL, zfp_type_double, zfp_num_per_dim,
                                     zfp_num_per_dim, zfp_num_per_dim);
            field_2d  = zfp_field_2d(NULL, zfp_type_double, zfp_num_per_dim,
                                     zfp_num_per_dim);
            field_1d  = zfp_field_1d(NULL, zfp_type_double, zfp_num_per_dim);
        } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
            fieldType = zfp_type_float;
            field_3d  = zfp_field_3d(NULL, zfp_type_float, zfp_num_per_dim,
                                     zfp_num_per_dim, zfp_num_per_dim);
            field_2d  = zfp_field_2d(NULL, zfp_type_float, zfp_num_per_dim,
                                     zfp_num_per_dim);
            field_1d  = zfp_field_1d(NULL, zfp_type_float, zfp_num_per_dim);
        } else {
            throw std::invalid_argument(
                "Invalid input type for sendtype in ZFPCompression");
        }

        // std::cout << "ZFP Element Order set to: " << eleOrder <<
        // std::endl;

        // NOTE: setting rate and accuracy should always be called after
        // this function
    }

    void setUpForMultiVariable(const size_t &eleOrder_in,
                               const size_t &numVars_in) {
        close_and_free_all();

        // set the new values
        eleOrder        = eleOrder_in;
        numVars         = numVars_in;
        useMultiVars    = true;

        zfp_num_per_dim = eleOrder - 1;
        zfp_dim1_decomp = numVars;
        zfp_dim2_decomp = zfp_num_per_dim * numVars;
        zfp_dim3_decomp = zfp_num_per_dim * zfp_num_per_dim * numVars;
        zfp_dim4_decomp =
            zfp_num_per_dim * zfp_num_per_dim * zfp_num_per_dim * numVars;

        // finally open the new streams
        zfp4d = zfp_stream_open(NULL);
        zfp3d = zfp_stream_open(NULL);
        zfp2d = zfp_stream_open(NULL);
        zfp1d = zfp_stream_open(NULL);

        // then the fields

        if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
            fieldType = zfp_type_double;
            field_4d =
                zfp_field_4d(NULL, zfp_type_double, numVars, zfp_num_per_dim,
                             zfp_num_per_dim, zfp_num_per_dim);
            field_3d = zfp_field_3d(NULL, zfp_type_double, numVars,
                                    zfp_num_per_dim, zfp_num_per_dim);
            field_2d =
                zfp_field_2d(NULL, zfp_type_double, numVars, zfp_num_per_dim);
            field_1d = zfp_field_1d(NULL, zfp_type_double, numVars);
        } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
            fieldType = zfp_type_float;
            field_4d =
                zfp_field_4d(NULL, zfp_type_float, numVars, zfp_num_per_dim,
                             zfp_num_per_dim, zfp_num_per_dim);
            field_3d = zfp_field_3d(NULL, zfp_type_float, numVars,
                                    zfp_num_per_dim, zfp_num_per_dim);
            field_2d =
                zfp_field_2d(NULL, zfp_type_float, numVars, zfp_num_per_dim);
            field_1d = zfp_field_1d(NULL, zfp_type_float, numVars);
        } else {
            throw std::invalid_argument(
                "Invalid input type for sendtype in ZFPCompression");
        }
    }

    void setRate(const double rate_in) {
        rate = rate_in;

        if (zfp3d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
            fieldType = zfp_type_double;
            zfp_stream_set_rate(zfp3d, rate, zfp_type_double, 3, 0);
            zfp_stream_set_rate(zfp2d, rate, zfp_type_double, 2, 0);
            zfp_stream_set_rate(zfp1d, rate, zfp_type_double, 1, 0);
        } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
            fieldType = zfp_type_float;
            zfp_stream_set_rate(zfp3d, rate, zfp_type_float, 3, 0);
            zfp_stream_set_rate(zfp2d, rate, zfp_type_float, 2, 0);
            zfp_stream_set_rate(zfp1d, rate, zfp_type_float, 1, 0);
        } else {
            throw std::invalid_argument(
                "Invalid input type for sendtype in ZFPCompression");
        }

        assert(zfp_stream_compression_mode(zfp3d) == zfp_mode_fixed_rate);
        assert(zfp_stream_compression_mode(zfp2d) == zfp_mode_fixed_rate);
        assert(zfp_stream_compression_mode(zfp1d) == zfp_mode_fixed_rate);

        mode_set = "rate";
        // std::cout << "ZFP Rate set to: " << rate << std::endl;

        // and then set 4d field, only initialized if we set up the whole
        // thing ready to go
        if (useMultiVars) {
            if (zfp4d == nullptr)
                throw std::invalid_argument(
                    "ZFP Wasn't properly initialized for some reason!");

            if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
                zfp_stream_set_rate(zfp4d, rate, zfp_type_double, 4, 0);
            } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
                zfp_stream_set_rate(zfp4d, rate, zfp_type_float, 4, 0);
            }
            assert(zfp_stream_compression_mode(zfp4d) == zfp_mode_fixed_rate);
        }
    }

    void setPrecision(const unsigned int precision_in) {
        precision = precision_in;

        if (zfp3d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        zfp_stream_set_precision(zfp3d, precision);
        zfp_stream_set_precision(zfp2d, precision);
        zfp_stream_set_precision(zfp1d, precision);

        if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
            fieldType = zfp_type_double;
        } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
            fieldType = zfp_type_float;
        } else {
            throw std::invalid_argument(
                "Invalid input type for sendtype in ZFPCompression");
        }

        assert(zfp_stream_compression_mode(zfp3d) == zfp_mode_fixed_precision);
        assert(zfp_stream_compression_mode(zfp2d) == zfp_mode_fixed_precision);
        assert(zfp_stream_compression_mode(zfp1d) == zfp_mode_fixed_precision);

        mode_set = "precision";
        // std::cout << "ZFP Rate set to: " << rate << std::endl;

        // and then set 4d field, only initialized if we set up the whole
        // thing ready to go
        if (useMultiVars) {
            if (zfp4d == nullptr)
                throw std::invalid_argument(
                    "ZFP Wasn't properly initialized for some reason!");
            zfp_stream_set_precision(zfp4d, precision);

            assert(zfp_stream_compression_mode(zfp4d) ==
                   zfp_mode_fixed_precision);
        }
    }

    void setAccuracy(const double tolerance_in) {
        tolerance = tolerance_in;

        if (zfp3d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        zfp_stream_set_accuracy(zfp3d, tolerance);
        zfp_stream_set_accuracy(zfp2d, tolerance);
        zfp_stream_set_accuracy(zfp1d, tolerance);

        if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
            fieldType = zfp_type_double;
        } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
            fieldType = zfp_type_float;
        } else {
            throw std::invalid_argument(
                "Invalid input type for sendtype in ZFPCompression");
        }

        assert(zfp_stream_compression_mode(zfp3d) == zfp_mode_fixed_accuracy);
        assert(zfp_stream_compression_mode(zfp2d) == zfp_mode_fixed_accuracy);
        assert(zfp_stream_compression_mode(zfp1d) == zfp_mode_fixed_accuracy);

        mode_set = "accuracy";
        // std::cout << "ZFP Tolerance set to: " << rate << std::endl;

        // and then set 4d field, only initialized if we set up the whole
        // thing ready to go
        if (useMultiVars) {
            if (zfp4d == nullptr)
                throw std::invalid_argument(
                    "ZFP Wasn't properly initialized for some reason!");

            zfp_stream_set_accuracy(zfp4d, 1e-1);
            assert(zfp_stream_compression_mode(zfp4d) ==
                   zfp_mode_fixed_accuracy);
        }
    }

    void close_and_free_all() {
        close_all_streams();
        free_all_fields();
    }

    void close_all_streams() {
        if (zfp4d != nullptr) zfp_stream_close(zfp4d);

        if (zfp3d != nullptr) zfp_stream_close(zfp3d);

        if (zfp2d != nullptr) zfp_stream_close(zfp2d);

        if (zfp1d != nullptr) zfp_stream_close(zfp1d);

        zfp4d = nullptr;
        zfp3d = nullptr;
        zfp2d = nullptr;
        zfp1d = nullptr;
    }

    void free_all_fields() {
        if (field_4d != nullptr) zfp_field_free(field_4d);

        if (field_3d != nullptr) zfp_field_free(field_3d);

        if (field_2d != nullptr) zfp_field_free(field_2d);

        if (field_1d != nullptr) zfp_field_free(field_1d);

        field_4d = nullptr;
        field_3d = nullptr;
        field_2d = nullptr;
        field_1d = nullptr;
    }

    template <typename T>
    size_t do_4d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_4d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_3d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_3d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_2d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_2d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_1d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_1d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_0d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_0d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    /**
     * @brief: Set the CTX Send Type to prepare the inputs
     *
     * NOTE: You *must* run the set_rate or set_accuracy methods *after* calling
     * this one, the ZFP algorithms will not be prepared properly by calling
     * this function!
     */
    void setCtxSendType(ot::CTXSendType sendtype_in) {
        m_ctxSendType = sendtype_in;
    }

    inline ot::CTXSendType getCtxSendType() const { return m_ctxSendType; }

#ifdef _DEBUG_ZFP_
    void print_rmse_and_reset() {
        std::cout << "4D RMSE: " << rmse.get_rmse() << std::endl;
        rmse.reset();
    }
#endif

   private:
    zfp_stream *zfp4d             = nullptr;
    zfp_stream *zfp3d             = nullptr;
    zfp_stream *zfp2d             = nullptr;
    zfp_stream *zfp1d             = nullptr;
    std::string mode_set          = "none";
    int zfp_dim0_decomp           = 0;
    int zfp_dim1_decomp           = 0;
    int zfp_dim2_decomp           = 0;
    int zfp_dim3_decomp           = 0;
    int zfp_dim4_decomp           = 0;
    size_t eleOrder               = 0;
    size_t numVars                = 0;
    double rate                   = 20.0;
    double tolerance              = 1e-2f;
    unsigned int precision        = 24;
    size_t zfp_num_per_dim        = 0;
    zfp_field *field_4d           = nullptr;
    zfp_field *field_3d           = nullptr;
    zfp_field *field_2d           = nullptr;
    zfp_field *field_1d           = nullptr;
    bool useMultiVars             = false;

    zfp_type fieldType            = zfp_type_double;

    ot::CTXSendType m_ctxSendType = ot::CTXSendType::CTX_DOUBLE;

#ifdef _DEBUG_ZFP_
    RunningRMSE rmse;
#endif
};

// A ZFPCompression object to use "globally"
extern ZFPCompression zfpblockwise;

}  // namespace ZFPAlgorithms

#include <blosc.h>

#include <iostream>

namespace BLOSCAlgorithms {

/**
 * Compresses a block of data using the Blosc library.
 * This function compresses a given array of doubles using the specified
 * Blosc compressor and compression level.
 *
 * Before calling this function, ensure blosc_init() has been called to
 * initialize the Blosc library. After using the compressed data,
 * blosc_destroy() should be called for proper cleanup.
 *
 * @param blosc_compressor The compression algorithm to be used. Must be one
 * of "blosclz", "lz4", "lz4hc", "zlib", or "zstd".
 * @param clevel Choose the compression level (1-9, where 9 is highest
 * compression)
 * @param n The number of elements in the original data array.
 * @param originalData Pointer to the original array of doubles to be
 * compressed.
 * @param byteStreamSize Reference to an integer where the size of the
 * resulting bytestream will be stored.
 * @return Pointer to the compressed data bytestream.
 * @throw std::runtime_error if compression fails.
 */
unsigned char *compressData(const char *blosc_compressor, int clevel, int n,
                            double *originalData, int &byteStreamSize);

/**
 * Decompresses a bytestream using the Blosc library.
 * It decompresses a bytestream that was created by the compressData
 * function.
 *
 * Before calling this function, ensure blosc_init() has been called to
 * initialize the Blosc library. After using the compressed data,
 * blosc_destroy() should be called for proper cleanup.
 *
 * @param byteStream Pointer to the bytestream containing the compressed
 * data.
 * @param byteStreamSize The size of the bytestream.
 * @return Pointer to the decompressed data array.
 * @throw std::runtime_error if decompression fails or if input is invalid.
 */
double *decompressData(unsigned char *byteStream, int byteStreamSize);

/**
 * Decompresses a bytestream using the Blosc library.
 * It decompresses a bytestream that was created by the compressData
 * function.
 *
 * Before calling this function, ensure blosc_init() has been called to
 * initialize the Blosc library. After using the compressed data,
 * blosc_destroy() should be called for proper cleanup.
 *
 * @param byteStream Pointer to the bytestream containing the compressed
 * data.
 * @param byteStreamSize The size of the bytestream.
 * @param outBuff The output buffer.
 * @throw std::runtime_error if decompression fails or if input is invalid.
 */
void decompressData(unsigned char *byteStream, int byteStreamSize,
                    double *outBuff);

class BloscCompression {
   public:
    BloscCompression(const size_t &eleOrder             = 6,
                     const std::string &bloscCompressor = "lz4",
                     const int &clevel = 4, const int &doShuffle = 1)
        : eleOrder(eleOrder),
          bloscCompressor(bloscCompressor),
          clevel(clevel),
          doShuffle(doShuffle) {
        // TODO: init shouldn't be called here, it should be in some other
        // initialization
        blosc_init();
        blosc_set_compressor(bloscCompressor.c_str());
        int max_threads = omp_get_max_threads();
        std::cout << "MAXIMUM NUMBER OF THREADS AVAILABLE TO BLOSC: "
                  << max_threads << std::endl;
        if (max_threads > 1) blosc_set_nthreads(max_threads);

        calculateSizes();
    }

    ~BloscCompression() {
        // TODO: destroy shouldn't be called here, it should be in some
        // other destruction
        blosc_destroy();
    }

    void setEleOrder(size_t eleOrder_in) {
        eleOrder = eleOrder_in;

        calculateSizes();
    }

    void setCompressor(const std::string &bloscCompressor_in) {
        bloscCompressor = bloscCompressor_in;

        blosc_set_compressor(bloscCompressor.c_str());
    }

    void calculateSizes() {
        size_t points_1d        = eleOrder - 1;

        // calculate the number of bytes based on the element order

        size_t bytes_dtype      = ot::getCTXSendTypeSize(m_ctxSendType);

        blosc_original_bytes_1d = bytes_dtype * points_1d;
        blosc_original_bytes_2d = points_1d * blosc_original_bytes_1d;
        blosc_original_bytes_3d = points_1d * blosc_original_bytes_2d;

        // then with the overhead for the maximum possible amount it could
        // take. This guarantees success, but will basically never take this
        // much.
        blosc_original_bytes_overhead_1d =
            blosc_original_bytes_1d + BLOSC_MAX_OVERHEAD;
        blosc_original_bytes_overhead_2d =
            blosc_original_bytes_2d + BLOSC_MAX_OVERHEAD;
        blosc_original_bytes_overhead_3d =
            blosc_original_bytes_3d + BLOSC_MAX_OVERHEAD;
    }

    void setUpForMultiVariable(const size_t &eleOrder_in,
                               const size_t &numVars_in) {
        eleOrder                = eleOrder_in;
        numVars                 = numVars_in;
        useMultiVars            = true;

        size_t points_1d        = eleOrder - 1;

        size_t bytes_dtype      = ot::getCTXSendTypeSize(m_ctxSendType);

        blosc_original_bytes_1d = bytes_dtype * numVars;
        blosc_original_bytes_2d = bytes_dtype * numVars * points_1d;
        blosc_original_bytes_3d = bytes_dtype * numVars * points_1d * points_1d;
        blosc_original_bytes_4d =
            bytes_dtype * numVars * points_1d * points_1d * points_1d;

        // then with the overhead for the maximum possible amount it could
        // take. This guarantees success, but will basically never take this
        // much.
        blosc_original_bytes_overhead_1d =
            blosc_original_bytes_1d + BLOSC_MAX_OVERHEAD;
        blosc_original_bytes_overhead_2d =
            blosc_original_bytes_2d + BLOSC_MAX_OVERHEAD;
        blosc_original_bytes_overhead_3d =
            blosc_original_bytes_3d + BLOSC_MAX_OVERHEAD;
        blosc_original_bytes_overhead_4d =
            blosc_original_bytes_4d + BLOSC_MAX_OVERHEAD;
    }

    template <typename T>
    size_t do_4d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_4d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_3d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_3d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_2d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_2d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_1d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_1d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    template <typename T>
    size_t do_0d_compression(T *originalMatrix, unsigned char *outputArray,
                             size_t batchSize = 1);
    template <typename T>
    size_t do_0d_decompression(unsigned char *compressedBuffer, T *outputArray,
                               size_t batchSize = 1);

    void setCtxSendType(ot::CTXSendType sendtype_in) {
        m_ctxSendType = sendtype_in;
    }

    inline ot::CTXSendType getCtxSendType() const { return m_ctxSendType; }

   private:
    size_t eleOrder;
    size_t numVars    = 1;
    bool useMultiVars = false;
    // blosc settings
    std::string bloscCompressor;
    int clevel;
    int doShuffle;

    // tracking original sizes
    size_t blosc_original_bytes_4d;
    size_t blosc_original_bytes_3d;
    size_t blosc_original_bytes_2d;
    size_t blosc_original_bytes_1d;

    size_t blosc_original_bytes_overhead_4d;
    size_t blosc_original_bytes_overhead_3d;
    size_t blosc_original_bytes_overhead_2d;
    size_t blosc_original_bytes_overhead_1d;

    ot::CTXSendType m_ctxSendType = ot::CTXSendType::CTX_DOUBLE;

    // overhead bytes
};

extern BloscCompression bloscblockwise;

}  // namespace BLOSCAlgorithms

namespace dendro_compress {

class PsuedoNoneCompression {
   public:
    PsuedoNoneCompression(const size_t &eleOrder = 6, const size_t &numVars = 1)
        : m_eleOrder(eleOrder), m_numVars(numVars) {
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts = pointsPerDim * pointsPerDim * pointsPerDim * m_numVars;
        m_total2DPts = pointsPerDim * pointsPerDim * m_numVars;
        m_total1DPts = pointsPerDim * m_numVars;
        m_total0DPts = m_numVars;
    }

    void set_sizes(const size_t &eleOrder = 6, const size_t &numVars = 1) {
        m_eleOrder                = eleOrder;
        m_numVars                 = numVars;
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts = pointsPerDim * pointsPerDim * pointsPerDim * m_numVars;
        m_total2DPts = pointsPerDim * pointsPerDim * m_numVars;
        m_total1DPts = pointsPerDim * m_numVars;
        m_total0DPts = m_numVars;
    }

    template <typename T>
    size_t do_3d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, originalMatrix,
                    batchSize * m_total3DPts * sizeof(T));
        return batchSize * m_total3DPts * sizeof(T);
    }

    template <typename T>
    size_t do_3d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, compressedBuffer,
                    batchSize * m_total3DPts * sizeof(T));
        return batchSize * m_total3DPts * sizeof(T);
    }

    template <typename T>
    size_t do_2d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, originalMatrix,
                    batchSize * m_total2DPts * sizeof(T));
        return batchSize * m_total2DPts * sizeof(T);
    }

    template <typename T>
    size_t do_2d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, compressedBuffer,
                    batchSize * m_total2DPts * sizeof(T));
        return batchSize * m_total2DPts * sizeof(T);
    }

    template <typename T>
    size_t do_1d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, originalMatrix,
                    batchSize * m_total1DPts * sizeof(T));
        return batchSize * m_total1DPts * sizeof(T);
    }

    template <typename T>
    size_t do_1d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, compressedBuffer,
                    batchSize * m_total1DPts * sizeof(T));
        return batchSize * m_total1DPts * sizeof(T);
    }

    template <typename T>
    size_t do_0d_compression(const T *originalMatrix,
                             unsigned char *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, originalMatrix,
                    batchSize * m_total0DPts * sizeof(T));
        return batchSize * m_total0DPts * sizeof(T);
    }

    template <typename T>
    size_t do_0d_decompression(const unsigned char *compressedBuffer,
                               T *outputArray, size_t batchSize = 1) {
        std::memcpy(outputArray, compressedBuffer,
                    batchSize * m_total0DPts * sizeof(T));
        return batchSize * m_total0DPts * sizeof(T);
    }

   private:
    unsigned int m_eleOrder;
    unsigned int m_pointsPerDim;
    unsigned int m_numVars;
    unsigned int m_total3DPts;
    unsigned int m_total2DPts;
    unsigned int m_total1DPts;
    unsigned int m_total0DPts;
};

// TODO: bilateral filtering, non-local means (though it's really expensive),
// Savitzky-Golay (polynomial)

class GaussianFiltering {
   public:
    GaussianFiltering(
        const unsigned int &eleOrder = 6, const double &sigma = 1.0,
        const unsigned int &radius     = 2,
        const ot::CTXSendType sendtype = ot::CTXSendType::CTX_DOUBLE)
        : m_eleOrder(eleOrder),
          m_sigma(sigma),
          m_kernelRadius(radius),
          m_ctxSendType(sendtype) {
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts              = pointsPerDim * pointsPerDim * pointsPerDim;
        m_total2DPts              = pointsPerDim * pointsPerDim;
        m_total1DPts              = pointsPerDim;
        m_total0DPts              = 1;

        // load_chebyshev_matrices();
    }

    void set_ctx_send_type(const ot::CTXSendType sendtype) {
        m_ctxSendType = sendtype;
    }

    void set_sizes(const unsigned int &eleOrder = 6) {
        m_eleOrder                = eleOrder;
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts              = pointsPerDim * pointsPerDim * pointsPerDim;
        m_total2DPts              = pointsPerDim * pointsPerDim;
        m_total1DPts              = pointsPerDim;
        m_total0DPts              = 1;

        create_filter_kernel();
    }

    void set_sigma(const double sigma) { m_sigma = sigma; }
    void set_radius(const unsigned int radius) { m_kernelRadius = radius; }

    void create_filter_kernel() {
        // m_kernelRadius = static_cast<int>(m_sigma * 3.0);

        if (m_kernelRadius > m_total1DPts) {
            std::cerr << "ERROR: Gaussian filtering requires fewer than " +
                             std::to_string(m_total1DPts) +
                             " based on input sizes! Choose a smaller sigma!"
                      << std::endl;
            exit(EXIT_FAILURE);
        }

        // if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
        {
            double sum           = 0.0f;

            m_dKernel            = std::vector<double>(2 * m_kernelRadius + 1);

            double itwoSigmaSqrd = 1.0 / (2.0 * m_sigma * m_sigma);

            for (int i = -m_kernelRadius; i <= m_kernelRadius; ++i) {
                m_dKernel[i + m_kernelRadius] =
                    exp(-(double)(i * i) * itwoSigmaSqrd);
                sum += m_dKernel[i + m_kernelRadius];
            }

            // then we have to normalize the kernel
            for (auto &val : m_dKernel) {
                val /= sum;
            }

            // prepare the temporary vectors after building the kernel
            m_dTemp  = std::vector<double>(m_total3DPts);
            m_dTemp2 = std::vector<double>(m_total3DPts);
        }
        // } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
        {
            float sum          = 0.0f;

            m_fKernel          = std::vector<float>(2 * m_kernelRadius + 1);

            float twoSigmaSqrd = 1.0 / (2 * m_sigma * m_sigma);

            for (int i = -m_kernelRadius; i <= m_kernelRadius; ++i) {
                m_fKernel[i + m_kernelRadius] =
                    exp(-((float)i * (float)i) / twoSigmaSqrd);
                sum += m_fKernel[i + m_kernelRadius];
            }

            // then we have to normalize the kernel
            for (auto &val : m_fKernel) {
                val /= sum;
            }

            // prepare the temporary vectors after building the kernel
            m_fTemp  = std::vector<float>(m_total3DPts);
            m_fTemp2 = std::vector<float>(m_total3DPts);
        }

        // exit(0);

        // now we have the data
    }

    template <typename T>
    void do_3d_filtering(T *originalMatrix, size_t batchSize = 1) {
        T *matrixLoc = originalMatrix;

        T sum;
        // fix warping at edges by renormalizing the filter here
        T weightSum;

        std::vector<T> &kernel = getKernel<T>();
        std::vector<T> &temp   = getTempVar<T>();
        std::vector<T> &temp2  = getTemp2Var<T>();

        for (unsigned int batch = 0; batch < batchSize; batch++) {
            // apply the filter across the x axis
            for (int z = 0; z < m_total1DPts; ++z) {
                for (int y = 0; y < m_total1DPts; ++y) {
                    for (int x = 0; x < m_total1DPts; ++x) {
                        sum       = 0.0;
                        weightSum = 0.0;
                        for (int i = -m_kernelRadius; i <= m_kernelRadius;
                             ++i) {
                            int idx = x + i;
                            if (idx >= 0 && idx < m_total1DPts) {
                                sum += matrixLoc[index(idx, y, z)] *
                                       kernel[i + m_kernelRadius];
                                weightSum += kernel[i + m_kernelRadius];
                            }
                        }
                        temp[index(x, y, z)] = sum / weightSum;
                    }
                }
            }

            // then we apply to the yaxis
            for (int z = 0; z < m_total1DPts; ++z) {
                for (int x = 0; x < m_total1DPts; ++x) {
                    for (int y = 0; y < m_total1DPts; ++y) {
                        sum       = 0.0;
                        weightSum = 0.0;
                        for (int i = -m_kernelRadius; i <= m_kernelRadius;
                             ++i) {
                            int idx = y + i;
                            if (idx >= 0 && idx < m_total1DPts) {
                                sum += temp[index(x, idx, z)] *
                                       kernel[i + m_kernelRadius];
                                weightSum += kernel[i + m_kernelRadius];
                            }
                        }
                        temp2[index(x, y, z)] = sum / weightSum;
                    }
                }
            }

            // then we apply to the zaxis
            for (int y = 0; y < m_total1DPts; ++y) {
                for (int x = 0; x < m_total1DPts; ++x) {
                    for (int z = 0; z < m_total1DPts; ++z) {
                        sum       = 0.0;
                        weightSum = 0.0;
                        for (int i = -m_kernelRadius; i <= m_kernelRadius;
                             ++i) {
                            int idx = z + i;
                            if (idx >= 0 && idx < m_total1DPts) {
                                sum += temp2[index(x, y, idx)] *
                                       kernel[i + m_kernelRadius];
                                weightSum += kernel[i + m_kernelRadius];
                            }
                        }
                        matrixLoc[index(x, y, z)] = sum / weightSum;
                    }
                }
            }

            // advance forward
            matrixLoc += m_total3DPts;
        }
        // done
    }

    template <typename T>
    void do_2d_filtering(T *originalMatrix, size_t batchSize = 1) {
        T *matrixLoc = originalMatrix;

        T sum;
        T weightSum;

        std::vector<T> &kernel = getKernel<T>();
        std::vector<T> &temp   = getTempVar<T>();

        for (unsigned int batch = 0; batch < batchSize; batch++) {
            // apply the filter across the x axis
            for (int y = 0; y < m_total1DPts; ++y) {
                for (int x = 0; x < m_total1DPts; ++x) {
                    sum       = 0.0;
                    weightSum = 0.0;
                    for (int i = -m_kernelRadius; i <= m_kernelRadius; ++i) {
                        int idx = x + i;
                        if (idx >= 0 && idx < m_total1DPts) {
                            sum += matrixLoc[index2d(idx, y)] *
                                   kernel[i + m_kernelRadius];
                            weightSum += kernel[i + m_kernelRadius];
                        }
                    }
                    temp[index2d(x, y)] = sum / weightSum;
                }
            }

            // then we apply to the yaxis
            for (int x = 0; x < m_total1DPts; ++x) {
                for (int y = 0; y < m_total1DPts; ++y) {
                    sum       = 0.0;
                    weightSum = 0.0;
                    for (int i = -m_kernelRadius; i <= m_kernelRadius; ++i) {
                        int idx = y + i;
                        if (idx >= 0 && idx < m_total1DPts) {
                            sum += temp[index2d(x, idx)] *
                                   kernel[i + m_kernelRadius];
                            weightSum += kernel[i + m_kernelRadius];
                        }
                    }
                    matrixLoc[index2d(x, y)] = sum / weightSum;
                }
            }

            // advance forward
            matrixLoc += m_total2DPts;
        }
        // then we're done!
    }

    template <typename T>
    void do_1d_filtering(T *originalMatrix, size_t batchSize = 1) {
        T *matrixLoc = originalMatrix;

        T sum;
        T weightSum;

        std::vector<T> &kernel = getKernel<T>();

        for (unsigned int batch = 0; batch < batchSize; batch++) {
            // apply the filter across the x axis
            for (int x = 0; x < m_total1DPts; ++x) {
                sum       = 0.0;
                weightSum = 0.0;
                for (int i = -m_kernelRadius; i <= m_kernelRadius; ++i) {
                    int idx = x + i;
                    if (idx >= 0 && idx < m_total1DPts) {
                        sum += matrixLoc[idx] * kernel[i + m_kernelRadius];
                        weightSum += kernel[i + m_kernelRadius];
                    }
                }
                matrixLoc[x] = sum / weightSum;
            }

            // advance forward
            matrixLoc += m_total1DPts;
        }
        // then we're done!
    }

    template <typename T>
    constexpr std::vector<T> &getKernel() {
        if constexpr (std::is_same_v<T, double>) {
            return m_dKernel;
        } else {
            return m_fKernel;
        }
    }

    template <typename T>
    constexpr std::vector<T> &getTempVar() {
        if constexpr (std::is_same_v<T, double>) {
            return m_dTemp;
        } else {
            return m_fTemp;
        }
    }

    template <typename T>
    constexpr std::vector<T> &getTemp2Var() {
        if constexpr (std::is_same_v<T, double>) {
            return m_dTemp2;
        } else {
            return m_fTemp2;
        }
    }

   private:
    __attribute__((always_inline)) inline int index(int i, int j, int k) const {
        return (k)*m_total1DPts * m_total1DPts + (j)*m_total1DPts + i;
    }

    __attribute__((always_inline)) inline int index2d(int i, int j) const {
        return (j)*m_total1DPts + i;
    }

    ot::CTXSendType m_ctxSendType;

    unsigned int m_eleOrder;
    unsigned int m_pointsPerDim;
    unsigned int m_total3DPts;
    unsigned int m_total2DPts;
    unsigned int m_total1DPts;
    unsigned int m_total0DPts;

    double m_sigma;

    int m_kernelRadius;

    std::vector<double> m_dKernel;
    std::vector<float> m_fKernel;

    std::vector<double> m_dTemp;
    std::vector<float> m_fTemp;
    std::vector<double> m_dTemp2;
    std::vector<float> m_fTemp2;
};

template <typename T>
std::vector<T> load_chebyshev_matrix(std::string &path, unsigned int eleOrder,
                                     unsigned int ndim, unsigned int nPoly) {
    std::string path_full;

    if constexpr (std::is_same_v<T, double>) {
        path_full = path + "/cheb_ele" + std::to_string(eleOrder) + "_r" +
                    std::to_string(nPoly) + "_d_" + std::to_string(ndim) +
                    "d.bin";
    } else {
        path_full = path + "/cheb_ele" + std::to_string(eleOrder) + "_r" +
                    std::to_string(nPoly) + "_f_" + std::to_string(ndim) +
                    "d.bin";
    }

    std::size_t numEles     = pow(eleOrder - 1, ndim);
    std::size_t numPolyEles = pow(nPoly, ndim);

    std::ifstream file(path_full, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(
            "ERROR: Could not open chebyshev matrix file " + path_full);
    }

    // check file size
    size_t expected_size = numEles * numPolyEles * sizeof(T);
    if (std::filesystem::file_size(path_full) != expected_size) {
        throw std::runtime_error(
            "ERROR: Chebyshev Matrix File did not have the right size! "
            "Expected " +
            std::to_string(expected_size) + " bytes. Found " +
            std::to_string(std::filesystem::file_size(path_full)) + " bytes.");
    }

    std::vector<T> matrix(numEles * numPolyEles);
    file.read(reinterpret_cast<char *>(matrix.data()),
              matrix.size() * sizeof(T));
    return matrix;
}

class ChebyshevFiltering {
   public:
    ChebyshevFiltering(
        const unsigned int &eleOrder       = 6,
        const unsigned int &nPolyReduction = 4,
        const ot::CTXSendType sendtype     = ot::CTXSendType::CTX_DOUBLE)
        : m_eleOrder(eleOrder),
          m_nPolyReduction(nPolyReduction),
          m_ctxSendType(sendtype) {
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts              = pointsPerDim * pointsPerDim * pointsPerDim;
        m_total2DPts              = pointsPerDim * pointsPerDim;
        m_total1DPts              = pointsPerDim;
        m_total0DPts              = 1;

        // load_chebyshev_matrices();
    }

    void set_ctx_send_type(const ot::CTXSendType sendtype) {
        m_ctxSendType = sendtype;
    }

    void set_sizes(const unsigned int &eleOrder       = 6,
                   const unsigned int &nPolyReduction = 2) {
        m_eleOrder                = eleOrder;
        m_nPolyReduction          = nPolyReduction;
        unsigned int pointsPerDim = (m_eleOrder - 1);
        m_pointsPerDim            = pointsPerDim;

        m_total3DPts              = pointsPerDim * pointsPerDim * pointsPerDim;
        m_total2DPts              = pointsPerDim * pointsPerDim;
        m_total1DPts              = pointsPerDim;
        m_total0DPts              = 1;

        load_chebyshev_matrices();
    }

    void load_chebyshev_matrices() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // call after setting up!
        if (m_ctxSendType == ot::CTXSendType::CTX_DOUBLE) {
            std::vector<double> A1d = load_chebyshev_matrix<double>(
                m_basePath, m_eleOrder, 1, m_nPolyReduction);

            m_dA_chebfilt_dim1 =
                std::vector<double>(m_total1DPts * m_total1DPts, 0.0);

            // matrix size is nPolyReduction x total1DPts
            int M        = m_nPolyReduction;
            int K        = m_total1DPts;

            double alpha = 1.0, beta = 0.0;
            // need to compute A^T x A
            char transa = 'T', transb = 'N';

            // remember LDA is equal to M, and LDB is also equal to M, because
            // we're "squaring" the matrix
            dgemm_(&transa, &transb, &K, &K, &M, &alpha, A1d.data(), &M,
                   A1d.data(), &M, &beta, m_dA_chebfilt_dim1.data(), &K);

            // --------------------------------------
            // LOAD 2D
            std::vector<double> A2d = load_chebyshev_matrix<double>(
                m_basePath, m_eleOrder, 2, m_nPolyReduction);

            m_dA_chebfilt_dim2 =
                std::vector<double>(m_total2DPts * m_total2DPts, 0.0);

            // matrix size is nPolyReduction^2 x total1DPts
            K = m_total2DPts;
            // M updated to be  the nPolyReduction squared
            M = m_nPolyReduction * m_nPolyReduction;

            // dgemm_ is the same
            dgemm_(&transa, &transb, &K, &K, &M, &alpha, A2d.data(), &M,
                   A2d.data(), &M, &beta, m_dA_chebfilt_dim2.data(), &K);

            // --------------------------------------
            // LOAD 3D
            std::vector<double> A3d = load_chebyshev_matrix<double>(
                m_basePath, m_eleOrder, 3, m_nPolyReduction);

            m_dA_chebfilt_dim3 =
                std::vector<double>(m_total3DPts * m_total3DPts, 0.0);

            // matrix size is nPolyReduction^2 x total1DPts
            K = m_total3DPts;
            // M updated to be  the nPolyReduction squared
            M = m_nPolyReduction * m_nPolyReduction * m_nPolyReduction;

            // dgemm_ is the same
            dgemm_(&transa, &transb, &K, &K, &M, &alpha, A3d.data(), &M,
                   A3d.data(), &M, &beta, m_dA_chebfilt_dim3.data(), &K);

        } else if (m_ctxSendType == ot::CTXSendType::CTX_FLOAT) {
            std::vector<float> A1d = load_chebyshev_matrix<float>(
                m_basePath, m_eleOrder, 1, m_nPolyReduction);

            m_fA_chebfilt_dim1 =
                std::vector<float>(m_total1DPts * m_total1DPts, 0.0);

            // matrix size is nPolyReduction x total1DPts
            int M       = m_nPolyReduction;
            int K       = m_total1DPts;

            float alpha = 1.0, beta = 0.0;
            // need to compute A^T x A
            char transa = 'T', transb = 'N';

            // remember LDA is equal to M, and LDB is also equal to M, because
            // we're "squaring" the matrix
            sgemm_(&transa, &transb, &K, &K, &M, &alpha, A1d.data(), &M,
                   A1d.data(), &M, &beta, m_fA_chebfilt_dim1.data(), &K);

            // --------------------------------------
            // LOAD 2D
            std::vector<float> A2d = load_chebyshev_matrix<float>(
                m_basePath, m_eleOrder, 2, m_nPolyReduction);

            m_fA_chebfilt_dim2 =
                std::vector<float>(m_total2DPts * m_total2DPts, 0.0);

            // matrix size is nPolyReduction^2 x total1DPts
            K = m_total2DPts;
            // M updated to be  the nPolyReduction squared
            M = m_nPolyReduction * m_nPolyReduction;

            // dgemm_ is the same
            sgemm_(&transa, &transb, &K, &K, &M, &alpha, A2d.data(), &M,
                   A2d.data(), &M, &beta, m_fA_chebfilt_dim2.data(), &K);

            // check the matrix

            // if (rank == 0) {
            //     for (unsigned int i = 0; i < m_total2DPts; i++) {
            //         for (unsigned int j = 0; j < m_total2DPts; j++) {
            //             std::cout << m_fA_chebfilt_dim2[j + i * m_total2DPts]
            //                       << " ";
            //         }
            //         std::cout << std::endl;
            //     }
            // }

            // --------------------------------------
            // LOAD 3D
            std::vector<float> A3d = load_chebyshev_matrix<float>(
                m_basePath, m_eleOrder, 3, m_nPolyReduction);

            m_fA_chebfilt_dim3 =
                std::vector<float>(m_total3DPts * m_total3DPts, 0.0);

            // matrix size is nPolyReduction^2 x total1DPts
            K = m_total3DPts;
            // M updated to be  the nPolyReduction squared
            M = m_nPolyReduction * m_nPolyReduction * m_nPolyReduction;

            // dgemm_ is the same
            sgemm_(&transa, &transb, &K, &K, &M, &alpha, A3d.data(), &M,
                   A3d.data(), &M, &beta, m_fA_chebfilt_dim3.data(), &K);

            // check the matrix
        }
    }

    template <typename T>
    void do_array_norm(T *array, const size_t count, T &minVal, T &maxVal) {
        minVal  = *std::min_element(array, array + count);
        maxVal  = *std::max_element(array, array + count);
        T range = maxVal - minVal;

        if (minVal < -1.0 || maxVal > 1.0) {
            // if we're outside -1 and or 1, then we need to do full
            // normalization
            for (size_t i = 0; i < count; i++) {
                array[i] = 2.0 * ((array[i] - minVal) / range) - 1.0;
            }
        } else if (range > 1e-16) {
            // apply a shift if we're within this particular range to center us
            // around 0
            T shift = -(minVal + maxVal) / 2.0;
            for (size_t i = 0; i < count; i++) {
                array[i] += shift;
            }
        } else {
            // otherwise do nothing, we're close to zero or something
        }
    }
    template <typename T>
    void undo_array_norm(T *array, const size_t count, const T minVal,
                         const T maxVal) {
        T range = maxVal - minVal;

        if (minVal < -1.0 || maxVal > 1.0) {
            // if we're outside -1 and or 1, then we need to do full
            // denormalization
            for (size_t i = 0; i < count; i++) {
                array[i] = ((array[i] + 1.0) / 2.0) * range + minVal;
            }
        } else if (range > 1e-16) {
            // apply a shift if we're within the vals
            T shift = -(minVal + maxVal) / 2.0;
            for (size_t i = 0; i < count; i++) {
                array[i] -= shift;
            }
        } else {
            // otherwise do nothing, we didn't do anything above
        }
    }

    template <typename T>
    void do_3d_filtering(T *originalMatrix, size_t batchSize = 1) {
        T *matrixLoc = originalMatrix;
        T maxVal, minVal;

        std::vector<T> temp_buffer(m_total3DPts);

        // std::cout << "BATCH SIZE: " << batchSize << std::endl;

        int M = m_total3DPts, N = m_total3DPts, S = 1;
        T alpha = 1.0, beta = 0.0;

        // matrix is symmetric, N or T doesn't matter, just whichever is fastest
        char TRANS = 'N';

        // NOTE: might be faster to normalize entire buffer (as allotted by
        // batchSize)

        for (unsigned int batch = 0; batch < batchSize; batch++) {
            // copy the data over to temp_buffer
            std::copy(matrixLoc, matrixLoc + m_total3DPts, temp_buffer.data());

            do_array_norm(temp_buffer.data(), m_total3DPts, minVal, maxVal);

            // std::cout << "MIN VAL: " << minVal << " MAX VAL: " << maxVal
            //           << std::endl;
            //
            // std::cout << "VAL[0] BEFORE: " << temp_buffer[0] << std::endl;

            // then we just do gemv_ with the array

            if constexpr (std::is_same_v<T, double>) {
                dgemv_(&TRANS, &M, &N, &alpha, m_dA_chebfilt_dim3.data(), &M,
                       temp_buffer.data(), &S, &beta, matrixLoc, &S);
            } else if constexpr (std::is_same_v<T, float>) {
                sgemv_(&TRANS, &M, &N, &alpha, m_fA_chebfilt_dim3.data(), &M,
                       temp_buffer.data(), &S, &beta, matrixLoc, &S);
            }

            // std::cout << "VAL[0] AFTER: " << matrixLoc[0] << std::endl;

            // then denorm
            undo_array_norm(matrixLoc, m_total3DPts, minVal, maxVal);

            // move matrixLoc ahead by the number of points we just processed
            matrixLoc += m_total3DPts;
        }
    }

    template <typename T>
    void do_2d_filtering(T *originalMatrix, size_t batchSize = 1) {
        return;
        T *matrixLoc = originalMatrix;
        T maxVal, minVal;

        std::vector<T> temp_buffer(m_total2DPts);

        int M = m_total2DPts, N = m_total2DPts, S = 1;
        T alpha = 1.0, beta = 0.0;

        // matrix is symmetric, N or T doesn't matter, just whichever is fastest
        char TRANS = 'N';

        // NOTE: might be faster to normalize entire buffer (as allotted by
        // batchSize)

        for (unsigned int batch = 0; batch < batchSize; batch++) {
            // copy the data over to temp_buffer
            std::copy(matrixLoc, matrixLoc + m_total2DPts, temp_buffer.data());

            do_array_norm(temp_buffer.data(), m_total2DPts, minVal, maxVal);

            // then we just do gemv_ with the array

            if constexpr (std::is_same_v<T, double>) {
                dgemv_(&TRANS, &M, &N, &alpha, m_dA_chebfilt_dim2.data(), &M,
                       temp_buffer.data(), &S, &beta, matrixLoc, &S);
            } else if constexpr (std::is_same_v<T, float>) {
                sgemv_(&TRANS, &M, &N, &alpha, m_fA_chebfilt_dim2.data(), &M,
                       temp_buffer.data(), &S, &beta, matrixLoc, &S);
            }

            // then denorm
            undo_array_norm(matrixLoc, m_total2DPts, minVal, maxVal);

            // move matrixLoc ahead by the number of points we just processed
            matrixLoc += m_total2DPts;
        }
    }

    template <typename T>
    void do_1d_filtering(T *originalMatrix, size_t batchSize = 1) {
        return;
        T *matrixLoc = originalMatrix;
        T maxVal, minVal;

        std::vector<T> temp_buffer(m_total1DPts);

        int M = m_total1DPts, N = m_total1DPts, S = 1;
        T alpha = 1.0, beta = 0.0;

        // matrix is symmetric, N or T doesn't matter, just whichever is fastest
        char TRANS = 'N';

        // NOTE: might be faster to normalize entire buffer (as allotted by
        // batchSize)

        for (unsigned int batch = 0; batch < batchSize; batch++) {
            // copy the data over to temp_buffer
            std::copy(matrixLoc, matrixLoc + m_total1DPts, temp_buffer.data());

            do_array_norm(temp_buffer.data(), m_total1DPts, minVal, maxVal);

            // then we just do gemv_ with the array

            if constexpr (std::is_same_v<T, double>) {
                dgemv_(&TRANS, &M, &N, &alpha, m_dA_chebfilt_dim1.data(), &M,
                       temp_buffer.data(), &S, &beta, matrixLoc, &S);
            } else if constexpr (std::is_same_v<T, float>) {
                sgemv_(&TRANS, &M, &N, &alpha, m_fA_chebfilt_dim1.data(), &M,
                       temp_buffer.data(), &S, &beta, matrixLoc, &S);
            }

            // then denorm
            undo_array_norm(matrixLoc, m_total1DPts, minVal, maxVal);

            // move matrixLoc ahead by the number of points we just processed
            matrixLoc += m_total1DPts;
        }
    }

   private:
    ot::CTXSendType m_ctxSendType;

    std::string m_basePath = "../dendrolib";

    unsigned int m_eleOrder;
    unsigned int m_nPolyReduction;
    unsigned int m_pointsPerDim;
    unsigned int m_total3DPts;
    unsigned int m_total2DPts;
    unsigned int m_total1DPts;
    unsigned int m_total0DPts;

    std::vector<double> m_dA_chebfilt_dim1;
    std::vector<double> m_dA_chebfilt_dim2;
    std::vector<double> m_dA_chebfilt_dim3;

    std::vector<float> m_fA_chebfilt_dim1;
    std::vector<float> m_fA_chebfilt_dim2;
    std::vector<float> m_fA_chebfilt_dim3;
};

extern GaussianFiltering gaussfilter;
extern ChebyshevFiltering chebyfilter;

/**
 * @brief Calculates the Mean Absolute Error (MAE) between two arrays of values.
 *
 * @tparam T The floating point data type of the elements in the input arrays
 *
 * @param x Pointer to the first array of true values
 * @param y Pointer to the second array of calculated values
 * @param total_pts The number of elements in the arrays
 *
 * @return The Mean Absolute Error (MAE) as a value of type T.
 */
template <typename T>
T calculate_mae(T *x, T *y, std::size_t total_pts) {
    T mae = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        T error = std::abs(x[i] - y[i]);
        mae += error;
    }
    return mae / total_pts;
}

template <typename T>
T calculate_mae(T *absError, std::size_t total_pts) {
    T mae = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        mae += absError[i];
    }
    return mae / total_pts;
}

template <typename T>
T calculate_mse(T *x, T *y, std::size_t total_pts) {
    T mse = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        T error = std::abs(x[i] - y[i]);
        mse += error * error;
    }
    return mse / total_pts;
}

template <typename T>
T calculate_mse(T *absError, std::size_t total_pts) {
    T mse = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        mse += absError[i] * absError[i];
    }
    return mse / total_pts;
}

template <typename T>
T calculate_rmse(T *x, T *y, std::size_t total_pts) {
    return std::sqrt(calculate_mse(x, y, total_pts));
}

template <typename T>
T calculate_rmse(T *absError, std::size_t total_pts) {
    return std::sqrt(calculate_mse(absError, total_pts));
}

template <typename T>
T calculate_max_error(T *x, T *y, std::size_t total_pts) {
    T max_error = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        T error   = std::abs(x[i] - y[i]);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

template <typename T>
T calculate_max_error(T *absError, std::size_t total_pts) {
    T max_error = 0;
    for (size_t i = 0; i < total_pts; ++i) {
        max_error = std::max(max_error, absError[i]);
    }
    return max_error;
}

template <typename T>
T calculate_min_error(T *x, T *y, std::size_t total_pts) {
    T min_error = std::numeric_limits<T>::max();
    for (size_t i = 0; i < total_pts; ++i) {
        T error   = std::abs(x[i] - y[i]);
        min_error = std::min(min_error, error);
    }
    return min_error;
}

template <typename T>
T calculate_min_error(T *absError, std::size_t total_pts) {
    T min_error = std::numeric_limits<T>::max();
    for (size_t i = 0; i < total_pts; ++i) {
        min_error = std::min(min_error, absError[i]);
    }
    return min_error;
}

enum CompressionType {
    NONE = 0,
    ZFP,
    CHEBYSHEV,
    BLOSC,
    TORCH_SCRIPT,
    ONNX_MODEL
};
static const char *COMPRESSION_TYPE_NAMES[] = {
    "NONE", "ZFP", "CHEBYSHEV", "BLOSC", "TORCH_SCRIPT", "ONNX_MODEL"};

enum FilterType { F_NONE = 0, F_GAUSSIAN, F_CHEBYSHEV };
static const char *FILTER_TYPE_NAMES[] = {"F_NONE", "F_GAUSSIAN",
                                          "F_CHEBYSHEV"};

// then the global option
extern CompressionType COMPRESSION_OPTION;
extern FilterType COMPRESSION_FILTER_OPTION;

struct CompressionOptions {
    size_t eleOrder             = 6;
    size_t numVars              = 1;
    // options just for blosc
    std::string bloscCompressor = "lz4";
    int bloscClevel             = 5;
    int bloscDoShuffle          = 1;

    // options just for ZFP

    // Options: accuracy, rate
    std::string zfpMode         = "accuracy";
    double zfpRate              = 5.0;
    double zfpAccuracyTolerance = 1e-6;
    unsigned int zfpPrecision   = 24;

    // options for chebyshev
    size_t chebyNReduced        = 3;

    // options for ML
    std::string encoder_3d_path;
    std::string decoder_3d_path;
    std::string encoder_2d_path;
    std::string decoder_2d_path;
    std::string encoder_1d_path;
    std::string decoder_1d_path;
    std::string encoder_0d_path;
    std::string decoder_0d_path;
};

std::ostream &operator<<(std::ostream &out, const CompressionOptions opts);

std::ostream &operator<<(std::ostream &out, const CompressionType t);
std::ostream &operator<<(std::ostream &out, const FilterType t);

void set_compression_options(
    CompressionType compT, const CompressionOptions &compOpt,
    const ot::CTXSendType sendType = ot::CTXSendType::CTX_DOUBLE,
    const FilterType filterT       = FilterType::F_NONE);

template <typename T>
std::size_t blockwise_compression(
    T *buffer, unsigned char *compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig> &blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);

template <typename T>
std::size_t blockwise_decompression(
    T *buffer, unsigned char *compressBuffer, const size_t numBlocks,
    const std::vector<sm_config::SMConfig> &blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder);

#define __DENDRO_DEFAULT_BATCH_SIZE__ 32

template <typename T>
std::size_t blockwise_all_dof_compression(
    T *buffer, unsigned char *compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig> &blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4> &blockDimCounts,
    const std::array<unsigned int, 4> &blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize = __DENDRO_DEFAULT_BATCH_SIZE__);

template <typename T>
std::size_t blockwise_all_dof_decompression(
    T *buffer, unsigned char *compressBuffer, const size_t numBlocks,
    const size_t dof,
    const std::vector<sm_config::SMConfig> &blockConfiguration,
    const size_t blockConfigOffset,
    const std::array<unsigned int, 4> &blockDimCounts,
    const std::array<unsigned int, 4> &blockDimOffsets, const size_t eleorder,
    const unsigned int batchSize = __DENDRO_DEFAULT_BATCH_SIZE__);
}  // namespace dendro_compress
