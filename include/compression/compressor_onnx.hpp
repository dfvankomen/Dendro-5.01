#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "compression_base.hpp"

namespace dendrocompression {

template <typename T>
Ort::Value createOnnxTensorFromData(const T *original_matrix,
                                    const unsigned int nPoints,
                                    const unsigned int nBatches,
                                    std::vector<float> &floatBuffer,
                                    const Ort::MemoryInfo &meminfo,
                                    std::vector<long> &inputShape) {
    if constexpr (std::is_same_v<T, double>) {
        // convert to floats
        std::transform(original_matrix, original_matrix + (nPoints * nBatches),
                       floatBuffer.begin(),
                       [](double d) { return static_cast<float>(d); });

        // then return the tensor data
        return Ort::Value::CreateTensor<float>(
            meminfo, const_cast<float *>(floatBuffer.data()),
            floatBuffer.size(), inputShape.data(), inputShape.size());
    } else if constexpr (std::is_same_v<T, float>) {
        // no need to convert if we need the floats as output
        return Ort::Value::CreateTensor<float>(
            meminfo, const_cast<float *>(original_matrix), nPoints * nBatches,
            inputShape.data(), inputShape.size());
    } else {
        std::cerr << "Somehow ONNX Tensor Creation FAILED in templating!"
                  << std::endl;
        exit(-1);
        return Ort::Value(nullptr);
    }
}

template <typename T>
class ONNXCompressor : public Compression<T> {
   private:
    std::string encoder_3d_path_;
    std::string decoder_3d_path_;
    std::string encoder_2d_path_;
    std::string decoder_2d_path_;
    std::string encoder_1d_path_;
    std::string decoder_1d_path_;
    std::string encoder_0d_path_;
    std::string decoder_0d_path_;

    std::unique_ptr<Ort::Session> encoder_3d_;
    std::unique_ptr<Ort::Session> decoder_3d_;
    std::unique_ptr<Ort::Session> encoder_2d_;
    std::unique_ptr<Ort::Session> decoder_2d_;
    std::unique_ptr<Ort::Session> encoder_1d_;
    std::unique_ptr<Ort::Session> decoder_1d_;
    std::unique_ptr<Ort::Session> encoder_0d_;
    std::unique_ptr<Ort::Session> decoder_0d_;

    unsigned int n_outs_3d_encoder_;
    unsigned int n_outs_2d_encoder_;
    unsigned int n_outs_1d_encoder_;
    unsigned int n_outs_0d_encoder_;

    std::vector<int64_t> input_shape_3d_;
    std::vector<int64_t> input_shape_2d_;
    std::vector<int64_t> input_shape_1d_;
    std::vector<int64_t> input_shape_0d_;

    std::vector<int64_t> decoder_shape_3d_;
    std::vector<int64_t> decoder_shape_2d_;
    std::vector<int64_t> decoder_shape_1d_;
    std::vector<int64_t> decoder_shape_0d_;

    std::vector<float> double_to_float_buffer_3d_;
    std::vector<float> double_to_float_buffer_2d_;
    std::vector<float> double_to_float_buffer_1d_;
    std::vector<float> double_to_float_buffer_0d_;

    std::string input_name_3d_;
    std::string output_name_3d_;
    std::string input_name_2d_;
    std::string output_name_2d_;
    std::string input_name_1d_;
    std::string output_name_1d_;
    std::string input_name_0d_;
    std::string output_name_0d_;

    std::string decoder_input_name_3d_;
    std::string decoder_output_name_3d_;
    std::string decoder_input_name_2d_;
    std::string decoder_output_name_2d_;
    std::string decoder_input_name_1d_;
    std::string decoder_output_name_1d_;
    std::string decoder_input_name_0d_;
    std::string decoder_output_name_0d_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;

    // default CPU selection
    ExecutionProviderType selected_execution_provider_ =
        ExecutionProviderType::CPU;

    bool treat_variables_as_batch_3d_ = false;
    bool treat_variables_as_batch_2d_ = false;
    bool treat_variables_as_batch_1d_ = false;
    bool treat_variables_as_batch_0d_ = false;

    void probe_and_setup_model(
        Ort::Session &encoder, Ort::Session &decoder,
        const std::string &model_dim_str, std::vector<int64_t> &input_shape,
        bool &treat_vars_as_batch, unsigned int &n_outs_encoder,
        std::string &encoder_input_name, std::string &encoder_output_name,
        std::string &decoder_input_name, std::string &decoder_output_name,
        std::vector<int64_t> &decoder_shape) {
        Ort::AllocatorWithDefaultOptions allocator;

        // Step 1. model's expected input shape from metadata
        Ort::TypeInfo input_type_info = encoder.GetInputTypeInfo(0);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> model_expected_shape = tensor_info.GetShape();

        // Step 2: make sure the model has the expected number of dimensions
        size_t expected_dims                      = input_shape.size();
        if (model_expected_shape.size() != expected_dims) {
            throw std::runtime_error(
                "ERROR: Mismatched dimensions encountered when loading "
                "encoder.");
        }

        // Step 3: how to handle the "variables" dimension
        int64_t channel_dim_size = model_expected_shape[1];

        if (channel_dim_size == 1) {
            // if the channel dimension is 1, then we need to "flatten"
            treat_vars_as_batch = true;
        } else if (channel_dim_size == this->num_vars_) {
            // normal operation, all variables are together
            treat_vars_as_batch = false;
        } else if (channel_dim_size < 0) {
            // this is a handler for ambiguous channel dimension
            std::cerr << "Warning: " << model_dim_str
                      << " encoder has a dynamic channel dimension. "
                      << "Assuming it will handle all " << this->num_vars_
                      << " variables as channels.\n";
            treat_vars_as_batch = false;
        } else {
            // if it's zero or any other positive integer, throw an error
            throw std::runtime_error(
                "ERROR: Unexpected channel dimension size for " +
                model_dim_str + " encoder. Model expects " +
                std::to_string(channel_dim_size) +
                " channels, but code is configured for 1 or " +
                std::to_string(this->num_vars_) + ".");
        }

        // Step 4: check the names and get meta data for inputs/outputs
        auto enc_input_name_ptr  = encoder.GetInputNameAllocated(0, allocator);
        encoder_input_name       = enc_input_name_ptr.get();
        auto enc_output_name_ptr = encoder.GetOutputNameAllocated(0, allocator);
        encoder_output_name      = enc_output_name_ptr.get();

        auto dec_input_name_ptr  = decoder.GetInputNameAllocated(0, allocator);
        decoder_input_name       = dec_input_name_ptr.get();
        auto dec_output_name_ptr = decoder.GetOutputNameAllocated(0, allocator);
        decoder_output_name      = dec_output_name_ptr.get();

        // Step 5: Test the encoder for shape information
        std::vector<int64_t> test_run_shape = tensor_info.GetShape();
        // Replace symbolic dimensions with concrete dimensions
        for (auto &dim : test_run_shape) {
            if (dim < 0) {
                dim = 1;
            }
        }

        size_t test_tensor_size =
            std::accumulate(test_run_shape.begin(), test_run_shape.end(), 1LL,
                            std::multiplies<int64_t>());
        std::vector<float> test_data(test_tensor_size, 1.0f);

        Ort::Value test_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, test_data.data(), test_data.size(),
            test_run_shape.data(), test_run_shape.size());

        const char *input_names[]  = {encoder_input_name.c_str()};
        const char *output_names[] = {encoder_output_name.c_str()};

        auto output_tensors = encoder.Run(Ort::RunOptions{nullptr}, input_names,
                                          &test_tensor, 1, output_names, 1);

        // shape info gives information about one item (flattened or one block
        // of all vars if not)
        auto output_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        n_outs_encoder          = output_tensor_info.GetElementCount();

        // 6. Setup the base shape for the decoder input tensor.
        decoder_shape           = {1, static_cast<int64_t>(n_outs_encoder)};
    }

    Ort::Value createOnnxTensorFromDataInternal(
        const T *original_matrix, size_t total_elements,
        std::vector<float> &float_buffer, const std::vector<int64_t> &shape) {
        // NOTE: this is a simplified data, other data conversions will be
        // required!
        if constexpr (std::is_same_v<T, double>) {
            // float converstion
            std::transform(original_matrix, original_matrix + total_elements,
                           float_buffer.begin(),
                           [](double d) { return static_cast<float>(d); });

            // create the tensor
            return Ort::Value::CreateTensor<float>(
                memory_info_, float_buffer.data(), total_elements, shape.data(),
                shape.size());
        } else if constexpr (std::is_same_v<T, float>) {
            // nothing to convert if we already have floats
            return Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float *>(original_matrix),
                total_elements, shape.data(), shape.size());
        }
    }

    void load_models() {
        // then attempt to load 3D
        encoder_3d_ = std::make_unique<Ort::Session>(
            env_, encoder_3d_path_.c_str(), session_options_);
        decoder_3d_ = std::make_unique<Ort::Session>(
            env_, decoder_3d_path_.c_str(), session_options_);

        // 2D
        encoder_2d_ = std::make_unique<Ort::Session>(
            env_, encoder_2d_path_.c_str(), session_options_);
        decoder_2d_ = std::make_unique<Ort::Session>(
            env_, decoder_2d_path_.c_str(), session_options_);

        // 1D
        encoder_1d_ = std::make_unique<Ort::Session>(
            env_, encoder_1d_path_.c_str(), session_options_);
        decoder_1d_ = std::make_unique<Ort::Session>(
            env_, decoder_1d_path_.c_str(), session_options_);

        encoder_0d_ = std::make_unique<Ort::Session>(
            env_, encoder_0d_path_.c_str(), session_options_);
        decoder_0d_ = std::make_unique<Ort::Session>(
            env_, decoder_0d_path_.c_str(), session_options_);

        // Probe and setup each dimension's models.
        probe_and_setup_model(*encoder_3d_, *decoder_3d_, "3D", input_shape_3d_,
                              treat_variables_as_batch_3d_, n_outs_3d_encoder_,
                              input_name_3d_, output_name_3d_,
                              decoder_input_name_3d_, decoder_output_name_3d_,
                              decoder_shape_3d_);

        probe_and_setup_model(*encoder_2d_, *decoder_2d_, "2D", input_shape_2d_,
                              treat_variables_as_batch_2d_, n_outs_2d_encoder_,
                              input_name_2d_, output_name_2d_,
                              decoder_input_name_2d_, decoder_output_name_2d_,
                              decoder_shape_2d_);

        probe_and_setup_model(*encoder_1d_, *decoder_1d_, "1D", input_shape_1d_,
                              treat_variables_as_batch_1d_, n_outs_1d_encoder_,
                              input_name_1d_, output_name_1d_,
                              decoder_input_name_1d_, decoder_output_name_1d_,
                              decoder_shape_1d_);

        probe_and_setup_model(*encoder_0d_, *decoder_0d_, "0D", input_shape_0d_,
                              treat_variables_as_batch_0d_, n_outs_0d_encoder_,
                              input_name_0d_, output_name_0d_,
                              decoder_input_name_0d_, decoder_output_name_0d_,
                              decoder_shape_0d_);
    }

   public:
    ONNXCompressor(
        unsigned int ele_order, unsigned int num_vars,
        const std::string &encoder_3d_path, const std::string &decoder_3d_path,
        const std::string &encoder_2d_path, const std::string &decoder_2d_path,
        const std::string &encoder_1d_path, const std::string &decoder_1d_path,
        const std::string &encoder_0d_path, const std::string &decoder_0d_path,
        ExecutionProviderType execution_provider = ExecutionProviderType::CPU)
        : Compression<T>(ele_order, num_vars),
          encoder_3d_path_(encoder_3d_path),
          decoder_3d_path_(decoder_3d_path),
          encoder_2d_path_(encoder_2d_path),
          decoder_2d_path_(decoder_2d_path),
          encoder_1d_path_(encoder_1d_path),
          decoder_1d_path_(decoder_1d_path),
          encoder_0d_path_(encoder_0d_path),
          decoder_0d_path_(encoder_0d_path),
          memory_info_(
              Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          selected_execution_provider_(execution_provider) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // set up the execution provider...
        switch (selected_execution_provider_) {
            case dendrocompression::ExecutionProviderType::CUDA: {
                // building the NVIDIA CUDA provider
                std::cout << "INFO[dendrocompression:ONNX]: Attempting to "
                             "enable CUDA execution "
                             "provider (NVIDIA)..."
                          << std::endl;
                OrtCUDAProviderOptions cuda_options{};
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                break;
            }
            case dendrocompression::ExecutionProviderType::ROCM: {
                // AMD provider
                std::cout << "INFO[dendrocompression:ONNX]: Attempting to "
                             "enable ROCm execution "
                             "provider (AMD)...."
                          << std::endl;
                OrtROCMProviderOptions rocm_options{};
                session_options_.AppendExecutionProvider_ROCM(rocm_options);
                break;
            }
            case dendrocompression::ExecutionProviderType::OpenVINO: {
                // openvino, which is an alternative provider specifically for
                // intel
                std::cout << "INFO[dendrocompression:ONNX]: Attempting to "
                             "enable OpenVINO execution "
                             "provider (Intel)..."
                          << std::endl;
                OrtOpenVINOProviderOptions openvino_options{};
                session_options_.AppendExecutionProvider_OpenVINO(
                    openvino_options);
            }
            // NOTE: we can also technically add DirectML, which would give
            // support Windows, as it uses DirectX 12. We don't support Windows
            // currently, but the option does exist for future
            case dendrocompression::ExecutionProviderType::CPU:
            default:
                std::cout << "INFO[dendrocompression:ONNX]: Using CPU "
                             "execution provider"
                          << std::endl;
                break;
        }

        // then set up the sizes and buffers
        input_shape_3d_ = {1, this->num_vars_, this->n_, this->n_, this->n_};
        input_shape_2d_ = {1, this->num_vars_, this->n_, this->n_};
        input_shape_1d_ = {1, this->num_vars_, this->n_};
        input_shape_0d_ = {1, this->num_vars_};

        double_to_float_buffer_3d_ = std::vector<float>(this->total_3d_pts_);
        double_to_float_buffer_2d_ = std::vector<float>(this->total_2d_pts_);
        double_to_float_buffer_1d_ = std::vector<float>(this->total_1d_pts_);
        double_to_float_buffer_0d_ = std::vector<float>(this->total_0d_pts_);

        load_models();
    }

    ~ONNXCompressor() = default;

    std::unique_ptr<Compression<T>> clone() const override {
        auto cloned = std::make_unique<ONNXCompressor>(
            this->ele_order_, this->num_vars_, this->encoder_3d_path_,
            this->decoder_3d_path_, this->encoder_2d_path_,
            this->decoder_2d_path_, this->encoder_1d_path_,
            this->decoder_1d_path_, this->encoder_0d_path_,
            this->decoder_0d_path_, this->selected_execution_provider_);

        // constructor should call load models and do the session initilaization
        return cloned;
    }

    CompressionType get_compression_type() const override {
        return CompressionType::COMP_ONNX_MODEL;
    }

    std::string to_string() const override { return "ONNXCompressor"; }

    std::size_t do_compress_3d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        if (!encoder_3d_) {
            throw std::runtime_error(
                "ERROR: The 3D encoder is not initialized!");
        }

        std::vector<int64_t> current_input_shape = input_shape_3d_;
        unsigned int effective_batch_size;

        if (treat_variables_as_batch_3d_) {
            effective_batch_size   = batch_size * this->num_vars_;
            current_input_shape[0] = effective_batch_size;
            current_input_shape[1] = 1;
        } else {
            effective_batch_size   = batch_size;
            current_input_shape[0] = effective_batch_size;
        }

        size_t total_elements = this->total_3d_pts_ * batch_size;
        if (double_to_float_buffer_3d_.size() < total_elements) {
            double_to_float_buffer_3d_.resize(total_elements);
        }

        Ort::Value tensor_data = createOnnxTensorFromDataInternal(
            original_matrix, total_elements, double_to_float_buffer_3d_,
            current_input_shape);

        const char *input_names[]  = {input_name_3d_.c_str()};
        const char *output_names[] = {output_name_3d_.c_str()};

        auto output_tensors =
            encoder_3d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, 1);

        const float *output_data = output_tensors[0].GetTensorData<float>();
        size_t total_output_bytes =
            n_outs_3d_encoder_ * sizeof(float) * effective_batch_size;
        std::memcpy(output_array, output_data, total_output_bytes);

        return total_output_bytes;
    }

    std::size_t do_decompress_3d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        if (!decoder_3d_) {
            throw std::runtime_error(
                "ERROR: The 3D decoder is not initialized!");
        }

        unsigned int effective_batch_size = treat_variables_as_batch_3d_
                                                ? (batch_size * this->num_vars_)
                                                : batch_size;
        size_t compressed_elements = n_outs_3d_encoder_ * effective_batch_size;

        std::vector<int64_t> current_decoder_input_shape = decoder_shape_3d_;
        current_decoder_input_shape[0]                   = effective_batch_size;

        std::vector<int64_t> final_output_shape          = input_shape_3d_;
        if (treat_variables_as_batch_3d_) {
            final_output_shape[0] = effective_batch_size;
            final_output_shape[1] = 1;
        } else {
            final_output_shape[0] = effective_batch_size;
        }

        const float *float_input_array =
            reinterpret_cast<const float *>(compressed_buffer);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(float_input_array),
            compressed_elements, current_decoder_input_shape.data(),
            current_decoder_input_shape.size());

        const char *input_names[]    = {decoder_input_name_3d_.c_str()};
        const char *output_names[]   = {decoder_output_name_3d_.c_str()};
        size_t total_output_elements = this->total_3d_pts_ * batch_size;

        Ort::Value tensor_data       = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(float_input_array),
            n_outs_3d_encoder_ * batch_size, decoder_shape_3d_.data(),
            decoder_shape_3d_.size());
        const float *tensor_values = tensor_data.GetTensorData<float>();

        if constexpr (std::is_same_v<T, double>) {
            // output to the buffer array so we're not deleting it
            if (double_to_float_buffer_3d_.size() <
                this->total_3d_pts_ * batch_size) {
                double_to_float_buffer_3d_.resize(this->total_3d_pts_ *
                                                  batch_size);
            }

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, double_to_float_buffer_3d_.data(),
                total_output_elements, final_output_shape.data(),
                final_output_shape.size());

            decoder_3d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, &output_tensor, 1);

            // now the data has been written to output tensor so we don't have
            // to worry about *more copies*, we just convert back to doubles
            std::transform(
                double_to_float_buffer_3d_.begin(),
                double_to_float_buffer_3d_.begin() + total_output_elements,
                output_array, [](float f) { return static_cast<T>(f); });
        } else if constexpr (std::is_same_v<T, float>) {
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, reinterpret_cast<float *>(output_array),
                total_output_elements, final_output_shape.data(),
                final_output_shape.size());

            decoder_3d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, &output_tensor, 1);

            // here, we have the data just within output_tensor, which mapped
            // directly to our output array, so we are done
        }

        return compressed_elements * sizeof(float);
    }

    std::size_t do_compress_2d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        if (!encoder_2d_) {
            throw std::runtime_error(
                "ERROR: The 2D encoder is not initialized!");
        }

        std::vector<int64_t> current_input_shape = input_shape_2d_;
        unsigned int effective_batch_size;

        if (treat_variables_as_batch_2d_) {
            effective_batch_size   = batch_size * this->num_vars_;
            current_input_shape[0] = effective_batch_size;
            current_input_shape[1] = 1;
        } else {
            effective_batch_size   = batch_size;
            current_input_shape[0] = effective_batch_size;
        }

        size_t total_elements = this->total_2d_pts_ * batch_size;
        if (double_to_float_buffer_2d_.size() < total_elements) {
            double_to_float_buffer_2d_.resize(total_elements);
        }

        Ort::Value tensor_data = createOnnxTensorFromDataInternal(
            original_matrix, total_elements, double_to_float_buffer_2d_,
            current_input_shape);

        const char *input_names[]  = {input_name_2d_.c_str()};
        const char *output_names[] = {output_name_2d_.c_str()};

        auto output_tensors =
            encoder_2d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, 1);

        const float *output_data = output_tensors[0].GetTensorData<float>();
        size_t total_output_bytes =
            n_outs_2d_encoder_ * sizeof(float) * effective_batch_size;
        std::memcpy(output_array, output_data, total_output_bytes);

        return total_output_bytes;
    }

    std::size_t do_decompress_2d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        if (!decoder_2d_) {
            throw std::runtime_error(
                "ERROR: The 2D decoder is not initialized!");
        }

        unsigned int effective_batch_size = treat_variables_as_batch_2d_
                                                ? (batch_size * this->num_vars_)
                                                : batch_size;
        size_t compressed_elements = n_outs_2d_encoder_ * effective_batch_size;

        std::vector<int64_t> current_decoder_input_shape = decoder_shape_2d_;
        current_decoder_input_shape[0]                   = effective_batch_size;

        std::vector<int64_t> final_output_shape          = input_shape_2d_;
        if (treat_variables_as_batch_2d_) {
            final_output_shape[0] = effective_batch_size;
            final_output_shape[1] = 1;
        } else {
            final_output_shape[0] = effective_batch_size;
        }

        const float *float_input_array =
            reinterpret_cast<const float *>(compressed_buffer);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(float_input_array),
            compressed_elements, current_decoder_input_shape.data(),
            current_decoder_input_shape.size());

        const char *input_names[]    = {decoder_input_name_2d_.c_str()};
        const char *output_names[]   = {decoder_output_name_2d_.c_str()};
        size_t total_output_elements = this->total_2d_pts_ * batch_size;

        Ort::Value tensor_data       = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(float_input_array),
            n_outs_2d_encoder_ * batch_size, decoder_shape_2d_.data(),
            decoder_shape_2d_.size());
        const float *tensor_values = tensor_data.GetTensorData<float>();

        if constexpr (std::is_same_v<T, double>) {
            // output to the buffer array so we're not deleting it
            if (double_to_float_buffer_2d_.size() <
                this->total_2d_pts_ * batch_size) {
                double_to_float_buffer_2d_.resize(this->total_2d_pts_ *
                                                  batch_size);
            }

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, double_to_float_buffer_2d_.data(),
                total_output_elements, final_output_shape.data(),
                final_output_shape.size());

            decoder_2d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, &output_tensor, 1);

            // now the data has been written to output tensor so we don't have
            // to worry about *more copies*, we just convert back to doubles
            std::transform(
                double_to_float_buffer_2d_.begin(),
                double_to_float_buffer_2d_.begin() + total_output_elements,
                output_array, [](float f) { return static_cast<T>(f); });
        } else if constexpr (std::is_same_v<T, float>) {
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, reinterpret_cast<float *>(output_array),
                total_output_elements, final_output_shape.data(),
                final_output_shape.size());

            decoder_2d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, &output_tensor, 1);

            // here, we have the data just within output_tensor, which mapped
            // directly to our output array, so we are done
        }

        return compressed_elements * sizeof(float);
    }

    std::size_t do_compress_1d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        if (!encoder_1d_) {
            throw std::runtime_error(
                "ERROR: The 1D encoder is not initialized!");
        }

        std::vector<int64_t> current_input_shape = input_shape_1d_;
        unsigned int effective_batch_size;

        if (treat_variables_as_batch_1d_) {
            effective_batch_size   = batch_size * this->num_vars_;
            current_input_shape[0] = effective_batch_size;
            current_input_shape[1] = 1;
        } else {
            effective_batch_size   = batch_size;
            current_input_shape[0] = effective_batch_size;
        }

        size_t total_elements = this->total_1d_pts_ * batch_size;
        if (double_to_float_buffer_1d_.size() < total_elements) {
            double_to_float_buffer_1d_.resize(total_elements);
        }

        Ort::Value tensor_data = createOnnxTensorFromDataInternal(
            original_matrix, total_elements, double_to_float_buffer_1d_,
            current_input_shape);

        const char *input_names[]  = {input_name_1d_.c_str()};
        const char *output_names[] = {output_name_1d_.c_str()};

        auto output_tensors =
            encoder_1d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, 1);

        const float *output_data = output_tensors[0].GetTensorData<float>();
        size_t total_output_bytes =
            n_outs_1d_encoder_ * sizeof(float) * effective_batch_size;
        std::memcpy(output_array, output_data, total_output_bytes);

        return total_output_bytes;
    }

    std::size_t do_decompress_1d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        if (!decoder_1d_) {
            throw std::runtime_error(
                "ERROR: The 1D decoder is not initialized!");
        }

        unsigned int effective_batch_size = treat_variables_as_batch_1d_
                                                ? (batch_size * this->num_vars_)
                                                : batch_size;
        size_t compressed_elements = n_outs_1d_encoder_ * effective_batch_size;

        std::vector<int64_t> current_decoder_input_shape = decoder_shape_1d_;
        current_decoder_input_shape[0]                   = effective_batch_size;

        std::vector<int64_t> final_output_shape          = input_shape_1d_;
        if (treat_variables_as_batch_1d_) {
            final_output_shape[0] = effective_batch_size;
            final_output_shape[1] = 1;
        } else {
            final_output_shape[0] = effective_batch_size;
        }

        const float *float_input_array =
            reinterpret_cast<const float *>(compressed_buffer);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(float_input_array),
            compressed_elements, current_decoder_input_shape.data(),
            current_decoder_input_shape.size());

        const char *input_names[]    = {decoder_input_name_1d_.c_str()};
        const char *output_names[]   = {decoder_output_name_1d_.c_str()};
        size_t total_output_elements = this->total_1d_pts_ * batch_size;

        Ort::Value tensor_data       = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(float_input_array),
            n_outs_1d_encoder_ * batch_size, decoder_shape_1d_.data(),
            decoder_shape_1d_.size());
        const float *tensor_values = tensor_data.GetTensorData<float>();

        if constexpr (std::is_same_v<T, double>) {
            // output to the buffer array so we're not deleting it
            if (double_to_float_buffer_1d_.size() <
                this->total_1d_pts_ * batch_size) {
                double_to_float_buffer_1d_.resize(this->total_1d_pts_ *
                                                  batch_size);
            }

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, double_to_float_buffer_1d_.data(),
                total_output_elements, final_output_shape.data(),
                final_output_shape.size());

            decoder_1d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, &output_tensor, 1);

            // now the data has been written to output tensor so we don't have
            // to worry about *more copies*, we just convert back to doubles
            std::transform(
                double_to_float_buffer_1d_.begin(),
                double_to_float_buffer_1d_.begin() + total_output_elements,
                output_array, [](float f) { return static_cast<T>(f); });
        } else if constexpr (std::is_same_v<T, float>) {
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, reinterpret_cast<float *>(output_array),
                total_output_elements, final_output_shape.data(),
                final_output_shape.size());

            decoder_1d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &input_tensor, 1, output_names, &output_tensor, 1);

            // here, we have the data just within output_tensor, which mapped
            // directly to our output array, so we are done
        }

        return compressed_elements * sizeof(float);
    }

    std::size_t do_compress_0d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        std::memcpy(output_array, original_matrix,
                    batch_size * this->total_0d_bytes_);
        return batch_size * this->total_0d_bytes_;
    }

    std::size_t do_decompress_0d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        std::memcpy(output_array, compressed_buffer,
                    batch_size * this->total_0d_bytes_);
        return batch_size * this->total_0d_bytes_;
    }

    std::size_t do_compress_flat(T *const original_matrix,
                                 unsigned char *const output_array,
                                 unsigned int n_pts) override {
        std::memcpy(output_array, original_matrix, n_pts * sizeof(T));
        return n_pts * sizeof(T);
    }

    std::size_t do_decompress_flat(unsigned char *const compressed_buffer,
                                   T *const output_array,
                                   unsigned int n_pts) override {
        std::memcpy(output_array, compressed_buffer, n_pts * sizeof(T));
        return n_pts * sizeof(T);
    }
};

}  // namespace dendrocompression
