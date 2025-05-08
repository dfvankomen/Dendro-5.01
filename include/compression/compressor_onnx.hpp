#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
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

        std::cout << "LOADED MODELS" << std::endl;

        // TODO: 0D, will need checks for it

        // allocator that helps us get the input and output names
        Ort::AllocatorWithDefaultOptions allocator;

        // --------
        // 3d Checks

        // CALCULATE THE OUTPUT SIZE OF THE ENCODER TO STORE INTERNALLY
        std::vector<float> test_data(this->total_3d_pts_, 1.0);

        Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(test_data.data()),
            test_data.size(), input_shape_3d_.data(), input_shape_3d_.size());

        // this returns a smart pointer, which we'll just clear at the end of
        // the function anyway
        auto output_name_ptr =
            encoder_3d_->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        output_name_3d_     = output_name_ptr.get();

        auto input_name_ptr = encoder_3d_->GetInputNameAllocated(0, allocator);
        input_name_3d_      = input_name_ptr.get();

        std::cout << "ENCODER SHAPE: ";
        for (auto i : input_shape_3d_) {
            std::cout << i << " ";
        }
        std::cout << std::endl;

        const char *input_names_3d[]  = {input_name_3d_.c_str()};
        const char *output_names_3d[] = {output_name_3d_.c_str()};
        auto output = encoder_3d_->Run(Ort::RunOptions{nullptr}, input_names_3d,
                                       &tensor_data, 1, output_names_3d, 1);

        n_outs_3d_encoder_ =
            output[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // decoder names
        output_name_ptr = decoder_3d_->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        decoder_output_name_3d_ = output_name_ptr.get();

        input_name_ptr = decoder_3d_->GetInputNameAllocated(0, allocator);
        decoder_input_name_3d_ = input_name_ptr.get();

        // now we can do the same for other dimensionalities

        // --------
        // 2d Checks
        test_data.resize(this->total_2d_pts_);
        // override tensor data
        tensor_data = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(test_data.data()),
            test_data.size(), input_shape_2d_.data(), input_shape_2d_.size());
        output_name_ptr = encoder_2d_->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        output_name_2d_ = output_name_ptr.get();

        input_name_ptr  = encoder_2d_->GetInputNameAllocated(0, allocator);
        input_name_2d_  = input_name_ptr.get();

        const char *input_names_2d[]  = {input_name_2d_.c_str()};
        const char *output_names_2d[] = {output_name_2d_.c_str()};
        output = encoder_2d_->Run(Ort::RunOptions{nullptr}, input_names_2d,
                                  &tensor_data, 1, output_names_2d, 1);

        n_outs_2d_encoder_ =
            output[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // decoder names
        output_name_ptr = decoder_2d_->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        decoder_output_name_2d_ = output_name_ptr.get();

        input_name_ptr = decoder_2d_->GetInputNameAllocated(0, allocator);
        decoder_input_name_2d_ = input_name_ptr.get();

        // --------
        // 1d Checks
        test_data.resize(this->total_1d_pts_);
        // override tensor data
        tensor_data = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(test_data.data()),
            test_data.size(), input_shape_1d_.data(), input_shape_1d_.size());
        output_name_ptr = encoder_1d_->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        output_name_1d_ = output_name_ptr.get();

        input_name_ptr  = encoder_1d_->GetInputNameAllocated(0, allocator);
        input_name_1d_  = input_name_ptr.get();

        const char *input_names_1d[]  = {input_name_1d_.c_str()};
        const char *output_names_1d[] = {output_name_1d_.c_str()};
        output = encoder_1d_->Run(Ort::RunOptions{nullptr}, input_names_1d,
                                  &tensor_data, 1, output_names_1d, 1);

        n_outs_1d_encoder_ =
            output[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // decoder names
        output_name_ptr = decoder_1d_->GetOutputNameAllocated(0, allocator);
        // fetch the string output
        decoder_output_name_1d_ = output_name_ptr.get();

        input_name_ptr = decoder_1d_->GetInputNameAllocated(0, allocator);
        decoder_input_name_1d_ = input_name_ptr.get();

        decoder_shape_3d_      = {1, n_outs_3d_encoder_};
        decoder_shape_2d_      = {1, n_outs_2d_encoder_};
        decoder_shape_1d_      = {1, n_outs_1d_encoder_};
    }

   public:
    ONNXCompressor(
        unsigned int ele_order, unsigned int num_vars,
        const std::string &encoder_3d_path, const std::string &decoder_3d_path,
        const std::string &encoder_2d_path, const std::string &decoder_2d_path,
        const std::string &encoder_1d_path, const std::string &decoder_1d_path,
        const std::string &encoder_0d_path, const std::string &decoder_0d_path)
        : Compression<T>(ele_order, num_vars),
          encoder_3d_path_(encoder_3d_path),
          decoder_3d_path_(decoder_3d_path),
          encoder_2d_path_(encoder_2d_path),
          decoder_2d_path_(decoder_2d_path),
          encoder_1d_path_(encoder_1d_path),
          decoder_1d_path_(decoder_1d_path),
          encoder_0d_path_(encoder_0d_path),
          decoder_0d_path_(encoder_0d_path),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                  OrtMemTypeDefault)) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

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
            this->decoder_0d_path_);

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
                "ERROR: The 3D encoder is not initialized! Make sure "
                "set_models() "
                "is called.");
        }

        // update batch size
        input_shape_3d_[0] = batch_size;

        std::cout << "ENCODER SHAPE: ";
        for (auto i : input_shape_3d_) {
            std::cout << i << " ";
        }
        std::cout << std::endl;

        // only resize if we're going to use this buffer
        if constexpr (std::is_same_v<T, double>) {
            if (double_to_float_buffer_3d_.size() <
                this->total_3d_pts_ * batch_size) {
                double_to_float_buffer_3d_.resize(this->total_3d_pts_ *
                                                  batch_size);
            }
        }

        Ort::Value tensor_data = createOnnxTensorFromData(
            original_matrix, this->total_3d_pts_, batch_size,
            double_to_float_buffer_3d_, memory_info_, input_shape_3d_);

        const float *tensor_values = tensor_data.GetTensorData<float>();

        const char *input_names[]  = {input_name_3d_.c_str()};
        const char *output_names[] = {output_name_3d_.c_str()};

        auto output = encoder_3d_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &tensor_data, 1, output_names, 1);

        const float *output_data = output[0].GetTensorData<float>();

        std::memcpy(output_array, output_data,
                    n_outs_3d_encoder_ * sizeof(float) * batch_size);
        return batch_size * n_outs_3d_encoder_ * sizeof(float);
    }

    std::size_t do_decompress_3d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        if (!decoder_3d_) {
            throw std::runtime_error(
                "ERROR: The 3D decoder is not initialized! Make sure "
                "set_models() "
                "is called.");
        }

        const float *floatInputArray =
            reinterpret_cast<const float *>(compressed_buffer);

        decoder_shape_3d_[0] = batch_size;

        if (double_to_float_buffer_3d_.size() <
            this->total_3d_pts_ * batch_size) {
            double_to_float_buffer_3d_.resize(this->total_3d_pts_ * batch_size);
        }

        Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(floatInputArray),
            n_outs_3d_encoder_ * batch_size, decoder_shape_3d_.data(),
            decoder_shape_3d_.size());
        const float *tensor_values = tensor_data.GetTensorData<float>();

        const char *input_names[]  = {decoder_input_name_3d_.c_str()};
        const char *output_names[] = {decoder_output_name_3d_.c_str()};

#if 1
        if constexpr (std::is_same_v<T, double>) {
            // output to the buffer array so we're not deleting it
            input_shape_3d_[0]       = batch_size;

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, double_to_float_buffer_3d_.data(),
                this->total_3d_pts_ * batch_size, input_shape_3d_.data(),
                input_shape_3d_.size());

            decoder_3d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, &output_tensor, 1);

            // now the data has been written to output tensor so we don't have
            // to worry about *more copies*, we just convert back to doubles
            std::transform(double_to_float_buffer_3d_.data(),
                           double_to_float_buffer_3d_.data() +
                               (this->total_3d_pts_ * batch_size),
                           output_array,
                           [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            input_shape_3d_[0]       = batch_size;

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, output_array, this->total_3d_pts_ * batch_size,
                input_shape_3d_.data(), input_shape_3d_.size());

            decoder_3d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, &output_tensor, 1);

            // here, we have the data just within output_tensor, which mapped
            // directly to our output array, so we are done
        }

#else
        auto output = decoder_3d_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &tensor_data, 1, output_names, 1);

        const float *output_data = output[0].GetTensorData<float>();

        if constexpr (std::is_same_v<T, double>) {
            std::transform(
                output_data, output_data + (this->total_3d_pts_ * batch_size),
                output_array, [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            std::memcpy(output_array, output_data,
                        this->total_3d_pts_ * sizeof(float) * batch_size);
        }
#endif
        return n_outs_3d_encoder_ * sizeof(float) * batch_size;
    }

    std::size_t do_compress_2d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, double>,
            "ONNX Compression only accepts doubles or floats as inputs!");

        if (!encoder_2d_) {
            throw std::runtime_error(
                "ERROR: The 2D encoder is not initialized! Make sure "
                "set_models() "
                "is called.");
        }

        input_shape_2d_[0] = batch_size;

        if (double_to_float_buffer_2d_.size() <
            this->total_2d_pts_ * batch_size) {
            double_to_float_buffer_2d_.resize(this->total_2d_pts_ * batch_size);
        }

        // tensor data can now come from this function
        Ort::Value tensor_data = createOnnxTensorFromData(
            original_matrix, this->total_2d_pts_, batch_size,
            double_to_float_buffer_2d_, memory_info_, input_shape_2d_);

        // const float* tensor_values = tensor_data.GetTensorData<float>();

        const char *input_names[]  = {input_name_2d_.c_str()};
        const char *output_names[] = {output_name_2d_.c_str()};

        auto output = encoder_2d_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &tensor_data, 1, output_names, 1);
        const float *output_data = output[0].GetTensorData<float>();

        std::memcpy(output_array, output_data,
                    n_outs_2d_encoder_ * sizeof(float) * batch_size);
        return n_outs_2d_encoder_ * sizeof(float) * batch_size;
    }

    std::size_t do_decompress_2d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        if (!decoder_2d_) {
            throw std::runtime_error(
                "ERROR: The 2D decoder is not initialized! Make sure "
                "set_models() "
                "is called.");
        }

        const float *floatInputArray =
            reinterpret_cast<const float *>(compressed_buffer);

        decoder_shape_2d_[0] = batch_size;

        if (double_to_float_buffer_2d_.size() <
            this->total_2d_pts_ * batch_size) {
            double_to_float_buffer_2d_.resize(this->total_2d_pts_ * batch_size);
        }

        Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(floatInputArray),
            n_outs_2d_encoder_ * batch_size, decoder_shape_2d_.data(),
            decoder_shape_2d_.size());
        const float *tensor_values = tensor_data.GetTensorData<float>();

        const char *input_names[]  = {decoder_input_name_2d_.c_str()};
        const char *output_names[] = {decoder_output_name_2d_.c_str()};

#if 1
        if constexpr (std::is_same_v<T, double>) {
            // output to the buffer array so we're not deleting it
            input_shape_2d_[0]       = batch_size;

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, double_to_float_buffer_2d_.data(),
                this->total_2d_pts_ * batch_size, input_shape_2d_.data(),
                input_shape_2d_.size());

            decoder_2d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, &output_tensor, 1);

            // now the data has been written to output tensor so we don't have
            // to worry about *more copies*, we just convert back to doubles
            std::transform(double_to_float_buffer_2d_.data(),
                           double_to_float_buffer_2d_.data() +
                               (this->total_2d_pts_ * batch_size),
                           output_array,
                           [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            input_shape_2d_[0]       = batch_size;

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, output_array, this->total_2d_pts_ * batch_size,
                input_shape_2d_.data(), input_shape_2d_.size());

            decoder_2d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, &output_tensor, 1);

            // here, we have the data just within output_tensor, which mapped
            // directly to our output array, so we are done
        }

#else
        auto output = decoder_2d_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &tensor_data, 1, output_names, 1);

        const float *output_data = output[0].GetTensorData<float>();

        // convert to doubles
        if constexpr (std::is_same_v<T, double>) {
            std::transform(
                output_data, output_data + (this->total_2d_pts_ * batch_size),
                output_array, [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            std::memcpy(output_array, output_data,
                        this->total_2d_pts_ * sizeof(float) * batch_size);
        }
#endif
        return n_outs_2d_encoder_ * sizeof(float) * batch_size;
    }

    std::size_t do_compress_1d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        if (!encoder_1d_) {
            throw std::runtime_error(
                "ERROR: The 1D encoder is not initialized! Make sure "
                "set_models() "
                "is called.");
        }

        input_shape_1d_[0] = batch_size;

        if (double_to_float_buffer_1d_.size() <
            this->total_1d_pts_ * batch_size) {
            double_to_float_buffer_1d_.resize(this->total_1d_pts_ * batch_size);
        }

        // convert to tensor data
        Ort::Value tensor_data = createOnnxTensorFromData(
            original_matrix, this->total_1d_pts_, batch_size,
            double_to_float_buffer_1d_, memory_info_, input_shape_1d_);

        // const float* tensor_values = tensor_data.GetTensorData<float>();

        const char *input_names[]  = {input_name_1d_.c_str()};
        const char *output_names[] = {output_name_1d_.c_str()};

        auto output = encoder_1d_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &tensor_data, 1, output_names, 1);
        const float *output_data = output[0].GetTensorData<float>();

        std::memcpy(output_array, output_data,
                    n_outs_1d_encoder_ * sizeof(float) * batch_size);
        return n_outs_1d_encoder_ * sizeof(float) * batch_size;
    }

    std::size_t do_decompress_1d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        if (!decoder_1d_) {
            throw std::runtime_error(
                "ERROR: The 1D decoder is not initialized! Make sure "
                "set_models() "
                "is called.");
        }

        const float *floatInputArray =
            reinterpret_cast<const float *>(compressed_buffer);

        decoder_shape_1d_[0] = batch_size;

        if (double_to_float_buffer_1d_.size() <
            this->total_1d_pts_ * batch_size) {
            double_to_float_buffer_1d_.resize(this->total_1d_pts_ * batch_size);
        }

        Ort::Value tensor_data = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float *>(floatInputArray),
            n_outs_1d_encoder_ * batch_size, decoder_shape_1d_.data(),
            decoder_shape_1d_.size());
        const float *tensor_values = tensor_data.GetTensorData<float>();

        const char *input_names[]  = {decoder_input_name_1d_.c_str()};
        const char *output_names[] = {decoder_output_name_1d_.c_str()};

#if 1
        if constexpr (std::is_same_v<T, double>) {
            // output to the buffer array so we're not deleting it
            input_shape_1d_[0]       = batch_size;

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, double_to_float_buffer_1d_.data(),
                this->total_1d_pts_ * batch_size, input_shape_1d_.data(),
                input_shape_1d_.size());

            decoder_1d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, &output_tensor, 1);

            // now the data has been written to output tensor so we don't have
            // to worry about *more copies*, we just convert back to doubles
            std::transform(double_to_float_buffer_1d_.data(),
                           double_to_float_buffer_1d_.data() +
                               (this->total_1d_pts_ * batch_size),
                           output_array,
                           [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            input_shape_1d_[0]       = batch_size;

            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, output_array, this->total_1d_pts_ * batch_size,
                input_shape_1d_.data(), input_shape_1d_.size());

            decoder_1d_->Run(Ort::RunOptions{nullptr}, input_names,
                             &tensor_data, 1, output_names, &output_tensor, 1);

            // here, we have the data just within output_tensor, which mapped
            // directly to our output array, so we are done
        }

#else
        auto output = decoder_1d_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &tensor_data, 1, output_names, 1);

        const float *output_data = output[0].GetTensorData<float>();

        // convert to doubles
        if constexpr (std::is_same_v<T, double>) {
            std::transform(
                output_data, output_data + (this->total_1d_pts_ * batch_size),
                output_array, [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            std::memcpy(output_array, output_data,
                        this->total_1d_pts_ * sizeof(float) * batch_size);
        }
#endif
        return n_outs_1d_encoder_ * sizeof(float) * batch_size;
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
