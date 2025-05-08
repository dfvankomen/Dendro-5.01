#pragma once

#include <torch/script.h>

#include <cassert>

#include "compression_base.hpp"

namespace dendrocompression {

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

template <typename T>
class TorchScriptCompressor : public Compression<T> {
   private:
    std::string encoder_3d_path_;
    std::string decoder_3d_path_;
    std::string encoder_2d_path_;
    std::string decoder_2d_path_;
    std::string encoder_1d_path_;
    std::string decoder_1d_path_;
    std::string encoder_0d_path_;
    std::string decoder_0d_path_;

    torch::jit::Module encoder_3d_;
    torch::jit::Module decoder_3d_;
    torch::jit::Module encoder_2d_;
    torch::jit::Module decoder_2d_;
    torch::jit::Module encoder_1d_;
    torch::jit::Module decoder_1d_;
    torch::jit::Module encoder_0d_;
    torch::jit::Module decoder_0d_;

    unsigned int n_outs_3d_encoder_;
    unsigned int n_outs_2d_encoder_;
    unsigned int n_outs_1d_encoder_;
    unsigned int n_outs_0d_encoder_;

    void load_models() {
        // then attempt to load 3D
        try {
            encoder_3d_ = torch::jit::load(encoder_3d_path_);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 3d encoder model! - attempted "
                         "to load: "
                      << encoder_3d_path_ << std::endl;
            exit(-1);
        }
        try {
            decoder_3d_ = torch::jit::load(decoder_3d_path_);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 3d decoder model! - attempted "
                         "to load: "
                      << decoder_3d_path_ << std::endl;
            exit(-1);
        }

        // 2D
        try {
            encoder_2d_ = torch::jit::load(encoder_2d_path_);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 2d encoder model! - attempted "
                         "to load: "
                      << encoder_2d_path_ << std::endl;
            exit(-1);
        }
        try {
            decoder_2d_ = torch::jit::load(decoder_2d_path_);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 2d decoder model! - attempted "
                         "to load: "
                      << decoder_2d_path_ << std::endl;
            exit(-1);
        }

        // 1D
        try {
            encoder_1d_ = torch::jit::load(encoder_1d_path_);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 1d encoder model! - attempted "
                         "to load: "
                      << encoder_1d_path_ << std::endl;
            exit(-1);
        }
        try {
            decoder_1d_ = torch::jit::load(decoder_1d_path_);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading 1d encoder model! - attempted "
                         "to load: "
                      << decoder_1d_path_ << std::endl;
            exit(-1);
        }

        // TODO: 0D, will need checks for it

        // now we can do quick checks on the output of the
        // encoder/decoder pairs
        std::vector<float> test_data(this->total_3d_pts_, 1.0);
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor output;
        torch::Tensor input_tensor;

        input_tensor = convertDataToModelType(test_data, "float");
        input_tensor =
            reshapeTensor3DBlock(input_tensor, this->num_vars_, this->n_, 1);
        inputs.push_back(input_tensor);

        try {
            output = encoder_3d_.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            // NOTE: torch throws a runtime_error if there's ever a
            // mismatch, anything else we don't want to handle
            std::cerr << "Error when attempting to run the 3d encoder, it's "
                         "possible the input size is incorrect!\n";
            exit(-1);
        }

        // index 0 is batch size, so we'll store this!
        n_outs_3d_encoder_ = output.sizes()[1];

        inputs.clear();
        inputs.push_back(output);

        try {
            output = decoder_3d_.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            std::cerr << "Error when attempting to run the 3d decoder on "
                         "initialization, it's "
                         "possible the input size is incorrect or it doesn't "
                         "match with the encoder!\n";
            exit(-1);
        }
        // if we were successful, we've at least got matching data

        // NOW CHECK 2D
        test_data.resize(this->total_2d_pts_);
        input_tensor = convertDataToModelType(test_data, "float");
        input_tensor =
            reshapeTensor2DBlock(input_tensor, this->num_vars_, this->n_, 1);
        inputs.clear();
        inputs.push_back(input_tensor);
        try {
            output = encoder_2d_.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            // NOTE: torch throws a runtime_error if there's ever a
            // mismatch, anything else we don't want to handle
            std::cerr << "Error when attempting to run the 2d encoder, it's "
                         "possible the input size is incorrect!\n";
            exit(-1);
        }

        // index 0 is batch size, so we'll store this!
        n_outs_2d_encoder_ = output.sizes()[1];

        inputs.clear();
        inputs.push_back(output);

        try {
            output = decoder_2d_.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            std::cerr << "Error when attempting to run the 2d decoder on "
                         "initialization, it's "
                         "possible the input size is incorrect or it doesn't "
                         "match with the encoder!\n";
            exit(-1);
        }

        // NOW CHECK 1D
        test_data.resize(this->total_1d_pts_);
        input_tensor = convertDataToModelType(test_data, "float");
        input_tensor =
            reshapeTensor1DBlock(input_tensor, this->num_vars_, this->n_, 1);
        inputs.clear();
        inputs.push_back(input_tensor);
        try {
            output = encoder_1d_.forward(inputs).toTensor();
        } catch (const std::runtime_error &e) {
            // NOTE: torch throws a runtime_error if there's ever a
            // mismatch, anything else we don't want to handle
            std::cerr << "Error when attempting to run the 1d encoder, it's "
                         "possible the input size is incorrect!\n";
            exit(-1);
        }

        // index 0 is batch size, so we'll store this!
        n_outs_1d_encoder_ = output.sizes()[1];

        inputs.clear();
        inputs.push_back(output);

        try {
            output = decoder_1d_.forward(inputs).toTensor();
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
    }

   public:
    TorchScriptCompressor(
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
          decoder_0d_path_(encoder_0d_path) {
        load_models();
    }

    ~TorchScriptCompressor() = default;

    std::unique_ptr<Compression<T>> clone() const override {
        return std::make_unique<TorchScriptCompressor>(*this);
    }

    CompressionType get_compression_type() const override {
        return CompressionType::COMP_TORCH_SCRIPT;
    }

    std::string to_string() const override { return "TorchScriptCompressor"; }

    std::size_t do_compress_3d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        // start by creating the vector size
        torch::Tensor input_data = convertDataToModelType(
            original_matrix, this->total_3d_pts_, batch_size, "float");

        input_data = reshapeTensor3DBlock(input_data, this->num_vars_, this->n_,
                                          batch_size);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_data);

        torch::Tensor output;
        try {
            output = encoder_3d_.forward(inputs).toTensor();
        } catch (const std::exception &e) {
            std::cerr << "Error during forward pass on 3d encoder: " << e.what()
                      << std::endl;
            exit(-1);
        }

        if (!output.is_contiguous()) {
            output = output.contiguous();
        }

        auto size = output.numel();

        if (size != n_outs_3d_encoder_ * batch_size) {
            std::cerr
                << "ERROR: Mismatch on 3d AutoEncoder output sizes: expected "
                << n_outs_3d_encoder_ << ", got " << size << std::endl;
            exit(-1);
        }

        float *floatOutputArray = reinterpret_cast<float *>(output_array);
        std::memcpy(floatOutputArray, output.data_ptr<float>(),
                    size * sizeof(float));

        return sizeof(float) * n_outs_3d_encoder_ * batch_size;
    }

    std::size_t do_decompress_3d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        // simple reinterpret cast
        float *floatInputArray   = reinterpret_cast<float *>(compressed_buffer);

        torch::Tensor input_data = convertDataToModelType(
            floatInputArray, n_outs_3d_encoder_, batch_size, "float");

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_data);

        torch::Tensor output;
        try {
            output = decoder_3d_.forward(inputs).toTensor();
        } catch (const std::exception &e) {
            std::cerr << "Error during forward pass in 3d decompression: "
                      << e.what() << std::endl;
            exit(-1);
        }

        if (!output.is_contiguous()) {
            output = output.contiguous();
        }

        auto size = output.numel();

        if (size != this->total_3d_pts_ * batch_size) {
            std::cerr << "ERROR: Mismatch on 3D Decoder output sizes: expected "
                      << this->total_3d_pts_ << ", got " << size << std::endl;
            exit(-1);
        }

        if constexpr (std::is_same_v<T, double>) {
            // then do a transform to our double outputs
            std::transform(
                output.data_ptr<float>(),
                output.data_ptr<float>() + this->total_3d_pts_ * batch_size,
                output_array, [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            // or just copy if the output array expects floats
            std::memcpy(output_array, output.data_ptr<float>(),
                        this->total_3d_pts_ * sizeof(float) * batch_size);
        }

        return sizeof(float) * n_outs_3d_encoder_ * batch_size;
    }

    std::size_t do_compress_2d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        // start by creating the vector size
        torch::Tensor input_data = convertDataToModelType(
            original_matrix, this->total_2d_pts_, batch_size, "float");

        input_data = reshapeTensor2DBlock(input_data, this->num_vars_, this->n_,
                                          batch_size);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_data);

        torch::Tensor output;
        try {
            output = encoder_2d_.forward(inputs).toTensor();
        } catch (const std::exception &e) {
            std::cerr << "Error during forward pass on 2d encoder: " << e.what()
                      << std::endl;
            exit(-1);
        }

        if (!output.is_contiguous()) {
            output = output.contiguous();
        }

        auto size = output.numel();

        if (size != n_outs_2d_encoder_ * batch_size) {
            std::cerr
                << "ERROR: Mismatch on 2d AutoEncoder output sizes: expected "
                << n_outs_2d_encoder_ << ", got " << size << std::endl;
            exit(-1);
        }

        float *floatOutputArray = reinterpret_cast<float *>(output_array);
        std::memcpy(floatOutputArray, output.data_ptr<float>(),
                    size * sizeof(float));

        return sizeof(float) * n_outs_2d_encoder_ * batch_size;
    }

    std::size_t do_decompress_2d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        // simple reinterpret cast
        float *floatInputArray   = reinterpret_cast<float *>(compressed_buffer);

        torch::Tensor input_data = convertDataToModelType(
            floatInputArray, n_outs_2d_encoder_, batch_size, "float");

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_data);

        torch::Tensor output;
        try {
            output = decoder_2d_.forward(inputs).toTensor();
        } catch (const std::exception &e) {
            std::cerr << "Error during forward pass in 2d decompression: "
                      << e.what() << std::endl;
            exit(-1);
        }

        if (!output.is_contiguous()) {
            output = output.contiguous();
        }

        auto size = output.numel();

        if (size != this->total_2d_pts_ * batch_size) {
            std::cerr << "ERROR: Mismatch on 2D Decoder output sizes: expected "
                      << this->total_2d_pts_ << ", got " << size << std::endl;
            exit(-1);
        }

        if constexpr (std::is_same_v<T, double>) {
            // then do a transform to our double outputs
            std::transform(
                output.data_ptr<float>(),
                output.data_ptr<float>() + this->total_2d_pts_ * batch_size,
                output_array, [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            // or just copy if the output array expects floats
            std::memcpy(output_array, output.data_ptr<float>(),
                        this->total_2d_pts_ * sizeof(float) * batch_size);
        }

        return sizeof(float) * n_outs_2d_encoder_ * batch_size;
    }

    std::size_t do_compress_1d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) override {
        // start by creating the vector size
        torch::Tensor input_data = convertDataToModelType(
            original_matrix, this->total_1d_pts_, batch_size, "float");

        input_data = reshapeTensor1DBlock(input_data, this->num_vars_, this->n_,
                                          batch_size);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_data);

        torch::Tensor output;
        try {
            output = encoder_1d_.forward(inputs).toTensor();
        } catch (const std::exception &e) {
            std::cerr << "Error during forward pass on 1d encoder: " << e.what()
                      << std::endl;
            exit(-1);
        }

        if (!output.is_contiguous()) {
            output = output.contiguous();
        }

        auto size = output.numel();

        if (size != n_outs_1d_encoder_ * batch_size) {
            std::cerr
                << "ERROR: Mismatch on 1d AutoEncoder output sizes: expected "
                << n_outs_1d_encoder_ << ", got " << size << std::endl;
            exit(-1);
        }

        float *floatOutputArray = reinterpret_cast<float *>(output_array);
        std::memcpy(floatOutputArray, output.data_ptr<float>(),
                    size * sizeof(float));

        return sizeof(float) * n_outs_1d_encoder_ * batch_size;
    }

    std::size_t do_decompress_1d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) override {
        // simple reinterpret cast
        float *floatInputArray   = reinterpret_cast<float *>(compressed_buffer);

        torch::Tensor input_data = convertDataToModelType(
            floatInputArray, n_outs_1d_encoder_, batch_size, "float");

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_data);

        torch::Tensor output;
        try {
            output = decoder_1d_.forward(inputs).toTensor();
        } catch (const std::exception &e) {
            std::cerr << "Error during forward pass in 1d decompression: "
                      << e.what() << std::endl;
            exit(-1);
        }

        if (!output.is_contiguous()) {
            output = output.contiguous();
        }

        auto size = output.numel();

        if (size != this->total_1d_pts_ * batch_size) {
            std::cerr << "ERROR: Mismatch on 1D Decoder output sizes: expected "
                      << this->total_1d_pts_ << ", got " << size << std::endl;
            exit(-1);
        }

        if constexpr (std::is_same_v<T, double>) {
            // then do a transform to our double outputs
            std::transform(
                output.data_ptr<float>(),
                output.data_ptr<float>() + this->total_1d_pts_ * batch_size,
                output_array, [](float d) { return static_cast<double>(d); });
        } else if constexpr (std::is_same_v<T, float>) {
            // or just copy if the output array expects floats
            std::memcpy(output_array, output.data_ptr<float>(),
                        this->total_1d_pts_ * sizeof(float) * batch_size);
        }

        return sizeof(float) * n_outs_1d_encoder_ * batch_size;
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
        std::memcpy(output_array, original_matrix,
                    n_pts * this->total_0d_bytes_);
        return n_pts * this->total_0d_bytes_;
    }

    std::size_t do_decompress_flat(unsigned char *const compressed_buffer,
                                   T *const output_array,
                                   unsigned int n_pts) override {
        std::memcpy(output_array, compressed_buffer,
                    n_pts * this->total_0d_bytes_);
        return n_pts * this->total_0d_bytes_;
    }
};

}  // namespace dendrocompression
