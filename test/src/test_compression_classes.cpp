#include <stdexcept>
#include <type_traits>

#include "compression.h"
#include "compression_base.hpp"
#include "compression_factory.hpp"
#include "compression_interface.hpp"
#include "compression_wrapper.hpp"

// Forward transform: 5-point lifting (in-place)
template <typename T>
void lift5Forward(std::vector<T> &x) {
    // Predict step (remove correlations)
    for (int i = 1; i < 4; i++)
        x[i] -= 0.5f * (x[i - 1] + x[i + 1]);  // Linear prediction

    // Update step (preserve average)
    T avg = 0.2f * (x[0] + x[1] + x[2] + x[3] + x[4]);
    for (T &v : x) v -= avg;
}

// Inverse transform: Reconstruct original 5 points
template <typename T>
void lift5Inverse(std::vector<T> &x) {
    // Undo update step
    T avg = 0.2f * (x[0] + x[1] + x[2] + x[3] + x[4]);
    for (T &v : x) v += avg;

    // Undo predict step
    for (int i = 3; i >= 1; i--) x[i] += 0.5f * (x[i - 1] + x[i + 1]);
}

template <typename T>
void forward5(std::vector<T> &x) {
    // Predict: Remove correlations (linear interpolation)
    T d0  = x[1] - x[0];
    T d1  = x[2] - (x[1] + x[0]) / 2;
    T d2  = x[3] - (x[2] + x[1]) / 2;
    T d3  = x[4] - (x[3] + x[2]) / 2;

    // Update: Preserve average
    T avg = (x[0] + x[1] + x[2] + x[3] + x[4]) / 5.0f;
    x[0]  = avg;  // Store average in x[0]
    x[1]  = d0;   // Details in x[1..4]
    x[2]  = d1;
    x[3]  = d2;
    x[4]  = d3;
}

template <typename T>
void inverse5(std::vector<T> &x) {
    T avg = x[0];  // Recover average
    T d0 = x[1], d1 = x[2], d2 = x[3], d3 = x[4];

    // Reconstruct original values
    x[0] = avg - (2 * d0 + 1.5f * d1 + d2 + 0.5f * d3) / 5.0f;
    x[1] = x[0] + d0;
    x[2] = (x[1] + x[0]) / 2 + d1;
    x[3] = (x[2] + x[1]) / 2 + d2;
    x[4] = (x[3] + x[2]) / 2 + d3;
}

template <typename T>
void threshold_wavelets(std::vector<T> &x, T tol) {
    for (int i = 1; i < 5; i++)  // Only threshold details (x[1..4])
        if (std::abs(x[i]) < tol) x[i] = 0.0f;
}

template <typename T>
void threshold_wavelets_quant(std::vector<T> &x, T tol) {
    for (auto &coeff : x) {
        T quantized = (double)(round(coeff / tol)) * tol;
        T error     = coeff - quantized;
        coeff       = quantized;
    }
}

template <typename T>
void smartQuantize(std::vector<T> &coeffs, int bits) {
    // Find max magnitude in block
    T max_val = 0.0f;
    for (T c : coeffs)
        if (std::abs(c) > max_val) max_val = std::abs(c);

    if (max_val == 0.0f) return;  // All zeros

    // Quantize to [-2^(bits-1), 2^(bits-1)-1]
    int max_q = (1 << (bits - 1)) - 1;
    T scale   = max_val / max_q;
    std::cout << "MAX Q: " << max_q << " SCALE: " << scale << std::endl;

    for (T &c : coeffs) {
        int q = static_cast<int>(std::round(c / scale));
        c     = static_cast<T>(q) * scale;  // Dequantize
    }
}

template <typename T>
void thresholdQuantize(std::vector<T> &coeffs, T tol, int bits) {
    T max_val = *std::max_element(coeffs.begin(), coeffs.end(), [](T a, T b) {
        return std::abs(a) < std::abs(b);
    });

    T scale   = (max_val - tol) / ((1 << (bits - 1)) - 1);

    for (T &c : coeffs) {
        if (std::abs(c) < tol) {
            c = 0.0f;  // Dead zone
        } else {
            int q = static_cast<int>(std::round(c / scale));
            c     = static_cast<T>(q) * scale;
        }
    }
}

int main() {
    // typedef float comptype;
    typedef double comptype;

    dendrocompression::register_compressors();

    std::unique_ptr<dendrocompression::CompressionInterface> compressor;

    const dendrocompression::CompressionType compressor_type =
        dendrocompression::CompressionType::COMP_ONNX_MODEL;

    std::string zfp_type   = "precision";
    double zfp_parameter   = 1e-3;

    unsigned int ele_order = 6;
    unsigned int num_vars  = 1;
    unsigned int npts_dim  = ele_order - 1;

    double x_start         = 0;
    double x_end           = 1.0;

    double dt              = (x_end - x_start) / (npts_dim - 1);

    std::vector<comptype> input_data(npts_dim * npts_dim * npts_dim);

    for (unsigned int k = 0; k < npts_dim; ++k) {
        double z = k * dt;
        for (unsigned int j = 0; j < npts_dim; ++j) {
            double y = k * dt;
            for (unsigned int i = 0; i < npts_dim; ++i) {
                double x                                      = k * dt;
                input_data[i + npts_dim * (j + npts_dim * k)] = x * y * z;
            }
        }
    }

    size_t num_bytes = input_data.size() * sizeof(comptype);

    std::cout << "ARRAY DATA: ";
    for (auto val : input_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "ORIGINAL NBYTES: " << num_bytes << std::endl;

    if (compressor_type == dendrocompression::CompressionType::COMP_ZFP) {
        if (std::is_same_v<double, comptype>) {
            compressor =
                std::make_unique<dendrocompression::CompressionWrapper<double>>(
                    dendrocompression::doubleCompressor.create(
                        compressor_type,
                        {ele_order, num_vars, zfp_type, zfp_parameter}));
        } else {
            compressor =
                std::make_unique<dendrocompression::CompressionWrapper<float>>(
                    dendrocompression::floatCompressor.create(
                        compressor_type,
                        {ele_order, num_vars, zfp_type, zfp_parameter}));
        }
    } else if (compressor_type ==
               dendrocompression::CompressionType::COMP_DUMMY) {
        if (std::is_same_v<double, comptype>) {
            compressor =
                std::make_unique<dendrocompression::CompressionWrapper<double>>(
                    dendrocompression::doubleCompressor.create(
                        compressor_type, {ele_order, num_vars}));
        } else {
            compressor =
                std::make_unique<dendrocompression::CompressionWrapper<float>>(
                    dendrocompression::floatCompressor.create(
                        compressor_type, {ele_order, num_vars}));
        }
    } else if (compressor_type ==
               dendrocompression::CompressionType::COMP_TORCH_SCRIPT) {
        if (std::is_same_v<double, comptype>) {
            compressor =
                std::make_unique<dendrocompression::CompressionWrapper<double>>(
                    dendrocompression::doubleCompressor.create(
                        compressor_type,
                        {
                            ele_order,
                            num_vars,
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_3d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_3d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_2d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_2d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_1d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_1d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_0d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_0d.pt"),
                        }));
        } else {
            compressor =
                std::make_unique<dendrocompression::CompressionWrapper<float>>(
                    dendrocompression::floatCompressor.create(
                        compressor_type,
                        {
                            ele_order,
                            num_vars,
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_3d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_3d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_2d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_2d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_1d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_1d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "ENCODER_singledim_equal_in_out_0d.pt"),
                            std::string("/home/denv/research/notebooks/"
                                        "DECODER_singledim_equal_in_out_0d.pt"),
                        }));
        }
    } else if (compressor_type ==
               dendrocompression::CompressionType::COMP_ONNX_MODEL) {
        if (std::is_same_v<double, comptype>) {
            compressor = std::make_unique<
                dendrocompression::CompressionWrapper<double>>(
                dendrocompression::doubleCompressor.create(
                    compressor_type,
                    {
                        ele_order,
                        num_vars,
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_3d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_3d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_2d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_2d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_1d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_1d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_0d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_0d.onnx"),
                    }));
        } else {
            compressor = std::make_unique<
                dendrocompression::CompressionWrapper<float>>(
                dendrocompression::floatCompressor.create(
                    compressor_type,
                    {
                        ele_order,
                        num_vars,
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_3d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_3d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_2d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_2d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_1d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_1d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "ENCODER_singledim_equal_in_out_0d.onnx"),
                        std::string("/home/denv/research/notebooks/"
                                    "DECODER_singledim_equal_in_out_0d.onnx"),
                    }));
        }
    } else {
        throw std::runtime_error(
            "This type of compressor isn't available yet!");
    }

    std::vector<comptype> compressed(input_data.size() * 2);
    std::vector<comptype> output_data(input_data.size(), 10000000.0);

    std::size_t nbytes = compressor->do_compress_3d(
        input_data.data(), reinterpret_cast<unsigned char *>(compressed.data()),
        1);

    std::size_t nbytes_out = compressor->do_decompress_3d(
        reinterpret_cast<unsigned char *>(compressed.data()),
        output_data.data(), 1);

    std::cout << "NBYTES/NBYTES_OUT: " << nbytes << "/" << nbytes_out
              << "   compession ratio: "
              << (comptype)num_bytes / (comptype)nbytes << std::endl;

    std::cout << "ARRAY DATA (DECOMPRESSED): ";
    for (auto val : output_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    comptype mse = dendro_compress::calculate_mse(
        input_data.data(), output_data.data(), input_data.size());

    std::cout << "MSE: " << mse << std::endl;

    nbytes = compressor->do_compress_2d(
        input_data.data(), reinterpret_cast<unsigned char *>(compressed.data()),
        4);

    nbytes_out = compressor->do_decompress_2d(
        reinterpret_cast<unsigned char *>(compressed.data()),
        output_data.data(), 4);

    std::cout << "2d: NBYTES/NBYTES_OUT: " << nbytes << "/" << nbytes_out
              << "   compession ratio: "
              << (comptype)num_bytes / (comptype)nbytes << std::endl;

    mse = dendro_compress::calculate_mse(input_data.data(), output_data.data(),
                                         input_data.size());

    std::cout << "2d: MSE: " << mse << std::endl;
    std::cout << "2d: ARRAY DATA (DECOMPRESSED): ";
    for (auto val : output_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 1d slices
    nbytes = compressor->do_compress_1d(
        input_data.data(), reinterpret_cast<unsigned char *>(compressed.data()),
        4 * 4);

    nbytes_out = compressor->do_decompress_1d(
        reinterpret_cast<unsigned char *>(compressed.data()),
        output_data.data(), 4 * 4);

    std::cout << "1d: NBYTES/NBYTES_OUT: " << nbytes << "/" << nbytes_out
              << "   compession ratio: "
              << (comptype)num_bytes / (comptype)nbytes << std::endl;

    mse = dendro_compress::calculate_mse(input_data.data(), output_data.data(),
                                         input_data.size());

    std::cout << "1d: MSE: " << mse << std::endl;
    std::cout << "1d: ARRAY DATA (DECOMPRESSED): ";
    for (auto val : output_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 0d slices
    nbytes = compressor->do_compress_0d(
        input_data.data(), reinterpret_cast<unsigned char *>(compressed.data()),
        4 * 4 * 4);

    nbytes_out = compressor->do_decompress_0d(
        reinterpret_cast<unsigned char *>(compressed.data()),
        output_data.data(), 4 * 4 * 4);

    std::cout << "0d: NBYTES/NBYTES_OUT: " << nbytes << "/" << nbytes_out
              << "   compession ratio: "
              << (comptype)num_bytes / (comptype)nbytes << std::endl;

    mse = dendro_compress::calculate_mse(input_data.data(), output_data.data(),
                                         input_data.size());

    std::cout << "0d: MSE: " << mse << std::endl;
    std::cout << "0d: ARRAY DATA (DECOMPRESSED): ";
    for (auto val : output_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 1d wavelet transform
    std::vector<comptype> test_vector(npts_dim);
    for (unsigned int i = 0; i < npts_dim; ++i) {
        double x       = i * dt;
        test_vector[i] = sin(x * 0.1);
    }
    // copy to truth
    std::vector<comptype> truth(test_vector);

    std::cout << std::endl << std::endl << "wavelet input: ";
    for (auto &i : test_vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    std::vector<comptype> temp(npts_dim);
    int n = npts_dim;
    while (n > 1) {
        int half = n / 2;
        for (int i = 0; i < half; i++) {
            temp[i] =
                (test_vector[2 * i] + test_vector[2 * i + 1]) / sqrt(2.0f);
            temp[half + i] =
                (test_vector[2 * i] - test_vector[2 * i + 1]) / sqrt(2.0f);
        }
        std::copy(temp.begin(), temp.begin() + n, test_vector.begin());
        n = half;  // process the next level
    }

    std::cout << "wavelet coeffs: ";
    for (auto &i : test_vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // undo wavelet stuff
    n = 2;
    while (n <= npts_dim) {
        std::cout << "n: " << n << std::endl;
        int half = n / 2;
        for (int i = 0; i < half; i++) {
            temp[2 * i] = (test_vector[2 * i] =
                               (test_vector[i] + test_vector[half + i])) /
                          sqrt(2.0f);
            temp[2 * i + 1] = (test_vector[2 * i + 1] =
                                   (test_vector[i] - test_vector[half + i])) /
                              sqrt(2.0f);
        }
        std::copy(temp.begin(), temp.end(), test_vector.begin());
        n *= 2;
    }

    std::cout << "reconstructed array: ";
    for (auto &i : test_vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // try the custom lifting wavelet
    test_vector = truth;
    std::cout << "truth (again): ";
    for (auto &i : test_vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    forward5(test_vector);
    std::cout << "wavelet coeffs: ";
    for (auto &i : test_vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    thresholdQuantize(test_vector, 0.0025, 4);
    std::cout << "wavelet coeffs (threshold): ";
    for (auto &i : test_vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    inverse5(test_vector);
    std::cout << "reconstructed array: ";
    for (auto &i : test_vector) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}
