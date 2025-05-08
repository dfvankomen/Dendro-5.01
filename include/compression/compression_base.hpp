#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

namespace dendrocompression {

enum CompressionType {
    COMP_NONE = 0,
    COMP_DUMMY,
    COMP_ZFP,
    COMP_CHEBYSHEV,
    COMP_BLOSC,
    COMP_TORCH_SCRIPT,
    COMP_ONNX_MODEL
};

template <typename T>
class Compression {
   protected:
    unsigned int n_;
    unsigned int ele_order_;

    unsigned int num_vars_;

    unsigned int total_3d_pts_;
    unsigned int total_2d_pts_;
    unsigned int total_1d_pts_;
    unsigned int total_0d_pts_;

    std::size_t total_3d_bytes_;
    std::size_t total_2d_bytes_;
    std::size_t total_1d_bytes_;
    std::size_t total_0d_bytes_;

    bool is_multivar_ = false;

    /**
     * @brief Protected constructor to initialize a Compression object
     * @param ele_order The element order (i.e. 6)
     * @param num_vars The number of variables per block (i.e. 1)
     */
    Compression(unsigned int ele_order, unsigned int num_vars)
        : ele_order_(ele_order), num_vars_(num_vars) {
        // compared to regular blocks, n_ is eleorder - 1, not eleorder * 2 + 1
        n_              = ele_order_ - 1;

        total_3d_pts_   = n_ * n_ * n_ * num_vars_;
        total_2d_pts_   = n_ * n_ * num_vars_;
        total_1d_pts_   = n_ * num_vars_;
        total_0d_pts_   = 1 * num_vars_;

        total_3d_bytes_ = total_3d_pts_ * sizeof(T);
        total_2d_bytes_ = total_2d_pts_ * sizeof(T);
        total_1d_bytes_ = total_1d_pts_ * sizeof(T);
        total_0d_bytes_ = total_0d_pts_ * sizeof(T);

        if (num_vars_ > 1) {
            is_multivar_ = true;
        } else {
            is_multivar_ = false;
        }
    }

    /**
     * @brief Copy constructor
     * @param obj The Compression object to copy from
     */
    Compression(const Compression &obj) {};

   public:
    virtual ~Compression() {};

    virtual std::unique_ptr<Compression> clone() const            = 0;

    virtual std::size_t do_compress_3d(T *const original_matrix,
                                       unsigned char *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_2d(T *const original_matrix,
                                       unsigned char *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_1d(T *const original_matrix,
                                       unsigned char *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_0d(T *const original_matrix,
                                       unsigned char *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_flat(T *const original_matrix,
                                         unsigned char *const output_array,
                                         unsigned int n_pts)      = 0;

    virtual std::size_t do_decompress_3d(unsigned char *const compressed_buffer,
                                         T *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_2d(unsigned char *const compressed_buffer,
                                         T *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_1d(unsigned char *const compressed_buffer,
                                         T *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_0d(unsigned char *const compressed_buffer,
                                         T *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_flat(
        unsigned char *const compressed_buffer, T *const output_array,
        unsigned int n_pts)                              = 0;

    /**
     * @brief Get the type of compressor being used
     * @return The compression type
     */
    virtual CompressionType get_compression_type() const = 0;

    virtual std::string to_string() const                = 0;
};

}  // namespace dendrocompression
