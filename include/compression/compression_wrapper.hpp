#pragma once
#include <memory>

#include "compression_base.hpp"
#include "compression_interface.hpp"

namespace dendrocompression {

template <typename T>
class CompressionWrapper : public CompressionInterface {
    std::unique_ptr<Compression<T>> compressor_;

   public:
    CompressionWrapper(std::unique_ptr<Compression<T>> compressor)
        : compressor_(std::move(compressor)) {}

    std::size_t do_compress_3d(void *const original_matrix,
                               void *const output_array,
                               unsigned int batch_size) override {
        return compressor_->do_compress_3d(
            static_cast<T *const>(original_matrix),
            static_cast<unsigned char *const>(output_array), batch_size);
    }
    std::size_t do_compress_2d(void *const original_matrix,
                               void *const output_array,
                               unsigned int batch_size) override {
        return compressor_->do_compress_2d(
            static_cast<T *const>(original_matrix),
            static_cast<unsigned char *const>(output_array), batch_size);
    }
    std::size_t do_compress_1d(void *const original_matrix,
                               void *const output_array,
                               unsigned int batch_size) override {
        return compressor_->do_compress_1d(
            static_cast<T *const>(original_matrix),
            static_cast<unsigned char *const>(output_array), batch_size);
    }
    std::size_t do_compress_0d(void *const original_matrix,
                               void *const output_array,
                               unsigned int batch_size) override {
        return compressor_->do_compress_0d(
            static_cast<T *const>(original_matrix),
            static_cast<unsigned char *const>(output_array), batch_size);
    }
    std::size_t do_compress_flat(void *const original_matrix,
                                 void *const output_array,
                                 unsigned int n_pts) override {
        return compressor_->do_compress_flat(
            static_cast<T *const>(original_matrix),
            static_cast<unsigned char *const>(output_array), n_pts);
    }

    std::size_t do_decompress_3d(void *const compressed_buffer,
                                 void *const output_array,
                                 unsigned int batch_size) override {
        return compressor_->do_decompress_3d(
            static_cast<unsigned char *const>(compressed_buffer),
            static_cast<T *const>(output_array), batch_size);
    }
    std::size_t do_decompress_2d(void *const compressed_buffer,
                                 void *const output_array,
                                 unsigned int batch_size) override {
        return compressor_->do_decompress_2d(
            static_cast<unsigned char *const>(compressed_buffer),
            static_cast<T *const>(output_array), batch_size);
    }
    std::size_t do_decompress_1d(void *const compressed_buffer,
                                 void *const output_array,
                                 unsigned int batch_size) override {
        return compressor_->do_decompress_1d(
            static_cast<unsigned char *const>(compressed_buffer),
            static_cast<T *const>(output_array), batch_size);
    }
    std::size_t do_decompress_0d(void *const compressed_buffer,
                                 void *const output_array,
                                 unsigned int batch_size) override {
        return compressor_->do_decompress_0d(
            static_cast<unsigned char *const>(compressed_buffer),
            static_cast<T *const>(output_array), batch_size);
    }
    std::size_t do_decompress_flat(void *const compressed_buffer,
                                   void *const output_array,
                                   unsigned int n_pts) override {
        return compressor_->do_decompress_flat(
            static_cast<unsigned char *const>(compressed_buffer),
            static_cast<T *const>(output_array), n_pts);
    }
};

}  // namespace dendrocompression
