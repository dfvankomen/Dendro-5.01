#pragma once

#include <cstring>
#include <memory>

#include "compression_base.hpp"

namespace dendrocompression {

template <typename T>
class DummyCompressor : public Compression<T> {
   public:
    DummyCompressor(unsigned int ele_order, unsigned int num_vars)
        : Compression<T>(ele_order, num_vars) {}

    ~DummyCompressor() = default;

    std::unique_ptr<Compression<T>> clone() const override {
        return std::make_unique<DummyCompressor>(*this);
    }

    CompressionType get_compression_type() const override {
        return CompressionType::COMP_DUMMY;
    }

    std::string to_string() const override { return "DummyCompressor"; }

    std::size_t do_compress_3d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        std::memcpy(output_array, original_matrix,
                    batch_size * this->total_3d_bytes_);
        return batch_size * this->total_3d_bytes_;
    }

    std::size_t do_decompress_3d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        std::memcpy(output_array, compressed_buffer,
                    batch_size * this->total_3d_bytes_);
        return batch_size * this->total_3d_bytes_;
    }

    std::size_t do_compress_2d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        std::memcpy(output_array, original_matrix,
                    batch_size * this->total_2d_bytes_);
        return batch_size * this->total_2d_bytes_;
    }

    std::size_t do_decompress_2d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        std::memcpy(output_array, compressed_buffer,
                    batch_size * this->total_2d_bytes_);
        return batch_size * this->total_2d_bytes_;
    }

    std::size_t do_compress_1d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        std::memcpy(output_array, original_matrix,
                    batch_size * this->total_1d_bytes_);
        return batch_size * this->total_1d_bytes_;
    }

    std::size_t do_decompress_1d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        std::memcpy(output_array, compressed_buffer,
                    batch_size * this->total_1d_bytes_);
        return batch_size * this->total_1d_bytes_;
    }

    std::size_t do_compress_0d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        std::memcpy(output_array, original_matrix,
                    batch_size * this->total_0d_bytes_);
        return batch_size * this->total_0d_bytes_;
    }

    std::size_t do_decompress_0d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        std::memcpy(output_array, compressed_buffer,
                    batch_size * this->total_0d_bytes_);
        return batch_size * this->total_0d_bytes_;
    }

    std::size_t do_compress_flat(T* const original_matrix,
                                 unsigned char* const output_array,
                                 unsigned int n_pts) override {
        std::memcpy(output_array, original_matrix,
                    n_pts * this->total_0d_bytes_);
        return n_pts * this->total_0d_bytes_;
    }

    std::size_t do_decompress_flat(unsigned char* const compressed_buffer,
                                   T* const output_array,
                                   unsigned int n_pts) override {
        std::memcpy(output_array, compressed_buffer,
                    n_pts * this->total_0d_bytes_);
        return n_pts * this->total_0d_bytes_;
    }
};

}  // namespace dendrocompression
