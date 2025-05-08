#pragma once

#include <cstddef>
namespace dendrocompression {
/**
 *
 *
 */
class CompressionInterface {
   public:
    virtual ~CompressionInterface()                               = default;

    virtual std::size_t do_compress_3d(void *const original_matrix,
                                       void *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_2d(void *const original_matrix,
                                       void *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_1d(void *const original_matrix,
                                       void *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_0d(void *const original_matrix,
                                       void *const output_array,
                                       unsigned int batch_size)   = 0;
    virtual std::size_t do_compress_flat(void *const original_matrix,
                                         void *const output_array,
                                         unsigned int n_pts)      = 0;

    virtual std::size_t do_decompress_3d(void *const compressed_buffer,
                                         void *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_2d(void *const compressed_buffer,
                                         void *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_1d(void *const compressed_buffer,
                                         void *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_0d(void *const compressed_buffer,
                                         void *const output_array,
                                         unsigned int batch_size) = 0;
    virtual std::size_t do_decompress_flat(void *const compressed_buffer,
                                           void *const output_array,
                                           unsigned int n_pts)    = 0;
};

}  // namespace dendrocompression
