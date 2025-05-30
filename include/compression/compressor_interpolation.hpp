#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "compression_base.hpp"

namespace dendrocompression {

template <typename T>
inline T lerp(T a, T b, T t) {
    // since we're not using c++20 yet...
    return a + t * (b - a);
}

template <typename T>
class InterpolationCompressor : public Compression<T> {
   private:
    std::vector<unsigned int> strides_;
    std::size_t n_reduced_;
    std::size_t n_reduced_2d_;
    std::size_t n_reduced_3d_;

    std::size_t n_reduced_1d_bytes_;
    std::size_t n_reduced_2d_bytes_;
    std::size_t n_reduced_3d_bytes_;

    std::size_t n_3d_;
    std::size_t n_2d_;

    unsigned int reduction_stride_;

    T scale_;

   public:
    InterpolationCompressor(unsigned int ele_order, unsigned int num_vars,
                            unsigned int reduction_stride = 2)
        : Compression<T>(ele_order, num_vars),
          reduction_stride_(reduction_stride) {
        // based on ele_order is how we'll do the interpolation

        n_2d_               = this->n_ * this->n_;
        n_3d_               = this->n_2d_ * this->n_;

        // NON-VARIABLE influenced, i.e. these are for later!
        n_reduced_          = (this->n_ / reduction_stride_) + 1;
        n_reduced_2d_       = n_reduced_ * n_reduced_;
        n_reduced_3d_       = n_reduced_2d_ * n_reduced_;

        n_reduced_1d_bytes_ = this->num_vars_ * n_reduced_ * sizeof(T);
        n_reduced_2d_bytes_ = this->num_vars_ * n_reduced_2d_ * sizeof(T);
        n_reduced_3d_bytes_ = this->num_vars_ * n_reduced_3d_ * sizeof(T);

        // calculate strides based on the number of points
        strides_.resize(this->n_reduced_);

        if (this->n_reduced_ == 1) {
            strides_[0] = 0;
        } else {
            // then fill in the strides
            for (unsigned int i = 0; i < n_reduced_; ++i) {
                strides_[i] = (i * (this->n_ - 1)) / (this->n_reduced_ - 1);
            }
        }

        // calculate the scale value for interpolation back up, we will do our
        // intput size of, say, 3 back to 5
        if (this->n_ == 1 || n_reduced_ == 1) {
            // single point, no interpolation
            this->scale_ = 0.0;
        } else {
            this->scale_ = static_cast<T>(this->n_reduced_ - 1) /
                           static_cast<T>(this->n_ - 1);
        }
    }

    ~InterpolationCompressor() = default;

    std::unique_ptr<Compression<T>> clone() const override {
        return std::make_unique<InterpolationCompressor>(*this);
    }

    CompressionType get_compression_type() const override {
        return CompressionType::COMP_INTERP;
    }

    std::string to_string() const override { return "InterpolationCompressor"; }

    std::size_t do_compress_3d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        const std::size_t uncompressed_size =
            this->total_3d_bytes_ * batch_size;

        T *output_translated = reinterpret_cast<T *>(output_array);

        for (unsigned int batch = 0; batch < batch_size; ++batch) {
            const unsigned int batch_offset_orig = batch * this->total_3d_pts_;
            const unsigned int batch_offset_out =
                batch * this->n_reduced_3d_ * this->num_vars_;
            for (unsigned int var = 0; var < this->num_vars_; ++var) {
                // var_offset needs to be calculated to get through the
                // variables
                const unsigned int var_offset_orig = var * this->n_3d_;
                const unsigned int var_offset_out  = var * this->n_reduced_3d_;
                for (unsigned int z = 0; z < this->n_reduced_; ++z) {
                    for (unsigned int y = 0; y < this->n_reduced_; ++y) {
                        for (unsigned int x = 0; x < this->n_reduced_; ++x) {
                            const unsigned int x5 = strides_[x];
                            const unsigned int y5 = strides_[y];
                            const unsigned int z5 = strides_[z];

                            std::size_t src_idx =
                                z5 * this->n_2d_ + y5 * this->n_ + x5;

                            // then write to the buffer
                            output_translated[batch_offset_out +
                                              var_offset_out +
                                              z * this->n_reduced_2d_ +
                                              y * this->n_reduced_ + x] =
                                original_matrix[batch_offset_orig +
                                                var_offset_orig + src_idx];
                        }
                    }
                }
            }
        }

        // then we just return the number of bytes we took, which is batch-size
        // times the nbytes 3d we calculated
        return this->n_reduced_3d_bytes_ * batch_size;
    }

    std::size_t do_decompress_3d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        const size_t uncompressed_size = this->total_3d_bytes_ * batch_size;

        // now we do the interpolation!

        // just trilinear works, probably
        T const *input_translated =
            reinterpret_cast<T const *>(compressed_buffer);

        if (this->n_reduced_ == 1) {
            // single batch size
            const size_t output_size = this->n_3d_;

            T *output_ptr            = output_array;
            const T *input_ptr = reinterpret_cast<const T *>(compressed_buffer);

            // fill with single input value from reduction
            for (unsigned int batch = 0; batch < batch_size; ++batch) {
                for (unsigned int var = 0; var < this->num_vars_; ++var) {
                    T single_value = input_ptr[batch * this->num_vars_ + var];

                    std::fill(output_ptr, output_ptr + output_size,
                              single_value);
                    output_ptr += output_size;
                }
            }
            return this->n_reduced_3d_bytes_ * batch_size;
        }

        for (unsigned int batch = 0; batch < batch_size; ++batch) {
            const unsigned int batch_offset_orig = batch * this->total_3d_pts_;
            const unsigned int batch_offset_comp =
                batch * this->n_reduced_3d_ * this->num_vars_;
            for (unsigned int var = 0; var < this->num_vars_; ++var) {
                // var_offset needs to be calculated to get through the
                // variables
                const unsigned int var_offset_orig = var * this->n_3d_;
                const unsigned int var_offset_comp = var * this->n_reduced_3d_;
                const unsigned int output_base =
                    batch_offset_orig + var_offset_orig;
                for (unsigned int z = 0; z < this->n_; ++z) {
                    T z_norm               = z * this->scale_;
                    int z0                 = static_cast<int>(z_norm);
                    int z1                 = std::min(z0 + 1,
                                                      static_cast<int>(this->n_reduced_) - 1);
                    T dz                   = z_norm - z0;
                    // offsets precompute to reduce additions
                    unsigned int z0_offset = batch_offset_comp +
                                             var_offset_comp +
                                             z0 * this->n_reduced_2d_;
                    unsigned int z1_offset = batch_offset_comp +
                                             var_offset_comp +
                                             z1 * this->n_reduced_2d_;
                    unsigned int z_out_offset = output_base + z * n_2d_;

                    // precompute inversion term
                    T dz_inv                  = 1.0f - dz;

                    for (unsigned int y = 0; y < this->n_; ++y) {
                        T y_norm = y * this->scale_;
                        int y0   = static_cast<int>(y_norm);
                        int y1   = std::min(
                            y0 + 1, static_cast<int>(this->n_reduced_) - 1);
                        T dy                      = y_norm - y0;

                        // offsets
                        unsigned int y0_offset    = y0 * this->n_reduced_;
                        unsigned int y1_offset    = y1 * this->n_reduced_;
                        unsigned int y_out_offset = z_out_offset + y * this->n_;

                        // precompute inversion term
                        T dy_inv                  = 1.0f - dy;

                        for (unsigned int x = 0; x < this->n_; ++x) {
                            T x_norm = x * this->scale_;
                            int x0   = static_cast<int>(x_norm);
                            int x1   = std::min(
                                x0 + 1, static_cast<int>(this->n_reduced_) - 1);
                            T dx     = x_norm - x0;
                            T dx_inv = 1.0f - dx;

                            // fetch 8 corners from input grid

                            T corner_000 =
                                input_translated[z0_offset + y0_offset + x0];
                            T corner_001 =
                                input_translated[z0_offset + y0_offset + x1];
                            T corner_010 =
                                input_translated[z0_offset + y1_offset + x0];
                            T corner_011 =
                                input_translated[z0_offset + y1_offset + x1];
                            T corner_100 =
                                input_translated[z1_offset + y0_offset + x0];
                            T corner_101 =
                                input_translated[z1_offset + y0_offset + x1];
                            T corner_110 =
                                input_translated[z1_offset + y1_offset + x0];
                            T corner_111 =
                                input_translated[z1_offset + y1_offset + x1];

                            // trilinear interpolation happens here
                            // T temp00 = corner_000 * dx_inv + corner_001 * dx;
                            // T temp01 = corner_010 * dx_inv + corner_011 * dx;
                            // T temp10 = corner_100 * dx_inv + corner_101 * dx;
                            // T temp11 = corner_110 * dx_inv + corner_111 * dx;
                            //
                            // T temp0  = temp00 * dy_inv + temp01 * dy;
                            // T temp1  = temp10 * dy_inv + temp11 * dy;
                            //
                            // T interpolated = temp0 * dz_inv + temp1 * dz;

                            // just call "lerp"
                            T interpolated =
                                lerp(lerp(lerp(corner_000, corner_001, dx),
                                          lerp(corner_010, corner_011, dx), dy),
                                     lerp(lerp(corner_100, corner_101, dx),
                                          lerp(corner_110, corner_111, dx), dy),
                                     dz);

                            // then put it in the output buffer!
                            output_array[y_out_offset + x] = interpolated;
                        }
                    }
                }
            }
        }

        return this->n_reduced_3d_bytes_ * batch_size;
    }

    std::size_t do_compress_2d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        const std::size_t uncompressed_size =
            this->total_2d_bytes_ * batch_size;

        T *output_translated = reinterpret_cast<T *>(output_array);

        for (unsigned int batch = 0; batch < batch_size; ++batch) {
            const unsigned int batch_offset_orig = batch * this->total_2d_pts_;
            const unsigned int batch_offset_out =
                batch * this->n_reduced_2d_ * this->num_vars_;
            for (unsigned int var = 0; var < this->num_vars_; ++var) {
                // var_offset needs to be calculated to get through the
                // variables
                const unsigned int var_offset_orig = var * this->n_2d_;
                const unsigned int var_offset_out  = var * this->n_reduced_2d_;
                for (unsigned int y = 0; y < this->n_reduced_; ++y) {
                    for (unsigned int x = 0; x < this->n_reduced_; ++x) {
                        const unsigned int x5 = strides_[x];
                        const unsigned int y5 = strides_[y];

                        std::size_t src_idx   = y5 * this->n_ + x5;

                        // then write to the buffer
                        output_translated[batch_offset_out + var_offset_out +
                                          y * this->n_reduced_ + x] =
                            original_matrix[batch_offset_orig +
                                            var_offset_orig + src_idx];
                    }
                }
            }
        }

        // then we just return the number of bytes we took, which is batch-size
        // times the nbytes 2d we calculated
        return this->n_reduced_2d_bytes_ * batch_size;
    }

    std::size_t do_decompress_2d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        const size_t uncompressed_size = this->total_2d_bytes_ * batch_size;

        // now we do the interpolation!

        // just trilinear works, probably
        T const *input_translated =
            reinterpret_cast<T const *>(compressed_buffer);

        if (this->n_reduced_ == 1) {
            // single batch size
            const size_t output_size = this->n_2d_;

            T *output_ptr            = output_array;
            const T *input_ptr = reinterpret_cast<const T *>(compressed_buffer);

            // fill with single input value from reduction
            for (unsigned int batch = 0; batch < batch_size; ++batch) {
                for (unsigned int var = 0; var < this->num_vars_; ++var) {
                    T single_value = input_ptr[batch * this->num_vars_ + var];

                    std::fill(output_ptr, output_ptr + output_size,
                              single_value);
                    output_ptr += output_size;
                }
            }
            return this->n_reduced_2d_bytes_ * batch_size;
        }

        for (unsigned int batch = 0; batch < batch_size; ++batch) {
            const unsigned int batch_offset_orig = batch * this->total_2d_pts_;
            const unsigned int batch_offset_comp =
                batch * this->n_reduced_2d_ * this->num_vars_;
            for (unsigned int var = 0; var < this->num_vars_; ++var) {
                // var_offset needs to be calculated to get through the
                // variables
                const unsigned int var_offset_orig = var * this->n_2d_;
                const unsigned int var_offset_comp = var * this->n_reduced_2d_;
                const unsigned int output_base =
                    batch_offset_orig + var_offset_orig;

                for (unsigned int y = 0; y < this->n_; ++y) {
                    T y_norm               = y * this->scale_;
                    int y0                 = static_cast<int>(y_norm);
                    int y1                 = std::min(y0 + 1,
                                                      static_cast<int>(this->n_reduced_) - 1);
                    T dy                   = y_norm - y0;

                    // offsets
                    unsigned int y0_offset = batch_offset_comp +
                                             var_offset_comp +
                                             y0 * this->n_reduced_;
                    unsigned int y1_offset = batch_offset_comp +
                                             var_offset_comp +
                                             y1 * this->n_reduced_;
                    unsigned int y_out_offset = output_base + y * this->n_;

                    // precompute inversion term
                    T dy_inv                  = 1.0f - dy;

                    for (unsigned int x = 0; x < this->n_; ++x) {
                        T x_norm = x * this->scale_;
                        int x0   = static_cast<int>(x_norm);
                        int x1   = std::min(
                            x0 + 1, static_cast<int>(this->n_reduced_) - 1);
                        T dx        = x_norm - x0;
                        T dx_inv    = 1.0f - dx;

                        // fetch 8 corners from input grid

                        T corner_00 = input_translated[y0_offset + x0];
                        T corner_01 = input_translated[y0_offset + x1];
                        T corner_10 = input_translated[y1_offset + x0];
                        T corner_11 = input_translated[y1_offset + x1];

                        // trilinear interpolation happens here
                        // T temp00 = corner_00 * dx_inv + corner_001 * dx;
                        // T temp01 = corner_10 * dx_inv + corner_011 * dx;

                        // T interpolated = temp00 * dy_inv + temp01 * dy;

                        // just call "lerp"
                        T interpolated =
                            lerp(lerp(corner_00, corner_01, dx),
                                 lerp(corner_10, corner_11, dx), dy);

                        // then put it in the output buffer!
                        output_array[y_out_offset + x] = interpolated;
                    }
                }
            }
        }

        return this->n_reduced_2d_bytes_ * batch_size;
    }

    std::size_t do_compress_1d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        const std::size_t uncompressed_size =
            this->total_1d_bytes_ * batch_size;

        T *output_translated = reinterpret_cast<T *>(output_array);

        for (unsigned int batch = 0; batch < batch_size; ++batch) {
            const unsigned int batch_offset_orig = batch * this->total_1d_pts_;
            const unsigned int batch_offset_out =
                batch * this->n_reduced_ * this->num_vars_;
            for (unsigned int var = 0; var < this->num_vars_; ++var) {
                // var_offset needs to be calculated to get through the
                // variables
                const unsigned int var_offset_orig = var * this->n_;
                const unsigned int var_offset_out  = var * this->n_reduced_;
                for (unsigned int x = 0; x < this->n_reduced_; ++x) {
                    const unsigned int x5 = strides_[x];

                    std::size_t src_idx   = x5;

                    // then write to the buffer
                    output_translated[batch_offset_out + var_offset_out + +x] =
                        original_matrix[batch_offset_orig + var_offset_orig +
                                        src_idx];
                }
            }
        }

        // then we just return the number of bytes we took, which is batch-size
        // times the nbytes 1d we calculated
        return this->n_reduced_1d_bytes_ * batch_size;
    }

    std::size_t do_decompress_1d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        const size_t uncompressed_size = this->total_1d_bytes_ * batch_size;

        // now we do the interpolation!

        // just trilinear works, probably
        T const *input_translated =
            reinterpret_cast<T const *>(compressed_buffer);

        if (this->n_reduced_ == 1) {
            // single batch size
            const size_t output_size = this->n_;

            T *output_ptr            = output_array;
            const T *input_ptr = reinterpret_cast<const T *>(compressed_buffer);

            // fill with single input value from reduction
            for (unsigned int batch = 0; batch < batch_size; ++batch) {
                for (unsigned int var = 0; var < this->num_vars_; ++var) {
                    T single_value = input_ptr[batch * this->num_vars_ + var];

                    std::fill(output_ptr, output_ptr + output_size,
                              single_value);
                    output_ptr += output_size;
                }
            }
            return this->n_reduced_1d_bytes_ * batch_size;
        }

        for (unsigned int batch = 0; batch < batch_size; ++batch) {
            const unsigned int batch_offset_orig = batch * this->total_1d_pts_;
            const unsigned int batch_offset_comp =
                batch * this->n_reduced_ * this->num_vars_;
            for (unsigned int var = 0; var < this->num_vars_; ++var) {
                // var_offset needs to be calculated to get through the
                // variables
                const unsigned int var_offset_orig = var * this->n_;
                const unsigned int var_offset_comp = var * this->n_reduced_;
                const unsigned int output_base =
                    batch_offset_orig + var_offset_orig;

                for (unsigned int x = 0; x < this->n_; ++x) {
                    T x_norm       = x * this->scale_;
                    int x0         = static_cast<int>(x_norm);
                    int x1         = std::min(x0 + 1,
                                              static_cast<int>(this->n_reduced_) - 1);
                    T dx           = x_norm - x0;
                    T dx_inv       = 1.0f - dx;

                    // fetch 8 corners from input grid

                    T corner_0     = input_translated[x0];
                    T corner_1     = input_translated[x1];

                    // trilinear interpolation happens here

                    // T interpolated = corner_0 * dx_inv + corner_1 * dx;

                    // just call "lerp"
                    T interpolated = lerp(corner_0, corner_1, dx);

                    // then put it in the output buffer!
                    output_array[output_base + x] = interpolated;
                }
            }
        }

        return this->n_reduced_1d_bytes_ * batch_size;
    }

    std::size_t do_compress_0d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        // no interpolation for 0d, just copy in
        std::memcpy(output_array, original_matrix,
                    batch_size * this->total_0d_bytes_);
        return batch_size * this->total_0d_bytes_;
    }

    std::size_t do_decompress_0d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        std::memcpy(output_array, compressed_buffer,
                    batch_size * this->total_0d_bytes_);
        return batch_size * this->total_0d_bytes_;
    }

    std::size_t do_compress_flat(T *const original_matrix,
                                 unsigned char *const output_array,
                                 unsigned int n_pts) {
        std::memcpy(output_array, original_matrix,
                    n_pts * this->total_0d_bytes_);
        return n_pts * this->total_0d_bytes_;
    }

    std::size_t do_decompress_flat(unsigned char *const compressed_buffer,
                                   T *const output_array, unsigned int n_pts) {
        std::memcpy(output_array, compressed_buffer,
                    n_pts * this->total_0d_bytes_);
        return n_pts * this->total_0d_bytes_;
    }
};

}  // namespace dendrocompression
