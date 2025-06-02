#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "compression_base.hpp"
#include "zfp.h"

namespace dendrocompression {

template <typename T>
class ZFPCompressor : public Compression<T> {
   private:
    std::string zfp_mode_;
    zfp_stream *zfp4d_   = nullptr;
    zfp_stream *zfp3d_   = nullptr;
    zfp_stream *zfp2d_   = nullptr;
    zfp_stream *zfp1d_   = nullptr;
    zfp_field *field_4d_ = nullptr;
    zfp_field *field_3d_ = nullptr;
    zfp_field *field_2d_ = nullptr;
    zfp_field *field_1d_ = nullptr;
    zfp_type field_type_;

    double rate_;
    double tolerance_;
    unsigned int precision_;

    void set_up_fields() {
        free_all_fields();
        if (this->is_multivar_) {
            field_4d_ = zfp_field_4d(NULL, field_type_, this->num_vars_,
                                     this->n_, this->n_, this->n_);
            field_3d_ = zfp_field_3d(NULL, field_type_, this->num_vars_,
                                     this->n_, this->n_);
            field_2d_ =
                zfp_field_2d(NULL, field_type_, this->num_vars_, this->n_);
            field_1d_ = zfp_field_1d(NULL, field_type_, this->num_vars_);
        } else {
            // no 4d field
            field_3d_ =
                zfp_field_3d(NULL, field_type_, this->n_, this->n_, this->n_);
            field_2d_ = zfp_field_2d(NULL, field_type_, this->n_, this->n_);
            field_1d_ = zfp_field_1d(NULL, field_type_, this->n_);
        }
    }

    void set_up_streams() {
        close_all_streams();
        zfp4d_ = zfp_stream_open(NULL);
        zfp3d_ = zfp_stream_open(NULL);
        zfp2d_ = zfp_stream_open(NULL);
        zfp1d_ = zfp_stream_open(NULL);
    }

    void set_up_precision() {
        set_up_fields();
        set_up_streams();

        // now we can set the values
        if (zfp3d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        zfp_stream_set_precision(zfp3d_, precision_);
        zfp_stream_set_precision(zfp2d_, precision_);
        zfp_stream_set_precision(zfp1d_, precision_);

        assert(zfp_stream_compression_mode(zfp3d_) == zfp_mode_fixed_precision);
        assert(zfp_stream_compression_mode(zfp2d_) == zfp_mode_fixed_precision);
        assert(zfp_stream_compression_mode(zfp1d_) == zfp_mode_fixed_precision);

        if (this->is_multivar_) {
            if (zfp4d_ == nullptr)
                throw std::invalid_argument(
                    "ZFP Wasn't properly initialized for some reason!");
            zfp_stream_set_precision(zfp4d_, precision_);
            assert(zfp_stream_compression_mode(zfp4d_) ==
                   zfp_mode_fixed_precision);
        }
    }

    void set_up_rate() {
        set_up_fields();
        set_up_streams();

        // now we can set the values
        if (zfp3d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        zfp_stream_set_rate(zfp3d_, rate_, field_type_, 3, 0);
        zfp_stream_set_rate(zfp2d_, rate_, field_type_, 2, 0);
        zfp_stream_set_rate(zfp1d_, rate_, field_type_, 1, 0);

        assert(zfp_stream_compression_mode(zfp3d_) == zfp_mode_fixed_rate);
        assert(zfp_stream_compression_mode(zfp2d_) == zfp_mode_fixed_rate);
        assert(zfp_stream_compression_mode(zfp1d_) == zfp_mode_fixed_rate);

        if (this->is_multivar_) {
            if (zfp4d_ == nullptr)
                throw std::invalid_argument(
                    "ZFP Wasn't properly initialized for some reason!");
            zfp_stream_set_rate(zfp4d_, rate_, field_type_, 4, 0);
            assert(zfp_stream_compression_mode(zfp4d_) == zfp_mode_fixed_rate);
        }
    }

    void set_up_accuracy() {
        set_up_fields();
        set_up_streams();

        // now we can set the values
        if (zfp3d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp2d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");
        if (zfp1d_ == nullptr)
            throw std::invalid_argument(
                "ZFP Wasn't properly initialized for some reason!");

        zfp_stream_set_accuracy(zfp3d_, tolerance_);
        zfp_stream_set_accuracy(zfp2d_, tolerance_);
        zfp_stream_set_accuracy(zfp1d_, tolerance_);

        assert(zfp_stream_compression_mode(zfp3d_) == zfp_mode_fixed_accuracy);
        assert(zfp_stream_compression_mode(zfp2d_) == zfp_mode_fixed_accuracy);
        assert(zfp_stream_compression_mode(zfp1d_) == zfp_mode_fixed_accuracy);

        if (this->is_multivar_) {
            if (zfp4d_ == nullptr)
                throw std::invalid_argument(
                    "ZFP Wasn't properly initialized for some reason!");
            zfp_stream_set_accuracy(zfp4d_, tolerance_);
            assert(zfp_stream_compression_mode(zfp4d_) ==
                   zfp_mode_fixed_accuracy);
        }
    }

    void close_and_free_all() {
        close_all_streams();
        free_all_fields();
    }

    void close_all_streams() {
        if (zfp4d_ != nullptr) zfp_stream_close(zfp4d_);

        if (zfp3d_ != nullptr) zfp_stream_close(zfp3d_);

        if (zfp2d_ != nullptr) zfp_stream_close(zfp2d_);

        if (zfp1d_ != nullptr) zfp_stream_close(zfp1d_);

        zfp4d_ = nullptr;
        zfp3d_ = nullptr;
        zfp2d_ = nullptr;
        zfp1d_ = nullptr;
    }

    void free_all_fields() {
        if (field_4d_ != nullptr) zfp_field_free(field_4d_);

        if (field_3d_ != nullptr) zfp_field_free(field_3d_);

        if (field_2d_ != nullptr) zfp_field_free(field_2d_);

        if (field_1d_ != nullptr) zfp_field_free(field_1d_);

        field_4d_ = nullptr;
        field_3d_ = nullptr;
        field_2d_ = nullptr;
        field_1d_ = nullptr;
    }

   public:
    ZFPCompressor(unsigned int ele_order, unsigned int num_vars,
                  const std::string &zfp_mode, const double zfp_parameter)
        : Compression<T>(ele_order, num_vars), zfp_mode_(zfp_mode) {
        // note that the ZFP parameter could be in the input rate, accuracy, or
        // precision. The precision, in that case, would be **directly cast** to
        // an integer (after calling a round).

        if constexpr (std::is_same_v<T, double>) {
            field_type_ = zfp_type_double;
        } else if constexpr (std::is_same_v<T, float>) {
            field_type_ = zfp_type_float;
        } else {
            throw std::invalid_argument(
                "ZFPCompressor was attempted with non-float/non-double data "
                "type. This isn't supported.");
        }

        if (zfp_mode_ == "precision") {
            precision_ = static_cast<unsigned int>(std::round(zfp_parameter));
            set_up_precision();
        } else if (zfp_mode_ == "accuracy") {
            tolerance_ = zfp_parameter;
            set_up_accuracy();
        } else if (zfp_mode_ == "rate") {
            rate_ = zfp_parameter;
            set_up_rate();
        }
    }

    ~ZFPCompressor() { close_and_free_all(); };

    std::unique_ptr<Compression<T>> clone() const override {
        return std::make_unique<ZFPCompressor>(*this);
    }

    CompressionType get_compression_type() const override {
        return CompressionType::COMP_ZFP;
    }

    std::string to_string() const override { return "ZFPCompressor"; }

    std::size_t do_compress_3d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        const size_t uncompressed_size = this->total_3d_bytes_ * batch_size;

        // batch-wise computation
        if (field_3d_ != nullptr) {
            zfp_field_free(field_3d_);
            field_3d_ = nullptr;
        }

        field_3d_ = zfp_field_3d(NULL, field_type_,
                                 this->num_vars_ * batch_size * this->n_,
                                 this->n_, this->n_);

        // TODO: the old set up for the compression, might want to consider
        // extending batch_size to another dimension, or num_vars or something

        zfp_field_set_pointer(field_3d_, original_matrix);

        size_t bufsize = zfp_stream_maximum_size(zfp3d_, field_3d_);
        if (bufsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR CALCULATING MAXIMUM SIZE in ZFP 3d Compress!");
        }

        bitstream *stream = stream_open(output_array + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 3d Compress!");
        }

        zfp_stream_set_bit_stream(zfp3d_, stream);

        size_t outsize = zfp_compress(zfp3d_, field_3d_);
        if (outsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR COMPRESSING DATA IN 3D ZFP STREAM!");
        }

        stream_close(stream);

        if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
            std::cerr
                << "CRITICAL ERROR COMPRESSING DATA IN 4D ZFP STREAM! The "
                   "compressed buffer is larger than the original!"
                << std::endl;
            std::cerr << "Number of points to compress: "
                      << zfp_dim4_decomp * batch_size << " ("
                      << zfp_dim4_decomp * sizeof(T) * batch_size
                      << " bytes), number of bytes in compressed stream: "
                      << outsize << std::endl;
#endif

            // just copy the raw data
            std::memcpy(output_array + sizeof(size_t), original_matrix,
                        uncompressed_size);
            outsize = uncompressed_size;
        }

        std::memcpy(output_array, &outsize, sizeof(outsize));
        return outsize + sizeof(size_t);
    }

    std::size_t do_decompress_3d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        const size_t uncompressed_size = this->total_3d_bytes_ * batch_size;

        // first extract out the buffer size
        size_t bufsize;
        std::memcpy(&bufsize, compressed_buffer, sizeof(size_t));

        if (bufsize == uncompressed_size) {
            std::memcpy(output_array, compressed_buffer + sizeof(size_t),
                        uncompressed_size);
            return bufsize + sizeof(size_t);
        }

        bitstream *stream =
            stream_open(compressed_buffer + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 3d Decompress!");
        }

        if (field_3d_ != nullptr) {
            zfp_field_free(field_3d_);
            field_3d_ = nullptr;
        }

        field_3d_ = zfp_field_3d(NULL, field_type_,
                                 this->num_vars_ * batch_size * this->n_,
                                 this->n_, this->n_);

        zfp_stream_set_bit_stream(zfp3d_, stream);
        zfp_field_set_pointer(field_3d_, output_array);

        size_t outsize = zfp_decompress(zfp3d_, field_3d_);
        if (!outsize) {
            throw std::runtime_error(
                "CRITICAL ERROR DECOMPRESSING DATA IN 4D ZFP STREAM!");
        }

        stream_close(stream);

        // remember, this is for the raw buffer, as it includes that data that
        // we're working with
        return bufsize + sizeof(size_t);
    }

    std::size_t do_compress_2d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        const size_t uncompressed_size = this->total_2d_bytes_ * batch_size;

        // batch-wise computation
        if (field_3d_ != nullptr) {
            zfp_field_free(field_3d_);
            field_3d_ = nullptr;
        }

        field_3d_ =
            zfp_field_3d(NULL, field_type_, this->num_vars_ * batch_size,
                         this->n_, this->n_);

        // TODO: the old set up for the compression, might want to consider
        // extending batch_size to another dimension, or num_vars or something

        zfp_field_set_pointer(field_3d_, original_matrix);

        size_t bufsize = zfp_stream_maximum_size(zfp3d_, field_3d_);
        if (bufsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR CALCULATING MAXIMUM SIZE in ZFP 3d Compress!");
        }

        bitstream *stream = stream_open(output_array + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 3d Compress!");
        }

        zfp_stream_set_bit_stream(zfp3d_, stream);

        size_t outsize = zfp_compress(zfp3d_, field_3d_);
        if (outsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR COMPRESSING DATA IN 3D ZFP STREAM!");
        }

        stream_close(stream);

        if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
            std::cerr
                << "CRITICAL ERROR COMPRESSING DATA IN 4D ZFP STREAM! The "
                   "compressed buffer is larger than the original!"
                << std::endl;
            std::cerr << "Number of points to compress: "
                      << zfp_dim4_decomp * batch_size << " ("
                      << zfp_dim4_decomp * sizeof(T) * batch_size
                      << " bytes), number of bytes in compressed stream: "
                      << outsize << std::endl;
#endif

            // just copy the raw data
            std::memcpy(output_array + sizeof(size_t), original_matrix,
                        uncompressed_size);
            outsize = uncompressed_size;
        }

        std::memcpy(output_array, &outsize, sizeof(outsize));
        return outsize + sizeof(size_t);
    }

    std::size_t do_decompress_2d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        const size_t uncompressed_size = this->total_2d_bytes_ * batch_size;

        // first extract out the buffer size
        size_t bufsize;
        std::memcpy(&bufsize, compressed_buffer, sizeof(size_t));

        if (bufsize == uncompressed_size) {
            std::memcpy(output_array, compressed_buffer + sizeof(size_t),
                        uncompressed_size);
            return bufsize + sizeof(size_t);
        }

        bitstream *stream =
            stream_open(compressed_buffer + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 3d Decompress!");
        }

        if (field_3d_ != nullptr) {
            zfp_field_free(field_3d_);
            field_3d_ = nullptr;
        }

        field_3d_ =
            zfp_field_3d(NULL, field_type_, this->num_vars_ * batch_size,
                         this->n_, this->n_);

        zfp_stream_set_bit_stream(zfp3d_, stream);
        zfp_field_set_pointer(field_3d_, output_array);

        size_t outsize = zfp_decompress(zfp3d_, field_3d_);
        if (!outsize) {
            throw std::runtime_error(
                "CRITICAL ERROR DECOMPRESSING DATA IN 4D ZFP STREAM!");
        }

        stream_close(stream);

        // remember, this is for the raw buffer, as it includes that data that
        // we're working with
        return bufsize + sizeof(size_t);
    }

    std::size_t do_compress_1d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        const size_t uncompressed_size = this->total_1d_bytes_ * batch_size;

        // batch-wise computation
        if (field_2d_ != nullptr) {
            zfp_field_free(field_2d_);
            field_2d_ = nullptr;
        }

        field_2d_ = zfp_field_2d(NULL, field_type_,
                                 this->num_vars_ * batch_size, this->n_);

        // TODO: the old set up for the compression, might want to consider
        // extending batch_size to another dimension, or num_vars or something

        zfp_field_set_pointer(field_2d_, original_matrix);

        size_t bufsize = zfp_stream_maximum_size(zfp2d_, field_2d_);
        if (bufsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR CALCULATING MAXIMUM SIZE in ZFP 1d Compress!");
        }

        bitstream *stream = stream_open(output_array + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 1d Compress!");
        }

        zfp_stream_set_bit_stream(zfp2d_, stream);

        size_t outsize = zfp_compress(zfp2d_, field_2d_);
        if (outsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR COMPRESSING DATA IN 1D ZFP STREAM!");
        }

        stream_close(stream);

        if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
            std::cerr
                << "CRITICAL ERROR COMPRESSING DATA IN 4D ZFP STREAM! The "
                   "compressed buffer is larger than the original!"
                << std::endl;
            std::cerr << "Number of points to compress: "
                      << zfp_dim4_decomp * batch_size << " ("
                      << zfp_dim4_decomp * sizeof(T) * batch_size
                      << " bytes), number of bytes in compressed stream: "
                      << outsize << std::endl;
#endif

            // just copy the raw data
            std::memcpy(output_array + sizeof(size_t), original_matrix,
                        uncompressed_size);
            outsize = uncompressed_size;
        }

        std::memcpy(output_array, &outsize, sizeof(outsize));
        return outsize + sizeof(size_t);
    }

    std::size_t do_decompress_1d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        const size_t uncompressed_size = this->total_1d_bytes_ * batch_size;

        // first extract out the buffer size
        size_t bufsize;
        std::memcpy(&bufsize, compressed_buffer, sizeof(size_t));

        if (bufsize == uncompressed_size) {
            std::memcpy(output_array, compressed_buffer + sizeof(size_t),
                        uncompressed_size);
            return bufsize + sizeof(size_t);
        }

        bitstream *stream =
            stream_open(compressed_buffer + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 1d Decompress!");
        }

        if (field_2d_ != nullptr) {
            zfp_field_free(field_2d_);
            field_2d_ = nullptr;
        }

        field_2d_ = zfp_field_2d(NULL, field_type_,
                                 this->num_vars_ * batch_size, this->n_);

        zfp_stream_set_bit_stream(zfp2d_, stream);
        zfp_field_set_pointer(field_2d_, output_array);

        size_t outsize = zfp_decompress(zfp2d_, field_2d_);
        if (!outsize) {
            throw std::runtime_error(
                "CRITICAL ERROR DECOMPRESSING DATA IN 1D ZFP STREAM!");
        }

        stream_close(stream);

        // remember, this is for the raw buffer, as it includes that data that
        // we're working with
        return bufsize + sizeof(size_t);
    }

    std::size_t do_compress_0d(T *const original_matrix,
                               unsigned char *const output_array,
                               unsigned int batch_size) {
        const size_t uncompressed_size = this->total_0d_bytes_ * batch_size;

        // batch-wise computation
        if (field_1d_ != nullptr) {
            zfp_field_free(field_1d_);
            field_1d_ = nullptr;
        }

        field_1d_ =
            zfp_field_1d(NULL, field_type_, this->num_vars_ * batch_size);

        // TODO: the old set up for the compression, might want to consider
        // extending batch_size to another dimension, or num_vars or something

        zfp_field_set_pointer(field_1d_, original_matrix);

        size_t bufsize = zfp_stream_maximum_size(zfp1d_, field_1d_);
        if (bufsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR CALCULATING MAXIMUM SIZE in ZFP 0d Compress!");
        }

        bitstream *stream = stream_open(output_array + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 0d Compress!");
        }

        zfp_stream_set_bit_stream(zfp1d_, stream);

        size_t outsize = zfp_compress(zfp1d_, field_1d_);
        if (outsize == 0) {
            throw std::runtime_error(
                "CRITICAL ERROR COMPRESSING DATA IN 0D ZFP STREAM!");
        }

        stream_close(stream);

        if (outsize > uncompressed_size) {
#ifdef __DENDRO_PRINT_ZFP_WARNING__
            std::cerr
                << "CRITICAL ERROR COMPRESSING DATA IN 4D ZFP STREAM! The "
                   "compressed buffer is larger than the original!"
                << std::endl;
            std::cerr << "Number of points to compress: "
                      << zfp_dim4_decomp * batch_size << " ("
                      << zfp_dim4_decomp * sizeof(T) * batch_size
                      << " bytes), number of bytes in compressed stream: "
                      << outsize << std::endl;
#endif

            // just copy the raw data
            std::memcpy(output_array + sizeof(size_t), original_matrix,
                        uncompressed_size);
            outsize = uncompressed_size;
        }

        std::memcpy(output_array, &outsize, sizeof(outsize));
        return outsize + sizeof(size_t);
    }

    std::size_t do_decompress_0d(unsigned char *const compressed_buffer,
                                 T *const output_array,
                                 unsigned int batch_size) {
        const size_t uncompressed_size = this->total_0d_bytes_ * batch_size;

        // first extract out the buffer size
        size_t bufsize;
        std::memcpy(&bufsize, compressed_buffer, sizeof(size_t));

        if (bufsize == uncompressed_size) {
            std::memcpy(output_array, compressed_buffer + sizeof(size_t),
                        uncompressed_size);
            return bufsize + sizeof(size_t);
        }

        bitstream *stream =
            stream_open(compressed_buffer + sizeof(size_t), bufsize);
        if (stream == nullptr) {
            throw std::runtime_error(
                "CRITICAL ERROR OPENING BITSTREAM IN ZFP 0d Decompress!");
        }

        if (field_1d_ != nullptr) {
            zfp_field_free(field_1d_);
            field_1d_ = nullptr;
        }

        field_1d_ =
            zfp_field_1d(NULL, field_type_, this->num_vars_ * batch_size);

        zfp_stream_set_bit_stream(zfp1d_, stream);
        zfp_field_set_pointer(field_1d_, output_array);

        size_t outsize = zfp_decompress(zfp1d_, field_1d_);
        if (!outsize) {
            throw std::runtime_error(
                "CRITICAL ERROR DECOMPRESSING DATA IN 0D ZFP STREAM!");
        }

        stream_close(stream);

        // remember, this is for the raw buffer, as it includes that data that
        // we're working with
        return bufsize + sizeof(size_t);
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
