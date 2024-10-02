#include "compression.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "zfp.h"
#include "zfp/bitstream.h"
// #include "mpi.h"

namespace ChebyshevAlgorithms {

// a "global" object that we can call from pretty much anywhere
ChebyshevCompression cheby{6, 3};

void ChebyshevCompression::set_compression_type(const size_t& eleOrder,
                                                const size_t& nReduced) {
    // TODO: autogenerate this
    if (eleOrder == 6) {
        if (nReduced == 1) {
            set_chebyshev_mat_ele6_out1_dim1();
            set_chebyshev_mat_ele6_out1_dim2();
            set_chebyshev_mat_ele6_out1_dim3();
        } else if (nReduced == 2) {
            set_chebyshev_mat_ele6_out2_dim1();
            set_chebyshev_mat_ele6_out2_dim2();
            set_chebyshev_mat_ele6_out2_dim3();
        } else if (nReduced == 3) {
            set_chebyshev_mat_ele6_out3_dim1();
            set_chebyshev_mat_ele6_out3_dim2();
            set_chebyshev_mat_ele6_out3_dim3();
        } else if (nReduced == 4) {
            set_chebyshev_mat_ele6_out4_dim1();
            set_chebyshev_mat_ele6_out4_dim2();
            set_chebyshev_mat_ele6_out4_dim3();
        } else if (eleOrder == 8) {
            if (nReduced == 1) {
                set_chebyshev_mat_ele8_out1_dim1();
                set_chebyshev_mat_ele8_out1_dim2();
                set_chebyshev_mat_ele8_out1_dim3();
            } else if (nReduced == 2) {
                set_chebyshev_mat_ele8_out2_dim1();
                set_chebyshev_mat_ele8_out2_dim2();
                set_chebyshev_mat_ele8_out2_dim3();
            } else if (nReduced == 3) {
                set_chebyshev_mat_ele8_out3_dim1();
                set_chebyshev_mat_ele8_out3_dim2();
                set_chebyshev_mat_ele8_out3_dim3();
            } else if (nReduced == 4) {
                set_chebyshev_mat_ele8_out4_dim1();
                set_chebyshev_mat_ele8_out4_dim2();
                set_chebyshev_mat_ele8_out4_dim3();
            }
        }
    }

    recalculate_byte_sizes();
}

void ChebyshevCompression::do_array_norm(double* array, const size_t count,
                                         double& minVal, double& maxVal) {
    minVal       = *std::min_element(array, array + count);
    maxVal       = *std::max_element(array, array + count);
    double range = maxVal - minVal;

    if (minVal < -1.0 || maxVal > 1.0) {
        // if we're outside -1 and or 1, then we need to do full normalization
        for (size_t i = 0; i < count; i++) {
            array[i] = 2.0 * ((array[i] - minVal) / range) - 1.0;
        }
    } else if (range > 1e-8) {
        // apply a shift if we're within the vals
        double shift = -(minVal + maxVal) / 2.0;
        for (size_t i = 0; i < count; i++) {
            array[i] += shift;
        }
    } else {
        // otherwise do nothing, we're close to zero or something
    }
}

void ChebyshevCompression::undo_array_norm(double* array, const size_t count,
                                           const double minVal,
                                           const double maxVal) {
    double range = maxVal - minVal;

    if (minVal < -1.0 || maxVal > 1.0) {
        // if we're outside -1 and or 1, then we need to do full
        // denormalization
        for (size_t i = 0; i < count; i++) {
            array[i] = ((array[i] + 1.0) / 2.0) * range + minVal;
        }
    } else if (range > 1e-8) {
        // apply a shift if we're within the vals
        double shift = -(minVal + maxVal) / 2.0;
        for (size_t i = 0; i < count; i++) {
            array[i] -= shift;
        }
    } else {
        // otherwise do nothing, we didn't do anything above
    }
}

size_t ChebyshevCompression::do_3d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    // the number will always be (eleorder - 1) ^3 because it's a 3d block,
    // turns out this number coencides with cheb_dim3_decomp

    double maxVal, minVal;
    do_array_norm(originalMatrix, cheb_dim3_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    // that *I* control
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    char TRANS = 'T';

    dgemv_(&TRANS, &cheb_dim3_decomp, &cheb_dim3_comp, &alpha, A_cheb_dim3,
           &cheb_dim3_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    // now we're "compressed", and we return the total number of bytes we
    // wrote:
    return bytes_3d;
}

size_t ChebyshevCompression::do_3d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim3_decomp, &cheb_dim3_comp, &alpha, A_cheb_dim3,
           &cheb_dim3_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim3_decomp, minVal, maxVal);

    // total number of doubles coming out
    return bytes_3d;
}

size_t ChebyshevCompression::do_2d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    double maxVal, minVal;
    do_array_norm(originalMatrix, cheb_dim2_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    char TRANS = 'T';
    // now do the matrix-vector multiplication
    dgemv_(&TRANS, &cheb_dim2_decomp, &cheb_dim2_comp, &alpha, A_cheb_dim2,
           &cheb_dim2_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    // total number of bytes that we wrote
    return bytes_2d;
}

size_t ChebyshevCompression::do_2d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim2_decomp, &cheb_dim2_comp, &alpha, A_cheb_dim2,
           &cheb_dim2_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim2_decomp, minVal, maxVal);

    // total number of compress values coming out, to advance. The doubles we
    // know are easy
    return bytes_2d;
}

size_t ChebyshevCompression::do_1d_compression(double* originalMatrix,
                                               unsigned char* outputArray) {
    double maxVal, minVal;

    do_array_norm(originalMatrix, cheb_dim1_decomp, minVal, maxVal);

    // copy min and max val
    std::memcpy(outputArray, &minVal, sizeof(double));
    std::memcpy(outputArray + sizeof(double), &maxVal, sizeof(double));

    // recast the output array as a double array, since it's just memory
    double* outputCast =
        reinterpret_cast<double*>(outputArray + 2 * sizeof(double));

    // do the matrix math
    char TRANSA         = 'T';
    char TRANSB         = 'N';

    double* temp_output = new double[3];

    // matrix multiplication because it works
    dgemv_(&TRANSA, &cheb_dim1_decomp, &cheb_dim1_comp, &alpha, A_cheb_dim1,
           &cheb_dim1_decomp, originalMatrix, &single_dim, &beta, outputCast,
           &single_dim);

    return bytes_1d;
}

size_t ChebyshevCompression::do_1d_decompression(
    unsigned char* compressedBuffer, double* outputArray) {
    // pass
    // so we need to get the min and max value
    double maxVal, minVal;

    unsigned char* ptr = compressedBuffer;

    std::memcpy(&minVal, compressedBuffer, sizeof(double));
    std::memcpy(&maxVal, compressedBuffer + sizeof(double), sizeof(double));

    // if (minVal > maxVal) {
    //     std::cout << "Detected minval larger than maxval in 1d decompression!
    //     "
    //               << maxVal << " " << minVal << std::endl;
    // }

    double* inputCast =
        reinterpret_cast<double*>(compressedBuffer + 2 * sizeof(double));

    char TRANS = 'N';

    dgemv_(&TRANS, &cheb_dim1_decomp, &cheb_dim1_comp, &alpha, A_cheb_dim1,
           &cheb_dim1_decomp, inputCast, &single_dim, &beta, outputArray,
           &single_dim);

    // undo the array normalization
    undo_array_norm(outputArray, cheb_dim1_decomp, minVal, maxVal);

    // total number of doubles coming out
    return bytes_1d;
}

}  // namespace ChebyshevAlgorithms

namespace ZFPAlgorithms {

// "global" object for ZFP algorithm that can be called by dendro
ZFPCompression zfpblockwise(6, 5.0);

size_t ZFPCompression::do_3d_compression(double* originalMatrix,
                                         unsigned char* outputArray) {
    // create a field
    zfp_field_set_pointer(field_3d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize    = zfp_stream_maximum_size(zfp3d, field_3d);

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp3d, stream);

    size_t outsize = zfp_compress(zfp3d, field_3d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 3D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

size_t ZFPCompression::do_3d_decompression(unsigned char* compressedBuffer,
                                           double* outputArray) {
    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);

    zfp_stream_set_bit_stream(zfp3d, stream);

    zfp_field_set_pointer(field_3d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp3d, field_3d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 3D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

size_t ZFPCompression::do_2d_compression(double* originalMatrix,
                                         unsigned char* outputArray) {
    // create a field
    zfp_field_set_pointer(field_2d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize    = zfp_stream_maximum_size(zfp2d, field_2d);

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp2d, stream);

    size_t outsize = zfp_compress(zfp2d, field_2d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 2D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

size_t ZFPCompression::do_2d_decompression(unsigned char* compressedBuffer,
                                           double* outputArray) {
    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);

    zfp_stream_set_bit_stream(zfp2d, stream);

    zfp_field_set_pointer(field_2d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp2d, field_2d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 2D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

size_t ZFPCompression::do_1d_compression(double* originalMatrix,
                                         unsigned char* outputArray) {
    // create a field
    zfp_field_set_pointer(field_1d, originalMatrix);

    // need to calculate the maximum size
    size_t bufsize    = zfp_stream_maximum_size(zfp1d, field_1d);

    // then we can open the stream, we go one past size_t to store room for the
    // final size needed in decompression
    bitstream* stream = stream_open(outputArray + sizeof(size_t), bufsize);

    // associate the bitstream with ZFP stream
    zfp_stream_set_bit_stream(zfp1d, stream);

    size_t outsize = zfp_compress(zfp1d, field_1d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR COMPRESSING DATA IN 1D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // close stream
    stream_close(stream);

    // make sure we store the number of bytes in our outsize!
    std::memcpy(outputArray, &outsize, sizeof(outsize));

    return outsize + sizeof(size_t);
}

size_t ZFPCompression::do_1d_decompression(unsigned char* compressedBuffer,
                                           double* outputArray) {
    // first extract out the buffer size
    size_t bufsize;

    std::memcpy(&bufsize, compressedBuffer, sizeof(size_t));

    bitstream* stream = stream_open(compressedBuffer + sizeof(size_t), bufsize);

    zfp_stream_set_bit_stream(zfp1d, stream);

    zfp_field_set_pointer(field_1d, outputArray);

    // do the decompression
    size_t outsize = zfp_decompress(zfp1d, field_1d);

    if (!outsize) {
        std::cerr << "CRITICAL ERROR DECOMPRESSING DATA IN 1D ZFP STREAM!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // make sure stream is closed
    stream_close(stream);

    // remember, this is for the raw buffer, as it includes that data that we're
    // working with
    return bufsize + sizeof(size_t);
}

}  // namespace ZFPAlgorithms

namespace BLOSCAlgorithms {

BloscCompression bloscblockwise(6, "lz4", 4, 1);

size_t BloscCompression::do_3d_compression(double* originalMatrix,
                                           unsigned char* outputArray) {
    // make sure the output array includes our header
    // std::cout << "attempting to compress " << blosc_original_bytes_3d
    //           << std::endl;
    int compressedSize = blosc_compress(clevel, doShuffle, sizeof(double),
                                        blosc_original_bytes_3d, originalMatrix,
                                        outputArray + sizeof(size_t),
                                        blosc_original_bytes_overhead_3d);

    // TODO: if compressed size is 0, we have to disregard the buffer
    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 3d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}
size_t BloscCompression::do_3d_decompression(unsigned char* compressedBuffer,
                                             double* outputArray) {
    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData =
        blosc_decompress(compressedBuffer + sizeof(size_t), outputArray,
                         blosc_original_bytes_3d);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 3d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

size_t BloscCompression::do_2d_compression(double* originalMatrix,
                                           unsigned char* outputArray) {
    // TODO: need some kind of better metric or way to know if we can compress
    // or not. Current idea is if it fails, we still do a copy. We attempt to
    // copy it back out into 32 bits and see if it's garbage? idk
    if (eleOrder <= 6) {
        std::memcpy(outputArray, originalMatrix, blosc_original_bytes_2d);
        return blosc_original_bytes_2d;
    }

    // make sure the output array includes our header
    int compressedSize = blosc_compress(clevel, doShuffle, sizeof(double),
                                        blosc_original_bytes_2d, originalMatrix,
                                        outputArray + sizeof(size_t),
                                        blosc_original_bytes_overhead_2d);

    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 2d!" << std::endl;
        exit(EXIT_FAILURE);
    } else if (compressedSize == blosc_original_bytes_overhead_2d) {
        // std::cerr << "ERROR: found a block that can't be compressed in 2D!"
        //           << std::endl;
        // exit(EXIT_FAILURE);
        // need some method of handling it a bit better
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}
size_t BloscCompression::do_2d_decompression(unsigned char* compressedBuffer,
                                             double* outputArray) {
    // TODO: see 2d_compression above, this needs to be handled better
    if (eleOrder <= 6) {
        std::memcpy(outputArray, compressedBuffer, blosc_original_bytes_2d);
        return blosc_original_bytes_2d;
    }
    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData =
        blosc_decompress(compressedBuffer + sizeof(size_t), outputArray,
                         blosc_original_bytes_2d);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 2d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

size_t BloscCompression::do_1d_compression(double* originalMatrix,
                                           unsigned char* outputArray) {
    // TODO: see 1d_compression above, this needs to be handled better
    if (eleOrder <= 6) {
        std::memcpy(outputArray, originalMatrix, blosc_original_bytes_1d);
        return blosc_original_bytes_1d;
    }

    // make sure the output array includes our header
    int compressedSize = blosc_compress(
        clevel, doShuffle, sizeof(double), blosc_original_bytes_1d,
        originalMatrix, outputArray + sizeof(size_t), blosc_original_bytes_1d);

    if (compressedSize < 0) {
        std::cerr << "Error compressing BLOSC in 1d!" << std::endl;
        exit(EXIT_FAILURE);
    } else if (compressedSize == 0) {
        // it failed to compress if we're at 0, which means garbage, so we want
        // to copy in the data
        std::cout << "FAILED in 1d Case" << std::endl;
    } else if (compressedSize == blosc_original_bytes_overhead_1d) {
        // we weren't able to get any compression!
    } else {
        // success
        std::cout << "SUCCCESS! Got a compressed 1d! Hooray!" << std::endl;
    }

    // loses precision only if compressedSize is less than 0, which we catch
    // above
    size_t outSize = (size_t)compressedSize;
    // store the value properly
    std::memcpy(outputArray, &outSize, sizeof(size_t));

    return outSize + sizeof(size_t);
}
size_t BloscCompression::do_1d_decompression(unsigned char* compressedBuffer,
                                             double* outputArray) {
    // TODO: see 1d_compression above, this needs to be handled better
    if (eleOrder <= 6) {
        std::memcpy(outputArray, compressedBuffer, blosc_original_bytes_1d);
        return blosc_original_bytes_1d;
    }

    // start by extracting the outSize
    size_t outSize;
    std::memcpy(&outSize, compressedBuffer, sizeof(size_t));

    // then do the decomrpession, we know the destination number of bytes
    int decompressedData =
        blosc_decompress(compressedBuffer + sizeof(size_t), outputArray,
                         blosc_original_bytes_1d);

    if (decompressedData < 0) {
        std::cerr << "Error decompressing BLOSC in 1d!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // return the number of bytes to advance the compressed buffer!
    return outSize + sizeof(size_t);
}

unsigned char* compressData(const char* blosc_compressor, int clevel, int n,
                            double* originalData, int& byteStreamSize) {
    blosc_set_compressor(blosc_compressor);
    int originalDataBytes         = n * sizeof(double);

    // Calculate the maximum possible size for the compressed data
    // This value is suggested to not be modified and does not affect the size
    // of the final compressed form
    int maxCompressedSize         = originalDataBytes + BLOSC_MAX_OVERHEAD;
    unsigned char* compressedData = new unsigned char[maxCompressedSize];

    // Blosc compression function: int blosc_ocmpress(int clevel, int shuffle,
    // size_t typesize, size_t nbytes, const void* src, void* dest, size_t
    // destsize); Parameters: clevel: Compression level (0-9, 0 being no
    // compression, 9 being maximum compression). shuffle: Bitshuffle filter for
    // data rearrangement.
    //          - Pass 0 for no shuffling, which can be used for data without a
    //          specific pattern or when compression speed is a priority.
    //          - Pass 1 for byte shuffle, which is effective for numerical data
    //          where each element is larger than a byte, as it aligns the least
    //          significant bits of the data types. Good for data with repeating
    //          patterns at the byte level.
    //          - Pass 2 for bit shuffle, which is more aggressive than byte
    //          shuffle and aligns the bits across data types. Useful for
    //          numerical data with repeating patterns at the bit level, often
    //          leading to better compression.
    // typesize: The size of the datatype in the array (in bytes).
    // nbytes: The number of bytes to compress from the source buffer.
    // src: Pointer to the data buffer to compress.
    // dest: Pointer to the buffer where the compressed data will be stored.
    // destsize: Maximum size of the destination buffer.
    int compressedSize =
        blosc_compress(clevel, 1, sizeof(double), originalDataBytes,
                       originalData, compressedData, maxCompressedSize);
    if (compressedSize < 0) {
        throw std::runtime_error("blosc could not compress data.");
    }

    // Allocate memory for the bytestream, including space for the size of the
    // original data
    byteStreamSize            = compressedSize + sizeof(originalDataBytes);
    unsigned char* bytestream = new unsigned char[byteStreamSize];
    // Copy compressed data to bytestream
    std::memcpy(bytestream, compressedData, compressedSize);
    // Pack originalDataBytes at the end of the bytestream
    std::memcpy(bytestream + compressedSize, &originalDataBytes,
                sizeof(originalDataBytes));
    delete[] compressedData;
    return bytestream;
}

double* decompressData(unsigned char* byteStream, int byteStreamSize) {
    // Check if byteStream is valid
    if (!byteStream || byteStreamSize <= 0) {
        return nullptr;
    }
    // Unpack originalDataBytes from the end of the byteStream
    int originalDataBytes;
    std::memcpy(&originalDataBytes,
                byteStream + (byteStreamSize - sizeof(originalDataBytes)),
                sizeof(originalDataBytes));
    double* decompressedData = new double[originalDataBytes / sizeof(double)];

    int decompressedSize =
        blosc_decompress(byteStream, decompressedData, originalDataBytes);

    // Check for decompression error
    if (decompressedSize < 0) {
        // Handle decompression error (e.g., return null or throw an exception)
        throw std::runtime_error("blosc could not decompress data.");
    }

    return decompressedData;
}

void decompressData(unsigned char* byteStream, int byteStreamSize,
                    double* outBuff) {
    // Check if byteStream is valid
    // if (!byteStream || byteStreamSize <= 0) {
    //     return nullptr;
    // }
    // Unpack originalDataBytes from the end of the byteStream

    int originalDataBytes;
    std::memcpy(&originalDataBytes,
                byteStream + (byteStreamSize - sizeof(originalDataBytes)),
                sizeof(originalDataBytes));
    // double* decompressedData = new double[originalDataBytes /
    // sizeof(double)];

    int decompressedSize =
        blosc_decompress(byteStream, outBuff, originalDataBytes);

    // Check for decompression error
    if (decompressedSize < 0) {
        // Handle decompression error (e.g., return null or throw an exception)
        throw std::runtime_error("blosc could not decompress data.");
    }
}
}  // namespace BLOSCAlgorithms

namespace dendro_compress {

CompressionType COMPRESSION_OPTION = CompressionType::ZFP;

void set_compression_options(CompressionType compT,
                             const CompressionOptions& compOpt) {
    dendro_compress::COMPRESSION_OPTION = compT;

    // std::cout << "Set compression option to: "
    //           << dendro_compress::COMPRESSION_OPTION << std::endl;

    // then set up the options for all types
    ZFPAlgorithms::zfpblockwise.setEleOrder(compOpt.eleOrder);
    if (compOpt.zfpMode == "accuracy") {
        ZFPAlgorithms::zfpblockwise.setAccuracy(compOpt.zfpAccuracyTolerance);
    } else if (compOpt.zfpMode == "rate") {
        ZFPAlgorithms::zfpblockwise.setRate(compOpt.zfpRate);
    }

    // set up for BLOSC
    BLOSCAlgorithms::bloscblockwise.setEleOrder(compOpt.eleOrder);
    BLOSCAlgorithms::bloscblockwise.setCompressor(compOpt.bloscCompressor);

    // set up for Chebyshev
    ChebyshevAlgorithms::cheby.set_compression_type(compOpt.eleOrder,
                                                    compOpt.chebyNReduced);
}

std::size_t single_block_compress_3d(double* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_3d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_3d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_3d_compression(buffer,
                                                                     bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_decompress_3d(unsigned char* buffer,
                                       double* bufferOut) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_3d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_3d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // chebyshev decompression
            return BLOSCAlgorithms::bloscblockwise.do_3d_decompression(
                buffer, bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_compress_2d(double* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_2d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_2d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_2d_compression(buffer,
                                                                     bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_decompress_2d(unsigned char* buffer,
                                       double* bufferOut) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_2d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_2d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_2d_decompression(
                buffer, bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_compress_1d(double* buffer, unsigned char* bufferOut,
                                     const size_t points_per_dim) {
    // check the compression option and do the compression
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp compression
            return ZFPAlgorithms::zfpblockwise.do_1d_compression(buffer,
                                                                 bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev compression
            return ChebyshevAlgorithms::cheby.do_1d_compression(buffer,
                                                                bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // zfp compression
            return BLOSCAlgorithms::bloscblockwise.do_1d_compression(buffer,
                                                                     bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN COMPRESSION OPTION FOUND IN COMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t single_block_decompress_1d(unsigned char* buffer,
                                       double* bufferOut) {
    switch (COMPRESSION_OPTION) {
        case dendro_compress::CompressionType::ZFP:
            // zfp decompression
            return ZFPAlgorithms::zfpblockwise.do_1d_decompression(buffer,
                                                                   bufferOut);
            break;
        case dendro_compress::CompressionType::CHEBYSHEV:
            // chebyshev decompression
            return ChebyshevAlgorithms::cheby.do_1d_decompression(buffer,
                                                                  bufferOut);
            break;
        case dendro_compress::CompressionType::BLOSC:
            // chebyshev decompression
            return BLOSCAlgorithms::bloscblockwise.do_1d_decompression(
                buffer, bufferOut);
            break;
        default:
            std::cerr << "UNKNOWN DECOMPRESSION OPTION FOUND IN DECOMPRESS 3D "
                      << COMPRESSION_OPTION << std::endl;
            exit(EXIT_FAILURE);
            break;
    }
}

std::size_t blockwise_compression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder) {
    unsigned char config;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1;
    const size_t total_points_1d = points_per_dim;
    const size_t total_points_2d = total_points_1d * points_per_dim;
    const size_t total_points_3d = total_points_2d * points_per_dim;

    // TODO: set the compression type elsewhere
    // ChebyshevAlgorithms::cheby.set_compression_type(eleorder, 2);
    // ChebyshevAlgorithms::cheby.print();

    std::size_t comp_offset      = 0;
    std::size_t orig_offset      = 0;

    for (size_t ib = 0; ib < numBlocks; ib++) {
        // decode the value
        config = blockConfiguration[blockConfigOffset + ib];

        xdim   = (((config >> 6) & 7u) == 1);
        ydim   = (((config >> 3) & 7u) == 1);
        zdim   = ((config & 7u) == 1);

        // get the "dimensionality" of the block
        ndim   = xdim + ydim + zdim;

        // now based on the ndim, we will set up our compression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                std::memcpy(compressBuffer + comp_offset, &buffer[orig_offset],
                            sizeof(double));
                comp_offset += sizeof(double);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_compress_1d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_compress_2d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_compress_3d(
                    &buffer[orig_offset], compressBuffer + comp_offset,
                    total_n_points);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise compression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }

    return comp_offset;
}

std::size_t blockwise_decompression(
    double* buffer, unsigned char* compressBuffer, const size_t numBlocks,
    const std::vector<unsigned char>& blockConfiguration,
    const size_t blockConfigOffset, const size_t eleorder) {
    unsigned char config;

    // booleans that store whether or not these dimensions are "active"
    bool xdim, ydim, zdim;
    uint32_t ndim;

    // these values are used to define the output side, since we're
    // decompressing back to our values. All of the decompression methods should
    // return how many bytes to advance the compression offset.
    size_t total_n_points        = 0;
    const size_t points_per_dim  = eleorder - 1;
    const size_t total_points_0d = 1;
    const size_t total_points_1d = points_per_dim;
    const size_t total_points_2d = total_points_1d * points_per_dim;
    const size_t total_points_3d = total_points_2d * points_per_dim;

    std::size_t comp_offset      = 0;
    std::size_t orig_offset      = 0;

    for (std::size_t ib = 0; ib < numBlocks; ib++) {
        config = blockConfiguration[blockConfigOffset + ib];

        xdim   = (((config >> 6) & 7u) == 1);
        ydim   = (((config >> 3) & 7u) == 1);
        zdim   = ((config & 7u) == 1);

        // get the "dimensionality" of the block
        ndim   = xdim + ydim + zdim;

        // now based on the ndim, we will use our decompression methods
        switch (ndim) {
            case 0:
                // no compression on a single point
                std::memcpy(&buffer[orig_offset], compressBuffer + comp_offset,
                            sizeof(double));
                comp_offset += sizeof(double);
                orig_offset += total_points_0d;
                break;
            case 1:
                comp_offset += single_block_decompress_1d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_1d;
                break;
            case 2:
                comp_offset += single_block_decompress_2d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_2d;
                break;
            case 3:
                comp_offset += single_block_decompress_3d(
                    compressBuffer + comp_offset, &buffer[orig_offset]);
                orig_offset += total_points_3d;
                break;
            default:
                std::cerr << "Invalid number of dimensions found when doing "
                             "blockwise decompression. Exiting!"
                          << std::endl;
                exit(0);
                break;
        }
    }
    return comp_offset;
}

std::ostream& operator<<(std::ostream& out, const CompressionOptions opts) {
    return out << "<Compression Options: eleorder " << opts.eleOrder
               << ", bloscCompressor " << opts.bloscCompressor
               << ", bloscCLevel " << opts.bloscClevel << ", bloscDoShuffle "
               << opts.bloscDoShuffle << ", zfpMode " << opts.zfpMode
               << ", zfpRate " << opts.zfpRate << ", zfpAccuracy "
               << opts.zfpAccuracyTolerance << ", chebyNReduced "
               << opts.chebyNReduced << ">";
}

std::ostream& operator<<(std::ostream& out, const CompressionType t) {
    return out << "<CompressionType: " << COMPRESSION_TYPE_NAMES[t] << ">";
}

}  // namespace dendro_compress
