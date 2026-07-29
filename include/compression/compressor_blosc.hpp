#pragma once

#include <blosc.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "compression_base.hpp"

namespace dendrocompression {

/**
 * @brief LOSSLESS compression via c-blosc (byte-shuffle + lz4/zstd/etc).
 *
 * Its reason for existing is the correctness gate, not performance. A lossless
 * codec MUST reproduce the uncompressed exchange bit for bit, which makes it a
 * strictly stronger check than the dummy codec -- dummy is a memcpy and never
 * exercises a real codec's framing, sizing or fallback paths at all. Any
 * difference it produces is a plumbing bug, and needs no error budget to
 * interpret. Expect a poor ratio (~1.1-1.3x is typical on float data); that is
 * not the point.
 *
 * ONE CALL, ONE HEADER -- and this is the fix for why blosc was abandoned.
 * The branch's original BLOSCAlgorithms compressed each small scatter-map block
 * separately, so a 5-element edge (40 bytes) paid BLOSC_MAX_OVERHEAD of framing
 * and the recorded verdict became "1d and 2d won't compress enough". Here the
 * whole batch handed to do_compress_Nd is compressed as a single byte stream:
 * blosc does not care about the dimensional structure, and one header per call
 * instead of one per block removes that failure mode. It also keeps the header
 * count far below the sizeof(size_t)-per-block slack that alloc_mpi_ctx
 * reserves, which per-group framing would blow through at high dof.
 *
 * Wire format per call:
 *
 *     [ size_t n ][ n bytes of blosc payload ]      when compression helped
 *     [ size_t 0 ][ raw_bytes of original data ]    otherwise
 *
 * n == 0 is the raw sentinel; a blosc payload is never zero-length, so it is
 * unambiguous. The raw fallback fires whenever blosc cannot beat the input --
 * which for the smallest dimensionalities is most of the time, exactly as the
 * original verdict predicted.
 *
 * NOT threaded by DENDRO_COMPRESSION_OMP: the output length is data-dependent,
 * so output offsets are not known in advance and the disjoint-write argument
 * that makes the quantize codec's threading bit-exact does not apply. blosc's
 * own internal threading is used instead (blosc_set_nthreads).
 */
template <typename T>
class BloscCompressor : public Compression<T> {
   private:
    std::string codec_;
    int clevel_;
    int doshuffle_;

    /** @brief blosc_init() is process-global and c-blosc 1.x does not refcount
     *  it, so a per-object init/destroy pair would tear the library down while a
     *  sibling compressor (float vs double -- setUpCompressor builds both) still
     *  holds it. Init once, never destroy. */
    static void ensure_init() {
        static bool done = false;
        if (done) return;
        done = true;
        blosc_init();
        const int nt = blosc_get_nthreads();
        (void)nt;
    }

    std::size_t compress_run(const T* in, unsigned char* out,
                             std::size_t raw_bytes) const {
        std::size_t n = 0;
        // Budget: never write more than the raw form would have taken, so the
        // caller's worst-case sizing always holds. blosc returns 0 when it
        // cannot fit inside destsize, which is exactly the fallback condition.
        const std::size_t budget = raw_bytes;
        int rc = 0;
        if (raw_bytes > 0) {
            rc = blosc_compress(clevel_, doshuffle_, sizeof(T), raw_bytes, in,
                                out + sizeof(std::size_t), budget);
        }
        if (rc > 0 && (std::size_t)rc < raw_bytes) {
            n = (std::size_t)rc;
            std::memcpy(out, &n, sizeof(std::size_t));
            return sizeof(std::size_t) + n;
        }
        if (rc < 0) {
            throw std::runtime_error(
                "BloscCompressor: blosc_compress failed (internal error)");
        }
        // rc == 0 (did not fit) or rc >= raw_bytes (no gain) -> store raw
        n = 0;
        std::memcpy(out, &n, sizeof(std::size_t));
        std::memcpy(out + sizeof(std::size_t), in, raw_bytes);
        return sizeof(std::size_t) + raw_bytes;
    }

    std::size_t decompress_run(const unsigned char* in, T* out,
                               std::size_t raw_bytes) const {
        std::size_t n = 0;
        std::memcpy(&n, in, sizeof(std::size_t));
        if (n == 0) {
            std::memcpy(out, in + sizeof(std::size_t), raw_bytes);
            return sizeof(std::size_t) + raw_bytes;
        }
        const int rc = blosc_decompress(in + sizeof(std::size_t), out,
                                       raw_bytes);
        if (rc <= 0 || (std::size_t)rc != raw_bytes) {
            throw std::runtime_error(
                "BloscCompressor: blosc_decompress produced the wrong size");
        }
        return sizeof(std::size_t) + n;
    }

   public:
    BloscCompressor(unsigned int ele_order, unsigned int num_vars,
                    const std::string& codec = "lz4", int clevel = 5,
                    int doshuffle = 1)
        : Compression<T>(ele_order, num_vars),
          codec_(codec),
          clevel_(clevel),
          doshuffle_(doshuffle) {
        ensure_init();
        if (blosc_set_compressor(codec_.c_str()) < 0) {
            throw std::runtime_error("BloscCompressor: unknown blosc codec '" +
                                     codec_ + "'");
        }
    }

    ~BloscCompressor() = default;

    std::unique_ptr<Compression<T>> clone() const override {
        return std::make_unique<BloscCompressor>(*this);
    }

    CompressionType get_compression_type() const override {
        return CompressionType::COMP_BLOSC;
    }

    std::string to_string() const override {
        std::ostringstream os;
        os << "BloscCompressor<" << codec_ << ", clevel=" << clevel_
           << ", shuffle=" << doshuffle_ << "> (lossless)";
        return os.str();
    }

    std::size_t do_compress_3d(T* const in, unsigned char* const out,
                               unsigned int batch) override {
        return compress_run(in, out, (std::size_t)batch * this->total_3d_bytes_);
    }
    std::size_t do_decompress_3d(unsigned char* const in, T* const out,
                                 unsigned int batch) override {
        return decompress_run(in, out,
                              (std::size_t)batch * this->total_3d_bytes_);
    }

    std::size_t do_compress_2d(T* const in, unsigned char* const out,
                               unsigned int batch) override {
        return compress_run(in, out, (std::size_t)batch * this->total_2d_bytes_);
    }
    std::size_t do_decompress_2d(unsigned char* const in, T* const out,
                                 unsigned int batch) override {
        return decompress_run(in, out,
                              (std::size_t)batch * this->total_2d_bytes_);
    }

    std::size_t do_compress_1d(T* const in, unsigned char* const out,
                               unsigned int batch) override {
        return compress_run(in, out, (std::size_t)batch * this->total_1d_bytes_);
    }
    std::size_t do_decompress_1d(unsigned char* const in, T* const out,
                                 unsigned int batch) override {
        return decompress_run(in, out,
                              (std::size_t)batch * this->total_1d_bytes_);
    }

    std::size_t do_compress_0d(T* const in, unsigned char* const out,
                               unsigned int batch) override {
        return compress_run(in, out, (std::size_t)batch * this->total_0d_bytes_);
    }
    std::size_t do_decompress_0d(unsigned char* const in, T* const out,
                                 unsigned int batch) override {
        return decompress_run(in, out,
                              (std::size_t)batch * this->total_0d_bytes_);
    }

    std::size_t do_compress_flat(T* const in, unsigned char* const out,
                                 unsigned int n_pts) override {
        return compress_run(in, out, (std::size_t)n_pts * this->total_0d_bytes_);
    }
    std::size_t do_decompress_flat(unsigned char* const in, T* const out,
                                   unsigned int n_pts) override {
        return decompress_run(in, out,
                              (std::size_t)n_pts * this->total_0d_bytes_);
    }
};

}  // namespace dendrocompression
