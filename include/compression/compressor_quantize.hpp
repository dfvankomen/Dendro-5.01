#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "compression_base.hpp"

#ifdef DENDRO_COMPRESSION_OMP
#include <omp.h>
#endif

namespace dendrocompression {

#ifdef DENDRO_COMPRESSION_OMP
/**
 * @brief Groups a thread needs before it is worth waking.
 *
 * Real per-peer payloads span roughly 43..2400 groups (measured on a distributed
 * mesh -- see the scatter-map profile), so one run hits both the
 * too-little-work and the plenty-of-work regime and the thread count has to
 * follow the work. Uncapped, a 43-group peer peaks at 4 threads and loses ~15%
 * of that going to 8; capped too aggressively it loses far more by dropping to
 * serial.
 *
 * Swept 4 / 8 / 16 / 32 groups-per-thread against no cap (16-bit, compress,
 * 8 threads): at 43 groups no-cap 14.9, gpt=4 16.8, gpt=8 18.0, gpt=16 15.1,
 * gpt=32 9.8 GB/s (32 forces serial and throws away a 2x win -- it was a bad
 * first guess). At 9600 groups all caps land ~75 vs 66 uncapped. 8 never loses
 * to no-cap and is best at the small end, so: 8. Run-to-run spread is ~10%, so
 * 4 and 8 are not meaningfully separable -- the point is that it is single
 * digits, not tens.
 *
 * Capping threads here is safe in a way it would not be for the solver's block
 * loops: this codec keeps no per-thread workspace and does no tid indexing, so
 * there is no pool to stay pinned to, and the output is byte-identical at any
 * thread count.
 */
constexpr unsigned int QUANT_GROUPS_PER_THREAD = 8u;

inline int quant_nthreads(unsigned int ngroups) {
    const int want = (int)(ngroups / QUANT_GROUPS_PER_THREAD);
    const int have = omp_get_max_threads();
    return want < 1 ? 1 : (want < have ? want : have);
}
#endif

/**
 * @brief Fixed-point quantization against a per-(block,variable) absmax.
 *
 * Layout, per scaling group (one variable's worth of one block, npg points):
 *
 *     [ T scale ][ Q q_0 ][ Q q_1 ] ... [ Q q_{npg-1} ]
 *
 * where Q is int8_t or int16_t and q_i = round(v_i * QMAX / scale).
 *
 * Three properties make this a good fit for the ghost exchange, and they are
 * the reason it exists alongside ZFP:
 *
 *  1. THE RATIO IS FIXED BY CONSTRUCTION and data-independent -- exactly
 *     npg*sizeof(T) / (sizeof(T) + npg*sizeof(Q)). Nothing has to be measured,
 *     buffers can be sized statically, and the exchange's variable-length
 *     framing becomes unnecessary (though it still works).
 *  2. IT IS SCALE INVARIANT. The per-group scale means a block whose values are
 *     all ~1e-30 still gets full Q-bit resolution *of that range*. A fixed
 *     reduced-precision format (fp16) instead has an absolute magnitude floor
 *     and flushes such a block toward zero. Given how much dynamic range a GR
 *     field spans between the wave zone and a puncture, that matters.
 *  3. IT IS EMBARRASSINGLY PARALLEL WITH DISJOINT WRITES. Every group is
 *     independent and its output offset is known in advance (fixed size), so
 *     the threaded path writes exactly the same bytes as the serial one -- see
 *     DENDRO_COMPRESSION_OMP below.
 *
 * Error is bounded at half a quantization step: |err| <= scale / (2*QMAX), i.e.
 * relative to the group's own largest magnitude. 16-bit gives ~1.5e-5 of the
 * block max, which is the same order as the double->float cast the exchange
 * already performs; 8-bit gives ~4e-3 and is very likely too coarse for a
 * stencil without a physics study behind it.
 *
 * Degenerate dimensionalities: for small groups the 4-byte scale header is not
 * amortized -- with n=5 and float, a 0-D group is 1 point and quantizing it
 * would EXPAND 4 bytes to 6. Whether each dimensionality quantizes at all is
 * therefore decided once in the constructor (quantizes_Nd_) from sizes alone,
 * so both directions agree without any per-group flag on the wire and the
 * output stays fixed-size.
 */
template <typename T, typename Q>
class QuantizeCompressor : public Compression<T> {
   private:
    // largest representable magnitude of the signed quantized type
    static constexpr int QMAX_ =
        (1 << (8 * static_cast<int>(sizeof(Q)) - 1)) - 1;

    // points per scaling group (one variable of one block), per dimensionality
    unsigned int npg_3d_, npg_2d_, npg_1d_, npg_0d_;
    // bytes on the wire per group, per dimensionality
    std::size_t gb_3d_, gb_2d_, gb_1d_, gb_0d_;
    // does quantizing this dimensionality actually shrink it?
    bool q_3d_, q_2d_, q_1d_, q_0d_;

    static constexpr std::size_t group_bytes(unsigned int npg, bool quantized) {
        return quantized ? sizeof(T) + npg * sizeof(Q) : npg * sizeof(T);
    }
    static constexpr bool pays(unsigned int npg) {
        return sizeof(T) + npg * sizeof(Q) < npg * sizeof(T);
    }

    /**
     * @brief Compress `ngroups` groups of `npg` points each.
     *
     * Each group is read from `in + g*npg` and written to `out + g*gb`, both
     * disjoint across g, which is what makes the parallel form bit-identical.
     */
    std::size_t compress_groups(T* const in, unsigned char* const out,
                                unsigned int ngroups, unsigned int npg,
                                std::size_t gb, bool quantized) const {
        if (!quantized) {
            std::memcpy(out, in, (std::size_t)ngroups * npg * sizeof(T));
            return (std::size_t)ngroups * gb;
        }
#ifdef DENDRO_COMPRESSION_OMP
#pragma omp parallel for num_threads(quant_nthreads(ngroups))
#endif
        for (unsigned int g = 0; g < ngroups; ++g) {
            const T* src         = in + (std::size_t)g * npg;
            unsigned char* dst   = out + (std::size_t)g * gb;

            // The absmax reduction is the compress path's bottleneck if left as
            // a plain loop: GCC will not reassociate a floating-point max
            // reduction without -ffast-math, so it becomes a serial dependency
            // chain while the quantize loop below vectorizes freely (verified
            // with -fopt-info-vec-missed). `omp simd reduction(max:)` grants the
            // reassociation explicitly.
            //
            // Reassociating is safe here regardless of NaN: `a > amax ? a : amax`
            // and max() both drop NaN, so every lane ignores it and the
            // horizontal combine does too -- a NaN in the input yields the same
            // amax either way.
            T amax = T(0);
#ifdef _OPENMP
#pragma omp simd reduction(max : amax)
#endif
            for (unsigned int i = 0; i < npg; ++i) {
                const T a = std::abs(src[i]);
                amax      = a > amax ? a : amax;
            }
            std::memcpy(dst, &amax, sizeof(T));

            // amax == 0 (an exactly flat group) => scale 0 => all codes 0, and
            // decompress reproduces zeros. No special case needed.
            const T sc = amax > T(0) ? T(QMAX_) / amax : T(0);
            Q* q       = reinterpret_cast<Q*>(dst + sizeof(T));
            for (unsigned int i = 0; i < npg; ++i) {
                // round-half-away-from-zero without a libm call; lrintf/nearbyint
                // measured ~2.7x slower here because they do not vectorize.
                // |src[i]*sc| <= QMAX_ by construction, so +-0.5 then truncating
                // lands in [-QMAX_, QMAX_] and cannot overflow Q.
                const T v = src[i] * sc;
                q[i]      = static_cast<Q>(
                    static_cast<int>(v + (v < T(0) ? T(-0.5) : T(0.5))));
            }
        }
        return (std::size_t)ngroups * gb;
    }

    std::size_t decompress_groups(unsigned char* const in, T* const out,
                                  unsigned int ngroups, unsigned int npg,
                                  std::size_t gb, bool quantized) const {
        if (!quantized) {
            std::memcpy(out, in, (std::size_t)ngroups * npg * sizeof(T));
            return (std::size_t)ngroups * gb;
        }
#ifdef DENDRO_COMPRESSION_OMP
#pragma omp parallel for num_threads(quant_nthreads(ngroups))
#endif
        for (unsigned int g = 0; g < ngroups; ++g) {
            const unsigned char* src = in + (std::size_t)g * gb;
            T* dst                   = out + (std::size_t)g * npg;

            T amax;
            std::memcpy(&amax, src, sizeof(T));
            const T isc    = amax / T(QMAX_);
            const Q* q     = reinterpret_cast<const Q*>(src + sizeof(T));
            for (unsigned int i = 0; i < npg; ++i) {
                dst[i] = static_cast<T>(q[i]) * isc;
            }
        }
        return (std::size_t)ngroups * gb;
    }

   public:
    QuantizeCompressor(unsigned int ele_order, unsigned int num_vars)
        : Compression<T>(ele_order, num_vars) {
        const unsigned int n = this->n_;
        npg_3d_              = n * n * n;
        npg_2d_              = n * n;
        npg_1d_              = n;
        npg_0d_              = 1;

        q_3d_                = pays(npg_3d_);
        q_2d_                = pays(npg_2d_);
        q_1d_                = pays(npg_1d_);
        q_0d_                = pays(npg_0d_);

        gb_3d_               = group_bytes(npg_3d_, q_3d_);
        gb_2d_               = group_bytes(npg_2d_, q_2d_);
        gb_1d_               = group_bytes(npg_1d_, q_1d_);
        gb_0d_               = group_bytes(npg_0d_, q_0d_);
    }

    ~QuantizeCompressor() = default;

    std::unique_ptr<Compression<T>> clone() const override {
        return std::make_unique<QuantizeCompressor>(*this);
    }

    CompressionType get_compression_type() const override {
        return CompressionType::COMP_QUANT;
    }

    std::string to_string() const override {
        std::ostringstream os;
        os << "QuantizeCompressor<" << (8 * sizeof(Q)) << "-bit>"
           << " [3d " << (q_3d_ ? "on" : "raw") << " 2d "
           << (q_2d_ ? "on" : "raw") << " 1d " << (q_1d_ ? "on" : "raw")
           << " 0d " << (q_0d_ ? "on" : "raw") << "]";
        return os.str();
    }

    std::size_t do_compress_3d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        return compress_groups(original_matrix, output_array,
                               batch_size * this->num_vars_, npg_3d_, gb_3d_,
                               q_3d_);
    }
    std::size_t do_decompress_3d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        return decompress_groups(compressed_buffer, output_array,
                                 batch_size * this->num_vars_, npg_3d_, gb_3d_,
                                 q_3d_);
    }

    std::size_t do_compress_2d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        return compress_groups(original_matrix, output_array,
                               batch_size * this->num_vars_, npg_2d_, gb_2d_,
                               q_2d_);
    }
    std::size_t do_decompress_2d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        return decompress_groups(compressed_buffer, output_array,
                                 batch_size * this->num_vars_, npg_2d_, gb_2d_,
                                 q_2d_);
    }

    std::size_t do_compress_1d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        return compress_groups(original_matrix, output_array,
                               batch_size * this->num_vars_, npg_1d_, gb_1d_,
                               q_1d_);
    }
    std::size_t do_decompress_1d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        return decompress_groups(compressed_buffer, output_array,
                                 batch_size * this->num_vars_, npg_1d_, gb_1d_,
                                 q_1d_);
    }

    std::size_t do_compress_0d(T* const original_matrix,
                               unsigned char* const output_array,
                               unsigned int batch_size) override {
        return compress_groups(original_matrix, output_array,
                               batch_size * this->num_vars_, npg_0d_, gb_0d_,
                               q_0d_);
    }
    std::size_t do_decompress_0d(unsigned char* const compressed_buffer,
                                 T* const output_array,
                                 unsigned int batch_size) override {
        return decompress_groups(compressed_buffer, output_array,
                                 batch_size * this->num_vars_, npg_0d_, gb_0d_,
                                 q_0d_);
    }

    /**
     * @brief Flat (unstructured) run of n_pts * num_vars values.
     *
     * Treated as one group per variable-sized chunk so the scale still adapts,
     * rather than one global scale over the whole run.
     */
    std::size_t do_compress_flat(T* const original_matrix,
                                 unsigned char* const output_array,
                                 unsigned int n_pts) override {
        return compress_groups(original_matrix, output_array, n_pts, npg_0d_,
                               gb_0d_, q_0d_);
    }
    std::size_t do_decompress_flat(unsigned char* const compressed_buffer,
                                   T* const output_array,
                                   unsigned int n_pts) override {
        return decompress_groups(compressed_buffer, output_array, n_pts,
                                 npg_0d_, gb_0d_, q_0d_);
    }
};

template <typename T>
using Quantize16Compressor = QuantizeCompressor<T, std::int16_t>;
template <typename T>
using Quantize8Compressor  = QuantizeCompressor<T, std::int8_t>;

}  // namespace dendrocompression
