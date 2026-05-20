// Phase-1 same-level scatter kernels for Mesh::unzip_scatter.
// Pure-integer index computation + contiguous row memcpy.
// See plan: same-level branch of unzip_scatter is a deterministic reindex;
// the original FP coordinate math + std::round/fabs in mesh.tcc:11091-11145 is unnecessary.

#ifndef DENDRO_MESH_UNZIP_SCATTER_KERNELS_H
#define DENDRO_MESH_UNZIP_SCATTER_KERNELS_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace dendro {
namespace unzip {

// Inputs are precomputed once per (element, block) pair in unzip_scatter.
//   dgWVec       : pointer to DG element values, layout dof * (eOrder+1)^3
//   uzWVec       : pointer to unzipped output buffer (full dof * unSz vector base)
//   dof          : number of variables (typically 1 in BSSN per-call)
//   unSz, dgSz   : per-variable strides in uzWVec and dgWVec
//   offset       : block base offset within uzWVec
//   lx, ly, lz   : block allocation extents (with padding)
//   i0, j0, k0   : signed start indices of this element inside the block array
//                  (may be negative when the element overlaps a ghost padding region)
//   The clip-to-block bounds [i_lo,i_hi) etc. are computed inside the kernel.

template <typename T, unsigned int EORDER>
inline void scatter_same_level_specialized(const T* __restrict__ dgWVec,
                                           T* __restrict__ uzWVec,
                                           unsigned int dof,
                                           std::size_t unSz, std::size_t dgSz,
                                           std::size_t offset,
                                           unsigned int lx, unsigned int ly,
                                           unsigned int lz, int i0, int j0,
                                           int k0) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "scatter kernel requires trivially copyable T");
    constexpr int eOp1   = (int)EORDER + 1;
    constexpr int eOp1Sq = eOp1 * eOp1;

    const int i_lo = std::max(0, -i0);
    const int i_hi = std::min(eOp1, (int)lx - i0);
    const int j_lo = std::max(0, -j0);
    const int j_hi = std::min(eOp1, (int)ly - j0);
    const int k_lo = std::max(0, -k0);
    const int k_hi = std::min(eOp1, (int)lz - k0);
    if (i_hi <= i_lo || j_hi <= j_lo || k_hi <= k_lo) return;

    const std::size_t row_bytes = (std::size_t)(i_hi - i_lo) * sizeof(T);
    const std::size_t lxy       = (std::size_t)lx * (std::size_t)ly;

    for (int k = k_lo; k < k_hi; ++k) {
        const std::size_t dst_zoff = (std::size_t)(k0 + k) * lxy;
        const std::size_t src_zoff = (std::size_t)k * (std::size_t)eOp1Sq;
        for (int j = j_lo; j < j_hi; ++j) {
            const std::size_t dst_off =
                offset + dst_zoff + (std::size_t)(j0 + j) * (std::size_t)lx +
                (std::size_t)(i0 + i_lo);
            const std::size_t src_off =
                src_zoff + (std::size_t)j * (std::size_t)eOp1 +
                (std::size_t)i_lo;
            for (unsigned int v = 0; v < dof; ++v) {
                std::memcpy(uzWVec + (std::size_t)v * unSz + dst_off,
                            dgWVec + (std::size_t)v * dgSz + src_off,
                            row_bytes);
            }
        }
    }
}

template <typename T>
inline void scatter_same_level_generic(const T* __restrict__ dgWVec,
                                       T* __restrict__ uzWVec,
                                       unsigned int eOrder, unsigned int dof,
                                       std::size_t unSz, std::size_t dgSz,
                                       std::size_t offset, unsigned int lx,
                                       unsigned int ly, unsigned int lz,
                                       int i0, int j0, int k0) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "scatter kernel requires trivially copyable T");
    const int eOp1   = (int)eOrder + 1;
    const int eOp1Sq = eOp1 * eOp1;

    const int i_lo = std::max(0, -i0);
    const int i_hi = std::min(eOp1, (int)lx - i0);
    const int j_lo = std::max(0, -j0);
    const int j_hi = std::min(eOp1, (int)ly - j0);
    const int k_lo = std::max(0, -k0);
    const int k_hi = std::min(eOp1, (int)lz - k0);
    if (i_hi <= i_lo || j_hi <= j_lo || k_hi <= k_lo) return;

    const std::size_t row_bytes = (std::size_t)(i_hi - i_lo) * sizeof(T);
    const std::size_t lxy       = (std::size_t)lx * (std::size_t)ly;

    for (int k = k_lo; k < k_hi; ++k) {
        const std::size_t dst_zoff = (std::size_t)(k0 + k) * lxy;
        const std::size_t src_zoff = (std::size_t)k * (std::size_t)eOp1Sq;
        for (int j = j_lo; j < j_hi; ++j) {
            const std::size_t dst_off =
                offset + dst_zoff + (std::size_t)(j0 + j) * (std::size_t)lx +
                (std::size_t)(i0 + i_lo);
            const std::size_t src_off =
                src_zoff + (std::size_t)j * (std::size_t)eOp1 +
                (std::size_t)i_lo;
            for (unsigned int v = 0; v < dof; ++v) {
                std::memcpy(uzWVec + (std::size_t)v * unSz + dst_off,
                            dgWVec + (std::size_t)v * dgSz + src_off,
                            row_bytes);
            }
        }
    }
}

template <typename T>
inline void scatter_same_level_dispatch(const T* __restrict__ dgWVec,
                                        T* __restrict__ uzWVec,
                                        unsigned int eOrder, unsigned int dof,
                                        std::size_t unSz, std::size_t dgSz,
                                        std::size_t offset, unsigned int lx,
                                        unsigned int ly, unsigned int lz,
                                        int i0, int j0, int k0) {
    switch (eOrder) {
        case 2:
            scatter_same_level_specialized<T, 2>(dgWVec, uzWVec, dof, unSz,
                                                 dgSz, offset, lx, ly, lz, i0,
                                                 j0, k0);
            return;
        case 4:
            scatter_same_level_specialized<T, 4>(dgWVec, uzWVec, dof, unSz,
                                                 dgSz, offset, lx, ly, lz, i0,
                                                 j0, k0);
            return;
        case 6:
            scatter_same_level_specialized<T, 6>(dgWVec, uzWVec, dof, unSz,
                                                 dgSz, offset, lx, ly, lz, i0,
                                                 j0, k0);
            return;
        case 8:
            scatter_same_level_specialized<T, 8>(dgWVec, uzWVec, dof, unSz,
                                                 dgSz, offset, lx, ly, lz, i0,
                                                 j0, k0);
            return;
        default:
            scatter_same_level_generic<T>(dgWVec, uzWVec, eOrder, dof, unSz,
                                          dgSz, offset, lx, ly, lz, i0, j0,
                                          k0);
            return;
    }
}

}  // namespace unzip
}  // namespace dendro

#endif  // DENDRO_MESH_UNZIP_SCATTER_KERNELS_H
