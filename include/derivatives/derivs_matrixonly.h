
#pragma once

#include <stdexcept>
#include <unordered_map>

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_utils.h"

// in-matrix filter options
#include "derivatives/filt_inmat.h"
#include "derivatives/filt_inmat_byufilter.h"
#include "derivatives/filt_inmat_kim.h"

#define DDERIVS_MAX_BLOCKS_INIT 8

namespace dendroderivs {

namespace detail {
template <typename T>
inline std::unique_ptr<InMatrixFilter> make_filter(
    const std::vector<double>& c) {
    return std::make_unique<T>(c);
}
using InMatrixFilterMaker =
    std::unique_ptr<InMatrixFilter> (*)(const std::vector<double>&);
}  // namespace detail

// registry-based dispatch — adding a new filter is one line in the map.
// inline-scoped so we keep the header-only setup; the map is constructed
// once per TU (acceptable; each entry is just two pointers)
inline std::unique_ptr<InMatrixFilter> createInMatrixFilterByType(
    const std::string &in_matrix_filter,
    const std::vector<double> &in_matrix_filter_coeffs) {
    static const std::unordered_map<std::string, detail::InMatrixFilterMaker>
        registry = {
            {"none",       detail::make_filter<NoneFilter_InMatrix>},
            {"BYUT4",      detail::make_filter<BYUT4Filter_InMatrix>},
            {"BYUT6",      detail::make_filter<BYUT6Filter_InMatrix>},
            {"BYUT8",      detail::make_filter<BYUT8Filter_InMatrix>},
            {"KIM",        detail::make_filter<KimFilter_InMatrix>},
            {"KIM_1_P6",   detail::make_filter<Kim1P6Filter_InMatrix>},
            {"KIM_2_P6",   detail::make_filter<Kim2P6Filter_InMatrix>},
            {"KIM_3_P6",   detail::make_filter<Kim3P6Filter_InMatrix>},
            {"KIM_4_P6",   detail::make_filter<Kim4P6Filter_InMatrix>},
            {"KIM_P6",     detail::make_filter<KimP6Filter_InMatrix>},
            {"A4",         detail::make_filter<A4_Filter_InMatrix>},
            {"KIM_06_P6",  detail::make_filter<Kim_06_P6_Filter_InMatrix>},
            {"KIM_075_P6", detail::make_filter<Kim_075_P6_Filter_InMatrix>},
            {"KIM_08_P6",  detail::make_filter<Kim_08_P6_Filter_InMatrix>},
            {"KIM_085_P6", detail::make_filter<Kim_085_P6_Filter_InMatrix>},
            {"KIM_09_P6",  detail::make_filter<Kim_09_P6_Filter_InMatrix>},
            {"KIM_09_P2",  detail::make_filter<Kim_09_P2_Filter_InMatrix>},
            {"KIM_08_P2",  detail::make_filter<Kim_08_P2_Filter_InMatrix>},
        };

    auto it = registry.find(in_matrix_filter);
    if (it == registry.end()) {
        throw std::invalid_argument("Unsupported 'In-Matrix' Filter Type: " +
                                    in_matrix_filter);
    }
    return it->second(in_matrix_filter_coeffs);
}

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage> createMatrixSystemForSingleSize(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries *diagEntries,
    const bool skip_leftright = false);

template <unsigned int DerivOrder>
std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeAllUniqueDiags(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries *diagEntries,
    const MatrixDiagonalEntries *diagEntriesLeft,
    const MatrixDiagonalEntries *diagEntriesRight,
    const MatrixDiagonalEntries *diagEntriesLeftRight,
    const bool skip_leftright = false);

template <unsigned int DerivOrder>
class MatrixCompactDerivs : public CompactDerivs {
   protected:
    std::vector<double> workspace_;
    unsigned int workspace_tot_;

    std::unordered_map<unsigned int, std::unique_ptr<DerivMatrixStorage>>
        D_storage_map_;

    // memoize the last-used storage pointer — avoids hash lookup when
    // block size doesn't change between calls (the common case)
    mutable unsigned int _cached_sz = 0;
    mutable DerivMatrixStorage *_cached_storage = nullptr;

    // interior and bounded entries for each P and Q matrix
    MatrixDiagonalEntries *diagEntries = nullptr;

    std::unique_ptr<InMatrixFilter> in_matrix_filter_;

   public:
    MatrixCompactDerivs(unsigned int ele_order,
                        const std::string &in_matrix_filter = "none",
                        const std::vector<double> &in_matrix_filter_coeffs =
                            std::vector<double>())
        : CompactDerivs{ele_order} {
        // establish workspace to be as large as our largest
        unsigned int workspace_size_calc = p_n * p_n * p_n * 2;
        workspace_        = std::vector<double>(workspace_size_calc, 0.0);
        workspace_tot_    = workspace_size_calc;

        // then call and build up the in_matrix_filter_
        in_matrix_filter_ = createInMatrixFilterByType(in_matrix_filter,
                                                       in_matrix_filter_coeffs);
    }

    ~MatrixCompactDerivs() {
        // make sure diagEntries is properly deleted to avoid memory leak
        delete diagEntries;
    }

    void set_maximum_block_size(size_t block_size) {
        workspace_tot_ = block_size * 2;
        workspace_.resize(workspace_tot_);
    }

   protected:
    // defensive: matmul_y_dim/matmul_z_dim need 2*Nx*Ny*Nz scratch. if the
    // user forgot to call set_maximum_block_size, or sees a block bigger than
    // anything previously seen, grow the workspace lazily. fast path is a
    // single size_t compare — cost is irrelevant next to the matmul itself.
    inline void ensure_workspace_for(const unsigned int *sz) {
        const size_t needed =
            size_t{2} * size_t{sz[0]} * size_t{sz[1]} * size_t{sz[2]};
        if (workspace_.size() < needed) {
            workspace_.resize(needed);
            workspace_tot_ = static_cast<unsigned int>(needed);
        }
    }

   public:

    /**
     * Pre-create derivative matrices for a specific grid dimension size.
     * Call at mesh setup to avoid lazy creation during timestepping.
     */
    void pre_create_for_size(unsigned int n) {
        get_storage_for_size(n);
    }

    /**
     * @brief Deep-copy constructor. Needed so `clone()` can duplicate the
     * instance, since MatrixCompactDerivs owns the MatrixDiagonalEntries
     * (raw pointer), a workspace vector, and the per-size D storage map.
     * Prefer `clone()` over direct copy; this is the mechanism that backs it.
     */
    MatrixCompactDerivs(const MatrixCompactDerivs &obj) : CompactDerivs(obj) {
        if (obj.diagEntries) {
            diagEntries = new MatrixDiagonalEntries{
                obj.diagEntries->PDiagInterior, obj.diagEntries->PDiagBoundary,
                obj.diagEntries->QDiagInterior, obj.diagEntries->QDiagBoundary};
        } else {
            diagEntries = nullptr;
        }

        workspace_ = obj.workspace_;

        for (const auto &pair : obj.D_storage_map_) {
            D_storage_map_[pair.first] =
                pair.second ? std::make_unique<DerivMatrixStorage>(*pair.second)
                            : nullptr;
        }
    }

    // fast-path storage lookup: integer compare for the common case,
    // falls back to hash map + lazy creation if the size changed
    DerivMatrixStorage *get_storage_for_size(unsigned int n) {
        if (n == _cached_sz && _cached_storage) {
            return _cached_storage;
        }

        auto it = D_storage_map_.find(n);
        if (it != D_storage_map_.end()) {
            _cached_sz      = n;
            _cached_storage = it->second.get();
            return _cached_storage;
        }

        // first time seeing this size — create matrices
        D_storage_map_.emplace(
            n, createMatrixSystemForSingleSize<DerivOrder>(
                   p_pw, n, diagEntries, false));

        _cached_sz      = n;
        _cached_storage = D_storage_map_[n].get();
        return _cached_storage;
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        auto *storage = get_storage_for_size(sz[0]);
        auto *D_use   = get_deriv_mat_by_bflag_x(storage, bflag);

        if constexpr (DerivOrder == 1) {
            matmul_x_dim(D_use->data(), du, u, 1.0 / dx, sz, bflag, p_pw);
        } else {
            matmul_x_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz, bflag,
                         p_pw);
        }
    }

    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        auto *storage = get_storage_for_size(sz[1]);
        auto *D_use   = get_deriv_mat_by_bflag_y(storage, bflag);
        ensure_workspace_for(sz);

        if constexpr (DerivOrder == 1) {
            matmul_y_dim(D_use->data(), du, u, 1.0 / dx, sz, workspace_.data(),
                         bflag, p_pw);
        } else {
            matmul_y_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data(), bflag, p_pw);
        }
    }

    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        auto *storage = get_storage_for_size(sz[2]);
        auto *D_use   = get_deriv_mat_by_bflag_z(storage, bflag);
        ensure_workspace_for(sz);

        if constexpr (DerivOrder == 1) {
            matmul_z_dim(D_use->data(), du, u, 1.0 / dx, sz, workspace_.data(),
                         bflag, p_pw);
        } else {
            matmul_z_dim(D_use->data(), du, u, 1.0 / (dx * dx), sz,
                         workspace_.data(), bflag, p_pw);
        }
    }

    // "_last" overrides: pass is_last_op=true to the matmul so it can
    // skip the y/z output padding. See the contract in
    // derivatives.h::do_grad_x_last.
    void do_grad_x_last(double *const du, const double *const u,
                        const double dx, const unsigned int *sz,
                        const unsigned int bflag) override {
        auto *storage      = get_storage_for_size(sz[0]);
        auto *D_use        = get_deriv_mat_by_bflag_x(storage, bflag);
        const double alpha = (DerivOrder == 1) ? 1.0 / dx : 1.0 / (dx * dx);
        matmul_x_dim(D_use->data(), du, u, alpha, sz, bflag, p_pw,
                     /*is_last_op=*/true);
    }

    void do_grad_y_last(double *const du, const double *const u,
                        const double dx, const unsigned int *sz,
                        const unsigned int bflag) override {
        auto *storage      = get_storage_for_size(sz[1]);
        auto *D_use        = get_deriv_mat_by_bflag_y(storage, bflag);
        const double alpha = (DerivOrder == 1) ? 1.0 / dx : 1.0 / (dx * dx);
        ensure_workspace_for(sz);
        matmul_y_dim(D_use->data(), du, u, alpha, sz, workspace_.data(),
                     bflag, p_pw, /*is_last_op=*/true);
    }

    // Fused d^2u/dxdy in a single call. Per active z-slice (k in [pw, nz-pw))
    //   tmp = D_x * u_slice            (full nx x ny, intermediate in L1)
    //   w_slice = tmp * D_y^T          (active y region only)
    // Saves vs. the chain grad_x; grad_y_last:
    //   - no full-z intermediate written to RAM (chain's grad_x must write all
    //     of v because the chain treats grad_x as a black-box operation; the
    //     fused function knows only active z is needed downstream)
    //   - ~46% FLOP reduction (chain's grad_x computes z-padding cells too)
    // Only available for 1st-order MatrixCompactDerivs (DerivOrder == 1);
    // mixed 2nd-derivs use two 1st-order applications, so this is the
    // relevant order. Output: w in i in [0, nx), j in [pw, ny-pw),
    // k in [pw, nz-pw). The y- and z-padding cells of w are NOT written.
    void do_grad_xy_last(double *const w, const double *const u,
                         const double dx, const unsigned int *sz,
                         const unsigned int bflag) {
        static_assert(DerivOrder == 1,
                      "do_grad_xy_last is only for 1st-order MatrixCompactDerivs");
        const unsigned int nx = sz[0];
        const unsigned int ny = sz[1];
        const unsigned int nz = sz[2];
        const unsigned int pw = this->p_pw;

        auto *sx = this->get_storage_for_size(nx);
        auto *sy = this->get_storage_for_size(ny);
        auto *Dx = get_deriv_mat_by_bflag_x(sx, bflag);
        auto *Dy = get_deriv_mat_by_bflag_y(sy, bflag);

        // pre-scale both Ds by 1/dx so each per-slice GEMM applies 1/dx,
        // and the product applies 1/dx^2 as required for d^2/dxdy
        const double alpha = 1.0 / dx;
        std::vector<double> Dx_scaled(nx * nx);
        std::vector<double> Dy_scaled(ny * ny);
        for (size_t i = 0; i < (size_t)nx * nx; i++)
            Dx_scaled[i] = Dx->data()[i] * alpha;
        for (size_t i = 0; i < (size_t)ny * ny; i++)
            Dy_scaled[i] = Dy->data()[i] * alpha;

        // per-slice intermediate buffer; (nx * ny) doubles fits in L1 at
        // typical block sizes. Sized once.
        std::vector<double> tmp((size_t)nx * ny);

        const unsigned int y_start   = pw;
        const unsigned int y_end     = ny - pw;
        const unsigned int ny_active = y_end - y_start;
        const unsigned int slice_sz  = nx * ny;
        const unsigned int z_start   = pw;
        const unsigned int z_end     = nz - pw;
        const unsigned int y_off     = y_start * nx;

        // Per-slice fused loop. tmp lives in L1 (only nx*ny doubles).
        // LIBXSMM JITs better kernels for small N=ny=13 than for the
        // batched N=ny*nz_active=91 alternative, and the per-slice
        // approach keeps tmp hot in L1 between the x and y steps.
        auto kx = get_or_create_kernel_x(nx, ny, nx);
        KernelType ky_skip(LIBXSMM_GEMM_FLAG_TRANS_B, nx, ny_active, ny,
                           nx, ny, nx, 1.0, 0.0);

        if (!kx || !ky_skip) {
            this->do_grad_x(tmp.data(), u, dx, sz, bflag);
            this->do_grad_y_last(w, tmp.data(), dx, sz, bflag);
            return;
        }

#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw_kx = kx.kernel();
        libxsmm_gemmfunction raw_ky = ky_skip.kernel();
        if (raw_kx && raw_ky) {
            libxsmm_gemm_param a_args, b_args;
            a_args.a.primary = (void *)Dx_scaled.data();
            a_args.c.primary = (void *)tmp.data();
            b_args.a.primary = (void *)tmp.data();
            b_args.b.primary = (void *)(Dy_scaled.data() + y_start);
            for (unsigned int k = z_start; k < z_end; k++) {
                a_args.b.primary = (void *)(u + k * slice_sz);
                raw_kx(&a_args);
                b_args.c.primary = (void *)(w + k * slice_sz + y_off);
                raw_ky(&b_args);
            }
            return;
        }
#endif
        for (unsigned int k = z_start; k < z_end; k++) {
            const double *u_slice = u + k * slice_sz;
            double *w_slice       = w + k * slice_sz;
            kx(Dx_scaled.data(), u_slice, tmp.data());
            ky_skip(tmp.data(), Dy_scaled.data() + y_start,
                    w_slice + y_off);
        }
    }

    // Fused d^2u/dxdz in a single call. Per active y-slab (j in [pw, ny-pw))
    //   tmp(nx, nz) = D_x * u_at_j(nx, nz, strided)
    //   w[:, j, z_active] = tmp * D_z^T (strided output)
    // Chain comparison: grad_x then grad_z. grad_z already skips y-padding
    // (project convention) so it reads tmp only at active y; grad_x writes
    // tmp at all y, wasting ~46% on the y-padding cells. Fused only
    // computes tmp at active y.
    void do_grad_xz_last(double *const w, const double *const u,
                         const double dx, const unsigned int *sz,
                         const unsigned int bflag) {
        static_assert(DerivOrder == 1,
                      "do_grad_xz_last is only for 1st-order MatrixCompactDerivs");
        const unsigned int nx = sz[0];
        const unsigned int ny = sz[1];
        const unsigned int nz = sz[2];
        const unsigned int pw = this->p_pw;

        auto *sx  = this->get_storage_for_size(nx);
        auto *sz_s = this->get_storage_for_size(nz);
        auto *Dx  = get_deriv_mat_by_bflag_x(sx, bflag);
        auto *Dz  = get_deriv_mat_by_bflag_z(sz_s, bflag);

        const double alpha = 1.0 / dx;
        std::vector<double> Dx_scaled(nx * nx);
        std::vector<double> Dz_scaled(nz * nz);
        for (size_t i = 0; i < (size_t)nx * nx; i++)
            Dx_scaled[i] = Dx->data()[i] * alpha;
        for (size_t i = 0; i < (size_t)nz * nz; i++)
            Dz_scaled[i] = Dz->data()[i] * alpha;

        std::vector<double> tmp((size_t)nx * nz);

        const unsigned int y_start   = pw;
        const unsigned int y_end     = ny - pw;
        const unsigned int z_start   = pw;
        const unsigned int z_end     = nz - pw;
        const unsigned int nz_active = z_end - z_start;
        const unsigned int ld_3d     = nx * ny;

        // x kernel: tmp(nx, nz) = D_x_scaled(nx, nx) * u_at_j(nx, nz, strided)
        //   M=nx, N=nz, K=nx; LDA=nx, LDB=ld_3d, LDC=nx; no trans
        KernelType kx_strided(LIBXSMM_GEMM_FLAG_NONE, nx, nz, nx,
                              nx, ld_3d, nx, 1.0, 0.0);
        // z kernel: w_at_j[:, z_active](nx, nz_active) = tmp * D_z^T
        //   M=nx, N=nz_active, K=nz; LDA=nx, LDB=nz, LDC=ld_3d; TRANS_B
        KernelType kz_skip(LIBXSMM_GEMM_FLAG_TRANS_B, nx, nz_active, nz,
                           nx, nz, ld_3d, 1.0, 0.0);

        if (!kx_strided || !kz_skip) {
            // fallback: do the chain with a heap intermediate
            std::vector<double> chain_tmp((size_t)nx * ny * nz);
            this->do_grad_x(chain_tmp.data(), u, dx, sz, bflag);
            this->do_grad_z(w, chain_tmp.data(), dx, sz, bflag);
            return;
        }

        // raw_fn dispatch saves the C++ wrapper overhead across the loop.
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw_kx = kx_strided.kernel();
        libxsmm_gemmfunction raw_kz = kz_skip.kernel();
        if (raw_kx && raw_kz) {
            libxsmm_gemm_param a_args, b_args;
            a_args.a.primary = (void *)Dx_scaled.data();
            a_args.c.primary = (void *)tmp.data();
            b_args.a.primary = (void *)tmp.data();
            b_args.b.primary = (void *)(Dz_scaled.data() + z_start);
            for (unsigned int j = y_start; j < y_end; j++) {
                a_args.b.primary = (void *)(u + j * nx);
                raw_kx(&a_args);
                b_args.c.primary =
                    (void *)(w + j * nx + z_start * ld_3d);
                raw_kz(&b_args);
            }
            return;
        }
#endif
        for (unsigned int j = y_start; j < y_end; j++) {
            const double *u_at_j = u + j * nx;
            double *w_at_j_zstart = w + j * nx + z_start * ld_3d;
            // step 1: tmp(nx, nz) = D_x * u_at_j (strided read)
            kx_strided(Dx_scaled.data(), u_at_j, tmp.data());
            // step 2: w_at_j[:, z_active] = tmp * D_z^T  (strided write)
            // D_z B pointer offset by z_start so kernel reads rows
            // [z_start, z_end) of D_z.
            kz_skip(tmp.data(), Dz_scaled.data() + z_start, w_at_j_zstart);
        }
    }

    // Fused d^2u/dydz in a single call. Two-pass over a local tmp buffer:
    //   pass 1: per z-slice (all z), tmp_slab(nx, ny_active) =
    //           u_slice(nx, ny) * D_y_active^T  (y-derivative, output skip)
    //   pass 2: per active y, w_at_j[:, z_active] = tmp_at_j * D_z^T
    //           (strided write, z output skip)
    // tmp is (nx, ny_active, nz); at eleorder=6 that's 13*7*13 = 1183
    // doubles = 9.5 KB, comfortably in L1. Chain saves vs. chain by
    // avoiding grad_y's writes to y-padding cells AND by reusing the
    // tmp from L1 instead of reading it from RAM in grad_z.
    void do_grad_yz_last(double *const w, const double *const u,
                         const double dx, const unsigned int *sz,
                         const unsigned int bflag) {
        static_assert(DerivOrder == 1,
                      "do_grad_yz_last is only for 1st-order MatrixCompactDerivs");
        const unsigned int nx = sz[0];
        const unsigned int ny = sz[1];
        const unsigned int nz = sz[2];
        const unsigned int pw = this->p_pw;

        auto *sy  = this->get_storage_for_size(ny);
        auto *sz_s = this->get_storage_for_size(nz);
        auto *Dy  = get_deriv_mat_by_bflag_y(sy, bflag);
        auto *Dz  = get_deriv_mat_by_bflag_z(sz_s, bflag);

        const double alpha = 1.0 / dx;
        std::vector<double> Dy_scaled(ny * ny);
        std::vector<double> Dz_scaled(nz * nz);
        for (size_t i = 0; i < (size_t)ny * ny; i++)
            Dy_scaled[i] = Dy->data()[i] * alpha;
        for (size_t i = 0; i < (size_t)nz * nz; i++)
            Dz_scaled[i] = Dz->data()[i] * alpha;

        const unsigned int y_start   = pw;
        const unsigned int z_start   = pw;
        const unsigned int ny_active = ny - 2 * pw;
        const unsigned int nz_active = nz - 2 * pw;
        const unsigned int slab_sz   = nx * ny_active;     // per z-slice in tmp
        const unsigned int ld_3d     = nx * ny;

        std::vector<double> tmp((size_t)slab_sz * nz);

        // y kernel: tmp_slab(nx, ny_active) = u_slice(nx, ny) * D_y_active^T
        //   M=nx, N=ny_active, K=ny; LDA=nx, LDB=ny, LDC=nx; TRANS_B
        KernelType ky_skip(LIBXSMM_GEMM_FLAG_TRANS_B, nx, ny_active, ny,
                           nx, ny, nx, 1.0, 0.0);
        // z kernel: w_at_j[:, z_active] = tmp_at_j * D_z^T
        //   M=nx, N=nz_active, K=nz; LDA=slab_sz (stride in tmp's k dim),
        //   LDB=nz, LDC=ld_3d (stride in w's k dim); TRANS_B
        KernelType kz_skip(LIBXSMM_GEMM_FLAG_TRANS_B, nx, nz_active, nz,
                           slab_sz, nz, ld_3d, 1.0, 0.0);

        if (!ky_skip || !kz_skip) {
            std::vector<double> chain_tmp((size_t)nx * ny * nz);
            this->do_grad_y(chain_tmp.data(), u, dx, sz, bflag);
            this->do_grad_z(w, chain_tmp.data(), dx, sz, bflag);
            return;
        }

        // pass 1: full z, y-derivative with output skip → tmp(nx, ny_active, nz)
        // Per-slice loop (can't be batched into a single GEMM because the
        // input A = u_slice changes per slice; GEMM has one A). Use raw_fn
        // dispatch to minimize per-call overhead across the nz iterations.
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        {
            libxsmm_gemmfunction raw_ky = ky_skip.kernel();
            if (raw_ky) {
                libxsmm_gemm_param args;
                args.b.primary = (void *)(Dy_scaled.data() + y_start);
                for (unsigned int k = 0; k < nz; k++) {
                    args.a.primary = (void *)(u + k * ld_3d);
                    args.c.primary = (void *)(tmp.data() + k * slab_sz);
                    raw_ky(&args);
                }
            } else {
                for (unsigned int k = 0; k < nz; k++) {
                    ky_skip(u + k * ld_3d, Dy_scaled.data() + y_start,
                            tmp.data() + k * slab_sz);
                }
            }
        }
#else
        for (unsigned int k = 0; k < nz; k++) {
            ky_skip(u + k * ld_3d, Dy_scaled.data() + y_start,
                    tmp.data() + k * slab_sz);
        }
#endif

        // pass 2: per active y, z-derivative with output skip and strided
        //          write into w. Raw dispatch.
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw_kz = kz_skip.kernel();
        if (raw_kz) {
            libxsmm_gemm_param args;
            args.b.primary = (void *)(Dz_scaled.data() + z_start);
            for (unsigned int ja = 0; ja < ny_active; ja++) {
                const unsigned int j = ja + y_start;
                args.a.primary = (void *)(tmp.data() + ja * nx);
                args.c.primary = (void *)(w + j * nx + z_start * ld_3d);
                raw_kz(&args);
            }
            return;
        }
#endif
        for (unsigned int ja = 0; ja < ny_active; ja++) {
            const unsigned int j  = ja + y_start;
            const double *tmp_aj  = tmp.data() + ja * nx;
            double *w_aj_zstart   = w + j * nx + z_start * ld_3d;
            kz_skip(tmp_aj, Dz_scaled.data() + z_start, w_aj_zstart);
        }
    }

    // batch overrides: pre-scale D once and apply to all variables,
    // keeping the scaled matrix and kernel hot in cache
    void do_grad_x_batch(double **du_arr, const double **u_arr,
                         unsigned int n_vars, const double dx,
                         const unsigned int *sz,
                         const unsigned int bflag) override {
        auto *storage = get_storage_for_size(sz[0]);
        auto *D_use   = get_deriv_mat_by_bflag_x(storage, bflag);
        const double alpha = (DerivOrder == 1) ? 1.0 / dx : 1.0 / (dx * dx);

        for (unsigned int v = 0; v < n_vars; v++)
            matmul_x_dim(D_use->data(), du_arr[v], u_arr[v], alpha, sz, bflag,
                         p_pw);
    }

    void do_grad_y_batch(double **du_arr, const double **u_arr,
                         unsigned int n_vars, const double dx,
                         const unsigned int *sz,
                         const unsigned int bflag) override {
        auto *storage = get_storage_for_size(sz[1]);
        auto *D_use   = get_deriv_mat_by_bflag_y(storage, bflag);
        const double alpha = (DerivOrder == 1) ? 1.0 / dx : 1.0 / (dx * dx);
        this->ensure_workspace_for(sz);

        for (unsigned int v = 0; v < n_vars; v++)
            matmul_y_dim(D_use->data(), du_arr[v], u_arr[v], alpha, sz,
                         workspace_.data(), bflag, p_pw);
    }

    void do_grad_z_batch(double **du_arr, const double **u_arr,
                         unsigned int n_vars, const double dx,
                         const unsigned int *sz,
                         const unsigned int bflag) override {
        auto *storage = get_storage_for_size(sz[2]);
        auto *D_use   = get_deriv_mat_by_bflag_z(storage, bflag);
        const double alpha = (DerivOrder == 1) ? 1.0 / dx : 1.0 / (dx * dx);
        this->ensure_workspace_for(sz);

        for (unsigned int v = 0; v < n_vars; v++)
            matmul_z_dim(D_use->data(), du_arr[v], u_arr[v], alpha, sz,
                         workspace_.data(), bflag, p_pw);
    }

    void init();
};

// ============================================================
// generic wrappers that eliminate per-scheme class boilerplate.
// instead of defining a class for each scheme, the factory can
// construct these directly with a diagonal-creation function.
// ============================================================

// for schemes without user coefficients (most kim, A4, C6, etc.)
using DiagCreatorFn = MatrixDiagonalEntries* (*)();

template <unsigned int DerivOrder>
class GenericMatrixDerivs : public MatrixCompactDerivs<DerivOrder> {
    DiagCreatorFn diag_fn_;
    DerivType dtype_;
    std::string name_;

   public:
    GenericMatrixDerivs(DiagCreatorFn fn, DerivType dt, std::string name,
                        unsigned int ele_order,
                        const std::string& filter = "none",
                        const std::vector<double>& fcoeffs = {})
        : MatrixCompactDerivs<DerivOrder>{ele_order, filter, fcoeffs},
          diag_fn_(fn), dtype_(dt), name_(std::move(name)) {
        this->diagEntries = diag_fn_();
        this->init();
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<GenericMatrixDerivs>(*this);
    }
    DerivType getDerivType() const override { return dtype_; }
    enum DerivOrder getDerivOrder() const override {
        return (DerivOrder == 1) ? D_FIRST_ORDER : D_SECOND_ORDER;
    }
    std::string toString() const override { return name_; }
    void set_maximum_block_size(size_t block_size) override {
        MatrixCompactDerivs<DerivOrder>::set_maximum_block_size(block_size);
    }
};

// for schemes that accept user coefficients (BYU families)
using DiagCreatorWithCoeffsFn = MatrixDiagonalEntries* (*)(const std::vector<double>&);

template <unsigned int DerivOrder>
class GenericMatrixDerivsWithCoeffs : public MatrixCompactDerivs<DerivOrder> {
    DiagCreatorWithCoeffsFn diag_fn_;
    DerivType dtype_;
    std::string name_;
    std::vector<double> coeffs_;
    unsigned int n_coeffs_;

   public:
    GenericMatrixDerivsWithCoeffs(DiagCreatorWithCoeffsFn fn, DerivType dt,
                                  std::string name, unsigned int n_coeffs,
                                  unsigned int ele_order,
                                  const std::string& filter = "none",
                                  const std::vector<double>& fcoeffs = {},
                                  const std::vector<double>& coeffs_in = {})
        : MatrixCompactDerivs<DerivOrder>{ele_order, filter, fcoeffs},
          diag_fn_(fn), dtype_(dt), name_(std::move(name)),
          n_coeffs_(n_coeffs) {
        // pad/truncate coefficients to expected count
        coeffs_.resize(n_coeffs_, 0.0);
        for (unsigned int i = 0; i < n_coeffs_ && i < coeffs_in.size(); i++)
            coeffs_[i] = coeffs_in[i];
        this->diagEntries = diag_fn_(coeffs_);
        this->init();
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<GenericMatrixDerivsWithCoeffs>(*this);
    }
    DerivType getDerivType() const override { return dtype_; }
    enum DerivOrder getDerivOrder() const override {
        return (DerivOrder == 1) ? D_FIRST_ORDER : D_SECOND_ORDER;
    }
    std::string toString() const override { return name_; }
    void set_maximum_block_size(size_t block_size) override {
        MatrixCompactDerivs<DerivOrder>::set_maximum_block_size(block_size);
    }
};

}  // namespace dendroderivs
