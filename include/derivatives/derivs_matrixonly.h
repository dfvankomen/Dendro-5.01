
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
std::unique_ptr<DerivMatrixStorage>
createMatrixSystemForSingleSizeInMatrixFilter(
    const unsigned int pw, const unsigned int n,
    const MatrixDiagonalEntries *diagEntries,
    const MatrixDiagonalEntries *filterEntries, const bool skip_leftright,
    const InMatFilterType filt_type);

template <unsigned int DerivOrder>
class MatrixCompactDerivs : public CompactDerivs {
   protected:
    std::vector<double> workspace_;
    unsigned int workspace_tot_;

    // reused scratch for the fused mixed-2nd kernels' L1 intermediate. grown
    // once and refilled per call, so the fused path does no per-call heap
    // allocation. not thread-safe to share (same as workspace_); the
    // block-parallel model clones per thread.
    std::vector<double> fused_tmp_;

    std::unordered_map<unsigned int, std::unique_ptr<DerivMatrixStorage>>
        D_storage_map_;

    // memoize the last-used storage pointer — avoids hash lookup when
    // block size doesn't change between calls (the common case)
    mutable unsigned int _cached_sz = 0;
    mutable DerivMatrixStorage *_cached_storage = nullptr;

    // hot-path memo. an instance is per-thread under the clone model, so
    // these need no lock: the kernel plan for the last block shape, and the
    // alpha-scaled operator for the last (D, alpha) seen on each axis. blocks
    // arrive in Morton order, so consecutive blocks nearly always share both,
    // and the shared kernel cache (shared_mutex + hash) and the per-call
    // re-scale stay off the timestep loop. reset on copy: the src pointers
    // key into THIS instance's D storage
    MatmulPlan plan_;
    ScaledOperator sop_[3];

    const MatmulPlan &plan_for(const unsigned int *sz) {
        if (!plan_.matches(sz, p_pw)) plan_ = build_matmul_plan(sz, p_pw);
        return plan_;
    }

    const double *scaled_op(int axis, const std::vector<double> *D,
                            double alpha, unsigned int n) {
        return sop_[axis].get(D->data(), alpha, n);
    }

    static constexpr double spacing_alpha(double dx) {
        return (DerivOrder == 1) ? 1.0 / dx : 1.0 / (dx * dx);
    }

    // the three axis applies behind do_grad_* / _last / _batch. fall back to
    // the BLAS path with the already-scaled operator if a kernel failed to JIT
    void apply_x(double *const du, const double *const u, const double dx,
                 const unsigned int *sz, const unsigned int bflag, bool last) {
        auto *storage       = get_storage_for_size(sz[0]);
        auto *D_use         = get_deriv_mat_by_bflag_x(storage, bflag);
        const double *Ds    = scaled_op(0, D_use, spacing_alpha(dx), sz[0]);
        const MatmulPlan &p = plan_for(sz);
        if (!matmul_x_apply(last ? p.kx_last : p.kx_int, Ds, du, u, sz, p_pw,
                            last))
            matmul_x_dim_old(Ds, du, u, 1.0, sz, bflag, p_pw);
    }
    void apply_y(double *const du, const double *const u, const double dx,
                 const unsigned int *sz, const unsigned int bflag, bool last) {
        auto *storage       = get_storage_for_size(sz[1]);
        auto *D_use         = get_deriv_mat_by_bflag_y(storage, bflag);
        const double *Ds    = scaled_op(1, D_use, spacing_alpha(dx), sz[1]);
        const MatmulPlan &p = plan_for(sz);
        if (!matmul_y_apply(last ? p.ky_last : p.ky_int, Ds, du, u, sz, p_pw,
                            last)) {
            ensure_workspace_for(sz);
            matmul_y_dim_old(Ds, du, u, 1.0, sz, workspace_.data(), bflag,
                             p_pw);
        }
    }
    void apply_z(double *const du, const double *const u, const double dx,
                 const unsigned int *sz, const unsigned int bflag) {
        auto *storage       = get_storage_for_size(sz[2]);
        auto *D_use         = get_deriv_mat_by_bflag_z(storage, bflag);
        const double *Ds    = scaled_op(2, D_use, spacing_alpha(dx), sz[2]);
        const MatmulPlan &p = plan_for(sz);
        if (!matmul_z_apply(p.kz, Ds, du, u, sz, p_pw)) {
            ensure_workspace_for(sz);
            matmul_z_dim_old(Ds, du, u, 1.0, sz, workspace_.data(), bflag,
                             p_pw);
        }
    }

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
    /**
     * @brief Build the four bflag variants of D for one block size.
     *
     * The single source of truth for "how does *this* scheme build D". Both
     * init() (eager, for the common sizes) and get_storage_for_size() (lazy, on
     * a cache miss) route through here, so a scheme that builds D some other way
     * overrides this one method and is correct on both paths.
     *
     * Only called on a cache miss, so the virtual dispatch never lands on the
     * hot path — and it's dwarfed by the LAPACK solve it wraps anyway.
     */
    virtual std::unique_ptr<DerivMatrixStorage> build_storage_for_size(
        unsigned int n, bool skip_leftright) {
        if (in_matrix_filter_->get_filter_type() == InMatFilterType::IMFT_NONE) {
            return createMatrixSystemForSingleSize<DerivOrder>(
                p_pw, n, diagEntries, skip_leftright);
        }
        return createMatrixSystemForSingleSizeInMatrixFilter<DerivOrder>(
            p_pw, n, diagEntries, in_matrix_filter_->get_diag_entries(),
            skip_leftright, in_matrix_filter_->get_filter_type());
    }

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

        // first time seeing this size — create matrices. goes through the
        // virtual so a scheme with its own build (in-matrix filter, boris'
        // per-side diagonals, CCFD's coupled system) stays correct here and not
        // just on the eager init() path.
        D_storage_map_.emplace(n, build_storage_for_size(n, false));

        _cached_sz      = n;
        _cached_storage = D_storage_map_[n].get();
        return _cached_storage;
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        apply_x(du, u, dx, sz, bflag, /*last=*/false);
    }

    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        apply_y(du, u, dx, sz, bflag, /*last=*/false);
    }

    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) {
        apply_z(du, u, dx, sz, bflag);
    }

    // "_last" overrides: the output is terminal, so the kernel skips the y/z
    // output padding too. See the contract in derivatives.h::do_grad_x_last.
    void do_grad_x_last(double *const du, const double *const u,
                        const double dx, const unsigned int *sz,
                        const unsigned int bflag) override {
        apply_x(du, u, dx, sz, bflag, /*last=*/true);
    }

    void do_grad_y_last(double *const du, const double *const u,
                        const double dx, const unsigned int *sz,
                        const unsigned int bflag) override {
        apply_y(du, u, dx, sz, bflag, /*last=*/true);
    }

    // Fused mixed second derivatives d^2u/dadb (1st-order engines only). Each
    // streams one slice/slab through both operators: step 1 differentiates
    // along the first axis into a small intermediate that stays in L1, step 2
    // applies the second axis from it straight into the active output. Both
    // steps use active shapes (M = active x rows, output columns active only)
    // and the memoized scaled operators / kernel plan, so a call does no
    // lookups, no re-scaling and no allocation. Output is defined on the
    // active region only, per the contract in derivs_utils.h. Bit-identical
    // to the chained form (same dot products, same order).
    // isotropic forwarders (kept for existing callers)
    void do_grad_xy_last(double *const w, const double *const u,
                         const double dx, const unsigned int *sz,
                         const unsigned int bflag) {
        do_grad_xy_last(w, u, dx, dx, sz, bflag);
    }
    void do_grad_xz_last(double *const w, const double *const u,
                         const double dx, const unsigned int *sz,
                         const unsigned int bflag) {
        do_grad_xz_last(w, u, dx, dx, sz, bflag);
    }
    void do_grad_yz_last(double *const w, const double *const u,
                         const double dy, const unsigned int *sz,
                         const unsigned int bflag) {
        do_grad_yz_last(w, u, dy, dy, sz, bflag);
    }

    void do_grad_xy_last(double *const w, const double *const u,
                         const double dx, const double dy,
                         const unsigned int *sz, const unsigned int bflag) {
        static_assert(DerivOrder == 1,
                      "do_grad_xy_last is only for 1st-order MatrixCompactDerivs");
        const unsigned int nx = sz[0], ny = sz[1], nz = sz[2], pw = this->p_pw;
        auto *Dx = get_deriv_mat_by_bflag_x(this->get_storage_for_size(nx), bflag);
        auto *Dy = get_deriv_mat_by_bflag_y(this->get_storage_for_size(ny), bflag);
        const double *Dxs   = scaled_op(0, Dx, 1.0 / dx, nx);
        const double *Dys   = scaled_op(1, Dy, 1.0 / dy, ny);
        const MatmulPlan &p = plan_for(sz);
        if (!p.kxy1 || !p.kxy2) {
            // chain through a block-sized intermediate
            fused_tmp_.resize((size_t)nx * ny * nz);
            this->do_grad_x(fused_tmp_.data(), u, dx, sz, bflag);
            this->do_grad_y_last(w, fused_tmp_.data(), dy, sz, bflag);
            return;
        }
        fused_tmp_.resize((size_t)p.ma * ny);
        double *tmp                 = fused_tmp_.data();
        const unsigned int slice_sz = nx * ny;
        const unsigned int c_off    = pw * nx + pw;
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw1 = p.kxy1.kernel(), raw2 = p.kxy2.kernel();
        if (raw1 && raw2) {
            libxsmm_gemm_param a1, a2;
            a1.a.primary = (void *)(Dxs + pw);
            a1.c.primary = (void *)tmp;
            a2.a.primary = (void *)tmp;
            a2.b.primary = (void *)(Dys + pw);
            for (unsigned int k = pw; k < nz - pw; k++) {
                a1.b.primary = (void *)(u + k * slice_sz);
                raw1(&a1);
                a2.c.primary = (void *)(w + k * slice_sz + c_off);
                raw2(&a2);
            }
            return;
        }
#endif
        for (unsigned int k = pw; k < nz - pw; k++) {
            p.kxy1(Dxs + pw, u + k * slice_sz, tmp);
            p.kxy2(tmp, Dys + pw, w + k * slice_sz + c_off);
        }
    }

    void do_grad_xz_last(double *const w, const double *const u,
                         const double dx, const double dz,
                         const unsigned int *sz, const unsigned int bflag) {
        static_assert(DerivOrder == 1,
                      "do_grad_xz_last is only for 1st-order MatrixCompactDerivs");
        const unsigned int nx = sz[0], ny = sz[1], nz = sz[2], pw = this->p_pw;
        auto *Dx = get_deriv_mat_by_bflag_x(this->get_storage_for_size(nx), bflag);
        auto *Dz = get_deriv_mat_by_bflag_z(this->get_storage_for_size(nz), bflag);
        const double *Dxs   = scaled_op(0, Dx, 1.0 / dx, nx);
        const double *Dzs   = scaled_op(2, Dz, 1.0 / dz, nz);
        const MatmulPlan &p = plan_for(sz);
        if (!p.kxz1 || !p.kxz2) {
            fused_tmp_.resize((size_t)nx * ny * nz);
            this->do_grad_x(fused_tmp_.data(), u, dx, sz, bflag);
            this->do_grad_z(w, fused_tmp_.data(), dz, sz, bflag);
            return;
        }
        fused_tmp_.resize((size_t)p.ma * nz);
        double *tmp              = fused_tmp_.data();
        const unsigned int ld_3d = nx * ny;
        const unsigned int c_off = pw + pw * ld_3d;
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw1 = p.kxz1.kernel(), raw2 = p.kxz2.kernel();
        if (raw1 && raw2) {
            libxsmm_gemm_param a1, a2;
            a1.a.primary = (void *)(Dxs + pw);
            a1.c.primary = (void *)tmp;
            a2.a.primary = (void *)tmp;
            a2.b.primary = (void *)(Dzs + pw);
            for (unsigned int j = pw; j < ny - pw; j++) {
                a1.b.primary = (void *)(u + j * nx);
                raw1(&a1);
                a2.c.primary = (void *)(w + j * nx + c_off);
                raw2(&a2);
            }
            return;
        }
#endif
        for (unsigned int j = pw; j < ny - pw; j++) {
            p.kxz1(Dxs + pw, u + j * nx, tmp);
            p.kxz2(tmp, Dzs + pw, w + j * nx + c_off);
        }
    }

    void do_grad_yz_last(double *const w, const double *const u,
                         const double dy, const double dz,
                         const unsigned int *sz, const unsigned int bflag) {
        static_assert(DerivOrder == 1,
                      "do_grad_yz_last is only for 1st-order MatrixCompactDerivs");
        const unsigned int nx = sz[0], ny = sz[1], nz = sz[2], pw = this->p_pw;
        auto *Dy = get_deriv_mat_by_bflag_y(this->get_storage_for_size(ny), bflag);
        auto *Dz = get_deriv_mat_by_bflag_z(this->get_storage_for_size(nz), bflag);
        const double *Dys   = scaled_op(1, Dy, 1.0 / dy, ny);
        const double *Dzs   = scaled_op(2, Dz, 1.0 / dz, nz);
        const MatmulPlan &p = plan_for(sz);
        if (!p.kyz1 || !p.kyz2) {
            fused_tmp_.resize((size_t)nx * ny * nz);
            this->do_grad_y(fused_tmp_.data(), u, dy, sz, bflag);
            this->do_grad_z(w, fused_tmp_.data(), dz, sz, bflag);
            return;
        }
        // tmp(ma, ny_active, nz): the y-derivative at active x rows and active
        // y columns for every z slice; pass 2 reads it at z-stride slab_sz
        const unsigned int ny_active = ny - 2 * pw;
        const unsigned int slab_sz   = p.ma * ny_active;
        fused_tmp_.resize((size_t)slab_sz * nz);
        double *tmp              = fused_tmp_.data();
        const unsigned int ld_3d = nx * ny;
        const unsigned int c_off = pw + pw * ld_3d;
#if DENDRO_DERIVS_USE_RAW_XSMM_DISPATCH
        libxsmm_gemmfunction raw1 = p.kyz1.kernel(), raw2 = p.kyz2.kernel();
        if (raw1 && raw2) {
            libxsmm_gemm_param a1, a2;
            a1.b.primary = (void *)(Dys + pw);
            for (unsigned int k = 0; k < nz; k++) {
                a1.a.primary = (void *)(u + k * ld_3d + pw);
                a1.c.primary = (void *)(tmp + k * slab_sz);
                raw1(&a1);
            }
            a2.b.primary = (void *)(Dzs + pw);
            for (unsigned int ja = 0; ja < ny_active; ja++) {
                a2.a.primary = (void *)(tmp + ja * p.ma);
                a2.c.primary = (void *)(w + (ja + pw) * nx + c_off);
                raw2(&a2);
            }
            return;
        }
#endif
        for (unsigned int k = 0; k < nz; k++)
            p.kyz1(u + k * ld_3d + pw, Dys + pw, tmp + k * slab_sz);
        for (unsigned int ja = 0; ja < ny_active; ja++)
            p.kyz2(tmp + ja * p.ma, Dzs + pw, w + (ja + pw) * nx + c_off);
    }

    // expose the fused mixed-2nd kernels to the facade (1st-order only;
    // 2nd-order operators don't compose this way). isotropic only — the
    // fused ops scale both Ds by 1/h, matching the facade's dx==dy guard.
    bool try_fused_grad_xy_last(double *const w, const double *const u,
                                const double dx, const double dy,
                                const unsigned int *sz,
                                const unsigned int bflag) override {
        if constexpr (DerivOrder == 1) {
            do_grad_xy_last(w, u, dx, dy, sz, bflag);
            return true;
        }
        return false;
    }
    bool try_fused_grad_xz_last(double *const w, const double *const u,
                                const double dx, const double dz,
                                const unsigned int *sz,
                                const unsigned int bflag) override {
        if constexpr (DerivOrder == 1) {
            do_grad_xz_last(w, u, dx, dz, sz, bflag);
            return true;
        }
        return false;
    }
    bool try_fused_grad_yz_last(double *const w, const double *const u,
                                const double dy, const double dz,
                                const unsigned int *sz,
                                const unsigned int bflag) override {
        if constexpr (DerivOrder == 1) {
            do_grad_yz_last(w, u, dy, dz, sz, bflag);
            return true;
        }
        return false;
    }

    // batch overrides: the scaled operator and the kernel plan are memoized
    // on the instance, so every variable after the first is just the apply
    void do_grad_x_batch(double **du_arr, const double **u_arr,
                         unsigned int n_vars, const double dx,
                         const unsigned int *sz,
                         const unsigned int bflag) override {
        for (unsigned int v = 0; v < n_vars; v++)
            apply_x(du_arr[v], u_arr[v], dx, sz, bflag, /*last=*/false);
    }

    void do_grad_y_batch(double **du_arr, const double **u_arr,
                         unsigned int n_vars, const double dx,
                         const unsigned int *sz,
                         const unsigned int bflag) override {
        for (unsigned int v = 0; v < n_vars; v++)
            apply_y(du_arr[v], u_arr[v], dx, sz, bflag, /*last=*/false);
    }

    void do_grad_z_batch(double **du_arr, const double **u_arr,
                         unsigned int n_vars, const double dx,
                         const unsigned int *sz,
                         const unsigned int bflag) override {
        for (unsigned int v = 0; v < n_vars; v++)
            apply_z(du_arr[v], u_arr[v], dx, sz, bflag);
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
