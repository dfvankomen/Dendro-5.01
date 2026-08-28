#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "derivatives/derivs_utils.h"
#include "filters.h"

namespace dendroderivs {

/**
 * @brief Matrix-form Kreiss-Oliger dissipation.
 *
 * The per-axis KO stencil (centred interior, one-sided rows at bflag faces) is
 * assembled as a dense n x n operator by probing the wrapped explicit filter
 * with unit vectors, so every entry is exactly the stencil coefficient, and
 * applied as three active-region GEMMs accumulating D_x u + D_y u + D_z u,
 * then output += coeff * ws. 2.5-3.4x faster per call than the stencil loops
 * at n = 13; agrees with them to roundoff, not bit-for-bit, hence the separate
 * "KO4Matrix" names (stencil "KO4" stays the default). One instance per thread.
 */
class MatrixKODiss : public Filters {
   protected:
    std::unique_ptr<Filters> stencil_;  ///< source of the operator; fallback
    std::string name_;

    // per axis, per block size: the four bflag variants of the operator
    std::unordered_map<unsigned int, std::unique_ptr<DerivMatrixStorage>>
        D_[3];

    // hot-path memos (per instance)
    MatmulPlan plan_;
    ScaledOperator sop_[3];
    std::vector<double> ws_;

    static unsigned int axis_bflag(int axis, BoundaryType b) {
        static const unsigned int lo[3] = {1u << OCT_DIR_LEFT, 1u << OCT_DIR_DOWN,
                                           1u << OCT_DIR_BACK};
        static const unsigned int hi[3] = {1u << OCT_DIR_RIGHT, 1u << OCT_DIR_UP,
                                           1u << OCT_DIR_FRONT};
        unsigned int bf = 0;
        if (b == LEFT_BOUNDARY || b == LEFTRIGHT_BOUNDARY) bf |= lo[axis];
        if (b == RIGHT_BOUNDARY || b == LEFTRIGHT_BOUNDARY) bf |= hi[axis];
        return bf;
    }

    // probe the explicit filter with unit vectors constant along the other two
    // axes (their stencils vanish exactly): output column = operator column
    std::unique_ptr<DerivMatrixStorage> build_storage(int axis,
                                                      unsigned int n) {
        auto st        = std::make_unique<DerivMatrixStorage>();
        st->dim_size   = n;
        const size_t nn = (size_t)n * n;
        st->D_original.assign(nn, 0.0);
        st->D_left.assign(nn, 0.0);
        st->D_right.assign(nn, 0.0);
        st->D_leftright.assign(nn, 0.0);

        const unsigned int sz[3] = {n, n, n};
        const size_t tot         = (size_t)n * n * n;
        std::vector<double> e(tot), out(tot), wx(tot), wy(tot), wz(tot);
        auto idx = [&](unsigned int a, unsigned int b, unsigned int c) {
            // point with `a` on `axis` and (b, c) on the other two axes
            unsigned int ijk[3];
            ijk[axis]           = a;
            ijk[(axis + 1) % 3] = b;
            ijk[(axis + 2) % 3] = c;
            return ijk[0] + n * (ijk[1] + n * ijk[2]);
        };

        for (BoundaryType b : {NO_BOUNDARY, LEFT_BOUNDARY, RIGHT_BOUNDARY,
                               LEFTRIGHT_BOUNDARY}) {
            std::vector<double> *D = get_deriv_mat_by_boundary(st.get(), b);
            const unsigned int bf  = axis_bflag(axis, b);
            for (unsigned int c = 0; c < n; c++) {
                std::fill(e.begin(), e.end(), 0.0);
                for (unsigned int q = 0; q < n; q++)
                    for (unsigned int r = 0; r < n; r++) e[idx(c, q, r)] = 1.0;
                std::fill(out.begin(), out.end(), 0.0);
                stencil_->do_full_filter(e.data(), out.data(), wx.data(),
                                         wy.data(), wz.data(), 1.0, 1.0, 1.0,
                                         1.0, sz, bf);
                for (unsigned int r = p_pw; r < n - p_pw; r++)
                    (*D)[r + n * c] = out[idx(r, p_pw, p_pw)];
            }
        }
        return st;
    }

    DerivMatrixStorage *storage(int axis, unsigned int n) {
        auto it = D_[axis].find(n);
        if (it != D_[axis].end()) return it->second.get();
        auto *raw = D_[axis].emplace(n, build_storage(axis, n)).first->second.get();
        return raw;
    }

    const MatmulPlan &plan_for(const unsigned int *sz) {
        if (!plan_.matches(sz, p_pw)) plan_ = build_matmul_plan(sz, p_pw);
        return plan_;
    }

    // ws = D_x u + D_y u + D_z u on the active region; false if a kernel is missing
    bool sum_axes(double *ws, const double *u, double dx, double dy, double dz,
                  const unsigned int *sz, unsigned int bflag) {
        const MatmulPlan &p = plan_for(sz);
        if (!p.kx_last || !p.ky_last_acc || !p.kz_acc) return false;
        const double *Dx = sop_[0].get(
            get_deriv_mat_by_bflag_x(storage(0, sz[0]), bflag)->data(), 1.0 / dx,
            sz[0]);
        const double *Dy = sop_[1].get(
            get_deriv_mat_by_bflag_y(storage(1, sz[1]), bflag)->data(), 1.0 / dy,
            sz[1]);
        const double *Dz = sop_[2].get(
            get_deriv_mat_by_bflag_z(storage(2, sz[2]), bflag)->data(), 1.0 / dz,
            sz[2]);
        matmul_x_apply(p.kx_last, Dx, ws, u, sz, p_pw, true);      // ws  = Dx u
        matmul_y_apply(p.ky_last_acc, Dy, ws, u, sz, p_pw, true);  // ws += Dy u
        matmul_z_apply(p.kz_acc, Dz, ws, u, sz, p_pw);             // ws += Dz u
        return true;
    }

    double *workspace_for(const unsigned int *sz) {
        const size_t need = (size_t)sz[0] * sz[1] * sz[2];
        if (ws_.size() < need) ws_.resize(need);
        return ws_.data();
    }

   public:
    MatrixKODiss(unsigned int ele_order, std::unique_ptr<Filters> stencil,
                 std::string name)
        : Filters(ele_order), stencil_(std::move(stencil)), name_(std::move(name)) {
        if (!stencil_)
            throw std::invalid_argument("MatrixKODiss: null stencil filter");
        // the production block size; other sizes are built on first use
        (void)storage(0, p_n);
        (void)storage(1, p_n);
        (void)storage(2, p_n);
    }

    MatrixKODiss(const MatrixKODiss &o)
        : Filters(o), stencil_(o.stencil_->clone()), name_(o.name_) {
        for (int a = 0; a < 3; a++)
            for (const auto &kv : o.D_[a])
                D_[a][kv.first] = std::make_unique<DerivMatrixStorage>(*kv.second);
        // plan_ / sop_ start empty: their keys point into THIS instance
    }

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<MatrixKODiss>(*this);
    }

    void do_full_filter(const double *const input, double *const output,
                        double *const workspace_x, double *const workspace_y,
                        double *const workspace_z, const double dx,
                        const double dy, const double dz, const double coeff,
                        const unsigned int *sz,
                        const unsigned int bflag) override {
        double *ws = workspace_for(sz);
        if (!sum_axes(ws, input, dx, dy, dz, sz, bflag)) {
            stencil_->do_full_filter(input, output, workspace_x, workspace_y,
                                     workspace_z, dx, dy, dz, coeff, sz, bflag);
            return;
        }
        const unsigned int nx = sz[0], ny = sz[1], nz = sz[2], pw = p_pw;
        for (unsigned int k = pw; k < nz - pw; k++)
            for (unsigned int j = pw; j < ny - pw; j++) {
                const unsigned int row = nx * (j + ny * k);
                for (unsigned int i = pw; i < nx - pw; i++)
                    output[row + i] += coeff * ws[row + i];
            }
    }

    void do_full_filter_field(const double *const input, double *const output,
                              double *const workspace_x,
                              double *const workspace_y,
                              double *const workspace_z, const double dx,
                              const double dy, const double dz,
                              const double *const coeff_field,
                              const unsigned int *sz,
                              const unsigned int bflag) override {
        double *ws = workspace_for(sz);
        if (!sum_axes(ws, input, dx, dy, dz, sz, bflag)) {
            stencil_->do_full_filter_field(input, output, workspace_x,
                                           workspace_y, workspace_z, dx, dy, dz,
                                           coeff_field, sz, bflag);
            return;
        }
        const unsigned int nx = sz[0], ny = sz[1], nz = sz[2], pw = p_pw;
        for (unsigned int k = pw; k < nz - pw; k++)
            for (unsigned int j = pw; j < ny - pw; j++) {
                const unsigned int row = nx * (j + ny * k);
                for (unsigned int i = pw; i < nx - pw; i++)
                    output[row + i] += coeff_field[row + i] * ws[row + i];
            }
    }

    std::string toString() const override { return name_; }
    bool do_filter_before() const override { return stencil_->do_filter_before(); }
    void set_maximum_block_size(size_t block_size) override {
        if (ws_.size() < block_size) ws_.resize(block_size);
        stencil_->set_maximum_block_size(block_size);
    }
    FilterFamily get_filter_family() const override {
        return stencil_->get_filter_family();
    }
};

}  // namespace dendroderivs
