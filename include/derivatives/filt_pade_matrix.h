#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "derivatives/derivs_matrixonly.h"
#include "derivatives/derivs_utils.h"
#include "filters.h"

namespace dendroderivs {

/**
 * @brief Compact (Pade) filter applied as a dissipation term (registry name
 * KIMF): rhs += (coeff/h) * sum_axis (F - I) u, with the Kim filter's 7-point
 * RHS so pw = 3 suffices where explicit KO caps at KO4.
 *
 * The Kim diagonals store R (implicit LHS) and S (high-pass, zero row sums),
 * so the single-size builder yields R^-1 S = F - I directly, closures and
 * bflag variants included. Same Nyquist damping scaling as KO; 8th-order
 * rolloff below it. Applied through the libxsmm plan like MatrixKODiss.
 * Gate: testKOSign. One instance per thread.
 */
class MatrixPadeFilter : public Filters {
   protected:
    std::string name_;
    MatrixDiagonalEntries entries_;  // R/S diagonals; strength stays per call

    // per block size: the four bflag variants of D = R^-1 S
    std::unordered_map<unsigned int, std::unique_ptr<DerivMatrixStorage>> D_;

    // per-instance memos: keys point into THIS instance (clone resets them)
    MatmulPlan plan_;
    ScaledOperator sop_[3];
    std::vector<double> ws_;

    DerivMatrixStorage *storage(unsigned int n) {
        auto it = D_.find(n);
        if (it != D_.end()) return it->second.get();
        auto st = createMatrixSystemForSingleSize<2>(p_pw, n, &entries_);
        return D_.emplace(n, std::move(st)).first->second.get();
    }

    const MatmulPlan &plan_for(const unsigned int *sz) {
        if (!plan_.matches(sz, p_pw)) plan_ = build_matmul_plan(sz, p_pw);
        return plan_;
    }

    bool sum_axes(double *ws, const double *u, double dx, double dy, double dz,
                  const unsigned int *sz, unsigned int bflag) {
        const MatmulPlan &p = plan_for(sz);
        if (!p.kx_last || !p.ky_last_acc || !p.kz_acc) return false;
        const double *Dx = sop_[0].get(
            get_deriv_mat_by_bflag_x(storage(sz[0]), bflag)->data(), 1.0 / dx, sz[0]);
        const double *Dy = sop_[1].get(
            get_deriv_mat_by_bflag_y(storage(sz[1]), bflag)->data(), 1.0 / dy, sz[1]);
        const double *Dz = sop_[2].get(
            get_deriv_mat_by_bflag_z(storage(sz[2]), bflag)->data(), 1.0 / dz, sz[2]);
        matmul_x_apply(p.kx_last, Dx, ws, u, sz, p_pw, true);
        matmul_y_apply(p.ky_last_acc, Dy, ws, u, sz, p_pw, true);
        matmul_z_apply(p.kz_acc, Dz, ws, u, sz, p_pw);
        return true;
    }

    double *workspace_for(const unsigned int *sz) {
        const size_t need = (size_t)sz[0] * sz[1] * sz[2];
        if (ws_.size() < need) ws_.resize(need);
        return ws_.data();
    }

   public:
    MatrixPadeFilter(unsigned int ele_order, const std::string &family,
                     const std::vector<double> &coeffs, std::string name)
        : Filters(ele_order),
          name_(std::move(name)),
          entries_(*createInMatrixFilterByType(family, coeffs)->get_diag_entries()) {
        (void)storage(p_n);  // production size now; other sizes on first use
    }

    MatrixPadeFilter(const MatrixPadeFilter &o)
        : Filters(o), name_(o.name_), entries_(o.entries_) {
        for (const auto &kv : o.D_)
            D_[kv.first] = std::make_unique<DerivMatrixStorage>(*kv.second);
    }

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<MatrixPadeFilter>(*this);
    }

    void do_full_filter(const double *const input, double *const output,
                        double *const, double *const, double *const,
                        const double dx, const double dy, const double dz,
                        const double coeff, const unsigned int *sz,
                        const unsigned int bflag) override {
        double *ws = workspace_for(sz);
        if (!sum_axes(ws, input, dx, dy, dz, sz, bflag))
            throw std::runtime_error(name_ + ": no GEMM kernels for this size");
        const unsigned int nx = sz[0], ny = sz[1], nz = sz[2], pw = p_pw;
        for (unsigned int k = pw; k < nz - pw; k++)
            for (unsigned int j = pw; j < ny - pw; j++) {
                const unsigned int row = nx * (j + ny * k);
                for (unsigned int i = pw; i < nx - pw; i++)
                    output[row + i] += coeff * ws[row + i];
            }
    }

    void do_full_filter_field(const double *const input, double *const output,
                              double *const, double *const, double *const,
                              const double dx, const double dy, const double dz,
                              const double *const coeff_field,
                              const unsigned int *sz,
                              const unsigned int bflag) override {
        double *ws = workspace_for(sz);
        if (!sum_axes(ws, input, dx, dy, dz, sz, bflag))
            throw std::runtime_error(name_ + ": no GEMM kernels for this size");
        const unsigned int nx = sz[0], ny = sz[1], nz = sz[2], pw = p_pw;
        for (unsigned int k = pw; k < nz - pw; k++)
            for (unsigned int j = pw; j < ny - pw; j++) {
                const unsigned int row = nx * (j + ny * k);
                for (unsigned int i = pw; i < nx - pw; i++)
                    output[row + i] += coeff_field[row + i] * ws[row + i];
            }
    }

    // three beta = 1 GEMMs into rhs; coeff folded into the scaled operators
    bool do_accumulate(double *const rhs, const double *const u,
                       const double coeff, const double dx, const double dy,
                       const double dz, const unsigned int *sz,
                       const unsigned int bflag) override {
        const MatmulPlan &p = plan_for(sz);
        if (!p.kx_last_acc || !p.ky_last_acc || !p.kz_acc) return false;
        const double *Dx = sop_[0].get(
            get_deriv_mat_by_bflag_x(storage(sz[0]), bflag)->data(), coeff / dx, sz[0]);
        const double *Dy = sop_[1].get(
            get_deriv_mat_by_bflag_y(storage(sz[1]), bflag)->data(), coeff / dy, sz[1]);
        const double *Dz = sop_[2].get(
            get_deriv_mat_by_bflag_z(storage(sz[2]), bflag)->data(), coeff / dz, sz[2]);
        bool ok = matmul_x_apply(p.kx_last_acc, Dx, rhs, u, sz, p_pw, true);
        ok = matmul_y_apply(p.ky_last_acc, Dy, rhs, u, sz, p_pw, true) && ok;
        ok = matmul_z_apply(p.kz_acc, Dz, rhs, u, sz, p_pw) && ok;
        return ok;
    }

    std::string toString() const override { return name_; }
    bool do_filter_before() const override { return false; }
    void set_maximum_block_size(size_t) override {}
    FilterFamily get_filter_family() const override {
        return dendroderivs::FilterFamily::FF_KO;
    }
};

}  // namespace dendroderivs
