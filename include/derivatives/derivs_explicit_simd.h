#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "derivatives.h"
#include "derivatives/derivs_explicit.h"

namespace dendroderivs {

/**
 * @brief Explicit central-difference engine with vectorized kernels
 * (registry names E4Simd / E6Simd / E8Simd).
 *
 * Interior blocks (bflag == 0): each axis is one `omp simd` loop per z-slice
 * over the contiguous (x,y)-plane span between the first and last active
 * point; padding cells inside the span are computed and never read. Output is
 * defined on the active region plus the full extent along axes a downstream
 * operator still differentiates (same contract as the matrix engines, see
 * derivs_utils.h). Boundary blocks delegate to the loop engine so physical
 * one-sided rows are the identical code. Agrees with the loop engine to
 * roundoff (factored coefficients, not bit-for-bit).
 */
template <unsigned int Order, unsigned int DerivOrder>
class ExplicitSimdDerivs : public Derivs {
    static_assert(Order == 4 || Order == 6 || Order == 8, "E4/E6/E8 only");
    static_assert(DerivOrder == 1 || DerivOrder == 2, "1st or 2nd derivative");
    static constexpr unsigned int H = Order / 2;  // stencil half-width

    std::unique_ptr<Derivs> fallback_;  // loop engine: bflag != 0

    // interior stencil weights, unscaled: c[0] centre (2nd only), c[m] for +-m
    static void weights(double *c) {
        if constexpr (DerivOrder == 1) {
            if constexpr (Order == 4) { c[1] = 2.0 / 3.0; c[2] = -1.0 / 12.0; }
            if constexpr (Order == 6) { c[1] = 3.0 / 4.0; c[2] = -3.0 / 20.0; c[3] = 1.0 / 60.0; }
            if constexpr (Order == 8) { c[1] = 4.0 / 5.0; c[2] = -1.0 / 5.0; c[3] = 4.0 / 105.0; c[4] = -1.0 / 280.0; }
        } else {
            if constexpr (Order == 4) { c[0] = -5.0 / 2.0; c[1] = 4.0 / 3.0; c[2] = -1.0 / 12.0; }
            if constexpr (Order == 6) { c[0] = -49.0 / 18.0; c[1] = 3.0 / 2.0; c[2] = -3.0 / 20.0; c[3] = 1.0 / 90.0; }
            if constexpr (Order == 8) { c[0] = -205.0 / 72.0; c[1] = 8.0 / 5.0; c[2] = -1.0 / 5.0; c[3] = 8.0 / 315.0; c[4] = -1.0 / 560.0; }
        }
    }

    // one simd loop of `len` points starting at `off` in every slice k in
    // [k0, k1), stencil stride s along the derivative axis
    static void apply(double *__restrict__ o, const double *__restrict__ u,
                      const double *c, double scale, long s, long off, long len,
                      unsigned int k0, unsigned int k1, size_t slice) {
        const double c0 = c[0] * scale, c1 = c[1] * scale, c2 = c[2] * scale;
        const double c3 = (H >= 3) ? c[3] * scale : 0.0, c4 = (H >= 4) ? c[4] * scale : 0.0;
        for (unsigned int k = k0; k < k1; k++) {
            const double *__restrict__ ur = u + slice * k + off;
            double *__restrict__ orow     = o + slice * k + off;
            if constexpr (DerivOrder == 1) {
#pragma omp simd
                for (long i = 0; i < len; i++) {
                    double v = c1 * (ur[i + s] - ur[i - s]) + c2 * (ur[i + 2 * s] - ur[i - 2 * s]);
                    if constexpr (H >= 3) v += c3 * (ur[i + 3 * s] - ur[i - 3 * s]);
                    if constexpr (H >= 4) v += c4 * (ur[i + 4 * s] - ur[i - 4 * s]);
                    orow[i] = v;
                }
            } else {
#pragma omp simd
                for (long i = 0; i < len; i++) {
                    double v = c0 * ur[i] + c1 * (ur[i + s] + ur[i - s]) + c2 * (ur[i + 2 * s] + ur[i - 2 * s]);
                    if constexpr (H >= 3) v += c3 * (ur[i + 3 * s] + ur[i - 3 * s]);
                    if constexpr (H >= 4) v += c4 * (ur[i + 4 * s] + ur[i - 4 * s]);
                    orow[i] = v;
                }
            }
        }
    }

    // axis 0/1/2; all_j: span covers every y row (x intermediate), else the
    // active rows; k0/k1: z-slice range
    void run(double *o, const double *u, int axis, double dx,
             const unsigned int *sz, bool all_j, unsigned int k0,
             unsigned int k1) const {
        const long nx = sz[0], ny = sz[1], pw = p_pw;
        const long s  = axis == 0 ? 1 : axis == 1 ? nx : nx * ny;
        const long off = all_j ? pw : pw + nx * pw;
        const long len = all_j ? nx * ny - 2 * pw : nx * (ny - 2 * pw) - 2 * pw;
        double c[5] = {0, 0, 0, 0, 0};
        weights(c);
        const double scale = (DerivOrder == 1) ? 1.0 / dx : 1.0 / (dx * dx);
        apply(o, u, c, scale, s, off, len, k0, k1, (size_t)nx * ny);
    }

   public:
    explicit ExplicitSimdDerivs(unsigned int ele_order) : Derivs(ele_order) {
        if (p_pw < H)
            throw std::invalid_argument("E" + std::to_string(Order) +
                                        "Simd needs a padding width >= " +
                                        std::to_string(H));
        if constexpr (Order == 4) {
            if constexpr (DerivOrder == 1) fallback_ = std::make_unique<ExplicitDerivsO4_DX>(ele_order);
            else fallback_ = std::make_unique<ExplicitDerivsO4_DXX>(ele_order);
        } else if constexpr (Order == 6) {
            if constexpr (DerivOrder == 1) fallback_ = std::make_unique<ExplicitDerivsO6_DX>(ele_order);
            else fallback_ = std::make_unique<ExplicitDerivsO6_DXX>(ele_order);
        } else {
            if constexpr (DerivOrder == 1) fallback_ = std::make_unique<ExplicitDerivsO8_DX>(ele_order);
            else fallback_ = std::make_unique<ExplicitDerivsO8_DXX>(ele_order);
        }
    }
    ExplicitSimdDerivs(const ExplicitSimdDerivs &o)
        : Derivs(o), fallback_(o.fallback_->clone()) {}

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitSimdDerivs>(*this);
    }

    // plain x: all y rows and z slices (may feed y/z); plain y: active rows,
    // all z (may feed z); z: active only
    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        if (bflag) return fallback_->do_grad_x(du, u, dx, sz, bflag);
        run(du, u, 0, dx, sz, true, 0, sz[2]);
    }
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        if (bflag) return fallback_->do_grad_y(du, u, dx, sz, bflag);
        run(du, u, 1, dx, sz, false, 0, sz[2]);
    }
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        if (bflag) return fallback_->do_grad_z(du, u, dx, sz, bflag);
        run(du, u, 2, dx, sz, false, p_pw, sz[2] - p_pw);
    }
    void do_grad_x_last(double *const du, const double *const u,
                        const double dx, const unsigned int *sz,
                        const unsigned int bflag) override {
        if (bflag) return fallback_->do_grad_x_last(du, u, dx, sz, bflag);
        run(du, u, 0, dx, sz, false, p_pw, sz[2] - p_pw);
    }
    void do_grad_y_last(double *const du, const double *const u,
                        const double dx, const unsigned int *sz,
                        const unsigned int bflag) override {
        if (bflag) return fallback_->do_grad_y_last(du, u, dx, sz, bflag);
        run(du, u, 1, dx, sz, false, p_pw, sz[2] - p_pw);
    }

    DerivType getDerivType() const override {
        return Order == 4 ? DerivType::D_E4 : Order == 6 ? DerivType::D_E6 : DerivType::D_E8;
    }
    enum DerivOrder getDerivOrder() const override {
        return DerivOrder == 1 ? D_FIRST_ORDER : D_SECOND_ORDER;
    }
    std::string toString() const override {
        return "E" + std::to_string(Order) + "Simd_" +
               (DerivOrder == 1 ? "FirstOrder" : "SecondOrder");
    }
    void set_maximum_block_size(size_t) override {}
};

}  // namespace dendroderivs
