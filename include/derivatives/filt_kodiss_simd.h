#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "derivatives/filt_kodiss_explicit.h"
#include "filters.h"

namespace dendroderivs {

/**
 * @brief Kreiss-Oliger dissipation as one span-vectorized pass
 * (registry names KO2Simd / KO4Simd / KO6Simd / KO8Simd).
 *
 * Interior blocks (bflag == 0): a single `omp simd` loop per z-slice over the
 * contiguous (x,y)-plane span between the first and last active point
 * evaluates the x, y and z stencils at each point and accumulates
 * coeff * (Dx + Dy + Dz) straight into the output. No per-axis workspaces are
 * written or re-read, so the pass is one read of u and one read-modify-write
 * of the output instead of three writes, four reads and an add pass. Padding
 * cells inside the span are computed and never read. The stencil expressions
 * are written term for term as in the loop kernels (filt_kodiss_explicit.cpp),
 * so the result is bit-identical to them where the compiler contracts both
 * the same way, and within roundoff otherwise. Boundary blocks delegate to the
 * wrapped stencil filter so the one-sided rows are the identical code.
 */
template <unsigned int Order>
class SimdKODiss : public Filters {
    static_assert(Order == 2 || Order == 4 || Order == 6 || Order == 8,
                  "KO2/4/6/8 only");
    static constexpr long R = Order / 2 + 1;  // stencil radius

    std::unique_ptr<Filters> stencil_;  // loop kernels: bflag != 0
    std::vector<double> ws_;            // do_accumulate on a boundary block

    // the per-axis stencil at p with stride s, same term order as the loop kernel
    static inline double __attribute__((always_inline))
    ko(const double *__restrict__ p, long s, double pre) {
        if constexpr (Order == 2)
            return pre * (1.0 * p[-2 * s] - 4.0 * p[-s] + 6.0 * p[0] -
                          4.0 * p[s] + 1.0 * p[2 * s]);
        else if constexpr (Order == 4)
            return pre * (-p[-3 * s] + 6.0 * p[-2 * s] - 15.0 * p[-s] +
                          20.0 * p[0] - 15.0 * p[s] + 6.0 * p[2 * s] - p[3 * s]);
        else if constexpr (Order == 6)
            return pre * (p[-4 * s] - 8.0 * p[-3 * s] + 28.0 * p[-2 * s] -
                          56.0 * p[-s] + 70.0 * p[0] - 56.0 * p[s] +
                          28.0 * p[2 * s] - 8.0 * p[3 * s] + p[4 * s]);
        else
            return pre * (p[-5 * s] - 10.0 * p[-4 * s] + 45.0 * p[-3 * s] -
                          120.0 * p[-2 * s] + 210.0 * p[-s] - 252.0 * p[0] +
                          210.0 * p[s] - 120.0 * p[2 * s] + 45.0 * p[3 * s] -
                          10.0 * p[4 * s] + 1.0 * p[5 * s]);
    }

    // the loop kernels' prefactor, spelled the same way so it rounds the same
    static inline double prefactor(double h) {
        if constexpr (Order == 2) return -1.0 / 16.0 / h;
        else if constexpr (Order == 4) return -1.0 / 64.0 / h;
        else if constexpr (Order == 6) return -1.0 / 256.0 / h;
        else return +1.0 / (1024.0 * h);  // sign fixed with the stencil 2026-08-29
    }

    // out += c(i) * (Dx + Dy + Dz) u over the active z-slices, one span each
    template <bool Field>
    void fused(const double *__restrict__ u, double *__restrict__ out,
               const double *__restrict__ cf, double c0, double dx, double dy,
               double dz, const unsigned int *sz) const {
        const long nx = sz[0], ny = sz[1], nz = sz[2], pw = p_pw;
        const long nxy = nx * ny;
        const long off = pw + nx * pw;
        const long len = nx * (ny - 2 * pw) - 2 * pw;
        const double px = prefactor(dx), py = prefactor(dy), pz = prefactor(dz);
        for (long k = pw; k < nz - pw; k++) {
            const double *__restrict__ ur = u + nxy * k + off;
            double *__restrict__ orow     = out + nxy * k + off;
            const double *__restrict__ cr = Field ? cf + nxy * k + off : nullptr;
#pragma omp simd
            for (long i = 0; i < len; i++) {
                const double v = ko(ur + i, 1, px) + ko(ur + i, nx, py) +
                                 ko(ur + i, nxy, pz);
                orow[i] += (Field ? cr[i] : c0) * v;
            }
        }
    }

   public:
    SimdKODiss(unsigned int ele_order, std::unique_ptr<Filters> stencil)
        : Filters(ele_order), stencil_(std::move(stencil)) {
        if (!stencil_) throw std::invalid_argument("SimdKODiss: null stencil filter");
        if ((long)p_pw < R)
            throw std::invalid_argument("KO" + std::to_string(Order) +
                                        "Simd needs a padding width >= " +
                                        std::to_string(R));
    }
    SimdKODiss(const SimdKODiss &o)
        : Filters(o), stencil_(o.stencil_->clone()) {}

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<SimdKODiss>(*this);
    }

    void do_full_filter(const double *const input, double *const output,
                        double *const workspace_x, double *const workspace_y,
                        double *const workspace_z, const double dx,
                        const double dy, const double dz, const double coeff,
                        const unsigned int *sz,
                        const unsigned int bflag) override {
        if (bflag)
            return stencil_->do_full_filter(input, output, workspace_x,
                                            workspace_y, workspace_z, dx, dy,
                                            dz, coeff, sz, bflag);
        fused<false>(input, output, nullptr, coeff, dx, dy, dz, sz);
    }

    void do_full_filter_field(const double *const input, double *const output,
                              double *const workspace_x,
                              double *const workspace_y,
                              double *const workspace_z, const double dx,
                              const double dy, const double dz,
                              const double *const coeff_field,
                              const unsigned int *sz,
                              const unsigned int bflag) override {
        if (bflag)
            return stencil_->do_full_filter_field(
                input, output, workspace_x, workspace_y, workspace_z, dx, dy,
                dz, coeff_field, sz, bflag);
        fused<true>(input, output, coeff_field, 0.0, dx, dy, dz, sz);
    }

    // the fused pass already is rhs += coeff * KO(u); boundary blocks go
    // through the stencil with an instance workspace
    bool do_accumulate(double *const rhs, const double *const u,
                       const double coeff, const double dx, const double dy,
                       const double dz, const unsigned int *sz,
                       const unsigned int bflag) override {
        if (bflag) {
            const size_t tot = (size_t)sz[0] * sz[1] * sz[2];
            if (ws_.size() < 3 * tot) ws_.resize(3 * tot);
            stencil_->do_full_filter(u, rhs, ws_.data(), ws_.data() + tot,
                                     ws_.data() + 2 * tot, dx, dy, dz, coeff,
                                     sz, bflag);
            return true;
        }
        fused<false>(u, rhs, nullptr, coeff, dx, dy, dz, sz);
        return true;
    }

    std::string toString() const override {
        return "KO" + std::to_string(Order) + "Simd";
    }
    bool do_filter_before() const override { return false; }
    void set_maximum_block_size(size_t) override {}
    FilterFamily get_filter_family() const override {
        return dendroderivs::FilterFamily::FF_KO;
    }
};

}  // namespace dendroderivs
