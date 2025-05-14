#pragma once

#include "derivatives.h"
#include "filters.h"

namespace dendroderivs {

template <unsigned int P>
void ko_deriv21_x(double *const Dxu, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv21_y(double *const Dyu, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv21_z(double *const Dzu, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv42_x(double *const Dxu, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv42_y(double *const Dyu, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv42_z(double *const Dzu, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv64_x(double *const Dxu, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv64_y(double *const Dyu, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv64_z(double *const Dzu, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag);

void inline do_ko_single(const double *const calc_dx,
                         const double *const calc_dy,
                         const double *const calc_dz, double *const output,
                         const double coeff, const unsigned int *sz,
                         const unsigned int bflag, const unsigned int PW) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    for (unsigned int k = PW; k < nz - PW; k++) {
        for (unsigned int j = PW; j < ny - PW; j++) {
            for (unsigned int i = PW; i < nx - PW; i++) {
                const unsigned int pp = i + nx * (j + ny * k);
                output[pp] += coeff * (calc_dx[pp] + calc_dy[pp] + calc_dz[pp]);
            }
        }
    }
}

class ExplicitKODissO2 : public Filters {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        kox_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koy_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koz_func;

   public:
    template <typename... Args>
    ExplicitKODissO2(unsigned int ele_order, Args &&...) : Filters(ele_order) {
        if (p_pw == 2) {
            kox_func = &ko_deriv21_x<3>;
            koy_func = &ko_deriv21_y<3>;
            koz_func = &ko_deriv21_z<3>;
        } else if (p_pw == 3) {
            kox_func = &ko_deriv21_x<3>;
            koy_func = &ko_deriv21_y<3>;
            koz_func = &ko_deriv21_z<3>;
        } else if (p_pw == 4) {
            kox_func = &ko_deriv21_x<4>;
            koy_func = &ko_deriv21_y<4>;
            koz_func = &ko_deriv21_z<4>;
        } else if (p_pw == 5) {
            kox_func = &ko_deriv21_x<5>;
            koy_func = &ko_deriv21_y<5>;
            koz_func = &ko_deriv21_z<5>;
        } else {
            throw std::invalid_argument(
                "Explicit KODissO4 requires a padding width of 3 to 5! pw=" +
                std::to_string(p_ele_order));
        }
    }

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<ExplicitKODissO2>(*this);
    }

    // for KO diss, input is our variable itself, and output is the in-place
    // modified u_rhs
    virtual void do_full_filter(const double *const input, double *const output,
                                double *const workspace_x,
                                double *const workspace_y,
                                double *const workspace_z, const double dx,
                                const double dy, const double dz,
                                const double coeff, const unsigned int *sz,
                                const unsigned int bflag) override {
        // calculate filt x, y, and z
        kox_func(workspace_x, input, dx, sz, bflag);
        koy_func(workspace_y, input, dy, sz, bflag);
        koz_func(workspace_z, input, dz, sz, bflag);

        do_ko_single(workspace_x, workspace_y, workspace_z, output, coeff, sz,
                     bflag, p_pw);
        // done
    }

    std::string toString() const override { return "ExplicitKODissO2"; };

    bool do_filter_before() const override { return false; }

    void set_maximum_block_size(size_t block_size) override {}

    FilterFamily get_filter_family() const override {
        return dendroderivs::FilterFamily::FF_KO;
    }
};

class ExplicitKODissO4 : public Filters {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        kox_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koy_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koz_func;

   public:
    template <typename... Args>
    ExplicitKODissO4(unsigned int ele_order, Args &&...) : Filters(ele_order) {
        if (p_pw == 3) {
            kox_func = &ko_deriv42_x<3>;
            koy_func = &ko_deriv42_y<3>;
            koz_func = &ko_deriv42_z<3>;
        } else if (p_pw == 4) {
            kox_func = &ko_deriv42_x<4>;
            koy_func = &ko_deriv42_y<4>;
            koz_func = &ko_deriv42_z<4>;
        } else if (p_pw == 5) {
            kox_func = &ko_deriv42_x<5>;
            koy_func = &ko_deriv42_y<5>;
            koz_func = &ko_deriv42_z<5>;
        } else {
            throw std::invalid_argument(
                "Explicit KODissO4 requires a padding width of 3 to 5! pw=" +
                std::to_string(p_ele_order));
        }
    }

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<ExplicitKODissO4>(*this);
    }

    // for KO diss, input is our variable itself, and output is the in-place
    // modified u_rhs
    virtual void do_full_filter(const double *const input, double *const output,
                                double *const workspace_x,
                                double *const workspace_y,
                                double *const workspace_z, const double dx,
                                const double dy, const double dz,
                                const double coeff, const unsigned int *sz,
                                const unsigned int bflag) override {
        kox_func(workspace_x, input, dx, sz, bflag);
        koy_func(workspace_y, input, dy, sz, bflag);
        koz_func(workspace_z, input, dz, sz, bflag);

        do_ko_single(workspace_x, workspace_y, workspace_z, output, coeff, sz,
                     bflag, p_pw);
        // done
    }

    std::string toString() const override { return "ExplicitKODissO4"; };

    bool do_filter_before() const override { return false; }

    void set_maximum_block_size(size_t block_size) override {}

    FilterFamily get_filter_family() const override {
        return dendroderivs::FilterFamily::FF_KO;
    }
};

class ExplicitKODissO6 : public Filters {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        kox_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koy_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koz_func;

   public:
    template <typename... Args>
    ExplicitKODissO6(unsigned int ele_order, Args &&...) : Filters(ele_order) {
        if (p_pw == 4) {
            kox_func = &ko_deriv64_x<4>;
            koy_func = &ko_deriv64_y<4>;
            koz_func = &ko_deriv64_z<4>;
        } else if (p_pw == 5) {
            kox_func = &ko_deriv64_x<5>;
            koy_func = &ko_deriv64_y<5>;
            koz_func = &ko_deriv64_z<5>;
        } else {
            throw std::invalid_argument(
                "Explicit KODissO4 requires a padding width of 4 to 5! pw=" +
                std::to_string(p_ele_order));
        }
    }

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<ExplicitKODissO6>(*this);
    }

    // for KO diss, input is our variable itself, and output is the in-place
    // modified u_rhs
    virtual void do_full_filter(const double *const input, double *const output,
                                double *const workspace_x,
                                double *const workspace_y,
                                double *const workspace_z, const double dx,
                                const double dy, const double dz,
                                const double coeff, const unsigned int *sz,
                                const unsigned int bflag) override {
        // calculate filt x, y, and z
        kox_func(workspace_x, input, dx, sz, bflag);
        koy_func(workspace_y, input, dy, sz, bflag);
        koz_func(workspace_z, input, dz, sz, bflag);

        do_ko_single(workspace_x, workspace_y, workspace_z, output, coeff, sz,
                     bflag, p_pw);
        // done
    }

    std::string toString() const override { return "ExplicitKODissO6"; };

    bool do_filter_before() const override { return false; }

    void set_maximum_block_size(size_t block_size) override {}

    FilterFamily get_filter_family() const override {
        return dendroderivs::FilterFamily::FF_KO;
    }
};

// NOT YET IMPLEMENTED
#if 0
class ExplicitKODissO8 : public Filters {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        kox_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koy_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        koz_func;

   public:
    template <typename... Args>
    ExplicitKODissO8(unsigned int ele_order, Args &&...) : Filters(ele_order) {
         if (p_pw == 5) {
            kox_func = &ko_deriv64_x<5>;
            koy_func = &ko_deriv64_y<5>;
            koz_func = &ko_deriv64_z<5>;
        } else {
            throw std::invalid_argument(
                "Explicit KODissO4 requires a padding width of 2 to 5! pw=" +
                std::to_string(p_ele_order));
        }
    }

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<ExplicitKODissO6>(*this);
    }

    // for KO diss, input is our variable itself, and output is the in-place
    // modified u_rhs
    virtual void do_full_filter(const double *const input, double *const output,
                                double *const workspace_x,
                                double *const workspace_y,
                                double *const workspace_z, const double dx,
                                const double dy, const double dz,
                                const double coeff, const unsigned int *sz,
                                const unsigned int bflag) override {
        // calculate filt x, y, and z
        kox_func(workspace_x, input, dx, sz, bflag);
        koy_func(workspace_y, input, dx, sz, bflag);
        koz_func(workspace_z, input, dx, sz, bflag);

        do_ko_single(workspace_x, workspace_y, workspace_z, output, coeff, sz,
                     bflag, p_pw);
        // done
    }

    std::string toString() const override { return "ExplicitKODissO6"; };

    bool do_filter_before() const override { return false; }

    void set_maximum_block_size(size_t block_size) override {}

    FilterFamily get_filter_family() const override {
        return dendroderivs::FilterFamily::FF_KO;
    }
};
#endif

}  // namespace dendroderivs
