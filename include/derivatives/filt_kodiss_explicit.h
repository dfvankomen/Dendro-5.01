#pragma once

#include "derivatives.h"

namespace dendroderivs {

template <unsigned int P>
void ko_deriv42_x(double *const Dxu, const double *const u, const double dx,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv42_y(double *const Dyu, const double *const u, const double dy,
                  const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void ko_deriv42_z(double *const Dzu, const double *const u, const double dz,
                  const unsigned int *sz, unsigned bflag);

enum FilterType { F_NONE = 0, F_KO4, F_KO6, F_KO8 };

class Filters {
   protected:
    unsigned int p_n;
    unsigned int p_pw;
    unsigned int p_ele_order;

    Filters(unsigned int ele_order) : p_ele_order{ele_order} {
        p_n  = p_ele_order * 2 + 1;
        p_pw = p_ele_order / 2;
    }

    Filters(const Filters &obj){};

   public:
    virtual ~Filters(){};

    virtual std::unique_ptr<Filters> clone() const   = 0;

    virtual void do_filt_x(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;
    virtual void do_filt_y(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;
    virtual void do_filt_z(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;

    virtual std::string toString() const             = 0;
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
        if (p_pw == 2) {
            kox_func = &ko_deriv42_x<2>;
            koy_func = &ko_deriv42_y<2>;
            koz_func = &ko_deriv42_z<2>;
        } else if (p_pw == 3) {
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
                "Explicit KODissO4 requires a padding width of 2 to 5! pw=" +
                std::to_string(p_ele_order));
        }
    }

    std::unique_ptr<Filters> clone() const override {
        return std::make_unique<ExplicitKODissO4>(*this);
    }

    void do_filt_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        kox_func(du, u, dx, sz, bflag);
    }
    void do_filt_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        koy_func(du, u, dx, sz, bflag);
    }
    void do_filt_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        koz_func(du, u, dx, sz, bflag);
    }

    std::string toString() const override { return "ExplicitKODissO4"; };
};

}  // namespace dendroderivs
