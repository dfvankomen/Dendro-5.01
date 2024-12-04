#pragma once

#include <string>

#include "derivatives.h"

namespace dendroderivs {

template <unsigned int P>
void deriv42_x(double *const Dxu, const double *const u, const double dx,
               const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv42_y(double *const Dyu, const double *const u, const double dy,
               const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv42_z(double *const Dzu, const double *const u, const double dz,
               const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv42_xx(double *const DxDxu, const double *const u, const double dx,
                const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv42_yy(double *const DyDyu, const double *const u, const double dy,
                const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv42_zz(double *const DzDzu, const double *const u, const double dz,
                const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv644_x(double *const Dxu, const double *const u, const double dx,
                const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv644_y(double *const Dyu, const double *const u, const double dy,
                const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv644_z(double *const Dzu, const double *const u, const double dz,
                const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv644_xx(double *const DxDxu, const double *const u, const double dx,
                 const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv644_yy(double *const DyDyu, const double *const u, const double dy,
                 const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv644_zz(double *const DzDzu, const double *const u, const double dz,
                 const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv8666_x(double *const Dxu, const double *const u, const double dx,
                 const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv8666_y(double *const Dyu, const double *const u, const double dy,
                 const unsigned int *sz, unsigned bflag);

template <unsigned int P>
void deriv8666_z(double *const Dzu, const double *const u, const double dz,
                 const unsigned int *sz, unsigned bflag);

class ExplicitDerivsO4_DX : public Derivs {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradx_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        grady_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradz_func;

   public:
    template <typename... Args>
    ExplicitDerivsO4_DX(unsigned int ele_order, Args &&...)
        : Derivs(ele_order) {
        if (p_pw == 2) {
            gradx_func = &deriv42_x<2>;
            grady_func = &deriv42_y<2>;
            gradz_func = &deriv42_z<2>;
        } else if (p_pw == 3) {
            gradx_func = &deriv42_x<3>;
            grady_func = &deriv42_y<3>;
            gradz_func = &deriv42_z<3>;
        } else if (p_pw == 4) {
            gradx_func = &deriv42_x<4>;
            grady_func = &deriv42_y<4>;
            gradz_func = &deriv42_z<4>;
        } else if (p_pw == 5) {
            gradx_func = &deriv42_x<5>;
            grady_func = &deriv42_y<5>;
            gradz_func = &deriv42_z<5>;
        } else {
            throw std::invalid_argument(
                "Explicit DerivsO4 DX requires a padding width of 2 to 5! pw=" +
                std::to_string(p_ele_order));
        }
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO4_DX>(*this);
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradx_func(du, u, dx, sz, bflag);
    }
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        grady_func(du, u, dx, sz, bflag);
    }
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradz_func(du, u, dx, sz, bflag);
    }

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_FirstOrder";
    };
};

class ExplicitDerivsO4_DXX : public Derivs {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradx_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        grady_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradz_func;

   public:
    template <typename... Args>
    ExplicitDerivsO4_DXX(unsigned int ele_order, Args &&...)
        : Derivs(ele_order) {
        if (p_pw == 2) {
            gradx_func = &deriv42_xx<2>;
            grady_func = &deriv42_yy<2>;
            gradz_func = &deriv42_zz<2>;
        } else if (p_pw == 3) {
            gradx_func = &deriv42_xx<3>;
            grady_func = &deriv42_yy<3>;
            gradz_func = &deriv42_zz<3>;
        } else if (p_pw == 4) {
            gradx_func = &deriv42_xx<4>;
            grady_func = &deriv42_yy<4>;
            gradz_func = &deriv42_zz<4>;
        } else if (p_pw == 5) {
            gradx_func = &deriv42_xx<5>;
            grady_func = &deriv42_yy<5>;
            gradz_func = &deriv42_zz<5>;
        } else {
            throw std::invalid_argument(
                "Explicit DerivsO4 DXX requires a padding width of 2 to 5!");
        }
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO4_DXX>(*this);
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradx_func(du, u, dx, sz, bflag);
    }
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        grady_func(du, u, dx, sz, bflag);
    }
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradz_func(du, u, dx, sz, bflag);
    }

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_SecondOrder";
    };
};

class ExplicitDerivsO6_DX : public Derivs {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradx_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        grady_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradz_func;

   public:
    template <typename... Args>
    ExplicitDerivsO6_DX(unsigned int ele_order, Args &&...)
        : Derivs(ele_order) {
        if (p_pw == 3) {
            gradx_func = &deriv644_x<3>;
            grady_func = &deriv644_y<3>;
            gradz_func = &deriv644_z<3>;
        } else if (p_pw == 4) {
            gradx_func = &deriv644_x<4>;
            grady_func = &deriv644_y<4>;
            gradz_func = &deriv644_z<4>;
        } else if (p_pw == 5) {
            gradx_func = &deriv644_x<5>;
            grady_func = &deriv644_y<5>;
            gradz_func = &deriv644_z<5>;
        } else {
            throw std::invalid_argument(
                "Explicit DerivsO4 DX requires a padding width of 2 to 5!");
        }
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO6_DX>(*this);
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradx_func(du, u, dx, sz, bflag);
    }
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        grady_func(du, u, dx, sz, bflag);
    }
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradz_func(du, u, dx, sz, bflag);
    }

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_FirstOrder";
    };
};

class ExplicitDerivsO6_DXX : public Derivs {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradx_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        grady_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradz_func;

   public:
    template <typename... Args>
    ExplicitDerivsO6_DXX(unsigned int ele_order, Args &&...)
        : Derivs(ele_order) {
        if (p_pw == 3) {
            gradx_func = &deriv644_xx<3>;
            grady_func = &deriv644_yy<3>;
            gradz_func = &deriv644_zz<3>;
        } else if (p_pw == 4) {
            gradx_func = &deriv644_xx<4>;
            grady_func = &deriv644_yy<4>;
            gradz_func = &deriv644_zz<4>;
        } else if (p_pw == 5) {
            gradx_func = &deriv644_xx<5>;
            grady_func = &deriv644_yy<5>;
            gradz_func = &deriv644_zz<5>;
        } else {
            throw std::invalid_argument(
                "Explicit DerivsO4 DXX requires a padding width of 2 to 5!");
        }
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO6_DXX>(*this);
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradx_func(du, u, dx, sz, bflag);
    }
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        grady_func(du, u, dx, sz, bflag);
    }
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradz_func(du, u, dx, sz, bflag);
    }

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_SecondOrder";
    };
};

class ExplicitDerivsO8_DX : public Derivs {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradx_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        grady_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradz_func;

   public:
    template <typename... Args>
    ExplicitDerivsO8_DX(unsigned int ele_order, Args &&...)
        : Derivs(ele_order) {
        if (p_pw == 4) {
            gradx_func = &deriv644_x<4>;
            grady_func = &deriv644_y<4>;
            gradz_func = &deriv644_z<4>;
        } else if (p_pw == 5) {
            gradx_func = &deriv644_x<5>;
            grady_func = &deriv644_y<5>;
            gradz_func = &deriv644_z<5>;
        } else {
            throw std::invalid_argument(
                "Explicit DerivsO4 DX requires a padding width of 2 to 5!");
        }
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO8_DX>(*this);
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradx_func(du, u, dx, sz, bflag);
    }
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        grady_func(du, u, dx, sz, bflag);
    }
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradz_func(du, u, dx, sz, bflag);
    }

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_FIRST_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_FirstOrder";
    };
};

class ExplicitDerivsO8_DXX : public Derivs {
   private:
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradx_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        grady_func;
    std::function<void(double *const, const double *, const double,
                       const unsigned int *, const unsigned int)>
        gradz_func;

   public:
    template <typename... Args>
    ExplicitDerivsO8_DXX(unsigned int ele_order, Args &&...)
        : Derivs(ele_order) {
        if (p_pw == 4) {
            gradx_func = &deriv644_xx<4>;
            grady_func = &deriv644_yy<4>;
            gradz_func = &deriv644_zz<4>;
        } else if (p_pw == 5) {
            gradx_func = &deriv644_xx<5>;
            grady_func = &deriv644_yy<5>;
            gradz_func = &deriv644_zz<5>;
        } else {
            throw std::invalid_argument(
                "Explicit DerivsO4 DXX requires a padding width of 2 to 5!");
        }
    }

    std::unique_ptr<Derivs> clone() const override {
        return std::make_unique<ExplicitDerivsO8_DXX>(*this);
    }

    void do_grad_x(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradx_func(du, u, dx, sz, bflag);
    }
    void do_grad_y(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        grady_func(du, u, dx, sz, bflag);
    }
    void do_grad_z(double *const du, const double *const u, const double dx,
                   const unsigned int *sz, const unsigned int bflag) override {
        gradz_func(du, u, dx, sz, bflag);
    }

    DerivType getDerivType() const override { return DerivType::D_E4; }
    DerivOrder getDerivOrder() const override {
        return DerivOrder::D_SECOND_ORDER;
    }

    std::string toString() const override {
        return "ExplicitDerivsO4_SecondOrder";
    };
};

}  // namespace dendroderivs
