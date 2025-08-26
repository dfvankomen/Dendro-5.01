#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "dendro.h"
#include "filters.h"
#include "libxsmm.h"

#define IDX(i, j, k) ((i) + nx * ((j) + ny * (k)))

namespace dendroderivs {

// A global check for the derivatives padding width, should be the same as
// Derivatives' pw used in the matrix multiplication routines to know the PW
// without having to pass it through
extern unsigned int DENDRO_DERIVS_PW;

enum DerivType {
    // default, does nothing, should return an error if anything is
    D_NONE = 0,
    // Explicit Schemes
    D_E4,
    D_E6,
    D_E8,
    // Explicit Schemes with matrix multiplication
    D_E4_MAT,
    D_E6_MAT,
    D_E8_MAT,
    // Jonathan Tyler tridiagonal schemes (compact)
    D_JTT4,
    D_JTT6,
    D_JTT8,
    D_JTT10,
    // Jonathan Tyler pentadiagonal schemes (compact)
    D_JTP6,
    D_JTP8,
    D_JTP10,
    // J.W. Kim 4th order derivatives
    D_K4,
    D_K4_07,
    D_K4_16,
    D_KB4,
    D_KIMBYU_1,
    D_KIMBYU_2,
    D_KIMBYU_3,
    D_KIMBYU_4,
    D_KIMBYU_5,
    D_KIMBYU_6,
    D_KIMBYU_7,
    D_KIMBYU_8,
    D_KIMBYU_9,
    D_KIMBYU_10,
    D_KIMBYU_11,
    D_KIMBYU_12,
    D_KIMBYU_13,
    D_KIMBYU_14,
    D_KIMBYU_15,
    D_KIMBYU_16,
    D_KIMBYU_17,
    D_KIMBYU_18,
    D_KIMBYU_19,
    D_C4_1,
    D_C4_2,
    D_C4_3,
    D_C4_4,
    D_C4_5,
    D_A6_1,
    D_A6_2,
    D_A6_3,
    D_C6_1,
    D_C6_2,
    D_C6_3,
    D_C6_4,
    D_C6_5,
    D_C6_6,
    D_C6_7,
    D_C6_8,
    D_C6_9,
    D_C6_10,
    D_C6_11,
    D_C6_12,
    D_C6_13,
    D_C6_14,
    D_C6_15,
    D_C6_16,
    D_UC6_1,
    D_UC6_2,
    D_UC6_3,
    D_UC6_4,
    D_UA4_4,
    D_UA4_5,
    D_UA4_6,
    // Brady Livescue Derivatives
    D_BL6,
    // BORIS derivatives
    D_BORISO4,
    D_BORISO6,
    D_BORISO6_ETA,
    // BYU TRIDIAGONAL FREE PARAMTERS
    D_BYUT4,
    D_BYUT4R1,
    D_BYUT4R2,
    D_BYUT4R4,
    D_BYUT6,
    D_BYUT6R1,
    D_BYUT6R2,
    D_BYUT6R4,
    D_BYUT6R3,
    D_BYUT64R3,
    D_BYUT8,
    // BYU PENTADIAGONAL FREE PARAMETERS (Nate and James)
    D_BYUP6,
    D_BYUP6R3,
    D_BYUP6R2,
    D_BYUP8,
    D_BYUP8B7,
    // TwoA and TwoUA Second-Order Derivatives
   D_2A_1,
   D_2A_2,
   D_2A_3,
   D_2A_4,
   D_2A_5,
   D_2A_6,
   D_2A_7,
   D_2A_8,
   D_2A_9,
   D_2A_10,
   D_2A_11,
   D_2A_12,
   D_2UA_1,
   D_2UA_2,    
   D_2B4_1,
   D_2B6_1,



};

enum DerivOrder { D_FIRST_ORDER = 0, D_SECOND_ORDER };

enum DerivFamily {
    DF_NONE = 0,
    DF_EXPLICIT,
    DF_JTT,    // Jonathan Tyler tridiagonal
    DF_JTP,    // Jonathan Tyler pentadiagonal
    DF_KIM,    // Kim
    DF_BL,     // Brady Livescue
    DF_BORIS,  // Boris
    DF_BYUT,   // BYU tridiagonal
    DF_BYUP    // BYU Pentadiagonal
};

/**
 * @class Derivs
 * @brief An Abstract base class for all dendro derivative calculations
 *
 * This class serves as a base for various derivative calculation methods.
 * This includes first and second order derivatives, explicit methods, and
 * compact methods. It primarily is to provide a common interface for the
 * different types and orders of derivatives.
 */
class Derivs {
   protected:
    unsigned int p_n;          ///< Size of the derivative calculation
    unsigned int p_pw;         ///< Size of the padding width
    unsigned int p_ele_order;  ///< Element order for the computations

    /**
     * @brief Protected constructor to initialize a Derivs object.
     * @param dtype The type of derivative.
     * @param dorder The order of derivative.
     * @param n The size or dimension of the derivative calculation.
     *
     * Do note that in the case of pure explicit schemes, n does not need
     * to implement anything. Additionally, do_grad should check the size of
     * stored n vs incoming n to avoid any errors.
     */
    Derivs(unsigned int ele_order) : p_ele_order{ele_order} {
        p_n  = p_ele_order * 2 + 1;
        p_pw = p_ele_order / 2;
    }

    /**
     * @brief Copy constructor.
     * @param obj The Derivs object to copy from.
     *
     * TODO: Implement full deep copy method for the derivatives, as
     * shallow copies would not be appropriate.
     */
    Derivs(const Derivs &obj) {};

   public:
    /**
     * @brief Destructor
     */
    virtual ~Derivs() {};

    virtual std::unique_ptr<Derivs> clone() const    = 0;

    /**
     * @brief Pure virtual function to calculate the derivative.
     * @param du Pointer to where calculated derivative will be stored.
     * @param u Pointer to the input array.
     * @param dx The step size or grid spacing.
     * @param sz The number of points expected to be calculated in each dim
     * @param bflag The boundary flag
     */
    virtual void do_grad_x(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;
    virtual void do_grad_y(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;
    virtual void do_grad_z(double *const du, const double *const u,
                           const double dx, const unsigned int *sz,
                           const unsigned int bflag) = 0;

    /**
     * @brief Get the type of derivative.
     * @return The derivative type.
     */
    virtual DerivType getDerivType() const           = 0;

    /**
     * @brief Get the order of derivative.
     * @return The derivative order.
     */
    virtual DerivOrder getDerivOrder() const         = 0;

    /**
     * @brief Get the size or dimension of the derivative calculation.
     * @return The size the derivative is expecting.
     */
    inline unsigned int getN() const { return p_n; }

    /**
     * @brief Get a string representation.
     * @return A string describing the Derivs object.
     */
    virtual std::string toString() const                   = 0;

    virtual void set_maximum_block_size(size_t block_size) = 0;
};

class DendroDerivatives {
   private:
    std::unique_ptr<Derivs> _first_deriv;
    std::unique_ptr<Derivs> _second_deriv;
    // NOTE: unique ptr will automatically delete the object once out of scope
    // or if this object is deleted

    std::unique_ptr<Filters> _filter;

    std::unique_ptr<double[]> _derivative_space;

    unsigned int _n_points_deriv_space;
    unsigned int _n_vars_deriv_space;

   public:
    DendroDerivatives(
        const std::string derivType_1 = "E4",
        const std::string derivType_2 = "E4", const unsigned int ele_order = 13,
        const std::vector<double> &coeffs_in_1 = std::vector<double>(),
        const std::vector<double> &coeffs_in_2 = std::vector<double>(),
        const unsigned int deriv1_matrixID     = 0,
        const unsigned int deriv2_matrixID     = 0,
        const std::string inMatrixFilterType_1 = "none",
        const std::string inMatrixFilterType_2 = "none",
        const std::vector<double> &in_matrix_coeffs_in_1 =
            std::vector<double>(),
        const std::vector<double> &in_matrix_coeffs_in_2 =
            std::vector<double>(),
        const std::string postRHSFilterType = "default");

    ~DendroDerivatives() = default;

    // Copy constructor
    DendroDerivatives(const DendroDerivatives &other)
        : _first_deriv(other._first_deriv ? other._first_deriv->clone()
                                          : nullptr),
          _second_deriv(other._second_deriv ? other._second_deriv->clone()
                                            : nullptr),
          _n_points_deriv_space(other._n_points_deriv_space),
          _n_vars_deriv_space(other._n_vars_deriv_space) {
        // if the incoming copy has data in derivative space, we need to copy it
        if (other._derivative_space) {
            _derivative_space = std::make_unique<double[]>(
                _n_points_deriv_space * _n_vars_deriv_space);

            std::copy(other._derivative_space.get(),
                      other._derivative_space.get() +
                          _n_points_deriv_space * _n_vars_deriv_space,
                      _derivative_space.get());
        }
    }

    // Copy assignment operator
    DendroDerivatives &operator=(const DendroDerivatives &other) {
        if (this != &other) {
            _first_deriv =
                other._first_deriv ? other._first_deriv->clone() : nullptr;
            _second_deriv =
                other._second_deriv ? other._second_deriv->clone() : nullptr;

            _n_points_deriv_space = other._n_points_deriv_space;
            _n_vars_deriv_space   = other._n_vars_deriv_space;

            if (other._derivative_space) {
                _derivative_space = std::make_unique<double[]>(
                    _n_points_deriv_space * _n_vars_deriv_space);

                std::copy(
                    other._derivative_space.get(),
                    other._derivative_space.get() +
                        other._n_points_deriv_space * other._n_vars_deriv_space,
                    _derivative_space.get());
            } else {
                _derivative_space.reset();
            }
        }
        return *this;
    }

    void allocate_derivative_space(const unsigned int n_points,
                                   const unsigned int n_var) {
        if (_derivative_space) {
            // delete the data if we're already at size
            _derivative_space.reset(new double[n_points * n_var]);
        } else {
            _derivative_space = std::make_unique<double[]>(n_points * n_var);
        }

        _n_points_deriv_space = n_points;
        _n_vars_deriv_space   = n_var;
    }

    double *get_derivative_space(const unsigned int var_no) const {
        return _derivative_space.get() + var_no * _n_points_deriv_space;
    }

    void grad_x(double *du, const double *u, double dx, const unsigned int *sz,
                unsigned int bflag) {
        _first_deriv->do_grad_x(du, u, dx, sz, bflag);
    }
    void grad_y(double *du, const double *u, double dx, const unsigned int *sz,
                unsigned int bflag) {
        _first_deriv->do_grad_y(du, u, dx, sz, bflag);
    }
    void grad_z(double *du, const double *u, double dx, const unsigned int *sz,
                unsigned int bflag) {
        _first_deriv->do_grad_z(du, u, dx, sz, bflag);
    }
    void grad_xx(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        _second_deriv->do_grad_x(du, u, dx, sz, bflag);
    }
    void grad_yy(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        _second_deriv->do_grad_y(du, u, dx, sz, bflag);
    }
    void grad_zz(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        _second_deriv->do_grad_z(du, u, dx, sz, bflag);
    }

    // deriv renaming of grad naming
    void deriv_x(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        _first_deriv->do_grad_x(du, u, dx, sz, bflag);
    }
    void deriv_y(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        _first_deriv->do_grad_y(du, u, dx, sz, bflag);
    }
    void deriv_z(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        _first_deriv->do_grad_z(du, u, dx, sz, bflag);
    }
    void deriv_xx(double *du, const double *u, double dx,
                  const unsigned int *sz, unsigned int bflag) {
        _second_deriv->do_grad_x(du, u, dx, sz, bflag);
    }
    void deriv_yy(double *du, const double *u, double dx,
                  const unsigned int *sz, unsigned int bflag) {
        _second_deriv->do_grad_y(du, u, dx, sz, bflag);
    }
    void deriv_zz(double *du, const double *u, double dx,
                  const unsigned int *sz, unsigned int bflag) {
        _second_deriv->do_grad_z(du, u, dx, sz, bflag);
    }

    void filter(const double *const input, double *const output,
                double *const workspace_x, double *const workspace_y,
                double *const workspace_z, const double dx, const double dy,
                const double dz, const double coeff, const unsigned int *sz,
                const unsigned int bflag) {
        _filter->do_full_filter(input, output, workspace_x, workspace_y,
                                workspace_z, dx, dy, dz, coeff, sz, bflag);
    }

    bool inline do_filter_before() { return _filter->do_filter_before(); }

    void set_maximum_block_size(size_t block_size) {
        _first_deriv->set_maximum_block_size(block_size);
        _second_deriv->set_maximum_block_size(block_size);
    }

    std::string toString() {
        return "DendroDerivs<" + _first_deriv->toString() + ", " +
               _second_deriv->toString() + ", " + _filter->toString() + ">";
    }

    FilterFamily get_filter_family() { return _filter->get_filter_family(); }
};

void inline initialize_derivatives() { libxsmm_init(); }

void inline finalize_derivatives() { libxsmm_finalize(); }

class DendroDerivsNotImplemented : public std::exception {
   private:
    std::string message_;

   public:
    explicit DendroDerivsNotImplemented(const std::string &msg)
        : message_(msg) {}
    const char *what() { return message_.c_str(); }
};

}  // namespace dendroderivs
