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
    D_A4_1,
    D_A4_2,
    D_A4_3,
    D_A4_4,
    D_A4_5,
    D_A4_6,
    D_A4_7,
    D_A4_8,
    D_A4_9,
    D_A4_10,
    D_A4_11,
    D_A4_12,
    D_A4_13,
    D_A4_14,
    D_A4_15,
    D_A4_16,
    D_A4_17,
    D_A4_18,
    D_A4_19,
    D_A4_20,
    D_B4_1,
    D_B4_2,
    D_C4_1,
    D_C4_2,
    D_C4_3,
    D_C4_4,
    D_C4_5,
    D_A6_1,
    D_A6_2,
    D_A6_3,
    D_A6_4,
    D_A6_5,
    D_A6_6,
    D_A6_7,
    D_A6_8,
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
   D_2B6_2,
   D_2B6_3,
   D_2B6_4,
   D_2B6_5,
   D_2B6_6,
   D_2B6_7,
   D_2B6_8,
   D_2B6_9,
      // C4 family (10 total)
    D_2C4_1,
    D_2C4_2,
    D_2C4_3,
    D_2C4_4,
    D_2C4_5,
    D_2C4_6,
    D_2C4_7,
    D_2C4_8,
    D_2C4_9,
    D_2C4_10,

    // C6 family (1..7, extended from 6)
    D_2C6_1,
    D_2C6_2,
    D_2C6_3,
    D_2C6_4,
    D_2C6_5,
    D_2C6_6,
    D_2C6_7,

    // A6 family (1..7, extended from 6)
    D_2A6_1,
    D_2A6_2,
    D_2A6_3,
    D_2A6_4,
    D_2A6_5,
    D_2A6_6,
    D_2A6_7,





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
 *
 * ## Thread safety
 *
 * A single Derivs instance is **not** safe to use concurrently from multiple
 * threads — internal state (workspace buffers, the per-size D storage map,
 * and the last-used-size memoization inside MatrixCompactDerivs) is mutated
 * during `do_grad_*`. The intended model for multi-threaded callers is:
 *
 *   1. Build one "prototype" Derivs instance on the main thread.
 *   2. Call `clone()` once per thread to get private copies. Clone cost is
 *      small (copies only the already-built matrices; workspaces re-allocate
 *      lazily on first call).
 *   3. Each thread exclusively owns its clone for the duration of the work.
 *
 * The process-wide libxsmm kernel caches in `derivs_utils.h` *are*
 * thread-safe (shared_mutex-protected). For best performance, call
 * `dendroderivs::prewarm_kernel_cache(...)` on the main thread after mesh
 * setup so the hot path only ever takes the shared read lock.
 *
 * ## Workspace sizing (matrix-based subclasses)
 *
 * `MatrixCompactDerivs<DerivOrder>` (the parent of every JTT/JTP/Kim/BORIS/
 * BYUT/BYUP/etc. instantiation) keeps a per-instance scratch buffer used by
 * `matmul_y_dim` / `matmul_z_dim` for the y/z direction transposes. The
 * buffer must hold `2 * Nx * Ny * Nz` doubles for the largest block this
 * instance will ever be called on.
 *
 * The recommended pattern is to call `set_maximum_block_size(max_cells)`
 * once per instance (and once per clone) after mesh setup, with `max_cells`
 * the largest `getAllocationSzX()*Y*Z` over all blocks on this rank. That
 * gives one clean allocation up front and zero growth during timestepping.
 *
 * As a safety net, every `do_grad_y/z` (and the `_batch` variants) checks
 * the workspace size and `resize()`s if a larger block ever arrives. The
 * fast path is a single size_t compare; the slow path is a `std::vector`
 * grow that happens at most a handful of times per run. Forgetting to call
 * `set_maximum_block_size` no longer corrupts memory, but it does cost a
 * heap reallocation on the first oversized block — prefer the explicit
 * call where possible.
 *
 * Explicit/stencil-only Derivs subclasses don't use a workspace and
 * implement `set_maximum_block_size` as a no-op.
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
     * @brief Copy constructor. Copies base-class members (ele_order, n, pw).
     * Derived classes must handle their own deep copy via clone() or their
     * own copy constructors.
     */
    Derivs(const Derivs &obj)
        : p_ele_order(obj.p_ele_order), p_n(obj.p_n), p_pw(obj.p_pw) {};

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
     * "_last" variants: the caller asserts that the output of this
     * derivative will NOT be further differentiated (i.e. this is the
     * last operation in its derivative chain — a solo grad_*, or the
     * last step of a mixed 2nd-order chain like
     * d^2u/dxdy = grad_y(grad_x(u)) where grad_y is last). When that
     * holds, the implementation may skip writing the padding cells of
     * the output, recovering a ~3.4x speedup on x and ~1.9x on y vs
     * the default safe path. Implementations that don't take advantage
     * of this hint fall back to the safe do_grad_* path.
     *
     * IMPORTANT: do NOT use the _last variants for intermediate
     * derivatives in mixed-2nd-order chains. The next operator in the
     * chain reads padding cells that the _last path leaves untouched.
     *
     * (grad_z does not have a _last variant: the project convention is
     * "z is always last in mixed chains", so do_grad_z already
     * unconditionally skips the y-padding output cells.)
     */
    virtual void do_grad_x_last(double *const du, const double *const u,
                                const double dx, const unsigned int *sz,
                                const unsigned int bflag) {
        do_grad_x(du, u, dx, sz, bflag);
    }
    virtual void do_grad_y_last(double *const du, const double *const u,
                                const double dx, const unsigned int *sz,
                                const unsigned int bflag) {
        do_grad_y(du, u, dx, sz, bflag);
    }

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
     * @brief Get the padding width (half-stencil reach in cells) for this
     * derivative. Callers that need to inflate a bounding box by the
     * deriv's reach (e.g. BSSN's per-block puncture-fallback policy) read
     * this to know how far the stencil reads outside the query region.
     */
    inline unsigned int getPaddingWidth() const { return p_pw; }

    /**
     * @brief Get the element order used to size the derivative.
     */
    inline unsigned int getEleOrder() const { return p_ele_order; }

    /**
     * @brief Get a string representation.
     * @return A string describing the Derivs object.
     */
    virtual std::string toString() const                   = 0;

    /**
     * @brief Pre-size the per-instance scratch workspace for the largest
     * block this Derivs will see. See the "Workspace sizing" section of the
     * `Derivs` class docstring for the full contract.
     *
     * **NOTE:** Call this once per instance (and once per `clone()`) right
     * after construction, with the max `Nx*Ny*Nz` over all blocks on this
     * rank. The matrix-based subclasses will lazily grow the workspace if
     * you skip this — but you'll pay a heap reallocation on the first
     * oversized block, and (for OMP parallel-for-over-blocks) those
     * reallocations land in the hot loop. Always cleaner to size up front.
     */
    virtual void set_maximum_block_size(size_t block_size) = 0;

    /**
     * Batch derivative computation for multiple variables.
     * Default implementation just loops; matrix-based classes override
     * to share the pre-scaled D matrix across all variables.
     */
    virtual void do_grad_x_batch(double **du_arr, const double **u_arr,
                                 unsigned int n_vars, const double dx,
                                 const unsigned int *sz,
                                 const unsigned int bflag) {
        for (unsigned int v = 0; v < n_vars; v++)
            do_grad_x(du_arr[v], u_arr[v], dx, sz, bflag);
    }
    virtual void do_grad_y_batch(double **du_arr, const double **u_arr,
                                 unsigned int n_vars, const double dx,
                                 const unsigned int *sz,
                                 const unsigned int bflag) {
        for (unsigned int v = 0; v < n_vars; v++)
            do_grad_y(du_arr[v], u_arr[v], dx, sz, bflag);
    }
    virtual void do_grad_z_batch(double **du_arr, const double **u_arr,
                                 unsigned int n_vars, const double dx,
                                 const unsigned int *sz,
                                 const unsigned int bflag) {
        for (unsigned int v = 0; v < n_vars; v++)
            do_grad_z(du_arr[v], u_arr[v], dx, sz, bflag);
    }

    /// Pre-create matrices for a specific grid dimension. Override in
    /// matrix-based classes; default is a no-op for stencil types.
    virtual void pre_create_for_size(unsigned int) {};

    // raw function pointer type for bypassing virtual dispatch on stencils.
    // explicit stencil classes override these to expose their cached function
    // pointers, letting DendroDerivatives call them directly.
    using RawStencilFn = void (*)(double *const, const double *const,
                                  const double, const unsigned int *,
                                  const unsigned int);

    virtual RawStencilFn get_raw_grad_x() const { return nullptr; }
    virtual RawStencilFn get_raw_grad_y() const { return nullptr; }
    virtual RawStencilFn get_raw_grad_z() const { return nullptr; }
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

    // cached raw function pointers for bypassing virtual dispatch.
    // extracted from the Derivs objects at construction time — nullptr
    // means the implementation doesn't expose raw stencils (e.g. matrix
    // methods) and we fall back to virtual dispatch.
    Derivs::RawStencilFn _raw_1st_grad_x = nullptr;
    Derivs::RawStencilFn _raw_1st_grad_y = nullptr;
    Derivs::RawStencilFn _raw_1st_grad_z = nullptr;
    Derivs::RawStencilFn _raw_2nd_grad_x = nullptr;
    Derivs::RawStencilFn _raw_2nd_grad_y = nullptr;
    Derivs::RawStencilFn _raw_2nd_grad_z = nullptr;

    // pull raw pointers from a Derivs object into the cache slots
    void _cache_raw_stencils() {
        if (_first_deriv) {
            _raw_1st_grad_x = _first_deriv->get_raw_grad_x();
            _raw_1st_grad_y = _first_deriv->get_raw_grad_y();
            _raw_1st_grad_z = _first_deriv->get_raw_grad_z();
        }
        if (_second_deriv) {
            _raw_2nd_grad_x = _second_deriv->get_raw_grad_x();
            _raw_2nd_grad_y = _second_deriv->get_raw_grad_y();
            _raw_2nd_grad_z = _second_deriv->get_raw_grad_z();
        }
    }

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

        _cache_raw_stencils();
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
        _cache_raw_stencils();
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
        if (_raw_1st_grad_x) _raw_1st_grad_x(du, u, dx, sz, bflag);
        else _first_deriv->do_grad_x(du, u, dx, sz, bflag);
    }
    void grad_y(double *du, const double *u, double dx, const unsigned int *sz,
                unsigned int bflag) {
        if (_raw_1st_grad_y) _raw_1st_grad_y(du, u, dx, sz, bflag);
        else _first_deriv->do_grad_y(du, u, dx, sz, bflag);
    }
    void grad_z(double *du, const double *u, double dx, const unsigned int *sz,
                unsigned int bflag) {
        if (_raw_1st_grad_z) _raw_1st_grad_z(du, u, dx, sz, bflag);
        else _first_deriv->do_grad_z(du, u, dx, sz, bflag);
    }
    void grad_xx(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        if (_raw_2nd_grad_x) _raw_2nd_grad_x(du, u, dx, sz, bflag);
        else _second_deriv->do_grad_x(du, u, dx, sz, bflag);
    }
    void grad_yy(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        if (_raw_2nd_grad_y) _raw_2nd_grad_y(du, u, dx, sz, bflag);
        else _second_deriv->do_grad_y(du, u, dx, sz, bflag);
    }
    void grad_zz(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        if (_raw_2nd_grad_z) _raw_2nd_grad_z(du, u, dx, sz, bflag);
        else _second_deriv->do_grad_z(du, u, dx, sz, bflag);
    }

    // "_last" variants: caller asserts the output will NOT be further
    // differentiated. Use only for solo 1st-derivatives or for the last
    // step of a mixed 2nd-order chain (e.g. grad_y_last when computing
    // d^2u/dxdy = grad_y(grad_x(u))). Skips writing the output's
    // padding cells, recovering a ~3.4x speedup on x and ~1.9x on y at
    // eleorder=6. See Derivs::do_grad_x_last for the full contract.
    //
    // grad_z has no _last variant because do_grad_z already
    // unconditionally skips by project convention.
    void grad_x_last(double *du, const double *u, double dx,
                     const unsigned int *sz, unsigned int bflag) {
        _first_deriv->do_grad_x_last(du, u, dx, sz, bflag);
    }
    void grad_y_last(double *du, const double *u, double dx,
                     const unsigned int *sz, unsigned int bflag) {
        _first_deriv->do_grad_y_last(du, u, dx, sz, bflag);
    }
    void grad_xx_last(double *du, const double *u, double dx,
                      const unsigned int *sz, unsigned int bflag) {
        _second_deriv->do_grad_x_last(du, u, dx, sz, bflag);
    }
    void grad_yy_last(double *du, const double *u, double dx,
                      const unsigned int *sz, unsigned int bflag) {
        _second_deriv->do_grad_y_last(du, u, dx, sz, bflag);
    }

    // deriv renaming of grad naming
    void deriv_x(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        grad_x(du, u, dx, sz, bflag);
    }
    void deriv_y(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        grad_y(du, u, dx, sz, bflag);
    }
    void deriv_z(double *du, const double *u, double dx, const unsigned int *sz,
                 unsigned int bflag) {
        grad_z(du, u, dx, sz, bflag);
    }
    void deriv_xx(double *du, const double *u, double dx,
                  const unsigned int *sz, unsigned int bflag) {
        grad_xx(du, u, dx, sz, bflag);
    }
    void deriv_yy(double *du, const double *u, double dx,
                  const unsigned int *sz, unsigned int bflag) {
        grad_yy(du, u, dx, sz, bflag);
    }
    void deriv_zz(double *du, const double *u, double dx,
                  const unsigned int *sz, unsigned int bflag) {
        grad_zz(du, u, dx, sz, bflag);
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

    /// Pre-create derivative matrices for a specific grid dimension size.
    /// Call at mesh setup to ensure no lazy allocation during timestepping.
    void pre_create_for_size(unsigned int n) {
        _first_deriv->pre_create_for_size(n);
        _second_deriv->pre_create_for_size(n);
    }

    /**
     * Compute first derivatives for multiple variables at once.
     * Keeps the D matrix hot in L1 cache across all variables,
     * significantly reducing cache pressure vs calling grad_x/y/z
     * individually for each variable.
     *
     * @param du_arr Array of n_vars output pointers
     * @param u_arr  Array of n_vars input pointers
     * @param n_vars Number of variables to process
     */
    /**
     * Batch derivative computation for multiple variables at once.
     * For matrix-based methods, this shares the pre-scaled D matrix
     * and kernel dispatch across all variables.
     */
    void grad_x_batch(double **du_arr, const double **u_arr,
                      unsigned int n_vars, double dx,
                      const unsigned int *sz, unsigned int bflag) {
        _first_deriv->do_grad_x_batch(du_arr, u_arr, n_vars, dx, sz, bflag);
    }
    void grad_y_batch(double **du_arr, const double **u_arr,
                      unsigned int n_vars, double dy,
                      const unsigned int *sz, unsigned int bflag) {
        _first_deriv->do_grad_y_batch(du_arr, u_arr, n_vars, dy, sz, bflag);
    }
    void grad_z_batch(double **du_arr, const double **u_arr,
                      unsigned int n_vars, double dz,
                      const unsigned int *sz, unsigned int bflag) {
        _first_deriv->do_grad_z_batch(du_arr, u_arr, n_vars, dz, sz, bflag);
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
