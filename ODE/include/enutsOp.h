/**
 * @file etsRefEl.h
 * @author Milinda Fernando (milinda@cs.utah.edu)
 * @brief Computes the block correction operators for explicit time steppers,
 * @version 0.1
 * @date 2020-03-28
 * School of Computing, University of Utah.
 * @copyright Copyright (c) 2020
 *
 */

/**
 * Note that these correction operations are constructed from picewise linear
 * approximation of F(u), [k] = CxPx[DU] -> (1) and the B computation comes from
 * the (1) and the Taylor exapansion of the RK stages.
 *
 */

#pragma once

#include <cmath>
#include <iostream>

#include "ts.h"

namespace ts {
class ENUTSOp {
   protected:
    /**@brief: ETS time stage computation coefficients. */
    std::vector<DendroScalar> m_uiAij;

    /**@brief Diagonal op. of time scaling $P_{\Delta t}$ */
    std::vector<DendroScalar> m_uiPi;

    /**@brief Diagonal op. of time scaling $P_{\Delta t}^{-1}$ */
    std::vector<DendroScalar> m_uiInvPi;

    /**@brief: Coefficient matrix from Taylor expansion of stage vectors i.e [K]
     * = C*P DU*/
    std::vector<DendroScalar> m_uiCij;

    /**@brief: Inverse of coefficient matrix C*/
    std::vector<DendroScalar> m_uiInvCij;

    /**@brief: Taylor expansion of the DU = (\partial_t u , \partial^2_t u ...
     * \partial^{p}_t u)^T*/
    std::vector<DendroScalar> m_uiBij;

    /**@brief: workspace matrix M*/
    std::vector<DendroScalar> m_uiMij;

    /**@brief: workspace vector in */
    std::vector<DendroScalar> m_uiVin;

    /**@brief: workspace vector out */
    std::vector<DendroScalar> m_uiVout;

    /**@brief: number of stages of the scheme*/
    unsigned int m_uiNumStages;

    /**@brief: time stepper type*/
    ETSType m_uiType;

    /**@brief: recently built stage-correction operators (Ccf / Cfc) */
    enum { OP_CACHE = 8 };
    struct OpCache {
        bool valid = false, cf = false;
        unsigned int rk = 0;
        DendroScalar dt_c = 0, dt_f = 0, dt = 0;
        DendroScalar M[25];
    };
    OpCache m_uiOpCache[OP_CACHE];
    unsigned int m_uiOpCacheNext = 0;

    /**@brief: C x P x B x P^{-1} x C^{-1} of Ccf (cf) or Cfc into M (d x d) */
    void buildM(bool cf, DendroScalar* M, unsigned int rk_s, DendroScalar dt_c,
                DendroScalar dt_f, DendroScalar dt);
    const DendroScalar* cachedM(bool cf, unsigned int rk_s, DendroScalar dt_c,
                                DendroScalar dt_f, DendroScalar dt);

   public:
    /**
     * @brief Construct a new ENUTSOp object
     * @param ts_type time stepper type.
     */
    ENUTSOp(ETSType type);

    /**
     * @brief Destroy the ENUTSOp object
     *
     */
    ~ENUTSOp() {};

    /**
     * @brief Diagonal op. of time scaling $P_{\Delta t}$ and its inverse
     * operator
     * @param dt : delta t that P is defined at.
     * @param rk_s : nunmber of stages considered to be interpolated.
     */
    void Pdt(DendroScalar dt, unsigned int rk_s);

    /**
     * @brief : computes the Taylor expansion operator for DU
     * @param dt : delta t that P is defined at.
     * @param rk_s : nunmber of stages considered to be interpolated.
     */
    void Bdt(DendroScalar dt, unsigned int rk_s);

    // stage correction routines begin

    /**
     * @brief perform the finer to coarser time stage corrections.
     * @param out : corrected or interpolated coarser stages from finer stages.
     * @param in  : finer stages.
     * @param sz: size of in/out vectors
     * @param rk_s : number of rk stages needs to be interpolated.
     * @param dt_c : coarser time step size.
     * @param dt_f : finer time step size.
     * @param dt : dt for the taylor expansion of DU
     * @param dof : number of dof.
     */
    void Cfc(DendroScalar** out, const DendroScalar** in,
             const unsigned int* sz, unsigned int rk_s, DendroScalar dt_c,
             DendroScalar dt_f, DendroScalar dt, unsigned int dof = 1,
             unsigned int nav = 0);

    /**
     * @brief perform the coarser to finer time stage corrections.
     * @param out : corrected or interpolated coarser stages from finer stages.
     * @param in  : finer stages.
     * @param sz: size of in/out vectors
     * @param rk_s : number of rk stages needs to be interpolated.
     * @param dt_c : coarser time step size.
     * @param dt_f : finer time step size.
     * @param dt : dt for the taylor expansion of DU
     * @param dof : number of dof.
     */
    void Ccf(DendroScalar** out, const DendroScalar** in,
             const unsigned int* sz, unsigned int rk_s, DendroScalar dt_c,
             DendroScalar dt_f, DendroScalar dt, unsigned int dof = 1,
             unsigned int nav = 0);

    // stage correction routines end

    /**
     * @brief row rk_s of the Ccf (cf) or Cfc operator: stage rk_s in the
     * target frame = sum_k row[k] * source stage k (k < number of source
     * stages used), the per-node form of Ccf / Cfc.
     */
    void stageRow(bool cf, DendroScalar* row, unsigned int rk_s,
                  DendroScalar dt_c, DendroScalar dt_f, DendroScalar dt);

    /**
     * @brief stageRow for a nonlinear right-hand side (RK4 only): the fourth
     * Taylor variable is dt^2 f_yy(f,f) = 4 g, g = k4 - 2 k3 + k1, instead of
     * dt^3 d^4y/dt^4 (which assumes a linear right-hand side).
     * row[0..3] multiply the source stages k1..k4, row[4] multiplies g.
     * nav = 4: all source stages exist, g is folded into row[0..3] and
     * row[4] = 0. nav < 4 (same-time conversion, dt = 0): row[k] = 0 for
     * k >= nav and g has to come from the source's previous step.
     * @return true when row[4] != 0 (the caller has to supply g).
     */
    bool stageRowNL(bool cf, DendroScalar* row, unsigned int rk_s,
                    DendroScalar dt_c, DendroScalar dt_f, DendroScalar dt,
                    unsigned int nav) const;

    /**
     * @brief out[n] = sum_{k<ns} m[k] * in[k][n], n < NN (out must not
     * overlap in); the node loop of Ccf / Cfc.
     */
    void applyRow(DendroScalar* out, const DendroScalar* const* in,
                  const DendroScalar* m, unsigned int ns, unsigned int NN) const;

    /**
     * @brief blk[dst[q]] = sum_{k<ns} m[k] * in[k][src[q]], q < n; same
     * summation order as Ccf / Cfc.
     */
    void gatherRow(DendroScalar* blk, const unsigned int* dst,
                   const unsigned int* src, unsigned int n,
                   const DendroScalar* const* in, const DendroScalar* m,
                   unsigned int ns) const;

    // below routines are for coarser to finer and finer to coarser time
    // corrections, which is needed resuming LTS time stepper after intergrid
    // transfer. U(t) correction For these corrections we need all the computed
    // stages.

    /**
     * @brief Performs correction operators for Ut
     * @param out
     * @param in
     * @param sz
     * @param dt_c
     * @param dt_f
     * @param dt
     * @param dof
     */
    void coarser_finer_ut_correction(DendroScalar** out,
                                     const DendroScalar** in,
                                     const unsigned int* sz, DendroScalar dt_c,
                                     DendroScalar dt_f, DendroScalar dt,
                                     unsigned int dof = 1);

    /**
     * @brief
     * @param out
     * @param in
     * @param sz
     * @param dt_c
     * @param dt_f
     * @param dt
     * @param dof
     */
    void finer_coarser_ut_correction(DendroScalar** out,
                                     const DendroScalar** in,
                                     const unsigned int* sz, DendroScalar dt_c,
                                     DendroScalar dt_f, DendroScalar dt,
                                     unsigned int dof = 1);
};

};  // namespace ts
