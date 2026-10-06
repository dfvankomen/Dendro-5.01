/**
 * @file tpid_params.h
 * @brief Parameter declarations for the TwoPunctures initial data solver.
 *
 * These are declared as extern -- the solver project must define them
 * (typically in parameters.cpp, read from the TOML file).
 */

#ifndef DENDRO_GR_TPID_PARAMS_H
#define DENDRO_GR_TPID_PARAMS_H

#include <string>

namespace TPID {

extern std::string FILE_PREFIX;

static const double TCP_epsilon = 1e-06;
static const int swap_xz = 0;
static const int use_sources = 0;
static const int rescale_sources = 0;
static const int use_external_initial_guess = 0;
static const int do_residuum_debug_output = 1;
static const int do_initial_debug_output = 1;
static const int multiply_old_lapse = 0;
static const double TCP_Tiny = 1.0e-15;
static const double TCP_Extend_Radius = 0.0;
static const int Newton_maxit = 15;

extern double par_b;
extern double par_m_plus;
extern double par_m_minus;
extern double par_q_plus;
extern double par_q_minus;
extern double par_p_plus;
extern double par_p_minus;
extern double target_M_plus;
extern double target_M_minus;
extern double par_alpha0;
extern double par_P_plus[3];
extern double par_P_minus[3];
extern double par_S_plus[3];
extern double par_S_minus[3];
extern double center_offset[3];
extern unsigned int give_bare_mass;
extern int initial_lapse;
extern unsigned int grid_setup_method;
extern int solve_momentum_constraint;
extern unsigned int verbose;
extern double adm_tol;
extern double Newton_tol;
extern double initial_lapse_psi_exponent;
extern unsigned int npoints_A;
extern unsigned int npoints_B;
extern unsigned int npoints_phi;
extern unsigned int nintegration_theta;
extern unsigned int nintegration_phi;
extern bool use_Dilaton;
extern bool use_Interpolated_solution;
extern bool replace_lapse_with_sqrt_chi;
extern double par_alpha0;
extern double par_q_plus;
extern double par_q_minus;
extern double par_p_plus;
extern double par_p_minus;
extern double target_M_plus;
extern double target_M_minus;

}  // namespace TPID

#endif  // DENDRO_GR_TPID_PARAMS_H
