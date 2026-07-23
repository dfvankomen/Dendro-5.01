#pragma once

#include "derivatives/derivs_ccfd.h"

/**
 * @file impl_ccfd.h
 * @brief Coefficient tables for the CCFD (combined compact finite difference)
 * schemes.
 *
 * One function per scheme returning its coupled coefficients; the same function
 * serves both the first- and second-order registries, since a CCFD scheme's
 * entries describe the whole coupled system and the Order template parameter
 * picks which half is sliced out. See derivs_ccfd.h for the formulation.
 *
 * Generated schemes (emitted by a Mathematica notebook, or by
 * scripts/ccfd_operators.py --emit-cpp) declare their prototypes here and are
 * #include'd into src/derivatives/impl_ccfd.cpp, in the same three-step pattern
 * the BYU families use. Generated files are NOT standalone translation units --
 * do not add them to CMake.
 */

namespace dendroderivs {

// classical 3-point CCFD, 6th order in both f' and f''
CCFDDiagonalEntries* createCCFD6Diagonals();

}  // namespace dendroderivs
