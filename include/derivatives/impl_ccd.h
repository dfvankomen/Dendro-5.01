#pragma once

#include "derivatives/derivs_ccd.h"

/**
 * @file impl_ccd.h
 * @brief Coefficient tables for the CCD (combined compact difference) schemes.
 *
 * One function per scheme returning its coupled coefficients; the same function
 * serves both the first- and second-order registries, since a CCD scheme's
 * entries describe the whole coupled system and the Order template parameter
 * picks which half is sliced out. See derivs_ccd.h for the formulation.
 *
 * Generated schemes (emitted by scripts/ccd_operators.py) declare their
 * prototypes here and are #include'd into src/derivatives/impl_ccd.cpp, in the
 * same three-step pattern the BYU families use. Generated files are NOT
 * standalone translation units -- do not add them to CMake.
 */

namespace dendroderivs {

// classical 3-point CCD, 6th order in both f' and f''
CCDDiagonalEntries* createCCD6Diagonals();

}  // namespace dendroderivs
