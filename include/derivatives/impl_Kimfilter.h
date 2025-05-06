#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

// TODO: remove this include
#include "refel.h"
namespace dendroderivs {
    MatrixDiagonalEntries* KimFilterO6(const std::vector<double>&);
    } // namespace dendroderivs