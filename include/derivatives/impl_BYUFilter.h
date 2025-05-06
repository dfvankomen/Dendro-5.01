#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

// TODO: remove this include
#include "refel.h"
namespace dendroderivs {
    MatrixDiagonalEntries* BYUT6Filter(const std::vector<double>&);
    } // namespace dendroderivs