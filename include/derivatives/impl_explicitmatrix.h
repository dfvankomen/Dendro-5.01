#pragma once

#include "derivatives/derivs_compact.h"
#include "derivatives/derivs_matrixonly.h"

namespace dendroderivs {

MatrixDiagonalEntries* createE4DiagonalsFirstOrder();
MatrixDiagonalEntries* createE4DiagonalsSecondOrder();
MatrixDiagonalEntries* createE6DiagonalsFirstOrder();
MatrixDiagonalEntries* createE6DiagonalsSecondOrder();
MatrixDiagonalEntries* createE8DiagonalsFirstOrder();
MatrixDiagonalEntries* createE8DiagonalsSecondOrder();

}  // namespace dendroderivs
