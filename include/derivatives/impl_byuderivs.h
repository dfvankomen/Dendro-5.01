#pragma once

#include "derivatives/derivs_banded.h"
#include "derivatives/derivs_matrixonly.h"

namespace dendroderivs {

std::vector<double> inline clean_coeffs(const std::vector<double>& coeffs_in,
                                        unsigned int max_coeffs) {
    std::vector<double> coeffs_out(max_coeffs, 0.0);

    // std::cout << "Applying coefficients: ";
    // as soon as one of these breaks, we exit, no need to check sizes
    for (unsigned int i = 0; i < max_coeffs && i < coeffs_in.size(); i++) {
        coeffs_out[i] = coeffs_in[i];
        // std::cout << coeffs_in[i] << " ";
    }
    // std::cout << std::endl;

    return coeffs_out;
}

void inline check_end_of_boundaries(std::vector<std::vector<double>>& coeff_in,
                                    const double threshold = 1e-10) {
    // we should only chekc the DIAGs and remove any values that are extremely
    // close to or equal to zero

    for (auto& vec : coeff_in) {
        // check the last value, if it's "bad" pop it back, otherwise it should
        // end
        while (!vec.empty() && std::abs(vec.back()) < threshold) {
            vec.pop_back();
        }
    }
}
MatrixDiagonalEntries* BYUDerivsT64R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R2DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R2DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R1DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R42DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT4R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R42DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT6R2DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R2DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R2DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsT8R3DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R3DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R32DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsSecondOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP8R4DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

MatrixDiagonalEntries* BYUDerivsP6R32DiagonalsFirstOrder(
    const std::vector<double>& D_coeffs);

// ---- A4 ----
MatrixDiagonalEntries* createA4_1_Diagonals();
MatrixDiagonalEntries* createA4_2_Diagonals();
MatrixDiagonalEntries* createA4_3_Diagonals();
MatrixDiagonalEntries* createA4_4_Diagonals();
MatrixDiagonalEntries* createA4_5_Diagonals();
MatrixDiagonalEntries* createA4_6_Diagonals();
MatrixDiagonalEntries* createA4_7_Diagonals();
MatrixDiagonalEntries* createA4_8_Diagonals();
MatrixDiagonalEntries* createA4_9_Diagonals();
MatrixDiagonalEntries* createA4_10_Diagonals();
MatrixDiagonalEntries* createA4_11_Diagonals();
MatrixDiagonalEntries* createA4_12_Diagonals();
MatrixDiagonalEntries* createA4_13_Diagonals();
MatrixDiagonalEntries* createA4_14_Diagonals();
MatrixDiagonalEntries* createA4_15_Diagonals();
MatrixDiagonalEntries* createA4_16_Diagonals();
MatrixDiagonalEntries* createA4_17_Diagonals();
MatrixDiagonalEntries* createA4_18_Diagonals();
MatrixDiagonalEntries* createA4_19_Diagonals();
MatrixDiagonalEntries* createA4_20_Diagonals();
// ---- B4 ----
MatrixDiagonalEntries* createB4_1_Diagonals();
MatrixDiagonalEntries* createB4_2_Diagonals();

MatrixDiagonalEntries* createC4_1_Diagonals();
MatrixDiagonalEntries* createC4_2_Diagonals();
MatrixDiagonalEntries* createC4_3_Diagonals();
MatrixDiagonalEntries* createC4_4_Diagonals();
MatrixDiagonalEntries* createC4_5_Diagonals();

MatrixDiagonalEntries* createA6_1_Diagonals();
MatrixDiagonalEntries* createA6_2_Diagonals();
MatrixDiagonalEntries* createA6_3_Diagonals();
MatrixDiagonalEntries* createA6_4_Diagonals();
MatrixDiagonalEntries* createA6_5_Diagonals();
MatrixDiagonalEntries* createA6_6_Diagonals();
MatrixDiagonalEntries* createA6_7_Diagonals();
MatrixDiagonalEntries* createA6_8_Diagonals();

//Second Derivatives
MatrixDiagonalEntries* create2B4_1_Diagonals();

// ---- 2B6 ----
MatrixDiagonalEntries* create2B6_1_Diagonals();
MatrixDiagonalEntries* create2B6_2_Diagonals();
MatrixDiagonalEntries* create2B6_3_Diagonals();
MatrixDiagonalEntries* create2B6_4_Diagonals();
MatrixDiagonalEntries* create2B6_5_Diagonals();
MatrixDiagonalEntries* create2B6_6_Diagonals();
MatrixDiagonalEntries* create2B6_7_Diagonals();
MatrixDiagonalEntries* create2B6_8_Diagonals();
MatrixDiagonalEntries* create2B6_9_Diagonals();
// ---- 2C4 ----
MatrixDiagonalEntries* create2C4_1_Diagonals();
MatrixDiagonalEntries* create2C4_2_Diagonals();
MatrixDiagonalEntries* create2C4_3_Diagonals();
MatrixDiagonalEntries* create2C4_4_Diagonals();
MatrixDiagonalEntries* create2C4_5_Diagonals();
MatrixDiagonalEntries* create2C4_6_Diagonals();
MatrixDiagonalEntries* create2C4_7_Diagonals();
MatrixDiagonalEntries* create2C4_8_Diagonals();
MatrixDiagonalEntries* create2C4_9_Diagonals();
MatrixDiagonalEntries* create2C4_10_Diagonals();

// ---- 2C6 ----
MatrixDiagonalEntries* create2C6_1_Diagonals();
MatrixDiagonalEntries* create2C6_2_Diagonals();
MatrixDiagonalEntries* create2C6_3_Diagonals();
MatrixDiagonalEntries* create2C6_4_Diagonals();
MatrixDiagonalEntries* create2C6_5_Diagonals();
MatrixDiagonalEntries* create2C6_6_Diagonals();
MatrixDiagonalEntries* create2C6_7_Diagonals();

// ---- 2A6 ----
MatrixDiagonalEntries* create2A6_1_Diagonals();
MatrixDiagonalEntries* create2A6_2_Diagonals();
MatrixDiagonalEntries* create2A6_3_Diagonals();
MatrixDiagonalEntries* create2A6_4_Diagonals();
MatrixDiagonalEntries* create2A6_5_Diagonals();
MatrixDiagonalEntries* create2A6_6_Diagonals();
MatrixDiagonalEntries* create2A6_7_Diagonals();

// ---- BYU A6 second-derivative generated operators ----
MatrixDiagonalEntries* createBYU_A6_2ND_R060_OP1_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R060_OP2_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R060_OP3_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R065_OP1_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R065_OP2_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R065_OP3_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP1_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP2_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R070_OP3_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R075_OP1_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R075_OP2_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R075_OP3_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R080_OP1_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R080_OP2_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R080_OP3_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R085_OP1_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R085_OP2_Diagonals(
    const std::vector<double>& D_coeffs);
MatrixDiagonalEntries* createBYU_A6_2ND_R085_OP3_Diagonals(
    const std::vector<double>& D_coeffs);

}  // namespace dendroderivs
