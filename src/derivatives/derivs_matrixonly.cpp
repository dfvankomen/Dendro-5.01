#include "derivatives/derivs_matrixonly.h"

#include "derivatives.h"
#include "derivatives/derivs_utils.h"
#include "lapac.h"
#include "mathUtils.h"
#include "refel.h"

namespace dendroderivs {

// TODO: implement do_grad_x, y, and z

template <unsigned int DerivOrder>
void MatrixCompactDerivs<DerivOrder>::init() {
    // P and Q should be initialized, now we have to invert Q and then multiply
    // with P

    // copy Q to Qinv
    // double *Qinv;
    std::vector<double> Pinv = P_;
    // std::cout << "P before inverse:" << std::endl;
    // printArray_2D_transpose(P_.data(), p_n, p_n);

    // normalize the matrix to be between -1 and 1 for more accuracy
    // double norm = normL2(Pinv.data(), p_n * p_n);
    // for (size_t i = 0; i < p_n * p_n; i++) {
    //     Pinv[i] /= norm;
    // }
    // std::cout << "NORM IS: " << norm << std::endl;

    std::cout << "Q in general:" << std::endl;
    printArray_2D_transpose(Q_.data(), p_n, p_n);

    std::cout << "P_norm before inverse:" << std::endl;
    printArray_2D_transpose(Pinv.data(), p_n, p_n);
    // lapack::inverse(Qinv.data(), p_n);
    lapack::iterative_inverse(P_.data(), Pinv.data(), p_n);
    std::cout << "P_inv_norm after inverse:" << std::endl;
    printArray_2D_transpose(Pinv.data(), p_n, p_n);

    // for (size_t i = 0; i < p_n * p_n; i++) {
    //     Pinv[i] *= norm;
    // }

    // std::cout << "P_norm after inverse:" << std::endl;
    // printArray_2D_transpose(Pinv.data(), p_n, p_n);

    // then do matrix multiplication to get D_

    lapack::square_matrix_multiplication(Pinv.data(), Q_.data(), D_.data(),
                                         p_n);

    std::cout << "D_ after multiplication:" << std::endl;
    printArray_2D_transpose(D_.data(), p_n, p_n);
}

template void MatrixCompactDerivs<1>::init();
template void MatrixCompactDerivs<2>::init();

}  // namespace dendroderivs
