/**
 * @file testProlongation.cpp
 * @brief Coordinate-free invariants for RefElement::I3D_Parent2Child.
 *
 * The 3D parent->child prolongation is assembled from the tensor kernels in
 * FEM/src/tensor.cpp, and nothing tested that assembly. Two invariants hold
 * regardless of node placement or element order:
 *
 *   1. The interpolation weights reaching each child node sum to 1. The
 *      matrices are stored for the tensor kernels, which contract over the
 *      first index, so that sum runs down a column.
 *   2. Therefore a constant parent field prolongates to the same constant, for
 *      every child.
 *
 * (2) is what a dropped or duplicated lane in the tensor kernels breaks, so it
 * fails loudly for the class of defect that reached eO=8 unnoticed. Checking
 * (1) separately localises a failure to the matrices rather than the assembly.
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <mpi.h>

#include <cmath>
#include <string>
#include <vector>

#include "refel.h"

namespace {

const std::vector<unsigned int> kOrders = {2, 4, 6, 8};

}  // namespace

TEST_CASE("1D child interpolation weights sum to one per output node") {
    for (unsigned int order : kOrders) {
        RefElement ref(3, order);
        const unsigned int nrp = order + 1;
        for (int child = 0; child < 2; ++child) {
            const double* M = child == 0 ? ref.getIMChild0() : ref.getIMChild1();
            for (unsigned int out_node = 0; out_node < nrp; ++out_node) {
                double sum = 0.0;
                for (unsigned int k = 0; k < nrp; ++k)
                    sum += M[k * nrp + out_node];
                CAPTURE(order);
                CAPTURE(child);
                CAPTURE(out_node);
                CHECK(std::fabs(sum - 1.0) < 1e-12);
            }
        }
    }
}

TEST_CASE("prolongating a constant reproduces it for every child") {
    for (unsigned int order : kOrders) {
        RefElement ref(3, order);
        const unsigned int nrp = order + 1;
        const std::size_t nPe  = (std::size_t)nrp * nrp * nrp;
        const double c         = 0.75;
        std::vector<double> in(nPe, c);

        for (unsigned int child = 0; child < 8; ++child) {
            std::vector<double> out(nPe, std::nan(""));
            ref.I3D_Parent2Child(in.data(), out.data(), child);

            int unwritten = 0, wrong = 0;
            double worst = 0.0;
            for (std::size_t i = 0; i < nPe; ++i) {
                if (std::isnan(out[i])) {
                    unwritten++;
                } else {
                    const double d = std::fabs(out[i] - c);
                    worst          = std::max(worst, d);
                    if (d > 1e-12) wrong++;
                }
            }
            CAPTURE(order);
            CAPTURE(child);
            CAPTURE(worst);
            CHECK(unwritten == 0);
            CHECK(wrong == 0);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const int res = ctx.run();
    MPI_Finalize();
    return res;
}
