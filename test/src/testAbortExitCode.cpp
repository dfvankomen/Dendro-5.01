// Oracle: test/expect_abort.cmake needs a nonzero exit plus the mode's message.
// ghost (2 ranks): End with no Begin. interp: rank 0 reads a point outside its
// element; rank 1, if any, waits. Catches: MPI_Abort(comm, 0); exit(0) in the
// singleton interp run (mpirun itself exits 1 on an unfinalized exit(0)).
// Usage: [mpirun -np 2] testAbortExitCode [ghost|interp] (ghost: 2 ranks)

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "daUtils.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    const std::string mode = (argc > 1) ? argv[1] : "ghost";
    const bool sizeOk      = npes == 2 || (mode == "interp" && npes == 1);
    if (!sizeOk || (mode != "ghost" && mode != "interp")) {
        if (!rank)
            std::fprintf(
                stderr,
                "usage: [mpirun -np 2] %s [ghost|interp] (ghost needs 2 "
                "ranks)\n",
                argv[0]);
        MPI_Abort(comm, 2);
    }

    m_uiMaxDepth              = 6;
    const unsigned int eOrder = 4;
    _InitializeHcurve(m_uiDim);

    const double d_min = -10.0, d_max = 10.0;
    std::function<double(double, double, double)> fr =
        [d_min, d_max](double x, double y, double z) {
            auto to = [d_min, d_max](double g) {
                return (g / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            };
            const double xx = to(x), yy = to(y), zz = to(z);
            return std::exp(-((xx + 2) * (xx + 2) + yy * yy + zz * zz)) +
                   std::exp(-((xx - 2) * (xx - 2) + yy * yy + zz * zz));
        };
    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(fr, tmpNodes, m_uiMaxDepth, 1e-3, eOrder, comm);
    ot::Mesh* mesh = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FDM,
        DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);

    const bool onAll = mesh->isActive() && mesh->getMPICommSize() == npes;
    if (onAll) {
        std::vector<double> vec(mesh->getDegOfFreedom(), 0.0);
        if (mode == "ghost") {
            mesh->readFromGhostEnd(vec.data(), 1);
        } else if (!mesh->getMPIRank()) {
            double x[3] = {5.0, 0.5, 0.5};  // outside [0,1]^3
            ot::da::lagrangeInterpElementToCoord(
                mesh, vec.data(), x, Point(0, 0, 0), Point(1, 1, 1),
                mesh->getElementLocalBegin(), eOrder);
        }
        MPI_Barrier(mesh->getMPICommunicator());
        if (!mesh->getMPIRank())
            std::printf("testAbortExitCode %s: library did NOT abort\n",
                        mode.c_str());
    } else if (!rank) {
        std::printf("testAbortExitCode: setup failed, mesh not on %d ranks\n",
                    npes);
    }
    delete mesh;
    MPI_Finalize();
    return onAll ? 0 : 1;
}
