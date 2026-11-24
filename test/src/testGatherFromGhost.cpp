// Oracle: gatherFromGhostEnd with dof 2 on 2 ranks returns, per owned node and
// variable, one copy per sharing rank equal to the owner's value; End on a
// vector without a Begin aborts. Catches: the per-variable stride taken as
// count*dof with the count bumped per variable, and the context defaulting to
// 0. Usage: mpirun -np 2 testGatherFromGhost values|unknown

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    const std::string mode = (argc > 1) ? argv[1] : "";
    if ((mode != "values" && mode != "unknown") || npes != 2) {
        if (!rank)
            std::fprintf(stderr, "usage: mpirun -np 2 %s values|unknown\n",
                         argv[0]);
        MPI_Abort(comm, 2);
    }

    m_uiMaxDepth              = 6;
    const unsigned int eOrder = 4, dof = 2;
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

    int local_fail = 0;
    if (mesh->isActive() && mesh->getMPICommSize() == 2) {
        const std::size_t n = mesh->getDegOfFreedom();
        std::vector<double> vec(n * dof, -1.0);
        for (unsigned int var = 0; var < dof; var++)
            for (std::size_t i = mesh->getNodeLocalBegin();
                 i < mesh->getNodeLocalEnd(); i++)
                vec[var * n + i] = 1000.0 * (var + 1) + (double)rank * 1e6 + i;
        // ghosts now hold their owners' values
        mesh->readFromGhostBegin(vec.data(), dof);
        mesh->readFromGhostEnd(vec.data(), dof);
        const std::vector<double> owner = vec;

        std::vector<std::vector<double>> gatherV;
        mesh->gatherFromGhostBegin(vec.data(), dof);
        if (mode == "unknown") {
            std::vector<double> other(n * dof, 0.0);
            mesh->gatherFromGhostEnd(other.data(), gatherV, dof);
            if (!rank)
                std::printf(
                    "testGatherFromGhost: End on an unknown vector NOT "
                    "detected\n");
            local_fail = 1;
        } else {
            mesh->gatherFromGhostEnd(vec.data(), gatherV, dof);
            std::size_t shared = 0, bad = 0;
            for (std::size_t k = 0; k < gatherV.size(); k++) {
                if (gatherV[k].empty()) continue;
                shared++;
                if (gatherV[k].size() % dof) {
                    bad++;
                    continue;
                }
                const std::size_t c = gatherV[k].size() / dof;
                for (unsigned int var = 0; var < dof; var++)
                    for (std::size_t j = 0; j < c; j++)
                        if (gatherV[k][var * c + j] != owner[var * n + k])
                            bad++;
            }
            std::printf(
                "  [rank %d] shared owned nodes %zu, wrong copies %zu\n", rank,
                shared, bad);
            if (bad) local_fail = 1;
            std::size_t total = 0;
            MPI_Allreduce(&shared, &total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
            if (total == 0) local_fail = 1;
        }
    } else {
        if (!rank)
            std::printf(
                "testGatherFromGhost: mesh not split over 2 active ranks\n");
        local_fail = 1;
    }

    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, comm);
    if (!rank)
        std::printf("testGatherFromGhost %s: %s\n", mode.c_str(),
                    global_fail ? "FAIL" : "PASS");
    delete mesh;
    MPI_Finalize();
    return global_fail ? 1 : 0;
}
