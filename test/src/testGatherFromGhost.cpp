// Oracle: gatherFromGhostEnd (dof 2) gives each shared owned node, per
// variable, the values its ghosting ranks (per the send map) hold, each rank's
// copies offset by rank and variable; End without a Begin aborts. Catches:
// wrong copy/variable layout (needs nodes with 2+ copies, np >= 3), context 0.
// Usage: mpirun -np <2..4> testGatherFromGhost values|unknown

#include <mpi.h>

#include <algorithm>
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
    if ((mode != "values" && mode != "unknown") || npes < 2 || npes > 4) {
        if (!rank)
            std::fprintf(stderr, "usage: mpirun -np <2..4> %s values|unknown\n",
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
    // grain 40: up to 4 active ranks; all must be active (checked below)
    ot::Mesh* mesh =
        ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1,
                       ot::SM_TYPE::FDM, 40, 0.1, DENDRO_DEFAULT_SF_K);

    int local_fail = 0;
    // exact compare: values < 2^22, offsets are multiples of 1/64
    auto off       = [](int r, unsigned int var) {
        return 0.25 * (r + 1) + 0.015625 * var;
    };
    if (mesh->isActive() && mesh->getMPICommSize() > 1) {
        const std::size_t n = mesh->getDegOfFreedom();
        const int ar        = mesh->getMPIRank();
        std::vector<double> vec(n * dof, -1.0);
        for (unsigned int var = 0; var < dof; var++)
            for (std::size_t i = mesh->getNodeLocalBegin();
                 i < mesh->getNodeLocalEnd(); i++)
                vec[var * n + i] = 1000.0 * (var + 1) + (double)ar * 1e6 + i;
        mesh->readFromGhostBegin(vec.data(), dof);
        mesh->readFromGhostEnd(vec.data(), dof);
        const std::vector<double> owner = vec;
        // each rank's ghost copies differ from the owner's by off(rank, var)
        for (unsigned int var = 0; var < dof; var++)
            for (std::size_t i = 0; i < n; i++)
                if (i < mesh->getNodeLocalBegin() ||
                    i >= mesh->getNodeLocalEnd())
                    vec[var * n + i] += off(ar, var);

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
            // which ranks ghost each owned node, from the send scatter map
            std::vector<std::vector<int>> ghosters(n);
            const auto& procs = mesh->getSendProcList();
            const auto& cnt   = mesh->getNodalSendCounts();
            const auto& offs  = mesh->getNodalSendOffsets();
            const auto& sm    = mesh->getSendNodeSM();
            for (unsigned int p : procs)
                for (unsigned int k = offs[p]; k < offs[p] + cnt[p]; k++)
                    ghosters[sm[k]].push_back(p);
            std::size_t shared = 0, multi = 0, bad = 0, vecBad = 0;
            for (std::size_t k = mesh->getNodeLocalBegin();
                 k < mesh->getNodeLocalEnd(); k++) {
                const std::size_t c = ghosters[k].size();
                const bool hasV     = k < gatherV.size() && !gatherV[k].empty();
                if (c == 0) {
                    if (hasV) bad++;
                    for (unsigned int var = 0; var < dof; var++)
                        if (vec[var * n + k] != owner[var * n + k]) vecBad++;
                    continue;
                }
                shared++;
                if (c >= 2) multi++;
                if (!hasV || gatherV[k].size() != c * dof) {
                    bad++;
                    continue;
                }
                for (unsigned int var = 0; var < dof; var++) {
                    std::vector<double> got(gatherV[k].begin() + var * c,
                                            gatherV[k].begin() + (var + 1) * c),
                        want;
                    for (int p : ghosters[k])
                        want.push_back(owner[var * n + k] + off(p, var));
                    std::sort(got.begin(), got.end());
                    std::sort(want.begin(), want.end());
                    if (got != want) bad++;
                    // NOTE: pins End's overwrite of the owner entry (last
                    // ghosting rank in receive order wins); relax if it goes
                    if (vec[var * n + k] !=
                        owner[var * n + k] + off(ghosters[k].back(), var))
                        vecBad++;
                }
            }
            std::printf(
                "  [rank %d] shared owned nodes %zu (%zu with 2+ "
                "copies), wrong gathers %zu, unexpected vec %zu\n",
                ar, shared, multi, bad, vecBad);
            if (bad || vecBad) local_fail = 1;
            std::size_t tot[2] = {shared, multi}, tot_g[2] = {0, 0};
            MPI_Allreduce(tot, tot_g, 2, MPI_UNSIGNED_LONG, MPI_SUM,
                          mesh->getMPICommunicator());
            // at 3+ ranks some node must have 2+ copies, or the layout is
            // untested
            if (tot_g[0] == 0 || (npes > 2 && tot_g[1] == 0)) local_fail = 1;
        }
    }
    if (!mesh->isActive() || mesh->getMPICommSize() != npes) {
        if (!rank)
            std::printf(
                "testGatherFromGhost: mesh not active on all %d ranks\n", npes);
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
