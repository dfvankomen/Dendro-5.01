// Oracle: with DENDRO_GHOST_RECV_CHECK on, a correct exchange passes and a
// receive completed with words unwritten aborts. Catches: an unwritten ghost.
//   clean  legacy and ctx paths: every ghost word written, paths bit-identical
//   hole   a PMPI MPI_Waitall wrapper restores the sentinel; must abort
// Usage: mpirun -np 2 testGhostRecvCheck clean|hole

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"

namespace {
// the check's double sentinel (ghostRecvCheck.h SENTINEL_F64), spelled out here
constexpr std::uint64_t kSentinel = 0x7ff4dead5eed0001ull;

MPI_Request* g_hole_reqs = nullptr;  // fire on the Waitall over this array
double* g_hole_ptr       = nullptr;
std::size_t g_hole_len   = 0;

bool is_sentinel(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b == kSentinel;
}
}  // namespace

// emulates the transport: the real wait completes, then part of the range
// reads as never written
extern "C" int MPI_Waitall(int count, MPI_Request reqs[], MPI_Status stats[]) {
    const int rc = PMPI_Waitall(count, reqs, stats);
    if (g_hole_reqs != nullptr && reqs == g_hole_reqs) {
        for (std::size_t i = 0; i < g_hole_len; i++)
            std::memcpy(g_hole_ptr + i, &kSentinel, sizeof(kSentinel));
        g_hole_reqs = nullptr;
    }
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    const std::string mode = (argc > 1) ? argv[1] : "";
    if ((mode != "clean" && mode != "hole") || npes != 2) {
        if (!rank) std::fprintf(stderr, "usage: mpirun -np 2 %s clean|hole\n", argv[0]);
        MPI_Abort(comm, 2);
    }
    // before any exchange: the library reads the switch once
    setenv("DENDRO_GHOST_RECV_CHECK", "1", 1);

    m_uiMaxDepth              = 6;
    const unsigned int eOrder = 6, dof = 4;
    _InitializeHcurve(m_uiDim);

    const double d_min = -10.0, d_max = 10.0;
    std::function<double(double, double, double)> fr = [d_min, d_max](double x, double y,
                                                                      double z) {
        auto to = [d_min, d_max](double g) {
            return (g / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
        };
        const double xx = to(x), yy = to(y), zz = to(z);
        return std::exp(-((xx + 2) * (xx + 2) + yy * yy + zz * zz)) +
               std::exp(-((xx - 2) * (xx - 2) + yy * yy + zz * zz));
    };
    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(fr, tmpNodes, m_uiMaxDepth, 1e-3, eOrder, comm);
    ot::Mesh* mesh = ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1,
                                    ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ, 0.1,
                                    DENDRO_DEFAULT_SF_K);

    int local_fail = 0;
    if (mesh->isActive() && mesh->getMPICommSize() == 2) {
        const std::size_t cgSz                   = mesh->getDegOfFreedom();
        const std::vector<unsigned int>& rProc   = mesh->getRecvProcList();
        const std::vector<unsigned int>& rCnt    = mesh->getNodalRecvCounts();
        const std::vector<unsigned int>& rOff    = mesh->getNodalRecvOffsets();
        const std::vector<unsigned int>& recvSM  = mesh->getRecvNodeSM();

        // owned nodes carry index values, ghosts start at -1
        auto fill = [&](double* v) {
            for (unsigned int var = 0; var < dof; var++)
                for (std::size_t i = 0; i < cgSz; i++)
                    v[var * cgSz + i] = (i >= mesh->getNodeLocalBegin() &&
                                         i < mesh->getNodeLocalEnd())
                                            ? 1000.0 * (var + 1) + (double)i
                                            : -1.0;
        };
        // ghost words still -1 (unwritten) or carrying the sentinel
        auto ghost_faults = [&](const double* v, std::size_t* sentinels) {
            std::size_t unset = 0;
            *sentinels        = 0;
            for (unsigned int p : rProc)
                for (unsigned int k = rOff[p]; k < rOff[p] + rCnt[p]; k++)
                    for (unsigned int var = 0; var < dof; var++) {
                        const double g = v[var * cgSz + recvSM[k]];
                        if (is_sentinel(g)) (*sentinels)++;
                        else if (g == -1.0) unset++;
                    }
            return unset;
        };

        std::vector<ot::AsyncExchangeContex> ctx;
        ot::alloc_mpi_ctx<double>(mesh, ctx, (int)dof, 1);
        std::vector<double> vCtx(cgSz * dof);
        fill(vCtx.data());

        if (mode == "hole" && rank == 0) {
            // part of variable 1 from rank 1 (a contiguous run)
            const unsigned int p = 1;
            g_hole_reqs          = ctx[0].m_recv_req.data();
            g_hole_ptr = (double*)ctx[0].getRecvBuffer() + dof * rOff[p] + rCnt[p];
            g_hole_len = rCnt[p] / 2;
        }
        mesh->readFromGhostBegin(ctx[0], vCtx.data(), dof);
        mesh->readFromGhostEnd(ctx[0], vCtx.data(), dof);

        std::size_t sent = 0;
        const std::size_t unset = ghost_faults(vCtx.data(), &sent);
        if (mode == "hole") {
            if (rank == 0) {
                std::printf("testGhostRecvCheck: hole NOT detected: %zu ghost words carry "
                            "the sentinel after the exchange\n", sent);
                local_fail = 1;
            }
        } else {
            std::vector<double> vLeg(cgSz * dof);
            fill(vLeg.data());
            mesh->readFromGhostBegin(vLeg.data(), dof);
            mesh->readFromGhostEnd(vLeg.data(), dof);
            std::size_t sentLeg = 0;
            const std::size_t unsetLeg = ghost_faults(vLeg.data(), &sentLeg);
            const bool same =
                std::memcmp(vCtx.data(), vLeg.data(), cgSz * dof * sizeof(double)) == 0;
            std::size_t nghost = 0;
            for (unsigned int p : rProc) nghost += (std::size_t)rCnt[p] * dof;
            std::printf("  [rank %d] ghost words %zu: ctx unset=%zu sentinel=%zu, legacy "
                        "unset=%zu sentinel=%zu, paths bit-identical=%d\n",
                        rank, nghost, unset, sent, unsetLeg, sentLeg, (int)same);
            if (nghost == 0 || unset || sent || unsetLeg || sentLeg || !same) local_fail = 1;
        }
        ot::dealloc_mpi_ctx<double>(mesh, ctx, (int)dof, 1);
    } else {
        if (!rank) std::printf("testGhostRecvCheck: mesh not split over 2 active ranks\n");
        local_fail = 1;
    }

    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, comm);
    if (!rank) std::printf("testGhostRecvCheck %s: %s\n", mode.c_str(), global_fail ? "FAIL" : "PASS");
    std::fflush(stdout);

    delete mesh;
    MPI_Finalize();
    return global_fail ? 1 : 0;
}
