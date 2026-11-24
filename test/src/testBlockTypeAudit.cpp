// Oracle: no UNZIP_INDEPENDENT block reads a ghost node (ghost CG nodes set to
// NaN, one unzip), and some blocks are independent. Catches: a classifier that
// flags a ghost-reading block independent, which DENDRO_UNZIP_OVERLAP unzips
// while the exchange is in flight.
// Usage: testBlockTypeAudit [maxDepth=6] [waveletTol=1e-3] [grain=8]

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"
#include "parUtils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    _InitializeHcurve(m_uiDim);
    m_uiMaxDepth             = (argc > 1) ? std::atoi(argv[1]) : 6;
    const double wavelet_tol = (argc > 2) ? std::atof(argv[2]) : 1e-3;
    const unsigned int grain = (argc > 3) ? std::atoi(argv[3]) : 8;

    DendroIntL bad_total     = 0;
    for (unsigned int eOrder : {4u, 6u}) {
        const double scale = 1.0 / (double)(1u << m_uiMaxDepth);
        std::function<double(double, double, double)> refine_fn =
            [scale](double x, double y, double z) {
                const double dx = x * scale - 0.5, dy = y * scale - 0.5,
                             dz = z * scale - 0.5;
                return std::exp(-120.0 * (dx * dx + dy * dy + dz * dz));
            };

        std::vector<ot::TreeNode> tmpNodes;
        function2Octree(refine_fn, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder,
                        MPI_COMM_WORLD);
        ot::Mesh* mesh = ot::createMesh(
            tmpNodes.data(), tmpNodes.size(), eOrder, MPI_COMM_WORLD, 1,
            ot::SM_TYPE::FDM, grain, 0.1, DENDRO_DEFAULT_SF_K);
        const double ext = (double)(1u << m_uiMaxDepth);
        mesh->setDomainBounds(Point(0.0, 0.0, 0.0), Point(ext, ext, ext));

        // collective on the active comm only; inactive ranks return 0
        DendroIntL counts[2]   = {0, 0};  // independent, read ghost data
        counts[1]              = mesh->auditBlockTypeIndependence(&counts[0]);
        DendroIntL counts_g[2] = {0, 0};
        par::Mpi_Allreduce(counts, counts_g, 2, MPI_MAX, MPI_COMM_WORLD);
        if (!rank)
            std::printf(
                "  np %d eOrder %u: %lld independent, %lld read ghosts\n", npes,
                eOrder, (long long)counts_g[0], (long long)counts_g[1]);
        bad_total += counts_g[1];
        if (counts_g[0] == 0) bad_total++;  // nothing audited
        delete mesh;
    }

    if (!rank)
        std::printf("testBlockTypeAudit np %d grain %u: %lld failure(s): %s\n",
                    npes, grain, (long long)bad_total,
                    bad_total == 0 ? "PASS" : "FAIL");
    MPI_Finalize();
    return bad_total == 0 ? 0 : 1;
}
