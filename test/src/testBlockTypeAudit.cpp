/**
 * @file testBlockTypeAudit.cpp
 * @brief Every block flagged UNZIP_INDEPENDENT must read no ghost data.
 *
 * The DENDRO_UNZIP_OVERLAP path unzips UNZIP_INDEPENDENT blocks while the
 * ghost exchange is in flight, so a block that is flagged independent but
 * reads a ghost node gets stale data. Mesh::auditBlockTypeIndependence poisons
 * every ghost CG node with NaN, unzips once, and counts the independent blocks
 * that catch a NaN; that count must be 0.
 *
 * Usage: testBlockTypeAudit [maxDepth=6] [waveletTol=1e-3] [grain=8]
 * A small grain keeps many ranks active, so most blocks touch a rank boundary.
 */

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

        char site[32];
        std::snprintf(site, sizeof(site), "np%d eOrder%u", npes, eOrder);
        // collective on the active comm only; inactive ranks return 0
        DendroIntL bad   = mesh->auditBlockTypeIndependence(site);
        DendroIntL bad_g = 0;
        par::Mpi_Allreduce(&bad, &bad_g, 1, MPI_MAX, MPI_COMM_WORLD);
        bad_total += bad_g;
        delete mesh;
    }

    if (!rank)
        std::printf(
            "testBlockTypeAudit np %d grain %u: %lld misclassified "
            "independent block(s): %s\n",
            npes, grain, (long long)bad_total,
            bad_total == 0 ? "PASS" : "FAIL");
    MPI_Finalize();
    return bad_total == 0 ? 0 : 1;
}
