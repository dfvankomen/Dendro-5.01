/**
 * @file testUnzipExact.cpp
 * @brief Whole-unzip exactness on a mesh containing 2:1 level jumps.
 *
 * Unzip is a copy at matching levels and a degree-eleOrder polynomial
 * interpolation across a level jump, so a polynomial of degree <= eleOrder in
 * each direction must come back exactly, at every sample inside the domain,
 * including the padding. Samples whose coordinates fall outside the domain are
 * legitimately unfilled and are skipped.
 *
 * This covers the assembled unzip rather than its kernels: a dropped lane, a
 * lost diagonal neighbour at a level jump, or a wrong scatter index all break
 * it. Mismatches are reported grouped by how many padding axes the sample sits
 * on, which localises a failure to interior / face / edge / corner.
 *
 * Usage: testUnzipExact [maxDepth] [waveletTol]
 */

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"
#include "octUtils.h"

namespace {

double poly(double x, double y, double z) {
    return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z + x * x * y - y * y * z +
           2.0 * x * y * z + z * z * z;
}

struct Census {
    unsigned long checked = 0, unfilled = 0, wrong = 0;
    double worst          = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    _InitializeHcurve(m_uiDim);

    m_uiMaxDepth              = (argc > 1) ? std::atoi(argv[1]) : 6;
    const double wavelet_tol  = (argc > 2) ? std::atof(argv[2]) : 1e-3;
    const double partition_tol = 0.1;
    const double SENTINEL     = -1.0e300;
    const double scale        = 1.0 / (double)(1u << m_uiMaxDepth);

    int failures = 0;
    for (unsigned int eOrder : {4u, 6u, 8u}) {
        // a bump refines the centre only, so the mesh carries 2:1 jumps
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
            ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ, partition_tol,
            DENDRO_DEFAULT_SF_K);
        if (!mesh->isActive()) {
            delete mesh;
            continue;
        }
        mesh->setDomainBounds(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0));

        std::function<void(double, double, double, double*)> f =
            [](double x, double y, double z, double* v) {
                v[0] = poly(x, y, z);
            };
        double* u       = mesh->createCGVector<double>(f, 1);
        double* u_unzip = mesh->createUnZippedVector<double>(1);
        for (DendroIntL i = 0; i < mesh->getDegOfFreedomUnZip(); i++)
            u_unzip[i] = SENTINEL;

        mesh->readFromGhostBegin(u, 1);
        mesh->readFromGhostEnd(u, 1);
        mesh->unzip(u, u_unzip, 1);

        const std::vector<ot::Block>& blks = mesh->getLocalBlockList();
        const ot::TreeNode* pNodes         = mesh->getAllElements().data();
        unsigned int lmin = m_uiMaxDepth, lmax = 0;
        Census census[4];

        for (size_t b = 0; b < blks.size(); b++) {
            const ot::TreeNode blkNode = blks[b].getBlockNode();
            const unsigned int bLev =
                pNodes[blks[b].getLocalElementBegin()].getLevel();
            lmin = std::min(lmin, bLev);
            lmax = std::max(lmax, bLev);

            const unsigned int PW = blks[b].get1DPadWidth();
            const unsigned int lx = blks[b].getAllocationSzX();
            const unsigned int ly = blks[b].getAllocationSzY();
            const unsigned int lz = blks[b].getAllocationSzZ();
            const DendroIntL off  = blks[b].getOffset();
            const double hx = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
            const double x0 = blkNode.minX() - PW * hx;
            const double y0 = blkNode.minY() - PW * hx;
            const double z0 = blkNode.minZ() - PW * hx;
            const double dmax = (double)(1u << m_uiMaxDepth);

            for (unsigned int k = 0; k < lz; k++) {
                const double zz = z0 + k * hx;
                for (unsigned int j = 0; j < ly; j++) {
                    const double yy = y0 + j * hx;
                    for (unsigned int i = 0; i < lx; i++) {
                        const double xx = x0 + i * hx;
                        if (xx < 0.0 || yy < 0.0 || zz < 0.0 || xx > dmax ||
                            yy > dmax || zz > dmax)
                            continue;

                        const int pad_axes =
                            (int)(i < PW || i >= lx - PW) +
                            (int)(j < PW || j >= ly - PW) +
                            (int)(k < PW || k >= lz - PW);
                        Census& c = census[pad_axes];
                        c.checked++;

                        const double got =
                            u_unzip[off + k * lx * ly + j * lx + i];
                        if (got == SENTINEL) {
                            c.unfilled++;
                            continue;
                        }
                        const double want =
                            poly(xx * scale, yy * scale, zz * scale);
                        const double err = std::fabs(got - want);
                        c.worst          = std::max(c.worst, err);
                        if (err > 1e-10 * (1.0 + std::fabs(want))) c.wrong++;
                    }
                }
            }
        }

        const char* names[4] = {"interior", "face", "edge", "corner"};
        unsigned long bad    = 0;
        if (!rank)
            std::printf("eOrder=%u blocks=%zu levels=%u..%u\n", eOrder,
                        blks.size(), lmin, lmax);
        for (int p = 0; p < 4; p++) {
            bad += census[p].unfilled + census[p].wrong;
            if (!rank)
                std::printf(
                    "  %-8s checked=%-9lu unfilled=%-7lu wrong=%-7lu "
                    "worst=%.3e\n",
                    names[p], census[p].checked, census[p].unfilled,
                    census[p].wrong, census[p].worst);
        }
        if (lmin == lmax) {
            if (!rank)
                std::printf("  ERROR: mesh is uniform, no 2:1 jump exercised\n");
            bad++;
        }
        if (bad) failures++;

        delete[] u;
        delete[] u_unzip;
        delete mesh;
    }

    unsigned long long total = failures;
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    if (!rank)
        std::printf("\nunzip exactness: %s\n", total ? "FAIL" : "PASS");
    MPI_Finalize();
    return total ? 1 : 0;
}
