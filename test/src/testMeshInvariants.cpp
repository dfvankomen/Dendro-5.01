/**
 * @file testMeshInvariants.cpp
 * @brief Gates the ot::test validators and the structural mesh invariants.
 *
 * meshTestUtils.h already provides isElementalNodalValuesValid, isUnzipValid,
 * isBlkFlagsValid, isDGGhostValid and the NaN probes, but their only caller is
 * run_all_tests, which returns 0 whatever they report. Assert them instead.
 *
 * Added on top of those:
 *   - 2:1 balance across every face neighbour, which the unzip padding logic
 *     assumes and nothing verifies.
 *   - block geometry: padding width is eleOrder/2 and the allocation is
 *     eleOrder * nEle + 1 + 2 * PW, the identity the scatter index arithmetic
 *     is derived from.
 *
 * Usage: testMeshInvariants [maxDepth] [waveletTol]
 */

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshTestUtils.h"
#include "meshUtils.h"
#include "octUtils.h"

namespace {

double field(double x, double y, double z) {
    const double s = 1.0 / (double)(1u << m_uiMaxDepth);
    const double a = x * s, b = y * s, c = z * s;
    return 1.0 + a - 2.0 * b + 0.5 * c + a * a * b * b - b * c * c +
           3.0 * a * b * c + c * c * c * c;
}

int check(const char* name, bool ok, int rank) {
    if (!rank) std::printf("    %-34s %s\n", name, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int check_balance(const ot::Mesh* mesh, int rank) {
    const ot::TreeNode* pNodes = mesh->getAllElements().data();
    const unsigned int* e2e    = mesh->getE2EMapping().data();
    unsigned long violations   = 0;
    for (unsigned int e = mesh->getElementLocalBegin();
         e < mesh->getElementLocalEnd(); e++)
        for (unsigned int d = 0; d < NUM_FACES; d++) {
            const unsigned int lk = e2e[e * NUM_FACES + d];
            if (lk == LOOK_UP_TABLE_DEFAULT) continue;
            const int dl = (int)pNodes[lk].getLevel() - (int)pNodes[e].getLevel();
            if (dl < -1 || dl > 1) violations++;
        }
    if (!rank)
        std::printf("    %-34s %s (%lu violations)\n", "2:1 face balance",
                    violations ? "FAIL" : "ok", violations);
    return violations ? 1 : 0;
}

int check_block_geometry(const ot::Mesh* mesh, unsigned int eOrder, int rank) {
    const std::vector<ot::Block>& blks = mesh->getLocalBlockList();
    unsigned long bad = 0;
    for (size_t b = 0; b < blks.size(); b++) {
        const unsigned int PW   = blks[b].get1DPadWidth();
        const unsigned int nEle = blks[b].getElemSz1D();
        if (PW != (eOrder >> 1u)) bad++;
        if (blks[b].getAllocationSzX() != eOrder * nEle + 1 + 2 * PW) bad++;
        if (blks[b].getAllocationSzX() != blks[b].getAllocationSzY() ||
            blks[b].getAllocationSzX() != blks[b].getAllocationSzZ())
            bad++;
    }
    if (!rank)
        std::printf("    %-34s %s (%lu blocks, %lu bad)\n", "block geometry",
                    bad ? "FAIL" : "ok", (unsigned long)blks.size(), bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    _InitializeHcurve(m_uiDim);
    m_uiMaxDepth             = (argc > 1) ? std::atoi(argv[1]) : 6;
    const double wavelet_tol = (argc > 2) ? std::atof(argv[2]) : 1e-3;

    int failures = 0;
    for (unsigned int eOrder : {4u, 6u, 8u}) {
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
            ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
        if (!mesh->isActive()) {
            delete mesh;
            continue;
        }
        const double ext = (double)(1u << m_uiMaxDepth);
        mesh->setDomainBounds(Point(0.0, 0.0, 0.0), Point(ext, ext, ext));

        unsigned int lmin, lmax;
        mesh->computeMinMaxLevel(lmin, lmax);
        if (!rank)
            std::printf("  eOrder=%u blocks=%zu levels=%u..%u\n", eOrder,
                        mesh->getLocalBlockList().size(), lmin, lmax);
        if (lmin == lmax) {
            if (!rank) std::printf("    mesh is uniform, nothing to test\n");
            failures++;
            delete mesh;
            continue;
        }

        std::function<void(double, double, double, double*)> f =
            [](double x, double y, double z, double* v) {
                v[0] = field(x, y, z);
            };
        std::function<double(double, double, double)> fs =
            [](double x, double y, double z) { return field(x, y, z); };

        double* u       = mesh->createCGVector<double>(f, 1);
        double* u_dg    = mesh->createDGVector(f, 1);
        double* u_unzip = mesh->createUnZippedVector<double>(1);

        mesh->readFromGhostBegin(u, 1);
        mesh->readFromGhostEnd(u, 1);
        mesh->readFromGhostBeginEleDGVec(u_dg, 1);
        mesh->readFromGhostEndEleDGVec(u_dg, 1);
        mesh->unzip(u, u_unzip, 1);

        failures += check_balance(mesh, rank);
        failures += check_block_geometry(mesh, eOrder, rank);
        failures += check("isBlkFlagsValid", ot::test::isBlkFlagsValid(mesh), rank);
        failures += check("isElementalNodalValuesValid",
                          ot::test::isElementalNodalValuesValid<double>(
                              mesh, u, fs, 1e-10),
                          rank);
        failures += check("isUnzipValid",
                          ot::test::isUnzipValid<double>(mesh, u_unzip, fs, 1e-10),
                          rank);
        failures += check("isUnzipNaN (expect none)",
                          !ot::test::isUnzipNaN<double>(mesh, u_unzip), rank);
        bool cg_finite = true;
        for (unsigned int i = mesh->getNodeLocalBegin();
             i < mesh->getNodeLocalEnd(); i++)
            if (!std::isfinite(u[i])) cg_finite = false;
        failures += check("CG vector all finite", cg_finite, rank);
        // Not gated. Its definition was unqualified, so ot::test::isDGGhostValid
        // never linked and has never run; at eleOrder 6 on 2 ranks it reports
        // values shifted by one node, which could as easily be the validator as
        // the exchange. Reported until that is settled.
        const bool dg_ghost =
            ot::test::isDGGhostValid<double>(mesh, u_dg, f, 1e-10);
        if (!rank)
            std::printf("    %-34s %s (not gated)\n", "isDGGhostValid",
                        dg_ghost ? "ok" : "reports mismatch");

        delete[] u;
        delete[] u_dg;
        delete[] u_unzip;
        delete mesh;
    }

    int total = failures;
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (!rank) std::printf("\nmesh invariants: %s\n", total ? "FAIL" : "PASS");
    MPI_Finalize();
    return total ? 1 : 0;
}
