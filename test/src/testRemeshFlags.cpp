// Oracle: isReMeshUnzip wavelets are interpolation residuals (order p inside,
// p/2 on boundary octants): on a uniform level-3 mesh degree p/2 splits none,
// degree p no interior element; a narrow Gaussian splits its peak element and
// none 0.45+ away. Catches: an inverted tolerance test, padding read on
// boundary octants. Usage: mpirun -np <n> testRemeshFlags

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
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
    m_uiMaxDepth = 8;
    _InitializeHcurve(m_uiDim);
    const double c[3] = {0.43, 0.52, 0.47};  // peak, off the element grid

    int failures      = 0;
    for (unsigned int p : {4u, 6u}) {  // element orders in use; p/2 = 2, 3
        // uniform level 3 (8^3 elements, 6^3 interior); domain [0,1]^3
        std::vector<ot::TreeNode> tmp;
        if (!rank) {
            const unsigned int lev = 3, n = 1u << lev,
                               h = 1u << (m_uiMaxDepth - lev);
            for (unsigned int k = 0; k < n; k++)
                for (unsigned int j = 0; j < n; j++)
                    for (unsigned int i = 0; i < n; i++)
                        tmp.push_back(ot::TreeNode(i * h, j * h, k * h, lev,
                                                   m_uiDim, m_uiMaxDepth));
        }
        ot::Mesh* mesh =
            ot::createMesh(tmp.data(), tmp.size(), p, comm, 1, ot::SM_TYPE::FDM,
                           DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
        mesh->setDomainBounds(Point(0, 0, 0), Point(1, 1, 1));
        int act = mesh->isActive(), nAct = 0;
        MPI_Allreduce(&act, &nAct, 1, MPI_INT, MPI_SUM, comm);
        if (!rank) std::printf("active ranks: %d of %d\n", nAct, npes);

        // tensor degree q; >= 1 on [0,1]^3, so relative and absolute agree
        auto poly = [](unsigned int q) {
            return std::function<void(double, double, double, double*)>(
                [q](double x, double y, double z, double* v) {
                    v[0] = (1.0 + x + std::pow(x, q)) * (2.0 - std::pow(y, q)) *
                           (1.0 + 0.5 * z + std::pow(z, q));
                });
        };
        std::function<void(double, double, double, double*)> gaussF =
            [&c](double x, double y, double z, double* v) {
                const double dx = x - c[0], dy = y - c[1], dz = z - c[2];
                v[0] = std::exp(-400.0 * (dx * dx + dy * dy + dz * dz));
            };
        // wavelets are relative to max(1, |u|); roundoff is ~1e-14
        std::function<double(double, double, double, double*)> tol =
            [](double, double, double, double*) { return 1e-8; };

        const unsigned int varId = 0;
        // field 0: degree p/2, 1: degree p, 2: Gaussian
        int nSplit[3] = {0, 0, 0}, interiorSplit = 0, farSplit = 0,
            peakSplit = 0, nInterior = 0;  // 6^3 expected: guards the check
        if (mesh->isActive())
            for (unsigned int i = mesh->getElementLocalBegin();
                 i < mesh->getElementLocalEnd(); i++)
                nInterior += !mesh->isBoundaryOctant(i);
        for (int field = 0; field < 3; field++) {
            double* u  = mesh->createCGVector<double>(field == 0   ? poly(p / 2)
                                                      : field == 1 ? poly(p)
                                                                   : gaussF,
                                                      1);
            double* uz = mesh->createUnZippedVector<double>(1);
            if (mesh->isActive()) {
                mesh->readFromGhostBegin(u, 1);
                mesh->readFromGhostEnd(u, 1);
                mesh->unzip(u, uz, 1);
            }
            // collective over the global comm: every rank calls it
            const double* uzs[1] = {uz};
            mesh->isReMeshUnzip(uzs, &varId, 1, tol);
            if (mesh->isActive()) {
                const auto& e  = mesh->getAllElements();
                const double G = (double)(1u << m_uiMaxDepth);
                for (unsigned int i = mesh->getElementLocalBegin();
                     i < mesh->getElementLocalEnd(); i++) {
                    if ((e[i].getFlag() >> NUM_LEVEL_BITS) != OCT_SPLIT)
                        continue;
                    nSplit[field]++;
                    if (field == 1 && !mesh->isBoundaryOctant(i))
                        interiorSplit++;
                    if (field != 2) continue;
                    const double h  = (1u << (m_uiMaxDepth - e[i].getLevel()));
                    const double dx = (e[i].minX() + h / 2) / G - c[0],
                                 dy = (e[i].minY() + h / 2) / G - c[1],
                                 dz = (e[i].minZ() + h / 2) / G - c[2];
                    // exp(-400 r^2) < 1e-8 beyond r = 0.22; 0.45 from the
                    // center leaves an element (half-diagonal 0.11) of margin
                    if (std::sqrt(dx * dx + dy * dy + dz * dz) >= 0.45)
                        farSplit++;
                    if (e[i].minX() <= c[0] * G && c[0] * G < e[i].minX() + h &&
                        e[i].minY() <= c[1] * G && c[1] * G < e[i].minY() + h &&
                        e[i].minZ() <= c[2] * G && c[2] * G < e[i].minZ() + h)
                        peakSplit++;
                }
            }
            delete[] u;
            delete[] uz;
        }
        int loc[7] = {nSplit[0], nSplit[1], interiorSplit, nSplit[2],
                      farSplit,  peakSplit, nInterior},
            glob[7];
        MPI_Allreduce(loc, glob, 7, MPI_INT, MPI_SUM, comm);
        const bool ok = glob[0] == 0 && glob[2] == 0 && glob[3] > 0 &&
                        glob[4] == 0 && glob[5] == 1 && glob[6] == 216;
        if (!rank)
            std::printf(
                "  np %d eOrder %u: degree %u %d split; degree %u %d "
                "split (%d interior of %d); Gaussian %d split (%d far, peak "
                "element %d): %s\n",
                npes, p, p / 2, glob[0], p, glob[1], glob[2], glob[6], glob[3],
                glob[4], glob[5], ok ? "ok" : "FAIL");
        if (!ok) failures++;
        delete mesh;
    }
    if (!rank)
        std::printf("testRemeshFlags np %d: %s\n", npes,
                    failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return failures ? 1 : 0;
}
