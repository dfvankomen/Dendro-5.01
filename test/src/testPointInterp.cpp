// Oracle: ot::da::interpolateToCoords reproduces a tensor-degree-eleOrder
// polynomial at random points, element corners and the domain's upper faces;
// exactly one rank reports each point. Catches: wrong element lookup or
// reference coordinates, points lost or doubled at partition boundaries or
// dropped on an upper domain face. Usage: mpirun -np <n> testPointInterp

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
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
    m_uiMaxDepth = 6;
    _InitializeHcurve(m_uiDim);
    const double lo = -2.0, hi = 2.0;  // non-unit domain: checks the mapping

    int failures = 0;
    for (unsigned int p : {4u, 6u}) {
        auto f = [p](double x, double y, double z, unsigned int v) {
            const double s = 0.5;  // keeps |x| <= 1 so x^p stays O(1)
            x *= s, y *= s, z *= s;
            return (1.0 + 0.2 * (v + 1) * x + std::pow(x, p)) *
                   (1.0 - 0.5 * y + std::pow(y, p)) *
                   (1.5 + z - 0.3 * std::pow(z, p));
        };
        std::function<void(double, double, double, double*)> fv =
            [&f](double x, double y, double z, double* v) {
                v[0] = f(x, y, z, 0);
                v[1] = f(x, y, z, 1);
            };
        std::function<double(double, double, double)> fr = [](double x,
                                                              double y,
                                                              double z) {
            const double s  = 1.0 / (1u << m_uiMaxDepth);
            const double dx = x * s - 0.3, dy = y * s - 0.6, dz = z * s - 0.5;
            return std::exp(-80.0 * (dx * dx + dy * dy + dz * dz));
        };
        std::vector<ot::TreeNode> tmp;
        function2Octree(fr, tmp, m_uiMaxDepth, 1e-3, p, comm);
        ot::Mesh* mesh =
            ot::createMesh(tmp.data(), tmp.size(), p, comm, 1, ot::SM_TYPE::FDM,
                           DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
        mesh->setDomainBounds(Point(lo, lo, lo), Point(hi, hi, hi));
        const double G      = (double)(1u << m_uiMaxDepth);
        const Point grid[2] = {Point(0, 0, 0), Point(G, G, G)};
        const Point dom[2]  = {Point(lo, lo, lo), Point(hi, hi, hi)};
        auto toDom = [&](unsigned int g) { return lo + g * (hi - lo) / G; };

        // the same points on every rank
        std::vector<double> pts;
        std::mt19937 gen(7);
        std::uniform_real_distribution<double> u(lo, hi);
        for (int i = 0; i < 300; i++)
            for (int d = 0; d < 3; d++) pts.push_back(u(gen));
        // corners of every 5th element, collected on all ranks
        std::vector<unsigned int> corners;
        if (mesh->isActive()) {
            const auto& e = mesh->getAllElements();
            for (unsigned int i = mesh->getElementLocalBegin();
                 i < mesh->getElementLocalEnd(); i += 5) {
                const unsigned int h = 1u << (m_uiMaxDepth - e[i].getLevel());
                corners.insert(
                    corners.end(),
                    {e[i].minX(), e[i].minY(), e[i].minZ(), e[i].minX() + h,
                     e[i].minY() + h, e[i].minZ() + h});
            }
        }
        int nc = corners.size();
        std::vector<int> cnt(npes), off(npes, 0);
        MPI_Allgather(&nc, 1, MPI_INT, cnt.data(), 1, MPI_INT, comm);
        for (int q = 1; q < npes; q++) off[q] = off[q - 1] + cnt[q - 1];
        std::vector<unsigned int> allCorners(off[npes - 1] + cnt[npes - 1]);
        MPI_Allgatherv(corners.data(), nc, MPI_UNSIGNED, allCorners.data(),
                       cnt.data(), off.data(), MPI_UNSIGNED, comm);
        for (unsigned int g : allCorners) pts.push_back(toDom(g));
        // the domain's upper faces: each face alone, each upper edge, the
        // corner; then the x face against the lower faces
        const double m = lo + 0.37 * (hi - lo);
        for (double x : {m, hi})
            for (double y : {m, hi})
                for (double z : {m, hi})
                    if (x == hi || y == hi || z == hi)
                        pts.insert(pts.end(), {x, y, z});
        for (double a : {lo, 0.5 * (lo + hi), hi})
            for (double b : {lo, hi}) pts.insert(pts.end(), {hi, a, b});
        const unsigned int n = pts.size() / 3;

        double* in           = mesh->createCGVector<double>(fv, 2);
        std::vector<int> found(n, 0), found_g(n, 0);
        double err = 0.0, fmax = 0.0;
        if (mesh->isActive()) {
            mesh->readFromGhostBegin(in, 2);
            mesh->readFromGhostEnd(in, 2);
            for (unsigned int v = 0; v < 2; v++) {
                std::vector<double> out(n, 0.0);
                std::vector<unsigned int> valid;
                ot::da::interpolateToCoords(
                    mesh, in + v * mesh->getDegOfFreedom(), pts.data(),
                    pts.size(), grid, dom, out.data(), valid);
                for (unsigned int i : valid) {
                    if (v == 0) found[i]++;
                    const double ex =
                        f(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2], v);
                    err  = std::max(err, std::fabs(out[i] - ex));
                    fmax = std::max(fmax, std::fabs(ex));
                }
            }
        }
        MPI_Allreduce(found.data(), found_g.data(), n, MPI_INT, MPI_SUM, comm);
        double err_g = 0.0, fmax_g = 0.0;
        MPI_Allreduce(&err, &err_g, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&fmax, &fmax_g, 1, MPI_DOUBLE, MPI_MAX, comm);
        unsigned int notOnce = 0, faceMiss = 0, faceN = 0;
        for (unsigned int i = 0; i < n; i++) {
            const bool face = pts[3 * i] == hi || pts[3 * i + 1] == hi ||
                              pts[3 * i + 2] == hi;
            faceN += face;
            faceMiss += face && found_g[i] != 1;
            notOnce += found_g[i] != 1;
        }
        // relative to max|f|; degree-p polynomials are exact, so a roundoff
        // bound
        const double rel = err_g / fmax_g;
        const bool ok    = rel < 1e-12 && notOnce == 0;
        if (!rank)
            std::printf(
                "  np %d eOrder %u: %u points (%u on an upper face, %u "
                "of those lost), %u not found exactly once, "
                "relative max error %.3e: %s\n",
                npes, p, n, faceN, faceMiss, notOnce, rel, ok ? "ok" : "FAIL");
        if (!ok) failures++;
        delete[] in;
        delete mesh;
    }
    if (!rank)
        std::printf("testPointInterp np %d: %s\n", npes,
                    failures ? "FAIL" : "PASS");
    MPI_Finalize();
    return failures ? 1 : 0;
}
