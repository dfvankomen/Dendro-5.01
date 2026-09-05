// Bit-exactness gate: Mesh::zip() vs Mesh::zip_ref(), memcmp over the CG vector.
// ZIP_GATE_SABOTAGE=1/2/3 injects a defect (bad value / skipped block / wrong owner).

#include <cmath>
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
#include "mpi.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (argc < 5) {
        if (!rank)
            std::fprintf(stderr,
                         "Usage: %s maxDepth wavelet_tol partition_tol "
                         "eleOrder [n_fields=8]\n",
                         argv[0]);
        MPI_Abort(comm, 1);
    }

    m_uiMaxDepth          = std::atoi(argv[1]);
    double wavelet_tol    = std::atof(argv[2]);
    double partition_tol  = std::atof(argv[3]);
    unsigned int eOrder   = (unsigned int)std::atoi(argv[4]);
    unsigned int n_fields = (argc > 5) ? (unsigned int)std::atoi(argv[5]) : 8u;

    _InitializeHcurve(m_uiDim);

    const double d_min = -10.0, d_max = 10.0;
    Point pt_min(d_min, d_min, d_min), pt_max(d_max, d_max, d_max);

    // two offset gaussians -> genuinely multi-level refinement
    std::function<void(double, double, double, double*)> func =
        [](double x, double y, double z, double* var) {
            const double ca[] = {-2.0, 0.0, 0.0};
            const double cb[] = {2.0, 0.0, 0.0};
            const double rra  = (x - ca[0]) * (x - ca[0]) +
                               (y - ca[1]) * (y - ca[1]) +
                               (z - ca[2]) * (z - ca[2]);
            const double rrb = (x - cb[0]) * (x - cb[0]) +
                               (y - cb[1]) * (y - cb[1]) +
                               (z - cb[2]) * (z - cb[2]);
            var[0] = std::exp(-rra) + std::exp(-rrb);
        };
    std::function<double(double, double, double)> fr =
        [func, d_min, d_max](double x, double y, double z) {
            const double xx =
                (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            const double yy =
                (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            const double zz =
                (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double v;
            func(xx, yy, zz, &v);
            return v;
        };

    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(fr, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder, comm);
    ot::Mesh* mesh = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FDM,
        DENDRO_DEFAULT_GRAIN_SZ, partition_tol, DENDRO_DEFAULT_SF_K);
    mesh->setDomainBounds(pt_min, pt_max);

    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);

    int local_fail = 0;

    // run on the original and a remeshed mesh (checks plan invalidation)
    auto check_mesh = [&](ot::Mesh* mesh, const char* tag) {
    if (mesh->isActive()) {
        const size_t cgSz = mesh->getDegOfFreedom();
        const size_t unSz = mesh->getDegOfFreedomUnZip();

        double* u         = mesh->createCGVector<double>(func, 1);
        double* u_unzip   = mesh->createUnZippedVector<double>(1);
        mesh->readFromGhostBegin(u, 1);
        mesh->readFromGhostEnd(u, 1);
        mesh->unzip(u, u_unzip, 1);

        std::vector<double> out_ref(cgSz), out_new(cgSz);

        if (!rank)
            std::printf(
                "testZipExact[%s]: eOrder=%u lev=[%u,%u] localElem=%u blocks=%zu "
                "cgSz=%zu unSz=%zu npes=%d fields=%u\n",
                tag, eOrder, lmin, lmax,
                mesh->getElementLocalEnd() - mesh->getElementLocalBegin(),
                mesh->getLocalBlockList().size(), cgSz, unSz, npes, n_fields);

        // analytic, index-valued and random patterns
        for (unsigned int f = 0; f < n_fields; f++) {
            if (f > 0) {
                // regenerate a distinct pattern in the unzip buffer directly
                std::srand(1234u + f);
                for (size_t i = 0; i < unSz; i++) {
                    if (f == 1)
                        u_unzip[i] = (double)i;            // index-valued
                    else if (f == 2)
                        u_unzip[i] = -(double)(i % 9973);  // sawtooth
                    else
                        u_unzip[i] =
                            (double)std::rand() / (double)RAND_MAX - 0.5;
                }
            }

            // poison both outputs so only zip's writes are compared
            std::memset(out_ref.data(), 0xA5, cgSz * sizeof(double));
            std::memset(out_new.data(), 0xA5, cgSz * sizeof(double));

            mesh->zip_ref(u_unzip, out_ref.data());
            mesh->zip(u_unzip, out_new.data());

#if defined(ZIP_GATE_SABOTAGE)
            // deliberate defect injection to prove the gate can fail
            if (cgSz > 0) {
#if ZIP_GATE_SABOTAGE == 1
                out_new[cgSz / 2] += 1e-16 * out_new[cgSz / 2] + 1e-300;
#elif ZIP_GATE_SABOTAGE == 2
                out_new[cgSz - 1] = 0.0;
#elif ZIP_GATE_SABOTAGE == 3
                for (size_t i = 0; i < cgSz; i += 4096) out_new[i] = -out_new[i];
#endif
            }
#endif

            const int bad =
                (std::memcmp(out_ref.data(), out_new.data(),
                             cgSz * sizeof(double)) != 0)
                    ? 1
                    : 0;
            if (bad) {
                local_fail = 1;
                size_t nd = 0, first = (size_t)-1;
                for (size_t i = 0; i < cgSz; i++) {
                    const uint64_t a =
                        *reinterpret_cast<const uint64_t*>(&out_ref[i]);
                    const uint64_t b =
                        *reinterpret_cast<const uint64_t*>(&out_new[i]);
                    if (a != b) {
                        if (first == (size_t)-1) first = i;
                        nd++;
                    }
                }
                std::printf(
                    "  [rank %d] field %u: MISMATCH  differing words=%zu/%zu "
                    "first=%zu ref=%.17g new=%.17g\n",
                    rank, f, nd, cgSz, first, out_ref[first], out_new[first]);
            } else if (!rank) {
                std::printf("  field %u: bit-exact (%zu doubles, memcmp==0)\n",
                            f, cgSz);
            }
        }

        mesh->destroyVector(u);
        mesh->destroyVector(u_unzip);
    }
    };

    check_mesh(mesh, "original");

    // ---- remesh round ----
    {
        std::vector<unsigned int> refine_flag;
        refine_flag.reserve(mesh->getNumLocalMeshElements());
        const ot::TreeNode* pN = mesh->getAllElements().data();
        for (unsigned int ele = mesh->getElementLocalBegin();
             ele < mesh->getElementLocalEnd(); ele++) {
            if (((ele + NUM_CHILDREN - 1) < mesh->getElementLocalEnd()) &&
                (pN[ele].getParent() ==
                 pN[ele + NUM_CHILDREN - 1].getParent())) {
                for (unsigned int c = 0; c < NUM_CHILDREN; c++)
                    refine_flag.push_back(OCT_COARSE);
                ele += (NUM_CHILDREN - 1);
            } else if ((ele % 10) == 0)
                refine_flag.push_back(OCT_SPLIT);
            else
                refine_flag.push_back(OCT_NO_CHANGE);
        }
        mesh->setMeshRefinementFlags(refine_flag);
        ot::Mesh* newMesh = mesh->ReMesh();
        newMesh->setDomainBounds(pt_min, pt_max);
        if (!rank) std::printf("-- after ReMesh --\n");
        check_mesh(newMesh, "remeshed");
        delete newMesh;
    }

    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, comm);
    if (!rank) {
#if defined(ZIP_GATE_SABOTAGE)
        std::printf("ZIP_GATE_SABOTAGE=%d active (gate is expected to FAIL)\n",
                    ZIP_GATE_SABOTAGE);
#endif
        std::printf("testZipExact: %s\n", global_fail ? "FAIL" : "PASS");
    }

    delete mesh;
    MPI_Finalize();
    return global_fail ? 1 : 0;
}
