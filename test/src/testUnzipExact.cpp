// Bit-exactness gate: Mesh::unzip_scatter_batch() vs unzip_scatter_batch_ref(),
// memcmp over every variable's whole unzipped vector.
// UNZIP_GATE_SABOTAGE=1/2/3 injects a defect (1 ulp / zeroed word / swapped vars).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <ctime>
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
                         "eleOrder [n_vars=6] [time_reps=0]\n",
                         argv[0]);
        MPI_Abort(comm, 1);
    }

    m_uiMaxDepth          = std::atoi(argv[1]);
    double wavelet_tol    = std::atof(argv[2]);
    double partition_tol  = std::atof(argv[3]);
    unsigned int eOrder   = (unsigned int)std::atoi(argv[4]);
    unsigned int n_vars   = (argc > 5) ? (unsigned int)std::atoi(argv[5]) : 6u;
    unsigned int n_reps   = (argc > 6) ? (unsigned int)std::atoi(argv[6]) : 0u;

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

    int local_fail = 0;

    // run on the original and a remeshed mesh (checks plan invalidation)
    auto check_mesh = [&](ot::Mesh* mesh, const char* tag) {
        unsigned int lmin, lmax;
        mesh->computeMinMaxLevel(lmin, lmax);
        if (!mesh->isActive()) return;
        const size_t cgSz = mesh->getDegOfFreedom();
        const size_t unSz = mesh->getDegOfFreedomUnZip();

        std::vector<double> in((size_t)n_vars * cgSz);
        std::vector<double> out_ref((size_t)n_vars * unSz),
            out_new((size_t)n_vars * unSz);
        std::vector<const double*> ins(n_vars);
        std::vector<double*> outs_ref(n_vars), outs_new(n_vars);
        for (unsigned int v = 0; v < n_vars; v++) {
            ins[v]      = in.data() + (size_t)v * cgSz;
            outs_ref[v] = out_ref.data() + (size_t)v * unSz;
            outs_new[v] = out_new.data() + (size_t)v * unSz;
        }

        if (!rank)
            std::printf(
                "testUnzipExact[%s]: eOrder=%u lev=[%u,%u] localElem=%u "
                "totalElem=%u blocks=%zu cgSz=%zu unSz=%zu npes=%d vars=%u\n",
                tag, eOrder, lmin, lmax,
                mesh->getElementLocalEnd() - mesh->getElementLocalBegin(),
                mesh->getAllElements().size() > 0
                    ? (unsigned int)mesh->getAllElements().size()
                    : 0u,
                mesh->getLocalBlockList().size(), cgSz, unSz, npes, n_vars);

        // analytic (ghost-synced), index-valued, sawtooth and random patterns
        const unsigned int n_patterns = 4;
        for (unsigned int p = 0; p < n_patterns; p++) {
            for (unsigned int v = 0; v < n_vars; v++) {
                double* x = in.data() + (size_t)v * cgSz;
                if (p == 0) {
                    double* u = mesh->createCGVector<double>(func, 1);
                    for (size_t i = 0; i < cgSz; i++) x[i] = u[i] * (1.0 + v);
                    mesh->destroyVector(u);
                    mesh->readFromGhostBegin(x, 1);
                    mesh->readFromGhostEnd(x, 1);
                } else {
                    std::srand(1234u + 97u * p + v + 7919u * rank);
                    for (size_t i = 0; i < cgSz; i++) {
                        if (p == 1)
                            x[i] = (double)i + 0.25 * v;
                        else if (p == 2)
                            x[i] = -(double)((i * (v + 3)) % 9973);
                        else
                            x[i] = (double)std::rand() / (double)RAND_MAX - 0.5;
                    }
                }
            }

            ot::g_lpt_block_order = (p % 2) == 1;  // cover both block orders

            // poison both outputs so untouched cells must agree too
            std::memset(out_ref.data(), 0xA5, out_ref.size() * sizeof(double));
            std::memset(out_new.data(), 0xA5, out_new.size() * sizeof(double));

            mesh->unzip_scatter_batch_ref(ins.data(), outs_ref.data(), n_vars);
            mesh->unzip_scatter_batch(ins.data(), outs_new.data(), n_vars);

#if defined(UNZIP_GATE_SABOTAGE)
            if (unSz > 0) {
#if UNZIP_GATE_SABOTAGE == 1
                double& d = out_new[(size_t)(n_vars - 1) * unSz + unSz / 2];
                uint64_t w;
                std::memcpy(&w, &d, sizeof(w));
                w ^= 1u;
                std::memcpy(&d, &w, sizeof(w));
#elif UNZIP_GATE_SABOTAGE == 2
                out_new[unSz - 1] = 0.0;
#elif UNZIP_GATE_SABOTAGE == 3
                if (n_vars > 1)
                    std::swap_ranges(out_new.begin(), out_new.begin() + unSz,
                                     out_new.begin() + unSz);
#endif
            }
#endif

            const size_t n_words = out_ref.size();
            if (std::memcmp(out_ref.data(), out_new.data(),
                            n_words * sizeof(double)) != 0) {
                local_fail = 1;
                size_t nd = 0, first = (size_t)-1;
                for (size_t i = 0; i < n_words; i++) {
                    if (std::memcmp(&out_ref[i], &out_new[i], sizeof(double))) {
                        if (first == (size_t)-1) first = i;
                        nd++;
                    }
                }
                std::printf(
                    "  [rank %d] pattern %u: MISMATCH differing words=%zu/%zu "
                    "first=%zu (var %zu) ref=%.17g new=%.17g\n",
                    rank, p, nd, n_words, first, first / unSz, out_ref[first],
                    out_new[first]);
            } else if (!rank) {
                std::printf("  pattern %u: bit-exact (%zu doubles)%s\n", p,
                            n_words, ot::g_lpt_block_order ? " lpt" : "");
            }
        }

        // timing: ref/new interleaved per rep, max over ranks per call
        if (n_reps > 0) {
            std::vector<double> t_ref(n_reps), t_new(n_reps), c_ref(n_reps),
                c_new(n_reps);
            auto cpu_now = []() {
                timespec ts;
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
                return ts.tv_sec + 1e-9 * ts.tv_nsec;
            };
            double c_last = 0.0;
            auto time_call = [&](bool use_ref) {
                MPI_Barrier(comm);
                const double c0 = cpu_now();
                const double t0 = MPI_Wtime();
                if (use_ref)
                    mesh->unzip_scatter_batch_ref(ins.data(), outs_ref.data(),
                                                  n_vars);
                else
                    mesh->unzip_scatter_batch(ins.data(), outs_new.data(),
                                              n_vars);
                double dt = MPI_Wtime() - t0, dmax;
                c_last    = cpu_now() - c0;
                MPI_Allreduce(&dt, &dmax, 1, MPI_DOUBLE, MPI_MAX, comm);
                return dmax;
            };
            for (unsigned int r = 0; r < n_reps; r++) {
                t_ref[r] = time_call(true);
                c_ref[r] = c_last;
                t_new[r] = time_call(false);
                c_new[r] = c_last;
            }
            for (auto* t : {&t_ref, &t_new, &c_ref, &c_new})
                std::sort(t->begin(), t->end());
            if (!rank)
                std::printf(
                    "  timing[%s] %u reps: ref med=%.3f min=%.3f ms | new "
                    "med=%.3f min=%.3f ms | speedup med=%.3fx min=%.3fx\n",
                    tag, n_reps, 1e3 * t_ref[n_reps / 2], 1e3 * t_ref[0],
                    1e3 * t_new[n_reps / 2], 1e3 * t_new[0],
                    t_ref[n_reps / 2] / t_new[n_reps / 2], t_ref[0] / t_new[0]);
            if (!rank)
                std::printf("  cpu[%s] rank0 process cpu med: ref=%.3f new=%.3f ms "
                            "ratio=%.3fx\n",
                            tag, 1e3 * c_ref[n_reps / 2], 1e3 * c_new[n_reps / 2],
                            c_ref[n_reps / 2] / c_new[n_reps / 2]);
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
#if defined(UNZIP_GATE_SABOTAGE)
        std::printf(
            "UNZIP_GATE_SABOTAGE=%d active (gate is expected to FAIL)\n",
            UNZIP_GATE_SABOTAGE);
#endif
        std::printf("testUnzipExact: %s\n", global_fail ? "FAIL" : "PASS");
    }

    delete mesh;
    MPI_Finalize();
    return global_fail ? 1 : 0;
}
