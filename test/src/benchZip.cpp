// Interleaved A/B microbenchmark for Mesh::zip().
//
// A = Mesh::zip_ref()  (verbatim pre-optimization baseline)
// B = Mesh::zip()      (optimized)
//
// Both live in the SAME binary, so the comparison is immune to the code-layout
// noise that plagues two-build comparisons. Runs A/B/A/B... and reports the
// distribution of the per-repetition RATIO (t_A/t_B), which is far more stable
// than either absolute.
//
// Also verifies bit-exactness on every repetition, so a "fast" answer that is
// wrong cannot be reported as a speedup.

#include <algorithm>
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
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_sec() { return MPI_Wtime(); }

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
                         "eleOrder [reps=21] [calls_per_rep=24]\n",
                         argv[0]);
        MPI_Abort(comm, 1);
    }
    m_uiMaxDepth         = std::atoi(argv[1]);
    double wavelet_tol   = std::atof(argv[2]);
    double partition_tol = std::atof(argv[3]);
    unsigned int eOrder  = (unsigned int)std::atoi(argv[4]);
    unsigned int reps    = (argc > 5) ? (unsigned int)std::atoi(argv[5]) : 21u;
    unsigned int ncalls  = (argc > 6) ? (unsigned int)std::atoi(argv[6]) : 24u;

    _InitializeHcurve(m_uiDim);
    const double d_min = -10.0, d_max = 10.0;
    Point pt_min(d_min, d_min, d_min), pt_max(d_max, d_max, d_max);

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

    if (!mesh->isActive()) {
        delete mesh;
        MPI_Finalize();
        return 0;
    }

    const size_t cgSz  = mesh->getDegOfFreedom();
    const size_t unSz  = mesh->getDegOfFreedomUnZip();
    const size_t nblk  = mesh->getLocalBlockList().size();
    const size_t nelem = mesh->getElementLocalEnd() - mesh->getElementLocalBegin();

    double* u          = mesh->createCGVector<double>(func, 1);
    double* u_unzip    = mesh->createUnZippedVector<double>(1);
    mesh->readFromGhostBegin(u, 1);
    mesh->readFromGhostEnd(u, 1);
    mesh->unzip(u, u_unzip, 1);

    std::vector<double> out_a(cgSz), out_b(cgSz);

    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp master
        nthreads = omp_get_num_threads();
    }
#endif

    if (!rank) {
        std::printf(
            "benchZip: eOrder=%u lev=[%u,%u] localElem=%zu blocks=%zu "
            "cgSz=%zu unSz=%zu ranks=%d threads=%d\n",
            eOrder, lmin, lmax, nelem, nblk, cgSz, unSz, npes, nthreads);
        std::printf("  reps=%u calls_per_rep=%u  blocks/thread=%.1f\n", reps,
                    ncalls, (double)nblk / (double)nthreads);

        // Block cost distribution. schedule(static) is only a good idea if
        // blocks are near-uniform in element count -- this prints whether
        // that premise actually holds for this mesh.
        const std::vector<ot::Block>& bl = mesh->getLocalBlockList();
        std::vector<size_t> ec(bl.size());
        size_t tot = 0;
        for (size_t b = 0; b < bl.size(); b++) {
            ec[b] = bl[b].getLocalElementEnd() - bl[b].getLocalElementBegin();
            tot += ec[b];
        }
        std::vector<size_t> es = ec;
        std::sort(es.begin(), es.end());
        const double meanE = (double)tot / (double)bl.size();
        double sdE = 0.0;
        for (size_t v : ec) sdE += ((double)v - meanE) * ((double)v - meanE);
        sdE = std::sqrt(sdE / bl.size());
        std::printf(
            "  elements/block: min=%zu p25=%zu med=%zu p75=%zu max=%zu "
            "mean=%.1f sd=%.1f (CV=%.2f)\n",
            es.front(), es[es.size() / 4], es[es.size() / 2],
            es[3 * es.size() / 4], es.back(), meanE, sdE, sdE / meanE);
    }

    // ZIPBENCH_ONLY=ref|new runs a single path only, so `perf stat` can
    // attribute hardware counters to one implementation without the other's
    // instructions mixed in. Timing output is meaningless in this mode.
    const char* only_env = std::getenv("ZIPBENCH_ONLY");
    const int only_ref   = (only_env && std::strcmp(only_env, "ref") == 0);
    const int only_new   = (only_env && std::strcmp(only_env, "new") == 0);
    if (only_env && !rank)
        std::printf("  ZIPBENCH_ONLY=%s (counter-attribution mode)\n", only_env);

    // warm-up: both paths in A/B mode (equal caches / first touch), only the
    // path under test in counter-attribution mode.
    for (unsigned int w = 0; w < 3; w++) {
        for (unsigned int c = 0; c < ncalls; c++) {
            if (!only_new) mesh->zip_ref(u_unzip, out_a.data());
            if (!only_ref) mesh->zip(u_unzip, out_b.data());
        }
    }

    std::vector<double> ta(reps), tb(reps), tc(reps), td(reps);
    int exact_fail = 0;
    const bool do_dyn = (std::getenv("ZIPBENCH_DYN") != nullptr);
    std::vector<double> out_c(do_dyn ? cgSz : 0);

    // ---- Hypothesis 2 prototype: precomputed (unzip_idx, cg_idx) plan ----
    // The ownership filter is mesh-invariant, so it can be resolved once into
    // a flat pair list and the hot loop becomes a branchless gather/scatter.
    // Built OUTSIDE the timed region: this measures the CEILING of the idea
    // (steady-state cost with a free plan), which is what decides whether the
    // rebuild-on-remesh + memory cost is worth paying.
    const bool do_plan = (std::getenv("ZIPBENCH_PLAN") != nullptr);
    std::vector<unsigned int> plan_uz, plan_cg;
    std::vector<double> out_d;
    double t_planbuild = 0.0;
    if (do_plan) {
        t_planbuild = now_sec();
        out_d.resize(cgSz);
        const std::vector<ot::Block>& bl = mesh->getLocalBlockList();
        const std::vector<ot::TreeNode>& allE = mesh->getAllElements();
        const std::vector<unsigned int>& dgm = mesh->getE2NMapping_DG();
        const std::vector<unsigned int>& cgm = mesh->getE2NMapping();
        const unsigned int eO = mesh->getElementOrder();
        const unsigned int npE = mesh->getNumNodesPerElement();
        const unsigned int eOp1 = eO + 1, eOp1Sq = (eO + 1) * (eO + 1);
        for (size_t b = 0; b < bl.size(); b++) {
            const ot::TreeNode bn = bl[b].getBlockNode();
            const unsigned int rl = bl[b].getRegularGridLev();
            const unsigned int lx = bl[b].getAllocationSzX();
            const unsigned int ly = bl[b].getAllocationSzY();
            const unsigned int off = bl[b].getOffset();
            const unsigned int pw = bl[b].get1DPadWidth();
            for (unsigned int e = bl[b].getLocalElementBegin();
                 e < bl[b].getLocalElementEnd(); e++) {
                const unsigned int ei =
                    (allE[e].getX() - bn.getX()) >> (m_uiMaxDepth - rl);
                const unsigned int ej =
                    (allE[e].getY() - bn.getY()) >> (m_uiMaxDepth - rl);
                const unsigned int ek =
                    (allE[e].getZ() - bn.getZ()) >> (m_uiMaxDepth - rl);
                const unsigned int eb = e * npE;
                for (unsigned int k = 0; k < eOp1; k++)
                    for (unsigned int j = 0; j < eOp1; j++)
                        for (unsigned int i = 0; i < eOp1; i++) {
                            const unsigned int n = k * eOp1Sq + j * eOp1 + i;
                            if ((unsigned int)(dgm[eb + n] - eb) < npE) {
                                plan_uz.push_back(off +
                                                  (ek * eO + k + pw) * ly * lx +
                                                  (ej * eO + j + pw) * lx +
                                                  (ei * eO + i + pw));
                                plan_cg.push_back(cgm[eb + n]);
                            }
                        }
            }
        }
        t_planbuild = now_sec() - t_planbuild;
        if (!rank) {
            const size_t pts = (size_t)nelem * (size_t)(eOp1 * eOp1 * eOp1);
            std::printf(
                "  PLAN: %zu owned pairs of %zu points (%.1f%% pass the "
                "ownership filter); plan memory = %.2f MB "
                "(%.2f bytes/point)\n",
                plan_uz.size(), pts, 100.0 * plan_uz.size() / (double)pts,
                2.0 * plan_uz.size() * sizeof(unsigned int) / 1048576.0,
                2.0 * plan_uz.size() * sizeof(unsigned int) / (double)pts);
            std::printf("  PLAN build cost = %.6e s (once per mesh)\n",
                        t_planbuild);
        }
    }

    for (unsigned int r = 0; r < reps; r++) {
        MPI_Barrier(comm);
        double t0 = now_sec();
        if (!only_new)
            for (unsigned int c = 0; c < ncalls; c++)
                mesh->zip_ref(u_unzip, out_a.data());
        double t1 = now_sec();

        MPI_Barrier(comm);
        double t2 = now_sec();
        if (!only_ref)
            for (unsigned int c = 0; c < ncalls; c++)
                mesh->zip(u_unzip, out_b.data());
        double t3 = now_sec();

        double t6 = t3, t7 = t3;
        if (do_plan) {
            MPI_Barrier(comm);
            t6 = now_sec();
            for (unsigned int c = 0; c < ncalls; c++) {
                const size_t np = plan_uz.size();
                const unsigned int* pu = plan_uz.data();
                const unsigned int* pc = plan_cg.data();
                double* zd = out_d.data();
#if defined(DENDRO_UNZIP_OMP)
#pragma omp parallel for schedule(static)
#endif
                for (size_t n = 0; n < np; n++) zd[pc[n]] = u_unzip[pu[n]];
            }
            t7 = now_sec();
        }
        double t4 = t3, t5 = t3;
        if (do_dyn) {
            MPI_Barrier(comm);
            t4 = now_sec();
            for (unsigned int c = 0; c < ncalls; c++)
                mesh->zip_dyn(u_unzip, out_c.data());
            t5 = now_sec();
        }

        ta[r] = (t1 - t0) / ncalls;
        tb[r] = (t3 - t2) / ncalls;
        tc[r] = (t5 - t4) / ncalls;
        td[r] = (t7 - t6) / ncalls;

        if (!only_env &&
            std::memcmp(out_a.data(), out_b.data(), cgSz * sizeof(double)) != 0)
            exact_fail = 1;
        if (do_dyn && !only_env &&
            std::memcmp(out_a.data(), out_c.data(), cgSz * sizeof(double)) != 0)
            exact_fail = 1;
        if (do_plan && !only_env &&
            std::memcmp(out_a.data(), out_d.data(), cgSz * sizeof(double)) != 0)
            exact_fail = 1;
    }

    // rank-max reduce the per-rep times (slowest rank governs a real step)
    std::vector<double> ta_m(reps), tb_m(reps);
    MPI_Allreduce(ta.data(), ta_m.data(), reps, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(tb.data(), tb_m.data(), reps, MPI_DOUBLE, MPI_MAX, comm);
    int gf = 0;
    MPI_Allreduce(&exact_fail, &gf, 1, MPI_INT, MPI_MAX, comm);

    if (!rank) {
        std::vector<double> R(reps);
        for (unsigned int r = 0; r < reps; r++) R[r] = ta_m[r] / tb_m[r];
        std::vector<double> Rs = R;
        std::sort(Rs.begin(), Rs.end());
        auto pct = [&](double p) {
            return Rs[(size_t)(p * (Rs.size() - 1) + 0.5)];
        };
        std::vector<double> A = ta_m, B = tb_m;
        std::sort(A.begin(), A.end());
        std::sort(B.begin(), B.end());
        const double medA = A[A.size() / 2], medB = B[B.size() / 2];

        double mean = 0.0;
        for (double v : R) mean += v;
        mean /= R.size();
        double sd = 0.0;
        for (double v : R) sd += (v - mean) * (v - mean);
        sd = std::sqrt(sd / R.size());

        std::printf("\n  per-call zip time (rank-max, seconds)\n");
        std::printf("    A  zip_ref  median = %.6e   [min %.6e max %.6e]\n",
                    medA, A.front(), A.back());
        std::printf("    B  zip      median = %.6e   [min %.6e max %.6e]\n",
                    medB, B.front(), B.back());
        std::printf("\n  RATIO A/B (speedup) over %u interleaved reps\n", reps);
        std::printf(
            "    median = %.4f   mean = %.4f  sd = %.4f\n"
            "    p25    = %.4f   p75  = %.4f\n"
            "    min    = %.4f   max  = %.4f\n",
            pct(0.5), mean, sd, pct(0.25), pct(0.75), Rs.front(), Rs.back());
        if (do_dyn) {
            std::vector<double> C = tc;
            MPI_Barrier(MPI_COMM_SELF);
            std::sort(C.begin(), C.end());
            std::vector<double> RC(reps);
            for (unsigned int r = 0; r < reps; r++) RC[r] = tc[r] / tb[r];
            std::sort(RC.begin(), RC.end());
            std::printf(
                "\n  C  zip_dyn (optimized kernel + schedule(dynamic,1))\n"
                "     median = %.6e   ratio C/B (dynamic vs static) median = "
                "%.4f  [p25 %.4f p75 %.4f]\n",
                C[C.size() / 2], RC[RC.size() / 2],
                RC[(size_t)(0.25 * (RC.size() - 1) + 0.5)],
                RC[(size_t)(0.75 * (RC.size() - 1) + 0.5)]);
        }
        if (do_plan) {
            std::vector<double> D = td;
            std::sort(D.begin(), D.end());
            std::vector<double> RD(reps);
            for (unsigned int r = 0; r < reps; r++) RD[r] = tb[r] / td[r];
            std::sort(RD.begin(), RD.end());
            std::printf(
                "\n  D  zip_plan (precomputed pair list, plan build NOT "
                "timed)\n     median = %.6e   ratio B/D (plan vs optimized) "
                "median = %.4f  [p25 %.4f p75 %.4f]\n",
                D[D.size() / 2], RD[RD.size() / 2],
                RD[(size_t)(0.25 * (RD.size() - 1) + 0.5)],
                RD[(size_t)(0.75 * (RD.size() - 1) + 0.5)]);
        }
        std::printf("\n  bit-exactness across all reps: %s\n",
                    gf ? "*** FAIL ***" : "PASS (memcmp==0)");
    }

    mesh->destroyVector(u);
    mesh->destroyVector(u_unzip);
    delete mesh;
    MPI_Finalize();
    return gf ? 1 : 0;
}
