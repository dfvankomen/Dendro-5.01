// testGradOmp.cpp
//
// Microbenchmark that measures the wall-clock scaling of the OpenMP-
// parallelized grad() function in ODE/include/rkTransportUtils.h across
// thread counts. Builds a small octree mesh, allocates unzip buffers,
// and times grad_x/y/z over many iterations.
//
// Usage:
//   export OMP_NUM_THREADS=1; ./testGradOmp 6    # single-thread
//   export OMP_NUM_THREADS=4; ./testGradOmp 6
//   export OMP_NUM_THREADS=8; ./testGradOmp 6
//
// Run at multiple values of OMP_NUM_THREADS and compare. Output per run
// shows mean us per grad() call and throughput (nodes/second).
//
// The benchmark is single-MPI-rank by design — we're measuring thread
// scaling within a rank, not MPI scaling.

#include <chrono>
#include <iostream>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "dendroIO.h"
#include "dendro_mpi_init.h"
#include "functional"
#include "genPts_par.h"
#include "mesh.h"
#include "mpi.h"
#include "octUtils.h"
#include "rkTransportUtils.h"
#include "sfcSort.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    dendro_mpi_init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    // arg 1: maxDepth (refinement level). 5-6 gives reasonable block counts
    unsigned int maxDepth = (argc > 1) ? atoi(argv[1]) : 6;
    unsigned int eOrder   = (argc > 2) ? atoi(argv[2]) : 4;
    unsigned int n_iters  = (argc > 3) ? atoi(argv[3]) : 50;

    m_uiMaxDepth          = maxDepth;

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    if (!rank) {
        std::cout << "=== testGradOmp ===" << std::endl;
        std::cout << "maxDepth:       " << maxDepth << std::endl;
        std::cout << "eOrder:         " << eOrder << std::endl;
        std::cout << "MPI ranks:      " << npes << std::endl;
        std::cout << "OMP threads:    " << n_threads << std::endl;
        std::cout << "iters per dir:  " << n_iters << std::endl;
    }

    _InitializeHcurve(m_uiDim);

    // build a sine-function-adapted octree (same shape as transportEq
    // so block structure is realistic). wavelet_tol = 1e-4, partition_tol = 0.1
    const Point grid_min(0, 0, 0);
    const Point grid_max((1u << m_uiMaxDepth), (1u << m_uiMaxDepth),
                         (1u << m_uiMaxDepth));
    const Point d_min = adv_param::domain_min;
    const Point d_max = adv_param::domain_max;

    // a gaussian with sharp features — forces adaptive refinement so we get
    // a non-trivial block list even at modest maxDepth. centered at the
    // domain center with a tight width
    std::function<double(double, double, double)> seed_fn =
        [grid_min, grid_max](double x, double y, double z) {
            double cx    = 0.5 * (grid_max.x() + grid_min.x());
            double cy    = 0.5 * (grid_max.y() + grid_min.y());
            double cz    = 0.5 * (grid_max.z() + grid_min.z());
            double scale = 0.05 * (grid_max.x() - grid_min.x());
            double dx    = (x - cx) / scale;
            double dy    = (y - cy) / scale;
            double dz    = (z - cz) / scale;
            return std::exp(-0.5 * (dx * dx + dy * dy + dz * dz));
        };

    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(seed_fn, tmpNodes, m_uiMaxDepth, 1e-6, eOrder, comm);

    std::vector<ot::TreeNode> balOctree;
    ot::TreeNode root;
    SFC::parSort::SFC_treeSort(tmpNodes, balOctree, balOctree, balOctree, 0.1,
                               m_uiMaxDepth, root, ROOT_ROTATION, 1,
                               TS_REMOVE_DUPLICATES, 2, comm);
    std::swap(tmpNodes, balOctree);
    balOctree.clear();
    SFC::parSort::SFC_treeSort(tmpNodes, balOctree, balOctree, balOctree, 0.1,
                               m_uiMaxDepth, root, ROOT_ROTATION, 1,
                               TS_CONSTRUCT_OCTREE, 2, comm);
    std::swap(tmpNodes, balOctree);
    balOctree.clear();
    SFC::parSort::SFC_treeSort(tmpNodes, balOctree, balOctree, balOctree, 0.1,
                               m_uiMaxDepth, root, ROOT_ROTATION, 1,
                               TS_BALANCE_OCTREE, 2, comm);
    tmpNodes.clear();

    ot::Mesh* mesh = new ot::Mesh(balOctree, 1, eOrder, comm);

    const auto& blkList = mesh->getLocalBlockList();
    int local_blocks = (int)blkList.size();
    int min_blocks, max_blocks, total_blocks;
    MPI_Reduce(&local_blocks, &min_blocks, 1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&local_blocks, &max_blocks, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&local_blocks, &total_blocks, 1, MPI_INT, MPI_SUM, 0, comm);
    if (!rank) {
        std::cout << "blocks (min/max/total across ranks): " << min_blocks
                  << " / " << max_blocks << " / " << total_blocks << std::endl;
    }

    // unzip buffer size — one double per unzipped node
    const unsigned int uzipSz = mesh->getDegOfFreedomUnZip();
    std::vector<double> u_zip(mesh->getDegOfFreedom(), 0.0);
    std::vector<double> u_unzip(uzipSz, 0.0);
    std::vector<double> du_unzip(uzipSz, 0.0);
    std::cout << "unzip dof:      " << uzipSz << std::endl;

    // fill input with sine pattern (just to have realistic data; values
    // don't affect grad() time since it's stencil-based)
    for (size_t i = 0; i < uzipSz; i++) {
        u_unzip[i] = std::sin(0.01 * i);
    }

    // unzip/zip roundtrip correctness check — fills u_zip with node indices,
    // unzips then zips back, and expects the result to match bit-for-bit.
    // this exercises the parallel unzip/zip loops under whatever
    // OMP_NUM_THREADS is set
    std::vector<double> zip_ref(mesh->getDegOfFreedom());
    std::vector<double> zip_back(mesh->getDegOfFreedom());
    for (size_t i = 0; i < zip_ref.size(); i++) zip_ref[i] = 0.001 * i;
    mesh->readFromGhostBegin(zip_ref.data(), 1);
    mesh->readFromGhostEnd(zip_ref.data(), 1);
    mesh->unzip(zip_ref.data(), u_unzip.data(), 1);
    mesh->zip(u_unzip.data(), zip_back.data());

    size_t mismatches  = 0;
    double max_absdiff = 0.0;
    for (unsigned int i = mesh->getNodeLocalBegin();
         i < mesh->getNodeLocalEnd(); i++) {
        double d = std::abs(zip_ref[i] - zip_back[i]);
        if (d > 0.0) {
            mismatches++;
            if (d > max_absdiff) max_absdiff = d;
        }
    }
    std::cout << "unzip/zip roundtrip mismatches: " << mismatches
              << " (max abs diff " << max_absdiff << ")" << std::endl;

    // warmup: run each op multiple times so CPU frequency saturates and
    // plans are built before we measure
    for (int w = 0; w < 20; w++) {
        mesh->readFromGhostBegin(zip_ref.data(), 1);
        mesh->readFromGhostEnd(zip_ref.data(), 1);
        mesh->unzip(zip_ref.data(), u_unzip.data(), 1);
        mesh->zip(u_unzip.data(), zip_back.data());
        grad<double>(mesh, 0, u_unzip.data(), du_unzip.data());
        grad<double>(mesh, 1, u_unzip.data(), du_unzip.data());
        grad<double>(mesh, 2, u_unzip.data(), du_unzip.data());
    }

    auto time_it = [&](auto fn) -> double {
        MPI_Barrier(comm);  // align starts across ranks for meaningful max
        auto t0 = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < n_iters; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() /
               n_iters;
    };

    // ghost exchange — this is where MPI's inter-rank cost lives. for
    // npes=1 this is essentially free; for npes>1 it scales with surface
    // area per rank
    double us_ghost = time_it([&]() {
        mesh->readFromGhostBegin(zip_ref.data(), 1);
        mesh->readFromGhostEnd(zip_ref.data(), 1);
    });
    double us_unzip = time_it(
        [&]() { mesh->unzip(zip_ref.data(), u_unzip.data(), 1); });
    double us_zip = time_it(
        [&]() { mesh->zip(u_unzip.data(), zip_back.data()); });
    double us_x = time_it(
        [&]() { grad<double>(mesh, 0, u_unzip.data(), du_unzip.data()); });
    double us_y = time_it(
        [&]() { grad<double>(mesh, 1, u_unzip.data(), du_unzip.data()); });
    double us_z = time_it(
        [&]() { grad<double>(mesh, 2, u_unzip.data(), du_unzip.data()); });

    // ------------------------------------------------------------------
    // BSSN-like RHS stage: what dendrolib actually sees when a BSSN RK
    // stage runs. Exercises the dof>1 unzip/zip paths and the per-batch
    // (async_k) call pattern that Ctx::unzip and Ctx::zip follow.
    //
    // pattern matches BSSN_GR/src/bssnCtx.cpp::BSSNCtx::rhs:
    //   for i in 0..async_k:
    //       readFromGhostBegin(ctx, ptr, batch_sz)
    //       readFromGhostEnd(ctx, ptr, batch_sz)
    //       mesh->unzip(ptr, ptr, batch_sz)
    //   // ...bssnRHS compute (stand in with per-block scaling)...
    //   zip(unz_out, out)   // called once at dof=NUM_VARS
    //
    // NUM_VARS=24 and ASYNC_COMM_K=2 match typical BSSN configuration
    const unsigned int NUM_VARS      = 24;
    const unsigned int ASYNC_COMM_K  = 2;
    const unsigned int batch_sz      = NUM_VARS / ASYNC_COMM_K;
    const unsigned int zipSz         = mesh->getDegOfFreedom();

    std::vector<double> bssn_zip(zipSz * NUM_VARS, 0.0);
    std::vector<double> bssn_unz(uzipSz * NUM_VARS, 0.0);
    std::vector<double> bssn_zip_out(zipSz * NUM_VARS, 0.0);

    // populate zip with deterministic values so ghost exchange has work
    for (size_t i = 0; i < bssn_zip.size(); i++) {
        bssn_zip[i] = std::sin(0.003 * i);
    }

    // correctness: full dof=NUM_VARS unzip+zip roundtrip
    mesh->readFromGhostBegin(bssn_zip.data(), NUM_VARS);
    mesh->readFromGhostEnd(bssn_zip.data(), NUM_VARS);
    mesh->unzip(bssn_zip.data(), bssn_unz.data(), NUM_VARS);
    mesh->zip(bssn_unz.data(), bssn_zip_out.data(), NUM_VARS);
    size_t bssn_mism = 0;
    double bssn_maxd = 0.0;
    for (unsigned int v = 0; v < NUM_VARS; v++) {
        for (unsigned int i = mesh->getNodeLocalBegin();
             i < mesh->getNodeLocalEnd(); i++) {
            double d = std::abs(bssn_zip[v * zipSz + i] -
                                bssn_zip_out[v * zipSz + i]);
            if (d > 0.0) {
                bssn_mism++;
                if (d > bssn_maxd) bssn_maxd = d;
            }
        }
    }
    std::cout << "bssn-like roundtrip (dof=" << NUM_VARS
              << ") mismatches: " << bssn_mism << " (max abs diff "
              << bssn_maxd << ")" << std::endl;

    // warmup the bssn-like path
    for (int w = 0; w < 10; w++) {
        for (unsigned int i = 0; i < ASYNC_COMM_K; i++) {
            double* ptr = bssn_zip.data() + i * batch_sz * zipSz;
            double* uptr = bssn_unz.data() + i * batch_sz * uzipSz;
            mesh->readFromGhostBegin(ptr, batch_sz);
            mesh->readFromGhostEnd(ptr, batch_sz);
            mesh->unzip(ptr, uptr, batch_sz);
        }
        mesh->zip(bssn_unz.data(), bssn_zip_out.data(), NUM_VARS);
    }

    // time the BSSN-like unzip pipeline (pipelined async_k batches)
    double us_bssn_unz = time_it([&]() {
        for (unsigned int i = 0; i < ASYNC_COMM_K; i++) {
            double* ptr  = bssn_zip.data() + i * batch_sz * zipSz;
            double* uptr = bssn_unz.data() + i * batch_sz * uzipSz;
            mesh->readFromGhostBegin(ptr, batch_sz);
            mesh->readFromGhostEnd(ptr, batch_sz);
            mesh->unzip(ptr, uptr, batch_sz);
        }
    });

    // time the BSSN-like zip: dof=NUM_VARS batched into one parallel region
    double us_bssn_zip_batched = time_it([&]() {
        mesh->zip(bssn_unz.data(), bssn_zip_out.data(), NUM_VARS);
    });

    // also time the Ctx::zip-style external loop (NUM_VARS calls at dof=1)
    // to quantify the fork/join overhead the batched overload eliminates
    double us_bssn_zip_serial = time_it([&]() {
        for (unsigned int v = 0; v < NUM_VARS; v++) {
            mesh->zip(bssn_unz.data() + v * uzipSz,
                      bssn_zip_out.data() + v * zipSz);
        }
    });

    // aggregate per-rank timings: MAX approximates wall-clock (synced by
    // MPI_Barrier inside time_it), MIN shows the fastest rank
    auto reduce_max = [&](double local) {
        double r = 0;
        MPI_Reduce(&local, &r, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        return r;
    };
    double mx_ghost = reduce_max(us_ghost);
    double mx_unzip = reduce_max(us_unzip);
    double mx_zip   = reduce_max(us_zip);
    double mx_x     = reduce_max(us_x);
    double mx_y     = reduce_max(us_y);
    double mx_z     = reduce_max(us_z);
    double mx_bssn_unz        = reduce_max(us_bssn_unz);
    double mx_bssn_zip_batch  = reduce_max(us_bssn_zip_batched);
    double mx_bssn_zip_serial = reduce_max(us_bssn_zip_serial);

    // rough end-to-end model: RK45 runs 6 stages, each does 1 ghost exchange
    // + 1 unzip + 3 grads + 1 zip. use max-across-ranks to approximate the
    // wall-clock the slowest rank imposes on the whole job
    double per_stage = mx_ghost + mx_unzip + mx_x + mx_y + mx_z + mx_zip;
    double per_step  = 6.0 * per_stage;

    if (!rank) {
        std::cout << "\n---------- results (us/call, max across ranks) ----------"
                  << std::endl;
        std::cout << "ghost exchange: " << mx_ghost << std::endl;
        std::cout << "unzip:          " << mx_unzip << std::endl;
        std::cout << "zip:            " << mx_zip << std::endl;
        std::cout << "grad_x:         " << mx_x << std::endl;
        std::cout << "grad_y:         " << mx_y << std::endl;
        std::cout << "grad_z:         " << mx_z << std::endl;
        std::cout << "per-RK-stage:   " << per_stage << std::endl;
        std::cout << "per-RK45-step:  " << per_step
                  << "  (6 stages × pipeline)" << std::endl;

        std::cout << "\n---------- BSSN-like stage (dof=" << NUM_VARS
                  << ", async_k=" << ASYNC_COMM_K << ") ----------" << std::endl;
        std::cout << "unzip pipeline (ghost+unzip, " << ASYNC_COMM_K
                  << " batches of " << batch_sz << "): " << mx_bssn_unz
                  << std::endl;
        std::cout << "zip batched (dof=" << NUM_VARS
                  << " in one region):             " << mx_bssn_zip_batch
                  << std::endl;
        std::cout << "zip per-var loop (" << NUM_VARS << "x dof=1 calls):          "
                  << mx_bssn_zip_serial
                  << "   (speedup: "
                  << (mx_bssn_zip_serial / mx_bssn_zip_batch) << "x)"
                  << std::endl;
    }

    delete mesh;
    MPI_Finalize();
    return 0;
}
