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

    if (npes != 1) {
        if (!rank)
            std::cerr << "testGradOmp: single-rank benchmark — run with -np 1"
                      << std::endl;
        MPI_Finalize();
        return 1;
    }

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

    std::cout << "=== testGradOmp ===" << std::endl;
    std::cout << "maxDepth:       " << maxDepth << std::endl;
    std::cout << "eOrder:         " << eOrder << std::endl;
    std::cout << "OMP threads:    " << n_threads << std::endl;
    std::cout << "iters per dir:  " << n_iters << std::endl;

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
    std::cout << "num blocks:     " << blkList.size() << std::endl;
    std::cout << "local elements: "
              << (mesh->getElementLocalEnd() - mesh->getElementLocalBegin())
              << std::endl;

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

    // warmup: once per direction
    grad<double>(mesh, 0, u_unzip.data(), du_unzip.data());
    grad<double>(mesh, 1, u_unzip.data(), du_unzip.data());
    grad<double>(mesh, 2, u_unzip.data(), du_unzip.data());

    // timed loop — each direction separately so we can see per-axis cost
    auto time_dir = [&](unsigned int dir) -> double {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < n_iters; i++) {
            grad<double>(mesh, dir, u_unzip.data(), du_unzip.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() /
               n_iters;
    };

    double us_x = time_dir(0);
    double us_y = time_dir(1);
    double us_z = time_dir(2);

    std::cout << "\n---------- results ----------" << std::endl;
    std::cout << "grad_x:  " << us_x << " us/call" << std::endl;
    std::cout << "grad_y:  " << us_y << " us/call" << std::endl;
    std::cout << "grad_z:  " << us_z << " us/call" << std::endl;
    std::cout << "total:   " << (us_x + us_y + us_z) << " us/call"
              << std::endl;
    std::cout << "throughput: "
              << (uzipSz * n_threads / (us_x * 1e-6)) / 1e6
              << " M-nodes/sec/thread (x dir)" << std::endl;

    delete mesh;
    MPI_Finalize();
    return 0;
}
