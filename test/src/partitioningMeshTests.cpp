

#include <random>

#include "TreeNode.h"
#include "dendro.h"
#include "dendroIO.h"
#include "fdCoefficient.h"
#include "functional"
#include "genPts_par.h"
#include "mesh.h"
#include "meshTestUtils.h"
#include "mpi.h"
#include "oct2vtk.h"
#include "octUtils.h"
#include "oda.h"
#include "rawIO.h"
#include "rkTransportUtils.h"
#include "sfcSort.h"
#include "stencil.h"
#include "waveletAMR.h"
#include "waveletRefEl.h"

#define EXPORT_MESH

namespace temp_data {

std::random_device rd;
std::mt19937 gen(rd());

std::normal_distribution<double> dist(0.0, 1.0);

}  // namespace temp_data

void enforceMinDepth(ot::Mesh* mesh, const unsigned int minDepth,
                     unsigned int grain_sz, double ld_tol, unsigned int sf_k) {
    std::vector<unsigned int> refine_flags;
    const unsigned int eleLocalBegin = mesh->getElementLocalBegin();
    const unsigned int eleLocalEnd   = mesh->getElementLocalEnd();
    bool isOctChange                 = false;

    if (mesh->isActive()) {
        refine_flags.resize(mesh->getNumLocalMeshElements(), OCT_NO_CHANGE);
        const ot::TreeNode* pNodes = mesh->getAllElements().data();
        for (unsigned int ele = eleLocalBegin; ele < eleLocalEnd; ele++) {
            if (pNodes[ele].getLevel() < minDepth) {
                refine_flags[ele - eleLocalBegin] = OCT_SPLIT;
            }
        }

        // then set the refinement flags
        isOctChange = mesh->setMeshRefinementFlags(refine_flags);
    }
    // communicate refinement between cores.
    bool isOctChange_g;
    MPI_Allreduce(&isOctChange, &isOctChange_g, 1, MPI_CXX_BOOL, MPI_LOR,
                  mesh->getMPIGlobalCommunicator());

    if (isOctChange_g) {
        std::cout << "Remeshing to enforce mindepth..." << std::endl;
        ot::Mesh* newMesh = mesh->ReMesh(grain_sz, ld_tol, sf_k);

        std::swap(mesh, newMesh);
        delete newMesh;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (argc < 5) {
        if (!rank)
            std::cout << "Usage: " << argv[0]
                      << " maxDepth wavelet_tol partition_tol eleOrder"
                      << std::endl;
        MPI_Abort(comm, 0);
    }

    PartitioningOptions partitionOption =
        static_cast<PartitioningOptions>(atoi(argv[1]));
    m_uiMaxDepth                 = atoi(argv[2]);
    double wavelet_tol           = atof(argv[3]);
    double partition_tol         = atof(argv[4]);
    unsigned int eOrder          = atoi(argv[5]);
    unsigned int DENDRO_GRAIN_SZ = atoi(argv[6]);
    unsigned int minDepth        = atoi(argv[7]);
    unsigned int SPLIT_FIX       = 256;
    double LOAD_IMB_TOL          = 0.1;

    if (!rank) {
        std::cout << YLW << "Partitioning option: " << partitionOption
                  << std::endl;
        std::cout << YLW << "maxDepth: " << m_uiMaxDepth << NRM << std::endl;
        std::cout << YLW << "wavelet_tol: " << wavelet_tol << NRM << std::endl;
        std::cout << YLW << "partition_tol: " << partition_tol << NRM
                  << std::endl;
        std::cout << YLW << "eleOrder: " << eOrder << NRM << std::endl;
        std::cout << YLW << "GRAIN_SZ: " << DENDRO_GRAIN_SZ << NRM << std::endl;
        std::cout << YLW << "minDepth: " << minDepth << NRM << std::endl;
    }

    _InitializeHcurve(m_uiDim);

    // function that we need to interpolate.
    const double d_min = -5.5;
    const double d_max = 5.5;
    double dMin[]      = {d_min, d_min, d_min};
    double dMax[]      = {d_max, d_max, d_max};

    Point pt_min(d_min, d_min, d_min);
    Point pt_max(d_max, d_max, d_max);

    std::function<double(double, double, double)> func_flat =
        [d_min, d_max](const double x, const double y, const double z) {
            return 1e8 * temp_data::dist(temp_data::gen);
        };

    //@note that based on how the functions are defined (f(x), dxf(x), etc)
    // the
    // compuatational domain is equivalent to the grid domain.
    std::function<double(double, double, double)> func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (sin(2 * M_PI * xx) * sin(2 * M_PI * yy) *
                    sin(2 * M_PI * zz));
        };
    std::function<double(double, double, double)> dx_func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (2 * M_PI * (1.0 / (1u << m_uiMaxDepth) * (d_max - d_min))) *
                   (cos(2 * M_PI * xx) * sin(2 * M_PI * yy) *
                    sin(2 * M_PI * zz));
        };

    std::function<double(double, double, double)> dy_func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (2 * M_PI * (1.0 / (1u << m_uiMaxDepth) * (d_max - d_min))) *
                   (sin(2 * M_PI * xx) * cos(2 * M_PI * yy) *
                    sin(2 * M_PI * zz));
        };

    std::function<double(double, double, double)> dz_func =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (2 * M_PI * (1.0 / (1u << m_uiMaxDepth) * (d_max - d_min))) *
                   (sin(2 * M_PI * xx) * sin(2 * M_PI * yy) *
                    cos(2 * M_PI * zz));
        };

    std::function<double(double, double, double)> func_alt =
        [d_min, d_max](const double x, const double y, const double z) {
            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            // squared radial oscillation
            const double f  = (1.6 / 2.0) * 3.1415926;
            const double a  = 10.0;

            const double x0 = xx * xx + yy * yy + zz * zz;
            const double x1 = exp(-x0);

            // Main vars

            return a * (pow(x0, 2) * x1 * pow(sin(f * sqrt(x0)), 2) + x0 * x1);
        };

    // call function to octree

    // f2olmin is like the max depth we want to refine to.
    // if we don't have two puncture initial data, then it should just be
    // the max depth minus three
    unsigned int maxDepthIn;

    std::vector<ot::TreeNode> tmpNodes;
    // function2Octree(func_alt, tmpNodes, m_uiMaxDepth, wavelet_tol,
    // eOrder,
    //                 comm);
    function2Octree(func_flat, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder,
                    comm);

    // THIS MESH WILL NOT BE AFFECTED
    // MESH DEFAULT DATA
    ot::Mesh* mesh = ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder,
                                    comm, 1, ot::SM_TYPE::FDM, DENDRO_GRAIN_SZ,
                                    LOAD_IMB_TOL, SPLIT_FIX);
    // create a "copy" of the mesh (which will use the exact same
    // information as inputs)
    ot::Mesh* mesh_repartitioned = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FDM,
        DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);

    unsigned int lmin = 1, lmax = 5;

    // set mesh data on "post" build
    mesh->setDomainBounds(pt_min, pt_max);
    mesh->computeMinMaxLevel(lmin, lmax);
    mesh_repartitioned->setDomainBounds(pt_min, pt_max);
    mesh_repartitioned->computeMinMaxLevel(lmin, lmax);

#if 0
    enforceMinDepth(mesh, minDepth, DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
    enforceMinDepth(mesh_repartitioned, minDepth, DENDRO_GRAIN_SZ, LOAD_IMB_TOL,
                    SPLIT_FIX);
#endif

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef EXPORT_MESH
    std::string save_prefix = "test_mesh_npes" + std::to_string(npes);
    io::vtk::mesh2vtu(mesh, save_prefix.c_str(), 0, nullptr, nullptr, 0,
                      nullptr, nullptr);
#endif

    mesh_repartitioned->setPartitioningMethod(partitionOption);

    // now we can do some partitioning checks
    mesh_repartitioned->repartitionMeshGlobal();

#ifdef EXPORT_MESH
    std::string save_prefix_2 =
        "test_mesh_repartitioned_npes" + std::to_string(npes);
    io::vtk::mesh2vtu(mesh_repartitioned, save_prefix_2.c_str(), 0, nullptr,
                      nullptr, 0, nullptr, nullptr);
#endif

    /**
     *
     * INITIAL TESTS
     *
     */
    // if the partitioning option is "NoPartition" or "OriginalPartition"
    // then every output should be the same!

    if (partitionOption == PartitioningOptions::NoPartition ||
        partitionOption == PartitioningOptions::OriginalPartition) {
        // this is our identity comparison, we need to make sure that all
        // important data structures were properly implemented

        // basically we just need to do a bunch of assertions, which we'll
        // go ahead and call as a "comparison" operator in the Mesh

        int retval = mesh->compareCriticalMeshObjects(mesh_repartitioned);

        // then share the ret value across comms
        int summedval;

        MPI_Allreduce(&retval, &summedval, 1, MPI_INT, MPI_MIN, comm);

        if (!rank) {
            if (summedval == 0) {
                std::cout << GRN
                          << "SUCCESS! ALL comparision tests across all "
                             "processors succeeded on "
                             "NoPartion/OriginalPartition!"
                          << NRM << std::endl;
            } else {
                std::cout << RED
                          << "FAILURE! Not all comparison tests suceeded! "
                             "Check the logs!"
                          << NRM << std::endl;
            }
        }
    }

    // END CLEANUP
    delete mesh;
    delete mesh_repartitioned;

    if (!rank) {
        std::cout << "---------------------------------------" << std::endl;
        std::cout << "               FINISHED                " << std::endl;
        std::cout << "---------------------------------------" << std::endl;
    }

    MPI_Finalize();
}
