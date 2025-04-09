
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (argc < 6) {
        if (!rank)
            std::cout << "Usage: " << argv[0]
                      << " maxDepth wavelet_tol partition_tol eleOrder "
                         "grainSize initialData"
                      << std::endl;
        MPI_Abort(comm, 0);
    }

    m_uiMaxDepth                 = atoi(argv[1]);
    double wavelet_tol           = atof(argv[2]);
    double partition_tol         = atof(argv[3]);
    unsigned int eOrder          = atoi(argv[4]);
    unsigned int DENDRO_GRAIN_SZ = atoi(argv[5]);
    unsigned int initialData     = atoi(argv[6]);
    unsigned int SPLIT_FIX       = 256;
    double LOAD_IMB_TOL          = 0.1;

    if (!rank) {
        std::cout << YLW << "maxDepth: " << m_uiMaxDepth << NRM << std::endl;
        std::cout << YLW << "wavelet_tol: " << wavelet_tol << NRM << std::endl;
        std::cout << YLW << "partition_tol: " << partition_tol << NRM
                  << std::endl;
        std::cout << YLW << "eleOrder: " << eOrder << NRM << std::endl;
        std::cout << YLW << "GRAIN_SZ: " << DENDRO_GRAIN_SZ << NRM << std::endl;
    }

    _InitializeHcurve(m_uiDim);

    // function that we need to interpolate.
    double d_min = -5.0;
    double d_max = 5.0;
    if (initialData == 1) {
        // adjust d_min and d_max for nlsm data
        d_min = -15.0;
        d_max = 15.0;
    }

    double dMin[] = {d_min, d_min, d_min};
    double dMax[] = {d_max, d_max, d_max};

    Point pt_min(d_min, d_min, d_min);
    Point pt_max(d_max, d_max, d_max);
    //@note that based on how the functions are defined (f(x), dxf(x), etc) the
    // compuatational domain is equivalent to the grid domain.
    std::function<double(double, double, double)> func = [d_min, d_max,
                                                          initialData](
                                                             const double x,
                                                             const double y,
                                                             const double z) {
        double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
        double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
        double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

        if (initialData == 0) {
            // if((xx < -0.5 || xx > 0.5) || ( yy < -0.5 || yy > 0.5) || (zz
            // < -0.5 || zz > 0.5) )
            //     return 0.0;

            return (sin(2 * M_PI * xx) * sin(2 * M_PI * yy) *
                    sin(2 * M_PI * zz));
        } else if (initialData == 1) {
            // returns data type 2 (NLSMB) chi value, which simulates simple
            // "puncture-like" data
            const double amp1   = 11.5;
            const double amp2   = 11.5;
            const double delta1 = 0.5;
            const double delta2 = 0.5;
            const double xc1    = 1.4;
            const double yc1    = 0.7;
            const double zc1    = 0.0;
            const double xc2    = -1.4;
            const double yc2    = -0.7;
            const double zc2    = 0.0;
            const double epsx1  = 1.0;
            const double epsy1  = 1.0;
            const double epsz1  = 1.0;
            const double epsx2  = 1.0;
            const double epsy2  = 1.0;
            const double epsz2  = 1.0;
            const double R1     = 0.0;
            const double R2     = 0.0;
            const double nu1    = 1.0;
            const double nu2    = -1.0;
            const double Omega  = 0.0;

            double xx = (x / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double yy = (y / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;
            double zz = (z / (1u << m_uiMaxDepth)) * (d_max - d_min) + d_min;

            double rt1 =
                sqrt(epsx1 * (xx - xc1) * (xx - xc1) +
                     epsy1 * (yy - yc1) * (yy - yc1) + (zz - zc1) * (zz - zc1));
            rt1 += 1.0e-14;
            double chi1 =
                amp1 * exp(-(rt1 - R1) * (rt1 - R1) / (delta1 * delta1));

            double rt2 =
                sqrt(epsx2 * (xx - xc2) * (xx - xc2) +
                     epsy2 * (yy - yc2) * (yy - yc2) + (zz - zc2) * (zz - zc2));
            rt2 += 1.0e-14;
            double chi2 =
                amp2 * exp(-(rt2 - R2) * (rt2 - R2) / (delta2 * delta2));
            double dGdx1 = -2.0 * chi1 * (rt1 - R1) / (delta1 * delta1) *
                           epsx1 * (xx - xc1) / rt1;
            double dGdx2 = -2.0 * chi2 * (rt2 - R2) / (delta2 * delta2) *
                           epsx2 * (xx - xc2) / rt2;
            return chi1 + chi2;
        } else if (initialData == 2) {
            // squared radial oscillation
            const double f  = (1.6 / 2.0) * 3.1415926;
            const double a  = 10.0;

            const double x0 = xx * xx + yy * yy + zz * zz;
            const double x1 = exp(-x0);

            // Main vars

            return a * (pow(x0, 2) * x1 * pow(sin(f * sqrt(x0)), 2) + x0 * x1);
        } else if (initialData == 3) {
            // Rosenbrock function initialization
            const double freq = (1.0 / 2.0) * 3.1415926;
            return cos(freq * xx) * cos(freq * yy) * exp(zz) +
                   xx * xx * yy * yy * zz * zz;
        } else if (initialData == 4) {
            // Rosenbrock function initialization
            const double a     = 10.0;
            const double scale = 1.0;

            const double r     = sqrt(xx * xx + yy * yy + zz * zz);
            return a * exp(-1.0 * (r * r)) * pow(r / scale, 5.0);
        } else {
            std::cout << "UNKNOWN INITIAL DATA, EXITING" << std::endl;
            exit(0);
        }

        return 0.0;
    };

    // call function to octree

    if (!rank)
        std::cout << YLW << "Using function2Octree. AMR enabled " << NRM
                  << std::endl;

    // f2olmin is like the max depth we want to refine to.
    // if we don't have two puncture initial data, then it should just be
    // the max depth minus three
    unsigned int maxDepthIn;

    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(func, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder, comm);

    // create the mesh
    ot::Mesh* mesh = ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder,
                                    comm, 1, ot::SM_TYPE::FDM, DENDRO_GRAIN_SZ,
                                    LOAD_IMB_TOL, SPLIT_FIX);

    mesh->setDomainBounds(pt_min, pt_max);
    if (!rank) {
        std::cout << "Domain bounds set" << std::endl;
    }

    unsigned int lmin = 1, lmax = 5;
    mesh->computeMinMaxLevel(lmin, lmax);
    if (!rank) {
        std::cout << "Computed min max level:" << lmin << " " << lmax
                  << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::string save_prefix = "test_mesh_npes" + std::to_string(npes);

    io::vtk::mesh2vtu(mesh, save_prefix.c_str(), 0, nullptr, nullptr, 0,
                      nullptr, nullptr);

    if (!rank) {
        std::cout << "Now computing buildOctreeConnectivity...." << std::endl;
    }

    // now we can do some partitioning checks
    // mesh->buildOctreeConnectivity();
    mesh->repartitionMeshGlobal(false, true, "somewhatinterestingmesh");

    std::string save_prefix_2 =
        "test_mesh_repartitioned_npes" + std::to_string(npes);
    io::vtk::mesh2vtu(mesh, save_prefix_2.c_str(), 0, nullptr, nullptr, 0,
                      nullptr, nullptr);

    // END CLEANUP
    delete mesh;

    if (!rank) {
        std::cout << "---------------------------------------" << std::endl;
        std::cout << "               FINISHED                " << std::endl;
        std::cout << "---------------------------------------" << std::endl;
    }

    MPI_Finalize();
}
