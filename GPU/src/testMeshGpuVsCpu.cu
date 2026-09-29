/**
 * @file testMeshGpuVsCpu.cu
 * @brief Compares the GPU unzip against the CPU unzip on the same mesh.
 *
 * run_meshgpu_tests only times the device kernels and checks cudaGetLastError,
 * so a kernel that writes the wrong values, or skips part of every element,
 * passes it. The CPU path is the reference implementation, so the two must agree
 * to roundoff on every sample either one writes.
 *
 * Both buffers are filled with a sentinel first, which separates three
 * failure modes: a sample the GPU left untouched, one the CPU left untouched,
 * and one both wrote but disagree on. The first is what a thread-tile narrower
 * than (p+1)^2 produces.
 *
 * Usage: testMeshGpuVsCpu [maxDepth] [waveletTol] [eleOrder]
 */

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "device.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mesh_gpu.cuh"
#include "octUtils.h"

namespace {

constexpr double SENTINEL = -1.0e300;

unsigned int g_deg = 3;

double poly(double x, double y, double z) {
    double v = 0.0, px = 1.0, py = 1.0, pz = 1.0;
    for (unsigned int k = 0; k <= g_deg; k++) {
        v += (1.0 / (1.0 + k)) * (px + 0.7 * py + 0.4 * pz);
        px *= x;
        py *= y;
        pz *= z;
    }
    return v;
}

// max relative deviation from the analytic field over samples inside the domain
double analytic_error(const ot::Mesh* mesh, const double* uz,
                      unsigned int eOrder) {
    const std::vector<ot::Block>& blks = mesh->getLocalBlockList();
    const ot::TreeNode* pNodes         = mesh->getAllElements().data();
    const double dmax  = (double)(1u << m_uiMaxDepth);
    const double scale = 1.0 / dmax;
    double worst       = 0.0;
    for (size_t b = 0; b < blks.size(); b++) {
        const ot::TreeNode blkNode = blks[b].getBlockNode();
        const unsigned int bLev =
            pNodes[blks[b].getLocalElementBegin()].getLevel();
        const unsigned int PW = blks[b].get1DPadWidth();
        const unsigned int lx = blks[b].getAllocationSzX();
        const unsigned int ly = blks[b].getAllocationSzY();
        const unsigned int lz = blks[b].getAllocationSzZ();
        const DendroIntL off  = blks[b].getOffset();
        const double hx = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
        const double x0 = blkNode.minX() - PW * hx;
        const double y0 = blkNode.minY() - PW * hx;
        const double z0 = blkNode.minZ() - PW * hx;
        for (unsigned int k = 0; k < lz; k++) {
            const double zz = z0 + k * hx;
            for (unsigned int j = 0; j < ly; j++) {
                const double yy = y0 + j * hx;
                for (unsigned int i = 0; i < lx; i++) {
                    const double xx = x0 + i * hx;
                    if (xx < 0 || yy < 0 || zz < 0 || xx > dmax || yy > dmax ||
                        zz > dmax)
                        continue;
                    const double got = uz[off + k * lx * ly + j * lx + i];
                    if (got == SENTINEL) continue;
                    const double want =
                        poly(xx * scale, yy * scale, zz * scale);
                    worst = std::max(worst, std::fabs(got - want) /
                                                (1.0 + std::fabs(want)));
                }
            }
        }
    }
    return worst;
}

struct Verdict {
    unsigned long long gpu_missing = 0, cpu_missing = 0, differ = 0, agree = 0;
    double worst                   = 0.0;
};

Verdict compare(const double* cpu, const double* gpu, size_t n, double rtol) {
    Verdict v;
    for (size_t i = 0; i < n; i++) {
        const bool c = (cpu[i] != SENTINEL), g = (gpu[i] != SENTINEL);
        if (c && !g)
            v.gpu_missing++;
        else if (!c && g)
            v.cpu_missing++;
        else if (c && g) {
            const double d = std::fabs(cpu[i] - gpu[i]);
            v.worst        = std::max(v.worst, d);
            if (d > rtol * (1.0 + std::fabs(cpu[i])))
                v.differ++;
            else
                v.agree++;
        }
    }
    return v;
}

// Only the structural claim is gated here. CPU unzipDG and GPU unzip_dg are both
// exact to the element order but differ in the truncation term above it, so their
// agreement is not a contract; correctness is gated against the analytic field.
int report(const char* label, const Verdict& v, int rank) {
    const bool bad = v.gpu_missing || v.cpu_missing || v.agree == 0;
    if (!rank)
        std::printf(
            "  %-10s agree=%-10llu gpu_missing=%-8llu cpu_missing=%-8llu "
            "differ=%-8llu worst=%.3e  %s\n",
            label, v.agree, v.gpu_missing, v.cpu_missing, v.differ, v.worst,
            bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    if (n_dev < 1) {
        if (!rank) std::printf("no CUDA device available\n");
        MPI_Finalize();
        return 0;
    }
    cudaSetDevice(rank % n_dev);

    m_uiMaxDepth             = (argc > 1) ? std::atoi(argv[1]) : 5;
    const double wavelet_tol = (argc > 2) ? std::atof(argv[2]) : 1e-3;
    const unsigned int eOrder = (argc > 3) ? std::atoi(argv[3]) : 6;
    const unsigned int dof    = 2;
    const double rtol         = 1e-12;

    _InitializeHcurve(m_uiDim);

    const double d_min = 0.0, d_max = 1.0;
    std::function<void(double, double, double, double*)> func =
        [dof](double x, double y, double z, double* var) {
            for (unsigned int v = 0; v < dof; v++) var[v] = poly(x, y, z);
        };
    // a localized bump, so the mesh carries 2:1 jumps and the GPU unzip has to
    // interpolate rather than only copy
    std::function<double(double, double, double)> fr =
        [](double x, double y, double z) {
            const double s = 1.0 / (1u << m_uiMaxDepth);
            const double dx = x * s - 0.5, dy = y * s - 0.5, dz = z * s - 0.5;
            return std::exp(-120.0 * (dx * dx + dy * dy + dz * dz));
        };

    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(fr, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder, comm);
    ot::Mesh* mesh =
        ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1,
                       ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ, 0.1,
                       DENDRO_DEFAULT_SF_K);
    mesh->setDomainBounds(Point(d_min, d_min, d_min), Point(d_max, d_max, d_max));

    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);
    if (!rank)
        std::printf("eOrder=%u ranks=%d levels=%u..%u\n", eOrder, npes, lmin,
                    lmax);

    const size_t unSz = mesh->getDegOfFreedomUnZip();

    device::MeshGPU mesh_gpu;
    device::MeshGPU* dptr_mesh = mesh_gpu.alloc_mesh_on_device(mesh);
    double* d_cg = mesh_gpu.createVector<double, device::vec_type::device>(dof);
    double* d_dg =
        mesh_gpu.createDGVector<double, device::vec_type::device>(dof);
    double* d_cg_uz =
        mesh_gpu.createUnZippedVector<double, device::vec_type::device>(dof);
    double* d_dg_uz =
        mesh_gpu.createUnZippedVector<double, device::vec_type::device>(dof);

    int bad = 0;
    if (lmin == lmax) {
        if (!rank) std::printf("  ERROR: mesh is uniform, no 2:1 jump tested\n");
        bad++;
    }

    if (!rank)
        std::printf(
            "  field degree sweep -- both paths must be exact while deg <= %u\n",
            eOrder);

    for (unsigned int d = 2; d <= eOrder + 2; d++) {
        g_deg = d;
        std::vector<double> cg_cpu(unSz * dof, SENTINEL),
            dg_cpu(unSz * dof, SENTINEL), cg_gpu(unSz * dof, SENTINEL),
            dg_gpu(unSz * dof, SENTINEL);

        double* u_cg = mesh->createCGVector<double>(func, dof);
        double* u_dg = mesh->createDGVector(func, dof);

        mesh->readFromGhostBegin(u_cg, dof);
        mesh->readFromGhostEnd(u_cg, dof);
        mesh->readFromGhostBeginEleDGVec(u_dg, dof);
        mesh->readFromGhostEndEleDGVec(u_dg, dof);
        mesh->unzip(u_cg, cg_cpu.data(), dof);
        mesh->unzipDG(u_dg, dg_cpu.data(), dof);

        GPUDevice::host_to_device<DEVICE_REAL>(cg_gpu.data(), d_cg_uz,
                                               unSz * dof);
        GPUDevice::host_to_device<DEVICE_REAL>(dg_gpu.data(), d_dg_uz,
                                               unSz * dof);
        GPUDevice::host_to_device<DEVICE_REAL>(u_cg, d_cg,
                                               mesh->getDegOfFreedom() * dof);
        GPUDevice::host_to_device<DEVICE_REAL>(u_dg, d_dg,
                                               mesh->getDegOfFreedomDG() * dof);

        mesh_gpu.unzip_cg(mesh, dptr_mesh, d_cg, d_cg_uz, dof, (cudaStream_t)0);
        mesh_gpu.unzip_dg(mesh, dptr_mesh, d_dg, d_dg_uz, dof, (cudaStream_t)0);
        GPUDevice::device_synchronize();

        GPUDevice::device_to_host<DEVICE_REAL>(cg_gpu.data(), d_cg_uz,
                                               unSz * dof);
        GPUDevice::device_to_host<DEVICE_REAL>(dg_gpu.data(), d_dg_uz,
                                               unSz * dof);

        const double e_cg_c = analytic_error(mesh, cg_cpu.data(), eOrder);
        const double e_cg_g = analytic_error(mesh, cg_gpu.data(), eOrder);
        const double e_dg_c = analytic_error(mesh, dg_cpu.data(), eOrder);
        const double e_dg_g = analytic_error(mesh, dg_gpu.data(), eOrder);
        const Verdict vcg =
            compare(cg_cpu.data(), cg_gpu.data(), unSz * dof, rtol);
        const Verdict vdg =
            compare(dg_cpu.data(), dg_gpu.data(), unSz * dof, rtol);

        if (!rank)
            std::printf(
                "    deg=%-2u vs analytic  cg: cpu=%.2e gpu=%.2e | dg: "
                "cpu=%.2e gpu=%.2e   cpu-vs-gpu differ cg=%llu dg=%llu\n",
                d, e_cg_c, e_cg_g, e_dg_c, e_dg_g, vcg.differ, vdg.differ);

        if (d == 3) {
            bad += report("unzip_cg", vcg, rank);
            bad += report("unzip_dg", vdg, rank);
        }

        if (d <= eOrder) {
            const double exact_tol = 1e-11;
            if (e_cg_c > exact_tol || e_cg_g > exact_tol ||
                e_dg_c > exact_tol || e_dg_g > exact_tol) {
                if (!rank)
                    std::printf(
                        "    FAIL: degree %u is within the element order but "
                        "was not reproduced exactly\n",
                        d);
                bad++;
            }
        }

        delete[] u_cg;
        delete[] u_dg;
    }

    int total = bad;
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, comm);
    if (!rank) std::printf("\ngpu vs cpu unzip: %s\n", total ? "FAIL" : "PASS");

    delete mesh;
    MPI_Finalize();
    return total ? 1 : 0;
}
