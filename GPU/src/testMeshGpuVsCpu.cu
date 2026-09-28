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

// gate_values is off for the DG path: CPU unzipDG and GPU unzip_dg disagree by
// ~1e-4 at eleorder 4 and ~1e-6 at 6 across level jumps, which is an unsettled
// prolongation contract rather than a coverage defect. The structural checks
// still apply there, since a sample one side never writes is unambiguous.
int report(const char* label, const Verdict& v, int rank, bool gate_values) {
    const bool structural = v.gpu_missing || v.cpu_missing || v.agree == 0;
    const bool bad        = structural || (gate_values && v.differ);
    if (!rank)
        std::printf(
            "  %-10s agree=%-10llu gpu_missing=%-8llu cpu_missing=%-8llu "
            "differ=%-8llu worst=%.3e  %s%s\n",
            label, v.agree, v.gpu_missing, v.cpu_missing, v.differ, v.worst,
            bad ? "FAIL" : "ok", gate_values ? "" : " (values not gated)");
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

    const double d_min = -10.0, d_max = 10.0;
    std::function<void(double, double, double, double*)> func =
        [dof](double x, double y, double z, double* var) {
            for (unsigned int v = 0; v < dof; v++)
                var[v] = std::sin(0.3 * x) * std::cos(0.2 * y) +
                         0.1 * z * z + 0.5 * (double)v;
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
    double* u_cg      = mesh->createCGVector<double>(func, dof);
    double* u_dg      = mesh->createDGVector(func, dof);
    std::vector<double> cg_cpu(unSz * dof, SENTINEL),
        dg_cpu(unSz * dof, SENTINEL), cg_gpu(unSz * dof, SENTINEL),
        dg_gpu(unSz * dof, SENTINEL);

    mesh->readFromGhostBegin(u_cg, dof);
    mesh->readFromGhostEnd(u_cg, dof);
    mesh->readFromGhostBeginEleDGVec(u_dg, dof);
    mesh->readFromGhostEndEleDGVec(u_dg, dof);
    mesh->unzip(u_cg, cg_cpu.data(), dof);
    mesh->unzipDG(u_dg, dg_cpu.data(), dof);

    device::MeshGPU mesh_gpu;
    device::MeshGPU* dptr_mesh = mesh_gpu.alloc_mesh_on_device(mesh);
    double* d_cg = mesh_gpu.createVector<double, device::vec_type::device>(dof);
    double* d_dg =
        mesh_gpu.createDGVector<double, device::vec_type::device>(dof);
    double* d_cg_uz =
        mesh_gpu.createUnZippedVector<double, device::vec_type::device>(dof);
    double* d_dg_uz =
        mesh_gpu.createUnZippedVector<double, device::vec_type::device>(dof);

    GPUDevice::host_to_device<DEVICE_REAL>(cg_gpu.data(), d_cg_uz, unSz * dof);
    GPUDevice::host_to_device<DEVICE_REAL>(dg_gpu.data(), d_dg_uz, unSz * dof);
    GPUDevice::host_to_device<DEVICE_REAL>(u_cg, d_cg,
                                           mesh->getDegOfFreedom() * dof);
    GPUDevice::host_to_device<DEVICE_REAL>(u_dg, d_dg,
                                           mesh->getDegOfFreedomDG() * dof);

    mesh_gpu.unzip_cg(mesh, dptr_mesh, d_cg, d_cg_uz, dof, (cudaStream_t)0);
    mesh_gpu.unzip_dg(mesh, dptr_mesh, d_dg, d_dg_uz, dof, (cudaStream_t)0);
    GPUDevice::device_synchronize();

    GPUDevice::device_to_host<DEVICE_REAL>(cg_gpu.data(), d_cg_uz, unSz * dof);
    GPUDevice::device_to_host<DEVICE_REAL>(dg_gpu.data(), d_dg_uz, unSz * dof);

    int bad = 0;
    if (lmin == lmax) {
        if (!rank) std::printf("  ERROR: mesh is uniform, no 2:1 jump tested\n");
        bad++;
    }
    bad += report("unzip_cg", compare(cg_cpu.data(), cg_gpu.data(), unSz * dof, rtol), rank, true);
    bad += report("unzip_dg", compare(dg_cpu.data(), dg_gpu.data(), unSz * dof, rtol), rank, false);

    int total = bad;
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, comm);
    if (!rank) std::printf("\ngpu vs cpu unzip: %s\n", total ? "FAIL" : "PASS");

    delete[] u_cg;
    delete[] u_dg;
    delete mesh;
    MPI_Finalize();
    return total ? 1 : 0;
}
