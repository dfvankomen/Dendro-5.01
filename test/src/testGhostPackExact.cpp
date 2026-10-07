// Bit-exactness gate: Mesh::readFromGhostBegin/End (OMP pack/unpack) vs a
// plain serial reference exchange built from the same public accessors.
// GHOST_GATE_SABOTAGE=1/2/3 injects a defect to prove the gate can fail.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"

// Serial reference ghost exchange: identical arithmetic to the library
// pack/unpack, but with no threading and a private MPI tag.
template <typename T>
static void ghostExchangeRef(ot::Mesh* mesh, T* vec, unsigned int dof) {
    if (mesh->getMPICommSizeGlobal() == 1 || !mesh->isActive()) return;

    const std::vector<unsigned int>& sendCnt = mesh->getNodalSendCounts();
    const std::vector<unsigned int>& sendOff = mesh->getNodalSendOffsets();
    const std::vector<unsigned int>& recvCnt = mesh->getNodalRecvCounts();
    const std::vector<unsigned int>& recvOff = mesh->getNodalRecvOffsets();
    const std::vector<unsigned int>& sendProc = mesh->getSendProcList();
    const std::vector<unsigned int>& recvProc = mesh->getRecvProcList();
    const std::vector<unsigned int>& sendSM = mesh->getSendNodeSM();
    const std::vector<unsigned int>& recvSM = mesh->getRecvNodeSM();
    const unsigned int activeNpes = mesh->getMPICommSize();
    const unsigned int numActual  = mesh->getDegOfFreedom();

    const unsigned int sendBSz = sendOff[activeNpes - 1] + sendCnt[activeNpes - 1];
    const unsigned int recvBSz = recvOff[activeNpes - 1] + recvCnt[activeNpes - 1];

    std::vector<T> sendB(dof * sendBSz), recvB(dof * recvBSz);
    std::vector<MPI_Request> sendReq(sendProc.size()), recvReq(recvProc.size());
    const int refTag = 424242;
    MPI_Comm comm = mesh->getMPICommunicator();

    for (unsigned int p = 0; p < recvProc.size(); p++) {
        unsigned int proc_id = recvProc[p];
        MPI_Irecv(&recvB[dof * recvOff[proc_id]], dof * recvCnt[proc_id],
                  par::Mpi_datatype<T>::value(), proc_id, refTag, comm,
                  &recvReq[p]);
    }

    for (unsigned int send_p = 0; send_p < sendProc.size(); send_p++) {
        unsigned int proc_id = sendProc[send_p];
        for (unsigned int var = 0; var < dof; var++) {
            for (unsigned int k = sendOff[proc_id];
                 k < sendOff[proc_id] + sendCnt[proc_id]; k++) {
                sendB[dof * sendOff[proc_id] + var * sendCnt[proc_id] +
                      (k - sendOff[proc_id])] =
                    (vec + var * numActual)[sendSM[k]];
            }
        }
    }

    for (unsigned int p = 0; p < sendProc.size(); p++) {
        unsigned int proc_id = sendProc[p];
        MPI_Isend(&sendB[dof * sendOff[proc_id]], dof * sendCnt[proc_id],
                  par::Mpi_datatype<T>::value(), proc_id, refTag, comm,
                  &sendReq[p]);
    }

    if (!recvReq.empty())
        MPI_Waitall(recvReq.size(), recvReq.data(), MPI_STATUSES_IGNORE);
    if (!sendReq.empty())
        MPI_Waitall(sendReq.size(), sendReq.data(), MPI_STATUSES_IGNORE);

    for (unsigned int recv_p = 0; recv_p < recvProc.size(); recv_p++) {
        unsigned int proc_id = recvProc[recv_p];
        for (unsigned int var = 0; var < dof; var++) {
            for (unsigned int k = recvOff[proc_id];
                 k < recvOff[proc_id] + recvCnt[proc_id]; k++) {
                (vec + var * numActual)[recvSM[k]] =
                    recvB[dof * recvOff[proc_id] + var * recvCnt[proc_id] +
                          (k - recvOff[proc_id])];
            }
        }
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
            std::fprintf(stderr,
                         "Usage: %s maxDepth wavelet_tol partition_tol "
                         "eleOrder [dof=6]\n",
                         argv[0]);
        MPI_Abort(comm, 1);
    }

    m_uiMaxDepth         = std::atoi(argv[1]);
    double wavelet_tol   = std::atof(argv[2]);
    double partition_tol = std::atof(argv[3]);
    unsigned int eOrder  = (unsigned int)std::atoi(argv[4]);
    unsigned int dof     = (argc > 5) ? (unsigned int)std::atoi(argv[5]) : 6u;

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

    int local_fail = 0;

    if (mesh->isActive()) {
        const size_t cgSz = mesh->getDegOfFreedom();
        double* vecA = mesh->createCGVector<double>(0.0, dof);
        std::vector<double> vecB(cgSz * dof);

        if (!rank)
            std::printf(
                "testGhostPackExact: cgSz=%zu dof=%u npes=%d nSend=%u "
                "nRecv=%u\n",
                cgSz, dof, npes, mesh->getSendProcListSize(),
                mesh->getRecvProcListSize());

        for (unsigned int trial = 0; trial < 3; trial++) {
            std::srand(7 + trial);
            for (unsigned int var = 0; var < dof; var++) {
                for (size_t i = 0; i < cgSz; i++) {
                    double v;
                    if (trial == 0)
                        v = (double)(var * cgSz + i);  // index-valued
                    else if (trial == 1)
                        v = -(double)((var * cgSz + i) % 9973);  // sawtooth
                    else
                        v = (double)std::rand() / (double)RAND_MAX - 0.5;
                    vecA[var * cgSz + i] = v;
                }
            }
            std::memcpy(vecB.data(), vecA, cgSz * dof * sizeof(double));

            mesh->readFromGhostBegin(vecA, dof);
            mesh->readFromGhostEnd(vecA, dof);
            ghostExchangeRef(mesh, vecB.data(), dof);

#if defined(GHOST_GATE_SABOTAGE)
            if (cgSz > 0) {
#if GHOST_GATE_SABOTAGE == 1
                vecA[cgSz / 2] += 1e-16 * vecA[cgSz / 2] + 1e-300;
#elif GHOST_GATE_SABOTAGE == 2
                vecA[cgSz * dof - 1] = 0.0;
#elif GHOST_GATE_SABOTAGE == 3
                for (size_t i = 0; i < cgSz * dof; i += 4096) vecA[i] = -vecA[i];
#endif
            }
#endif

            const int bad = (std::memcmp(vecA, vecB.data(),
                                         cgSz * dof * sizeof(double)) != 0)
                                ? 1
                                : 0;
            if (bad) {
                local_fail = 1;
                size_t nd = 0, first = (size_t)-1;
                for (size_t i = 0; i < cgSz * dof; i++) {
                    if (vecA[i] != vecB[i]) {
                        if (first == (size_t)-1) first = i;
                        nd++;
                    }
                }
                std::printf(
                    "  [rank %d] trial %u: MISMATCH differing=%zu/%zu "
                    "first=%zu lib=%.17g ref=%.17g\n",
                    rank, trial, nd, cgSz * dof, first, vecA[first],
                    vecB[first]);
            } else if (!rank) {
                std::printf("  trial %u: bit-exact (%zu doubles, memcmp==0)\n",
                            trial, cgSz * dof);
            }
        }

        // optional laptop timing: GHOST_BENCH_REPS=<n> times pack+unpack only
        if (const char* repsEnv = std::getenv("GHOST_BENCH_REPS")) {
            const int reps = std::atoi(repsEnv);
            double t_pack = 0.0, t_unpack = 0.0, t0;
            for (int r = 0; r < reps; r++) {
                MPI_Barrier(comm);
                t0 = MPI_Wtime();
                mesh->readFromGhostBegin(vecA, dof);
                t_pack += MPI_Wtime() - t0;
                MPI_Barrier(comm);
                t0 = MPI_Wtime();
                mesh->readFromGhostEnd(vecA, dof);
                t_unpack += MPI_Wtime() - t0;
            }
            double maxPack, maxUnpack;
            MPI_Reduce(&t_pack, &maxPack, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&t_unpack, &maxUnpack, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (!rank)
                std::printf(
                    "GHOST_BENCH: reps=%d avg_pack=%.6e s avg_unpack=%.6e s "
                    "(laptop numbers)\n",
                    reps, maxPack / reps, maxUnpack / reps);
        }

        mesh->destroyVector(vecA);
    }

    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX, comm);
    if (!rank) {
#if defined(GHOST_GATE_SABOTAGE)
        std::printf("GHOST_GATE_SABOTAGE=%d active (gate is expected to FAIL)\n",
                    GHOST_GATE_SABOTAGE);
#endif
        std::printf("testGhostPackExact: %s\n", global_fail ? "FAIL" : "PASS");
    }

    delete mesh;
    MPI_Finalize();
    return global_fail ? 1 : 0;
}
