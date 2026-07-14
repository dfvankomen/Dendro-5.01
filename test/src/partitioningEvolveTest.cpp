// RK4 evolution test for the heat equation:
//   du/dt = nu * Laplacian(u)
//
// Builds two identical meshes (SFC partition + graph partition via
// repartitionMeshGlobal), evolves both with the same time stepper +
// stencil, periodically remeshes both, and verifies the solutions
// agree. The point is a "full stack" smoke test: createVector, ghost
// exchange, unzip, FD stencil, ReMesh / ReMeshRepartitioned, intergrid
// transfer.
//
// Spatial: 4th-order central FD for second derivatives on padded
// blocks. Time: classical RK4. Boundary: Dirichlet u=0 implied by
// skipping near-boundary cells in the stencil (same as Test 9 of
// partitioningMeshTests).
//
// Usage:
//   testPartitioningEvolve <partitionOpt> <maxDepth> <wavelet_tol>
//                          <part_tol> <eOrder> <grain_sz> <minDepth>
//                          [num_steps] [dt_factor] [remesh_every]
//
// partitionOpt: 0=NoPartition 1=Original 2=Random 3=Fastpart

#include <iomanip>
#include <map>
#include <tuple>
#include <unordered_map>

#include "TreeNode.h"
#include "dendro.h"
#include "functional"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"
#include "octUtils.h"
#include "sfcSort.h"

namespace {
struct TreeNodeHash {
    size_t operator()(const ot::TreeNode& t) const noexcept {
        // splitmix64-style mixing of x/y/z/level packed into 64 bits.
        // x/y/z fit in m_uiMaxDepth bits (≤31), level fits in 5 bits,
        // so all four pack losslessly into 64 bits.
        uint64_t h = (uint64_t)t.getX();
        h         = (h << 21) ^ (uint64_t)t.getY();
        h         = (h << 21) ^ (uint64_t)t.getZ();
        h         = (h << 6)  ^ (uint64_t)t.getLevel();
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return (size_t)h;
    }
};

const double NU            = 0.05;   // diffusion coeff
const double DOM_MIN       = -5.5;
const double DOM_MAX       = 5.5;

// Initial condition: Gaussian pulse at domain center. Diffuses during
// evolution so wavelet should coarsen over time — exercises adaptive
// refinement decisions (not just static refined mesh). Matches the
// NLSM-style localized-feature IC that exposes fastpart-on-adaptive
// pathologies.
double init_func(double xg, double yg, double zg) {
    const double scale = (DOM_MAX - DOM_MIN) / (1u << m_uiMaxDepth);
    const double xp    = xg * scale + DOM_MIN;
    const double yp    = yg * scale + DOM_MIN;
    const double zp    = zg * scale + DOM_MIN;
    // Gaussian at origin, width 0.8
    const double sigma = 0.8;
    const double r2    = xp * xp + yp * yp + zp * zp;
    return std::exp(-r2 / (2.0 * sigma * sigma));
}

// Compute Laplacian via 4th-order central FD on each block, then
// scatter back into a CG vector via getElementNodalValues' inverse
// (write per element — we use unzip→stencil→pack-back-to-element-CG).
//
// Implementation strategy: for each LOCAL element, gather its 125
// nodal values from the unzipped buffer at element coordinates,
// then write into the output CG via E2N_CG. For CGs shared with
// multiple elements, the value is the same (they all see the same
// per-element FD result at that physical position).
// Thin wrapper around Mesh::orphanPreGather, preserved so existing
// test call sites keep working.
void orphanPreGather(ot::Mesh* mesh, std::vector<double>& vec) {
    mesh->orphanPreGather(vec.data());
}

void compute_rhs(ot::Mesh* mesh, const std::vector<double>& uVec,
                 std::vector<double>& rhsVec, std::vector<double>& uzIn,
                 std::vector<double>& uzOut) {
    if (!mesh->isActive()) return;

    // unzip
    orphanPreGather(mesh, const_cast<std::vector<double>&>(uVec));
    mesh->performGhostExchange(const_cast<std::vector<double>&>(uVec));
    std::fill(uzOut.begin(), uzOut.end(), 0.0);
    mesh->unzip(uVec.data(), uzIn.data());

    const auto& blkList         = mesh->getLocalBlockList();
    const unsigned int eOrder   = mesh->getElementOrder();
    const unsigned int maxDepth = m_uiMaxDepth;

    // 4th-order central second-derivative stencil
    const double c2[5] = {-1.0, 16.0, -30.0, 16.0, -1.0};

    // for each block, compute Laplacian into uzOut at every interior
    // point (padding excluded — those just stay 0 since stencil reaches
    // outside)
    for (unsigned int b = 0; b < blkList.size(); b++) {
        const ot::Block& blk    = blkList[b];
        const unsigned int pw   = blk.get1DPadWidth();
        if (pw < 2) continue;
        const unsigned int lx   = blk.getAllocationSzX();
        const unsigned int ly   = blk.getAllocationSzY();
        const unsigned int lz   = blk.getAllocationSzZ();
        const DendroIntL offset = blk.getOffset();
        const unsigned int rL   = blk.getRegularGridLev();
        const unsigned int n1D  = blk.getElemSz1D();

        const double phys_per_grid =
            (DOM_MAX - DOM_MIN) / (double)(1u << maxDepth);
        const double hx_grid = (1u << (maxDepth - rL)) / (double)eOrder;
        const double h       = hx_grid * phys_per_grid;
        const double inv12h2 = 1.0 / (12.0 * h * h);

        const unsigned int iLo = pw;
        const unsigned int iHi = pw + n1D * eOrder;

        for (unsigned int k = iLo; k <= iHi; k++) {
            for (unsigned int j = iLo; j <= iHi; j++) {
                for (unsigned int i = iLo; i <= iHi; i++) {
                    auto IDX = [&](int ii, int jj, int kk) {
                        return offset + ii + lx * (jj + ly * kk);
                    };
                    double dxx = (c2[0] * uzIn[IDX(i - 2, j, k)] +
                                  c2[1] * uzIn[IDX(i - 1, j, k)] +
                                  c2[2] * uzIn[IDX(i, j, k)] +
                                  c2[3] * uzIn[IDX(i + 1, j, k)] +
                                  c2[4] * uzIn[IDX(i + 2, j, k)]) *
                                 inv12h2;
                    double dyy = (c2[0] * uzIn[IDX(i, j - 2, k)] +
                                  c2[1] * uzIn[IDX(i, j - 1, k)] +
                                  c2[2] * uzIn[IDX(i, j, k)] +
                                  c2[3] * uzIn[IDX(i, j + 1, k)] +
                                  c2[4] * uzIn[IDX(i, j + 2, k)]) *
                                 inv12h2;
                    double dzz = (c2[0] * uzIn[IDX(i, j, k - 2)] +
                                  c2[1] * uzIn[IDX(i, j, k - 1)] +
                                  c2[2] * uzIn[IDX(i, j, k)] +
                                  c2[3] * uzIn[IDX(i, j, k + 1)] +
                                  c2[4] * uzIn[IDX(i, j, k + 2)]) *
                                 inv12h2;
                    uzOut[IDX(i, j, k)] = NU * (dxx + dyy + dzz);
                }
            }
        }
    }

    // zip back: write block-interior unzipped values back into rhsVec
    // (CG). Use mesh's `zip` function.
    std::fill(rhsVec.begin(), rhsVec.end(), 0.0);
    mesh->zip(uzOut.data(), rhsVec.data());
}

// RK4 step: u_new = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
void rk4_step(ot::Mesh* mesh, std::vector<double>& uVec,
              std::vector<double>& tmpVec, std::vector<double>& k1,
              std::vector<double>& k2, std::vector<double>& k3,
              std::vector<double>& k4, std::vector<double>& uzIn,
              std::vector<double>& uzOut, double dt) {
    if (!mesh->isActive()) return;
    const size_t n = uVec.size();

    compute_rhs(mesh, uVec, k1, uzIn, uzOut);

    for (size_t i = 0; i < n; i++) tmpVec[i] = uVec[i] + 0.5 * dt * k1[i];
    compute_rhs(mesh, tmpVec, k2, uzIn, uzOut);

    for (size_t i = 0; i < n; i++) tmpVec[i] = uVec[i] + 0.5 * dt * k2[i];
    compute_rhs(mesh, tmpVec, k3, uzIn, uzOut);

    for (size_t i = 0; i < n; i++) tmpVec[i] = uVec[i] + dt * k3[i];
    compute_rhs(mesh, tmpVec, k4, uzIn, uzOut);

    for (size_t i = 0; i < n; i++)
        uVec[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);

    // After the RK4 update, orphan local CGs on this rank still hold
    // their pre-step value (no local element contributes via zip).
    // Pull the up-to-date value from a rank that has the position as
    // ghost-with-local-ref (where RK did update). Otherwise each
    // step leaves orphans lagging by one RK step's contribution.
    orphanPreGather(mesh, uVec);
}

// Redistribute a CG vector across a partition change. Both meshes must
// have the SAME element set (same TreeNodes); only rank assignments
// differ. We extract DG values per local element on `meshOld`, route
// each element's DG to the rank that owns the same TreeNode locally on
// `meshNew`, then write into vec_out via meshNew's E2N_CG.
template <typename T>
void redistributeAcrossPartition(ot::Mesh* meshOld, ot::Mesh* meshNew,
                                 const std::vector<T>& vecIn,
                                 std::vector<T>& vecOut, MPI_Comm comm) {
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    const unsigned int npe = meshNew->getNumNodesPerElement();
    double t0 = MPI_Wtime();

    // Step 1: each rank gathers all OTHER ranks' local TreeNodes from
    // meshNew, building a global TreeNode→rank map.
    std::vector<ot::TreeNode> myLocal;
    if (meshNew->isActive()) {
        const auto* pN = meshNew->getAllElements().data();
        for (unsigned int e = meshNew->getElementLocalBegin();
             e < meshNew->getElementLocalEnd(); e++)
            myLocal.push_back(pN[e]);
    }
    int myCount = (int)myLocal.size();

    std::vector<int> counts(npes), offs(npes, 0);
    MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int total = 0;
    for (int p = 0; p < npes; p++) {
        offs[p] = total;
        total += counts[p];
    }

    // Each TreeNode is sizeof(ot::TreeNode) bytes (~16 with x,y,z,level).
    std::vector<ot::TreeNode> allLocal(total);
    MPI_Allgatherv(myLocal.data(), myCount, par::Mpi_datatype<ot::TreeNode>::value(),
                   allLocal.data(), counts.data(), offs.data(),
                   par::Mpi_datatype<ot::TreeNode>::value(), comm);

    std::unordered_map<ot::TreeNode, int, TreeNodeHash> tnToRank;
    tnToRank.reserve((size_t)total);
    for (int p = 0; p < npes; p++) {
        for (int i = offs[p]; i < offs[p] + counts[p]; i++) {
            tnToRank.emplace(allLocal[i], p);
        }
    }
    double tMap1 = MPI_Wtime();

    // Step 2: walk meshOld's local elements, gather per-element DG
    // values from vecIn via E2N_CG. For each node, read vecIn at the
    // canonical owner's CG slot — that's the "source of truth" value
    // for that physical position under master's 1:1 convention. (We
    // don't apply hanging interpolation here; interpolation happens at
    // read time on the new mesh via its own getElementNodalValues.)
    std::vector<std::vector<ot::TreeNode>> sendTN(npes);
    std::vector<std::vector<T>> sendDG(npes);

    if (meshOld->isActive()) {
        const auto* pNO = meshOld->getAllElements().data();
        const auto& e2nO = meshOld->getE2NMapping();
        for (unsigned int e = meshOld->getElementLocalBegin();
             e < meshOld->getElementLocalEnd(); e++) {
            auto it = tnToRank.find(pNO[e]);
            if (it == tnToRank.end()) continue;
            int target = it->second;
            sendTN[target].push_back(pNO[e]);
            for (unsigned int n = 0; n < npe; n++) {
                sendDG[target].push_back(
                    (T)vecIn[e2nO[e * npe + n]]);
            }
        }
    }
    double tPack = MPI_Wtime();

    // Step 3: alltoallv for TreeNode list and DG values
    std::vector<int> sendCntTN(npes), sendCntDG(npes);
    std::vector<int> sendOffTN(npes, 0), sendOffDG(npes, 0);
    for (int p = 0; p < npes; p++) {
        sendCntTN[p] = (int)sendTN[p].size();
        sendCntDG[p] = (int)sendDG[p].size();
    }
    for (int p = 1; p < npes; p++) {
        sendOffTN[p] = sendOffTN[p - 1] + sendCntTN[p - 1];
        sendOffDG[p] = sendOffDG[p - 1] + sendCntDG[p - 1];
    }

    std::vector<int> recvCntTN(npes), recvCntDG(npes);
    std::vector<int> recvOffTN(npes, 0), recvOffDG(npes, 0);
    MPI_Alltoall(sendCntTN.data(), 1, MPI_INT, recvCntTN.data(), 1,
                 MPI_INT, comm);
    MPI_Alltoall(sendCntDG.data(), 1, MPI_INT, recvCntDG.data(), 1,
                 MPI_INT, comm);
    int totRecvTN = 0, totRecvDG = 0;
    for (int p = 0; p < npes; p++) {
        recvOffTN[p] = totRecvTN;
        recvOffDG[p] = totRecvDG;
        totRecvTN += recvCntTN[p];
        totRecvDG += recvCntDG[p];
    }

    std::vector<ot::TreeNode> flatTN;
    std::vector<T> flatDG;
    for (int p = 0; p < npes; p++) {
        flatTN.insert(flatTN.end(), sendTN[p].begin(), sendTN[p].end());
        flatDG.insert(flatDG.end(), sendDG[p].begin(), sendDG[p].end());
    }

    std::vector<ot::TreeNode> recvTN(totRecvTN);
    std::vector<T> recvDG(totRecvDG);
    MPI_Alltoallv(flatTN.data(), sendCntTN.data(), sendOffTN.data(),
                  par::Mpi_datatype<ot::TreeNode>::value(),
                  recvTN.data(), recvCntTN.data(), recvOffTN.data(),
                  par::Mpi_datatype<ot::TreeNode>::value(), comm);
    MPI_Alltoallv(flatDG.data(), sendCntDG.data(), sendOffDG.data(),
                  par::Mpi_datatype<T>::value(),
                  recvDG.data(), recvCntDG.data(), recvOffDG.data(),
                  par::Mpi_datatype<T>::value(), comm);
    double tAlltoall = MPI_Wtime();

    // Step 4: write into meshNew's CG via E2N_CG. Build TreeNode →
    // local idx for meshNew's local elements.
    meshNew->createVector(vecOut, (T)0);
    if (!meshNew->isActive()) return;
    std::unordered_map<ot::TreeNode, unsigned int, TreeNodeHash> tnToLocal;
    tnToLocal.reserve(
        (size_t)(meshNew->getElementLocalEnd() -
                 meshNew->getElementLocalBegin()));
    {
        const auto* pNN = meshNew->getAllElements().data();
        for (unsigned int e = meshNew->getElementLocalBegin();
             e < meshNew->getElementLocalEnd(); e++)
            tnToLocal.emplace(pNN[e], e);
    }

    const auto& e2n = meshNew->getE2NMapping();
    for (size_t i = 0; i < recvTN.size(); i++) {
        auto it = tnToLocal.find(recvTN[i]);
        if (it == tnToLocal.end()) continue;
        unsigned int e = it->second;
        for (unsigned int n = 0; n < npe; n++)
            vecOut[e2n[e * npe + n]] = recvDG[i * npe + n];
    }

    // Also populate meshNew's m_uiLocalNodalDG buffer (used by
    // performGhostExchange's DG-path). Each local element's nodal DG
    // values come from the same recvDG buffer keyed by TreeNode.
    {
        std::vector<double>& localDG = meshNew->getLocalNodalDGRef();
        const unsigned int numLocal =
            meshNew->getElementLocalEnd() - meshNew->getElementLocalBegin();
        localDG.assign(numLocal * npe, (double)0);
        for (size_t i = 0; i < recvTN.size(); i++) {
            auto it = tnToLocal.find(recvTN[i]);
            if (it == tnToLocal.end()) continue;
            unsigned int eLocal =
                it->second - meshNew->getElementLocalBegin();
            for (unsigned int n = 0; n < npe; n++)
                localDG[eLocal * npe + n] = (double)recvDG[i * npe + n];
        }
    }
    double tWrite = MPI_Wtime();
    if (rank == 0) {
        std::cout << "   [redist] agv=" << (tMap1 - t0) * 1000
                  << " pack=" << (tPack - tMap1) * 1000
                  << " a2av=" << (tAlltoall - tPack) * 1000
                  << " write=" << (tWrite - tAlltoall) * 1000 << " ms\n";
    }
}

// Compare two CG vectors at the same physical positions. Iterate
// LOCAL nodes only, decode physical via E2N_DG, query the other mesh
// via global lookup. Return max absolute difference.
double compare_meshes(ot::Mesh* meshA, ot::Mesh* meshB,
                      const std::vector<double>& uA,
                      const std::vector<double>& uB) {
    if (!meshA->isActive()) return 0;

    const ot::TreeNode* pNA = meshA->getAllElements().data();
    const ot::TreeNode* pNB = meshB->getAllElements().data();
    const unsigned int* e2nA = meshA->getE2NMapping().data();
    const unsigned int* e2nB = meshB->getE2NMapping().data();
    const unsigned int* dgA  = meshA->getE2NMapping_DG().data();
    const unsigned int* dgB  = meshB->getE2NMapping_DG().data();
    const unsigned int npe   = meshA->getNumNodesPerElement();
    const unsigned int eOrd  = meshA->getElementOrder();

    // Build a position→value map from meshB's local nodes
    std::map<std::tuple<unsigned int, unsigned int, unsigned int>, double>
        posMapB;
    for (unsigned int e = meshB->getElementLocalBegin();
         e < meshB->getElementLocalEnd(); e++) {
        for (unsigned int n = 0; n < npe; n++) {
            unsigned int cg = e2nB[e * npe + n];
            if (cg < meshB->getNodeLocalBegin() ||
                cg >= meshB->getNodeLocalEnd())
                continue;
            unsigned int dg = dgB[e * npe + n];
            unsigned int oe = dg / npe;
            unsigned int os = dg % npe;
            unsigned int len =
                1u << (m_uiMaxDepth - pNB[oe].getLevel());
            unsigned int x =
                pNB[oe].getX() + (os % (eOrd + 1)) * (len / eOrd);
            unsigned int y = pNB[oe].getY() +
                             ((os / (eOrd + 1)) % (eOrd + 1)) * (len / eOrd);
            unsigned int z = pNB[oe].getZ() +
                             (os / ((eOrd + 1) * (eOrd + 1))) * (len / eOrd);
            posMapB[{x, y, z}] = uB[cg];
        }
    }

    double maxDiff = 0;
    int    nMissing = 0;
    int    nChecked = 0;
    for (unsigned int e = meshA->getElementLocalBegin();
         e < meshA->getElementLocalEnd(); e++) {
        for (unsigned int n = 0; n < npe; n++) {
            unsigned int cg = e2nA[e * npe + n];
            if (cg < meshA->getNodeLocalBegin() ||
                cg >= meshA->getNodeLocalEnd())
                continue;
            unsigned int dg = dgA[e * npe + n];
            unsigned int oe = dg / npe;
            unsigned int os = dg % npe;
            unsigned int len =
                1u << (m_uiMaxDepth - pNA[oe].getLevel());
            unsigned int x =
                pNA[oe].getX() + (os % (eOrd + 1)) * (len / eOrd);
            unsigned int y = pNA[oe].getY() +
                             ((os / (eOrd + 1)) % (eOrd + 1)) * (len / eOrd);
            unsigned int z = pNA[oe].getZ() +
                             (os / ((eOrd + 1) * (eOrd + 1))) * (len / eOrd);
            auto it = posMapB.find({x, y, z});
            if (it == posMapB.end()) {
                nMissing++;
                continue;
            }
            double diff = std::abs(uA[cg] - it->second);
            if (diff > maxDiff) maxDiff = diff;
            nChecked++;
        }
    }
    return maxDiff;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (argc < 8) {
        if (!rank)
            std::cout << "Usage: " << argv[0]
                      << " partOpt maxDepth wavelet_tol part_tol "
                         "eOrder grainSz minDepth [steps] [dtFac] [remeshEvery]"
                      << std::endl;
        MPI_Finalize();
        return 0;
    }

    PartitioningOptions partitionOption =
        static_cast<PartitioningOptions>(atoi(argv[1]));
    m_uiMaxDepth                      = atoi(argv[2]);
    double wavelet_tol                = atof(argv[3]);
    double partition_tol              = atof(argv[4]);
    unsigned int eOrder               = atoi(argv[5]);
    unsigned int DENDRO_GRAIN_SZ      = atoi(argv[6]);
    unsigned int /*minDepth*/ unused0 = atoi(argv[7]);
    (void)unused0;
    int numSteps    = (argc > 8) ? atoi(argv[8]) : 50;
    double dtFactor = (argc > 9) ? atof(argv[9]) : 0.4;
    int remeshEvery = (argc > 10) ? atoi(argv[10]) : 0;
    double LOAD_IMB_TOL = 0.1;
    unsigned int SPLIT_FIX = 256;

    if (!rank) {
        std::cout << YLW << "partOpt=" << partitionOption
                  << " maxDepth=" << m_uiMaxDepth
                  << " wavelet_tol=" << wavelet_tol
                  << " eOrder=" << eOrder
                  << " steps=" << numSteps << " dtFac=" << dtFactor
                  << " remeshEvery=" << remeshEvery << NRM << std::endl;
    }

    _InitializeHcurve(m_uiDim);

    Point pt_min(DOM_MIN, DOM_MIN, DOM_MIN);
    Point pt_max(DOM_MAX, DOM_MAX, DOM_MAX);

    std::function<double(double, double, double)> initFunc =
        [](double xg, double yg, double zg) {
            return init_func(xg, yg, zg);
        };

    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(initFunc, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder,
                    comm);

    ot::Mesh* mesh_sfc = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FDM,
        DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
    ot::Mesh* mesh_repart = ot::createMesh(
        tmpNodes.data(), tmpNodes.size(), eOrder, comm, 1, ot::SM_TYPE::FDM,
        DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);

    mesh_sfc->setDomainBounds(pt_min, pt_max);
    mesh_repart->setDomainBounds(pt_min, pt_max);

    mesh_repart->setPartitioningMethod(partitionOption);
    mesh_repart->repartitionMeshGlobal();

    if (!rank) std::cout << "meshes built\n";

    // initial partition-quality snapshot (comparable SFC vs graph).
    mesh_sfc->dumpPartitionStats(std::cout, "sfc-initial");
    mesh_repart->dumpPartitionStats(std::cout, "graph-initial");

    auto runOne = [&](ot::Mesh*& mesh, const std::string& tag) {
        std::vector<double> u, tmp, k1, k2, k3, k4, uzIn, uzOut;
        mesh->createVector(u, initFunc);
        mesh->createVector(tmp);
        mesh->createVector(k1);
        mesh->createVector(k2);
        mesh->createVector(k3);
        mesh->createVector(k4);
        mesh->createUnZippedVector(uzIn);
        mesh->createUnZippedVector(uzOut);

        // estimate dt from CFL: heat eqn limit ~ h^2 / (6*nu)
        double hMin = (DOM_MAX - DOM_MIN) /
                      (double)(1u << m_uiMaxDepth) /
                      (double)eOrder;
        double dt = dtFactor * hMin * hMin / (6.0 * NU);

        if (!rank)
            std::cout << "[" << tag << "] dt=" << dt << "\n";

        // start the comms-volume tally fresh for this mesh's evolution.
        mesh->resetCommStats();

        for (int step = 1; step <= numSteps; step++) {
            rk4_step(mesh, u, tmp, k1, k2, k3, k4, uzIn, uzOut, dt);

            if (remeshEvery > 0 && step % remeshEvery == 0) {
                bool isGraph =
                    (partitionOption == PartitioningOptions::fastpart ||
                     partitionOption ==
                         PartitioningOptions::RandomPartition) &&
                    tag == "repart";

                double t0 = MPI_Wtime();
                // Pre-remesh L2 diag — captures state just after RK4
                // steps, before any partition/redistribute. If this
                // value matches between sfc and repart runs at the
                // same step, pre-remesh state is OK and the loss is in
                // the remesh+redistribute cycle itself.
                {
                    double preL2 = 0.0;
                    if (mesh->isActive())
                        for (unsigned int n = mesh->getNodeLocalBegin();
                             n < mesh->getNodeLocalEnd(); n++)
                            preL2 += u[n] * u[n];
                    double preL2_g = 0.0;
                    MPI_Allreduce(&preL2, &preL2_g, 1, MPI_DOUBLE, MPI_SUM,
                                  mesh->getMPIGlobalCommunicator());
                    if (!rank)
                        std::cout << "  [diag " << tag << " step=" << step
                                  << "] PRE-remesh localL2^2="
                                  << std::fixed << std::setprecision(6)
                                  << preL2_g << "\n";
                }
                // Sync ghost values on the current mesh BEFORE reading
                // nodal values in redistribute.
                orphanPreGather(mesh, u);
                mesh->performGhostExchange(u);
                double tGS = MPI_Wtime();

                // === ADAPTIVE REFINEMENT (flags set below, on the
                // mesh that ReMesh will actually read) ===
                unsigned int varIds[1] = {0};
                std::function<double(double, double, double, double*)>
                    waveletTolFunc =
                        [wavelet_tol](double, double, double, double*) {
                            return wavelet_tol;
                        };
                // For the SFC branch we run isReMeshUnzip here on the
                // mesh directly. For the graph branch we run it AFTER
                // converting to sfcTwin so that flags land on the mesh
                // ReMesh will read (conversion via createMesh + redistribute
                // doesn't carry flags).
                if (!isGraph) {
                    std::fill(uzIn.begin(), uzIn.end(), 0.0);
                    mesh->unzip(u.data(), uzIn.data());
                    const double* unzipVars[1] = {uzIn.data()};
                    mesh->isReMeshUnzip(unzipVars, varIds, 1,
                                         waveletTolFunc, 0.1);
                }
                // =============================

                // Graph path uses ReMeshRepartitioned: one createMesh +
                // one repartitionMeshGlobal + a TreeNode-keyed vec
                // redistribute. SFC path uses ReMesh + interGridTransfer
                // (the stock flag-based intergrid path).
                double tRM = 0, tS2G = 0;
                if (isGraph) {
                    // CORRECT graph-path for adaptive refinement (full
                    // sandwich, same pattern as NLSM integration):
                    //   0. If mesh is currently graph, convert to SFC
                    //      via createMesh + redistributeVec + flag
                    //      propagation. ReMesh/IGT require SFC-ordered
                    //      local elements.
                    //   1. ReMesh(SFC) -> newSfcMesh (refinement applied)
                    //   2. interGridTransfer: vec from SFC mesh to SFC
                    //      newMesh — populates NEW split-child elements
                    //      via prolongation, restricts coarsened parents
                    //   3. Build graph twin of newSfcMesh, redistribute
                    //      vec across the partition change
                    //
                    // NOTE: a shorter version using ReMeshRepartitioned +
                    // redistributeVec directly SKIPS step 2. For
                    // non-adaptive evolution that's fine, but adaptive
                    // wavelet flagging gives newly-split children value
                    // 0 because their TreeNodes don't exist in the
                    // source mesh for a TreeNode-keyed redistribute.

                    // Step 0: always do the graph→SFC conversion. When
                    // mesh is already SFC (first remesh) it's wasted work
                    // but correct; avoids subtle state carried on the
                    // mesh object between remeshes. This uses FDM (full
                    // mesh) so that the subsequent ReMesh call has the
                    // E2N it needs.
                    {
                        std::vector<ot::TreeNode> oct0;
                        if (mesh->isActive()) {
                            const auto* pN = mesh->getAllElements().data();
                            for (unsigned int e = mesh->getElementLocalBegin();
                                 e < mesh->getElementLocalEnd(); e++)
                                oct0.push_back(pN[e]);
                        }
                        ot::Mesh* sfcTwin = ot::createMesh(
                            oct0.data(), oct0.size(), eOrder, comm, 1,
                            ot::SM_TYPE::FDM, DENDRO_GRAIN_SZ,
                            LOAD_IMB_TOL, SPLIT_FIX);
                        sfcTwin->setDomainBounds(pt_min, pt_max);
                        std::vector<double> u_twin;
                        sfcTwin->createVector(u_twin, (double)0);
                        mesh->redistributeVec(sfcTwin, u.data(), u_twin.data());
                        delete mesh;
                        mesh = sfcTwin;
                        u    = std::move(u_twin);
                    }

                    // Set refinement flags on sfcTwin (the mesh we'll
                    // ReMesh). Flags aren't carried across createMesh +
                    // redistributeVec, so we re-run the wavelet here.
                    {
                        std::vector<double> uzTwin;
                        mesh->createUnZippedVector(uzTwin);
                        mesh->performGhostExchange(u);
                        std::fill(uzTwin.begin(), uzTwin.end(), 0.0);
                        mesh->unzip(u.data(), uzTwin.data());
                        const double* unzipVarsT[1] = {uzTwin.data()};
                        mesh->isReMeshUnzip(unzipVarsT, varIds, 1,
                                             waveletTolFunc, 0.1);
                    }

                    ot::Mesh* newSfcMesh = mesh->ReMesh(
                        DENDRO_GRAIN_SZ, LOAD_IMB_TOL, SPLIT_FIX);
                    std::vector<double> u_sfc_new;
                    if (newSfcMesh) {
                        newSfcMesh->setDomainBounds(pt_min, pt_max);
                        mesh->interGridTransfer(u, newSfcMesh);
                        // `u` is now on newSfcMesh (after IGT moved it).
                    }

                    // Build graph twin of newSfcMesh.
                    std::vector<ot::TreeNode> oct;
                    if (newSfcMesh->isActive()) {
                        const auto* pN = newSfcMesh->getAllElements().data();
                        for (unsigned int e = newSfcMesh->getElementLocalBegin();
                             e < newSfcMesh->getElementLocalEnd(); e++)
                            oct.push_back(pN[e]);
                    }
                    ot::Mesh* graphMesh = ot::createMesh(
                        oct.data(), oct.size(), eOrder, comm, 1,
                        ot::SM_TYPE::E2E_ONLY, DENDRO_GRAIN_SZ,
                        LOAD_IMB_TOL, SPLIT_FIX);
                    graphMesh->setDomainBounds(pt_min, pt_max);
                    graphMesh->setPartitioningMethod(partitionOption);
                    graphMesh->setScatterMapType(ot::SM_TYPE::FDM);
                    graphMesh->repartitionMeshGlobal();
                    tRM = MPI_Wtime();

                    // Sync ghost values on source BEFORE redistribute —
                    // redistributeVec iterates src's local elements and
                    // reads via E2N_CG which can point at ghost CG slots
                    // for boundary elements. If those slots are stale,
                    // wrong values propagate to dst.
                    newSfcMesh->performGhostExchange(u);

                    std::vector<double> u_new;
                    graphMesh->createVector(u_new, (double)0);
                    newSfcMesh->redistributeVec(graphMesh, u.data(),
                                                u_new.data());
                    // Sync ghost values — some local CG slots on
                    // graphMesh may only be referenced by ghost
                    // elements on this rank, so redistributeVec (which
                    // iterates src's LOCAL elements) won't have
                    // written to them. Ghost exchange propagates the
                    // owning-rank's values into those slots.
                    //
                    // Pre-gather orphan local CGs FIRST. redistribute
                    // wrote correct values to R_real's ghost slots
                    // (via matching TreeNode + sub), but if we run a
                    // plain ghost exchange now it will overwrite those
                    // correct ghost values with 0s read from orphan
                    // local CGs on the owning rank. The pre-gather
                    // pulls the correct value from R_real back to the
                    // orphan owner, so the subsequent exchange then
                    // re-broadcasts a consistent correct value.
                    orphanPreGather(graphMesh, u_new);
                    graphMesh->performGhostExchange(u_new);
                    delete newSfcMesh;
                    delete mesh;
                    mesh = graphMesh;
                    u    = std::move(u_new);
                } else {
                    ot::Mesh* newMesh = mesh->ReMesh(DENDRO_GRAIN_SZ,
                                                     LOAD_IMB_TOL, SPLIT_FIX);
                    if (newMesh) {
                        newMesh->setDomainBounds(pt_min, pt_max);
                        mesh->interGridTransfer(u, newMesh);
                        delete mesh;
                        mesh = newMesh;
                    }
                    tRM = MPI_Wtime();
                }
                tS2G = MPI_Wtime();

                // Always recreate state vectors after the remesh chain.
                mesh->createVector(tmp);
                mesh->createVector(k1);
                mesh->createVector(k2);
                mesh->createVector(k3);
                mesh->createVector(k4);
                mesh->createUnZippedVector(uzIn);
                mesh->createUnZippedVector(uzOut);

                // === DIAGNOSTICS ===
                // L2 of u over local CG slots only (ghost slots excluded).
                // Compare this between SFC and graph at matching steps:
                // if it diverges, redistribute is losing / duplicating
                // values. If local L2 matches but later a compare_meshes
                // check shows different u values at same TreeNode, the
                // issue is elsewhere (likely ghost exchange post-remesh).
                double localL2 = 0.0;
                if (mesh->isActive()) {
                    for (unsigned int n = mesh->getNodeLocalBegin();
                         n < mesh->getNodeLocalEnd(); n++)
                        localL2 += u[n] * u[n];
                }
                double globalL2 = 0.0;
                MPI_Allreduce(&localL2, &globalL2, 1, MPI_DOUBLE, MPI_SUM,
                              mesh->getMPIGlobalCommunicator());
                // Ghost-plus-local L2 (includes stale/newly-written ghost
                // slots). Diverging globally-total-L2 vs local-L2 reveals
                // ghost inconsistency.
                double totL2 = 0.0;
                if (mesh->isActive()) {
                    for (size_t n = 0; n < u.size(); n++)
                        totL2 += u[n] * u[n];
                }
                double totL2_g = 0.0;
                MPI_Allreduce(&totL2, &totL2_g, 1, MPI_DOUBLE, MPI_SUM,
                              mesh->getMPIGlobalCommunicator());
                // ===================

                double tEnd = MPI_Wtime();
                if (!rank) {
                    std::cout << std::fixed << std::setprecision(6)
                              << "  [diag " << tag << " step=" << step
                              << "] localL2^2=" << globalL2
                              << " allL2^2=" << totL2_g << "\n"
                              << std::setprecision(1)
                              << "  step " << step << " [" << tag
                              << "] remeshed -> "
                              << mesh->getNumLocalMeshElements()
                              << " ele/rank  ["
                              << "ghost=" << (tGS - t0) * 1000
                              << "ms remesh=" << (tRM - tGS) * 1000
                              << "ms redist=" << (tS2G - tRM) * 1000
                              << "ms realloc=" << (tEnd - tS2G) * 1000
                              << "ms  total=" << (tEnd - t0) * 1000
                              << "ms]" << std::scientific
                              << std::setprecision(6) << "\n";
                }
            }
        }

        // comms-volume tally + final partition snapshot for this run.
        mesh->dumpCommStats(std::cout, tag.c_str());
        mesh->dumpPartitionStats(std::cout, tag.c_str());

        return u;
    };

    auto u_sfc    = runOne(mesh_sfc, "sfc");
    auto u_repart = runOne(mesh_repart, "repart");

    // Sync ghost on both before comparing (so both have full local state).
    // Pre-gather orphans on the graph mesh so orphan local CGs are
    // refreshed before the final compare.
    orphanPreGather(mesh_sfc, u_sfc);
    mesh_sfc->performGhostExchange(u_sfc);
    orphanPreGather(mesh_repart, u_repart);
    mesh_repart->performGhostExchange(u_repart);

    double maxDiff =
        compare_meshes(mesh_sfc, mesh_repart, u_sfc, u_repart);
    double globalMax;
    MPI_Allreduce(&maxDiff, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    // Also report the L_inf norm of the solutions for context
    double localMax = 0;
    for (unsigned int e = mesh_sfc->getElementLocalBegin();
         e < mesh_sfc->getElementLocalEnd(); e++) {
        for (unsigned int n = 0; n < mesh_sfc->getNumNodesPerElement(); n++) {
            unsigned int cg =
                mesh_sfc->getE2NMapping()[e * mesh_sfc->getNumNodesPerElement() +
                                          n];
            if (cg >= mesh_sfc->getNodeLocalBegin() &&
                cg < mesh_sfc->getNodeLocalEnd())
                localMax = std::max(localMax, std::abs(u_sfc[cg]));
        }
    }
    double globalSolMax;
    MPI_Allreduce(&localMax, &globalSolMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (!rank) {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "================================\n";
        std::cout << "max |u_sfc - u_repart| = " << globalMax << "\n";
        std::cout << "max |u_sfc|            = " << globalSolMax << "\n";
        std::cout << "relative diff          = "
                  << (globalSolMax > 0 ? globalMax / globalSolMax : 0)
                  << "\n";
        std::cout << "================================\n";
        const double relDiff =
            globalMax / std::max(globalSolMax, 1e-30);
        const double perStep = relDiff / std::max(numSteps, 1);
        std::cout << "per-step drift        = " << perStep << "\n";
        if (perStep < 1e-6)
            std::cout << GRN
                      << "EVOLUTION MATCH (per-step drift < 1e-6)"
                      << (remeshEvery > 0 ? " [with remesh]" : "")
                      << NRM << "\n";
        else
            std::cout << RED
                      << "EVOLUTION DRIFT TOO LARGE"
                      << (remeshEvery > 0 ? " [with remesh]" : "")
                      << NRM << "\n";
    }

    delete mesh_sfc;
    delete mesh_repart;
    MPI_Finalize();
    return 0;
}
