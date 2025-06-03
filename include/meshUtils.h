/**
 * @file meshUtils.h
 * @author Milinda Fernando
 * @brief Contains utility function related to easily generate a mesh.
 * @version 0.1
 * @date 2020-01-01
 *
 * School of Computing, University of Utah.
 * @copyright Copyright (c) 2020
 *
 */

#pragma once
#include <functional>
#include <iostream>
#include <vector>

#include "asyncExchangeContex.h"
#include "dvec.h"
#include "mesh.h"
#include "octUtils.h"
#include "parUtils.h"
#include "waveletAMR.h"
namespace ot {

/**
 * @brief Create a Mesh object by an array of octants.
 *
 * @param oct : pointer to a list of octant.
 * @param num : number of
 * @param eleOrder: element order.
 * param comm: global communicator.
 * @param verbose: if > 0 prints additional infomation on the Mesh generation.
 * @param sm_type: scatter map type
 * @param grain_sz : grain size for the partitoning,
 * @param ld_tol : Load imbalance tolerance.
 * @param sf_k : splitter fix values.
 * @return Mesh mesh object.
 */
Mesh* createMesh(const ot::TreeNode* oct, unsigned int num,
                 unsigned int eleOrder, MPI_Comm comm, unsigned int verbose = 1,
                 ot::SM_TYPE sm_type   = ot::SM_TYPE::FDM,
                 unsigned int grain_sz = DENDRO_DEFAULT_GRAIN_SZ,
                 double ld_tol         = DENDRO_DEFAULT_LB_TOL,
                 unsigned int sf_k     = DENDRO_DEFAULT_SF_K,
                 unsigned int (*getWeight)(const ot::TreeNode*) = NULL);

/**
 * @brief Generates an adaptive mesh based on the Wavelet AMR
 *
 * @param func : spatially dependent function.
 * @param wtol : wavelet tolerance.
 * @param numVars : number of vars returned by the func.
 * @param eleOrder : element order
 * @param comm: global communicator.
 * @param refId : refinement var ids.
 * @param verbose: if > 0 prints additional infomation on the Mesh generation.
 * @param sm_type: scatter map type (FDM, FEM_CG, FEM_DG)
 * @param sz : number of refinement variables.
 * @param grain_sz : grains size,
 * @param ld_tol : load imbalance tolerance for flexible partitioning.
 * @param sf_k : splitter fix k value (for large case runs)
 * @return Mesh*
 */
Mesh* createWAMRMesh(std::function<void(double, double, double, double*)> func,
                     double wtol, unsigned int numVars, unsigned int eleOrder,
                     MPI_Comm comm, unsigned int verbose = 1,
                     ot::SM_TYPE sm_type  = ot::SM_TYPE::FDM,
                     unsigned int* refIds = NULL, unsigned int sz = 0,
                     unsigned int grain_sz = DENDRO_DEFAULT_GRAIN_SZ,
                     double ld_tol         = DENDRO_DEFAULT_LB_TOL,
                     unsigned int sf_k     = DENDRO_DEFAULT_SF_K);

/**
 * @brief Creates a mesh which is guranteed to converge
 * @param pMesh : Pointer to current mesh object
 * @param wtol : wavelet tolerance,
 * @param numVars : number of variables defined on the mesh.
 * @param eleOrder : element order.
 * @param refIds : refinment ids
 * @param sz : size of the refinement ids
 * @param maxiter : number of maximum iteration.
 */
void meshWAMRConvergence(
    ot::Mesh*& pMesh, std::function<void(double, double, double, double*)> func,
    double wtol, unsigned int numVars, unsigned int eleOrder,
    unsigned int* refIds = NULL, unsigned int sz = 0,
    unsigned int maxiter = 10);

/**
 * @brief computes block unzip ghost node dependancies.
 *
 * @param pMesh : pointer to the mesh object
 * @param blk : blk id
 * @param gid : vector of ghost node ids related to the unzip, if the block is
 * independent then the gid will have size zero.
 */
void computeBlockUnzipGhostNodes(const ot::Mesh* pMesh, unsigned int blk,
                                 std::vector<unsigned int>& gid);

/**
 * @brief computes the block unzip padding elements.
 * @param pMesh : pointer to the mesh object
 * @param blk : blk id
 * @param eid : vector of element ids.
 */
void computeBlockUnzipDepElements(const ot::Mesh* pMesh, unsigned int blk,
                                  std::vector<unsigned int>& eid);

/**
 * @brief compute the corresponding time step level for a given block id.
 *
 * @param pMesh : pointer to mesh object.
 * @param blk : local block id.
 * @return unsigned int : corresponding time
 */
unsigned int computeTLevel(const ot::Mesh* const pMesh, unsigned int blk);

/**
 * @brief slice the mesh
 * @param pMesh : mesh
 * @param s_val : point on the slice plain
 * @param s_normal : normal vector to the slice.
 * @param sids : elements on the slice
 * @return int
 */
int slice_mesh(const ot::Mesh* pMesh, unsigned int s_val[3],
               unsigned int s_normal[3], std::vector<unsigned int>& sids);

/**
 * @brief Create a Split Mesh object with x left will have lmax and x right will
 * have lmin level.
 *
 * @param eleOrder element order
 * @param lmin : level min
 * @param lmax : level max
 * @param comm : communicator mpi
 * @return Mesh*
 */
Mesh* createSplitMesh(unsigned int eleOrder, unsigned int lmin,
                      unsigned int lmax, MPI_Comm comm);

template <typename T>
void alloc_mpi_ctx(const Mesh* pMesh,
                   std::vector<AsyncExchangeContex>& ctx_list, int dof,
                   int async_k) {
    if (pMesh->getMPICommSizeGlobal() == 1 || !pMesh->isActive()) return;

    {
        const std::vector<unsigned int>& nodeSendCount =
            pMesh->getNodalSendCounts();
        const std::vector<unsigned int>& nodeSendOffset =
            pMesh->getNodalSendOffsets();

        const std::vector<unsigned int>& e_sf = pMesh->getElementSendOffsets();
        const std::vector<unsigned int>& e_sc = pMesh->getElementSendCounts();
        const std::vector<unsigned int>& e_rf = pMesh->getElementRecvOffsets();
        const std::vector<unsigned int>& e_rc = pMesh->getElementRecvCounts();

        const std::vector<unsigned int>& nodeRecvCount =
            pMesh->getNodalRecvCounts();
        const std::vector<unsigned int>& nodeRecvOffset =
            pMesh->getNodalRecvOffsets();

        const std::vector<unsigned int>& sendProcList =
            pMesh->getSendProcList();
        const std::vector<unsigned int>& recvProcList =
            pMesh->getRecvProcList();

        const std::vector<unsigned int>& sendNodeSM = pMesh->getSendNodeSM();
        const std::vector<unsigned int>& recvNodeSM = pMesh->getRecvNodeSM();

        const unsigned int activeNpes               = pMesh->getMPICommSize();
        const unsigned int nPe = pMesh->getNumNodesPerElement();

        const unsigned int sendBSzCg =
            (nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1]);
        const unsigned int recvBSzCg =
            (nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1]);

        const unsigned int sendBSzDg =
            (e_sf[activeNpes - 1] + e_sc[activeNpes - 1]) * nPe;
        const unsigned int recvBSzDg =
            (e_rf[activeNpes - 1] + e_rc[activeNpes - 1]) * nPe;

        const unsigned int sendBSz = std::max(sendBSzCg, sendBSzDg);
        const unsigned int recvBSz = std::max(recvBSzCg, recvBSzDg);

        ctx_list.resize(async_k);
        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);

            if (sendBSz)
                ctx_list[i].allocateSendBuffer(batch_sz * sendBSz * sizeof(T));

            if (recvBSz)
                ctx_list[i].allocateRecvBuffer(batch_sz * recvBSz * sizeof(T));

            ctx_list[i].m_send_req.resize(pMesh->getMPICommSize(),
                                          MPI_Request());
            ctx_list[i].m_recv_req.resize(pMesh->getMPICommSize(),
                                          MPI_Request());
        }
    }

    return;
}

template <typename T>
void dealloc_mpi_ctx(const Mesh* pMesh,
                     std::vector<AsyncExchangeContex>& ctx_list, int dof,
                     int async_k) {
    if (pMesh->getMPICommSizeGlobal() == 1 || !pMesh->isActive()) return;

    const std::vector<unsigned int>& nodeSendCount =
        pMesh->getNodalSendCounts();
    const std::vector<unsigned int>& nodeSendOffset =
        pMesh->getNodalSendOffsets();

    const std::vector<unsigned int>& nodeRecvCount =
        pMesh->getNodalRecvCounts();
    const std::vector<unsigned int>& nodeRecvOffset =
        pMesh->getNodalRecvOffsets();

    const std::vector<unsigned int>& sendProcList = pMesh->getSendProcList();
    const std::vector<unsigned int>& recvProcList = pMesh->getRecvProcList();

    const std::vector<unsigned int>& sendNodeSM   = pMesh->getSendNodeSM();
    const std::vector<unsigned int>& recvNodeSM   = pMesh->getRecvNodeSM();

    const unsigned int activeNpes                 = pMesh->getMPICommSize();

    const unsigned int sendBSz =
        nodeSendOffset[activeNpes - 1] + nodeSendCount[activeNpes - 1];
    const unsigned int recvBSz =
        nodeRecvOffset[activeNpes - 1] + nodeRecvCount[activeNpes - 1];

    for (unsigned int i = 0; i < ctx_list.size(); i++) {
        const unsigned int v_begin  = ((i * dof) / async_k);
        const unsigned int v_end    = (((i + 1) * dof) / async_k);
        const unsigned int batch_sz = (v_end - v_begin);

        if (sendBSz) ctx_list[i].deAllocateSendBuffer();

        if (recvBSz) ctx_list[i].deAllocateRecvBuffer();

        ctx_list[i].m_send_req.clear();
        ctx_list[i].m_recv_req.clear();
    }

    ctx_list.clear();
}

template <typename T>
inline T R_calc(const T comp_domain_max, const T comp_domain_min) {
    return comp_domain_max - comp_domain_min;
}

template <typename T>
inline T Rg_calc(const T octree_domain_max, const T octree_domain_min) {
    return octree_domain_max - octree_domain_min;
}

template <typename T>
inline T grid_to_domain(const T x, const T comp_domain_max,
                        const T comp_domain_min, const T octree_domain_max,
                        const T octree_domain_min) {
    return (((R_calc(comp_domain_max, comp_domain_min) /
              Rg_calc(octree_domain_max, octree_domain_min)) *
             (x - octree_domain_min)) +
            comp_domain_min);
}

template <typename T>
inline T grid_to_domain_x(const T x, const T comp_domain_max[3],
                          const T comp_domain_min[3],
                          const T octree_domain_max[3],
                          const T octree_domain_min[3]) {
    return grid_to_domain(x, comp_domain_max[0], comp_domain_min[0],
                          octree_domain_max[0], octree_domain_min[0]);
}

template <typename T>
inline T grid_to_domain_y(const T x, const T comp_domain_max[3],
                          const T comp_domain_min[3],
                          const T octree_domain_max[3],
                          const T octree_domain_min[3]) {
    return grid_to_domain(x, comp_domain_max[1], comp_domain_min[1],
                          octree_domain_max[1], octree_domain_min[1]);
}

template <typename T>
inline T grid_to_domain_z(const T x, const T comp_domain_max[3],
                          const T comp_domain_min[3],
                          const T octree_domain_max[3],
                          const T octree_domain_min[3]) {
    return grid_to_domain(x, comp_domain_max[2], comp_domain_min[2],
                          octree_domain_max[2], octree_domain_min[2]);
}

// better optimized
template <typename T>
struct PrecomputedRs {
    T R;
    T Rg;
    T odm;
    T cdm;
    T RoRg;
    T RgoRx;
};

template <typename T>
PrecomputedRs<T> precomputeR(const T comp_domain_max, const T comp_domain_min,
                             const T octree_domain_max,
                             const T octree_domain_min) {
    PrecomputedRs<T> ranges;
    ranges.R     = R_calc(comp_domain_max, comp_domain_min);
    ranges.Rg    = Rg_calc(octree_domain_max, octree_domain_min);
    ranges.odm   = octree_domain_min;
    ranges.cdm   = comp_domain_max;
    ranges.RoRg  = ranges.R / ranges.Rg;
    ranges.RgoRx = ranges.Rg / ranges.R;
    return ranges;
}

template <typename T>
inline T grid_to_domain(const T x, const PrecomputedRs<T>& r) {
    return (((r.RoRg) * (x - r.odm)) + r.cdm);
}

template <typename T>
inline T domain_to_grid(const T x, const PrecomputedRs<T>& r) {
    return (((r.RgoRx) * (x - r.cdm)) + r.odm);
}

template <typename T>
T normL2_volnorm(const Mesh* mesh, const T* vec, const T octree_domain_max[3],
                 const T octree_domain_min[3], const T comp_domain_max[3],
                 const T comp_domain_min[3]) {
    double l2_g = 0.0;

    PrecomputedRs<T> xr =
        precomputeR(comp_domain_max[0], comp_domain_min[0],
                    octree_domain_max[0], octree_domain_min[0]);
    PrecomputedRs<T> yr =
        precomputeR(comp_domain_max[1], comp_domain_min[1],
                    octree_domain_max[1], octree_domain_min[1]);
    PrecomputedRs<T> zr =
        precomputeR(comp_domain_max[2], comp_domain_min[2],
                    octree_domain_max[2], octree_domain_min[2]);

    if (mesh->isActive()) {
        MPI_Comm comm                     = mesh->getMPICommunicator();
        const unsigned int eleLocalBegin  = mesh->getElementLocalBegin();
        const unsigned int eleLocalEnd    = mesh->getElementLocalEnd();

        const unsigned int nodeLocalBegin = mesh->getNodeLocalBegin();
        const unsigned int nodeLocalEnd   = mesh->getNodeLocalEnd();
        const unsigned int nPe            = mesh->getNumNodesPerElement();

        const unsigned int* e2n_cg        = &(*(mesh->getE2NMapping().begin()));
        const unsigned int* e2n_dg  = &(*(mesh->getE2NMapping_DG().begin()));
        const unsigned int eleOrder = mesh->getElementOrder();
        const ot::TreeNode* pNodes  = mesh->getAllElements().data();

        double l2                   = 0.0;

        DendroIntL localGridPts     = 0;
        DendroIntL globalGridPts    = 0;
        std::vector<bool> accumulated;
        accumulated.resize(mesh->getDegOfFreedom(), false);

        for (unsigned int elem = eleLocalBegin; elem < eleLocalEnd; elem++) {
            for (unsigned int k = 0; k < (eleOrder + 1); k++)
                for (unsigned int j = 0; j < (eleOrder + 1); j++)
                    for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                        const unsigned int nodeLookUp_CG =
                            e2n_cg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        if ((nodeLookUp_CG >= nodeLocalBegin &&
                             nodeLookUp_CG < nodeLocalEnd) &&
                            !(accumulated[nodeLookUp_CG])) {
                            const unsigned int nodeLookUp_DG =
                                e2n_dg[elem * nPe +
                                       k * (eleOrder + 1) * (eleOrder + 1) +
                                       j * (eleOrder + 1) + i];
                            unsigned int ownerID, ii_x, jj_y, kk_z;

                            mesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);
                            const double len =
                                (double)(1u << (m_uiMaxDepth -
                                                pNodes[ownerID].getLevel()));

                            const double x_ot =
                                pNodes[ownerID].getX() +
                                ii_x * (len / ((double)eleOrder));
                            const double y_ot =
                                pNodes[ownerID].getY() +
                                jj_y * (len / ((double)eleOrder));
                            const double z_ot =
                                pNodes[ownerID].getZ() +
                                kk_z * (len / ((double)eleOrder));

                            const double dall_ot  = len / ((double)eleOrder);
                            // then we can convert directly based on the scaling
                            // to the dx, dy, and dz
                            const double dx       = dall_ot * xr.RoRg;
                            const double dy       = dall_ot * yr.RoRg;
                            const double dz       = dall_ot * zr.RoRg;

                            const double vol_term = dx * dy * dz;

                            const double x        = grid_to_domain(x_ot, xr);
                            const double y        = grid_to_domain(y_ot, yr);
                            const double z        = grid_to_domain(z_ot, zr);

                            Point grid_pt(x, y, z);

                            // l2 is then calculated like this:
                            // NOTE: that we're scaling the l2 term by the
                            // volume in dx/dy/dz
                            l2 += (vec[nodeLookUp_CG] * vec[nodeLookUp_CG]) *
                                  vol_term;
                            accumulated[nodeLookUp_CG] = true;
                            localGridPts++;
                        }
                    }
        }

        par::Mpi_Reduce(&l2, &l2_g, 1, MPI_SUM, 0, mesh->getMPICommunicator());
        // par::Mpi_Reduce(&localGridPts, &globalGridPts, 1, MPI_SUM, 0,
        //                 mesh->getMPICommunicator());

        // then divide the full reduction by the total volume:
        const double volume_total = (comp_domain_max[0] - comp_domain_min[0]) *
                                    (comp_domain_max[1] - comp_domain_min[1]) *
                                    (comp_domain_max[2] - comp_domain_min[2]);

        if (!(mesh->getMPIRank())) {
            l2_g = l2_g / volume_total;
        }
    }

    return sqrt(l2_g);
}

}  // namespace ot
