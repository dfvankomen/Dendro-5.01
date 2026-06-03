//
// Created by milinda on 9/2/16.
//

/**
 * @author: Milinda Shayamal Fernando
 * School of Computing , University of Utah
 *
 * @breif Contains the functions to generate the mesh data structure from the
 * 2:1 balanced linear octree.
 *
 * Assumptions:
 * 1). Assumes that octree is balanced and sorted.
 * 2). Assumes that there is no duplicate nodes.
 *
 *
 * Communicator switch assumptions.
 * 1). global rank=0 is always active.
 * 2). communicator split should be always contigious.
 *
 *
 *
 * */

#include "mesh.h"

#include "TreeNode.h"
#include "dendro.h"
#include "logger.h"
#include "octUtils.h"

// OpenMP threading in graph-partitioning hot paths is opt-in via the
// CMake flag DENDRO_ENABLE_OPENMP_PARTITIONING (which sets
// DENDRO_OMP_PART). DENDRO_OMP_PRAGMA(x) expands to `_Pragma(#x)` only
// when both DENDRO_OMP_PART and _OPENMP are defined; otherwise it
// expands to nothing, so the default build is byte-identical to a
// codebase with no OMP annotations at all.
#if defined(DENDRO_OMP_PART) && defined(_OPENMP)
  #include <omp.h>
  #define DENDRO_OMP_PRAGMA(x) _Pragma(#x)
  #define DENDRO_OMP_ACTIVE 1
#else
  #define DENDRO_OMP_PRAGMA(x)
  #define DENDRO_OMP_ACTIVE 0
#endif
double t_e2e;  // e2e map generation time
double t_e2n;  // e2n map generation time
double t_sm;   // sm map generation time
double t_blk;  // perform blk setup time.

double t_e2e_g[3];
double t_e2n_g[3];
double t_sm_g[3];
double t_blk_g[3];

// #define DEBUG_E2N_MAPPING_SM
// #define DEBUG_MESH_GENERATION

namespace ot {

namespace {
// scaled-int (x, y, z) phys-position key used by audit, buildZipPlan,
// and getElementNodalValues for unordered_map<phys, ...> lookups.
struct PhysKey3 {
    unsigned long long x, y, z;
    bool operator==(const PhysKey3& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};
struct PhysKey3Hash {
    size_t operator()(const PhysKey3& k) const {
        size_t h = std::hash<unsigned long long>()(k.x);
        h ^= std::hash<unsigned long long>()(k.y)
            + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<unsigned long long>()(k.z)
            + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
}  // anonymous namespace

Mesh::Mesh(std::vector<ot::TreeNode> &in, unsigned int k_s, unsigned int pOrder,
           unsigned int activeNpes, MPI_Comm comm, bool pBlockSetup,
           SM_TYPE smType, unsigned int grainSz, double ld_tol,
           unsigned int sf_k) {
    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Now constructing mesh object");
    m_uiCommGlobal     = comm;
    m_uiIsBlockSetup   = pBlockSetup;
    m_uiScatterMapType = smType;
    m_uiIsF2ESetup     = false;
    MPI_Comm_rank(m_uiCommGlobal, &m_uiGlobalRank);
    MPI_Comm_size(m_uiCommGlobal, &m_uiGlobalNpes);

    DendroIntL localSz = in.size();
    DendroIntL globalSz;

    par::Mpi_Allreduce(&localSz, &globalSz, 1, MPI_SUM, m_uiCommGlobal);

    m_uiIsActive = m_uiGlobalRank < activeNpes;

    par::splitComm2way(m_uiIsActive, &m_uiCommActive, m_uiCommGlobal);

    m_uiMeshDomain_min = 0;
    m_uiMeshDomain_max = (1u << (m_uiMaxDepth));

    m_uiStensilSz      = k_s;
    m_uiElementOrder   = pOrder;
    m_uiEL_i           = 0;

    if (m_uiDim == 2)
        m_uiNpE = (m_uiElementOrder + 1) * (m_uiElementOrder + 1);
    else if (m_uiDim == 3)
        m_uiNpE = (m_uiElementOrder + 1) * (m_uiElementOrder + 1) *
                  (m_uiElementOrder + 1);

    m_uiNumDirections = (1u << m_uiDim) - 2 * (m_uiDim - 2);

    m_uiRefEl         = RefElement(m_uiDim, m_uiElementOrder);

    if (!m_uiIsActive) {  // set internal data structures for inactive mesh
                          // instances.

        assert(in.size() == 0);
        if (in.size() != 0) {
            std::cout << "[COMM Shrink/Expansion error]: global_rank: "
                      << m_uiGlobalRank << " is active: " << m_uiIsActive
                      << " balOct sz: " << in.size() << std::endl;
            exit(0);
        }

        m_uiElementPreGhostBegin  = 0;
        m_uiElementPreGhostEnd    = 0;
        m_uiElementLocalBegin     = 0;
        m_uiElementLocalEnd       = 0;
        m_uiElementPostGhostBegin = 0;
        m_uiElementPostGhostEnd   = 0;

        m_uiNumPreGhostElements =
            m_uiElementPreGhostEnd - m_uiElementPreGhostBegin;
        m_uiNumLocalElements = m_uiElementLocalEnd = m_uiElementLocalBegin;
        m_uiNumPostGhostElements =
            m_uiElementPostGhostEnd - m_uiElementPostGhostBegin;

        m_uiNodePreGhostBegin  = 0;
        m_uiNodePreGhostEnd    = 0;
        m_uiNodeLocalBegin     = 0;
        m_uiNodeLocalEnd       = 0;
        m_uiNodePostGhostBegin = 0;
        m_uiNodePostGhostEnd   = 0;

        m_uiNumActualNodes     = 0;
        m_uiUnZippedVecSz      = 0;

        m_uiAllElements.clear();
        m_uiLocalSplitterElements.clear();
        m_uiAllLocalNode.clear();
        m_uiE2NMapping_CG.clear();
        m_uiE2NMapping_DG.clear();
        m_uiE2EMapping.clear();
        m_uiLocalBlockList.clear();

    } else {
        MPI_Comm_rank(m_uiCommActive, &m_uiActiveRank);
        MPI_Comm_size(m_uiCommActive, &m_uiActiveNpes);

        if (!m_uiActiveRank)
            std::cout << " [MPI_COMM_SWITCH]: Selected comm.size: "
                      << m_uiActiveNpes << " also hi" << std::endl;

        if (in.size() <= 1) {
            std::cout << "rank: " << m_uiActiveRank << " input octree of size "
                      << in.size() << " is too small for the current comm.  "
                      << std::endl;
            exit(0);
        }

        /*m_uiElementOrder=pOrder;
        m_uiRefEl=RefElement(1,m_uiElementOrder);*/

        double t_e2e_begin = MPI_Wtime();
        if (m_uiActiveNpes > 1)
            buildE2EMap(in, m_uiCommActive);
        else
            buildE2EMap(in);
        double t_e2e_end = MPI_Wtime();
        t_e2e            = t_e2e_end - t_e2e_begin;

        if (smType == SM_TYPE::E2E_ONLY) return;

        double t_e2n_begin = MPI_Wtime();
        if (smType == SM_TYPE::FDM) {
            // build the scatter map for finite difference computations
            buildE2NWithSM();
        } else if (smType == SM_TYPE::FEM_CG) {
            buildE2NWithSM();

        } else if (m_uiScatterMapType == SM_TYPE::FEM_DG) {
            buildE2N_DG();
        }

        double t_e2n_end  = MPI_Wtime();
        t_e2n             = t_e2n_end - t_e2n_begin;

        double t_sm_begin = MPI_Wtime();
        // if(m_uiActiveNpes>1)computeNodeScatterMaps(m_uiCommActive);
        // if(m_uiActiveNpes>1)computeNodalScatterMap(m_uiCommActive);
        // if(m_uiActiveNpes>1)computeNodalScatterMap1(m_uiCommActive);
        // if(m_uiActiveNpes>1)computeNodalScatterMap2(m_uiCommActive);
        if (m_uiActiveNpes > 1) {
            if (m_uiScatterMapType == SM_TYPE::FEM_DG) {
                // elemental scatter map.
                computeNodalScatterMapDG(m_uiCommActive);
                buildF2EMap();
            }
            // nodal scatter map is build above.
            // else
            // computeNodalScatterMap4(m_uiCommActive);
        }

        double t_sm_end    = MPI_Wtime();
        t_sm               = t_sm_end - t_sm_begin;

        double t_blk_begin = MPI_Wtime();
        if (m_uiIsBlockSetup) {
            performBlocksSetup(m_uiCoarsetBlkLev, NULL, 0);
            // computeSMSpecialPts();
            buildE2BlockMap();
            buildUnzipCanonicalWriterTable();
            buildZipPlan();
            // canonical block decomposition is correct on the SFC mesh.
            // stamp each local element with its block's anchor + meta
            // so this info can ride with the element through any
            // future partition exchange.
            deriveBlockInfoFromBlocks();
        }

        double t_blk_end = MPI_Wtime();
        t_blk            = t_blk_end - t_blk_begin;

#ifdef __PROFILE_MESH__
        par::computeOverallStats(&t_e2e, t_e2e_g, m_uiCommActive, "mesh e2e ");
        par::computeOverallStats(&t_e2n, t_e2n_g, m_uiCommActive,
                                 "mesh e2n (+ sm for fdm type) ");
        par::computeOverallStats(&t_sm, t_sm_g, m_uiCommActive,
                                 "mesh sm (DG type) ");
        par::computeOverallStats(&t_blk, t_blk_g, m_uiCommActive,
                                 "block setup ");
#endif

        if (m_uiActiveNpes > 1) {
            for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
                if (m_uiSendNodeCount[p] != 0) m_uiSendProcList.push_back(p);

                if (m_uiRecvNodeCount[p] != 0) m_uiRecvProcList.push_back(p);
            }

            m_uiSendBufferNodes.resize(m_uiSendNodeOffset[m_uiActiveNpes - 1] +
                                       m_uiSendNodeCount[m_uiActiveNpes - 1]);
            m_uiRecvBufferNodes.resize(m_uiRecvNodeOffset[m_uiActiveNpes - 1] +
                                       m_uiRecvNodeCount[m_uiActiveNpes - 1]);
        }

        // release comm counter memory
        if (m_uiActiveNpes > 1) {
            delete[] m_uiSplitterNodes;
            delete[] m_uiSendKeyCount;
            delete[] m_uiSendKeyOffset;
            delete[] m_uiSendOctCountRound1;
            delete[] m_uiSendOctOffsetRound1;
            delete[] m_uiSendOctCountRound2;
            delete[] m_uiSendOctOffsetRound2;

            delete[] m_uiSendKeyDiagCount;
            delete[] m_uiRecvKeyDiagCount;
            delete[] m_uiSendKeyDiagOffset;
            delete[] m_uiRecvKeyDiagOffset;

            delete[] m_uiSendOctCountRound1Diag;
            delete[] m_uiRecvOctCountRound1Diag;
            delete[] m_uiSendOctOffsetRound1Diag;
            delete[] m_uiRecvOctOffsetRound1Diag;

            delete[] m_uiRecvKeyCount;
            delete[] m_uiRecvKeyOffset;
            delete[] m_uiRecvOctCountRound1;
            delete[] m_uiRecvOctOffsetRound1;
            delete[] m_uiRecvOctCountRound2;
            delete[] m_uiRecvOctOffsetRound2;

            // note: @milinda I have moved sencNode count and offsets to
            // std::vector<unsigned int > because we need them to perform the
            // ghost exchange.
        }
    }

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished building the mesh!");
}

Mesh::Mesh(std::vector<ot::TreeNode> &in, unsigned int k_s, unsigned int pOrder,
           MPI_Comm comm, bool pBlockSetup, SM_TYPE smType,
           unsigned int grainSz, double ld_tol, unsigned int sf_k,
           unsigned int (*getWeight)(const ot::TreeNode *),
           unsigned int *blk_tags, unsigned int blk_tags_sz) {
    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Now constructing mesh object");
    m_uiCommGlobal     = comm;
    m_uiIsBlockSetup   = pBlockSetup;
    m_uiScatterMapType = smType;
    m_uiIsF2ESetup     = false;
    // now m_uiCoarsetBlkLev set by the cmake;
    // m_uiCoarsetBlkLev = 0;
    MPI_Comm_rank(m_uiCommGlobal, &m_uiGlobalRank);
    MPI_Comm_size(m_uiCommGlobal, &m_uiGlobalNpes);

    DendroIntL localSz = in.size();
    DendroIntL globalSz;

    par::Mpi_Allreduce(&localSz, &globalSz, 1, MPI_SUM, m_uiCommGlobal);
    int p_npes      = std::max(globalSz / grainSz, (DendroIntL)1);
    int p_npes_prev = binOp::getPrevHighestPowerOfTwo(p_npes);
    int p_npes_next = binOp::getNextHighestPowerOfTwo(p_npes);

    // if(!m_uiGlobalRank) std::cout<<"p_npes_prev: "<<p_npes_prev<<"
    // p_npes_next: "<<p_npes_next<<" p_npes: "<<p_npes<<" diff1:
    // "<<std::abs(p_npes_prev-p_npes)<<" diff2:
    // "<<std::abs(p_npes_next-p_npes)<<std::endl;
    (std::abs(p_npes_prev - p_npes) <= std::abs(p_npes_next - p_npes))
        ? p_npes = p_npes_prev
        : p_npes = p_npes_next;

    if (p_npes > m_uiGlobalNpes) p_npes = m_uiGlobalNpes;
    // quick fix to enforce the npes>=2 for any given grain size.
    if (p_npes <= 1 && m_uiGlobalNpes > 1) p_npes = 2;
    if (p_npes == m_uiGlobalNpes) {
        // m_uiCommActive=m_uiCommGlobal; // note : use MPI_Comm_dup which is
        // more safe than the assignment operator. (and MPI_Comm_free is always
        // possible for m_uiCommActive)
        MPI_Comm_dup(m_uiCommGlobal, &m_uiCommActive);
        m_uiIsActive = true;
    } else {
        assert(p_npes < m_uiGlobalNpes);
        // m_uiIsActive=(m_uiGlobalRank<(globalSz/grainSz));
        // m_uiIsActive=(m_uiGlobalRank<p_npes);
        m_uiIsActive = isRankSelected(m_uiGlobalNpes, m_uiGlobalRank, p_npes);
        par::splitComm2way(m_uiIsActive, &m_uiCommActive, m_uiCommGlobal);
    }

    shrinkOrExpandOctree(in, ld_tol, sf_k, m_uiIsActive, m_uiCommActive,
                         m_uiCommGlobal, getWeight);

    m_uiMeshDomain_min = 0;
    m_uiMeshDomain_max = (1u << (m_uiMaxDepth));

    m_uiStensilSz      = k_s;
    m_uiElementOrder   = pOrder;
    m_uiEL_i           = 0;

    if (m_uiDim == 2)
        m_uiNpE = (m_uiElementOrder + 1) * (m_uiElementOrder + 1);
    else if (m_uiDim == 3)
        m_uiNpE = (m_uiElementOrder + 1) * (m_uiElementOrder + 1) *
                  (m_uiElementOrder + 1);

    m_uiNumDirections = (1u << m_uiDim) - 2 * (m_uiDim - 2);

    m_uiRefEl         = RefElement(m_uiDim, m_uiElementOrder);

    if (!m_uiIsActive) {  // set internal data structures for inactive mesh
                          // instances.

        assert(in.size() == 0);
        if (in.size() != 0) {
            std::cout << "[COMM Shrink/Expansion error]: global_rank: "
                      << m_uiGlobalRank << " is active: " << m_uiIsActive
                      << " balOct sz: " << in.size() << std::endl;
            exit(0);
        }

        m_uiElementPreGhostBegin  = 0;
        m_uiElementPreGhostEnd    = 0;
        m_uiElementLocalBegin     = 0;
        m_uiElementLocalEnd       = 0;
        m_uiElementPostGhostBegin = 0;
        m_uiElementPostGhostEnd   = 0;

        m_uiNumPreGhostElements =
            m_uiElementPreGhostEnd - m_uiElementPreGhostBegin;
        m_uiNumLocalElements = m_uiElementLocalEnd = m_uiElementLocalBegin;
        m_uiNumPostGhostElements =
            m_uiElementPostGhostEnd - m_uiElementPostGhostBegin;

        m_uiNodePreGhostBegin  = 0;
        m_uiNodePreGhostEnd    = 0;
        m_uiNodeLocalBegin     = 0;
        m_uiNodeLocalEnd       = 0;
        m_uiNodePostGhostBegin = 0;
        m_uiNodePostGhostEnd   = 0;

        m_uiNumActualNodes     = 0;
        m_uiUnZippedVecSz      = 0;

        m_uiAllElements.clear();
        m_uiLocalSplitterElements.clear();
        m_uiAllLocalNode.clear();
        m_uiE2NMapping_CG.clear();
        m_uiE2NMapping_DG.clear();
        m_uiE2EMapping.clear();
        m_uiLocalBlockList.clear();

    } else {
        MPI_Comm_rank(m_uiCommActive, &m_uiActiveRank);
        MPI_Comm_size(m_uiCommActive, &m_uiActiveNpes);

        if (!m_uiActiveRank)
            std::cout << " [MPI_COMM_SWITCH]: Selected comm.size: "
                      << m_uiActiveNpes << " inside second mesh option "
                      << std::endl;

        if (in.size() <= 1) {
            std::cout << "rank: " << m_uiActiveRank << " input octree of size "
                      << in.size() << " is too small for the current comm.  "
                      << std::endl;
            exit(0);
        }

        /*m_uiElementOrder=pOrder;
        m_uiRefEl=RefElement(1,m_uiElementOrder);*/

        double t_e2e_begin = MPI_Wtime();
        if (m_uiActiveNpes > 1)
            buildE2EMap(in, m_uiCommActive);
        else
            buildE2EMap(in);
        double t_e2e_end = MPI_Wtime();
        t_e2e            = t_e2e_end - t_e2e_begin;

        if (smType == SM_TYPE::E2E_ONLY) return;

        double t_e2n_begin = MPI_Wtime();

        if (m_uiScatterMapType == SM_TYPE::FDM) {
            // build the scatter map for finite difference computations
            buildE2NWithSM();

        } else if (m_uiScatterMapType == SM_TYPE::FEM_CG) {
            buildE2NWithSM();
        } else if (m_uiScatterMapType == SM_TYPE::FEM_DG) {
            buildE2N_DG();
        }

        double t_e2n_end  = MPI_Wtime();
        t_e2n             = t_e2n_end - t_e2n_begin;

        double t_sm_begin = MPI_Wtime();
        // if(m_uiActiveNpes>1)computeNodeScatterMaps(m_uiCommActive);
        // if(m_uiActiveNpes>1)computeNodalScatterMap(m_uiCommActive);
        // if(m_uiActiveNpes>1)computeNodalScatterMap1(m_uiCommActive);
        // if(m_uiActiveNpes>1)computeNodalScatterMap2(m_uiCommActive);
        if (m_uiActiveNpes > 1) {
            if (m_uiScatterMapType == SM_TYPE::FEM_DG) {
                computeNodalScatterMapDG(m_uiCommActive);
                buildF2EMap();  // this is more of the elemental scatter map.
            }
            // Note: that the scatter map is updated from above.
            // else
            // computeNodalScatterMap4(m_uiCommActive);
        }

        double t_sm_end    = MPI_Wtime();
        t_sm               = t_sm_end - t_sm_begin;

        double t_blk_begin = MPI_Wtime();

        if (m_uiIsBlockSetup) {
            performBlocksSetup(m_uiCoarsetBlkLev, blk_tags, blk_tags_sz);
            // computeSMSpecialPts();
            buildE2BlockMap();
            buildUnzipCanonicalWriterTable();
            buildZipPlan();
            deriveBlockInfoFromBlocks();
        }

        double t_blk_end = MPI_Wtime();
        t_blk            = t_blk_end - t_blk_begin;

#ifdef __PROFILE_MESH__
        par::computeOverallStats(&t_e2e, t_e2e_g, m_uiCommActive, "mesh e2e ");
        par::computeOverallStats(&t_e2n, t_e2n_g, m_uiCommActive,
                                 "mesh e2n (+ sm for fdm type) ");
        par::computeOverallStats(&t_sm, t_sm_g, m_uiCommActive,
                                 "mesh sm (DG type) ");
        par::computeOverallStats(&t_blk, t_blk_g, m_uiCommActive,
                                 "block setup ");
#endif

        if (m_uiActiveNpes > 1) {
            for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
                if (m_uiSendNodeCount[p] != 0) m_uiSendProcList.push_back(p);

                if (m_uiRecvNodeCount[p] != 0) m_uiRecvProcList.push_back(p);
            }

            m_uiSendBufferNodes.resize(m_uiSendNodeOffset[m_uiActiveNpes - 1] +
                                       m_uiSendNodeCount[m_uiActiveNpes - 1]);
            m_uiRecvBufferNodes.resize(m_uiRecvNodeOffset[m_uiActiveNpes - 1] +
                                       m_uiRecvNodeCount[m_uiActiveNpes - 1]);
        }

        // release comm counter memory
        if (m_uiActiveNpes > 1) {
            delete[] m_uiSplitterNodes;
            delete[] m_uiSendKeyCount;
            delete[] m_uiSendKeyOffset;
            delete[] m_uiSendOctCountRound1;
            delete[] m_uiSendOctOffsetRound1;
            delete[] m_uiSendOctCountRound2;
            delete[] m_uiSendOctOffsetRound2;

            delete[] m_uiSendKeyDiagCount;
            delete[] m_uiRecvKeyDiagCount;
            delete[] m_uiSendKeyDiagOffset;
            delete[] m_uiRecvKeyDiagOffset;

            delete[] m_uiSendOctCountRound1Diag;
            delete[] m_uiRecvOctCountRound1Diag;
            delete[] m_uiSendOctOffsetRound1Diag;
            delete[] m_uiRecvOctOffsetRound1Diag;

            delete[] m_uiRecvKeyCount;
            delete[] m_uiRecvKeyOffset;
            delete[] m_uiRecvOctCountRound1;
            delete[] m_uiRecvOctOffsetRound1;
            delete[] m_uiRecvOctCountRound2;
            delete[] m_uiRecvOctOffsetRound2;

            // note: @milinda I have moved sencNode count and offsets to
            // std::vector<unsigned int > because we need them to perform the
            // ghost exchange.
        }
    }

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished building the mesh!");
}

Mesh::~Mesh() {
    m_uiPreGhostOctants.clear();
    m_uiPostGhostOctants.clear();

    m_uiEmbeddedOctree.clear();
    m_uiGhostOctants.clear();
    m_uiAllElements.clear();

    m_uiE2NMapping_CG.clear();
    m_uiE2NMapping_DG.clear();
    m_uiE2EMapping.clear();

    m_uiSendBufferElement.clear();
    m_uiScatterMapElementRound1.clear();
    m_uiSendBufferNodes.clear();
    m_uiRecvBufferNodes.clear();
    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();

    m_uiSendNodeOffset.clear();
    m_uiSendNodeCount.clear();
    m_uiRecvNodeOffset.clear();
    m_uiRecvNodeCount.clear();
    m_uiLocalSplitterElements.clear();

    m_uiSendProcList.clear();
    m_uiRecvProcList.clear();
    m_uiE2BlkMap.clear();

    if (m_uiCommActive != MPI_COMM_NULL && m_uiCommActive != MPI_COMM_WORLD) {
        MPI_Comm_free(&m_uiCommActive);
        m_uiCommActive = MPI_COMM_NULL;
    }
}

void Mesh::generateSearchKeys() {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    std::vector<SearchKey> skeys;
    std::vector<SearchKey>::iterator hint;
    ot::TreeNode *inPtr     = (&(*(m_uiEmbeddedOctree.begin())));
    unsigned int domain_max = 1u << (m_uiMaxDepth);
    SearchKey skey;
    DendroRegister unsigned int mySz;
    DendroRegister unsigned int myX;
    DendroRegister unsigned int myY;
    DendroRegister unsigned int myZ;
    DendroRegister unsigned int myLev;
    const unsigned int K = 1;

    for (int i = 0; i < m_uiEmbeddedOctree.size(); i++) {
        myLev = m_uiEmbeddedOctree[i].getLevel();

        mySz  = (1u << (m_uiMaxDepth - myLev));
        myX   = inPtr[i].getX();
        myY   = inPtr[i].getY();
        myZ   = inPtr[i].getZ();

        /* Below orientation is used when generating keys.
         *
         * [up]
         * Y
         * |     Z [front]
         * |    /
         * |   /
         * |  /
         * | /
         * -------------> X [right]
         */
        // Key generation along X axis.
        if ((myX + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX + K * mySz), myY, myZ, m_uiMaxDepth,
                                       m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT);
        }
        if (myX > 0) {
            hint = skeys.emplace(skeys.end(),
                                 SearchKey((myX - 1), myY, myZ, m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT);
        }

        // Key generation along Y axis.
        if ((myY + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey(myX, (myY + K * mySz), myZ, m_uiMaxDepth,
                                       m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP);
        }
        if (myY > 0) {
            hint = skeys.emplace(skeys.end(),
                                 SearchKey(myX, (myY - 1), myZ, m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN);
        }

        if (m_uiDim == 3) {
            if ((myZ + K * mySz) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey(myX, myY, (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(i);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_FRONT);
            }

            if (myZ > 0) {
                hint = skeys.emplace(
                    skeys.end(), SearchKey(myX, myY, (myZ - 1), m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
                hint->addOwner(i);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_BACK);
            }
        }
    }

    if (m_uiActiveNpes > 1) {
        for (unsigned int i = 0; i < 2 * m_uiActiveNpes; i++) {
            skeys.emplace(skeys.end(), SearchKey(m_uiLocalSplitterElements[i]));
        }
    }

    SearchKey rootSkey(m_uiDim, m_uiMaxDepth);
    std::vector<SearchKey> tmpSKeys;
    SFC::seqSort::SFC_treeSort(&(*(skeys.begin())), skeys.size(), tmpSKeys,
                               tmpSKeys, tmpSKeys, m_uiMaxDepth, m_uiMaxDepth,
                               rootSkey, ROOT_ROTATION, 1, TS_SORT_ONLY);
    assert(seq::test::isSorted(skeys));

    Key tmpKey;
    unsigned int skip = 0;
    for (unsigned int e = 0; e < (skeys.size()); e++) {
        tmpKey = Key(skeys[e].getX(), skeys[e].getY(), skeys[e].getZ(),
                     skeys[e].getLevel(), m_uiDim, m_uiMaxDepth);
        if (skeys[e].getOwner() >= 0) {
            tmpKey.addOwner(skeys[e].getOwner());
            tmpKey.addStencilIndexAndDirection(
                K - 1, skeys[e].getStencilIndexDirectionList());
        }

        skip = 1;
        while (((e + skip) < skeys.size()) && (skeys[e] == skeys[e + skip])) {
            if (skeys[e + skip].getOwner() >= 0) {
                tmpKey.addOwner(skeys[e + skip].getOwner());
                tmpKey.addStencilIndexAndDirection(
                    K - 1, skeys[e + skip].getStencilIndexDirectionList());
            }
            skip++;
        }

        m_uiKeys.push_back(tmpKey);
        e += (skip - 1);
    }

    skeys.clear();
    // if(!m_uiActiveRank) std::cout<<"key gen 0 ended "<<std::endl;
}

void Mesh::generateGhostElementSearchKeys() {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    std::vector<SearchKey> skeys;
    std::vector<SearchKey>::iterator hint;
    ot::TreeNode *inPtr     = (&(*(m_uiAllElements.begin())));
    unsigned int domain_max = 1u << (m_uiMaxDepth);
    SearchKey skey;
    DendroRegister unsigned int mySz;
    DendroRegister unsigned int myX;
    DendroRegister unsigned int myY;
    DendroRegister unsigned int myZ;
    DendroRegister unsigned int myLev;
    const unsigned int K = 1;

    for (unsigned int i = m_uiElementPreGhostBegin; i < m_uiElementPreGhostEnd;
         i++) {
        myLev = inPtr[i].getLevel();

        mySz  = (1u << (m_uiMaxDepth - myLev));
        myX   = inPtr[i].getX();
        myY   = inPtr[i].getY();
        myZ   = inPtr[i].getZ();
        // domain_max=(1u<<my)
        //  We can skip the morton index -0  because that key is mapped to the
        //  current element. So No need to search for that. Note: we do not need
        //  to perform any boundary checks when generating the keys because, we
        //  are skipping all the level one octatns.

        // for (unsigned int K = 1; K <= m_uiStensilSz; K++) {

        /** *
         * Below orientation is used when generating keys.
         *
         * [up]
         * Y
         * |     Z [front]
         * |    /
         * |   /
         * |  /
         * | /
         * -------------> X [right]
         */

        // Key generation along X axis.
        if ((myX + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX + K * mySz), myY, myZ, m_uiMaxDepth,
                                       m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT);
        }
        if (myX > 0) {
            hint = skeys.emplace(skeys.end(),
                                 SearchKey((myX - 1), myY, myZ, m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT);
        }
        // Key generation along Y axis.
        if ((myY + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey(myX, (myY + K * mySz), myZ, m_uiMaxDepth,
                                       m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP);
        }
        if (myY > 0) {
            hint = skeys.emplace(skeys.end(),
                                 SearchKey(myX, (myY - 1), myZ, m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN);
        }

        if (m_uiDim == 3) {
            if ((myZ + K * mySz) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey(myX, myY, (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(i);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_FRONT);
            }
            if (myZ > 0) {
                hint = skeys.emplace(
                    skeys.end(), SearchKey(myX, myY, (myZ - 1), m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
                hint->addOwner(i);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_BACK);
            }
        }

    }  // end for i

    for (unsigned int i = m_uiElementPostGhostBegin;
         i < m_uiElementPostGhostEnd; i++) {
        myLev = inPtr[i].getLevel();

        mySz  = (1u << (m_uiMaxDepth - myLev));
        myX   = inPtr[i].getX();
        myY   = inPtr[i].getY();
        myZ   = inPtr[i].getZ();
        // domain_max=(1u<<my)
        //  We can skip the morton index -0  because that key is mapped to the
        //  current element. So No need to search for that. Note: we do not need
        //  to perform any boundary checks when generating the keys because, we
        //  are skipping all the level one octatns.

        /** *
         * Below orientation is used when generating keys.
         *
         * [up]
         * Y
         * |     Z [front]
         * |    /
         * |   /
         * |  /
         * | /
         * -------------> X [right]
         */

        // Key generation along X axis.
        if ((myX + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX + K * mySz), myY, myZ, m_uiMaxDepth,
                                       m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT);
        }
        if (myX > 0) {
            hint = skeys.emplace(skeys.end(),
                                 SearchKey((myX - 1), myY, myZ, m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT);
        }
        // Key generation along Y axis.
        if ((myY + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey(myX, (myY + K * mySz), myZ, m_uiMaxDepth,
                                       m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP);
        }
        if (myY > 0) {
            hint = skeys.emplace(skeys.end(),
                                 SearchKey(myX, (myY - 1), myZ, m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
            hint->addOwner(i);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN);
        }

        if (m_uiDim == 3) {
            if ((myZ + K * mySz) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey(myX, myY, (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(i);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_FRONT);
            }
            if (myZ > 0) {
                hint = skeys.emplace(
                    skeys.end(), SearchKey(myX, myY, (myZ - 1), m_uiMaxDepth,
                                           m_uiDim, m_uiMaxDepth));
                hint->addOwner(i);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_BACK);
            }
        }
    }

    SearchKey rootSkey(m_uiDim, m_uiMaxDepth);
    std::vector<SearchKey> tmpSKeys;
    SFC::seqSort::SFC_treeSort(&(*(skeys.begin())), skeys.size(), tmpSKeys,
                               tmpSKeys, tmpSKeys, m_uiMaxDepth, m_uiMaxDepth,
                               rootSkey, ROOT_ROTATION, 1, TS_SORT_ONLY);
    assert(seq::test::isSorted(skeys));

    // std::cout<<"rank: "<<m_uiActiveRank<<" skeys: "<<skeys.size()<<" pre
    // local post: "<<m_uiElementPreGhostEnd<<" "<<m_uiElementLocalEnd<<"
    // "<<m_uiElementPostGhostEnd<<std::endl;

    Key tmpKey;
    unsigned int skip = 0;
    for (unsigned int e = 0; e < (skeys.size()); e++) {
        tmpKey = Key(skeys[e].getX(), skeys[e].getY(), skeys[e].getZ(),
                     skeys[e].getLevel(), m_uiDim, m_uiMaxDepth);
        if (skeys[e].getOwner() >= 0) {
            tmpKey.addOwner(skeys[e].getOwner());
            tmpKey.addStencilIndexAndDirection(
                K - 1, skeys[e].getStencilIndexDirectionList());
        }

        skip = 1;
        while (((e + skip) < skeys.size()) && (skeys[e] == skeys[e + skip])) {
            if (skeys[e + skip].getOwner() >= 0) {
                tmpKey.addOwner(skeys[e + skip].getOwner());
                tmpKey.addStencilIndexAndDirection(
                    K - 1, skeys[e + skip].getStencilIndexDirectionList());
            }
            skip++;
        }

        m_uiGhostKeys.push_back(tmpKey);
        e += (skip - 1);
    }

    skeys.clear();
    // if(!m_uiActiveRank) std::cout<<"key gen 1 ended "<<std::endl;
}

void Mesh::generateBdyElementDiagonalSearchKeys() {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    std::vector<SearchKey> skeys;
    std::vector<SearchKey>::iterator hint;
    ot::TreeNode *inPtr     = (&(*(m_uiAllElements.begin())));
    unsigned int domain_max = 1u << (m_uiMaxDepth);
    SearchKey skey;
    unsigned int elementLookUp;
    DendroRegister unsigned int mySz;
    DendroRegister unsigned int myX;
    DendroRegister unsigned int myY;
    DendroRegister unsigned int myZ;
    DendroRegister unsigned int myLev;
    const unsigned int K = 1;

    // note : this is just to find the boundary of the local octree. boundary
    // octants should be face 2 distance from the ghsot elemnts.
    std::vector<unsigned int> bdyID_L1;
    std::vector<unsigned int> bdyID_L2;

    for (unsigned int i = m_uiElementLocalBegin; i < m_uiElementLocalEnd; i++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++) {
            elementLookUp = m_uiE2EMapping[i * m_uiNumDirections + dir];
            if ((elementLookUp != LOOK_UP_TABLE_DEFAULT) &&
                ((elementLookUp < m_uiElementLocalBegin) ||
                 (elementLookUp >= m_uiElementLocalEnd))) {
                bdyID_L1.push_back(i);
                break;
            }
        }
    }

    for (unsigned int i = 0; i < bdyID_L1.size(); i++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++) {
            elementLookUp =
                m_uiE2EMapping[bdyID_L1[i] * m_uiNumDirections + dir];
            if ((elementLookUp >= m_uiElementLocalBegin) &&
                (elementLookUp < m_uiElementLocalEnd)) {
                bdyID_L2.push_back(elementLookUp);
            }
        }
    }

    // merger level 1 & level 2 bdy octants.
    bdyID_L1.insert(bdyID_L1.end(), bdyID_L2.begin(), bdyID_L2.end());

    // remove duplicates.
    std::sort(bdyID_L1.begin(), bdyID_L1.end());
    bdyID_L1.erase(std::unique(bdyID_L1.begin(), bdyID_L1.end()),
                   bdyID_L1.end());

    const unsigned int *bdyID = &(*(bdyID_L1.begin()));

    for (unsigned int e = 0; e < bdyID_L1.size(); e++) {
        myLev = inPtr[bdyID[e]].getLevel();

        mySz  = (1u << (m_uiMaxDepth - myLev));
        myX   = inPtr[bdyID[e]].getX();
        myY   = inPtr[bdyID[e]].getY();
        myZ   = inPtr[bdyID[e]].getZ();

        // Edge keys

        if (myX > 0 && myY > 0) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX - 1), (myY - 1), (myZ),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_DOWN);

            if ((myZ + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX - 1), (myY - 1), (myZ + ((K * mySz) >> 1u)),
                              m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_DOWN);

                if ((myZ + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX - 1), (myY - 1), (myZ + ((K * mySz))),
                                  m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_DOWN);
                }
            }
        }

        if (myX > 0 && (myY + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX - 1), (myY + K * mySz), (myZ),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_UP);

            if ((myZ + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX - 1), (myY + K * mySz),
                              (myZ + ((K * mySz) >> 1u)), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_UP);

                if ((myZ + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX - 1), (myY + K * mySz),
                                  (myZ + ((K * mySz))), m_uiMaxDepth, m_uiDim,
                                  m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_UP);
                }
            }
        }

        if (myX > 0 && myZ > 0) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX - 1), (myY), (myZ - 1),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_BACK);

            if ((myY + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX - 1), (myY + ((K * mySz) >> 1u)), (myZ - 1),
                              m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_BACK);

                if ((myY + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX - 1), (myY + ((K * mySz))), (myZ - 1),
                                  m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_BACK);
                }
            }
        }

        if (myX > 0 && (myZ + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX - 1), (myY), (myZ + K * mySz),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_FRONT);

            if ((myY + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX - 1), (myY + ((K * mySz) >> 1u)),
                              (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_FRONT);

                if ((myY + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(), SearchKey((myX - 1), (myY + ((K * mySz))),
                                               (myZ + K * mySz), m_uiMaxDepth,
                                               m_uiDim, m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1,
                                                      OCT_DIR_LEFT_FRONT);
                }
            }
        }

        if ((myX + K * mySz) < domain_max && myY > 0) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX + K * mySz), (myY - 1), (myZ),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_DOWN);

            if ((myZ + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + K * mySz), (myY - 1),
                              (myZ + ((K * mySz) >> 1u)), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_DOWN);

                if ((myZ + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX + K * mySz), (myY - 1),
                                  (myZ + ((K * mySz))), m_uiMaxDepth, m_uiDim,
                                  m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1,
                                                      OCT_DIR_RIGHT_DOWN);
                }
            }
        }

        if ((myX + K * mySz) < domain_max && (myY + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(),
                SearchKey((myX + K * mySz), (myY + K * mySz), (myZ),
                          m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_UP);

            if ((myZ + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + K * mySz), (myY + K * mySz),
                              (myZ + ((K * mySz) >> 1u)), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_UP);

                if ((myZ + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX + K * mySz), (myY + K * mySz),
                                  (myZ + ((K * mySz))), m_uiMaxDepth, m_uiDim,
                                  m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_UP);
                }
            }
        }

        if ((myX + K * mySz) < domain_max && myZ > 0) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX + K * mySz), (myY), (myZ - 1),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_BACK);

            if ((myY + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + K * mySz), (myY + ((K * mySz) >> 1u)),
                              (myZ - 1), m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_BACK);

                if ((myY + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX + K * mySz), (myY + ((K * mySz))),
                                  (myZ - 1), m_uiMaxDepth, m_uiDim,
                                  m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1,
                                                      OCT_DIR_RIGHT_BACK);
                }
            }
        }

        if ((myX + K * mySz) < domain_max && (myZ + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(),
                SearchKey((myX + K * mySz), (myY), (myZ + K * mySz),
                          m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_FRONT);

            if ((myY + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + K * mySz), (myY + ((K * mySz) >> 1u)),
                              (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_FRONT);

                if ((myY + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX + K * mySz), (myY + ((K * mySz))),
                                  (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                                  m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1,
                                                      OCT_DIR_RIGHT_FRONT);
                }
            }
        }

        if (myY > 0 && myZ > 0) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX), (myY - 1), (myZ - 1),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN_BACK);

            if ((myX + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + ((K * mySz) >> 1u)), (myY - 1), (myZ - 1),
                              m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN_BACK);

                if ((myX + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX + ((K * mySz))), (myY - 1), (myZ - 1),
                                  m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN_BACK);
                }
            }
        }

        if (myY > 0 && (myZ + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX), (myY - 1), (myZ + K * mySz),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN_FRONT);

            if ((myX + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + ((K * mySz) >> 1u)), (myY - 1),
                              (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_DOWN_FRONT);

                if ((myX + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(), SearchKey((myX + ((K * mySz))), (myY - 1),
                                               (myZ + K * mySz), m_uiMaxDepth,
                                               m_uiDim, m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1,
                                                      OCT_DIR_DOWN_FRONT);
                }
            }
        }

        if ((myY + K * mySz) < domain_max && myZ > 0) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX), (myY + K * mySz), (myZ - 1),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP_BACK);

            if ((myX + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + ((K * mySz) >> 1u)), (myY + K * mySz),
                              (myZ - 1), m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP_BACK);

                if ((myX + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX + ((K * mySz))), (myY + K * mySz),
                                  (myZ - 1), m_uiMaxDepth, m_uiDim,
                                  m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP_BACK);
                }
            }
        }

        if ((myY + K * mySz) < domain_max && (myZ + K * mySz) < domain_max) {
            hint = skeys.emplace(
                skeys.end(),
                SearchKey((myX), (myY + K * mySz), (myZ + K * mySz),
                          m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP_FRONT);

            if ((myX + ((K * mySz) >> 1u)) < domain_max) {
                hint = skeys.emplace(
                    skeys.end(),
                    SearchKey((myX + ((K * mySz) >> 1u)), (myY + K * mySz),
                              (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                              m_uiMaxDepth));
                hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP_FRONT);

                if ((myX + ((K * mySz))) < domain_max) {
                    hint = skeys.emplace(
                        skeys.end(),
                        SearchKey((myX + ((K * mySz))), (myY + K * mySz),
                                  (myZ + K * mySz), m_uiMaxDepth, m_uiDim,
                                  m_uiMaxDepth));
                    hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
                    hint->addStencilIndexAndDirection(K - 1, OCT_DIR_UP_FRONT);
                }
            }
        }

        // Vertex Keys.
        if ((myX > 0) && (myY > 0) && (myZ > 0)) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX - 1), (myY - 1), (myZ - 1),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_DOWN_BACK);
        }

        if (((myX + K * mySz) < domain_max) && (myY > 0) && (myZ > 0)) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX + K * mySz), (myY - 1), (myZ - 1),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_DOWN_BACK);
        }

        if ((myX > 0) && ((myY + K * mySz) < domain_max) && (myZ > 0)) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX - 1), (myY + K * mySz), (myZ - 1),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_UP_BACK);
        }

        if (((myX + K * mySz) < domain_max) &&
            ((myY + K * mySz) < domain_max) && (myZ > 0)) {
            hint = skeys.emplace(
                skeys.end(),
                SearchKey((myX + K * mySz), (myY + K * mySz), (myZ - 1),
                          m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_UP_BACK);
        }

        if ((myX > 0) && (myY > 0) && ((myZ + K * mySz) < domain_max)) {
            hint = skeys.emplace(
                skeys.end(), SearchKey((myX - 1), (myY - 1), (myZ + K * mySz),
                                       m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_DOWN_FRONT);
        }

        if (((myX + K * mySz) < domain_max) && (myY > 0) &&
            ((myZ + K * mySz) < domain_max)) {
            hint = skeys.emplace(
                skeys.end(),
                SearchKey((myX + K * mySz), (myY - 1), (myZ + K * mySz),
                          m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_DOWN_FRONT);
        }

        if ((myX > 0) && ((myY + K * mySz) < domain_max) &&
            ((myZ + K * mySz) < domain_max)) {
            hint = skeys.emplace(
                skeys.end(),
                SearchKey((myX - 1), (myY + K * mySz), (myZ + K * mySz),
                          m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_LEFT_UP_FRONT);
        }

        if (((myX + K * mySz) < domain_max) &&
            ((myY + K * mySz) < domain_max) &&
            ((myZ + K * mySz) < domain_max)) {
            hint = skeys.emplace(
                skeys.end(),
                SearchKey((myX + K * mySz), (myY + K * mySz), (myZ + K * mySz),
                          m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
            hint->addOwner(bdyID[e] - m_uiElementLocalBegin);
            hint->addStencilIndexAndDirection(K - 1, OCT_DIR_RIGHT_UP_FRONT);
        }
    }

    if (m_uiActiveNpes > 1) {
        for (unsigned int i = 0; i < 2 * m_uiActiveNpes; i++) {
            // tmpKey = Key(m_uiSplitterElementsWGhost[i].getX(),
            // m_uiSplitterElementsWGhost[i].getY(),
            // m_uiSplitterElementsWGhost[i].getZ(), m_uiMaxDepth,m_uiDim,
            // m_uiMaxDepth); tmpKey=Key(m_uiLocalSplitterElements[i]);
            skeys.emplace(skeys.end(), SearchKey(m_uiLocalSplitterElements[i]));
        }
    }

    SearchKey rootSkey(m_uiDim, m_uiMaxDepth);
    std::vector<SearchKey> tmpSKeys;
    SFC::seqSort::SFC_treeSort(&(*(skeys.begin())), skeys.size(), tmpSKeys,
                               tmpSKeys, tmpSKeys, m_uiMaxDepth, m_uiMaxDepth,
                               rootSkey, ROOT_ROTATION, 1, TS_SORT_ONLY);
    assert(seq::test::isSorted(skeys));

    Key tmpKey;
    unsigned int skip = 0;
    for (unsigned int e = 0; e < (skeys.size()); e++) {
        tmpKey = Key(skeys[e].getX(), skeys[e].getY(), skeys[e].getZ(),
                     skeys[e].getLevel(), m_uiDim, m_uiMaxDepth);
        if (skeys[e].getOwner() >= 0) {
            tmpKey.addOwner(skeys[e].getOwner());
            tmpKey.addStencilIndexAndDirection(
                K - 1, skeys[e].getStencilIndexDirectionList());
        }

        skip = 1;
        while (((e + skip) < skeys.size()) && (skeys[e] == skeys[e + skip])) {
            if (skeys[e + skip].getOwner() >= 0) {
                tmpKey.addOwner(skeys[e + skip].getOwner());
                tmpKey.addStencilIndexAndDirection(
                    K - 1, skeys[e + skip].getStencilIndexDirectionList());
            }
            skip++;
        }

        m_uiKeysDiag.push_back(tmpKey);
        e += (skip - 1);
    }

    skeys.clear();
    // if(!m_uiActiveRank) std::cout<<"key gen 2 ended "<<std::endl;
}

void Mesh::buildE2EMap(std::vector<ot::TreeNode> &in, MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    dendro::logger::debug(dendro::logger::Scope{"MESH"},
                          "Now building E2E map");

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    std::swap(m_uiEmbeddedOctree, in);
    in.clear();

    // Below Key vector and ot::TreeNode vector is being use for sorting Keys
    // and treeNodes repeatedly. So make sure you clear them after using them in
    // SFC_TreeSort.
    std::vector<Key> tmpKeys;
    std::vector<ot::TreeNode> tmpNodes;

    Key rootKey(0, 0, 0, (OCT_KEY_NONE | 0), m_uiDim, m_uiMaxDepth);
    ot::TreeNode rootNode(0, 0, 0, 0, m_uiDim, m_uiMaxDepth);

    assert(m_uiEmbeddedOctree.size() >
           1);  // m_uiEmbedded octree cannot be empty.  (Remove this assertion
                // once we handle this case. )
    assert(par::test::isUniqueAndSorted(m_uiEmbeddedOctree, comm));

    // AllGather of the the max local octant. This will be used to split the
    // keys among the processors.
    m_uiLocalSplitterElements.resize(2 * npes);
    ot::TreeNode localMinMaxElement[2];
    localMinMaxElement[0] = m_uiEmbeddedOctree.front();
    localMinMaxElement[1] = m_uiEmbeddedOctree.back();
    par::Mpi_Allgather(localMinMaxElement,
                       &(*(m_uiLocalSplitterElements.begin())), 2, comm);

    // ------------------------------------------------------
    // 2- generate the keys. (These are the keys for the local elements
    // \tau_{loc})
    // ------------------------------------------------------
    generateSearchKeys();

#ifdef DEBUG_MESH_GENERATION
    if (!rank) std::cout << "Key generation completed " << std::endl;
#endif

#ifdef DEBUG_MESH_GENERATION
    /* unsigned int localKeySz=keys_vec.size();
        unsigned int globalKeySz=0;
        par::Mpi_Reduce(&localKeySz,&globalKeySz,1,MPI_SUM,0,comm);
        if(!rank) std::cout<<" Total number of keys generated:
       "<<globalKeySz<<std::endl;*/
#endif

    // ------------------------------------------------------
    // 4- Compute Face Neighbors (By sending owners of the key to the correct
    // proc.
    // ------------------------------------------------------

    SFC::seqSort::SFC_treeSort(&(*(m_uiKeys.begin())), m_uiKeys.size(), tmpKeys,
                               tmpKeys, tmpKeys, m_uiMaxDepth, m_uiMaxDepth,
                               rootKey, 0, 1, TS_SORT_ONLY);
    tmpKeys.clear();
    assert(seq::test::isUniqueAndSorted(m_uiKeys));

#ifdef DEBUG_MESH_GENERATION
    treeNodesTovtk(m_uiKeys, rank, "m_uiKeys", false);
#endif

    m_uiSendKeyCount        = new unsigned int[npes];
    m_uiRecvKeyCount        = new unsigned int[npes];

    m_uiSendKeyOffset       = new unsigned int[npes];
    m_uiRecvKeyOffset       = new unsigned int[npes];

    m_uiSendOctCountRound1  = new unsigned int[npes];
    m_uiRecvOctCountRound1  = new unsigned int[npes];

    m_uiSendOctOffsetRound1 = new unsigned int[npes];
    m_uiRecvOctOffsetRound1 = new unsigned int[npes];

    Key *m_uiKeysPtr        = &(*(m_uiKeys.begin()));

    std::vector<Key> splitterElements;
    splitterElements.resize(2 * npes);
    for (unsigned int p = 0; p < 2 * npes; p++)
        splitterElements[p] = Key(
            m_uiLocalSplitterElements
                [p]);  // Key(m_uiLocalSplitterElements[p].getX(),m_uiLocalSplitterElements[p].getY(),m_uiLocalSplitterElements[p].getZ(),m_uiMaxDepth,m_uiDim,m_uiMaxDepth);

    assert(seq::test::isUniqueAndSorted(splitterElements));

    // search element splitters in the keys, to determine who owns the keys.
    SFC::seqSearch::SFC_treeSearch(&(*(splitterElements.begin())),
                                   &(*(m_uiKeys.begin())), 0,
                                   splitterElements.size(), 0, m_uiKeys.size(),
                                   m_uiMaxDepth, m_uiMaxDepth, ROOT_ROTATION);

    // send owners of the keys to the key owner. (R1-a ghost exchange.)
    m_uiGhostElementIDsToBeSent.clear();
    unsigned int sBegin = 0;
    unsigned int sEnd;

    for (unsigned int p = 0; p < npes; p++) {
        m_uiSendKeyCount[p] = 0;
        assert((splitterElements[2 * p].getFlag() & OCT_FOUND));
        assert((splitterElements[2 * p + 1].getFlag() & OCT_FOUND));
        ;
        assert(m_uiKeys[splitterElements[2 * p].getSearchResult()] ==
               splitterElements[2 * p]);
        assert(m_uiKeys[splitterElements[2 * p + 1].getSearchResult()] ==
               splitterElements[2 * p + 1]);
        sBegin = splitterElements[2 * p].getSearchResult();
        // sEnd=splitterElements[2*p+1].getSearchResult()+1;
        (p < (npes - 1))
            ? sEnd = splitterElements[2 * p + 2].getSearchResult() + 1
            : sEnd = m_uiKeys.size();
        if (p != m_uiActiveRank)
            for (unsigned int k = sBegin; k < sEnd; k++) {
                for (unsigned int w = 0;
                     w < m_uiKeysPtr[k].getOwnerList()->size(); w++)
                    m_uiGhostElementIDsToBeSent.push_back(
                        (*(m_uiKeysPtr[k].getOwnerList()))[w]);

                m_uiSendKeyCount[p] += m_uiKeysPtr[k].getOwnerList()->size();
            }
    }

    par::Mpi_Alltoall(m_uiSendKeyCount, m_uiRecvKeyCount, 1, comm);

    m_uiSendKeyOffset[0] = 0;
    m_uiRecvKeyOffset[0] = 0;
    omp_par::scan(m_uiSendKeyCount, m_uiSendKeyOffset, npes);
    omp_par::scan(m_uiRecvKeyCount, m_uiRecvKeyOffset, npes);

    assert(m_uiGhostElementIDsToBeSent.size() ==
           (m_uiSendKeyOffset[npes - 1] + m_uiSendKeyCount[npes - 1]));

    // we need to collect both face edge & vertex neighbors as level 1 ghost
    // layer. here we collect the face neighbours later we will collect the edge
    // and vertex neighbors.
    std::vector<unsigned int> *scatterMapSend_R1 =
        new std::vector<unsigned int>[npes];

    std::set<unsigned int> tmpSendElementIds;
    for (unsigned int p = 0; p < npes; p++) {
        scatterMapSend_R1[p] = std::vector<unsigned int>();
        tmpSendElementIds.insert(
            (m_uiGhostElementIDsToBeSent.begin() + m_uiSendKeyOffset[p]),
            (m_uiGhostElementIDsToBeSent.begin() + m_uiSendKeyCount[p] +
             m_uiSendKeyOffset[p]));
        m_uiSendOctCountRound1[p] = tmpSendElementIds.size();
        scatterMapSend_R1[p].insert(scatterMapSend_R1[p].end(),
                                    tmpSendElementIds.begin(),
                                    tmpSendElementIds.end());
        tmpSendElementIds.clear();
    }

    m_uiGhostElementIDsToBeSent.clear();

    par::Mpi_Alltoall(m_uiSendOctCountRound1, m_uiRecvOctCountRound1, 1, comm);

    m_uiSendOctOffsetRound1[0] = 0;
    m_uiRecvOctOffsetRound1[0] = 0;

    omp_par::scan(m_uiSendOctCountRound1, m_uiSendOctOffsetRound1, npes);
    omp_par::scan(m_uiRecvOctCountRound1, m_uiRecvOctOffsetRound1, npes);

    for (unsigned int p = 0; p < npes; p++)
        for (unsigned int k = 0; k < scatterMapSend_R1[p].size(); k++)
            m_uiSendBufferElement.push_back(
                m_uiEmbeddedOctree[scatterMapSend_R1[p][k]]);

    m_uiGhostOctants.resize(m_uiRecvOctOffsetRound1[npes - 1] +
                            m_uiRecvOctCountRound1[npes - 1]);

    par::Mpi_Alltoallv(
        &(*(m_uiSendBufferElement.begin())), (int *)m_uiSendOctCountRound1,
        (int *)m_uiSendOctOffsetRound1, &(*(m_uiGhostOctants.begin())),
        (int *)m_uiRecvOctCountRound1, (int *)m_uiRecvOctOffsetRound1, comm);

    std::swap(m_uiAllElements, m_uiEmbeddedOctree);
    m_uiEmbeddedOctree.clear();
    m_uiAllElements.insert(m_uiAllElements.end(), m_uiGhostOctants.begin(),
                           m_uiGhostOctants.end());

    SFC::seqSort::SFC_treeSort(&(*(m_uiAllElements.begin())),
                               m_uiAllElements.size(), tmpNodes, tmpNodes,
                               tmpNodes, m_uiMaxDepth, m_uiMaxDepth, rootNode,
                               0, 1, TS_REMOVE_DUPLICATES);
    std::swap(m_uiAllElements, tmpNodes);
    tmpNodes.clear();

    assert(seq::test::isUniqueAndSorted(m_uiAllElements));

    m_uiGhostOctants.clear();

    m_uiElementPreGhostBegin  = 0;
    m_uiElementPreGhostEnd    = 0;
    m_uiElementLocalBegin     = 0;
    m_uiElementLocalEnd       = 0;
    m_uiElementPostGhostBegin = 0;
    m_uiElementPostGhostEnd   = 0;

    for (unsigned int k = 0; k < m_uiAllElements.size(); k++) {
        if (m_uiAllElements[k] == localMinMaxElement[0]) {
            m_uiElementLocalBegin = k;
            break;
        }
    }

    for (unsigned int k = (m_uiAllElements.size() - 1); k > 0; k--) {
        if (m_uiAllElements[k] == localMinMaxElement[1])
            m_uiElementLocalEnd = k + 1;
    }

    m_uiElementPreGhostEnd    = m_uiElementLocalBegin;
    m_uiElementPostGhostBegin = m_uiElementLocalEnd;
    m_uiElementPostGhostEnd   = m_uiAllElements.size();

    m_uiNumLocalElements      = (m_uiElementLocalEnd - m_uiElementLocalBegin);
    m_uiNumPreGhostElements =
        (m_uiElementPreGhostEnd - m_uiElementPreGhostBegin);
    m_uiNumPostGhostElements =
        (m_uiElementPostGhostEnd - m_uiElementPostGhostBegin);

    m_uiNumTotalElements = m_uiNumPreGhostElements + m_uiNumLocalElements +
                           m_uiNumPostGhostElements;

    const unsigned int FACE_EXC_ELE_BEGIN = m_uiElementLocalBegin;
    const unsigned int FACE_EXC_ELE_END   = m_uiElementLocalEnd;

    // E2E Mapping for the Round 1 Ghost Exchange. Later we will perform another
    // round of ghost exchange, (for the ghost elements that are hanging ) and
    // build the correct complete E2E mapping.
    SFC::seqSearch::SFC_treeSearch(&(*(m_uiKeys.begin())),
                                   &(*(m_uiAllElements.begin())), 0,
                                   m_uiKeys.size(), 0, m_uiAllElements.size(),
                                   m_uiMaxDepth, m_uiMaxDepth, 0);

    std::vector<unsigned int> *ownerList;
    std::vector<unsigned int> *stencilIndexDirection;
    unsigned int result;
    unsigned int direction;

    m_uiKeysPtr = &(*(m_uiKeys.begin()));
    m_uiE2EMapping.resize(m_uiAllElements.size() * m_uiNumDirections,
                          LOOK_UP_TABLE_DEFAULT);

    /**Neighbour list is made from
     *  first x axis. (left to right)
     *  second y axis. (down to up)
     *  third z axis. (back to front)**/
    for (unsigned int k = 0; k < m_uiKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();
        if (!(OCT_FOUND & m_uiKeysPtr[k].getFlag()))
            continue;  // Note that some keys might not be found due to absence
                       // of diagonal keys at this stage. (but all the keys
                       // should be found after diagonal neighbour exchange. )
        stencilIndexDirection = m_uiKeysPtr[k].getStencilIndexDirectionList();
        result                = m_uiKeysPtr[k].getSearchResult();
        if (ownerList->size())
            assert(m_uiAllElements[result].isAncestor(m_uiKeys[k]) ||
                   m_uiAllElements[result] ==
                       m_uiKeys[k]);  // To check the result found in the
                                      // treeSearch is correct or not.

        for (unsigned int w = 0; w < ownerList->size(); w++) {
            direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
            m_uiE2EMapping[(((*ownerList)[w]) + m_uiElementLocalBegin) *
                               m_uiNumDirections +
                           direction] = result;
        }
    }

    m_uiGhostKeys.clear();
    generateGhostElementSearchKeys();

    SFC::seqSearch::SFC_treeSearch(
        &(*(m_uiGhostKeys.begin())), &(*(m_uiAllElements.begin())), 0,
        m_uiGhostKeys.size(), 0, m_uiAllElements.size(), m_uiMaxDepth,
        m_uiMaxDepth, ROOT_ROTATION);

    m_uiKeysPtr = &(*(m_uiGhostKeys.begin()));
    // Note : Since the ghost elements are not complete it is not required to
    // find all the keys in the ghost.
    for (unsigned int k = 0; k < m_uiGhostKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if ((OCT_FOUND & m_uiKeysPtr[k].getFlag())) {
            stencilIndexDirection =
                m_uiKeysPtr[k].getStencilIndexDirectionList();
            result = m_uiKeysPtr[k].getSearchResult();
            if (ownerList->size())
                assert(
                    m_uiAllElements[result].isAncestor(m_uiGhostKeys[k]) ||
                    m_uiAllElements[result] ==
                        m_uiGhostKeys[k]);  // To check the result found in the
                                            // treeSearch is correct or not.

            for (unsigned int w = 0; w < ownerList->size(); w++) {
                direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
                m_uiE2EMapping[(((*ownerList)[w])) * m_uiNumDirections +
                               direction] = result;
            }
        }
    }

    // ----------------------------------------------------------
    // 5- Compute missing face keys
    // ----------------------------------------------------------

    //  computing the owner ship of the ghost elements.
    std::vector<unsigned int> elementOwner;
    computeElementOwnerRanks(elementOwner);

    unsigned int lookUp;
    unsigned int gOwner;
    std::vector<unsigned int> *missedSendID =
        new std::vector<unsigned int>[npes];  // store the missed id after
                                              // sending the ghost keys to it
                                              // self.
    // for local elements.
    for (unsigned int ele = m_uiElementLocalBegin; ele < m_uiElementLocalEnd;
         ele++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++) {
            lookUp = m_uiE2EMapping[ele * m_uiNumDirections + dir];
            if (lookUp != LOOK_UP_TABLE_DEFAULT &&
                ((lookUp < m_uiElementLocalBegin) ||
                 (lookUp >= m_uiElementLocalEnd))) {
                gOwner = elementOwner[lookUp];
                assert(gOwner != rank);
                assert(gOwner < npes && gOwner >= 0);
                if (!std::binary_search(scatterMapSend_R1[gOwner].begin(),
                                        scatterMapSend_R1[gOwner].end(),
                                        (ele - m_uiElementLocalBegin)))
                    missedSendID[gOwner].push_back(ele - m_uiElementLocalBegin);
            }
        }
    }

    // for pre ghost
    for (unsigned int ele = m_uiElementPreGhostBegin;
         ele < m_uiElementPreGhostEnd; ele++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++) {
            lookUp = m_uiE2EMapping[ele * m_uiNumDirections + dir];
            if (lookUp >= m_uiElementLocalBegin &&
                lookUp < m_uiElementLocalEnd) {
                gOwner = elementOwner[ele];
                assert(gOwner != rank);
                assert(gOwner < npes && gOwner >= 0);
                if (!std::binary_search(scatterMapSend_R1[gOwner].begin(),
                                        scatterMapSend_R1[gOwner].end(),
                                        (lookUp - m_uiElementLocalBegin)))
                    missedSendID[gOwner].push_back(lookUp -
                                                   m_uiElementLocalBegin);
            }
        }
    }

    // for post ghost
    for (unsigned int ele = m_uiElementPostGhostBegin;
         ele < m_uiElementPostGhostEnd; ele++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++) {
            lookUp = m_uiE2EMapping[ele * m_uiNumDirections + dir];
            if (lookUp >= m_uiElementLocalBegin &&
                lookUp < m_uiElementLocalEnd) {
                gOwner = elementOwner[ele];
                assert(gOwner != rank);
                assert(gOwner < npes && gOwner >= 0);
                if (!std::binary_search(scatterMapSend_R1[gOwner].begin(),
                                        scatterMapSend_R1[gOwner].end(),
                                        (lookUp - m_uiElementLocalBegin)))
                    missedSendID[gOwner].push_back(lookUp -
                                                   m_uiElementLocalBegin);
            }
        }
    }

    m_uiSendBufferElement.clear();
    for (unsigned int p = 0; p < npes; p++) {
        if (missedSendID[p].size() != 0) {
            std::sort(missedSendID[p].begin(), missedSendID[p].end());
            missedSendID[p].erase(
                std::unique(missedSendID[p].begin(), missedSendID[p].end()),
                missedSendID[p].end());
        }

        m_uiSendOctCountRound1[p] = missedSendID[p].size();
        for (unsigned int i = 0; i < missedSendID[p].size(); i++)
            m_uiSendBufferElement.push_back(
                m_uiAllElements[missedSendID[p][i] + m_uiElementLocalBegin]);

        scatterMapSend_R1[p].insert(scatterMapSend_R1[p].end(),
                                    missedSendID[p].begin(),
                                    missedSendID[p].end());
        std::sort(scatterMapSend_R1[p].begin(), scatterMapSend_R1[p].end());
        missedSendID[p].clear();
    }

    elementOwner.clear();

    par::Mpi_Alltoall(m_uiSendOctCountRound1, m_uiRecvOctCountRound1, 1, comm);

    m_uiSendOctOffsetRound1[0] = 0;
    m_uiRecvOctOffsetRound1[0] = 0;

    omp_par::scan(m_uiSendOctCountRound1, m_uiSendOctOffsetRound1, npes);
    omp_par::scan(m_uiRecvOctCountRound1, m_uiRecvOctOffsetRound1, npes);

    m_uiGhostOctants.clear();
    m_uiGhostOctants.resize(m_uiRecvOctOffsetRound1[npes - 1] +
                            m_uiRecvOctCountRound1[npes - 1]);

    // exchange missed keys.
    par::Mpi_Alltoallv(
        &(*(m_uiSendBufferElement.begin())), (int *)m_uiSendOctCountRound1,
        (int *)m_uiSendOctOffsetRound1, &(*(m_uiGhostOctants.begin())),
        (int *)m_uiRecvOctCountRound1, (int *)m_uiRecvOctOffsetRound1, comm);

    m_uiAllElements.insert(m_uiAllElements.end(), m_uiGhostOctants.begin(),
                           m_uiGhostOctants.end());

    SFC::seqSort::SFC_treeSort(&(*(m_uiAllElements.begin())),
                               m_uiAllElements.size(), tmpNodes, tmpNodes,
                               tmpNodes, m_uiMaxDepth, m_uiMaxDepth, rootNode,
                               0, 1, TS_REMOVE_DUPLICATES);
    std::swap(m_uiAllElements, tmpNodes);
    tmpNodes.clear();

    assert(seq::test::isUniqueAndSorted(m_uiAllElements));

    m_uiGhostOctants.clear();

    m_uiElementPreGhostBegin  = 0;
    m_uiElementPreGhostEnd    = 0;
    m_uiElementLocalBegin     = 0;
    m_uiElementLocalEnd       = 0;
    m_uiElementPostGhostBegin = 0;
    m_uiElementPostGhostEnd   = 0;

    for (unsigned int k = 0; k < m_uiAllElements.size(); k++) {
        if (m_uiAllElements[k] == localMinMaxElement[0]) {
            m_uiElementLocalBegin = k;
            break;
        }
    }

    for (unsigned int k = (m_uiAllElements.size() - 1); k > 0; k--) {
        if (m_uiAllElements[k] == localMinMaxElement[1])
            m_uiElementLocalEnd = k + 1;
    }

    m_uiElementPreGhostEnd    = m_uiElementLocalBegin;
    m_uiElementPostGhostBegin = m_uiElementLocalEnd;
    m_uiElementPostGhostEnd   = m_uiAllElements.size();

    m_uiNumLocalElements      = (m_uiElementLocalEnd - m_uiElementLocalBegin);
    m_uiNumPreGhostElements =
        (m_uiElementPreGhostEnd - m_uiElementPreGhostBegin);
    m_uiNumPostGhostElements =
        (m_uiElementPostGhostEnd - m_uiElementPostGhostBegin);

    m_uiNumTotalElements = m_uiNumPreGhostElements + m_uiNumLocalElements +
                           m_uiNumPostGhostElements;

    const unsigned int FACE_MISNG_EXC_ELE_BEGIN = m_uiElementLocalBegin;
    const unsigned int FACE_MISNG_EXC_ELE_END   = m_uiElementLocalEnd;

    // E2E Mapping for the Round face-1 & face-2 face ghost exchange.
    SFC::seqSearch::SFC_treeSearch(&(*(m_uiKeys.begin())),
                                   &(*(m_uiAllElements.begin())), 0,
                                   m_uiKeys.size(), 0, m_uiAllElements.size(),
                                   m_uiMaxDepth, m_uiMaxDepth, 0);

    m_uiKeysPtr = &(*(m_uiKeys.begin()));
    m_uiE2EMapping.resize(m_uiAllElements.size() * m_uiNumDirections,
                          LOOK_UP_TABLE_DEFAULT);

    /**Neighbour list is made from
     *  first x axis. (left to right)
     *  second y axis. (down to up)
     *  third z axis. (back to front)**/
    for (unsigned int k = 0; k < m_uiKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();
        if (ownerList->size() && (!((OCT_FOUND & m_uiKeysPtr[k].getFlag())))) {
            std::cout << "rank: " << m_uiActiveRank
                      << "[E2E Error]: Local  face key missing after R1 face-1 "
                         "& face-2 ghost exchange: "
                      << m_uiKeysPtr[k] << std::endl;
            exit(0);
        }

        stencilIndexDirection = m_uiKeysPtr[k].getStencilIndexDirectionList();
        result                = m_uiKeysPtr[k].getSearchResult();
        if (ownerList->size())
            assert(m_uiAllElements[result].isAncestor(m_uiKeys[k]) ||
                   m_uiAllElements[result] ==
                       m_uiKeys[k]);  // To check the result found in the
                                      // treeSearch is correct or not.

        for (unsigned int w = 0; w < ownerList->size(); w++) {
            direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
            m_uiE2EMapping[(((*ownerList)[w]) + m_uiElementLocalBegin) *
                               m_uiNumDirections +
                           direction] = result;
        }
    }

    m_uiGhostKeys.clear();
    generateGhostElementSearchKeys();

    SFC::seqSearch::SFC_treeSearch(
        &(*(m_uiGhostKeys.begin())), &(*(m_uiAllElements.begin())), 0,
        m_uiGhostKeys.size(), 0, m_uiAllElements.size(), m_uiMaxDepth,
        m_uiMaxDepth, ROOT_ROTATION);

    m_uiKeysPtr = &(*(m_uiGhostKeys.begin()));
    // Note : Since the ghost elements are not complete it is not required to
    // find all the keys in the ghost.
    for (unsigned int k = 0; k < m_uiGhostKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if ((OCT_FOUND & m_uiKeysPtr[k].getFlag())) {
            stencilIndexDirection =
                m_uiKeysPtr[k].getStencilIndexDirectionList();
            result = m_uiKeysPtr[k].getSearchResult();
            if (ownerList->size())
                assert(
                    m_uiAllElements[result].isAncestor(m_uiGhostKeys[k]) ||
                    m_uiAllElements[result] ==
                        m_uiGhostKeys[k]);  // To check the result found in the
                                            // treeSearch is correct or not.

            for (unsigned int w = 0; w < ownerList->size(); w++) {
                direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
                m_uiE2EMapping[(((*ownerList)[w])) * m_uiNumDirections +
                               direction] = result;
                // Note : Following is done to enforce that the local elements
                // that points to ghost elements has the inverse mapping.
            }
        }
    }

    // ---------------------------------------------------------------
    // 6- Perform exchange for edge and vertex neighbors.
    // ---------------------------------------------------------------

    m_uiKeysDiag.clear();
    generateBdyElementDiagonalSearchKeys();

#ifdef DEBUG_MESH_GENERATION
    treeNodesTovtk(m_uiKeysDiag, rank, "m_uiKeyDiag");
#endif

    tmpKeys.clear();
    SFC::seqSort::SFC_treeSort(&(*(m_uiKeysDiag.begin())), m_uiKeysDiag.size(),
                               tmpKeys, tmpKeys, tmpKeys, m_uiMaxDepth,
                               m_uiMaxDepth, rootKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);
    tmpKeys.clear();

    for (unsigned int i = 0; i < 2 * npes; i++)
        splitterElements[i] = ot::Key(m_uiLocalSplitterElements[i]);

    assert(seq::test::isUniqueAndSorted(splitterElements));

    SFC::seqSearch::SFC_treeSearch(
        &(*(splitterElements.begin())), &(*(m_uiKeysDiag.begin())), 0,
        splitterElements.size(), 0, m_uiKeysDiag.size(), m_uiMaxDepth,
        m_uiMaxDepth, ROOT_ROTATION);

    m_uiKeysPtr = &(*(m_uiKeysDiag.begin()));
    m_uiGhostElementIDsToBeSent.clear();
    m_uiSendBufferElement.clear();

    m_uiSendKeyDiagCount        = new unsigned int[npes];
    m_uiRecvKeyDiagCount        = new unsigned int[npes];

    m_uiSendKeyDiagOffset       = new unsigned int[npes];
    m_uiRecvKeyDiagOffset       = new unsigned int[npes];

    m_uiSendOctCountRound1Diag  = new unsigned int[npes];
    m_uiRecvOctCountRound1Diag  = new unsigned int[npes];

    m_uiSendOctOffsetRound1Diag = new unsigned int[npes];
    m_uiRecvOctOffsetRound1Diag = new unsigned int[npes];

    for (unsigned int p = 0; p < npes; p++) m_uiSendKeyDiagCount[p] = 0;

    m_uiSendBufferElement.clear();

    for (unsigned int p = 0; p < npes; p++) {
        assert((splitterElements[2 * p].getFlag() & OCT_FOUND));
        assert((splitterElements[2 * p + 1].getFlag() & OCT_FOUND));
        ;
        assert(m_uiKeysDiag[splitterElements[2 * p].getSearchResult()] ==
               splitterElements[2 * p]);
        assert(m_uiKeysDiag[splitterElements[2 * p + 1].getSearchResult()] ==
               splitterElements[2 * p + 1]);

        sBegin = splitterElements[2 * p].getSearchResult();
        // sEnd=splitterElements[2*p+1].getSearchResult()+1;
        (p < (m_uiActiveNpes - 1))
            ? sEnd = splitterElements[2 * p + 2].getSearchResult() + 1
            : sEnd = m_uiKeysDiag.size();
        if (p != m_uiActiveRank)
            for (unsigned int i = sBegin; i < sEnd; i++) {
                if (m_uiKeysDiag[i].getOwnerList()->size()) {
                    m_uiSendBufferElement.push_back(m_uiKeysDiag[i]);
                    m_uiSendKeyDiagCount[p]++;
                }
            }
    }

    par::Mpi_Alltoall(m_uiSendKeyDiagCount, m_uiRecvKeyDiagCount, 1, comm);

    m_uiSendKeyDiagOffset[0] = 0;
    m_uiRecvKeyDiagOffset[0] = 0;

    omp_par::scan(m_uiSendKeyDiagCount, m_uiSendKeyDiagOffset, npes);
    omp_par::scan(m_uiRecvKeyDiagCount, m_uiRecvKeyDiagOffset, npes);

    std::vector<ot::TreeNode> recvDiagKeyOct;
    recvDiagKeyOct.resize(m_uiRecvKeyDiagOffset[npes - 1] +
                          m_uiRecvKeyDiagCount[npes - 1]);

    par::Mpi_Alltoallv(
        &(*(m_uiSendBufferElement.begin())), (int *)m_uiSendKeyDiagCount,
        (int *)m_uiSendKeyDiagOffset, &(*(recvDiagKeyOct.begin())),
        (int *)m_uiRecvKeyDiagCount, (int *)m_uiRecvKeyDiagOffset, comm);

#ifdef DEBUG_MESH_GENERATION
    treeNodesTovtk(recvDiagKeyOct, rank, "recvDiagKeyOct");
#endif

    std::vector<ot::Key> recvDiagKey_keys;
    std::vector<ot::Key>::iterator itKey;
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = m_uiRecvKeyDiagOffset[p];
             e < (m_uiRecvKeyDiagOffset[p] + m_uiRecvKeyDiagCount[p]); e++) {
            itKey = recvDiagKey_keys.emplace(recvDiagKey_keys.end(),
                                             ot::Key(recvDiagKeyOct[e]));
            itKey->addOwner(p);
        }
    }

    SFC::seqSearch::SFC_treeSearch(
        &(*(recvDiagKey_keys.begin())), &(*(m_uiAllElements.begin())), 0,
        recvDiagKey_keys.size(), m_uiElementLocalBegin, m_uiElementLocalEnd,
        m_uiMaxDepth, m_uiMaxDepth, ROOT_ROTATION);

    for (unsigned int p = 0; p < npes; p++) {
        m_uiSendOctCountRound1Diag[p] = 0;
        missedSendID[p].clear();
    }

    // note that not all the recv keys needs to be found since recvkey send is
    // overlapped among the processors.
    for (unsigned int e = 0; e < recvDiagKey_keys.size(); e++) {
        if (!(recvDiagKey_keys[e].getFlag() & OCT_FOUND)) continue;

        ownerList = recvDiagKey_keys[e].getOwnerList();
        result    = recvDiagKey_keys[e].getSearchResult();
        for (unsigned int w = 0; w < ownerList->size(); w++) {
            missedSendID[(*ownerList)[w]].push_back(result -
                                                    m_uiElementLocalBegin);
        }
    }

    m_uiSendBufferElement.clear();
    for (unsigned int p = 0; p < npes; p++) {
        std::sort(missedSendID[p].begin(), missedSendID[p].end());
        missedSendID[p].erase(
            std::unique(missedSendID[p].begin(), missedSendID[p].end()),
            missedSendID[p].end());
        m_uiSendOctCountRound1Diag[p] = missedSendID[p].size();

        for (unsigned int e = 0; e < missedSendID[p].size(); e++) {
            m_uiSendBufferElement.push_back(
                m_uiAllElements[missedSendID[p][e] + m_uiElementLocalBegin]);
        }

        scatterMapSend_R1[p].insert(scatterMapSend_R1[p].end(),
                                    missedSendID[p].begin(),
                                    missedSendID[p].end());
        missedSendID[p].clear();
    }

    delete[] missedSendID;

    par::Mpi_Alltoall(m_uiSendOctCountRound1Diag, m_uiRecvOctCountRound1Diag, 1,
                      comm);

    m_uiSendOctOffsetRound1Diag[0] = 0;
    m_uiRecvOctOffsetRound1Diag[0] = 0;

    omp_par::scan(m_uiSendOctCountRound1Diag, m_uiSendOctOffsetRound1Diag,
                  npes);
    omp_par::scan(m_uiRecvOctCountRound1Diag, m_uiRecvOctOffsetRound1Diag,
                  npes);

    m_uiGhostOctants.clear();

    m_uiGhostOctants.resize(m_uiRecvOctOffsetRound1Diag[npes - 1] +
                            m_uiRecvOctCountRound1Diag[npes - 1]);
    par::Mpi_Alltoallv(
        &(*(m_uiSendBufferElement.begin())), (int *)m_uiSendOctCountRound1Diag,
        (int *)m_uiSendOctOffsetRound1Diag, &(*(m_uiGhostOctants.begin())),
        (int *)m_uiRecvOctCountRound1Diag, (int *)m_uiRecvOctOffsetRound1Diag,
        comm);

    m_uiAllElements.insert(m_uiAllElements.end(), m_uiGhostOctants.begin(),
                           m_uiGhostOctants.end());
    m_uiGhostOctants.clear();

    SFC::seqSort::SFC_treeSort(&(*(m_uiAllElements.begin())),
                               m_uiAllElements.size(), tmpNodes, tmpNodes,
                               tmpNodes, m_uiMaxDepth, m_uiMaxDepth, rootNode,
                               0, 1, TS_REMOVE_DUPLICATES);
    std::swap(m_uiAllElements, tmpNodes);
    tmpNodes.clear();

    assert(seq::test::isUniqueAndSorted(m_uiAllElements));

    m_uiElementPreGhostBegin  = 0;
    m_uiElementPreGhostEnd    = 0;
    m_uiElementLocalBegin     = 0;
    m_uiElementLocalEnd       = 0;
    m_uiElementPostGhostBegin = 0;
    m_uiElementPostGhostEnd   = 0;

    for (unsigned int k = 0; k < m_uiAllElements.size(); k++) {
        if (m_uiAllElements[k] == localMinMaxElement[0]) {
            m_uiElementLocalBegin = k;
            break;
        }
    }

    for (unsigned int k = (m_uiAllElements.size() - 1); k > 0; k--) {
        if (m_uiAllElements[k] == localMinMaxElement[1])
            m_uiElementLocalEnd = k + 1;
    }

    m_uiElementPreGhostEnd    = m_uiElementLocalBegin;
    m_uiElementPostGhostBegin = m_uiElementLocalEnd;
    m_uiElementPostGhostEnd   = m_uiAllElements.size();

    m_uiNumLocalElements      = (m_uiElementLocalEnd - m_uiElementLocalBegin);
    m_uiNumPreGhostElements =
        (m_uiElementPreGhostEnd - m_uiElementPreGhostBegin);
    m_uiNumPostGhostElements =
        (m_uiElementPostGhostEnd - m_uiElementPostGhostBegin);

    m_uiNumTotalElements = m_uiNumPreGhostElements + m_uiNumLocalElements +
                           m_uiNumPostGhostElements;

    const unsigned int EDGE_VERTEX_EXC_ELE_BEGIN = m_uiElementLocalBegin;
    const unsigned int EDGE_VERTEX_EXC_ELE_END   = m_uiElementLocalEnd;

    // E2E Mapping for the Round 1 Ghost Exchange. Later we will perform another
    // round of ghost exchange, (for the ghost elements that are hanging ) and
    // build the correct complete E2E mapping.

    SFC::seqSearch::SFC_treeSearch(&(*(m_uiKeys.begin())),
                                   &(*(m_uiAllElements.begin())), 0,
                                   m_uiKeys.size(), 0, m_uiAllElements.size(),
                                   m_uiMaxDepth, m_uiMaxDepth, 0);

    m_uiKeysPtr = &(*(m_uiKeys.begin()));
    m_uiE2EMapping.resize(m_uiAllElements.size() * m_uiNumDirections,
                          LOOK_UP_TABLE_DEFAULT);

    /**Neighbour list is made from
     *  first x axis. (left to right)
     *  second y axis. (down to up)
     *  third z axis. (back to front)**/

    for (unsigned int k = 0; k < m_uiKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if (ownerList->size() && (!((OCT_FOUND & m_uiKeysPtr[k].getFlag())))) {
            std::cout << "rank: " << m_uiActiveRank
                      << "[E2E Error]: Local  face key missing after R1 (face "
                         "edge vertex) ghost exchange: "
                      << m_uiKeysPtr[k] << std::endl;
            exit(0);
        }

        if (ownerList->size())
            assert((OCT_FOUND &
                    m_uiKeysPtr[k]
                        .getFlag()));  // Note that all the keys should be found
                                       // locally due to the fact that we have
                                       // exchanged ghost elements.
        stencilIndexDirection = m_uiKeysPtr[k].getStencilIndexDirectionList();
        result                = m_uiKeysPtr[k].getSearchResult();
        if (ownerList->size())
            assert(m_uiAllElements[result].isAncestor(m_uiKeys[k]) ||
                   m_uiAllElements[result] ==
                       m_uiKeys[k]);  // To check the result found in the
                                      // treeSearch is correct or not.
        for (unsigned int w = 0; w < ownerList->size(); w++) {
            direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
            m_uiE2EMapping[(((*ownerList)[w]) + m_uiElementLocalBegin) *
                               m_uiNumDirections +
                           direction] = result;
        }
    }

    m_uiGhostKeys.clear();
    generateGhostElementSearchKeys();

    SFC::seqSearch::SFC_treeSearch(
        &(*(m_uiGhostKeys.begin())), &(*(m_uiAllElements.begin())), 0,
        m_uiGhostKeys.size(), 0, m_uiAllElements.size(), m_uiMaxDepth,
        m_uiMaxDepth, ROOT_ROTATION);

    m_uiKeysPtr = &(*(m_uiGhostKeys.begin()));
    // Note : Since the ghost elements are not complete it is not required to
    // find all the keys in the ghost.
    for (unsigned int k = 0; k < m_uiGhostKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if ((OCT_FOUND & m_uiKeysPtr[k].getFlag())) {
            stencilIndexDirection =
                m_uiKeysPtr[k].getStencilIndexDirectionList();
            result = m_uiKeysPtr[k].getSearchResult();
            if (ownerList->size())
                assert(
                    m_uiAllElements[result].isAncestor(m_uiGhostKeys[k]) ||
                    m_uiAllElements[result] ==
                        m_uiGhostKeys[k]);  // To check the result found in the
                                            // treeSearch is correct or not.

            for (unsigned int w = 0; w < ownerList->size(); w++) {
                direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
                m_uiE2EMapping[(((*ownerList)[w])) * m_uiNumDirections +
                               direction] = result;
                // Note : Following is done to enforce that the local elements
                // that points to ghost elements has the inverse mapping.
            }
        }
    }

    SFC::seqSearch::SFC_treeSearch(
        &(*(m_uiKeysDiag.begin())), &(*(m_uiAllElements.begin())), 0,
        m_uiKeysDiag.size(), 0, m_uiAllElements.size(), m_uiMaxDepth,
        m_uiMaxDepth, ROOT_ROTATION);

    m_uiKeysPtr = &(*(m_uiKeysDiag.begin()));
    for (unsigned int k = 0; k < m_uiKeysDiag.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if (ownerList->size() && (!((OCT_FOUND & m_uiKeysPtr[k].getFlag())))) {
            std::cout << "rank: " << m_uiActiveRank
                      << "[E2E Error]: Local edge or vertex key missing after "
                         "R1 (face edge vertex) ghost exchange: "
                      << m_uiKeysPtr[k] << std::endl;
            exit(0);  // no point in continuing if this fails E2N fails for sure
                      // :)
        }
    }

    // merge the face neighbors with edge and vertex neighbors,
    // Note that m_uiScatterMapElementRound1 should only contain the local
    // element IDs. (pure local ID (ID's before even merging with ghost. ))
    // these IDs indicate that my local element is an ghost element to some
    // other processor. Since we have all the neighours for the boundary nodes
    // we can determine whether the each local element is hanging or not. If is
    // it hanging it will be participated in the layer 2 ghost level exchange.

    m_uiScatterMapElementRound1.clear();

    m_uiSendEleCount.resize(npes);
    m_uiRecvEleCount.resize(npes);
    m_uiSendEleOffset.resize(npes);
    m_uiRecvEleOffset.resize(npes);

    for (unsigned int p = 0; p < npes; p++) {
        std::sort(scatterMapSend_R1[p].begin(), scatterMapSend_R1[p].end());
        scatterMapSend_R1[p].erase(std::unique(scatterMapSend_R1[p].begin(),
                                               scatterMapSend_R1[p].end()),
                                   scatterMapSend_R1[p].end());
        m_uiScatterMapElementRound1.insert(m_uiScatterMapElementRound1.end(),
                                           scatterMapSend_R1[p].begin(),
                                           scatterMapSend_R1[p].end());
        m_uiSendOctCountRound1[p] = scatterMapSend_R1[p].size();
        m_uiSendEleCount[p]       = scatterMapSend_R1[p].size();
        scatterMapSend_R1[p].clear();
    }

    // finalized round 1 ghost exchange elements including face, edge and
    // vertex.

    par::Mpi_Alltoall(m_uiSendEleCount.data(), m_uiRecvEleCount.data(), 1,
                      m_uiCommActive);
    m_uiSendEleOffset[0] = 0;
    m_uiRecvEleOffset[0] = 0;

    omp_par::scan(m_uiSendEleCount.data(), m_uiSendEleOffset.data(), npes);
    omp_par::scan(m_uiRecvEleCount.data(), m_uiRecvEleOffset.data(), npes);

    for (unsigned int p = 0; p < npes; p++)
        m_uiSendOctOffsetRound1[p] = m_uiSendEleOffset[p];

    delete[] scatterMapSend_R1;

    // push the diagonal ghost layer 1 keys. this includes the face
    std::vector<ot::TreeNode> gKeys_R1;
    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++)
        gKeys_R1.push_back(m_uiAllElements[e]);

    for (unsigned int e = m_uiElementPostGhostBegin;
         e < m_uiElementPostGhostEnd; e++)
        gKeys_R1.push_back(m_uiAllElements[e]);

    // --------------------------------------------------------------------
    // 8 R2 ghost exchange
    // --------------------------------------------------------------------
    m_uiSendOctCountRound2  = new unsigned int[npes];
    m_uiRecvOctCountRound2  = new unsigned int[npes];

    m_uiSendOctOffsetRound2 = new unsigned int[npes];
    m_uiRecvOctOffsetRound2 = new unsigned int[npes];

    tmpSendElementIds.clear();
    m_uiSendBufferElement.clear();
    unsigned int elementLookup;
    std::set<unsigned int> *tmpSendEleIdR2 = new std::set<unsigned int>[npes];
    std::pair<std::set<unsigned int>::iterator, bool> setHintUint;

    for (unsigned int p = 0; p < npes; p++) {
        m_uiSendOctCountRound2[p] = 0;
        for (unsigned int ele = m_uiSendOctOffsetRound1[p];
             ele < (m_uiSendOctOffsetRound1[p] + m_uiSendOctCountRound1[p]);
             ele++) {
            for (unsigned int dir = 0; dir < m_uiNumDirections; dir++) {
                elementLookup =
                    m_uiE2EMapping[(m_uiScatterMapElementRound1[ele] +
                                    m_uiElementLocalBegin) *
                                       m_uiNumDirections +
                                   dir];

                if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                    (m_uiAllElements[elementLookup].getLevel() <=
                     m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                      m_uiElementLocalBegin)]
                         .getLevel())) {
                    setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
                }
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_LEFT, OCT_DIR_DOWN, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_LEFT, OCT_DIR_UP, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_LEFT, OCT_DIR_BACK, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_LEFT, OCT_DIR_FRONT, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_RIGHT, OCT_DIR_DOWN, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_RIGHT, OCT_DIR_UP, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_RIGHT, OCT_DIR_BACK, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_RIGHT, OCT_DIR_FRONT, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_DOWN, OCT_DIR_BACK, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_DOWN, OCT_DIR_FRONT, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_UP, OCT_DIR_BACK, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }

            OCT_DIR_DIAGONAL_E2E(
                (m_uiScatterMapElementRound1[ele] + m_uiElementLocalBegin),
                OCT_DIR_UP, OCT_DIR_FRONT, elementLookup);
            if ((elementLookup != LOOK_UP_TABLE_DEFAULT) &&
                (m_uiAllElements[elementLookup].getLevel() <=
                 m_uiAllElements[(m_uiScatterMapElementRound1[ele] +
                                  m_uiElementLocalBegin)]
                     .getLevel())) {
                setHintUint = tmpSendEleIdR2[p].emplace(elementLookup);
            }
        }
    }

    m_uiSendBufferElement.clear();
    std::vector<unsigned int> common_data;
    for (unsigned int pi = 0; pi < npes; pi++) {
        tmpSendElementIds.clear();
        tmpSendElementIds.insert(tmpSendEleIdR2[pi].begin(),
                                 tmpSendEleIdR2[pi].end());

        for (auto it = tmpSendElementIds.begin(); it != tmpSendElementIds.end();
             ++it) {
            m_uiSendBufferElement.push_back(m_uiAllElements[*it]);
        }

        m_uiSendOctCountRound2[pi] = tmpSendElementIds.size();
    }

    delete[] tmpSendEleIdR2;

    par::Mpi_Alltoall(m_uiSendOctCountRound2, m_uiRecvOctCountRound2, 1, comm);
    m_uiSendOctOffsetRound2[0] = 0;
    m_uiRecvOctOffsetRound2[0] = 0;

    omp_par::scan(m_uiSendOctCountRound2, m_uiSendOctOffsetRound2, npes);
    omp_par::scan(m_uiRecvOctCountRound2, m_uiRecvOctOffsetRound2, npes);

    m_uiGhostOctants.clear();
    m_uiGhostOctants.resize(m_uiRecvOctOffsetRound2[npes - 1] +
                            m_uiRecvOctCountRound2[npes - 1]);
    assert(m_uiSendBufferElement.size() == (m_uiSendOctOffsetRound2[npes - 1] +
                                            m_uiSendOctCountRound2[npes - 1]));
    par::Mpi_Alltoallv(
        &(*(m_uiSendBufferElement.begin())), (int *)m_uiSendOctCountRound2,
        (int *)m_uiSendOctOffsetRound2, &(*(m_uiGhostOctants.begin())),
        (int *)m_uiRecvOctCountRound2, (int *)m_uiRecvOctOffsetRound2, comm);

    m_uiAllElements.insert(m_uiAllElements.end(), m_uiGhostOctants.begin(),
                           m_uiGhostOctants.end());

    tmpNodes.clear();
    SFC::seqSort::SFC_treeSort(&(*(m_uiAllElements.begin())),
                               m_uiAllElements.size(), tmpNodes, tmpNodes,
                               tmpNodes, m_uiMaxDepth, m_uiMaxDepth, rootNode,
                               0, 1, TS_REMOVE_DUPLICATES);
    std::swap(m_uiAllElements, tmpNodes);
    tmpNodes.clear();

    assert(seq::test::isUniqueAndSorted(m_uiAllElements));
    m_uiGhostOctants.clear();

    m_uiElementPreGhostBegin  = 0;
    m_uiElementPreGhostEnd    = 0;
    m_uiElementLocalBegin     = 0;
    m_uiElementLocalEnd       = 0;
    m_uiElementPostGhostBegin = 0;
    m_uiElementPostGhostEnd   = 0;

    for (unsigned int k = 0; k < m_uiAllElements.size(); k++) {
        if (m_uiAllElements[k] == localMinMaxElement[0]) {
            m_uiElementLocalBegin = k;
            break;
        }
    }

    for (unsigned int k = (m_uiAllElements.size() - 1); k > 0; k--) {
        if (m_uiAllElements[k] == localMinMaxElement[1])
            m_uiElementLocalEnd = k + 1;
    }

    m_uiElementPreGhostEnd    = m_uiElementLocalBegin;
    m_uiElementPostGhostBegin = m_uiElementLocalEnd;
    m_uiElementPostGhostEnd   = m_uiAllElements.size();

    m_uiNumLocalElements      = (m_uiElementLocalEnd - m_uiElementLocalBegin);
    m_uiNumPreGhostElements =
        (m_uiElementPreGhostEnd - m_uiElementPreGhostBegin);
    m_uiNumPostGhostElements =
        (m_uiElementPostGhostEnd - m_uiElementPostGhostBegin);

    m_uiNumTotalElements = m_uiNumPreGhostElements + m_uiNumLocalElements +
                           m_uiNumPostGhostElements;

    // E2E Mapping for the Round 2 Ghost Exchange. (Final E2E Mapping )
    SFC::seqSearch::SFC_treeSearch(&(*(m_uiKeys.begin())),
                                   &(*(m_uiAllElements.begin())), 0,
                                   m_uiKeys.size(), 0, m_uiAllElements.size(),
                                   m_uiMaxDepth, m_uiMaxDepth, 0);

    m_uiKeysPtr = &(*(m_uiKeys.begin()));
    m_uiE2EMapping.clear();
    m_uiE2EMapping.resize(m_uiAllElements.size() * m_uiNumDirections,
                          LOOK_UP_TABLE_DEFAULT);

    // * Neighbour list is made from
    // *  first x axis. (left to right)
    // *  second y axis. (down to up)
    // *  third z axis. (back to front)
    for (unsigned int k = 0; k < m_uiKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if (ownerList->size())
            assert((OCT_FOUND &
                    m_uiKeysPtr[k]
                        .getFlag()));  // Note that all the keys should be found
                                       // locally due to the fact that we have
                                       // exchanged ghost elements.
        stencilIndexDirection = m_uiKeysPtr[k].getStencilIndexDirectionList();
        result                = m_uiKeysPtr[k].getSearchResult();
        if (ownerList->size())
            assert(m_uiAllElements[result].isAncestor(m_uiKeys[k]) ||
                   m_uiAllElements[result] ==
                       m_uiKeys[k]);  // To check the result found in the
                                      // treeSearch is correct or not.
        for (unsigned int w = 0; w < ownerList->size(); w++) {
            direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
            m_uiE2EMapping[(((*ownerList)[w]) + m_uiElementLocalBegin) *
                               m_uiNumDirections +
                           direction] = result;
            // Note : Following is done to enforce that the local elements that
            // points to ghost elements has the inverse mapping.
            if (result < m_uiElementLocalBegin || result >= m_uiElementLocalEnd)
                m_uiE2EMapping[result * m_uiNumDirections + (1u ^ direction)] =
                    (((*ownerList)[w]) +
                     m_uiElementLocalBegin);  // Note This depends on the
                                              // OCT_DIR numbering.
        }
    }

    assert(seq::test::checkE2EMapping(
        m_uiE2EMapping, m_uiAllElements, m_uiElementLocalBegin,
        m_uiElementLocalEnd, 1, m_uiNumDirections));

    m_uiGhostKeys.clear();
    generateGhostElementSearchKeys();

    SFC::seqSearch::SFC_treeSearch(
        &(*(m_uiGhostKeys.begin())), &(*(m_uiAllElements.begin())), 0,
        m_uiGhostKeys.size(), 0, m_uiAllElements.size(), m_uiMaxDepth,
        m_uiMaxDepth, ROOT_ROTATION);

    m_uiKeysPtr = &(*(m_uiGhostKeys.begin()));
    // Note : Since the ghost elements are not complete it is not required to
    // find all the keys in the ghost.
    for (unsigned int k = 0; k < m_uiGhostKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();
        if ((OCT_FOUND & m_uiKeysPtr[k].getFlag())) {
            stencilIndexDirection =
                m_uiKeysPtr[k].getStencilIndexDirectionList();
            result = m_uiKeysPtr[k].getSearchResult();
            if (ownerList->size())
                assert(
                    m_uiAllElements[result].isAncestor(m_uiGhostKeys[k]) ||
                    m_uiAllElements[result] ==
                        m_uiGhostKeys[k]);  // To check the result found in the
                                            // treeSearch is correct or not.

            for (unsigned int w = 0; w < ownerList->size(); w++) {
                direction = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
                m_uiE2EMapping[(((*ownerList)[w])) * m_uiNumDirections +
                               direction] = result;
                // Note : Following is done to enforce that the local elements
                // that points to ghost elements has the inverse mapping.
            }
        }
    }

    // -------------------------------------------------------------------------

    // Note: Note that m_uiAlllNodes need not to be sorted and contains
    // duplicates globally.

    // to identify the true ghost (level 1) elements.
    // Defn: True level 1 ghost element defined as any ghost element who shares
    // a face (now edge and vertex) with local element.
    m_uiGhostElementRound1Index.clear();
    unsigned int r1_count = 0;
    m_uiGhostElementRound1Index.resize(gKeys_R1.size());

    m_uiIsNodalMapValid.clear();
    m_uiIsNodalMapValid.resize(m_uiAllElements.size(), true);

    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++) {
        if (m_uiAllElements[e] == gKeys_R1[r1_count]) {
            m_uiGhostElementRound1Index[r1_count] = e;
            r1_count++;
        } else {
            m_uiIsNodalMapValid[e] = false;
        }

        if (r1_count == gKeys_R1.size()) break;
    }

    if (r1_count < gKeys_R1.size()) {
        for (unsigned int e = m_uiElementPostGhostBegin;
             e < m_uiElementPostGhostEnd; e++) {
            if (m_uiAllElements[e] == gKeys_R1[r1_count]) {
                m_uiGhostElementRound1Index[r1_count] = e;
                r1_count++;
            } else {
                m_uiIsNodalMapValid[e] = false;
            }

            if (r1_count == gKeys_R1.size()) break;
        }
    }

    for (unsigned int i = 0; i < m_uiGhostElementRound1Index.size(); i++)
        if (!m_uiIsNodalMapValid[m_uiGhostElementRound1Index[i]])
            std::cout << "invalid nodal elemental map" << std::endl;

    // clear and compute the send & recv proc list for elemental ghost exchange

    m_uiElementSendProcList.clear();
    m_uiElementRecvProcList.clear();

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        if (m_uiSendEleCount[p] > 0) m_uiElementSendProcList.push_back(p);

        if (m_uiRecvEleCount[p] > 0) m_uiElementRecvProcList.push_back(p);
    }

    // if(m_uiActiveRank==1)
    // {
    //     for(unsigned int i=0;  i < m_uiScatterMapElementRound1.size(); i++)
    //         std::cout<<YLW<<" rank: "<<m_uiActiveRank<< " send ele :
    //         "<<m_uiAllElements[ m_uiElementLocalBegin +
    //         m_uiScatterMapElementRound1[i]]<<NRM<<"\n";

    //     for(unsigned int i=0;  i < m_uiGhostElementRound1Index.size(); i++)
    //         std::cout<<GRN<<" rank: "<<m_uiActiveRank<< " recv ele :
    //         "<<m_uiAllElements[m_uiGhostElementRound1Index[i]]<<NRM<<"\n";
    // }

    // MPI_Barrier(m_uiCommActive);

    // if(m_uiActiveRank==0)
    // {
    //     for(unsigned int i=0;  i < m_uiScatterMapElementRound1.size(); i++)
    //         std::cout<<YLW<<" rank: "<<m_uiActiveRank<< " send ele :
    //         "<<m_uiAllElements[ m_uiElementLocalBegin +
    //         m_uiScatterMapElementRound1[i]]<<NRM<<"\n";

    //     for(unsigned int i=0;  i < m_uiGhostElementRound1Index.size(); i++)
    //         std::cout<<GRN<<" rank: "<<m_uiActiveRank<< " recv ele :
    //         "<<m_uiAllElements[m_uiGhostElementRound1Index[i]]<<NRM<<"\n";
    // }

    // std::cout<<" rank: "<<m_uiActiveRank<<" m_uiGR1 Size:
    // "<<m_uiGhostElementRound1Index.size()<<std::endl;

    // BELOW CODE IS RISKY. YOU SHOULD NOT ADD R2 GHOST AS R1 GHOST WHEN
    // GENERATING SM. THIS IS NOT CORRECT. IF WE HAVE DONE THE R1 GHOST EXCHANGE
    // EXCHANGE CORRECTLY WE SHOULD NOT NEED THE BELOW CODE. KEEPING THIS IF WE
    // NEEDED A QUICK FIX. AGAIN DO NOT ENABLE THE BELOW CODE. !!!!
    /*
    //m_uiGhostElementRound1Index.clear();
    SFC::seqSearch::SFC_treeSearch(&(*(m_uiKeys.begin())),&(*(m_uiAllElements.begin())),0,m_uiKeys.size(),0,m_uiAllElements.size(),m_uiMaxDepth,m_uiMaxDepth,ROOT_ROTATION);
    SFC::seqSearch::SFC_treeSearch(&(*(m_uiKeysDiag.begin())),&(*(m_uiAllElements.begin())),0,m_uiKeysDiag.size(),0,m_uiAllElements.size(),m_uiMaxDepth,m_uiMaxDepth,ROOT_ROTATION);


    m_uiKeysPtr=&(*(m_uiKeys.begin()));
    for (unsigned int k = 0; k < m_uiKeys.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if(ownerList->size() && (!((OCT_FOUND & m_uiKeysPtr[k].getFlag())))) {
            std::cout<<"rank: "<<m_uiActiveRank<<"[E2E Error]: Local key missing
    after R1 & R1 Diag & R2  ghost exchange: "<<m_uiKeysPtr[k]<<std::endl;
            exit(0); // no point in continuing if this fails E2N fails for sure
    :)
        }

        if(ownerList->size())
        {
            result=m_uiKeysPtr[k].getSearchResult();
            if((result<m_uiElementLocalBegin) || (result>=m_uiElementLocalEnd))
                m_uiGhostElementRound1Index.push_back(result);
        }


    }

    m_uiKeysPtr=&(*(m_uiKeysDiag.begin()));
    for (unsigned int k = 0; k < m_uiKeysDiag.size(); k++) {
        ownerList = m_uiKeysPtr[k].getOwnerList();

        if(ownerList->size() && (!((OCT_FOUND & m_uiKeysPtr[k].getFlag())))) {
            std::cout<<"rank: "<<m_uiActiveRank<<"[E2E Error]: Local Diag and
    corner key missing after R1 & R1 Diag & R2  ghost exchange:
    "<<m_uiKeysPtr[k]<<std::endl; exit(0); // no point in continuing if this
    fails E2N fails for sure :)
        }

        if(ownerList->size())
        {
            result=m_uiKeysPtr[k].getSearchResult();
            if((result<m_uiElementLocalBegin) || (result>=m_uiElementLocalEnd))
                m_uiGhostElementRound1Index.push_back(result);
        }

    }
    std::sort(m_uiGhostElementRound1Index.begin(),m_uiGhostElementRound1Index.end());
    m_uiGhostElementRound1Index.erase(std::unique(m_uiGhostElementRound1Index.begin(),m_uiGhostElementRound1Index.end()),m_uiGhostElementRound1Index.end());*/

#ifdef DEBUG_MESH_GENERATION
    std::vector<ot::TreeNode> r1GhostElements;
    std::vector<ot::TreeNode> localElements;
    std::vector<ot::TreeNode> ghostElements;

    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++)
        ghostElements.push_back(m_uiAllElements[e]);

    for (unsigned int e = m_uiElementPostGhostBegin;
         e < m_uiElementPostGhostEnd; e++)
        ghostElements.push_back(m_uiAllElements[e]);

    for (unsigned int e = m_uiElementLocalBegin; e < m_uiElementLocalEnd; e++)
        localElements.push_back(m_uiAllElements[e]);

    for (unsigned int e = 0; e < m_uiGhostElementRound1Index.size(); e++)
        r1GhostElements.push_back(
            m_uiAllElements[m_uiGhostElementRound1Index[e]]);

    treeNodesTovtk(r1GhostElements, rank, "r1GhostElements");
    treeNodesTovtk(localElements, rank, "localElements");
    treeNodesTovtk(ghostElements, rank, "ghostElements");
#endif

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished building E2E Map!");
}

void Mesh::buildE2EMap(std::vector<ot::TreeNode> &in) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    /* addBoundaryNodesType1(in, m_uiPositiveBoundaryOctants, m_uiDim,
     m_uiMaxDepth); m_uiMaxDepth = m_uiMaxDepth + 1;*/

    std::swap(m_uiEmbeddedOctree, in);
    /*  m_uiEmbeddedOctree.insert(m_uiEmbeddedOctree.end(),
     * m_uiPositiveBoundaryOctants.begin(),m_uiPositiveBoundaryOctants.end());*/

    std::vector<ot::TreeNode> tmpNodes;
    ot::TreeNode rootNode(0, 0, 0, 0, m_uiDim, m_uiMaxDepth);
    // clear the in and positive boundary octants;
    // m_uiPositiveBoundaryOctants.clear();
    in.clear();

    // std::cout<<"rank: "<<m_uiActiveRank<<" par::sort begin:
    // "<<m_uiMaxDepth<<" "<<m_uiDim<<" embedded octreeSize:
    // "<<m_uiEmbeddedOctree.size()<<std::endl;
    SFC::seqSort::SFC_treeSort(&(*(m_uiEmbeddedOctree.begin())),
                               m_uiEmbeddedOctree.size(), tmpNodes, tmpNodes,
                               tmpNodes, m_uiMaxDepth, m_uiMaxDepth, rootNode,
                               0, 1, TS_REMOVE_DUPLICATES);

    std::swap(tmpNodes, m_uiEmbeddedOctree);
    tmpNodes.clear();

    generateSearchKeys();  // generates keys for sequential case.

    std::swap(m_uiAllElements, m_uiEmbeddedOctree);
    m_uiEmbeddedOctree.clear();

    // update the loop counters.
    m_uiElementPreGhostBegin  = 0;
    m_uiElementPreGhostEnd    = 0;
    m_uiElementLocalBegin     = 0;
    m_uiElementLocalEnd       = m_uiAllElements.size();
    m_uiElementPostGhostBegin = m_uiElementLocalEnd;
    m_uiElementPostGhostEnd   = m_uiElementLocalEnd;

    m_uiNumLocalElements      = (m_uiElementLocalEnd - m_uiElementLocalBegin);
    m_uiNumPreGhostElements =
        (m_uiElementPreGhostEnd - m_uiElementPreGhostBegin);
    m_uiNumPostGhostElements =
        (m_uiElementPostGhostEnd - m_uiElementPostGhostBegin);

    m_uiNumTotalElements = m_uiNumPreGhostElements + m_uiNumLocalElements +
                           m_uiNumPostGhostElements;

    m_uiLocalSplitterElements.resize(2);
    m_uiLocalSplitterElements[0] = m_uiAllElements.front();
    m_uiLocalSplitterElements[1] = m_uiAllElements.back();

    // 1b - allocate  & initialize E2E mapping.
    m_uiE2EMapping.resize(m_uiAllElements.size() * m_uiNumDirections,
                          LOOK_UP_TABLE_DEFAULT);
    SFC::seqSearch::SFC_treeSearch(&(*(m_uiKeys.begin())),
                                   &(*(m_uiAllElements.begin())), 0,
                                   m_uiKeys.size(), 0, m_uiAllElements.size(),
                                   m_uiMaxDepth, m_uiMaxDepth, 0);

    std::vector<unsigned int> *ownerList;
    std::vector<unsigned int> *stencilIndexDirection;
    unsigned int result;
    unsigned int dir;
    // unsigned int stencilIndex;
    Key *m_uiKeysPtr = &(*(m_uiKeys.begin()));

    m_uiE2EMapping.resize(m_uiAllElements.size() * m_uiNumDirections,
                          LOOK_UP_TABLE_DEFAULT);

    /**Neighbour list is made from
     *  first x axis. (left to right)
     *  second y axis. (down to up)
     *  third z axis. (back to front)**/

    for (unsigned int k = 0; k < m_uiKeys.size(); k++) {
        if (!(OCT_FOUND & m_uiKeysPtr[k].getFlag()))
            std::cout << "key: " << m_uiKeysPtr[k]
                      << " not found: " << std::endl;
        assert(
            (OCT_FOUND &
             m_uiKeysPtr[k].getFlag()));  // Note that all the keys should be
                                          // found locally due to the fact that
                                          // we have exchanged ghost elements.
        ownerList             = m_uiKeysPtr[k].getOwnerList();
        stencilIndexDirection = m_uiKeysPtr[k].getStencilIndexDirectionList();
        result                = m_uiKeysPtr[k].getSearchResult();
        assert(m_uiAllElements[result].isAncestor(m_uiKeys[k]) ||
               m_uiAllElements[result] ==
                   m_uiKeys[k]);  // To check the result found in the treeSearch
                                  // is correct or not.
        for (unsigned int w = 0; w < ownerList->size(); w++) {
            dir = ((*stencilIndexDirection)[w]) & KEY_DIR_OFFSET;
            // stencilIndex = (((*stencilIndexDirection)[w]) & (KS_MAX << 3u))
            // >> 3u; std::cout<<"dir: "<<dir<<" stencil Index:
            // "<<stencilIndex<<"owner index: "<<(*ownerList)[w]<<std::endl;
            m_uiE2EMapping[(((*ownerList)[w]) + m_uiElementLocalBegin) *
                               m_uiNumDirections +
                           dir] = result;
            // Note : Following is done to enforce that the local elements that
            // points to ghost elements has the inverse mapping.
            if (result < m_uiElementLocalBegin ||
                result >= m_uiElementLocalEnd) {
                m_uiE2EMapping[result * m_uiNumDirections + (1u ^ dir)] =
                    (((*ownerList)[w]) +
                     m_uiElementLocalBegin);  // Note This depends on the
                                              // OCT_DIR numbering.
                assert(false);  // for sequential case this cannot be true.
            }
        }
    }

    assert(seq::test::checkE2EMapping(
        m_uiE2EMapping, m_uiAllElements, m_uiElementPreGhostBegin,
        m_uiElementPostGhostEnd, 1, m_uiNumDirections));
#ifdef DEBUG_MESH_GENERATION
    treeNodesTovtk(m_uiAllElements, m_uiActiveRank, "m_uiAllElements");
    // std::cout<<"rank: "<<rank<<"pre begin: "<<m_uiElementPreGhostBegin<<" pre
    // end: "<<m_uiElementPreGhostEnd<<" local begin:
    // "<<m_uiElementLocalBegin<<" local end: "<<m_uiElementLocalEnd<<" post
    // begin: "<<m_uiElementPostGhostBegin<<" post end:
    // "<<m_uiElementPostGhostEnd<<std::endl;
#endif

    // Note: Note that m_uiAlllNodes need not to be sorted and contains
    // duplicates globally.

    m_uiIsNodalMapValid.clear();
    m_uiIsNodalMapValid.resize(m_uiAllElements.size(), true);

#ifdef DEBUG_MESH_GENERATION
    /*  std::vector<ot::TreeNode> missedKeys;
            for (unsigned int k = 0; k < m_uiKeys.size(); k++) {
                if (!(OCT_FOUND & m_uiKeys[k].getFlag())) {
                    std::cout << "rank: " << m_uiActiveRank << " key : " <<
       m_uiKeys[k] << " not found!" << " Search index count: "
                            << searchResults.size() << std::endl;
                    missedKeys.push_back(m_uiKeys[k]);
                }
            }

        treeNodesTovtk(missedKeys,rank,"missedKeys");*/
#endif

    std::cout << "Seq: E2E mapping Ended" << std::endl;
}

void Mesh::buildE2NWithSM() {
    if (!m_uiIsActive) return;

    dendro::logger::debug(dendro::logger::Scope{"MESH"},
                          "Now building E2N with the scattermap");

    // 1. first build all data structures for element order 2. (this serves as
    // auxilary data strucutre to figure out hanging node information)
    const unsigned int eleOrder = m_uiElementOrder;
    const unsigned int pp       = 2;
    m_uiElementOrder            = pp;
    if (m_uiDim == 2)
        m_uiNpE = (pp + 1) * (pp + 1);
    else if (m_uiDim == 3)
        m_uiNpE = (pp + 1) * (pp + 1) * (pp + 1);

    buildE2NMap();

    if (m_uiActiveNpes > 1) computeNodalScatterMap4(m_uiCommActive);

    // 2. Expand to full order and rebuild scatter maps
    buildE2NWithSMRepartitioned(eleOrder);

    // 3. Derive per-element ownership masks from the now-finalized
    // cascade. The masks freeze cascade's correct ownership decisions
    // into a compact, transportable form. Future repartition exchanges
    // ship masks alongside oct_data so the receiver doesn't need a
    // working post-partition cascade.
    deriveOwnerMasksFromCascade();

    // Sanity check: at this point (initial SFC mesh build) the
    // cascade is canonically correct, and mask derivation just sampled
    // it. Validation should report zero disagreements.
    static const char* mask_dbg_env =
        std::getenv("DENDRO_VALIDATE_MASK");
    if (mask_dbg_env && mask_dbg_env[0] == '1'
        && mask_dbg_env[1] == '\0') {
        const size_t nDisagree =
            this->validateOwnerMasksAgainstCurrentCascade();
        std::cout << "[mask-validate r" << m_uiActiveRank
                  << "] post-buildE2NWithSM (SFC) cascade vs mask"
                  << " disagreements=" << nDisagree
                  << " (expect 0)" << std::endl;
    }
}

void Mesh::buildE2NWithSMRepartitioned(unsigned int eleOrder) {
    if (!m_uiIsActive) return;

    int rank  = m_uiActiveRank;
    int npes  = m_uiActiveNpes;
    const unsigned int pp = 2;

    // Use face edge vertex hanging information to modifying the data
    // strucutres to the specified element order.
    std::vector<unsigned int> e2n_dg;
    std::vector<unsigned int> e2n_cg;

    const unsigned int nPe_1d = (eleOrder + 1);
    const unsigned int nPe_2d = (eleOrder + 1) * (eleOrder + 1);
    const unsigned int nPe_3d =
        (eleOrder + 1) * (eleOrder + 1) * (eleOrder + 1);

    e2n_dg.resize(nPe_3d * m_uiNumTotalElements);
    e2n_cg.resize(nPe_3d * m_uiNumTotalElements);

    unsigned int ownerID, ii_x, jj_y, kk_z;
// idx for the element order 2
#define IDX2(i, j, k) k *(pp + 1) * (pp + 1) + j *(pp + 1) + i

// idx for the element order p.
#define IDXp(i, j, k) k *(eleOrder + 1) * (eleOrder + 1) + j *(eleOrder + 1) + i

    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPostGhostEnd;
         e++) {
        for (unsigned int n = 0; n < nPe_3d; n++)
            e2n_dg[e * (nPe_3d) + n] = e * (nPe_3d) + n;

        // OCT_DIR_LEFT
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 1, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            for (unsigned int d2 = 0; d2 < nPe_1d; d2++)
                for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                    e2n_dg[e * (nPe_3d) + IDXp(0, d1, d2)] =
                        ownerID * (nPe_3d) + IDXp(eleOrder, d1, d2);
        }

        // OCT_DIR_RIGHT
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 1, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.
            for (unsigned int d2 = 0; d2 < nPe_1d; d2++)
                for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                    e2n_dg[e * (nPe_3d) + IDXp(eleOrder, d1, d2)] =
                        ownerID * (nPe_3d) + IDXp(0, d1, d2);
        }

        // OCT_DIR_DOWN
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 0, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.
            for (unsigned int d2 = 0; d2 < nPe_1d; d2++)
                for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                    e2n_dg[e * (nPe_3d) + IDXp(d1, 0, d2)] =
                        ownerID * (nPe_3d) + IDXp(d1, eleOrder, d2);
        }

        // OCT_DIR_UP
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 2, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.
            for (unsigned int d2 = 0; d2 < nPe_1d; d2++)
                for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                    e2n_dg[e * (nPe_3d) + IDXp(d1, eleOrder, d2)] =
                        ownerID * (nPe_3d) + IDXp(d1, 0, d2);
        }

        // OCT_DIR_BACK
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 1, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.
            for (unsigned int d2 = 0; d2 < nPe_1d; d2++)
                for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                    e2n_dg[e * (nPe_3d) + IDXp(d1, d2, 0)] =
                        ownerID * (nPe_3d) + IDXp(d1, d2, eleOrder);
        }

        // OCT_DIR_FRONT
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 1, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.
            for (unsigned int d2 = 0; d2 < nPe_1d; d2++)
                for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                    e2n_dg[e * (nPe_3d) + IDXp(d1, d2, eleOrder)] =
                        ownerID * (nPe_3d) + IDXp(d1, d2, 0);
        }

        // LEFT FACE EDGES
        // --------------------------------------------------------
        unsigned int f1 = 0;
        unsigned int f2 = 0;

        // LEFT FACE EDGES
        // --------------------------------------------------------

        // OCT_DIR_LEFT_DOWN
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 0, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {
            assert(kk_z == 1);
            // e does not own the edge.
            f1 = (ii_x * eleOrder) / pp;
            f2 = (jj_y * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(0, 0, d1)] =
                    ownerID * (nPe_3d) + IDXp(f1, f2, d1);
        }

        // OCT_DIR_LEFT_UP
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 2, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (ii_x * eleOrder) / pp;
            f2 = (jj_y * eleOrder) / pp;
            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(0, eleOrder, d1)] =
                    ownerID * (nPe_3d) + IDXp(f1, f2, d1);
        }

        // OCT_DIR_LEFT_BACK
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 1, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {
            // e does not own the face.
            f1 = (ii_x * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;
            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(0, d1, 0)] =
                    ownerID * (nPe_3d) + IDXp(f1, d1, f2);
        }

        // OCT_DIR_LEFT_FRONT
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 1, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (ii_x * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(0, d1, eleOrder)] =
                    ownerID * (nPe_3d) + IDXp(f1, d1, f2);
        }

        // RIGHT FACE EDGES
        // -------------------------------------------------------

        // OCT_DIR_RIGHT_DOWN
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 0, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (ii_x * eleOrder) / pp;
            f2 = (jj_y * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(eleOrder, 0, d1)] =
                    ownerID * (nPe_3d) + IDXp(f1, f2, d1);
        }

        // OCT_DIR_RIGHT_UP
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 2, 1)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (ii_x * eleOrder) / pp;
            f2 = (jj_y * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(eleOrder, eleOrder, d1)] =
                    ownerID * (nPe_3d) + IDXp(f1, f2, d1);
        }

        // OCT_DIR_RIGHT_BACK
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 1, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {
            // e does not own the face.
            f1 = (ii_x * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;
            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(eleOrder, d1, 0)] =
                    ownerID * (nPe_3d) + IDXp(f1, d1, f2);
        }

        // OCT_DIR_RIGHT_FRONT
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 1, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (ii_x * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;
            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(eleOrder, d1, eleOrder)] =
                    ownerID * (nPe_3d) + IDXp(f1, d1, f2);
        }

        // BACK FACE EDGES ----------------------------------------------------

        // OCT_DIR_BACK_DOWN
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 0, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (jj_y * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(d1, 0, 0)] =
                    ownerID * (nPe_3d) + IDXp(d1, f1, f2);
        }

        // OCT_DIR_BACK_UP
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 2, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (jj_y * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(d1, eleOrder, 0)] =
                    ownerID * (nPe_3d) + IDXp(d1, f1, f2);
        }

        // FRONT FACE EDGES ----------------------------------------------------

        // OCT_DIR_FRONT_DOWN
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 0, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (jj_y * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(d1, 0, eleOrder)] =
                    ownerID * (nPe_3d) + IDXp(d1, f1, f2);
        }

        // OCT_DIR_FRONT_UP
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(1, 2, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID) {  // e does not own the face.

            f1 = (jj_y * eleOrder) / pp;
            f2 = (kk_z * eleOrder) / pp;

            for (unsigned int d1 = 0; d1 < nPe_1d; d1++)
                e2n_dg[e * (nPe_3d) + IDXp(d1, eleOrder, eleOrder)] =
                    ownerID * (nPe_3d) + IDXp(d1, f1, f2);
        }

        // VERTICES  --  (coner 2^dim)
        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 0, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(0, 0, 0)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);

        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 0, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(eleOrder, 0, 0)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);

        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 2, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(0, eleOrder, 0)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);

        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 2, 0)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(eleOrder, eleOrder, 0)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);

        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 0, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(0, 0, eleOrder)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);

        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 0, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(eleOrder, 0, eleOrder)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);

        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(0, 2, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(0, eleOrder, eleOrder)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);

        dg2eijk(m_uiE2NMapping_DG[e * m_uiNpE + IDX2(2, 2, 2)], ownerID, ii_x,
                jj_y, kk_z);
        if (e != ownerID)
            e2n_dg[e * (nPe_3d) + IDXp(eleOrder, eleOrder, eleOrder)] =
                ownerID * (nPe_3d) + IDXp((ii_x * eleOrder) / pp,
                                          (jj_y * eleOrder) / pp,
                                          (kk_z * eleOrder) / pp);
    }

    e2n_cg = e2n_dg;
    std::sort(e2n_dg.begin(), e2n_dg.end());
    e2n_dg.erase(std::unique(e2n_dg.begin(), e2n_dg.end()), e2n_dg.end());

    std::vector<unsigned int> cg2dg;
    cg2dg.resize(e2n_dg.size());
    cg2dg = e2n_dg;

    e2n_dg.resize(nPe_3d * m_uiNumTotalElements);
    e2n_dg = e2n_cg;

    std::vector<unsigned int> dg2cg;
    dg2cg.resize(nPe_3d * m_uiNumTotalElements, LOOK_UP_TABLE_DEFAULT);

    for (unsigned int i = 0; i < cg2dg.size(); i++) dg2cg[cg2dg[i]] = i;

    for (unsigned int i = 0; i < e2n_cg.size(); i++)
        e2n_cg[i] = dg2cg[e2n_cg[i]];

    // for(unsigned int i=0;i< e2n_cg.size(); i++)
    // {
    //     std::cout<<"i: "<<i<<" e2n_dg: "<<e2n_dg[i]<<"
    //     m_uiE2N_DG:"<<m_uiE2NMapping_DG[i]<<std::endl;;
    // }
    const unsigned int numCGNodes = cg2dg.size();

    unsigned int n_dg;
    unsigned int dir;
    unsigned int ib, ie, jb, je, kb, ke;

    if (m_uiActiveNpes > 1) {
        std::vector<unsigned int> sendNodeCount;
        std::vector<unsigned int> recvNodeCount;
        std::vector<unsigned int> sendNodeSM;
        std::vector<unsigned int> recvNodeSM;

        sendNodeCount.resize(m_uiActiveNpes, 0);
        recvNodeCount.resize(m_uiActiveNpes, 0);

        // send SM
        for (unsigned int m = 0; m < m_uiActiveNpes; m++) {
            for (unsigned int n = m_uiSendNodeOffset[m];
                 n < (m_uiSendNodeOffset[m] + m_uiSendNodeCount[m]); n++) {
                n_dg = m_uiCG2DG[m_uiScatterMapActualNodeSend[n]];
                dg2eijk(n_dg, ownerID, ii_x, jj_y, kk_z);

                // std::cout << " owner "<<ownerID<<" ii_x: "<<ii_x<<" jj_y:
                // "<<jj_y<<" kk_z: "<<kk_z<<std::endl;

                if (ii_x == 0) {
                    ib = 0;
                    ie = 0;
                }
                if (jj_y == 0) {
                    jb = 0;
                    je = 0;
                }
                if (kk_z == 0) {
                    kb = 0;
                    ke = 0;
                }

                if (ii_x == 1) {
                    ib = 1;
                    ie = eleOrder - 1;
                }
                if (jj_y == 1) {
                    jb = 1;
                    je = eleOrder - 1;
                }
                if (kk_z == 1) {
                    kb = 1;
                    ke = eleOrder - 1;
                }

                if (ii_x == 2) {
                    ib = eleOrder;
                    ie = eleOrder;
                }
                if (jj_y == 2) {
                    jb = eleOrder;
                    je = eleOrder;
                }
                if (kk_z == 2) {
                    kb = eleOrder;
                    ke = eleOrder;
                }

                for (unsigned int k = kb; k <= ke; k++)
                    for (unsigned int j = jb; j <= je; j++)
                        for (unsigned int i = ib; i <= ie; i++) {
                            sendNodeCount[m]++;
                            sendNodeSM.push_back(
                                e2n_cg[ownerID * nPe_3d + IDXp(i, j, k)]);
                        }
            }
        }

        // recv SM
        for (unsigned int m = 0; m < m_uiActiveNpes; m++) {
            for (unsigned int n = m_uiRecvNodeOffset[m];
                 n < (m_uiRecvNodeOffset[m] + m_uiRecvNodeCount[m]); n++) {
                n_dg = m_uiCG2DG[m_uiScatterMapActualNodeRecv[n]];
                dg2eijk(n_dg, ownerID, ii_x, jj_y, kk_z);

                if (ii_x == 0) {
                    ib = 0;
                    ie = 0;
                }
                if (jj_y == 0) {
                    jb = 0;
                    je = 0;
                }
                if (kk_z == 0) {
                    kb = 0;
                    ke = 0;
                }

                if (ii_x == 1) {
                    ib = 1;
                    ie = eleOrder - 1;
                }
                if (jj_y == 1) {
                    jb = 1;
                    je = eleOrder - 1;
                }
                if (kk_z == 1) {
                    kb = 1;
                    ke = eleOrder - 1;
                }

                if (ii_x == 2) {
                    ib = eleOrder;
                    ie = eleOrder;
                }
                if (jj_y == 2) {
                    jb = eleOrder;
                    je = eleOrder;
                }
                if (kk_z == 2) {
                    kb = eleOrder;
                    ke = eleOrder;
                }

                for (unsigned int k = kb; k <= ke; k++)
                    for (unsigned int j = jb; j <= je; j++)
                        for (unsigned int i = ib; i <= ie; i++) {
                            recvNodeCount[m]++;
                            recvNodeSM.push_back(
                                e2n_cg[ownerID * nPe_3d + IDXp(i, j, k)]);
                        }
            }
        }

        // for(unsigned int k=0 ; k < m_uiScatterMapActualNodeSend.size(); k++)
        // {
        //     std::cout<<" k : "<<k<< " scatter map :
        //     "<<m_uiScatterMapActualNodeSend[k]<<" "<<sendNodeSM[k]<<"
        //     "<<std::endl;
        // }

        // for(unsigned int k=0 ; k < m_uiScatterMapActualNodeRecv.size(); k++)
        // {
        //     std::cout<<" k : "<<k<< " scatter map :
        //     "<<m_uiScatterMapActualNodeRecv[k]<<" "<<recvNodeSM[k]<<"
        //     "<<std::endl;
        // }

        // up date the scatter maps.
        std::swap(m_uiSendNodeCount, sendNodeCount);
        std::swap(m_uiRecvNodeCount, recvNodeCount);

        m_uiSendNodeOffset[0] = 0;
        m_uiRecvNodeOffset[0] = 0;

        omp_par::scan(m_uiSendNodeCount.data(), m_uiSendNodeOffset.data(),
                      m_uiActiveNpes);
        omp_par::scan(m_uiRecvNodeCount.data(), m_uiRecvNodeOffset.data(),
                      m_uiActiveNpes);

        std::swap(m_uiScatterMapActualNodeSend, sendNodeSM);
        std::swap(m_uiScatterMapActualNodeRecv, recvNodeSM);
    }

    // update the nodal bounds.
    m_uiNumActualNodes = numCGNodes;
    m_uiElementOrder   = eleOrder;

    if (m_uiDim == 2)
        m_uiNpE = (m_uiElementOrder + 1) * (m_uiElementOrder + 1);
    else
        m_uiNpE = (m_uiElementOrder + 1) * (m_uiElementOrder + 1) *
                  (m_uiElementOrder + 1);

    std::swap(m_uiE2NMapping_CG, e2n_cg);
    std::swap(m_uiE2NMapping_DG, e2n_dg);
    std::swap(m_uiCG2DG, cg2dg);
    std::swap(m_uiDG2CG, dg2cg);

    m_uiNodePreGhostBegin   = UINT_MAX;
    m_uiNodeLocalBegin      = UINT_MAX;
    m_uiNodePostGhostBegin  = UINT_MAX;

    unsigned int preOwner   = UINT_MAX;
    unsigned int localOwner = UINT_MAX;
    unsigned int postOwner  = UINT_MAX;

    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPostGhostEnd;
         e++) {
        unsigned int tmpIndex;
        for (unsigned int k = 0; k < m_uiNpE; k++) {
            tmpIndex = (m_uiE2NMapping_DG[e * m_uiNpE + k] / m_uiNpE);
            if ((tmpIndex >= m_uiElementPreGhostBegin) &&
                (tmpIndex < m_uiElementPreGhostEnd) &&
                /*(preOwner>=(m_uiE2NMapping_CG[e * m_uiNpE + k])/m_uiNpE) &&*/
                (m_uiNodePreGhostBegin > m_uiE2NMapping_DG[e * m_uiNpE + k])) {
                // preOwner = m_uiE2NMapping_CG[e * m_uiNpE + k]/m_uiNpE;
                m_uiNodePreGhostBegin = m_uiE2NMapping_DG[e * m_uiNpE + k];
            }

            if ((tmpIndex >= m_uiElementLocalBegin) &&
                (tmpIndex < m_uiElementLocalEnd) &&
                /*(localOwner >=(m_uiE2NMapping_CG[e * m_uiNpE + k])/m_uiNpE)
                   &&*/
                (m_uiNodeLocalBegin > m_uiE2NMapping_DG[e * m_uiNpE + k])) {
                // localOwner = m_uiE2NMapping_CG[e * m_uiNpE + k]/m_uiNpE;
                m_uiNodeLocalBegin = m_uiE2NMapping_DG[e * m_uiNpE + k];
            }

            if ((tmpIndex >= m_uiElementPostGhostBegin) &&
                (tmpIndex < m_uiElementPostGhostEnd) &&
                /*(postOwner >=(m_uiE2NMapping_CG[e * m_uiNpE + k])/m_uiNpE)
                   &&*/
                (m_uiNodePostGhostBegin > m_uiE2NMapping_DG[e * m_uiNpE + k])) {
                // postOwner = m_uiE2NMapping_CG[e * m_uiNpE + k]/m_uiNpE;
                m_uiNodePostGhostBegin = m_uiE2NMapping_DG[e * m_uiNpE + k];
            }
        }
    }

    assert(m_uiNodeLocalBegin !=
           UINT_MAX);  // local node begin should be found.
    assert(m_uiDG2CG[m_uiNodeLocalBegin] != LOOK_UP_TABLE_DEFAULT);
    m_uiNodeLocalBegin = m_uiDG2CG
        [m_uiNodeLocalBegin];  //(std::lower_bound(E2N_DG_Sorted.begin(),E2N_DG_Sorted.end(),m_uiNodeLocalBegin)-E2N_DG_Sorted.begin());
    if (m_uiNodePreGhostBegin == UINT_MAX) {
        m_uiNodePreGhostBegin = 0;
        m_uiNodePreGhostEnd   = 0;
        assert(m_uiNodeLocalBegin == 0);
    } else {
        assert(m_uiDG2CG[m_uiNodePreGhostBegin] != LOOK_UP_TABLE_DEFAULT);
        m_uiNodePreGhostBegin = m_uiDG2CG
            [m_uiNodePreGhostBegin];  //(std::lower_bound(E2N_DG_Sorted.begin(),E2N_DG_Sorted.end(),m_uiNodePreGhostBegin)-E2N_DG_Sorted.begin());
        m_uiNodePreGhostEnd = m_uiNodeLocalBegin;
    }

    if (m_uiNodePostGhostBegin == UINT_MAX) {
        m_uiNodeLocalEnd       = m_uiCG2DG.size();  // E2N_DG_Sorted.size();
        m_uiNodePostGhostBegin = m_uiNodeLocalEnd;
        m_uiNodePostGhostEnd   = m_uiNodeLocalEnd;
    } else {
        assert(m_uiDG2CG[m_uiNodePostGhostBegin] != LOOK_UP_TABLE_DEFAULT);
        m_uiNodePostGhostBegin = m_uiDG2CG
            [m_uiNodePostGhostBegin];  //(std::lower_bound(E2N_DG_Sorted.begin(),E2N_DG_Sorted.end(),m_uiNodePostGhostBegin)-E2N_DG_Sorted.begin());
        m_uiNodeLocalEnd     = m_uiNodePostGhostBegin;
        m_uiNodePostGhostEnd = m_uiCG2DG.size();  // E2N_DG_Sorted.size();
    }

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished building E2N map!");

    // orphan fix: find local CGs with no local-element E2N_CG
    // reference, then redirect references (from ghost elements) to
    // the "real" CG at the same physical position.
    //
    // Orphans arise because the face/edge/vertex rewiring writes to
    // the SAME e2n_dg slot multiple times (corners overlap all of
    // face/edge/vertex rewrites; last-write wins). The canonical DG
    // ends up different depending on which rewrite processed last.
    // If a local element's sub cascade terminates at self-DG
    // initially (before rewiring), but all its rewrite passes move
    // it elsewhere, AND a ghost element still points at the self-DG
    // via its own rewriting, sort-unique keeps the self-DG. It
    // becomes a local CG that no local element references — orphan.
    //
    // Fix: for each orphan CG X, find the "real" CG Y that the local
    // canonical element actually points to. Redirect every E2N_CG
    // entry that references X (on any element) to Y instead.
    //
    // env gate: DENDRO_DISABLE_ORPHAN_FIX=1 skips the orphan-fix.
    // Lets us A/B-test it without recompiling.
    const char* _orphan_env = std::getenv("DENDRO_DISABLE_ORPHAN_FIX");
    const bool _orphan_enabled =
        !(_orphan_env && _orphan_env[0] == '1' && _orphan_env[1] == '\0');
    if (_orphan_enabled && m_uiIsActive) {
        const unsigned int npe = m_uiNpE;
        const unsigned int nLB = m_uiNodeLocalBegin;
        const unsigned int nLE = m_uiNodeLocalEnd;
        std::vector<unsigned char> refByLocal(nLE - nLB, 0);
        for (unsigned int e = m_uiElementLocalBegin;
             e < m_uiElementLocalEnd; e++) {
            for (unsigned int n = 0; n < npe; n++) {
                unsigned int cg = m_uiE2NMapping_CG[e * npe + n];
                if (cg >= nLB && cg < nLE) refByLocal[cg - nLB] = 1;
            }
        }

        std::unordered_map<unsigned int, unsigned int> redirect;
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            if (refByLocal[cg - nLB]) continue;
            unsigned int dg = m_uiCG2DG[cg];
            unsigned int A   = dg / npe;
            unsigned int n_A = dg % npe;
            if (A < m_uiElementLocalBegin || A >= m_uiElementLocalEnd)
                continue;
            unsigned int realCG = m_uiE2NMapping_CG[A * npe + n_A];
            if (realCG == cg) continue;
            redirect[cg] = realCG;
        }

        if (!redirect.empty()) {
            for (auto& kv : redirect) {
                unsigned int y = kv.second;
                while (true) {
                    auto it = redirect.find(y);
                    if (it == redirect.end()) break;
                    y = it->second;
                }
                kv.second = y;
            }
            auto applyRedirect = [&](std::vector<unsigned int>& vec) {
                size_t n = 0;
                for (auto& v : vec) {
                    auto it = redirect.find(v);
                    if (it != redirect.end()) {
                        v = it->second;
                        ++n;
                    }
                }
                return n;
            };
            size_t rewiredE2N   = applyRedirect(m_uiE2NMapping_CG);
            size_t rewiredSend  = applyRedirect(m_uiScatterMapActualNodeSend);
            size_t rewiredRecv  = applyRedirect(m_uiScatterMapActualNodeRecv);
            std::cout << "[orphan-fix r" << m_uiActiveRank
                      << "] orphans=" << redirect.size()
                      << " e2n=" << rewiredE2N
                      << " sendSM=" << rewiredSend
                      << " recvSM=" << rewiredRecv << std::endl;
        }
    }

    // Pass A: restore E2N_DG <-> cg2dg consistency.
    //
    // After all face/edge/vertex/corner cascades and the orphan-fix
    // above, some local cg's may have cg2dg[cg] = (e,n) where
    // E2N_DG[e*NpE+n] != e*NpE+n. zip uses E2N_DG self-ownership to
    // decide canonical writers (mesh.tcc:5356-5360); a stale cg2dg
    // makes zip skip the slot, leaving it at 0.
    //
    // Walk every local element and pick, for each cg in the local
    // range, the smallest (e,n) pair where E2N_DG self-owns and
    // E2N_CG == cg. Update cg2dg accordingly. Slots with no local
    // self-owned writer get stashed in m_uiPassACgsWithoutLocalCanonical
    // for cross-rank reconciliation by reconcileCrossRankDuplicateCGs.
    //
    // For "no local canonical" cgs, RESTORE the per-rank invariant
    // "every local cg has a local element with E2N_DG self-owned" by
    // promoting one local (e_local, sub) to canonical: rewrite
    // E2N_DG[e_local*NpE+sub] = e_local*NpE+sub. Also redirect every
    // OTHER (elem, sub) on this rank that references this cg via
    // E2N_CG to point its E2N_DG at the new canonical, so the
    // cascade is consistent with the rewrite. This guarantees the
    // cg-owner rank's zip writes the cg from a local element (the
    // "Path 1" structural invariant).
    //
    // SFC partitioning satisfies this invariant by construction;
    // graph cascades can violate it because per-rank cascades
    // independently route canonical writers and may put one outside
    // the cg-owner rank's local element set.
    m_uiPassACgsWithoutLocalCanonical.clear();
    // Pass A default-OFF as of 2026-05-28: with the orphan-fill cg2dg-key
    // fix + Fix B + post-axpy sync, Pass A is redundant for both EM4 and
    // NLSM (bit-identical with it disabled — see
    // project_nlsm_needs_full_machinery). Re-enable with
    // DENDRO_ENABLE_PASS_A=1 for A/B without recompiling.
    const char* _passA_env = std::getenv("DENDRO_ENABLE_PASS_A");
    const bool _passA_enabled =
        (_passA_env && _passA_env[0] == '1' && _passA_env[1] == '\0');
    if (_passA_enabled && m_uiIsActive) {
        const unsigned int npe = m_uiNpE;
        const unsigned int nLB = m_uiNodeLocalBegin;
        const unsigned int nLE = m_uiNodeLocalEnd;

        std::vector<unsigned int> candE(nLE - nLB, LOOK_UP_TABLE_DEFAULT);
        std::vector<unsigned int> candN(nLE - nLB, LOOK_UP_TABLE_DEFAULT);
        // Fallback candidates: any local (e,n) where E2N_CG==cg
        // (not requiring self-owned). Used to promote a local elem
        // to canonical when no local self-owned candidate exists.
        std::vector<unsigned int> fallbackE(nLE - nLB,
                                            LOOK_UP_TABLE_DEFAULT);
        std::vector<unsigned int> fallbackN(nLE - nLB,
                                            LOOK_UP_TABLE_DEFAULT);

        for (unsigned int e = m_uiElementLocalBegin;
             e < m_uiElementLocalEnd; e++) {
            for (unsigned int n = 0; n < npe; n++) {
                const unsigned int cg = m_uiE2NMapping_CG[e * npe + n];
                if (cg < nLB || cg >= nLE) continue;     // not local cg
                unsigned int idx = cg - nLB;
                const unsigned int dg = m_uiE2NMapping_DG[e * npe + n];
                if (dg == e * npe + n) {
                    if (candE[idx] == LOOK_UP_TABLE_DEFAULT) {
                        candE[idx] = e;
                        candN[idx] = n;
                    }
                }
                if (fallbackE[idx] == LOOK_UP_TABLE_DEFAULT) {
                    fallbackE[idx] = e;
                    fallbackN[idx] = n;
                }
            }
        }

        // Path 1 attempts (promote local elem to canonical;
        // orphan-redirect E2N_CG to local cg) made things worse
        // because they CREATED cross-rank duplicate writers: two
        // ranks each promoted their own local elem at the same
        // phys_pos, computed different rhs values from their
        // different blocks (not just different padding — bflag
        // logic and one-sided stencils can differ across ranks),
        // and ghost exchange surfaced the conflict.
        //
        // For now we do only the safe step: sync cg2dg with the
        // existing local self-owned writer when one exists. Cgs
        // without a local self-owned writer are stashed for
        // diagnosis; their CG slots remain unwritten (zip skips).
        //
        // The proper structural fix requires global cg-ownership
        // consolidation (one owner rank per phys_pos), which is
        // beyond this iteration.
        size_t rewritten = 0, missing = 0;
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            unsigned int idx = cg - nLB;
            if (candE[idx] == LOOK_UP_TABLE_DEFAULT) {
                missing++;
                m_uiPassACgsWithoutLocalCanonical.push_back(cg);
                continue;
            }
            const unsigned int newDG = candE[idx] * npe + candN[idx];
            const unsigned int oldDG = m_uiCG2DG[cg];
            if (oldDG != newDG) {
                if (oldDG < m_uiDG2CG.size()
                    && m_uiDG2CG[oldDG] == cg)
                    m_uiDG2CG[oldDG] = LOOK_UP_TABLE_DEFAULT;
                m_uiCG2DG[cg]    = newDG;
                m_uiDG2CG[newDG] = cg;
                rewritten++;
            }
        }

        if (rewritten || missing) {
            std::cout << "[passA r" << m_uiActiveRank
                      << "] cg2dg-rewrites=" << rewritten
                      << " no-local-canonical=" << missing
                      << std::endl;
        }
    }

    // Pass D + Pass E default-OFF as of 2026-05-28: redundant for both
    // EM4 and NLSM given the orphan-fill cg2dg-key fix + Fix B + post-axpy
    // sync (bit-identical with them disabled; the old "NLSM regresses
    // without Pass D" reason is subsumed by those fixes — see
    // project_nlsm_needs_full_machinery). Re-enable with
    // DENDRO_ENABLE_PASS_DE=1 for A/B without recompiling.
    const char* _passDE_env = std::getenv("DENDRO_ENABLE_PASS_DE");
    const bool _passDE_enabled =
        (_passDE_env && _passDE_env[0] == '1' && _passDE_env[1] == '\0');

    // Pass D: cross-rank cg-ownership consolidation.
    //
    // Graph cascades can leave the same phys_pos OWNED (in [nLB, nLE))
    // by MULTIPLE ranks. Each rank's zip writes a different value
    // (different blocks → different stencil padding sources) to its
    // local cg, ghost exchange surfaces the conflict, and the
    // SFC-vs-graph trajectory diverges.
    //
    // Fix: per phys_pos, smallest claiming rank wins. Loser ranks'
    // local cgs are demoted: every (elem, sub) on the loser rank
    // referencing the local cg via E2N_CG gets redirected to a
    // ghost cg at the same phys_pos. The ghost cg's value comes
    // from the winner via standard ghost exchange.
    //
    // SFC partitions don't trigger Pass D (cascades naturally
    // assign each phys_pos to exactly one rank). On graph, EM4
    // observes ~28 cross-rank duplicates per rank and diverges
    // from SFC by 1.7e-9 at step 1; Pass D should bring this to
    // machine precision.
    if (_passDE_enabled && m_uiIsActive && m_uiActiveNpes > 1) {
        const unsigned int npe = m_uiNpE;
        const unsigned int eOrd = m_uiElementOrder;
        const unsigned int nLB = m_uiNodeLocalBegin;
        const unsigned int nLE = m_uiNodeLocalEnd;
        const auto* pN = m_uiAllElements.data();
        const size_t nElTot = m_uiAllElements.size();
        MPI_Comm comm = m_uiCommActive;

        // Encode phys_pos as 3 uint64_t (integer scaled coords).
        auto encodeKey = [&](unsigned int e, unsigned int n,
                             unsigned long long& x,
                             unsigned long long& y,
                             unsigned long long& z) {
            unsigned long long len =
                (unsigned long long)1
                << (m_uiMaxDepth - pN[e].getLevel());
            unsigned int ni = n % (eOrd + 1);
            unsigned int nj = (n / (eOrd + 1)) % (eOrd + 1);
            unsigned int nk = n / ((eOrd + 1) * (eOrd + 1));
            x = (unsigned long long)pN[e].getX() * eOrd
                + (unsigned long long)ni * len;
            y = (unsigned long long)pN[e].getY() * eOrd
                + (unsigned long long)nj * len;
            z = (unsigned long long)pN[e].getZ() * eOrd
                + (unsigned long long)nk * len;
        };

        // Per local cg X: phys_pos via cg2dg's (e, n). Also compute
        // the smallest-TreeNode (deterministic global-canonical
        // surrogate) of any LOCAL element on this rank that has a
        // sub at this phys_pos. Pinning to the smallest-TreeNode-elem
        // matches SFC's natural cascade choice closely enough to
        // bring the np=4 EM4 face residual from 1.157e-7 to 2.225e-8.
        //
        // NOTE: The TRUE dendro cascade rule is "smallest level then
        // ot::TreeNode::operator< (Hilbert NCA)" -- see
        // meshE2NUtils.tcc:1919-1927 CORNER_NODE_MAP. We tried it
        // (2026-04-27) and the rescue picked SFC-matching TreeNodes
        // with bit-identical writer rhs — yet the vec residual
        // REGRESSED to baseline. Root cause: Pass D's demote-to-
        // ghost lookup picks "first-found ghost cg at phys_pos" via
        // element walk, which does NOT necessarily resolve to the
        // winner rank's ghost view. With the cascade rule, the
        // winner shifts to ranks that don't have a clean ghost-cg
        // path on the loser's side, so the sync mirrors a wrong
        // value. With smallest-packTN, the winner happens to have a
        // pre-existing ghost cg on the loser. See
        // docs/cascade_rule_match_2026-04-27.md for full diagnosis.
        // Fixing this requires explicit scatter-map-aware demote-to-
        // ghost lookup; out of scope for this commit. Returning to
        // smallest-packTN for now.
        //
        // TreeNode encoding for ordering: (level, X, Y, Z) packed
        // into a single uint64. Smaller level -> smaller packed val
        // (coarser-side wins, matching SFC).
        auto packTN = [&](unsigned int e) -> unsigned long long {
            unsigned long long lev = (unsigned long long)pN[e].getLevel()
                                     & 0xFFULL;
            unsigned long long X = (unsigned long long)pN[e].getX()
                                   & 0xFFFFFFFULL;
            unsigned long long Y = (unsigned long long)pN[e].getY()
                                   & 0xFFFFFFFULL;
            return (lev << 56) | (X << 28) | Y;
        };

        // Build a phys_pos-indexed local map of "smallest TreeNode
        // of any local elem at this phys_pos" — used to advertise.
        const unsigned long long PACK_INF = ~0ULL;
        // Two TN advertisements per phys_pos:
        //  - myMinTN: smallest packTN of any local elem at phys_pos.
        //  - myMinTN_self: smallest packTN with self-owned filter
        //    (E2N_DG[e*npe+n] == e*npe+n at this phys_pos). Used to
        //    detect phantom-cascade phys_pos (no rank globally has a
        //    writer) and pick a winner that actually CAN write.
        std::unordered_map<PhysKey3, unsigned long long, PhysKey3Hash>
            myMinTN, myMinTN_self;
        for (unsigned int e = m_uiElementLocalBegin;
             e < m_uiElementLocalEnd; e++) {
            unsigned long long tn = packTN(e);
            for (unsigned int n = 0; n < npe; n++) {
                unsigned long long x, y, z;
                encodeKey(e, n, x, y, z);
                PhysKey3 k{x, y, z};
                auto it = myMinTN.find(k);
                if (it == myMinTN.end() || tn < it->second)
                    myMinTN[k] = tn;
                if (m_uiE2NMapping_DG[e * npe + n] == e * npe + n) {
                    auto its = myMinTN_self.find(k);
                    if (its == myMinTN_self.end() || tn < its->second)
                        myMinTN_self[k] = tn;
                }
            }
        }

        std::vector<unsigned long long> myX, myY, myZ, myMinTNArr,
            myMinTNSelfArr;
        std::vector<unsigned int> myCg;
        myX.reserve(nLE - nLB);
        myY.reserve(nLE - nLB);
        myZ.reserve(nLE - nLB);
        myMinTNArr.reserve(nLE - nLB);
        myMinTNSelfArr.reserve(nLE - nLB);
        myCg.reserve(nLE - nLB);
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            unsigned int dg = m_uiCG2DG[cg];
            if (dg == LOOK_UP_TABLE_DEFAULT) continue;
            unsigned int e = dg / npe;
            unsigned int n = dg % npe;
            if (e >= nElTot) continue;
            unsigned long long x, y, z;
            encodeKey(e, n, x, y, z);
            unsigned long long minTN = PACK_INF;
            unsigned long long minTNSelf = PACK_INF;
            auto it = myMinTN.find(PhysKey3{x, y, z});
            if (it != myMinTN.end()) minTN = it->second;
            auto its = myMinTN_self.find(PhysKey3{x, y, z});
            if (its != myMinTN_self.end()) minTNSelf = its->second;
            myX.push_back(x);
            myY.push_back(y);
            myZ.push_back(z);
            myMinTNArr.push_back(minTN);
            myMinTNSelfArr.push_back(minTNSelf);
            myCg.push_back(cg);
        }
        int myCount = (int)myX.size();
        std::vector<int> counts(m_uiActiveNpes), offs(m_uiActiveNpes, 0);
        MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      comm);
        int total = 0;
        for (int p = 0; p < m_uiActiveNpes; p++) {
            offs[p] = total;
            total += counts[p];
        }
        std::vector<unsigned long long> allX(total), allY(total),
            allZ(total), allMinTN(total), allMinTNSelf(total);
        std::vector<unsigned int> allCg(total);
        MPI_Allgatherv(myX.data(), myCount, MPI_UINT64_T, allX.data(),
                       counts.data(), offs.data(), MPI_UINT64_T, comm);
        MPI_Allgatherv(myY.data(), myCount, MPI_UINT64_T, allY.data(),
                       counts.data(), offs.data(), MPI_UINT64_T, comm);
        MPI_Allgatherv(myZ.data(), myCount, MPI_UINT64_T, allZ.data(),
                       counts.data(), offs.data(), MPI_UINT64_T, comm);
        MPI_Allgatherv(myMinTNArr.data(), myCount, MPI_UINT64_T,
                       allMinTN.data(), counts.data(), offs.data(),
                       MPI_UINT64_T, comm);
        MPI_Allgatherv(myMinTNSelfArr.data(), myCount, MPI_UINT64_T,
                       allMinTNSelf.data(), counts.data(), offs.data(),
                       MPI_UINT64_T, comm);
        MPI_Allgatherv(myCg.data(), myCount, MPI_UNSIGNED, allCg.data(),
                       counts.data(), offs.data(), MPI_UNSIGNED, comm);

        // Per phys_pos: collect (rank, cg, minTN) claims. The winner
        // is the rank advertising the GLOBALLY smallest minTN. This
        // matches SFC's natural cascade choice (the rank holding the
        // smallest-TreeNode element at this phys_pos as a LOCAL
        // element wins; ranks without a local elem at the phys_pos
        // advertise PACK_INF and can't win).
        struct PhysKey {
            unsigned long long x, y, z;
            bool operator==(const PhysKey& o) const {
                return x == o.x && y == o.y && z == o.z;
            }
        };
        struct PhysKeyHash {
            size_t operator()(const PhysKey& k) const {
                size_t h = std::hash<unsigned long long>()(k.x);
                h ^= std::hash<unsigned long long>()(k.y)
                    + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= std::hash<unsigned long long>()(k.z)
                    + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };
        // phys_pos -> vector of (rank, cg, minTN_any, minTN_self) claims
        struct Claim {
            int rank;
            unsigned int cg;
            unsigned long long minTN;
            unsigned long long minTNSelf;
        };
        std::unordered_map<PhysKey, std::vector<Claim>, PhysKeyHash> claims;
        claims.reserve(total);
        for (int p = 0; p < m_uiActiveNpes; p++) {
            for (int i = offs[p]; i < offs[p] + counts[p]; i++) {
                PhysKey k{allX[i], allY[i], allZ[i]};
                claims[k].push_back(
                    Claim{p, allCg[i], allMinTN[i], allMinTNSelf[i]});
            }
        }

        // env gate: DENDRO_DISABLE_PASS_D_RESCUE=1 disables the
        // phantom-cascade rescue.
        const char* _rescue_env =
            std::getenv("DENDRO_DISABLE_PASS_D_RESCUE");
        const bool _rescue_enabled =
            !(_rescue_env && _rescue_env[0] == '1'
              && _rescue_env[1] == '\0');

        // For my rank: find phys_pos where I'm a loser. For each, my
        // local cg needs to be redirected to a ghost cg at the same
        // phys_pos.
        // ALSO: for phys_pos where NO rank has a self-owned writer
        // (allMinTN_self == PACK_INF for every claimant — phantom
        // cascade), pick a winner from minTN_any and have the winner
        // promote a local (elem, sub) to self-owned. Without this
        // rescue, vec[cg] at the phys_pos never gets written by
        // zip, propagating as O(physics) face residual.
        std::unordered_map<unsigned int, unsigned int> redirect;
        size_t demoted = 0;
        size_t rescued = 0;
        for (auto& kv : claims) {
            const auto& kvec = kv.second;
            bool any_self = false;
            for (auto& c : kvec) {
                if (c.minTNSelf != PACK_INF) { any_self = true; break; }
            }
            const bool rescue_case = !any_self;
            if (kvec.size() <= 1 && !rescue_case) continue;
            // Winner: smallest minTN; tiebreak smallest rank.
            int winner = kvec[0].rank;
            unsigned long long winnerTN =
                rescue_case ? kvec[0].minTN : kvec[0].minTNSelf;
            for (auto& c : kvec) {
                unsigned long long cTN =
                    rescue_case ? c.minTN : c.minTNSelf;
                if (cTN < winnerTN
                    || (cTN == winnerTN && c.rank < winner)) {
                    winner = c.rank;
                    winnerTN = cTN;
                }
            }
            if (rescue_case && !_rescue_enabled) continue;
            if (winnerTN == PACK_INF) continue;

            if (winner == m_uiActiveRank) {
                if (rescue_case) {
                    unsigned int myLocalCgWin = LOOK_UP_TABLE_DEFAULT;
                    for (auto& c : kvec) {
                        if (c.rank == m_uiActiveRank) {
                            myLocalCgWin = c.cg;
                            break;
                        }
                    }
                    if (myLocalCgWin == LOOK_UP_TABLE_DEFAULT) continue;
                    // Find local (e, n) at this phys_pos with the
                    // SMALLEST packTN — matches the elem we used to
                    // compute the advertised minTN_any. Picking by
                    // first-found instead can grab a finer-level
                    // elem that wasn't the advertised candidate.
                    unsigned int newE = LOOK_UP_TABLE_DEFAULT;
                    unsigned int newN = 0;
                    unsigned long long bestTN = PACK_INF;
                    for (unsigned int e = m_uiElementLocalBegin;
                         e < m_uiElementLocalEnd; e++) {
                        unsigned long long etn = packTN(e);
                        if (etn >= bestTN) continue;
                        for (unsigned int n = 0; n < npe; n++) {
                            unsigned long long ex, ey, ez;
                            encodeKey(e, n, ex, ey, ez);
                            if (ex == kv.first.x && ey == kv.first.y
                                && ez == kv.first.z) {
                                newE = e;
                                newN = n;
                                bestTN = etn;
                                break;
                            }
                        }
                    }
                    if (newE == LOOK_UP_TABLE_DEFAULT) continue;
                    const unsigned int newDG = newE * npe + newN;
                    // Promote: self-own the slot, point its E2N_CG at
                    // my local cg, fix the cg2dg/dg2cg pointers.
                    m_uiE2NMapping_DG[newDG] = newDG;
                    m_uiE2NMapping_CG[newDG] = myLocalCgWin;
                    const unsigned int oldDG = m_uiCG2DG[myLocalCgWin];
                    if (oldDG != LOOK_UP_TABLE_DEFAULT
                        && oldDG < m_uiDG2CG.size()
                        && m_uiDG2CG[oldDG] == myLocalCgWin)
                        m_uiDG2CG[oldDG] = LOOK_UP_TABLE_DEFAULT;
                    m_uiCG2DG[myLocalCgWin] = newDG;
                    m_uiDG2CG[newDG]        = myLocalCgWin;
                    rescued++;
                }
                continue;
            }
            // I'm a loser; find my local cg at this phys_pos
            unsigned int myLocalCg = LOOK_UP_TABLE_DEFAULT;
            for (auto& c : kvec) {
                if (c.rank == m_uiActiveRank) {
                    myLocalCg = c.cg;
                    break;
                }
            }
            if (myLocalCg == LOOK_UP_TABLE_DEFAULT) continue;
            // Find a ghost cg at the same phys_pos via walk over
            // m_uiAllElements. Picks the smallest-indexed ghost cg.
            unsigned int ghostCg = LOOK_UP_TABLE_DEFAULT;
            for (size_t e = 0; e < nElTot && ghostCg == LOOK_UP_TABLE_DEFAULT;
                 e++) {
                for (unsigned int n = 0; n < npe; n++) {
                    unsigned long long ex, ey, ez;
                    encodeKey((unsigned int)e, n, ex, ey, ez);
                    if (ex != kv.first.x) continue;
                    if (ey != kv.first.y) continue;
                    if (ez != kv.first.z) continue;
                    unsigned int cg2 =
                        m_uiE2NMapping_CG[e * npe + n];
                    // ghost = NOT in [nLB, nLE)
                    if (cg2 < nLB || cg2 >= nLE) {
                        if (ghostCg == LOOK_UP_TABLE_DEFAULT
                            || cg2 < ghostCg)
                            ghostCg = cg2;
                    }
                }
            }
            if (ghostCg == LOOK_UP_TABLE_DEFAULT) continue;
            redirect[myLocalCg] = ghostCg;
            demoted++;
        }
        if (rescued) {
            std::cout << "[passD-rescue r" << m_uiActiveRank
                      << "] phantom-cascade promoted=" << rescued
                      << std::endl;
        }

        // Stash demoted local cgs (so Pass E doesn't undo demote)
        // and the demoted→ghost cg map (so syncDemotedLocalCgs
        // can mirror ghost values into demoted local slots after
        // each ghost exchange).
        m_uiPassDDemotedLocalCgs.clear();
        m_uiPassDDemotedLocalCgs.reserve(redirect.size());
        m_uiPassDDemotedToGhostCg.clear();
        m_uiPassDDemotedToGhostCg.reserve(redirect.size());
        for (auto& kv : redirect) {
            m_uiPassDDemotedLocalCgs.insert(kv.first);
            m_uiPassDDemotedToGhostCg[kv.first] = kv.second;
        }

        if (!redirect.empty()) {
            // Apply redirect to E2N_CG (and scatter maps for safety,
            // though those get rebuilt by repartitionMeshGlobal).
            auto applyRedirect = [&](std::vector<unsigned int>& vec) {
                size_t n = 0;
                for (auto& v : vec) {
                    auto it = redirect.find(v);
                    if (it != redirect.end()) {
                        v = it->second;
                        ++n;
                    }
                }
                return n;
            };
            size_t rE2N = applyRedirect(m_uiE2NMapping_CG);
            size_t rSend =
                applyRedirect(m_uiScatterMapActualNodeSend);
            size_t rRecv =
                applyRedirect(m_uiScatterMapActualNodeRecv);
            std::cout << "[passD r" << m_uiActiveRank
                      << "] demoted=" << demoted
                      << " e2n=" << rE2N
                      << " sendSM=" << rSend
                      << " recvSM=" << rRecv << std::endl;
        }
    }

    // Pass E: on-rank orphan-redirect.
    //
    // After Pass A and Pass D, some local cgs may STILL be orphans:
    // no local elem refs them via E2N_CG (cg2dg points to a ghost
    // elem). On these (winner) ranks, no zip writer means the cg
    // stays at 0 each rhs call → graph drifts from SFC.
    //
    // Fix: find a local (elem, sub) at the same phys_pos as cg2dg[cg]'s
    // element. Redirect E2N_CG of that local elem from its current
    // (ghost) cg to this local cg, and promote its E2N_DG to self-
    // owned. Now the local elem is the canonical writer of the local
    // cg slot, and zip writes correctly.
    //
    // Safe to do AFTER Pass D: cross-rank duplicates are resolved,
    // so promoting on a loser rank doesn't happen (Pass D moved
    // their local elements off the local cg already).
    if (_passDE_enabled && m_uiIsActive) {
        const unsigned int npe = m_uiNpE;
        const unsigned int eOrd = m_uiElementOrder;
        const unsigned int nLB = m_uiNodeLocalBegin;
        const unsigned int nLE = m_uiNodeLocalEnd;
        const auto* pN = m_uiAllElements.data();

        // Recompute refByLocal AFTER Pass D's redirects (Pass D
        // changed E2N_CG; some local cgs that were not orphan
        // before may still not be orphan; some MAY have become
        // orphan-but-thats-fine since Pass D removed their refs
        // intentionally).
        std::vector<unsigned char> refByLocal(nLE - nLB, 0);
        for (unsigned int e = m_uiElementLocalBegin;
             e < m_uiElementLocalEnd; e++) {
            for (unsigned int n = 0; n < npe; n++) {
                unsigned int cg = m_uiE2NMapping_CG[e * npe + n];
                if (cg >= nLB && cg < nLE) refByLocal[cg - nLB] = 1;
            }
        }

        size_t fixed = 0;
        size_t promotedB = 0;  // unused; kept for log compatibility
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            if (refByLocal[cg - nLB]) continue;
            // Skip cgs that Pass D intentionally demoted.
            if (m_uiPassDDemotedLocalCgs.count(cg)) continue;
            unsigned int dg = m_uiCG2DG[cg];
            if (dg == LOOK_UP_TABLE_DEFAULT) continue;
            unsigned int oe = dg / npe;
            unsigned int os = dg % npe;
            if (oe >= m_uiAllElements.size()) continue;
            if (oe >= m_uiElementLocalBegin
                && oe < m_uiElementLocalEnd
                && m_uiE2NMapping_CG[oe * npe + os] == cg)
                continue;
            // phys_pos via cg2dg's element
            const unsigned long long lenT =
                (unsigned long long)1
                << (m_uiMaxDepth - pN[oe].getLevel());
            const unsigned int ni_t = os % (eOrd + 1);
            const unsigned int nj_t = (os / (eOrd + 1)) % (eOrd + 1);
            const unsigned int nk_t = os / ((eOrd + 1) * (eOrd + 1));
            const unsigned long long tx =
                (unsigned long long)pN[oe].getX() * eOrd
                + (unsigned long long)ni_t * lenT;
            const unsigned long long ty =
                (unsigned long long)pN[oe].getY() * eOrd
                + (unsigned long long)nj_t * lenT;
            const unsigned long long tz =
                (unsigned long long)pN[oe].getZ() * eOrd
                + (unsigned long long)nk_t * lenT;

            // Find a local (e, n) at the same phys_pos.
            unsigned int newE = LOOK_UP_TABLE_DEFAULT;
            unsigned int newN = LOOK_UP_TABLE_DEFAULT;
            for (unsigned int e = m_uiElementLocalBegin;
                 e < m_uiElementLocalEnd
                 && newE == LOOK_UP_TABLE_DEFAULT; e++) {
                const unsigned long long len_e =
                    (unsigned long long)1
                    << (m_uiMaxDepth - pN[e].getLevel());
                for (unsigned int n = 0; n < npe; n++) {
                    const unsigned int ni = n % (eOrd + 1);
                    const unsigned int nj =
                        (n / (eOrd + 1)) % (eOrd + 1);
                    const unsigned int nk =
                        n / ((eOrd + 1) * (eOrd + 1));
                    const unsigned long long ex =
                        (unsigned long long)pN[e].getX() * eOrd
                        + (unsigned long long)ni * len_e;
                    if (ex != tx) continue;
                    const unsigned long long ey =
                        (unsigned long long)pN[e].getY() * eOrd
                        + (unsigned long long)nj * len_e;
                    if (ey != ty) continue;
                    const unsigned long long ez =
                        (unsigned long long)pN[e].getZ() * eOrd
                        + (unsigned long long)nk * len_e;
                    if (ez != tz) continue;
                    newE = e;
                    newN = n;
                    break;
                }
            }
            if (newE == LOOK_UP_TABLE_DEFAULT) continue;
            // Redirect this local elem's E2N_CG from old (ghost) cg
            // to this local cg, and promote E2N_DG to self-owned.
            const unsigned int newDG = newE * npe + newN;
            m_uiE2NMapping_CG[newE * npe + newN] = cg;
            m_uiE2NMapping_DG[newDG] = newDG;
            // Sync cg2dg/dg2cg.
            const unsigned int oldDG = m_uiCG2DG[cg];
            if (oldDG < m_uiDG2CG.size() && m_uiDG2CG[oldDG] == cg)
                m_uiDG2CG[oldDG] = LOOK_UP_TABLE_DEFAULT;
            m_uiCG2DG[cg]    = newDG;
            m_uiDG2CG[newDG] = cg;
            fixed++;
        }
        if (fixed || promotedB) {
            std::cout << "[passE r" << m_uiActiveRank
                      << "] orphan-fixed=" << fixed
                      << " caseB-promoted=" << promotedB
                      << std::endl;
        }
    }

    // Pass C (DISABLED — see docs/partitioning_session_log.md for details):
    //
    // Goal: within-rank phys_pos consolidation. Graph repartition can
    // leave a rank with multiple CGs at the same physical position
    // (e.g., a local CG plus a post-ghost CG, both mapped from
    // elements at the same (x, y, z)). After ghost exchange these
    // slots can hold different values, so unzip into different
    // blocks reads inconsistent values for the same phys_pos and
    // the laplacian becomes wrong.
    //
    // Approach attempted: build phys_pos -> {cgs} map; for phys_pos
    // with multiple cgs, pick canonical (prefer local cg, else
    // smallest) and redirect E2N_CG entries pointing to non-canonical.
    // Also rewrite E2N_DG so canonical writer is unique.
    //
    // Result on NLSM mesh: step-0 dCHI EXPLODED from ~5e-17 to 0.74-1.28.
    // Even though TEST 8bo + 8br pass (test mesh has consistent cascade),
    // NLSM's mesh has a deeper inconsistency: two (elem, sub) pairs that
    // share a CG via E2N_CG (so should be at the same phys_pos) actually
    // have E2N_DG-decoded phys_pos that DIFFER. Pass C merges them into a
    // single cg slot, but createVector pass 2 then writes
    //   vec[cg] = func(phys_pos_via_E2N_DG)
    // for both pairs, with two different phys_pos values, last-write-wins
    // — depending on iteration order, vec[cg] holds an IC value that
    // doesn't match the actual phys_pos. Step-0 IC ends up scrambled.
    //
    // Root cause is upstream: the master cascade in
    // buildE2NWithSMRepartitioned chains DG ownership in a way that
    // creates pairs sharing E2N_CG but not E2N_DG-decodable to the
    // same phys_pos. Diagnosing/fixing the cascade itself is the
    // next-session task. Pass C as a post-cascade cleanup is the
    // wrong layer.
    if (false) if (m_uiIsActive) {
        const unsigned int npe   = m_uiNpE;
        const unsigned int eOrd  = m_uiElementOrder;
        const unsigned int nLB   = m_uiNodeLocalBegin;
        const unsigned int nLE   = m_uiNodeLocalEnd;
        const auto* pN           = m_uiAllElements.data();
        const size_t nElTot      = m_uiAllElements.size();

        // physPosKey -> first-seen cg, then collect duplicates.
        // Key is (x, y, z) as scaled integer coords (full uint64 each
        // to avoid overflow when m_uiMaxDepth + log2(eOrder) > 21).
        struct PhysKey {
            unsigned long long x, y, z;
            bool operator==(const PhysKey& o) const {
                return x == o.x && y == o.y && z == o.z;
            }
        };
        struct PhysKeyHash {
            size_t operator()(const PhysKey& k) const {
                size_t h = std::hash<unsigned long long>()(k.x);
                h ^= std::hash<unsigned long long>()(k.y)
                    + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= std::hash<unsigned long long>()(k.z)
                    + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };
        std::unordered_map<PhysKey, unsigned int, PhysKeyHash> firstCg;
        std::unordered_map<PhysKey, std::vector<unsigned int>,
                           PhysKeyHash> dupCgs;

        auto encodeKey = [&](unsigned int e, unsigned int n) -> PhysKey {
            unsigned long long len =
                (unsigned long long)1
                << (m_uiMaxDepth - pN[e].getLevel());
            unsigned int ni = n % (eOrd + 1);
            unsigned int nj = (n / (eOrd + 1)) % (eOrd + 1);
            unsigned int nk = n / ((eOrd + 1) * (eOrd + 1));
            PhysKey k;
            k.x = (unsigned long long)pN[e].getX() * eOrd
                  + (unsigned long long)ni * len;
            k.y = (unsigned long long)pN[e].getY() * eOrd
                  + (unsigned long long)nj * len;
            k.z = (unsigned long long)pN[e].getZ() * eOrd
                  + (unsigned long long)nk * len;
            return k;
        };

        // Track (e, n) pairs that will need E2N_DG fixup after
        // redirect. Without this, multiple pairs can end up "self-
        // owned" pointing to the same canonical cg, and zip would
        // write the cg multiple times from inconsistent block
        // buffers.
        std::vector<std::pair<unsigned int, unsigned int>> toRewireDG;

        // Skip elements without a valid nodal map (R2/isGhostTwo
        // elements may have stale or partial E2N data, which would
        // create false-positive duplicates).
        for (size_t e = 0; e < nElTot; e++) {
            if (e < m_uiIsNodalMapValid.size() && !m_uiIsNodalMapValid[e])
                continue;
            for (unsigned int n = 0; n < npe; n++) {
                unsigned int cg = m_uiE2NMapping_CG[e * npe + n];
                PhysKey key = encodeKey((unsigned int)e, n);
                auto it = firstCg.find(key);
                if (it == firstCg.end()) {
                    firstCg[key] = cg;
                } else if (it->second != cg) {
                    auto& vec = dupCgs[key];
                    if (vec.empty()) vec.push_back(it->second);
                    if (std::find(vec.begin(), vec.end(), cg)
                        == vec.end())
                        vec.push_back(cg);
                }
            }
        }

        // Build redirect map: non-canonical cg -> canonical cg.
        std::unordered_map<unsigned int, unsigned int> redirect;
        for (auto& kv : dupCgs) {
            auto& cgs = kv.second;
            if (cgs.size() <= 1) continue;
            // Pick canonical: prefer local cg (in [nLB, nLE)).
            // Tiebreak among multiple locals: smallest cg.
            unsigned int canonical = LOOK_UP_TABLE_DEFAULT;
            for (auto cg : cgs) {
                if (cg >= nLB && cg < nLE) {
                    if (canonical == LOOK_UP_TABLE_DEFAULT
                        || cg < canonical)
                        canonical = cg;
                }
            }
            if (canonical == LOOK_UP_TABLE_DEFAULT) {
                // No local; pick smallest cg as canonical.
                canonical = *std::min_element(cgs.begin(), cgs.end());
            }
            for (auto cg : cgs) {
                if (cg != canonical) redirect[cg] = canonical;
            }
        }

        if (!redirect.empty()) {
            // Resolve redirect chains.
            for (auto& kv : redirect) {
                unsigned int y = kv.second;
                while (true) {
                    auto it = redirect.find(y);
                    if (it == redirect.end()) break;
                    y = it->second;
                }
                kv.second = y;
            }

            // Identify (e, n) whose E2N_CG was a redirect source.
            // We need their E2N_DG rewritten post-redirect so that
            // the canonical writer of the redirected-to cg is the
            // ONLY one with E2N_DG self-owned. Otherwise zip writes
            // the cg multiple times from different block buffers.
            // Skip invalid-nodal-map elements.
            for (size_t e = 0; e < nElTot; e++) {
                if (e < m_uiIsNodalMapValid.size()
                    && !m_uiIsNodalMapValid[e])
                    continue;
                for (unsigned int n = 0; n < npe; n++) {
                    unsigned int cg = m_uiE2NMapping_CG[e * npe + n];
                    if (redirect.find(cg) != redirect.end()) {
                        toRewireDG.emplace_back((unsigned int)e, n);
                    }
                }
            }

            auto applyRedirect = [&](std::vector<unsigned int>& vec) {
                size_t n = 0;
                for (auto& v : vec) {
                    auto it = redirect.find(v);
                    if (it != redirect.end()) {
                        v = it->second;
                        ++n;
                    }
                }
                return n;
            };
            size_t rE2N    = applyRedirect(m_uiE2NMapping_CG);
            size_t rSend   = applyRedirect(m_uiScatterMapActualNodeSend);
            size_t rRecv   = applyRedirect(m_uiScatterMapActualNodeRecv);

            // Rewire E2N_DG for redirected pairs to point to the
            // canonical writer (cg2dg[newCg]). This collapses
            // duplicate canonical-writer claims onto a single
            // (elem, sub).
            //
            // Currently disabled (false branch) to isolate behavior.
            size_t rDG = 0;
            if (false) {
                for (auto& [e, n] : toRewireDG) {
                    unsigned int newCg = m_uiE2NMapping_CG[e * npe + n];
                    if (newCg < m_uiCG2DG.size()) {
                        unsigned int newDG = m_uiCG2DG[newCg];
                        if (newDG != LOOK_UP_TABLE_DEFAULT
                            && m_uiE2NMapping_DG[e * npe + n] != newDG) {
                            m_uiE2NMapping_DG[e * npe + n] = newDG;
                            rDG++;
                        }
                    }
                }
            }

            std::cout << "[passC r" << m_uiActiveRank
                      << "] dup-phys-pos=" << redirect.size()
                      << " e2n=" << rE2N
                      << " e2n_dg=" << rDG
                      << " sendSM=" << rSend
                      << " recvSM=" << rRecv << std::endl;
        }
    }

}

void Mesh::buildE2NMap() {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

#ifdef DEBUG_E2N_MAPPING
    std::vector<ot::TreeNode> invalidatedPreGhost;
    std::vector<ot::TreeNode> invalidatedPostGhost;
    // if(!m_uiActiveRank)  std::cout<<"E2E  rank :
    // "<<m_uiActiveRank<<std::endl; if(!m_uiActiveRank)
    for (unsigned int e = 0; e < m_uiAllElements.size(); e++) {
        // if(m_uiAllElements[e].getLevel()!=1) {
        // std::cout << "Element : "<<e<<" " << m_uiAllElements[e] << " : Node
        // List :";
        for (unsigned int k = 0; k < m_uiNumDirections; k++) {
            //  std::cout << " " << m_uiE2EMapping[e * m_uiNumDirections + k];
            if (m_uiE2EMapping[e * m_uiNumDirections + k] !=
                LOOK_UP_TABLE_DEFAULT)
                assert(m_uiE2EMapping[e * m_uiNumDirections + k] <
                       m_uiAllElements.size());
        }

        // std::cout << std::endl;
        // }
    }
#endif

    // update the E2E mapping with fake elements.
    unsigned int lookUp = 0;
    unsigned int lev1   = 0;
    unsigned int lev2   = 0;

    unsigned int child;
    unsigned int parent;

#ifdef DEBUG_E2N_MAPPING
    for (unsigned int ge = m_uiElementPreGhostBegin;
         ge < m_uiElementPreGhostEnd; ge++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++)
            if (m_uiE2EMapping[ge * m_uiNumDirections + dir] !=
                LOOK_UP_TABLE_DEFAULT)
                assert((m_uiE2EMapping[ge * m_uiNumDirections + dir] >=
                        m_uiElementLocalBegin) &&
                       (m_uiE2EMapping[ge * m_uiNumDirections + dir] <
                        m_uiElementLocalEnd));
    }

    for (unsigned int ge = m_uiElementPostGhostBegin;
         ge < m_uiElementPostGhostEnd; ge++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++)
            if (m_uiE2EMapping[ge * m_uiNumDirections + dir] !=
                LOOK_UP_TABLE_DEFAULT)
                assert((m_uiE2EMapping[ge * m_uiNumDirections + dir] >=
                        m_uiElementLocalBegin) &&
                       (m_uiE2EMapping[ge * m_uiNumDirections + dir] <
                        m_uiElementLocalEnd));
    }

#endif

    assert(m_uiNumTotalElements == m_uiAllElements.size());
    assert((m_uiElementPostGhostEnd - m_uiElementPreGhostBegin) > 0);
    assert(m_uiNumTotalElements ==
           ((m_uiElementPostGhostEnd - m_uiElementPreGhostBegin)));

    m_uiE2NMapping_CG.resize(m_uiNumTotalElements * m_uiNpE);
    m_uiE2NMapping_DG.resize(m_uiNumTotalElements * m_uiNpE);

    // initialize the DG mapping. // this order is mandotory.
    for (unsigned int e = 0; e < (m_uiNumTotalElements); e++)
        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
             k++)  // z coordinate
            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                 j++)  // y coordinate
                for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                     i++)  // x coordinate
                    m_uiE2NMapping_CG[e * m_uiNpE +
                                      k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                        e * m_uiNpE +
                        k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i;

#ifdef DEBUG_E2N_MAPPING
    MPI_Barrier(MPI_COMM_WORLD);
    if (!m_uiActiveRank) std::cout << "Invalid nodes removed " << std::endl;
    // treeNodesTovtk(invalidatedPreGhost,m_uiActiveRank,"invalidPre");
    // treeNodesTovtk(invalidatedPostGhost,m_uiActiveRank,"invalidPost");
#endif

    // 2. Removing the duplicate nodes from the mapping.
    unsigned int ownerIndexChild;
    unsigned int ownerIndexParent;

    unsigned int child_i, child_j, child_k;
    unsigned int parent_i, parent_j, parent_k;

#ifdef DEBUG_E2N_MAPPING
    std::vector<ot::TreeNode> cusE2ECheck;

    if (!m_uiActiveRank && m_uiActiveNpes > 1) {
        unsigned int eleID = 2;
        // cusE2ECheck.push_back(m_uiAllElements[eleID]);
        for (unsigned int dir = 0; dir < (m_uiNumDirections); dir++) {
            if (m_uiE2EMapping[eleID * m_uiNumDirections + dir] !=
                LOOK_UP_TABLE_DEFAULT)
                cusE2ECheck.push_back(
                    m_uiAllElements[m_uiE2EMapping[eleID * m_uiNumDirections +
                                                   dir]]);
        }

        treeNodesTovtk(cusE2ECheck, m_uiActiveRank, "cusE2ECheck");
    }

    unsigned int eleVtk   = 2;
    unsigned int eleVtkAt = 1;

    unsigned int lookUp1, lookUp2, lookUp3, lookUp4;
    unsigned int edgeOwner;
    unsigned int cornerNodeOwner;

    unsigned int cornerNodeOwnerIndex;
    unsigned int cornerNodeChildIndex;
#endif

    bool parentChildLevEqual = false;
    std::vector<unsigned int>
        faceChildIndex;  // indices that are being updated for a face
    std::vector<unsigned int>
        faceOwnerIndex;  // indices that are being used for updates. for a face

    std::vector<unsigned int>
        edgeChildIndex;  // indices that are being updated for an edge
    std::vector<unsigned int>
        edgeOwnerIndex;  // indices that are being used for updates. for an edge

    // Iterate in SFC (TreeNode) order so that CG-mapping chains
    // resolve deterministically. For an SFC-sorted m_uiAllElements
    // (the default after createMesh) this is identity; for a
    // repartitioned mesh the physical layout is (trank, eid) and the
    // permutation is needed to keep the chain resolution correct.
    std::vector<unsigned int> __sfcOrder(m_uiAllElements.size());
    std::iota(__sfcOrder.begin(), __sfcOrder.end(), 0u);
    std::sort(__sfcOrder.begin(), __sfcOrder.end(),
              [this](unsigned int a, unsigned int b) {
                  return m_uiAllElements[a] < m_uiAllElements[b];
              });

    for (unsigned int __i = 0; __i < __sfcOrder.size(); __i++) {
        unsigned int e = __sfcOrder[__i];
        // 1. All local nodes for a given element is automatically mapped for a
        // given element by construction of the DG indexing.

        // 2. Map internal nodes on each face,  with the corresponding face.
        parentChildLevEqual = false;
        lookUp = m_uiE2EMapping[e * m_uiNumDirections + OCT_DIR_LEFT];
        if (lookUp != LOOK_UP_TABLE_DEFAULT) {
            lev1 = m_uiAllElements[e].getLevel();
            lev2 = m_uiAllElements[lookUp].getLevel();
            if (lev1 == lev2) {
                parentChildLevEqual = true;
                // Rank-independent SFC tie-break: every rank sees the
                // same TreeNode ordering, so all ranks agree on owners.
                lev1 = (m_uiAllElements[e] < m_uiAllElements[lookUp]) ? 0 : 1;
                lev2 = 1 - lev1;
                assert(e != lookUp);
            }

            child  = ((lev1 > lev2) ? e : lookUp);
            parent = ((lev1 < lev2) ? e : lookUp);

            if (child == e) {
                // if(m_uiActiveRank==0 && e==324)std::cout<<"m_uiActiveRank:
                // "<<m_uiActiveRank<<" LEFT ele: "<<e<<std::endl;
                assert(parent == lookUp);

                faceNodesIndex(child, OCT_DIR_LEFT, faceChildIndex, true);
                faceNodesIndex(parent, OCT_DIR_RIGHT, faceOwnerIndex, true);

                assert(faceChildIndex.size() == faceOwnerIndex.size());

                for (unsigned int index = 0; index < faceChildIndex.size();
                     index++)
                    m_uiE2NMapping_CG[faceChildIndex[index]] =
                        m_uiE2NMapping_CG[faceOwnerIndex[index]];

                OCT_DIR_LEFT_INTERNAL_EDGE_MAP(
                    child, parent, parentChildLevEqual, edgeChildIndex,
                    edgeOwnerIndex);  // maps all the edges in the left face
            }
        }

        parentChildLevEqual = false;
        lookUp = m_uiE2EMapping[e * m_uiNumDirections + OCT_DIR_RIGHT];
        if (lookUp != LOOK_UP_TABLE_DEFAULT) {
            lev1 = m_uiAllElements[e].getLevel();
            lev2 = m_uiAllElements[lookUp].getLevel();
            if (lev1 == lev2) {
                // Rank-independent SFC tie-break: every rank sees the
                // same TreeNode ordering, so all ranks agree on owners.
                lev1 = (m_uiAllElements[e] < m_uiAllElements[lookUp]) ? 0 : 1;
                lev2 = 1 - lev1;
                parentChildLevEqual = true;
                assert(e != lookUp);
            }

            child  = ((lev1 > lev2) ? e : lookUp);
            parent = ((lev1 < lev2) ? e : lookUp);

            if (child == e) {
                // if(m_uiActiveRank==0 && e==324)std::cout<<"m_uiActiveRank:
                // "<<m_uiActiveRank<<" RIGHT ele: "<<e<<std::endl;
                assert(parent == lookUp);
                faceNodesIndex(child, OCT_DIR_RIGHT, faceChildIndex, true);
                faceNodesIndex(parent, OCT_DIR_LEFT, faceOwnerIndex, true);

                assert(faceChildIndex.size() == faceOwnerIndex.size());

                for (unsigned int index = 0; index < faceChildIndex.size();
                     index++)
                    m_uiE2NMapping_CG[faceChildIndex[index]] =
                        m_uiE2NMapping_CG[faceOwnerIndex[index]];

                OCT_DIR_RIGHT_INTERNAL_EDGE_MAP(
                    child, parent, parentChildLevEqual, edgeChildIndex,
                    edgeOwnerIndex);  // maps all the edges in the right face
            }
        }

        parentChildLevEqual = false;
        lookUp = m_uiE2EMapping[e * m_uiNumDirections + OCT_DIR_DOWN];
        if (lookUp != LOOK_UP_TABLE_DEFAULT) {
            lev1 = m_uiAllElements[e].getLevel();
            lev2 = m_uiAllElements[lookUp].getLevel();
            if (lev1 == lev2) {
                // Rank-independent SFC tie-break: every rank sees the
                // same TreeNode ordering, so all ranks agree on owners.
                lev1 = (m_uiAllElements[e] < m_uiAllElements[lookUp]) ? 0 : 1;
                lev2 = 1 - lev1;
                parentChildLevEqual = true;
                assert(e != lookUp);
            }

            child  = ((lev1 > lev2) ? e : lookUp);
            parent = ((lev1 < lev2) ? e : lookUp);

            if (child == e) {
                // if(m_uiActiveRank==0 && e==324)std::cout<<"m_uiActiveRank:
                // "<<m_uiActiveRank<<" DOWN ele: "<<e<<std::endl;
                assert(parent == lookUp);
                faceNodesIndex(child, OCT_DIR_DOWN, faceChildIndex, true);
                faceNodesIndex(parent, OCT_DIR_UP, faceOwnerIndex, true);

                assert(faceChildIndex.size() == faceOwnerIndex.size());

                for (unsigned int index = 0; index < faceChildIndex.size();
                     index++)
                    m_uiE2NMapping_CG[faceChildIndex[index]] =
                        m_uiE2NMapping_CG[faceOwnerIndex[index]];

                OCT_DIR_DOWN_INTERNAL_EDGE_MAP(
                    child, parent, parentChildLevEqual, edgeChildIndex,
                    edgeOwnerIndex);  // maps all the edges in the DOWN face
            }
        }

        parentChildLevEqual = false;
        lookUp = m_uiE2EMapping[e * m_uiNumDirections + OCT_DIR_UP];
        if (lookUp != LOOK_UP_TABLE_DEFAULT) {
            lev1 = m_uiAllElements[e].getLevel();
            lev2 = m_uiAllElements[lookUp].getLevel();
            if (lev1 == lev2) {
                // Rank-independent SFC tie-break: every rank sees the
                // same TreeNode ordering, so all ranks agree on owners.
                lev1 = (m_uiAllElements[e] < m_uiAllElements[lookUp]) ? 0 : 1;
                lev2 = 1 - lev1;
                parentChildLevEqual = true;
                assert(e != lookUp);
            }

            child  = ((lev1 > lev2) ? e : lookUp);
            parent = ((lev1 < lev2) ? e : lookUp);

            if (child == e) {
                // if(m_uiActiveRank==0 && e==324)std::cout<<"m_uiActiveRank:
                // "<<m_uiActiveRank<<" UP ele: "<<e<<std::endl;
                assert(parent == lookUp);
                faceNodesIndex(child, OCT_DIR_UP, faceChildIndex, true);
                faceNodesIndex(parent, OCT_DIR_DOWN, faceOwnerIndex, true);

                assert(faceChildIndex.size() == faceOwnerIndex.size());

                for (unsigned int index = 0; index < faceChildIndex.size();
                     index++)
                    m_uiE2NMapping_CG[faceChildIndex[index]] =
                        m_uiE2NMapping_CG[faceOwnerIndex[index]];

                OCT_DIR_UP_INTERNAL_EDGE_MAP(
                    child, parent, parentChildLevEqual, edgeChildIndex,
                    edgeOwnerIndex);  // maps all the edges in the UP face
            }
        }

        parentChildLevEqual = false;
        lookUp = m_uiE2EMapping[e * m_uiNumDirections + OCT_DIR_BACK];
        if (lookUp != LOOK_UP_TABLE_DEFAULT) {
            lev1 = m_uiAllElements[e].getLevel();
            lev2 = m_uiAllElements[lookUp].getLevel();
            if (lev1 == lev2) {
                // Rank-independent SFC tie-break: every rank sees the
                // same TreeNode ordering, so all ranks agree on owners.
                lev1 = (m_uiAllElements[e] < m_uiAllElements[lookUp]) ? 0 : 1;
                lev2 = 1 - lev1;
                parentChildLevEqual = true;
                assert(e != lookUp);
            }

            child  = ((lev1 > lev2) ? e : lookUp);
            parent = ((lev1 < lev2) ? e : lookUp);

            if (child == e) {
                // if(m_uiActiveRank==0 && e==324)std::cout<<"m_uiActiveRank:
                // "<<m_uiActiveRank<<" BACK ele: "<<e<<std::endl;
                assert(parent == lookUp);
                faceNodesIndex(child, OCT_DIR_BACK, faceChildIndex, true);
                faceNodesIndex(parent, OCT_DIR_FRONT, faceOwnerIndex, true);

                assert(faceChildIndex.size() == faceOwnerIndex.size());

                for (unsigned int index = 0; index < faceChildIndex.size();
                     index++)
                    m_uiE2NMapping_CG[faceChildIndex[index]] =
                        m_uiE2NMapping_CG[faceOwnerIndex[index]];

                OCT_DIR_BACK_INTERNAL_EDGE_MAP(
                    child, parent, parentChildLevEqual, edgeChildIndex,
                    edgeOwnerIndex);  // maps all the edges in the back face
            }
        }

        parentChildLevEqual = false;
        lookUp = m_uiE2EMapping[e * m_uiNumDirections + OCT_DIR_FRONT];
        if (lookUp != LOOK_UP_TABLE_DEFAULT) {
            lev1 = m_uiAllElements[e].getLevel();
            lev2 = m_uiAllElements[lookUp].getLevel();
            if (lev1 == lev2) {
                // Rank-independent SFC tie-break: every rank sees the
                // same TreeNode ordering, so all ranks agree on owners.
                lev1 = (m_uiAllElements[e] < m_uiAllElements[lookUp]) ? 0 : 1;
                lev2 = 1 - lev1;
                parentChildLevEqual = true;
                assert(e != lookUp);
            }

            child  = ((lev1 > lev2) ? e : lookUp);
            parent = ((lev1 < lev2) ? e : lookUp);

            if (child == e) {
                // if(m_uiActiveRank==0 && e==324)std::cout<<"m_uiActiveRank:
                // "<<m_uiActiveRank<<"FRONT ele: "<<e<<std::endl;

                assert(parent == lookUp);
                faceNodesIndex(child, OCT_DIR_FRONT, faceChildIndex, true);
                faceNodesIndex(parent, OCT_DIR_BACK, faceOwnerIndex, true);

                assert(faceChildIndex.size() == faceOwnerIndex.size());

                for (unsigned int index = 0; index < faceChildIndex.size();
                     index++)
                    m_uiE2NMapping_CG[faceChildIndex[index]] =
                        m_uiE2NMapping_CG[faceOwnerIndex[index]];

                OCT_DIR_FRONT_INTERNAL_EDGE_MAP(
                    child, parent, parentChildLevEqual, edgeChildIndex,
                    edgeOwnerIndex);  // maps all the edges in the front face
            }
        }

        CORNER_NODE_MAP(e);

        /*if(m_uiActiveRank==0 && e==284)
        {

            std::vector<ot::TreeNode> cusEleCheck;
            unsigned int ownerID,ii_x,jj_y,kk_z; // DG index to ownerID and ijk
        decomposition variable. unsigned int x,y,z,sz;
            cusEleCheck.push_back(m_uiAllElements[e]);
            for(unsigned int node=0;node<m_uiNpE;node++)
            {

                dg2eijk(m_uiE2NMapping_CG[e*m_uiNpE+node],ownerID,ii_x,jj_y,kk_z);

                x=m_uiAllElements[ownerID].getX();
                y=m_uiAllElements[ownerID].getY();
                z=m_uiAllElements[ownerID].getZ();
                sz=1u<<(m_uiMaxDepth-m_uiAllElements[ownerID].getLevel());
                cusEleCheck.push_back(m_uiAllElements[ownerID]);

                cusEleCheck.push_back(ot::TreeNode((x + ii_x *
        sz/m_uiElementOrder), (y + jj_y * sz/m_uiElementOrder), (z + kk_z *
        sz/m_uiElementOrder), m_uiMaxDepth,m_uiDim, m_uiMaxDepth));

            }

            treeNodesTovtk(cusEleCheck,e,"cusE2N_1");

        }*/
    }

    /* for(unsigned int
     e=m_uiElementPreGhostBegin;e<m_uiElementPostGhostEnd;e++)
     {
         if(m_uiActiveRank==0 && e==284)
         {

                 std::vector<ot::TreeNode> cusEleCheck;
                 unsigned int ownerID,ii_x,jj_y,kk_z; // DG index to ownerID and
     ijk decomposition variable. unsigned int x,y,z,sz;
                 cusEleCheck.push_back(m_uiAllElements[e]);
                 for(unsigned int node=0;node<m_uiNpE;node++)
                 {

                     dg2eijk(m_uiE2NMapping_CG[e*m_uiNpE+node],ownerID,ii_x,jj_y,kk_z);

                     x=m_uiAllElements[ownerID].getX();
                     y=m_uiAllElements[ownerID].getY();
                     z=m_uiAllElements[ownerID].getZ();
                     sz=1u<<(m_uiMaxDepth-m_uiAllElements[ownerID].getLevel());
                     cusEleCheck.push_back(m_uiAllElements[ownerID]);

                     cusEleCheck.push_back(ot::TreeNode((x + ii_x *
     sz/m_uiElementOrder), (y + jj_y * sz/m_uiElementOrder), (z + kk_z *
     sz/m_uiElementOrder), m_uiMaxDepth,m_uiDim, m_uiMaxDepth));

                 }

                 treeNodesTovtk(cusEleCheck,e,"cusE2N_2");

         }

     }*/

#ifdef DEBUG_E2N_MAPPING
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned int eleIndex;
    if (m_uiActiveRank == 0)
        std::cout << "E2N  rank : " << m_uiActiveRank << std::endl;
    if (m_uiActiveRank == 0)
        for (unsigned int e = 0; e < m_uiAllElements.size(); e++) {
            // if(m_uiAllElements[e].getLevel()!=1) {
            std::cout << "Element : " << e << " " << m_uiAllElements[e]
                      << " : Node List :";
            for (unsigned int k = 0; k < m_uiNpE; k++) {
                std::cout << " " << m_uiE2NMapping_CG[e * m_uiNpE + k];
            }

            std::cout << std::endl;
            //}
        }

#endif

    // assert(seq::test::checkE2NMapping(m_uiE2EMapping,
    // m_uiE2NMapping_CG,m_uiAllElements,m_uiNumDirections,m_uiElementOrder));
    std::vector<unsigned int> E2N_DG_Sorted;
    std::vector<unsigned int> dg2dg_p;  // dg to dg prime
    E2N_DG_Sorted.resize(m_uiE2NMapping_CG.size());
    E2N_DG_Sorted.assign(m_uiE2NMapping_CG.begin(), m_uiE2NMapping_CG.end());

    m_uiDG2CG.resize(m_uiAllElements.size() * m_uiNpE, LOOK_UP_TABLE_DEFAULT);
    dg2dg_p.resize(m_uiAllElements.size() * m_uiNpE, LOOK_UP_TABLE_DEFAULT);

    // 3. Update DG indexing with CG indexing.
    std::sort(E2N_DG_Sorted.begin(), E2N_DG_Sorted.end());
    E2N_DG_Sorted.erase(std::unique(E2N_DG_Sorted.begin(), E2N_DG_Sorted.end()),
                        E2N_DG_Sorted.end());

    unsigned int owner1, ii_x1, jj_y1, kk_z1;
    unsigned int owner2, ii_x2, jj_y2, kk_z2;
    unsigned int old_val;
    unsigned int new_val;
    unsigned int nsz;

    SearchKey tmpSKey;
    Key tmpKey;
    SearchKey rootSKey(m_uiDim, m_uiMaxDepth);
    std::vector<SearchKey> tmpSKeys;
    std::vector<Key> cgNodes;
    std::vector<SearchKey> skeys_cg;
    std::vector<SearchKey>::iterator hintSKey;
    unsigned int skip = 1;
    unsigned int i_cg, i_dg;
    std::vector<unsigned int> *ownerList_ptr;

    for (unsigned int index = 0; index < E2N_DG_Sorted.size(); index++) {
        dg2eijk(E2N_DG_Sorted[index], owner1, ii_x1, jj_y1, kk_z1);
        assert(owner1 < m_uiAllElements.size());
        nsz = 1u << (m_uiMaxDepth - m_uiAllElements[owner1].getLevel());
        assert(nsz % m_uiElementOrder == 0);
        hintSKey = skeys_cg.emplace(
            skeys_cg.end(),
            SearchKey((m_uiAllElements[owner1].getX()) +
                          (ii_x1 * nsz / m_uiElementOrder),
                      (m_uiAllElements[owner1].getY()) +
                          (jj_y1 * nsz / m_uiElementOrder),
                      (m_uiAllElements[owner1].getZ()) +
                          (kk_z1 * nsz / m_uiElementOrder),
                      m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
        hintSKey->addOwner(E2N_DG_Sorted[index]);
    }

    SFC::seqSort::SFC_treeSort(&(*(skeys_cg.begin())), skeys_cg.size(),
                               tmpSKeys, tmpSKeys, tmpSKeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

    for (unsigned int e = 0; e < skeys_cg.size(); e++) {
        skip    = 1;
        tmpSKey = skeys_cg[e];

        tmpKey = Key(skeys_cg[e].getX(), skeys_cg[e].getY(), skeys_cg[e].getZ(),
                     skeys_cg[e].getLevel(), m_uiDim, m_uiMaxDepth + 1);
        tmpKey.addOwner(skeys_cg[e].getOwner());

        while (((e + skip) < skeys_cg.size()) &&
               (skeys_cg[e] == skeys_cg[e + skip])) {
            tmpKey.addOwner(skeys_cg[e + skip].getOwner());
            dg2eijk(tmpSKey.getOwner(), owner1, ii_x1, jj_y1, kk_z1);
            dg2eijk(skeys_cg[e + skip].getOwner(), owner2, ii_x2, jj_y2, kk_z2);

            lev1 = m_uiAllElements[owner1].getLevel();
            lev2 = m_uiAllElements[owner2].getLevel();

            if (lev1 == lev2) {
                lev1 = owner1;
                lev2 = owner2;
            }
            assert(lev1 != lev2);

            if (lev1 > lev2) tmpSKey.addOwner(skeys_cg[e + skip].getOwner());

            skip++;
        }
        m_uiCG2DG.push_back(tmpSKey.getOwner());
        tmpKey.setSearchResult(tmpSKey.getOwner());
        cgNodes.push_back(tmpKey);
        e += (skip - 1);
    }

    std::sort(m_uiCG2DG.begin(), m_uiCG2DG.end());

    for (unsigned int i = 0; i < m_uiCG2DG.size(); i++)
        m_uiDG2CG[m_uiCG2DG[i]] = i;

    for (unsigned int i = 0; i < cgNodes.size(); i++) {
        ownerList_ptr = cgNodes[i].getOwnerList();
        if (ownerList_ptr->size() > 1) {
            i_cg = (std::lower_bound(m_uiCG2DG.begin(), m_uiCG2DG.end(),
                                     cgNodes[i].getSearchResult()) -
                    m_uiCG2DG.begin());
            assert(i_cg < m_uiCG2DG.size());
            for (unsigned int w = 0; w < ownerList_ptr->size(); w++) {
                m_uiDG2CG[(*(ownerList_ptr))[w]] = i_cg;
                dg2dg_p[(*(ownerList_ptr))[w]]   = cgNodes[i].getSearchResult();
            }
        }
    }

    for (unsigned int i = 0; i < m_uiE2NMapping_CG.size(); i++)
        if (dg2dg_p[m_uiE2NMapping_CG[i]] != LOOK_UP_TABLE_DEFAULT)
            m_uiE2NMapping_CG[i] = dg2dg_p[m_uiE2NMapping_CG[i]];

#ifdef DEBUG_E2N_MAPPING
    // MPI_Barrier(MPI_COMM_WORLD);
    if (!m_uiActiveRank)
        std::cout << "m_uiActiveRank: " << m_uiActiveRank
                  << "Number of actual nodes: " << (E2N_DG_Sorted.size())
                  << std::endl;
#endif

    m_uiNodePreGhostBegin   = UINT_MAX;
    m_uiNodeLocalBegin      = UINT_MAX;
    m_uiNodePostGhostBegin  = UINT_MAX;

    unsigned int preOwner   = UINT_MAX;
    unsigned int localOwner = UINT_MAX;
    unsigned int postOwner  = UINT_MAX;

    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPostGhostEnd;
         e++) {
        unsigned int tmpIndex;
        for (unsigned int k = 0; k < m_uiNpE; k++) {
            tmpIndex = (m_uiE2NMapping_CG[e * m_uiNpE + k] / m_uiNpE);
            assert(tmpIndex == (((m_uiE2NMapping_CG[e * m_uiNpE + k]) /
                                 (m_uiElementOrder + 1)) /
                                (m_uiElementOrder + 1)) /
                                   (m_uiElementOrder + 1));
            if ((tmpIndex >= m_uiElementPreGhostBegin) &&
                (tmpIndex < m_uiElementPreGhostEnd) &&
                /*(preOwner>=(m_uiE2NMapping_CG[e * m_uiNpE + k])/m_uiNpE) &&*/
                (m_uiNodePreGhostBegin > m_uiE2NMapping_CG[e * m_uiNpE + k])) {
                // preOwner = m_uiE2NMapping_CG[e * m_uiNpE + k]/m_uiNpE;
                m_uiNodePreGhostBegin = m_uiE2NMapping_CG[e * m_uiNpE + k];
            }

            if ((tmpIndex >= m_uiElementLocalBegin) &&
                (tmpIndex < m_uiElementLocalEnd) &&
                /*(localOwner >=(m_uiE2NMapping_CG[e * m_uiNpE + k])/m_uiNpE)
                   &&*/
                (m_uiNodeLocalBegin > m_uiE2NMapping_CG[e * m_uiNpE + k])) {
                // localOwner = m_uiE2NMapping_CG[e * m_uiNpE + k]/m_uiNpE;
                m_uiNodeLocalBegin = m_uiE2NMapping_CG[e * m_uiNpE + k];
            }

            if ((tmpIndex >= m_uiElementPostGhostBegin) &&
                (tmpIndex < m_uiElementPostGhostEnd) &&
                /*(postOwner >=(m_uiE2NMapping_CG[e * m_uiNpE + k])/m_uiNpE)
                   &&*/
                (m_uiNodePostGhostBegin > m_uiE2NMapping_CG[e * m_uiNpE + k])) {
                // postOwner = m_uiE2NMapping_CG[e * m_uiNpE + k]/m_uiNpE;
                m_uiNodePostGhostBegin = m_uiE2NMapping_CG[e * m_uiNpE + k];
            }
        }
    }

    assert(m_uiNodeLocalBegin !=
           UINT_MAX);  // local node begin should be found.
    assert(m_uiDG2CG[m_uiNodeLocalBegin] != LOOK_UP_TABLE_DEFAULT);
    m_uiNodeLocalBegin = m_uiDG2CG
        [m_uiNodeLocalBegin];  //(std::lower_bound(E2N_DG_Sorted.begin(),E2N_DG_Sorted.end(),m_uiNodeLocalBegin)-E2N_DG_Sorted.begin());
    if (m_uiNodePreGhostBegin == UINT_MAX) {
        m_uiNodePreGhostBegin = 0;
        m_uiNodePreGhostEnd   = 0;
        assert(m_uiNodeLocalBegin == 0);
    } else {
        assert(m_uiDG2CG[m_uiNodePreGhostBegin] != LOOK_UP_TABLE_DEFAULT);
        m_uiNodePreGhostBegin = m_uiDG2CG
            [m_uiNodePreGhostBegin];  //(std::lower_bound(E2N_DG_Sorted.begin(),E2N_DG_Sorted.end(),m_uiNodePreGhostBegin)-E2N_DG_Sorted.begin());
        m_uiNodePreGhostEnd = m_uiNodeLocalBegin;
    }

    if (m_uiNodePostGhostBegin == UINT_MAX) {
        m_uiNodeLocalEnd       = m_uiCG2DG.size();  // E2N_DG_Sorted.size();
        m_uiNodePostGhostBegin = m_uiNodeLocalEnd;
        m_uiNodePostGhostEnd   = m_uiNodeLocalEnd;
    } else {
        assert(m_uiDG2CG[m_uiNodePostGhostBegin] != LOOK_UP_TABLE_DEFAULT);
        m_uiNodePostGhostBegin = m_uiDG2CG
            [m_uiNodePostGhostBegin];  //(std::lower_bound(E2N_DG_Sorted.begin(),E2N_DG_Sorted.end(),m_uiNodePostGhostBegin)-E2N_DG_Sorted.begin());
        m_uiNodeLocalEnd     = m_uiNodePostGhostBegin;
        m_uiNodePostGhostEnd = m_uiCG2DG.size();  // E2N_DG_Sorted.size();
    }

    m_uiNumActualNodes = cgNodes.size();

    m_uiE2NMapping_DG.assign(m_uiE2NMapping_CG.begin(),
                             m_uiE2NMapping_CG.end());

    for (unsigned int i = 0; i < m_uiE2NMapping_CG.size(); i++) {
        assert(m_uiDG2CG[m_uiE2NMapping_CG[i]] != LOOK_UP_TABLE_DEFAULT);
        m_uiE2NMapping_CG[i] = m_uiDG2CG[m_uiE2NMapping_CG[i]];
    }

    dg2dg_p.clear();
    E2N_DG_Sorted.clear();

#ifdef DEBUG_E2N_MAPPING
    MPI_Barrier(MPI_COMM_WORLD);
    if (m_uiActiveRank) std::cout << " DG to CG index updated " << std::endl;
#endif

    /*for(unsigned int
    e=m_uiElementPreGhostBegin;e<m_uiElementPostGhostEnd;e++)
    {
        if(m_uiActiveRank==2 && e==28)
        {

            std::vector<ot::TreeNode> cusEleCheck;
            unsigned int ownerID,ii_x,jj_y,kk_z; // DG index to ownerID and
    ijk decomposition variable. unsigned int x,y,z,sz;
            cusEleCheck.push_back(m_uiAllElements[e]);
            for(unsigned int node=0;node<m_uiNpE;node++)
            {

                dg2eijk(m_uiE2NMapping_DG[e*m_uiNpE+node],ownerID,ii_x,jj_y,kk_z);

                x=m_uiAllElements[ownerID].getX();
                y=m_uiAllElements[ownerID].getY();
                z=m_uiAllElements[ownerID].getZ();
                sz=1u<<(m_uiMaxDepth-m_uiAllElements[ownerID].getLevel());
                cusEleCheck.push_back(m_uiAllElements[ownerID]);

                cusEleCheck.push_back(ot::TreeNode((x + ii_x *
    sz/m_uiElementOrder), (y + jj_y * sz/m_uiElementOrder), (z + kk_z *
    sz/m_uiElementOrder), m_uiMaxDepth,m_uiDim, m_uiMaxDepth));

            }

            treeNodesTovtk(cusEleCheck,e,"cusE2N_3");

        }

    }*/

    /*unsigned int eleIndex;
    if(!m_uiActiveRank)  std::cout<<"E2N  rank :
    "<<m_uiActiveRank<<std::endl; if(!m_uiActiveRank) for(unsigned int
    e=0;e<m_uiAllElements.size();e++)
        {
            if(m_uiAllElements[e]==ot::TreeNode(24, 12, 40,
    4,m_uiDim,m_uiMaxDepth)) { std::cout << "rank: "<<m_uiActiveRank<<"
    Element : "<<e<<" " << m_uiAllElements[e] << " : Node List :"; for
    (unsigned int k = 0; k < m_uiNpE; k++) { std::cout << " " <<
    m_uiE2NMapping_DG[e * m_uiNpE + k];
            }

            std::cout << std::endl;
            }

        }*/
    //--------------------------------------------------------PRINT THE E2N
    // MAP------------------------------------------------------------------------------------
    /*for(unsigned int w=0;w<m_uiE2NMapping_CG.size();w++)
        std::cout<<"w: "<<w<<" -> : "<<m_uiE2NMapping_CG[w]<<std::endl;*/

    //--------------------------------------------------------PRINT THE E2N
    // MAP------------------------------------------------------------------------------------

#ifdef DEBUG_E2N_MAPPING
    // MPI_Barrier(MPI_COMM_WORLD);
    if (!m_uiActiveRank)
        std::cout << "[NODE] rank:  " << m_uiActiveRank << " pre ( "
                  << m_uiNodePreGhostBegin << ", " << m_uiNodePreGhostEnd
                  << ") local ( " << m_uiNodeLocalBegin << ", "
                  << m_uiNodeLocalEnd << ")"
                  << " post (" << m_uiNodePostGhostBegin << " , "
                  << m_uiNodePostGhostEnd << ")" << std::endl;
    if (!m_uiActiveRank)
        std::cout << "[ELEMENT] rank:  " << m_uiActiveRank << " pre ( "
                  << m_uiElementPreGhostBegin << ", " << m_uiElementPreGhostEnd
                  << ") local ( " << m_uiElementLocalBegin << ", "
                  << m_uiElementLocalEnd << ")"
                  << " post (" << m_uiElementPostGhostBegin << " , "
                  << m_uiElementPostGhostEnd << ")" << std::endl;

    if (m_uiActiveRank)
        std::cout << "[NODE] rank:  " << m_uiActiveRank << " pre ( "
                  << m_uiNodePreGhostBegin << ", " << m_uiNodePreGhostEnd
                  << ") local ( " << m_uiNodeLocalBegin << ", "
                  << m_uiNodeLocalEnd << ")"
                  << " post (" << m_uiNodePostGhostBegin << " , "
                  << m_uiNodePostGhostEnd << ")" << std::endl;
    if (m_uiActiveRank)
        std::cout << "[ELEMENT] rank:  " << m_uiActiveRank << " pre ( "
                  << m_uiElementPreGhostBegin << ", " << m_uiElementPreGhostEnd
                  << ") local ( " << m_uiElementLocalBegin << ", "
                  << m_uiElementLocalEnd << ")"
                  << " post (" << m_uiElementPostGhostBegin << " , "
                  << m_uiElementPostGhostEnd << ")" << std::endl;

#endif

    //---------------------------------------print out the E2N mapping of fake
    // elements. (This is done for only the fake elements. )
    //---------------------------------------------------------------------------------

    /* MPI_Barrier(MPI_COMM_WORLD);
        if(!m_uiActiveRank){
            std::cout<<"rank: "<<m_uiActiveRank<<"fake element e2n mapping.
       "<<std::endl; std::cout<<"number of Fake Elements :
       "<<fakeElements_vec.size()<<std::endl; std::cout<<"number of FakeElement
       Nodes: "<<m_uiNumFakeNodes<<std::endl; std::cout<<"[Fake ELEMENT] rank:
       "<<m_uiActiveRank<<" pre ( "<<m_uiFElementPreGhostBegin<<",
       "<<m_uiFElementPreGhostEnd<<") local ( "<<m_uiFElementLocalBegin<<",
       "<<m_uiFElementLocalEnd<<")"<<" post ("<<m_uiFElementPostGhostBegin<<" ,
       "<<m_uiFElementPostGhostEnd<<")"<<std::endl; for(unsigned int
       e=0;e<fakeElements_vec.size();e++)
        {
                std::cout << "Element : "<<e<<" " << fakeElements_vec[e] << " :
       Node List :"; for (unsigned int k = 0; k < m_uiNpE; k++) {

                    std::cout << " " << fakeElement2Node_CG[e * m_uiNpE + k];

                }

                std::cout << std::endl;


        }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(m_uiActiveRank==1){
            std::cout<<"rank: "<<m_uiActiveRank<<"fake element e2n mapping.
       "<<std::endl; std::cout<<"number of Fake Elements :
       "<<fakeElements_vec.size()<<std::endl; std::cout<<"number of FakeElement
       Nodes: "<<m_uiNumFakeNodes<<std::endl; std::cout<<"[Fake ELEMENT] rank:
       "<<m_uiActiveRank<<" pre ( "<<m_uiFElementPreGhostBegin<<",
       "<<m_uiFElementPreGhostEnd<<") local ( "<<m_uiFElementLocalBegin<<",
       "<<m_uiFElementLocalEnd<<")"<<" post ("<<m_uiFElementPostGhostBegin<<" ,
       "<<m_uiFElementPostGhostEnd<<")"<<std::endl; for(unsigned int
       e=0;e<fakeElements_vec.size();e++)
            {
                std::cout << "Element : "<<e<<" " << fakeElements_vec[e] << " :
       Node List :"; for (unsigned int k = 0; k < m_uiNpE; k++) {

                    std::cout << " " << fakeElement2Node_CG[e * m_uiNpE + k];

                }

                std::cout << std::endl;


            }
        }
        MPI_Barrier(MPI_COMM_WORLD);*/

    //-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // 8. Change the size of the original E2N mapping and copy fake element to
    // node mapping at the end of the actual element to node mapping.
    // m_uiE2NMapping_CG.resize(m_uiNumTotalElements*m_uiNpE+(m_uiFElementPostGhostEnd-m_uiFElementPreGhostBegin)*m_uiNpE);
    // memcpy(&(*(m_uiE2NMapping_CG.begin()+(m_uiNumTotalElements*m_uiNpE))),&(*(fakeElement2Node_CG.begin())),sizeof(unsigned
    // int )*(m_uiFElementPostGhostEnd-m_uiFElementPreGhostBegin)*m_uiNpE);

    //---------------------------------------print out the final e2n mapping of
    // all, actual and fake element to node
    // mapping.--------------------------------------------------------------------------

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // if(!m_uiActiveRank) std::cout<<"E2N Mapping ended"<<std::endl;
}


void Mesh::buildFEM_E2N() {
    // todo we don't need to build the full e2n mapping only for the partition
    // boundary and R1 ghost elements.
    buildE2NMap();
    // for(unsigned int
    // ele=m_uiElementPreGhostBegin;ele<m_uiElementPreGhostEnd;ele++)
    // {
    //     for(unsigned int node=0;node<m_uiNpE;node++)
    //     {
    //         const unsigned nodeLookUp = m_uiE2NMapping_CG [ele*m_uiNpE +
    //         node]; if(nodeLookUp>=m_uiNodeLocalBegin &&
    //         nodeLookUp<m_uiNodeLocalEnd)
    //         {
    //             m_uiGhostElementRound1Index.push_back(ele);
    //             break;
    //         }
    //     }
    // }

    // for(unsigned int
    // ele=m_uiElementPostGhostBegin;ele<m_uiElementPostGhostEnd;ele++)
    // {
    //     for(unsigned int node=0;node<m_uiNpE;node++)
    //     {
    //         const unsigned nodeLookUp = m_uiE2NMapping_CG [ele*m_uiNpE +
    //         node]; if(nodeLookUp>=m_uiNodeLocalBegin &&
    //         nodeLookUp<m_uiNodeLocalEnd)
    //         {
    //             m_uiGhostElementRound1Index.push_back(ele);
    //             break;
    //         }
    //     }
    // }

    // std::sort(m_uiGhostElementRound1Index.begin(),
    // m_uiGhostElementRound1Index.end());
    // m_uiGhostElementRound1Index.erase(std::unique(m_uiGhostElementRound1Index.begin(),
    // m_uiGhostElementRound1Index.end()),m_uiGhostElementRound1Index.end());

    // m_uiCG2DG.clear();
    // buildE2NMap();
}

void Mesh::buildE2N_DG() {
    dendro::logger::debug(dendro::logger::Scope{"MESH"},
                          "Now constructing E2N map (DG)");
    unsigned int lookUp = 0;
    unsigned int lev1   = 0;
    unsigned int lev2   = 0;

    unsigned int child;
    unsigned int parent;

#ifdef DEBUG_E2N_MAPPING
    for (unsigned int ge = m_uiElementPreGhostBegin;
         ge < m_uiElementPreGhostEnd; ge++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++)
            if (m_uiE2EMapping[ge * m_uiNumDirections + dir] !=
                LOOK_UP_TABLE_DEFAULT)
                assert((m_uiE2EMapping[ge * m_uiNumDirections + dir] >=
                        m_uiElementLocalBegin) &&
                       (m_uiE2EMapping[ge * m_uiNumDirections + dir] <
                        m_uiElementLocalEnd));
    }

    for (unsigned int ge = m_uiElementPostGhostBegin;
         ge < m_uiElementPostGhostEnd; ge++) {
        for (unsigned int dir = 0; dir < m_uiNumDirections; dir++)
            if (m_uiE2EMapping[ge * m_uiNumDirections + dir] !=
                LOOK_UP_TABLE_DEFAULT)
                assert((m_uiE2EMapping[ge * m_uiNumDirections + dir] >=
                        m_uiElementLocalBegin) &&
                       (m_uiE2EMapping[ge * m_uiNumDirections + dir] <
                        m_uiElementLocalEnd));
    }

#endif

    assert(m_uiNumTotalElements == m_uiAllElements.size());
    assert((m_uiElementPostGhostEnd - m_uiElementPreGhostBegin) > 0);
    assert(m_uiNumTotalElements ==
           ((m_uiElementPostGhostEnd - m_uiElementPreGhostBegin)));

    m_uiE2NMapping_CG.resize(m_uiNumTotalElements * m_uiNpE);
    m_uiE2NMapping_DG.resize(m_uiNumTotalElements * m_uiNpE);

    // initialize the DG mapping. // this order is mandotory.
    for (unsigned int e = 0; e < (m_uiNumTotalElements); e++)
        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
             k++)  // z coordinate
            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                 j++)  // y coordinate
                for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                     i++)  // x coordinate
                    m_uiE2NMapping_CG[e * m_uiNpE +
                                      k * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      j * (m_uiElementOrder + 1) + i] =
                        e * m_uiNpE +
                        k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i;

    m_uiNodePreGhostBegin  = m_uiElementPreGhostBegin * m_uiNpE;
    m_uiNodePreGhostEnd    = m_uiElementPreGhostEnd * m_uiNpE;

    m_uiNodeLocalBegin     = m_uiElementLocalBegin * m_uiNpE;
    m_uiNodeLocalEnd       = m_uiElementLocalEnd * m_uiNpE;

    m_uiNodePostGhostBegin = m_uiElementPostGhostBegin * m_uiNpE;
    m_uiNodePostGhostEnd   = m_uiElementPostGhostEnd * m_uiNpE;

    m_uiNumActualNodes     = (m_uiNodePreGhostEnd - m_uiNodePreGhostBegin) +
                         (m_uiNodeLocalEnd - m_uiNodeLocalBegin) +
                         (m_uiNodeLocalEnd - m_uiNodeLocalBegin);

    m_uiE2NMapping_DG = m_uiE2NMapping_CG;
    m_uiCG2DG         = m_uiE2NMapping_CG;
    // m_uiCG2DG.resize(m_uiE2NMapping_CG.size(),1);

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished constructing E2N Map (DG)!");
}

void Mesh::deriveOwnerMasksFromCascade() {
    // Sized to match m_uiAllElements (local + ghost). At this point in
    // construction, m_uiAllElements is finalized and E2N_DG is the
    // cascade's authoritative ownership map.
    const size_t nElTot = m_uiAllElements.size();
    m_uiOwnerMask.assign(nElTot, 0u);
    if (!m_uiIsActive || nElTot == 0) return;

    const unsigned int npe  = m_uiNpE;
    const unsigned int eOrd = m_uiElementOrder;
    if (eOrd == 0) return;  // degenerate; nothing to do

    // For each of the 27 eOrd=2 representative coordinates
    // (ni27, nj27, nk27) ∈ {0, 1, 2}^3, compute the actual sub-index
    // at the mesh's eOrd. The "interior" representative maps to the
    // element's interior midpoint; "boundary low/high" map to the
    // element's 0 / eOrd ends.
    auto subForRep = [&](unsigned int rep) -> unsigned int {
        const unsigned int ni27 = rep % 3;
        const unsigned int nj27 = (rep / 3) % 3;
        const unsigned int nk27 = rep / 9;
        // Pick a STABLE representative within each category for
        // higher orders. The mid-point is `eOrd / 2` (integer
        // division). For eOrd=2 this collapses to the natural sub
        // index 0/1/2 and the formula is exact.
        const unsigned int ni = (ni27 == 0) ? 0u
                              : (ni27 == 2) ? eOrd
                              : (eOrd / 2);
        const unsigned int nj = (nj27 == 0) ? 0u
                              : (nj27 == 2) ? eOrd
                              : (eOrd / 2);
        const unsigned int nk = (nk27 == 0) ? 0u
                              : (nk27 == 2) ? eOrd
                              : (eOrd / 2);
        return nk * (eOrd + 1) * (eOrd + 1) + nj * (eOrd + 1) + ni;
    };

    for (size_t e = 0; e < nElTot; e++) {
        uint32_t mask = 0u;
        for (unsigned int r = 0; r < 27; r++) {
            const unsigned int sub = subForRep(r);
            const unsigned int dg  =
                m_uiE2NMapping_DG[e * npe + sub];
            // Self-owned canonical writer test: the cascade's
            // canonical DG slot for this (elem, sub) points back at
            // this elem.
            if (dg / npe == (unsigned int)e) mask |= (1u << r);
        }
        m_uiOwnerMask[e] = mask;
    }
}

void Mesh::deriveBlockInfoFromBlocks() {
    // sized to match m_uiAllElements (local + ghost). ghost slots stay
    // zero-valued (invalid) — only local elements have block info on
    // any given rank. that's fine because block info travels with each
    // element from its source rank's POV; whoever ships a given
    // element ships ITS canonical block info.
    const size_t nElTot = m_uiAllElements.size();
    m_uiBlockInfo.assign(nElTot, CanonicalBlockInfo{});
    if (!m_uiIsActive) return;
    if (m_uiLocalBlockList.empty()) return;

    for (const auto& blk : m_uiLocalBlockList) {
        const ot::TreeNode anchor = blk.getBlockNode();
        const uint32_t regLev     = blk.getRegularGridLev() & 0xFFu;
        const uint32_t rotID      = blk.getRotationID() & 0xFFu;
        const uint32_t meta       = regLev | (rotID << 8) | (1u << 31);

        for (DendroIntL e : blk) {
            if ((size_t)e >= nElTot) continue;
            CanonicalBlockInfo bi;
            bi.anchorX     = anchor.getX();
            bi.anchorY     = anchor.getY();
            bi.anchorZ     = anchor.getZ();
            bi.anchorLevel = anchor.getLevel();
            bi.meta        = meta;
            m_uiBlockInfo[e] = bi;
        }
    }
}

size_t Mesh::buildBlocksFromCanonicalInfo() {
    if (!m_uiIsActive) return 0;
    if (m_uiBlockInfo.empty()) return 0;

    // bucket key: pack anchor (x, y, z, level) into a 128-bit-ish key.
    // x/y/z fit in m_uiMaxDepth bits (typically <30), level in 5 bits.
    // since coords could span up to 2^30, we use a tuple key under a
    // map. block count is ~elements/30, so map overhead is negligible.
    struct AnchorKey {
        uint32_t x, y, z, lev;
        bool operator<(const AnchorKey& o) const {
            if (x != o.x) return x < o.x;
            if (y != o.y) return y < o.y;
            if (z != o.z) return z < o.z;
            return lev < o.lev;
        }
    };

    std::map<AnchorKey, std::vector<DendroIntL>> blockGroups;
    std::map<AnchorKey, uint32_t> blockMetaForAnchor;

    for (DendroIntL e = (DendroIntL)m_uiElementLocalBegin;
         e < (DendroIntL)m_uiElementLocalEnd; e++) {
        if ((size_t)e >= m_uiBlockInfo.size()) continue;
        const auto& bi = m_uiBlockInfo[e];
        if (!bi.isValid()) continue;
        AnchorKey k{bi.anchorX, bi.anchorY, bi.anchorZ, bi.anchorLevel};
        blockGroups[k].push_back(e);
        // first writer wins; meta should be identical for any element
        // tagged into the same block, by construction
        if (blockMetaForAnchor.find(k) == blockMetaForAnchor.end())
            blockMetaForAnchor[k] = bi.meta;
    }

    if (blockGroups.empty()) return 0;

    m_uiLocalBlockList.clear();
    m_uiLocalBlockList.reserve(blockGroups.size());

    for (auto& kv : blockGroups) {
        const AnchorKey& key = kv.first;
        std::vector<DendroIntL>& indices = kv.second;
        const uint32_t meta = blockMetaForAnchor[key];
        const unsigned int regLev = meta & 0xFFu;
        const unsigned int rotID  = (meta >> 8) & 0xFFu;

        ot::TreeNode anchorTN(key.x, key.y, key.z, key.lev, m_uiDim,
                              m_uiMaxDepth);
        m_uiLocalBlockList.emplace_back(anchorTN, rotID, regLev, indices,
                                        m_uiElementOrder);
    }

    // diagnostic: cross-rank block sharing detection. if any anchor
    // appears on more than one rank, we have partial blocks. partial
    // blocks may have missing interior elements (deeper than the R1
    // ghost layer reaches), which corrupts FD stencils.
    if (DENDRO_PROBE_GETENV("DENDRO_BLOCK_DIAG")) {
        const int rank = m_uiActiveRank;
        const int npes = m_uiActiveNpes;
        const size_t nLocalBlocks = m_uiLocalBlockList.size();
        size_t totalGlobal = 0;
        MPI_Allreduce(&nLocalBlocks, &totalGlobal, 1, MPI_UINT64_T,
                      MPI_SUM, m_uiCommActive);

        // count unique anchors globally via collision detection
        std::vector<uint32_t> localAnchors;
        localAnchors.reserve(nLocalBlocks * 4);
        for (const auto& kv : blockGroups) {
            localAnchors.push_back(kv.first.x);
            localAnchors.push_back(kv.first.y);
            localAnchors.push_back(kv.first.z);
            localAnchors.push_back(kv.first.lev);
        }
        std::vector<int> recvCounts(npes, 0);
        int sendCount = (int)localAnchors.size();
        MPI_Allgather(&sendCount, 1, MPI_INT, recvCounts.data(), 1,
                      MPI_INT, m_uiCommActive);
        std::vector<int> recvOff(npes, 0);
        for (int i = 1; i < npes; i++)
            recvOff[i] = recvOff[i-1] + recvCounts[i-1];
        const int totalSize = recvOff[npes-1] + recvCounts[npes-1];
        std::vector<uint32_t> allAnchors(totalSize);
        MPI_Allgatherv(localAnchors.data(), sendCount, MPI_UINT32_T,
                       allAnchors.data(), recvCounts.data(),
                       recvOff.data(), MPI_UINT32_T, m_uiCommActive);

        // count anchors that appear on >1 rank
        std::map<AnchorKey, int> anchorRankCount;
        std::map<AnchorKey, std::vector<int>> anchorRanks;
        for (int r = 0; r < npes; r++) {
            const int n = recvCounts[r] / 4;
            for (int i = 0; i < n; i++) {
                AnchorKey k{allAnchors[recvOff[r] + 4*i],
                            allAnchors[recvOff[r] + 4*i + 1],
                            allAnchors[recvOff[r] + 4*i + 2],
                            allAnchors[recvOff[r] + 4*i + 3]};
                anchorRanks[k].push_back(r);
            }
        }
        size_t splitAnchors = 0;
        for (const auto& kv : anchorRanks)
            if (kv.second.size() > 1) splitAnchors++;
        if (rank == 0) {
            std::cout << "[block-diag] total local blocks per rank=" << totalGlobal
                      << " unique anchors=" << anchorRanks.size()
                      << " split-across-ranks=" << splitAnchors
                      << " (split-rate=" << (anchorRanks.empty() ? 0.0 :
                          (double)splitAnchors / anchorRanks.size())
                      << ")" << std::endl;
        }
    }

    // refresh m_uiDmin/m_uiDmax from local element levels (these are
    // normally populated by the local block decomposition; downstream
    // code uses them in validity assertions).
    m_uiDmin = m_uiMaxDepth;
    m_uiDmax = 0u;
    for (DendroIntL e = (DendroIntL)m_uiElementLocalBegin;
         e < (DendroIntL)m_uiElementLocalEnd; e++) {
        const unsigned int L = m_uiAllElements[e].getLevel();
        if (L < m_uiDmin) m_uiDmin = L;
        if (L > m_uiDmax) m_uiDmax = L;
    }

    return m_uiLocalBlockList.size();
}

// OCT_DIR axis components: each direction d ∈ [0, 26) has components
// (dx, dy, dz) ∈ {-1, 0, +1}. d=26 (OCT_DIR_INTERNAL) is the identity.
// Used by patchE2NCgFromMasks() to walk the ownership cascade across
// element-to-element boundaries.
static constexpr int OCT_DIR_COMP[26][3] = {
    {-1,  0,  0},  // 0  LEFT
    { 1,  0,  0},  // 1  RIGHT
    { 0, -1,  0},  // 2  DOWN
    { 0,  1,  0},  // 3  UP
    { 0,  0, -1},  // 4  BACK
    { 0,  0,  1},  // 5  FRONT
    {-1, -1,  0},  // 6  LEFT_DOWN
    {-1,  1,  0},  // 7  LEFT_UP
    {-1,  0, -1},  // 8  LEFT_BACK
    {-1,  0,  1},  // 9  LEFT_FRONT
    { 1, -1,  0},  // 10 RIGHT_DOWN
    { 1,  1,  0},  // 11 RIGHT_UP
    { 1,  0, -1},  // 12 RIGHT_BACK
    { 1,  0,  1},  // 13 RIGHT_FRONT
    { 0, -1, -1},  // 14 DOWN_BACK
    { 0, -1,  1},  // 15 DOWN_FRONT
    { 0,  1, -1},  // 16 UP_BACK
    { 0,  1,  1},  // 17 UP_FRONT
    {-1, -1, -1},  // 18 LEFT_DOWN_BACK
    { 1, -1, -1},  // 19 RIGHT_DOWN_BACK
    {-1,  1, -1},  // 20 LEFT_UP_BACK
    { 1,  1, -1},  // 21 RIGHT_UP_BACK
    {-1, -1,  1},  // 22 LEFT_DOWN_FRONT
    { 1, -1,  1},  // 23 RIGHT_DOWN_FRONT
    {-1,  1,  1},  // 24 LEFT_UP_FRONT
    { 1,  1,  1},  // 25 RIGHT_UP_FRONT
};

size_t Mesh::patchE2NCgFromMasks() {
    if (!m_uiIsActive) return 0;
    if (m_uiOwnerMask.empty()) return 0;
    const size_t nElTot = m_uiAllElements.size();
    if (m_uiOwnerMask.size() != nElTot) return 0;

    const unsigned int npe  = m_uiNpE;
    const unsigned int eOrd = m_uiElementOrder;
    if (eOrd == 0) return 0;

    // If the mask vector is uniformly zero, it means no owner data was
    // shipped/derived for this mesh (e.g. ReMesh-created intermediate
    // meshes or test paths that don't inject the EM4 graph-twin mask).
    // Skip patching entirely — the unfixed cascade is the only data we
    // have, and emitting patched=0/unresolved≈npe*nLocal warnings is
    // noisy without telling us anything actionable.
    {
        bool any_bits = false;
        for (size_t e = 0; e < nElTot; e++) {
            if (m_uiOwnerMask[e] != 0u) { any_bits = true; break; }
        }
        if (!any_bits) return 0;
    }

    const unsigned int eLB = m_uiElementLocalBegin;
    const unsigned int eLE = m_uiElementLocalEnd;
    const unsigned int numDirs = m_uiNumDirections;
    const ot::TreeNode* pN = m_uiAllElements.data();

    // representative for sub n at actual eOrd: collapse each axis to
    // {0, 1, 2} via {0, interior, eOrd}.
    auto repForSub = [&](unsigned int sub) -> unsigned int {
        const unsigned int ni = sub % (eOrd + 1);
        const unsigned int nj = (sub / (eOrd + 1)) % (eOrd + 1);
        const unsigned int nk = sub / ((eOrd + 1) * (eOrd + 1));
        const unsigned int ni27 = (ni == 0) ? 0u : (ni == eOrd) ? 2u : 1u;
        const unsigned int nj27 = (nj == 0) ? 0u : (nj == eOrd) ? 2u : 1u;
        const unsigned int nk27 = (nk == 0) ? 0u : (nk == eOrd) ? 2u : 1u;
        return nk27 * 9 + nj27 * 3 + ni27;
    };

    auto mirrorRep = [](unsigned int r, int dx, int dy, int dz)
        -> unsigned int {
        unsigned int ni = r % 3;
        unsigned int nj = (r / 3) % 3;
        unsigned int nk = r / 9;
        if (dx != 0) ni = 2 - ni;
        if (dy != 0) nj = 2 - nj;
        if (dz != 0) nk = 2 - nk;
        return nk * 9 + nj * 3 + ni;
    };

    auto mirrorSub = [&](unsigned int sub, int dx, int dy, int dz)
        -> unsigned int {
        unsigned int ni = sub % (eOrd + 1);
        unsigned int nj = (sub / (eOrd + 1)) % (eOrd + 1);
        unsigned int nk = sub / ((eOrd + 1) * (eOrd + 1));
        if (dx != 0) ni = eOrd - ni;
        if (dy != 0) nj = eOrd - nj;
        if (dz != 0) nk = eOrd - nk;
        return nk * (eOrd + 1) * (eOrd + 1) + nj * (eOrd + 1) + ni;
    };

    // Direction d is compatible with rep r iff for each axis where d
    // has a nonzero component, r's coord on that axis is at the
    // matching boundary. Interior-axis reps don't cross that axis.
    auto compatible = [](unsigned int r, int dx, int dy, int dz) -> bool {
        const unsigned int ni27 = r % 3;
        const unsigned int nj27 = (r / 3) % 3;
        const unsigned int nk27 = r / 9;
        if (dx != 0 && ni27 == 1) return false;
        if (dx == -1 && ni27 != 0) return false;
        if (dx == +1 && ni27 != 2) return false;
        if (dy != 0 && nj27 == 1) return false;
        if (dy == -1 && nj27 != 0) return false;
        if (dy == +1 && nj27 != 2) return false;
        if (dz != 0 && nk27 == 1) return false;
        if (dz == -1 && nk27 != 0) return false;
        if (dz == +1 && nk27 != 2) return false;
        return true;
    };

    // phys_pos fallback: when the 1-step E2EMapping walk can't find a
    // canonical owner (which happens at corners with multi-level
    // transitions, or under random/aggressive partitioning where the
    // 1-step neighbor is not in the local+ghost set in the right
    // direction), search ALL elements for one whose mask bit is set at
    // the same physical position. Returns owner_e + owner_sub on the
    // owner's eOrd grid. Operates entirely on local+ghost so no
    // communication is needed at this stage.
    auto findOwnerByPhysPos = [&](unsigned int e, unsigned int sub,
                                   unsigned int& owner_e_out,
                                   unsigned int& owner_sub_out) -> bool {
        const unsigned int L_e     = pN[e].getLevel();
        const unsigned int cellSz_e = 1u << (m_uiMaxDepth - L_e);
        const unsigned int ni = sub % (eOrd + 1);
        const unsigned int nj = (sub / (eOrd + 1)) % (eOrd + 1);
        const unsigned int nk = sub / ((eOrd + 1) * (eOrd + 1));
        // Compute target phys_pos in scaled-int space (multiplied by
        // eOrd) so the divisibility check is exact integer arithmetic.
        const uint64_t pxE = (uint64_t)pN[e].getX() * eOrd
                           + (uint64_t)ni * cellSz_e;
        const uint64_t pyE = (uint64_t)pN[e].getY() * eOrd
                           + (uint64_t)nj * cellSz_e;
        const uint64_t pzE = (uint64_t)pN[e].getZ() * eOrd
                           + (uint64_t)nk * cellSz_e;

        for (unsigned int cand = 0; cand < nElTot; cand++) {
            if (cand == e) continue;
            const uint32_t mask_c = m_uiOwnerMask[cand];
            if (mask_c == 0u) continue;
            const ot::TreeNode& tn = pN[cand];
            const unsigned int L_c     = tn.getLevel();
            const unsigned int cellSz_c = 1u << (m_uiMaxDepth - L_c);
            const uint64_t cX = (uint64_t)tn.getX() * eOrd;
            const uint64_t cY = (uint64_t)tn.getY() * eOrd;
            const uint64_t cZ = (uint64_t)tn.getZ() * eOrd;
            // bbox check (closed interval).
            if (pxE < cX || pyE < cY || pzE < cZ) continue;
            const uint64_t dxE = pxE - cX;
            const uint64_t dyE = pyE - cY;
            const uint64_t dzE = pzE - cZ;
            if (dxE > (uint64_t)eOrd * cellSz_c) continue;
            if (dyE > (uint64_t)eOrd * cellSz_c) continue;
            if (dzE > (uint64_t)eOrd * cellSz_c) continue;
            // sub-index alignment on cand's eOrd grid.
            if (dxE % cellSz_c != 0) continue;
            if (dyE % cellSz_c != 0) continue;
            if (dzE % cellSz_c != 0) continue;
            const uint64_t ni_c64 = dxE / cellSz_c;
            const uint64_t nj_c64 = dyE / cellSz_c;
            const uint64_t nk_c64 = dzE / cellSz_c;
            const unsigned int ni_c = (unsigned int)ni_c64;
            const unsigned int nj_c = (unsigned int)nj_c64;
            const unsigned int nk_c = (unsigned int)nk_c64;
            const unsigned int ni27 = (ni_c == 0) ? 0u
                                    : (ni_c == eOrd) ? 2u : 1u;
            const unsigned int nj27 = (nj_c == 0) ? 0u
                                    : (nj_c == eOrd) ? 2u : 1u;
            const unsigned int nk27 = (nk_c == 0) ? 0u
                                    : (nk_c == eOrd) ? 2u : 1u;
            const unsigned int rep_c = nk27 * 9 + nj27 * 3 + ni27;
            if ((mask_c >> rep_c) & 1u) {
                owner_e_out   = cand;
                owner_sub_out = nk_c * (eOrd + 1) * (eOrd + 1)
                              + nj_c * (eOrd + 1) + ni_c;
                return true;
            }
        }
        return false;
    };

    size_t patched         = 0;
    size_t patched_via_e2e = 0;
    size_t patched_via_pos = 0;
    size_t unresolved      = 0;

    for (unsigned int e = eLB; e < eLE; e++) {
        const uint32_t mask_e = m_uiOwnerMask[e];
        for (unsigned int sub = 0; sub < npe; sub++) {
            const unsigned int r = repForSub(sub);
            const bool maskSaysE = (mask_e >> r) & 1u;
            const unsigned int dg_cur =
                m_uiE2NMapping_DG[e * npe + sub];
            const bool cascadeSaysE = (dg_cur / npe == e);

            if (maskSaysE == cascadeSaysE) continue;
            if (maskSaysE && !cascadeSaysE) continue;  // under-claim,
                                                       // skipped in this pass
            // !maskSaysE && cascadeSaysE → over-claim. Find true owner
            // via E2EMapping + mask check; fall back to phys_pos search.

            unsigned int owner_e   = LOOK_UP_TABLE_DEFAULT;
            unsigned int owner_sub = 0;
            int owner_dx = 0, owner_dy = 0, owner_dz = 0;
            bool found_via_e2e = false;
            for (unsigned int d = 0; d < numDirs && d < 26; d++) {
                const int dx = OCT_DIR_COMP[d][0];
                const int dy = OCT_DIR_COMP[d][1];
                const int dz = OCT_DIR_COMP[d][2];
                if (!compatible(r, dx, dy, dz)) continue;
                const unsigned int n =
                    m_uiE2EMapping[e * numDirs + d];
                if (n == LOOK_UP_TABLE_DEFAULT) continue;
                if (n >= nElTot) continue;
                // Same-level fast path. Different-level cases fall
                // through to the phys_pos search below.
                if (pN[n].getLevel() != pN[e].getLevel()) continue;
                const unsigned int r_on_n =
                    mirrorRep(r, dx, dy, dz);
                if ((m_uiOwnerMask[n] >> r_on_n) & 1u) {
                    owner_e  = n;
                    owner_dx = dx;
                    owner_dy = dy;
                    owner_dz = dz;
                    found_via_e2e = true;
                    break;
                }
            }

            if (found_via_e2e) {
                owner_sub = mirrorSub(sub, owner_dx, owner_dy, owner_dz);
            } else {
                // Fallback: search all elements by phys_pos.
                if (!findOwnerByPhysPos(e, sub, owner_e, owner_sub)) {
                    unresolved++;
                    if (unresolved <= 8) {
                        const unsigned int L_e =
                            pN[e].getLevel();
                        const unsigned int cellSz_e =
                            1u << (m_uiMaxDepth - L_e);
                        const unsigned int ni = sub % (eOrd + 1);
                        const unsigned int nj =
                            (sub / (eOrd + 1)) % (eOrd + 1);
                        const unsigned int nk =
                            sub / ((eOrd + 1) * (eOrd + 1));
                        const unsigned int px =
                            pN[e].getX() + (ni * cellSz_e) / eOrd;
                        const unsigned int py =
                            pN[e].getY() + (nj * cellSz_e) / eOrd;
                        const unsigned int pz =
                            pN[e].getZ() + (nk * cellSz_e) / eOrd;
                        std::cout << "[mask-patch r"
                                  << m_uiActiveRank
                                  << "] WARNING unresolved over-claim:"
                                  << " e=" << e
                                  << " sub=" << sub
                                  << " rep=" << r
                                  << " gridPos=(" << px
                                  << "," << py
                                  << "," << pz << ")"
                                  << " level=" << L_e
                                  << std::endl;
                    }
                    continue;
                }
            }

            // Owner found. Build the redirect.
            const unsigned int owner_dg =
                owner_e * npe + owner_sub;
            const unsigned int owner_cg =
                m_uiE2NMapping_CG[owner_dg];

            // Stash old local cg as demoted; sync helper will mirror
            // owner's value into it post-exchange.
            const unsigned int old_cg = m_uiE2NMapping_CG[e * npe + sub];
            if (old_cg != owner_cg
                && old_cg >= m_uiNodeLocalBegin
                && old_cg <  m_uiNodeLocalEnd) {
                m_uiPassDDemotedLocalCgs.insert(old_cg);
                m_uiPassDDemotedToGhostCg[old_cg] = owner_cg;
            }

            // Redirect E2N_CG / E2N_DG to point at the owner's slot.
            const unsigned int old_dg_cur = dg_cur;
            m_uiE2NMapping_CG[e * npe + sub] = owner_cg;
            m_uiE2NMapping_DG[e * npe + sub] = owner_dg;

            // Sync cg2dg / dg2cg.
            if (e * npe + sub < m_uiDG2CG.size())
                m_uiDG2CG[e * npe + sub] = LOOK_UP_TABLE_DEFAULT;
            patched++;
            if (found_via_e2e) patched_via_e2e++;
            else               patched_via_pos++;

            // Optional debug. Gate on DENDRO_PATCH_DBG=1.
            static const char* dbg_env =
                DENDRO_PROBE_GETENV("DENDRO_PATCH_DBG");
            if (dbg_env && dbg_env[0] == '1' && dbg_env[1] == '\0') {
                const unsigned int L_e_dbg = pN[e].getLevel();
                const unsigned int cellSz_e_dbg =
                    1u << (m_uiMaxDepth - L_e_dbg);
                const unsigned int ni_dbg = sub % (eOrd + 1);
                const unsigned int nj_dbg =
                    (sub / (eOrd + 1)) % (eOrd + 1);
                const unsigned int nk_dbg =
                    sub / ((eOrd + 1) * (eOrd + 1));
                const unsigned int px_dbg =
                    pN[e].getX() + (ni_dbg * cellSz_e_dbg) / eOrd;
                const unsigned int py_dbg =
                    pN[e].getY() + (nj_dbg * cellSz_e_dbg) / eOrd;
                const unsigned int pz_dbg =
                    pN[e].getZ() + (nk_dbg * cellSz_e_dbg) / eOrd;
                std::cout << "[mask-patch r" << m_uiActiveRank
                          << " " << (found_via_e2e ? "e2e" : "pos")
                          << "] e=" << e
                          << " sub=" << sub
                          << " rep=" << r
                          << " gridPos=(" << px_dbg
                          << "," << py_dbg
                          << "," << pz_dbg << ")"
                          << " owner_e=" << owner_e
                          << " owner_sub=" << owner_sub
                          << " old_dg=" << old_dg_cur
                          << " new_dg=" << owner_dg
                          << " new_cg=" << owner_cg
                          << std::endl;
            }
        }
    }

    std::cout << "[mask-patch r" << m_uiActiveRank
              << "] patched=" << patched
              << " (via_e2e=" << patched_via_e2e
              << " via_pos=" << patched_via_pos << ")"
              << " unresolved=" << unresolved << std::endl;
    return patched;
}

size_t Mesh::validateOwnerMasksAgainstCurrentCascade() const {
    if (!m_uiIsActive) return 0;
    if (m_uiOwnerMask.empty()) return 0;
    const size_t nElTot = m_uiAllElements.size();
    if (m_uiOwnerMask.size() != nElTot) {
        std::cout << "[mask-validate r" << m_uiActiveRank
                  << "] m_uiOwnerMask size " << m_uiOwnerMask.size()
                  << " != m_uiAllElements " << nElTot << std::endl;
        return nElTot;  // signal "everything is suspect"
    }

    const unsigned int npe  = m_uiNpE;
    const unsigned int eOrd = m_uiElementOrder;
    if (eOrd == 0) return 0;

    auto subForRep = [&](unsigned int rep) -> unsigned int {
        const unsigned int ni27 = rep % 3;
        const unsigned int nj27 = (rep / 3) % 3;
        const unsigned int nk27 = rep / 9;
        const unsigned int ni = (ni27 == 0) ? 0u
                              : (ni27 == 2) ? eOrd
                              : (eOrd / 2);
        const unsigned int nj = (nj27 == 0) ? 0u
                              : (nj27 == 2) ? eOrd
                              : (eOrd / 2);
        const unsigned int nk = (nk27 == 0) ? 0u
                              : (nk27 == 2) ? eOrd
                              : (eOrd / 2);
        return nk * (eOrd + 1) * (eOrd + 1) + nj * (eOrd + 1) + ni;
    };

    // Only count disagreements for LOCAL elements. Ghost elements are
    // expected to have mask=set bits for representatives they own
    // GLOBALLY (carried from their pre-partition rank), but cascade
    // post-partition won't mark them self-owned because they aren't
    // local — that's a rank shift, not a bug.
    const unsigned int eLB = m_uiElementLocalBegin;
    const unsigned int eLE = m_uiElementLocalEnd;
    size_t disagreements = 0;
    size_t mask1_cascade0 = 0;  // mask says I own; cascade disagrees
    size_t mask0_cascade1 = 0;  // cascade says I own; mask disagrees
    size_t total_local_decisions = 0;
    for (size_t e = eLB; e < eLE; e++) {
        const uint32_t mask = m_uiOwnerMask[e];
        for (unsigned int r = 0; r < 27; r++) {
            const unsigned int sub = subForRep(r);
            const unsigned int dg  = m_uiE2NMapping_DG[e * npe + sub];
            const bool cascadeSaysE = (dg / npe == (unsigned int)e);
            const bool maskSaysE    = (mask >> r) & 1u;
            total_local_decisions++;
            if (cascadeSaysE != maskSaysE) {
                disagreements++;
                if (maskSaysE) mask1_cascade0++;
                else           mask0_cascade1++;
            }
        }
    }
    std::cout << "[mask-validate r" << m_uiActiveRank
              << "] local_decisions=" << total_local_decisions
              << " mask1_cascade0=" << mask1_cascade0
              << " mask0_cascade1=" << mask0_cascade1
              << std::endl;
    return disagreements;
}

void Mesh::buildE2BlockMap() {
    if (!m_uiIsActive) return;

    dendro::logger::debug(dendro::logger::Scope{"MESH"},
                          "Now building the element to block map");

    // clear all the maps.
    const unsigned int num_all_elements = m_uiAllElements.size();
    m_e2b_unzip_counts.resize(num_all_elements, 0);
    m_e2b_unzip_offset.resize(num_all_elements, 0);

    std::vector<unsigned int> eid;
    eid.reserve((NUM_CHILDREN + NUM_EDGES + NUM_FACES + 1) * 4);

    // 1: count block-element relationships
    for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
        this->blkUnzipElementIDs(blk, eid);
        // iterator handles both sfc and non-sfc blocks
        for (unsigned int elem : m_uiLocalBlockList[blk])
            m_e2b_unzip_counts[elem]++;

        for (unsigned int i = 0; i < eid.size(); i++) {
            const unsigned int elem = eid[i];
            m_e2b_unzip_counts[elem]++;
        }
    }
    m_e2b_unzip_offset[0] = 0;
    omp_par::scan(m_e2b_unzip_counts.data(), m_e2b_unzip_offset.data(),
                  m_e2b_unzip_counts.size());

    const unsigned int e2b_map_size = m_e2b_unzip_offset[num_all_elements - 1] +
                                      m_e2b_unzip_counts[num_all_elements - 1];
    m_e2b_unzip_map.resize(e2b_map_size, LOOK_UP_TABLE_DEFAULT);

    for (unsigned int i = 0; i < m_e2b_unzip_counts.size(); i++)
        m_e2b_unzip_counts[i] = 0;

    for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
        this->blkUnzipElementIDs(blk, eid);
        for (unsigned int elem : m_uiLocalBlockList[blk]) {
            m_e2b_unzip_map[m_e2b_unzip_offset[elem] +
                            m_e2b_unzip_counts[elem]] = blk;
            m_e2b_unzip_counts[elem]++;
        }

        for (unsigned int i = 0; i < eid.size(); i++) {
            const unsigned int elem                   = eid[i];
            m_e2b_unzip_map[m_e2b_unzip_offset[elem] +
                            m_e2b_unzip_counts[elem]] = blk;
            m_e2b_unzip_counts[elem]++;
        }
    }

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished building the element to block map!");

    return;
}

void Mesh::buildUnzipCanonicalWriterTable() {
    // build a per-(block_buffer offset) canonical-writer table so that
    // multi-writer block-padding slots produce partition-invariant output.
    //
    // strategy: replay unzip_scatter's write-slot enumeration (without
    // actually writing values), collect per-slot candidate writers, and
    // for each slot with >1 candidate elect ONE canonical winner using a
    // partition-invariant rule. the rule mirrors canon_mode=2's natural
    // "last writer wins" order (level desc, x asc, y asc, z asc) but
    // narrowed to JUST the writers of the slot — this is robust against
    // global iteration-order accidents and rank-specific ghost compositions
    // that could otherwise let a non-canonical writer slip through.

    m_uiUnzipCanonWriterBuilt = false;
    m_uiUnzipCanonWriter.clear();
    if (!m_uiIsActive) {
        m_uiUnzipCanonWriterBuilt = true;
        return;
    }
    if (m_uiLocalBlockList.empty()) {
        m_uiUnzipCanonWriterBuilt = true;
        return;
    }

    const unsigned int unSz   = this->getDegOfFreedomUnZip();
    if (unSz == 0) {
        m_uiUnzipCanonWriterBuilt = true;
        return;
    }
    m_uiUnzipCanonWriter.assign(unSz, LOOK_UP_TABLE_DEFAULT);

    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const ot::Block* blkList   = m_uiLocalBlockList.data();
    const unsigned int eOrder  = m_uiElementOrder;

    const double d_compar_tol = 1e-10;

    // candidate writers per (absolute bbuf offset). usually <=4 candidates.
    // most slots are single-writer; we lazy-allocate via map.
    std::unordered_map<unsigned int, std::vector<unsigned int>> candidates;
    candidates.reserve(unSz / 8);

    std::vector<ot::TreeNode> childOct;
    childOct.reserve(NUM_CHILDREN);

    auto record = [&](unsigned int idx, unsigned int ele) {
        candidates[idx].push_back(ele);
    };

    for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
        if (m_e2b_unzip_counts[ele] == 0) continue;

        for (unsigned int ii = 0; ii < m_e2b_unzip_counts[ele]; ii++) {
            const unsigned int e2b_offset = m_e2b_unzip_offset[ele];
            const unsigned int blk        = m_e2b_unzip_map[e2b_offset + ii];
            if (blk == LOOK_UP_TABLE_DEFAULT ||
                blk >= m_uiLocalBlockList.size())
                continue;

            const ot::TreeNode blkNode = blkList[blk].getBlockNode();
            const unsigned int PW      = blkList[blk].get1DPadWidth();
            const unsigned int lx      = blkList[blk].getAllocationSzX();
            const unsigned int ly      = blkList[blk].getAllocationSzY();
            const unsigned int lz      = blkList[blk].getAllocationSzZ();
            const unsigned int offset  = blkList[blk].getOffset();
            const unsigned int bLev    = blkList[blk].getRegularGridLev();
            const double hx = (1u << (m_uiMaxDepth - bLev)) / (double)eOrder;
            const double xmin = blkNode.minX() - PW * hx;
            const double xmax = blkNode.maxX() + PW * hx;
            const double ymin = blkNode.minY() - PW * hx;
            const double ymax = blkNode.maxY() + PW * hx;
            const double zmin = blkNode.minZ() - PW * hx;
            const double zmax = blkNode.maxZ() + PW * hx;

            if (pNodes[ele].getLevel() == bLev) {
                // same-level: direct copy at every (i,j,k) in ele
                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / hh;
                for (unsigned int k = 0; k < eOrder + 1; k++) {
                    double zz = pNodes[ele].minZ() + k * hh;
                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;
                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    if (kkz < 0 || kkz >= (int)lz) continue;
                    for (unsigned int j = 0; j < eOrder + 1; j++) {
                        double yy = pNodes[ele].minY() + j * hh;
                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;
                        if (yy < ymin || yy > ymax) continue;
                        const int jjy = std::round((yy - ymin) * invhh);
                        if (jjy < 0 || jjy >= (int)ly) continue;
                        for (unsigned int i = 0; i < eOrder + 1; i++) {
                            double xx = pNodes[ele].minX() + i * hh;
                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;
                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            if (iix < 0 || iix >= (int)lx) continue;
                            record(offset + kkz * lx * ly + jjy * lx + iix,
                                   ele);
                        }
                    }
                }
            } else if (pNodes[ele].getLevel() > bLev) {
                // finer: writes at every-other (i,j,k) of finer ele's grid
                const double hh =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double)eOrder;
                const double invhh = 1.0 / (2 * hh);
                const unsigned int cb = (eOrder % 2 == 0) ? 0 : 1;
                for (unsigned int k = cb; k < eOrder + 1; k += 2) {
                    double zz = pNodes[ele].minZ() + k * hh;
                    if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                    if (fabs(zz - zmax) < d_compar_tol) zz = zmax;
                    if (zz < zmin || zz > zmax) continue;
                    const int kkz = std::round((zz - zmin) * invhh);
                    if (kkz < 0 || kkz >= (int)lz) continue;
                    for (unsigned int j = cb; j < eOrder + 1; j += 2) {
                        double yy = pNodes[ele].minY() + j * hh;
                        if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                        if (fabs(yy - ymax) < d_compar_tol) yy = ymax;
                        if (yy < ymin || yy > ymax) continue;
                        const int jjy = std::round((yy - ymin) * invhh);
                        if (jjy < 0 || jjy >= (int)ly) continue;
                        for (unsigned int i = cb; i < eOrder + 1; i += 2) {
                            double xx = pNodes[ele].minX() + i * hh;
                            if (fabs(xx - xmin) < d_compar_tol) xx = xmin;
                            if (fabs(xx - xmax) < d_compar_tol) xx = xmax;
                            if (xx < xmin || xx > xmax) continue;
                            const int iix = std::round((xx - xmin) * invhh);
                            if (iix < 0 || iix >= (int)lx) continue;
                            record(offset + kkz * lx * ly + jjy * lx + iix,
                                   ele);
                        }
                    }
                }
            } else {
                // coarser: each child child's (i,j,k) maps via p2c
                childOct.clear();
                pNodes[ele].addChildren(childOct);
                for (unsigned int child = 0; child < NUM_CHILDREN; child++) {
                    if ((childOct[child].maxX() < xmin ||
                         childOct[child].minX() >= xmax) ||
                        (childOct[child].maxY() < ymin ||
                         childOct[child].minY() >= ymax) ||
                        (childOct[child].maxZ() < zmin ||
                         childOct[child].minZ() >= zmax))
                        continue;

                    const double hh =
                        (1u << (m_uiMaxDepth - childOct[child].getLevel())) /
                        (double)eOrder;
                    const double invhh = 1.0 / hh;
                    for (unsigned int k = 0; k < eOrder + 1; k++) {
                        double zz = childOct[child].minZ() + k * hh;
                        if (fabs(zz - zmin) < d_compar_tol) zz = zmin;
                        if (fabs(zz - zmax) < d_compar_tol) zz = zmax;
                        if (zz < zmin || zz > zmax) continue;
                        const int kkz = std::round((zz - zmin) * invhh);
                        if (kkz < 0 || kkz >= (int)lz) continue;
                        for (unsigned int j = 0; j < eOrder + 1; j++) {
                            double yy = childOct[child].minY() + j * hh;
                            if (fabs(yy - ymin) < d_compar_tol) yy = ymin;
                            if (fabs(yy - ymax) < d_compar_tol) yy = ymax;
                            if (yy < ymin || yy > ymax) continue;
                            const int jjy = std::round((yy - ymin) * invhh);
                            if (jjy < 0 || jjy >= (int)ly) continue;
                            for (unsigned int i = 0; i < eOrder + 1; i++) {
                                double xx = childOct[child].minX() + i * hh;
                                if (fabs(xx - xmin) < d_compar_tol)
                                    xx = xmin;
                                if (fabs(xx - xmax) < d_compar_tol)
                                    xx = xmax;
                                if (xx < xmin || xx > xmax) continue;
                                const int iix =
                                    std::round((xx - xmin) * invhh);
                                if (iix < 0 || iix >= (int)lx) continue;
                                record(offset + kkz * lx * ly + jjy * lx +
                                           iix,
                                       ele);
                            }
                        }
                    }
                }
            }
        }
    }


    // pick canonical winner per multi-writer slot: rule mirrors
    // canon_mode=2 — level asc then xyz desc. equivalently, canonical =
    // smallest level, largest x, largest y, largest z (the natural
    // "outermost-corner coarsest" candidate).
    auto canon_better = [&](unsigned int cand, unsigned int cur) {
        const ot::TreeNode& a = pNodes[cand];
        const ot::TreeNode& b = pNodes[cur];
        const unsigned int la = a.getLevel(), lb = b.getLevel();
        if (la != lb) return la < lb;
        if (a.getX() != b.getX()) return a.getX() > b.getX();
        if (a.getY() != b.getY()) return a.getY() > b.getY();
        return a.getZ() > b.getZ();
    };

    size_t mw_slots = 0;
    size_t mw_max   = 0;
    for (auto it = candidates.begin(); it != candidates.end(); ++it) {
        auto& eles = it->second;
        if (eles.size() <= 1) continue;
        mw_slots++;
        if (eles.size() > mw_max) mw_max = eles.size();
        unsigned int best = eles[0];
        for (unsigned int e : eles)
            if (canon_better(e, best)) best = e;
        m_uiUnzipCanonWriter[it->first] = best;
    }

    m_uiUnzipCanonWriterBuilt = true;

    if (m_uiActiveRank == 0) {
        std::cout << "[unzip-canon r0] slots=" << unSz
                  << " multi_writer_slots=" << mw_slots
                  << " max_writers_per_slot=" << mw_max
                  << std::endl;
    }

    // PHASE 0 PROBE — dump canon_writer_tbl per (block_anchor, slot phys)
    // for diffing graph vs skip and measuring how many slots elect a
    // different canonical writer between modes.
    //
    // Gate: DENDRO_CANON_DUMP_DIR + optional DENDRO_CANON_DUMP_CALL_ID:
    //   - if CALL_ID unset: dump at EVERY call (file per call_id)
    //   - if CALL_ID=<n>: dump only at the n-th call (one-shot)
    {
        static const char* canon_dump_dir =
            DENDRO_PROBE_GETENV("DENDRO_CANON_DUMP_DIR");
        static const char* canon_dump_call_env =
            DENDRO_PROBE_GETENV("DENDRO_CANON_DUMP_CALL_ID");
        static const int canon_dump_call =
            canon_dump_call_env ? std::atoi(canon_dump_call_env) : -1;
        static int canon_build_count = 0;
        const bool dump_this_call =
            canon_dump_dir && (canon_dump_call < 0
                               || canon_build_count == canon_dump_call);
        if (dump_this_call) {
            const int canon_dump_call_use = canon_build_count;
            // (use canon_build_count below to label the file uniquely)
            (void)canon_dump_call;  // suppress unused warning when -1
            char fn[1024];
            std::snprintf(fn, sizeof(fn),
                          "%s/canon_writer_call%d_r%d.txt",
                          canon_dump_dir, canon_dump_call_use,
                          (int)m_uiActiveRank);
            FILE* fp = std::fopen(fn, "w");
            if (fp) {
                std::fprintf(fp,
                    "# rank=%d call=%d numBlocks=%zu unSz=%u eOrd=%u\n"
                    "# blk anchor_lev anchor_x anchor_y anchor_z "
                    "regLev sx sy sz off "
                    "i j k phys_x phys_y phys_z "
                    "writer_idx writer_lev writer_x writer_y writer_z\n",
                    (int)m_uiActiveRank, canon_dump_call_use,
                    m_uiLocalBlockList.size(), unSz, eOrder);
                for (unsigned int b = 0; b < m_uiLocalBlockList.size(); b++) {
                    const auto& blk = m_uiLocalBlockList[b];
                    const auto& an  = blk.getBlockNode();
                    const unsigned int regLev = blk.getRegularGridLev();
                    const unsigned int sx     = blk.getAllocationSzX();
                    const unsigned int sy     = blk.getAllocationSzY();
                    const unsigned int sz     = blk.getAllocationSzZ();
                    const unsigned int off    = blk.getOffset();
                    const unsigned int PW     = blk.get1DPadWidth();
                    const double hx =
                        (1u << (m_uiMaxDepth - regLev)) / (double)eOrder;
                    const double bx0 = an.minX() - PW * hx;
                    const double by0 = an.minY() - PW * hx;
                    const double bz0 = an.minZ() - PW * hx;
                    for (unsigned int k = 0; k < sz; k++)
                    for (unsigned int j = 0; j < sy; j++)
                    for (unsigned int i = 0; i < sx; i++) {
                        const unsigned int slot =
                            off + k * sx * sy + j * sx + i;
                        if (slot >= m_uiUnzipCanonWriter.size()) continue;
                        const unsigned int w = m_uiUnzipCanonWriter[slot];
                        if (w == LOOK_UP_TABLE_DEFAULT) continue;
                        if (w >= m_uiAllElements.size()) continue;
                        const auto& wn = m_uiAllElements[w];
                        const double phx = bx0 + i * hx;
                        const double phy = by0 + j * hx;
                        const double phz = bz0 + k * hx;
                        std::fprintf(fp,
                            "%u %u %u %u %u  %u %u %u %u %u  "
                            "%u %u %u %g %g %g  "
                            "%u %u %u %u %u\n",
                            b, an.getLevel(), an.getX(), an.getY(), an.getZ(),
                            regLev, sx, sy, sz, off,
                            i, j, k, phx, phy, phz,
                            w, wn.getLevel(), wn.getX(), wn.getY(), wn.getZ());
                    }
                }
                std::fclose(fp);
            }
        }
        canon_build_count++;
    }
}

size_t Mesh::canonicalizeHangingFaceRoutingTN() {
    if (!m_uiIsActive) return 0;
    if (m_uiE2NMapping_CG.empty()) return 0;
    if (m_uiE2NMapping_DG.empty()) return 0;
    if (m_uiAllElements.empty()) return 0;

    const unsigned int npe     = m_uiNpE;
    const ot::TreeNode* pNodes = m_uiAllElements.data();
    const size_t allElemSize   = m_uiAllElements.size();
    const unsigned int eAllEnd =
        (unsigned int)std::min<size_t>(allElemSize,
                                       (size_t)m_uiNumTotalElements);

    // build TN -> local_idx hash on m_uiAllElements (one pass).
    struct TNHash {
        size_t operator()(const ot::TreeNode& t) const noexcept {
            size_t h = std::hash<unsigned int>()(t.getLevel());
            h ^= std::hash<unsigned int>()(t.getX())
                + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<unsigned int>()(t.getY())
                + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<unsigned int>()(t.getZ())
                + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct TNEq {
        bool operator()(const ot::TreeNode& a,
                        const ot::TreeNode& b) const noexcept {
            return a.getLevel() == b.getLevel()
                && a.getX() == b.getX()
                && a.getY() == b.getY()
                && a.getZ() == b.getZ();
        }
    };
    std::unordered_map<ot::TreeNode, unsigned int, TNHash, TNEq> tn2idx;
    tn2idx.reserve(eAllEnd * 2);
    for (unsigned int e = 0; e < eAllEnd; e++) {
        tn2idx.emplace(pNodes[e], e);
    }

    static const char* dbg_env =
        DENDRO_PROBE_GETENV("DENDRO_E2N_CANON_TN_DBG");
    const bool dbg_on =
        (dbg_env && dbg_env[0] == '1' && dbg_env[1] == '\0');

    size_t patched      = 0;
    size_t parent_local = 0;
    size_t parent_miss  = 0;
    size_t already_ok   = 0;
    size_t same_lev     = 0;

    for (unsigned int e = 0; e < eAllEnd; e++) {
        const unsigned int eLev = pNodes[e].getLevel();
        if (eLev == 0) continue;
        for (unsigned int n = 0; n < npe; n++) {
            const unsigned int slot = e * npe + n;
            const unsigned int dg = m_uiE2NMapping_DG[slot];
            const unsigned int oe = dg / npe;
            if (oe >= allElemSize) continue;
            const unsigned int oLev = pNodes[oe].getLevel();
            if (oLev == eLev) {
                same_lev++;
                continue;  // not hanging; handled by phys-pos audit
            }

            const ot::TreeNode parentTN = pNodes[e].getParent();
            auto it = tn2idx.find(parentTN);
            if (it == tn2idx.end()) {
                parent_miss++;
                continue;
            }
            parent_local++;
            const unsigned int parent_idx = it->second;
            const unsigned int parent_slot = parent_idx * npe + n;
            // already canonical?
            if (parent_slot == dg) {
                already_ok++;
                continue;
            }
            m_uiE2NMapping_DG[slot] = parent_slot;
            if (parent_slot < m_uiE2NMapping_CG.size()) {
                m_uiE2NMapping_CG[slot] =
                    m_uiE2NMapping_CG[parent_slot];
            }
            patched++;
        }
    }

    if (dbg_on
        || (patched > 0 && m_uiActiveRank == 0)) {
        std::cout << "[E2N canon-TN r" << m_uiActiveRank
                  << "] patched=" << patched
                  << " already_ok=" << already_ok
                  << " parent_local=" << parent_local
                  << " parent_miss=" << parent_miss
                  << " same_lev=" << same_lev
                  << " tn2idx_size=" << tn2idx.size()
                  << std::endl;
    }
    return patched;
}

size_t Mesh::auditAndRepairE2NCgPhysPos() {
    if (!m_uiIsActive) return 0;
    if (m_uiE2NMapping_CG.empty()) return 0;
    if (m_uiCG2DG.empty()) return 0;

    const unsigned int npe        = m_uiNpE;
    const unsigned int eOrd       = m_uiElementOrder;
    const unsigned int nLB        = m_uiNodeLocalBegin;
    const unsigned int nTotal_cg  = m_uiNumActualNodes;
    const ot::TreeNode* pNodes    = m_uiAllElements.data();
    const size_t allElemSize      = m_uiAllElements.size();

    auto computePos = [&](unsigned int e, unsigned int n,
                          PhysKey3& k) -> bool {
        if (e >= allElemSize) return false;
        const unsigned int lev = pNodes[e].getLevel();
        if (lev > m_uiMaxDepth) return false;
        const unsigned long long len =
            (unsigned long long)1 << (m_uiMaxDepth - lev);
        const unsigned int ni = n % (eOrd + 1);
        const unsigned int nj = (n / (eOrd + 1)) % (eOrd + 1);
        const unsigned int nk = n / ((eOrd + 1) * (eOrd + 1));
        k.x = (unsigned long long)pNodes[e].getX() * eOrd
              + (unsigned long long)ni * len;
        k.y = (unsigned long long)pNodes[e].getY() * eOrd
              + (unsigned long long)nj * len;
        k.z = (unsigned long long)pNodes[e].getZ() * eOrd
              + (unsigned long long)nk * len;
        return true;
    };

    static const char* dbg_env = DENDRO_PROBE_GETENV("DENDRO_E2N_AUDIT_DBG");
    const bool dbg_on =
        (dbg_env && dbg_env[0] == '1' && dbg_env[1] == '\0');

    // Phase 0: identify cgs with a canonical writer. A cg has a
    // canonical writer if some (e, n) has E2N_CG[e*npe+n]==cg AND
    // E2N_DG[e*npe+n]==e*npe+n (self-owned). Without one, zip skips
    // the slot and the cg holds whatever IC value was last set
    // (typically zero). E2N_CG entries that point to dangling cgs
    // are functionally dead — readers get stale data.
    const unsigned int eAllEnd =
        (unsigned int)std::min<size_t>(m_uiAllElements.size(),
                                       (size_t)m_uiNumTotalElements);
    std::vector<bool> hasCanonicalWriter(nTotal_cg, false);
    for (unsigned int e = 0; e < eAllEnd; e++) {
        for (unsigned int n = 0; n < npe; n++) {
            const unsigned int slot = e * npe + n;
            const unsigned int dg = m_uiE2NMapping_DG[slot];
            if (dg != slot) continue;  // not self-owned
            const unsigned int cg = m_uiE2NMapping_CG[slot];
            if (cg == LOOK_UP_TABLE_DEFAULT) continue;
            if (cg >= nTotal_cg) continue;
            hasCanonicalWriter[cg] = true;
        }
    }

    // Phase 1: build phys_pos → cg lookups. Two maps:
    //   localCgByPos: any cg at this phys_pos (LOCAL preferred over
    //                 ghost). Used for the existing same-level
    //                 wrong-phys patch path.
    //   canonCgByPos: cg WITH a canonical writer at this phys_pos
    //                 (LOCAL preferred over ghost). Used for the new
    //                 dangling-cg redirect path.
    std::unordered_map<PhysKey3, unsigned int, PhysKey3Hash> localCgByPos;
    std::unordered_map<PhysKey3, unsigned int, PhysKey3Hash> canonCgByPos;
    localCgByPos.reserve(nTotal_cg);
    canonCgByPos.reserve(nTotal_cg);
    const unsigned int nLE = m_uiNodeLocalEnd;
    // tie-break helper for choosing the canonical cg at a phys when
    // multiple cgs are at the same phys (e.g., shared boundaries between
    // adjacent same-level elements). The choice MUST be partition-
    // invariant; without this, the audit's choice depends on CG iteration
    // order which depends on the rank's local layout, producing different
    // canonical owners between modes (root cause of the long-haul drift
    // seed at step 14→15, see findings_2026-05-26).
    //
    // Rule: prefer LOCAL > GHOST; prefer with-canonical-writer > without;
    // tiebreak on the OWNER element TN (level asc, x/y/z asc) — Morton-
    // sortable, partition-invariant.
    auto better_canon = [&](unsigned int new_cg, unsigned int cur_cg) -> bool {
        const bool nLocal = (new_cg >= nLB && new_cg < nLE);
        const bool cLocal = (cur_cg >= nLB && cur_cg < nLE);
        if (nLocal != cLocal) return nLocal;  // prefer LOCAL
        const bool nCanon = hasCanonicalWriter[new_cg];
        const bool cCanon = hasCanonicalWriter[cur_cg];
        if (nCanon != cCanon) return nCanon;  // prefer with-canon-writer
        // tiebreak by owner TN: (level asc, x asc, y asc, z asc).
        const unsigned int n_dg = m_uiCG2DG[new_cg];
        const unsigned int c_dg = m_uiCG2DG[cur_cg];
        const unsigned int n_oe = n_dg / npe;
        const unsigned int c_oe = c_dg / npe;
        if (n_oe >= allElemSize || c_oe >= allElemSize) return false;
        const ot::TreeNode& nN = pNodes[n_oe];
        const ot::TreeNode& cN = pNodes[c_oe];
        if (nN.getLevel() != cN.getLevel())
            return nN.getLevel() < cN.getLevel();
        if (nN.getX() != cN.getX()) return nN.getX() < cN.getX();
        if (nN.getY() != cN.getY()) return nN.getY() < cN.getY();
        if (nN.getZ() != cN.getZ()) return nN.getZ() < cN.getZ();
        // same owner TN — tie-break on sub_n
        return (n_dg % npe) < (c_dg % npe);
    };

    // build localCgByPos / canonCgByPos by folding every cg with
    // better_canon (a strict total order: LOCAL>GHOST, with-canon>without,
    // tn-tiebreak). result is independent of cg iteration order.
    //
    // when DENDRO_OMP_PART + _OPENMP are both defined, each thread folds
    // its slice into a thread-local map, then a serial merge folds those
    // into the canonical maps. determinism comes from better_canon being
    // a strict total order — the max-of-set is well-defined regardless
    // of insertion order.
#if DENDRO_OMP_ACTIVE
    const int nThr = omp_get_max_threads();
#else
    const int nThr = 1;
#endif
    std::vector<std::unordered_map<PhysKey3, unsigned int, PhysKey3Hash>>
        tlLocal(nThr);
    std::vector<std::unordered_map<PhysKey3, unsigned int, PhysKey3Hash>>
        tlCanon(nThr);
    {
        const size_t guess = (size_t)((nTotal_cg - nLB) / nThr + 1);
        for (int t = 0; t < nThr; t++) {
            tlLocal[t].reserve(guess);
            tlCanon[t].reserve(guess);
        }
    }
    DENDRO_OMP_PRAGMA(omp parallel for schedule(static))
    for (unsigned int cg = nLB; cg < nTotal_cg; cg++) {
#if DENDRO_OMP_ACTIVE
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const unsigned int dg = m_uiCG2DG[cg];
        const unsigned int e  = dg / npe;
        const unsigned int n  = dg % npe;
        PhysKey3 k;
        if (!computePos(e, n, k)) continue;
        const bool newIsCanon = hasCanonicalWriter[cg];
        auto& mLocal = tlLocal[tid];
        auto itL = mLocal.find(k);
        if (itL == mLocal.end()) {
            mLocal.emplace(k, cg);
        } else if (better_canon(cg, itL->second)) {
            itL->second = cg;
        }
        if (newIsCanon) {
            auto& mCanon = tlCanon[tid];
            auto itC = mCanon.find(k);
            if (itC == mCanon.end()) {
                mCanon.emplace(k, cg);
            } else if (better_canon(cg, itC->second)) {
                itC->second = cg;
            }
        }
    }
    // serial merge — deterministic by t order + better_canon tiebreak.
    for (int t = 0; t < nThr; t++) {
        for (auto& kv : tlLocal[t]) {
            auto it = localCgByPos.find(kv.first);
            if (it == localCgByPos.end()) {
                localCgByPos.emplace(kv.first, kv.second);
            } else if (better_canon(kv.second, it->second)) {
                it->second = kv.second;
            }
        }
        for (auto& kv : tlCanon[t]) {
            auto it = canonCgByPos.find(kv.first);
            if (it == canonCgByPos.end()) {
                canonCgByPos.emplace(kv.first, kv.second);
            } else if (better_canon(kv.second, it->second)) {
                it->second = kv.second;
            }
        }
    }

    // Phase 2: walk ALL elements (locals + ghosts), audit each sub.
    // Ghost elements need auditing because their E2N_CG entries are
    // read by unzip on this rank (writing to block buffers). A ghost
    // elem with wrong cascade routing corrupts the block buffer at
    // shared corners. Walking ghost elements is safe because the
    // scatter-map rebuild that follows reads from this rank's E2N_CG
    // and naturally reflects the corrections.
    size_t patched           = 0;
    size_t bug_class         = 0;
    size_t legitimate_hang   = 0;
    size_t finer_owner_seen  = 0;
    size_t unresolved        = 0;
    size_t passd_skipped     = 0;
    size_t dangling_class    = 0;
    size_t dangling_patched  = 0;
    size_t dangling_unres    = 0;

    auto auditOnce = [&]() {
        size_t local_patched = 0;
        size_t local_bug     = 0;
        size_t local_hang    = 0;
        size_t local_finer   = 0;
        size_t local_unres   = 0;
        size_t local_passd   = 0;
        size_t local_dangling_class   = 0;
        size_t local_dangling_patched = 0;
        size_t local_dangling_unres   = 0;
        // race-safe under OMP: each (e, n) slot is written by exactly one
        // thread (the one owning that e), localCgByPos / canonCgByPos /
        // m_uiCG2DG are read-only in this phase, counters are reductions.
        DENDRO_OMP_PRAGMA(omp parallel for schedule(dynamic, 64)
            reduction(+ : local_patched, local_bug, local_hang, local_finer,
                          local_unres, local_passd, local_dangling_class,
                          local_dangling_patched, local_dangling_unres))
        for (unsigned int e = 0; e < eAllEnd; e++) {
            const unsigned int eLev = pNodes[e].getLevel();
            for (unsigned int n = 0; n < npe; n++) {
                const unsigned int slot = e * npe + n;
                const unsigned int cgIdx = m_uiE2NMapping_CG[slot];
                if (cgIdx == LOOK_UP_TABLE_DEFAULT) continue;
                if (cgIdx >= m_uiCG2DG.size()) continue;

                PhysKey3 expected;
                if (!computePos(e, n, expected)) continue;

                const unsigned int dg = m_uiCG2DG[cgIdx];
                const unsigned int oe = dg / npe;
                const unsigned int on = dg % npe;
                if (oe >= allElemSize) continue;

                PhysKey3 actual;
                if (!computePos(oe, on, actual)) continue;


                // Dangling-cg case: cg has no canonical writer at all.
                // Look up an alternative canonical cg at the OWNER's
                // phys_pos (where the routing currently points to) and
                // redirect. Works for both same-phys cases
                // (actual == expected, e.g. direct duplicate) and
                // hanging-edge cases (owner is coarser parent at a
                // different phys; we still want the canonical CG at
                // the parent's phys so the p2c interp reads the right
                // value). Skip cgs PassD owns (cross-rank consolidation
                // handles them separately).
                if (cgIdx < nTotal_cg && !hasCanonicalWriter[cgIdx] &&
                    m_uiPassDDemotedLocalCgs.count(cgIdx) == 0 &&
                    m_uiPassDDemotedToGhostCg.count(cgIdx) == 0) {
                    local_dangling_class++;
                    auto itC = canonCgByPos.find(actual);
                    // expected-phys fallback for misrouted non-hanging
                    // slots: graph E2N can route a non-hanging interior
                    // slot to a dangling corner cg (e.g. +X+Y edge slot
                    // sharing the +X+Y+Z corner's cg). owner phys then
                    // has no canonical alternative, but the slot's
                    // expected phys typically does — and even if that
                    // cg is itself dangling, its value is populated by
                    // post-axpy sync / passD / ghost exchange. gated by
                    // !isNodeHanging because hanging-edge p2c reads
                    // must keep routing to the parent's phys for the
                    // p2c kernel inside getElementNodalValues.
                    if (itC == canonCgByPos.end()) {
                        const unsigned int ni = n % (eOrd + 1);
                        const unsigned int nj =
                            (n / (eOrd + 1)) % (eOrd + 1);
                        const unsigned int nk =
                            n / ((eOrd + 1) * (eOrd + 1));
                        if (!this->isNodeHanging(e, ni, nj, nk)) {
                            auto itExp = canonCgByPos.find(expected);
                            if (itExp != canonCgByPos.end()) {
                                itC = itExp;
                            } else {
                                auto itExpL = localCgByPos.find(expected);
                                if (itExpL != localCgByPos.end()) {
                                    itC = itExpL;
                                }
                            }
                        }
                    }
                    if (itC == canonCgByPos.end()) {
                        local_dangling_unres++;
                        if (dbg_on) {
                            std::cout << "[E2N audit r"
                                      << m_uiActiveRank
                                      << "] DANGLING-UNRES elem=" << e
                                      << " sub=" << n
                                      << " cgIdx=" << cgIdx
                                      << " ownerPhys=("
                                      << actual.x << "," << actual.y
                                      << "," << actual.z
                                      << ") expectedPhys=("
                                      << expected.x << "," << expected.y
                                      << "," << expected.z
                                      << ") (no canonical cg at phys)"
                                      << std::endl;
                        }
                        continue;
                    }
                    const unsigned int newCg = itC->second;
                    if (newCg == cgIdx) continue;
                    if (dbg_on) {
                        std::cout << "[E2N audit r"
                                  << m_uiActiveRank
                                  << "] DANGLING-PATCH elem=" << e
                                  << " sub=" << n
                                  << " elemTN=(" << pNodes[e].getX()
                                  << "," << pNodes[e].getY()
                                  << "," << pNodes[e].getZ()
                                  << ",lev" << eLev
                                  << ") expectedPhys=(" << expected.x
                                  << "," << expected.y
                                  << "," << expected.z
                                  << ") ownerPhys=(" << actual.x
                                  << "," << actual.y
                                  << "," << actual.z
                                  << ") oldCg=" << cgIdx
                                  << " newCg=" << newCg
                                  << std::endl;
                    }
                    m_uiE2NMapping_CG[slot] = newCg;
                    if (newCg < m_uiCG2DG.size()) {
                        const unsigned int newDg = m_uiCG2DG[newCg];
                        if (slot < m_uiE2NMapping_DG.size()) {
                            m_uiE2NMapping_DG[slot] = newDg;
                        }
                    }
                    local_dangling_patched++;
                    continue;
                }

                if (actual == expected) continue;

                const unsigned int oLev = pNodes[oe].getLevel();

                if (oLev < eLev) {
                    // Hanging-face/edge p2c: child sub routes to
                    // coarser parent's CG. The routing is "legitimate"
                    // structurally, but graph and SFC can pick DIFFERENT
                    // lev<eLev parents sharing the cg position, giving
                    // partition-dependent canonical-owner selection.
                    // Canonicalize: redirect to canonical cg at the
                    // MORTON-TREE PARENT's grid point at the child sub
                    // indices (using parent spacing). This makes
                    // routing partition-invariant.
                    // see docs/findings_2026-05-14d.md follow-up.
                    local_hang++;
                    // default-OFF as of 2026-05-28: redundant for both
                    // EM4 and NLSM given orphan-fill fix + Fix B +
                    // post-axpy sync. DENDRO_E2N_HANG_CANONICALIZE=1
                    // re-enables for A/B.
                    static const char* hcEnv =
                        std::getenv("DENDRO_E2N_HANG_CANONICALIZE");
                    static const bool hcOn =
                        hcEnv && hcEnv[0] == '1' && hcEnv[1] == '\0';
                    if (!hcOn) continue;
                    if (eLev == 0) continue;
                    const unsigned long long childSize =
                        1ULL << (m_uiMaxDepth - eLev);
                    const unsigned long long parentSize = childSize * 2;
                    const unsigned long long pmask = parentSize - 1;
                    const unsigned long long parentOrigX =
                        (unsigned long long)pNodes[e].getX() & ~pmask;
                    const unsigned long long parentOrigY =
                        (unsigned long long)pNodes[e].getY() & ~pmask;
                    const unsigned long long parentOrigZ =
                        (unsigned long long)pNodes[e].getZ() & ~pmask;
                    const unsigned int ni = n % (eOrd + 1);
                    const unsigned int nj =
                        (n / (eOrd + 1)) % (eOrd + 1);
                    const unsigned int nk =
                        n / ((eOrd + 1) * (eOrd + 1));
                    PhysKey3 parentExpected;
                    parentExpected.x = parentOrigX * eOrd
                        + (unsigned long long)ni * parentSize;
                    parentExpected.y = parentOrigY * eOrd
                        + (unsigned long long)nj * parentSize;
                    parentExpected.z = parentOrigZ * eOrd
                        + (unsigned long long)nk * parentSize;
                    if (parentExpected == actual) continue;  // already canonical
                    // try canonical (has canonical writer) first, fall
                    // back to localCgByPos (any cg incl ghost).
                    unsigned int newCg = LOOK_UP_TABLE_DEFAULT;
                    auto itP = canonCgByPos.find(parentExpected);
                    if (itP != canonCgByPos.end()) {
                        newCg = itP->second;
                    } else {
                        auto itL = localCgByPos.find(parentExpected);
                        if (itL != localCgByPos.end()) {
                            newCg = itL->second;
                        }
                    }
                    if (newCg == LOOK_UP_TABLE_DEFAULT) {
                        if (dbg_on) {
                            std::cout << "[E2N audit r" << m_uiActiveRank
                                      << "] HANG-CANONICALIZE-UNRES"
                                      << " elem=" << e
                                      << " sub=" << n
                                      << " elemTN=(" << pNodes[e].getX()
                                      << "," << pNodes[e].getY()
                                      << "," << pNodes[e].getZ()
                                      << ",lev" << eLev
                                      << ") parentExpectedPhys=("
                                      << parentExpected.x << ","
                                      << parentExpected.y << ","
                                      << parentExpected.z
                                      << ") actualPhys=("
                                      << actual.x << "," << actual.y
                                      << "," << actual.z
                                      << ") cgIdx=" << cgIdx
                                      << " ownerLev=" << oLev
                                      << std::endl;
                        }
                        continue;
                    }
                    if (newCg == cgIdx) continue;
                    if (dbg_on) {
                        std::cout << "[E2N audit r" << m_uiActiveRank
                                  << "] HANG-CANONICALIZE-PATCH"
                                  << " elem=" << e
                                  << " sub=" << n
                                  << " elemTN=(" << pNodes[e].getX()
                                  << "," << pNodes[e].getY()
                                  << "," << pNodes[e].getZ()
                                  << ",lev" << eLev
                                  << ") parentExpectedPhys=("
                                  << parentExpected.x << ","
                                  << parentExpected.y << ","
                                  << parentExpected.z
                                  << ") oldActualPhys=("
                                  << actual.x << "," << actual.y
                                  << "," << actual.z
                                  << ") oldCg=" << cgIdx
                                  << " newCg=" << newCg
                                  << std::endl;
                    }
                    m_uiE2NMapping_CG[slot] = newCg;
                    if (newCg < m_uiCG2DG.size()) {
                        const unsigned int newDg = m_uiCG2DG[newCg];
                        if (slot < m_uiE2NMapping_DG.size()) {
                            m_uiE2NMapping_DG[slot] = newDg;
                        }
                    }
                    local_patched++;
                    continue;
                }
                if (oLev > eLev) {
                    // Owner finer than this element — unexpected. This
                    // happens when E2N_CG routing for a hanging-face
                    // child node points to a corrupted (often finer)
                    // cg in graph mode. The CORRECT routing for a
                    // hanging-face child node at child-sub indices
                    // (ni, nj, nk) is to the parent's grid point at
                    // the SAME indices but with PARENT spacing.
                    // see docs/findings_2026-05-14d.md follow-up.
                    local_finer++;
                    // default-OFF as of 2026-05-28 (see hcOn above).
                    static const char* fnEnv =
                        std::getenv("DENDRO_E2N_HANG_CANONICALIZE");
                    static const bool fnOn =
                        fnEnv && fnEnv[0] == '1' && fnEnv[1] == '\0';
                    if (!fnOn) continue;
                    if (eLev == 0) continue;  // no parent
                    const unsigned long long childSize =
                        1ULL << (m_uiMaxDepth - eLev);
                    const unsigned long long parentSize = childSize * 2;
                    const unsigned long long pmask = parentSize - 1;
                    const unsigned long long parentOrigX =
                        (unsigned long long)pNodes[e].getX() & ~pmask;
                    const unsigned long long parentOrigY =
                        (unsigned long long)pNodes[e].getY() & ~pmask;
                    const unsigned long long parentOrigZ =
                        (unsigned long long)pNodes[e].getZ() & ~pmask;
                    const unsigned int ni = n % (eOrd + 1);
                    const unsigned int nj =
                        (n / (eOrd + 1)) % (eOrd + 1);
                    const unsigned int nk =
                        n / ((eOrd + 1) * (eOrd + 1));
                    PhysKey3 parentExpected;
                    parentExpected.x = parentOrigX * eOrd
                        + (unsigned long long)ni * parentSize;
                    parentExpected.y = parentOrigY * eOrd
                        + (unsigned long long)nj * parentSize;
                    parentExpected.z = parentOrigZ * eOrd
                        + (unsigned long long)nk * parentSize;
                    auto itP = canonCgByPos.find(parentExpected);
                    if (itP == canonCgByPos.end()) {
                        if (dbg_on) {
                            std::cout << "[E2N audit r" << m_uiActiveRank
                                      << "] FINER-OWNER-UNRES elem=" << e
                                      << " sub=" << n
                                      << " elemTN=(" << pNodes[e].getX()
                                      << "," << pNodes[e].getY()
                                      << "," << pNodes[e].getZ()
                                      << ",lev" << eLev
                                      << ") expectedChildPhys=("
                                      << expected.x << "," << expected.y
                                      << "," << expected.z
                                      << ") expectedParentPhys=("
                                      << parentExpected.x << ","
                                      << parentExpected.y << ","
                                      << parentExpected.z
                                      << ") actualPhys=("
                                      << actual.x << "," << actual.y
                                      << "," << actual.z
                                      << ") cgIdx=" << cgIdx
                                      << " ownerLev=" << oLev
                                      << std::endl;
                        }
                        continue;
                    }
                    const unsigned int newCg = itP->second;
                    if (newCg == cgIdx) continue;
                    if (dbg_on) {
                        std::cout << "[E2N audit r" << m_uiActiveRank
                                  << "] FINER-OWNER-PATCH elem=" << e
                                  << " sub=" << n
                                  << " elemTN=(" << pNodes[e].getX()
                                  << "," << pNodes[e].getY()
                                  << "," << pNodes[e].getZ()
                                  << ",lev" << eLev
                                  << ") parentPhys=("
                                  << parentExpected.x << ","
                                  << parentExpected.y << ","
                                  << parentExpected.z
                                  << ") oldCg=" << cgIdx
                                  << " ownerLev=" << oLev
                                  << " newCg=" << newCg
                                  << std::endl;
                    }
                    m_uiE2NMapping_CG[slot] = newCg;
                    if (newCg < m_uiCG2DG.size()) {
                        const unsigned int newDg = m_uiCG2DG[newCg];
                        if (slot < m_uiE2NMapping_DG.size()) {
                            m_uiE2NMapping_DG[slot] = newDg;
                        }
                    }
                    local_patched++;
                    continue;
                }

                // Same level + phys_pos mismatch == bug class.
                local_bug++;

                if (m_uiPassDDemotedLocalCgs.count(cgIdx) ||
                    m_uiPassDDemotedToGhostCg.count(cgIdx)) {
                    local_passd++;
                    continue;
                }

                auto it = localCgByPos.find(expected);
                if (it == localCgByPos.end()) {
                    local_unres++;
                    if (dbg_on) {
                        std::cout << "[E2N audit r" << m_uiActiveRank
                                  << "] UNRESOLVED elem=" << e
                                  << " sub=" << n
                                  << " elemTN=(" << pNodes[e].getX()
                                  << "," << pNodes[e].getY()
                                  << "," << pNodes[e].getZ()
                                  << ",lev" << eLev
                                  << ") expectedPhys=("
                                  << expected.x << "," << expected.y
                                  << "," << expected.z
                                  << ") actualPhys=("
                                  << actual.x << "," << actual.y
                                  << "," << actual.z
                                  << ") cgIdx=" << cgIdx
                                  << " ownerElem=" << oe
                                  << std::endl;
                    }
                    continue;
                }

                const unsigned int newCg = it->second;
                if (newCg == cgIdx) continue;

                if (dbg_on) {
                    std::cout << "[E2N audit r" << m_uiActiveRank
                              << "] PATCH elem=" << e
                              << " sub=" << n
                              << " elemTN=(" << pNodes[e].getX()
                              << "," << pNodes[e].getY()
                              << "," << pNodes[e].getZ()
                              << ",lev" << eLev
                              << ") expectedPhys=("
                              << expected.x << "," << expected.y
                              << "," << expected.z
                              << ") oldCg=" << cgIdx
                              << " ownerOld=" << oe
                              << " newCg=" << newCg
                              << std::endl;
                }

                m_uiE2NMapping_CG[slot] = newCg;
                // Also update E2N_DG so REBUILD NODAL SCATTER MAPS
                // and any downstream DG-based reads pick up the new
                // canonical owner. Pattern matches patchE2NCgFromMasks
                // at mesh.cpp:6010-6015.
                if (newCg < m_uiCG2DG.size()) {
                    const unsigned int newDg = m_uiCG2DG[newCg];
                    if (slot < m_uiE2NMapping_DG.size()) {
                        m_uiE2NMapping_DG[slot] = newDg;
                    }
                }
                local_patched++;
            }
        }
        bug_class       += local_bug;
        legitimate_hang += local_hang;
        finer_owner_seen+= local_finer;
        unresolved      += local_unres;
        passd_skipped   += local_passd;
        patched         += local_patched;
        dangling_class  += local_dangling_class;
        dangling_patched+= local_dangling_patched;
        dangling_unres  += local_dangling_unres;
        return local_patched + local_dangling_patched;
    };

    // Up to 3 passes; chains of wrong routings can require iteration
    // because patching A → A' may reveal that B (which A used to
    // alias to) is also wrong.
    size_t lastPass = auditOnce();
    int passes      = 1;
    while (lastPass > 0 && passes < 3) {
        lastPass = auditOnce();
        passes++;
    }

    if (dbg_on
        || ((patched + dangling_patched) > 0 && m_uiActiveRank == 0)) {
        std::cout << "[E2N audit r" << m_uiActiveRank
                  << "] passes=" << passes
                  << " patched=" << patched
                  << " bug_class=" << bug_class
                  << " hanging=" << legitimate_hang
                  << " finer_owner=" << finer_owner_seen
                  << " unresolved=" << unresolved
                  << " passd_skipped=" << passd_skipped
                  << " dangling_class=" << dangling_class
                  << " dangling_patched=" << dangling_patched
                  << " dangling_unres=" << dangling_unres
                  << std::endl;
    }

    return patched + dangling_patched;
}

void Mesh::buildZipPlan() {
    m_uiZipPlanCg.clear();
    m_uiZipPlanUnzipIdx.clear();
    if (!m_uiIsActive) return;
    if (!m_uiIsBlockSetup) return;
    if (m_uiLocalBlockList.empty()) return;

    const unsigned int npe  = m_uiNpE;
    const unsigned int eOrd = m_uiElementOrder;
    const unsigned int nLB  = m_uiNodeLocalBegin;
    const unsigned int nLE  = m_uiNodeLocalEnd;
    const ot::TreeNode* pNodes = m_uiAllElements.data();

    // Reserve a generous starting size: typical write count is ~one
    // per local cg.
    const size_t reserveN = (nLE > nLB) ? (nLE - nLB) : 0;
    m_uiZipPlanCg.reserve(reserveN);
    m_uiZipPlanUnzipIdx.reserve(reserveN);

    // Env gate: DENDRO_USE_LEGACY_PLAN_BUILD=1 falls back to the
    // cascade-self-owned-scan for plan construction (Stage 1
    // behavior). Default uses the explicit primary-pick path below
    // (Stage 2): plan ownership is determined by allgather +
    // smallest-packTN rule, not by E2N_DG self-ownership. This makes
    // the plan independent of Pass A/D/E's E2N_DG rewrites.
    static const char* legacy_plan_env =
        std::getenv("DENDRO_USE_LEGACY_PLAN_BUILD");
    static const bool use_legacy_plan = legacy_plan_env
        && legacy_plan_env[0] == '1' && legacy_plan_env[1] == '\0';

    if (use_legacy_plan || m_uiActiveNpes <= 1) {
        // Stage 1 path: cascade-self-owned scan. Used when the env
        // gate is set, or in the single-rank case where no allgather
        // is needed.
        for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
            const auto& block        = m_uiLocalBlockList[blk];
            const ot::TreeNode blkNode = block.getBlockNode();
            const unsigned int regLev  = block.getRegularGridLev();
            const unsigned int lx      = block.getAllocationSzX();
            const unsigned int ly      = block.getAllocationSzY();
            const unsigned int offset    = block.getOffset();
            const unsigned int paddWidth = block.get1DPadWidth();
            for (unsigned int elem : m_uiLocalBlockList[blk]) {
                const unsigned int ei =
                    (pNodes[elem].getX() - blkNode.getX()) >>
                    (m_uiMaxDepth - regLev);
                const unsigned int ej =
                    (pNodes[elem].getY() - blkNode.getY()) >>
                    (m_uiMaxDepth - regLev);
                const unsigned int ek =
                    (pNodes[elem].getZ() - blkNode.getZ()) >>
                    (m_uiMaxDepth - regLev);
                for (unsigned int k = 0; k < eOrd + 1; k++)
                    for (unsigned int j = 0; j < eOrd + 1; j++)
                        for (unsigned int i = 0; i < eOrd + 1; i++) {
                            const unsigned int sub =
                                k * (eOrd + 1) * (eOrd + 1)
                                + j * (eOrd + 1) + i;
                            const unsigned int dg =
                                m_uiE2NMapping_DG[elem * npe + sub];
                            if (dg / npe != elem) continue;
                            const unsigned int target_cg =
                                m_uiE2NMapping_CG[elem * npe + sub];
                            const unsigned int unzip_idx =
                                offset
                                + (ek * eOrd + k + paddWidth) * (ly * lx)
                                + (ej * eOrd + j + paddWidth) * lx
                                + (ei * eOrd + i + paddWidth);
                            m_uiZipPlanCg.push_back(target_cg);
                            m_uiZipPlanUnzipIdx.push_back(unzip_idx);
                        }
            }
        }
        return;
    }

    // ---- Stage 2: explicit primary-pick plan build ----
    //
    // Per phys_pos in the global mesh, exactly one rank should write
    // the value via zip. Ownership is decided by smallest-packTN
    // rule (matches current Pass D's primary pick): the rank holding
    // the local elem with the smallest TreeNode at this phys_pos
    // wins. Tiebreak by smallest rank.
    //
    // For each phys_pos where THIS rank is the primary, we add a
    // plan entry: (target_cg = local cg at this phys_pos that we
    // want to write into, writer_elem, writer_sub).
    //
    // Default ownership rule: smallest-packTN-lex on (level, X, Y).
    // This matches what Pass D currently uses and gives a proven
    // 5× improvement on EM4 np=4 vs no rescue (1.157e-7 → 2.2e-8).
    //
    // Opt-in DENDRO_USE_CASCADE_RULE=1 switches to dendro's true
    // cascade rule (smallest level then ot::TreeNode::operator< /
    // Hilbert NCA, see meshE2NUtils.tcc CORNER_NODE_MAP). Cascade
    // rule matches SFC's natural canonical pick at corners — but
    // currently regresses because the standard scatter maps were
    // built from the pre-Pass-D cascade and don't include the
    // (primary_rank, primary_cg) → ghost path needed for the new
    // sync to find the right ghost cg. Future work: rebuild scatter
    // maps after primary pick to enable bit-perfect cascade-rule.
    static const char* cascade_env =
        std::getenv("DENDRO_USE_CASCADE_RULE");
    static const bool use_packtn =
        !(cascade_env && cascade_env[0] == '1' && cascade_env[1] == '\0');
    auto packTN = [&](unsigned int e) -> unsigned long long {
        unsigned long long lev = (unsigned long long)pNodes[e].getLevel()
                                 & 0xFFULL;
        unsigned long long X = (unsigned long long)pNodes[e].getX()
                               & 0xFFFFFFFULL;
        unsigned long long Y = (unsigned long long)pNodes[e].getY()
                               & 0xFFFFFFFULL;
        return (lev << 56) | (X << 28) | Y;
    };
    const unsigned long long PACK_INF = ~0ULL;

    auto encodeKey = [&](unsigned int e, unsigned int n,
                         unsigned long long& x,
                         unsigned long long& y,
                         unsigned long long& z) {
        unsigned long long len =
            (unsigned long long)1 << (m_uiMaxDepth - pNodes[e].getLevel());
        unsigned int ni = n % (eOrd + 1);
        unsigned int nj = (n / (eOrd + 1)) % (eOrd + 1);
        unsigned int nk = n / ((eOrd + 1) * (eOrd + 1));
        x = (unsigned long long)pNodes[e].getX() * eOrd
            + (unsigned long long)ni * len;
        y = (unsigned long long)pNodes[e].getY() * eOrd
            + (unsigned long long)nj * len;
        z = (unsigned long long)pNodes[e].getZ() * eOrd
            + (unsigned long long)nk * len;
    };

    // Cascade-rule TreeNode comparator. Returns true iff a < b
    // under (level, ot::TreeNode::operator<). With use_packtn,
    // falls back to the lexicographic packTN ordering.
    auto tnLT = [&](unsigned int ea, unsigned int eb) -> bool {
        if (use_packtn) return packTN(ea) < packTN(eb);
        const unsigned int la = pNodes[ea].getLevel();
        const unsigned int lb = pNodes[eb].getLevel();
        if (la != lb) return la < lb;
        return pNodes[ea] < pNodes[eb];
    };

    // Build phys_pos -> (smallest-TreeNode-local-elem, sub) on this
    // rank under the cascade rule. We track elem and sub so the
    // primary writer is directly available without a second walk.
    struct LocalWriter {
        unsigned int elem;
        unsigned int sub;
    };
    // Helper: does (elem, sub) on this rank land in BLOCK INTERIOR
    // (i.e., the unzip buffer position for this (elem, sub) is in the
    // [PW, lx-1-PW] interior region where solverRHS actually writes)?
    // Plan-zip reads this position and writes to the target cg. If
    // it's in PADDING, the position holds 0 (uninitialized) or a
    // stale value, so we'd zip 0 into the canonical cg — corrupting
    // it across the network via the side-channel sync.
    //
    // SFC meshes don't trip this because the SFC partition tends to
    // place each phys_pos's canonical-owner element in a block where
    // the canonical sub lands in interior. Graph partitioning at
    // level-transition corners shuffles ownership such that the
    // smallest-TN local elem (myMin) often has its corner sub in
    // PADDING of its own block — yielding a zero write that then
    // propagates to all loser ranks via the side-channel sync.
    auto isInteriorWriter = [&](unsigned int e, unsigned int n) -> bool {
        if (e < m_uiElementLocalBegin || e >= m_uiElementLocalEnd)
            return false;
        if ((e - m_uiElementLocalBegin) >= m_uiE2BlkMap.size())
            return false;
        const unsigned int blk =
            m_uiE2BlkMap[e - m_uiElementLocalBegin];
        if (blk == LOOK_UP_TABLE_DEFAULT
            || blk >= m_uiLocalBlockList.size()) return false;
        const auto& block         = m_uiLocalBlockList[blk];
        const ot::TreeNode blkNode = block.getBlockNode();
        const unsigned int regLev  = block.getRegularGridLev();
        const unsigned int lx      = block.getAllocationSzX();
        const unsigned int ly      = block.getAllocationSzY();
        const unsigned int lz      = block.getAllocationSzZ();
        const unsigned int PW      = block.get1DPadWidth();
        const unsigned int ei =
            (pNodes[e].getX() - blkNode.getX()) >>
            (m_uiMaxDepth - regLev);
        const unsigned int ej =
            (pNodes[e].getY() - blkNode.getY()) >>
            (m_uiMaxDepth - regLev);
        const unsigned int ek =
            (pNodes[e].getZ() - blkNode.getZ()) >>
            (m_uiMaxDepth - regLev);
        const unsigned int ni  = n % (eOrd + 1);
        const unsigned int nj  = (n / (eOrd + 1)) % (eOrd + 1);
        const unsigned int nk  = n / ((eOrd + 1) * (eOrd + 1));
        const unsigned int bi  = ei * eOrd + ni + PW;
        const unsigned int bj  = ej * eOrd + nj + PW;
        const unsigned int bk  = ek * eOrd + nk + PW;
        return (bi >= PW && bi <= (lx - 1 - PW))
            && (bj >= PW && bj <= (ly - 1 - PW))
            && (bk >= PW && bk <= (lz - 1 - PW));
    };

    // Cascade-owner check: is THIS (e, n) the canonical owner of the
    // cg at this phys per the cascade (E2N_DG)? Partition-invariant
    // after auditAndRepairE2NCgPhysPos. Preferring cascade-owner over
    // smallest-TN-local breaks the partition-dependence we hit on
    // EM4 step-1 substage-0 (see findings_2026-05-14g.md): in graph
    // mode r2 had only the cascade owner local, in SFC r2 had ALSO a
    // smaller TN local, so the old smallest-TN-local rule picked
    // different writers between modes and the two writers' blocks
    // produced 1-ULP different rhs at the shared face cell.
    auto isCascadeOwner = [&](unsigned int e, unsigned int n) -> bool {
        return m_uiE2NMapping_DG[e * npe + n] / npe == e;
    };
    // Optional gate: DENDRO_ZIPPLAN_USE_SMALLEST_TN=1 reverts to the
    // legacy smallest-TN-local pick (no cascade-owner preference).
    // Useful for A/B testing.
    static const char* legacy_pick_env =
        std::getenv("DENDRO_ZIPPLAN_USE_SMALLEST_TN");
    static const bool legacy_pick =
        legacy_pick_env && legacy_pick_env[0] == '1'
        && legacy_pick_env[1] == '\0';

    std::unordered_map<PhysKey3, LocalWriter, PhysKey3Hash> myMin;
    // single-source preference rule: interior > padding, cascade-owner >
    // non-owner, smaller TN < larger TN. strict total order in practice,
    // so max-of-set is well-defined regardless of insertion order — safe
    // to parallelize via thread-local maps + serial merge under OMP.
    auto tryReplaceMyMin = [&](
        std::unordered_map<PhysKey3, LocalWriter, PhysKey3Hash>& m,
        const PhysKey3& k, unsigned int e, unsigned int n,
        bool interior, bool is_owner) {
        auto it = m.find(k);
        if (it == m.end()) {
            m[k] = LocalWriter{e, n};
            return;
        }
        const bool cur_interior =
            isInteriorWriter(it->second.elem, it->second.sub);
        const bool cur_is_owner =
            !legacy_pick
            && isCascadeOwner(it->second.elem, it->second.sub);
        if (interior && !cur_interior) {
            m[k] = LocalWriter{e, n};
        } else if (interior == cur_interior) {
            if (is_owner && !cur_is_owner) {
                m[k] = LocalWriter{e, n};
            } else if (is_owner == cur_is_owner) {
                if (tnLT(e, it->second.elem))
                    m[k] = LocalWriter{e, n};
            }
        }
    };

#if DENDRO_OMP_ACTIVE
    const int nThr_mm = omp_get_max_threads();
#else
    const int nThr_mm = 1;
#endif
    std::vector<std::unordered_map<PhysKey3, LocalWriter, PhysKey3Hash>>
        tlMyMin(nThr_mm);
    {
        const size_t numElem =
            (size_t)(m_uiElementLocalEnd - m_uiElementLocalBegin);
        const size_t guess = (numElem * npe) / nThr_mm + 1;
        for (int t = 0; t < nThr_mm; t++) tlMyMin[t].reserve(guess);
    }
    DENDRO_OMP_PRAGMA(omp parallel for schedule(static))
    for (unsigned int e = m_uiElementLocalBegin;
         e < m_uiElementLocalEnd; e++) {
#if DENDRO_OMP_ACTIVE
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& m = tlMyMin[tid];
        for (unsigned int n = 0; n < npe; n++) {
            unsigned long long x, y, z;
            encodeKey(e, n, x, y, z);
            PhysKey3 k{x, y, z};
            const bool interior = isInteriorWriter(e, n);
            const bool is_owner =
                !legacy_pick && isCascadeOwner(e, n);
            tryReplaceMyMin(m, k, e, n, interior, is_owner);
        }
    }
    // serial merge: deterministic by t order + preference rule.
    for (int t = 0; t < nThr_mm; t++) {
        for (auto& kv : tlMyMin[t]) {
            const bool interior =
                isInteriorWriter(kv.second.elem, kv.second.sub);
            const bool is_owner =
                !legacy_pick
                && isCascadeOwner(kv.second.elem, kv.second.sub);
            tryReplaceMyMin(myMin, kv.first,
                            kv.second.elem, kv.second.sub,
                            interior, is_owner);
        }
    }

    // For each LOCAL cg, find its phys_pos via cg2dg. We advertise
    // (phys_pos, my_local_cg, my smallest-TreeNode local elem here:
    //  X, Y, Z, level). After allgather, the cascade-rule winner per
    // phys_pos wins.
    const unsigned int TN_LEV_NONE = (unsigned int)-1;
    std::vector<unsigned long long> myX, myY, myZ;
    std::vector<unsigned int> myTNX, myTNY, myTNZ, myTNLev;
    std::vector<unsigned int> myCg;
    // myInterior[i] = 1 if my advertised writer for this phys_pos is in
    // BLOCK INTERIOR (i.e. solverRHS will fill that buffer position), 0
    // if it's in PADDING (zip would read 0/stale). Used in the cross-
    // rank winner pick to prefer interior writers globally.
    std::vector<unsigned int> myInterior;
    myX.reserve(reserveN);
    myY.reserve(reserveN);
    myZ.reserve(reserveN);
    myTNX.reserve(reserveN);
    myTNY.reserve(reserveN);
    myTNZ.reserve(reserveN);
    myTNLev.reserve(reserveN);
    myCg.reserve(reserveN);
    myInterior.reserve(reserveN);
    const size_t nElTot = m_uiAllElements.size();
    // Advertise ALL cgs on this rank (local + ghost) at every phys_pos
    // they hold. Including ghost cgs lets the cross-rank syncZipNonPrimary
    // recv list cover ghost destinations, not just non-primary local
    // destinations. Without ghost coverage, ranks that have only ghost
    // cgs at a level-transition phys_pos (i.e. no local elem there)
    // never receive the primary's post-axpy value via sync — and unzip
    // on those ranks reads stale IC from those ghosts.
    const unsigned int nTotal_cg = m_uiNumActualNodes;
    for (unsigned int cg = 0; cg < nTotal_cg; cg++) {
        if (cg >= m_uiCG2DG.size()) continue;
        unsigned int dg = m_uiCG2DG[cg];
        if (dg == LOOK_UP_TABLE_DEFAULT) continue;
        unsigned int e = dg / npe;
        unsigned int n = dg % npe;
        if (e >= nElTot) continue;
        unsigned long long x, y, z;
        encodeKey(e, n, x, y, z);
        unsigned int tnX = 0, tnY = 0, tnZ = 0, tnLev = TN_LEV_NONE;
        unsigned int interior = 0u;
        // Only LOCAL cgs claim a tnLev != TN_LEV_NONE (cascade winner
        // pick uses myMin, which is built from local elements only).
        // Ghost cgs are added with TN_LEV_NONE so they NEVER win the
        // cross-rank cascade comparison — they only serve as recv
        // destinations for the primary's value.
        const bool is_local = (cg >= nLB && cg < nLE);
        if (is_local) {
            auto it = myMin.find(PhysKey3{x, y, z});
            if (it != myMin.end()) {
                unsigned int we = it->second.elem;
                unsigned int wn = it->second.sub;
                tnX   = pNodes[we].getX();
                tnY   = pNodes[we].getY();
                tnZ   = pNodes[we].getZ();
                tnLev = pNodes[we].getLevel();
                interior = isInteriorWriter(we, wn) ? 1u : 0u;
            }
        }
        myX.push_back(x);
        myY.push_back(y);
        myZ.push_back(z);
        myTNX.push_back(tnX);
        myTNY.push_back(tnY);
        myTNZ.push_back(tnZ);
        myTNLev.push_back(tnLev);
        myCg.push_back(cg);
        myInterior.push_back(interior);
    }


    int myCount = (int)myX.size();
    std::vector<int> counts(m_uiActiveNpes), offs(m_uiActiveNpes, 0);
    MPI_Allgather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  m_uiCommActive);
    int total = 0;
    for (int p = 0; p < m_uiActiveNpes; p++) {
        offs[p] = total;
        total += counts[p];
    }
    std::vector<unsigned long long> allX(total), allY(total), allZ(total);
    std::vector<unsigned int> allTNX(total), allTNY(total),
        allTNZ(total), allTNLev(total), allCg(total);
    MPI_Allgatherv(myX.data(), myCount, MPI_UINT64_T, allX.data(),
                   counts.data(), offs.data(), MPI_UINT64_T,
                   m_uiCommActive);
    MPI_Allgatherv(myY.data(), myCount, MPI_UINT64_T, allY.data(),
                   counts.data(), offs.data(), MPI_UINT64_T,
                   m_uiCommActive);
    MPI_Allgatherv(myZ.data(), myCount, MPI_UINT64_T, allZ.data(),
                   counts.data(), offs.data(), MPI_UINT64_T,
                   m_uiCommActive);
    auto agU = [&](std::vector<unsigned int>& src,
                   std::vector<unsigned int>& dst) {
        MPI_Allgatherv(src.data(), myCount, MPI_UNSIGNED, dst.data(),
                       counts.data(), offs.data(), MPI_UNSIGNED,
                       m_uiCommActive);
    };
    agU(myTNX, allTNX);
    agU(myTNY, allTNY);
    agU(myTNZ, allTNZ);
    agU(myTNLev, allTNLev);
    agU(myCg, allCg);
    std::vector<unsigned int> allInterior(total);
    agU(myInterior, allInterior);

    // Cascade-rule cross-rank comparator. claim "a wins over b"
    // means a is strictly smaller under (level, ot::TreeNode::<).
    // claim with TN_LEV_NONE never wins (that rank advertised no
    // local elem at this phys_pos).
    auto claimLT = [&](unsigned int aX, unsigned int aY,
                       unsigned int aZ, unsigned int aL,
                       unsigned int bX, unsigned int bY,
                       unsigned int bZ, unsigned int bL) -> bool {
        if (aL == TN_LEV_NONE) return false;
        if (bL == TN_LEV_NONE) return true;
        if (use_packtn) {
            unsigned long long pa =
                ((unsigned long long)(aL & 0xFFULL) << 56)
                | ((unsigned long long)(aX & 0xFFFFFFFULL) << 28)
                | (unsigned long long)(aY & 0xFFFFFFFULL);
            unsigned long long pb =
                ((unsigned long long)(bL & 0xFFULL) << 56)
                | ((unsigned long long)(bX & 0xFFFFFFFULL) << 28)
                | (unsigned long long)(bY & 0xFFFFFFFULL);
            return pa < pb;
        }
        if (aL != bL) return aL < bL;
        ot::TreeNode ta(aX, aY, aZ, aL, m_uiDim, m_uiMaxDepth);
        ot::TreeNode tb(bX, bY, bZ, bL, m_uiDim, m_uiMaxDepth);
        return ta < tb;
    };

    // Build phys_pos -> (winner_rank, winner_cg). Strict preference:
    //   1. interior writers win over padding writers (so plan-zip
    //      reads from a buffer position that solverRHS actually wrote).
    //   2. among writers in the same interior class, cascade-rule
    //      smallest TreeNode wins.
    //   3. tiebreak: smallest rank.
    struct Winner {
        int rank;
        unsigned int cg;
        unsigned int tnX, tnY, tnZ, tnLev;
        unsigned int interior;
    };

    std::unordered_map<PhysKey3, Winner, PhysKey3Hash> winners;
    winners.reserve(total);
    for (int p = 0; p < m_uiActiveNpes; p++) {
        for (int i = offs[p]; i < offs[p] + counts[p]; i++) {
            PhysKey3 k{allX[i], allY[i], allZ[i]};
            auto it = winners.find(k);
            if (it == winners.end()) {
                winners[k] = Winner{p, allCg[i],
                                    allTNX[i], allTNY[i], allTNZ[i],
                                    allTNLev[i], allInterior[i]};
            } else {
                const Winner& w = it->second;
                bool replace = false;
                if (allInterior[i] != w.interior) {
                    replace = (allInterior[i] > w.interior);
                } else if (claimLT(allTNX[i], allTNY[i], allTNZ[i],
                                   allTNLev[i],
                                   w.tnX, w.tnY, w.tnZ, w.tnLev)) {
                    replace = true;
                } else if (!claimLT(w.tnX, w.tnY, w.tnZ, w.tnLev,
                                    allTNX[i], allTNY[i], allTNZ[i],
                                    allTNLev[i])
                           && p < w.rank) {
                    replace = true;
                }
                if (replace)
                    it->second = Winner{p, allCg[i],
                                        allTNX[i], allTNY[i], allTNZ[i],
                                        allTNLev[i], allInterior[i]};
            }
        }
    }

    // ---- Cleanup: remove cascade-primary cgs from PassD's demoted
    // mirror list ----
    //
    // patchE2NCgFromMasks (mask-patch) populates m_uiPassDDemotedToGhostCg
    // when it redirects an over-claim from `old_cg` to `owner_cg`. After
    // the redirect, no LOCAL element on this rank references `old_cg`
    // via E2N_CG (it's orphan). PassD's mirror then copies a ghost cg's
    // value into `old_cg` on every readFromGhostEnd.
    //
    // BUT cascade's cross-rank winner pick (above) can independently
    // designate `old_cg` as the PRIMARY for its phys_pos: plan-zip
    // writes the correct value into `old_cg`, the integrator's axpy
    // updates it, and syncZipNonPrimary later sends `old_cg` to all
    // non-primary ranks. If PassD then mirrors a stale ghost back into
    // `old_cg` after the standard ghost exchange, it overwrites the
    // correct post-axpy value with stale IC data — silently corrupting
    // every RK stage past stage 1 of step 0 at level-transition corners.
    //
    // Fix: walk winners[] and erase any (rank == me, cg == winner_cg)
    // entry from m_uiPassDDemotedToGhostCg. Cascade primaries are NOT
    // demoted; they're the source of truth.

    size_t passDCleanupErased = 0;
    bool dbgWinnerCleanup =
        (DENDRO_PROBE_GETENV("DENDRO_PASSD_CLEANUP_DBG") != nullptr);
    for (auto& kv : winners) {
        if (kv.second.rank != m_uiActiveRank) continue;
        if (m_uiPassDDemotedToGhostCg.erase(kv.second.cg)) passDCleanupErased++;
        m_uiPassDDemotedLocalCgs.erase(kv.second.cg);
    }
    if (dbgWinnerCleanup) {
        std::cout << "[passd-cleanup r" << m_uiActiveRank
                  << "] erased " << passDCleanupErased
                  << " winner-cgs from PassDDemotedToGhostCg"
                  << std::endl;
    }

    // ---- Side-channel non-primary sync via direct Alltoallv ----
    //
    // For each phys_pos with multiple claims (cross-rank duplicates):
    //   - The "primary" rank holds the winner cg. zip writes to it.
    //   - All other "non-primary" ranks have a local cg at the same
    //     phys_pos that needs to receive the primary's value.
    //
    // We build a separate Alltoallv channel that delivers
    // primary_cg → non_primary_cg in one MPI exchange per ghost
    // exchange, INDEPENDENT of the standard scatter map (which was
    // built before Pass A/D/E rewrites and may not have the right
    // primary→ghost paths).
    //
    // Each rank computes its own send/recv list deterministically
    // from the global allgathered claims. To keep send/recv ordering
    // consistent across rank pairs, we iterate phys_pos in a sorted
    // order.
    m_uiZipSyncSendCounts.assign(m_uiActiveNpes, 0);
    m_uiZipSyncRecvCounts.assign(m_uiActiveNpes, 0);
    m_uiZipSyncSendOffsets.assign(m_uiActiveNpes, 0);
    m_uiZipSyncRecvOffsets.assign(m_uiActiveNpes, 0);
    m_uiZipSyncSendCg.clear();
    m_uiZipSyncRecvCg.clear();
    m_uiZipLocalDupSrc.clear();
    m_uiZipLocalDupDst.clear();
    if (m_uiActiveNpes > 1) {
        // Build deterministic phys_pos iteration: sort claim list by
        // phys_pos key. Direct sort of allgathered data; both ranks
        // produce the same sorted order, so when rank A says "send
        // cg X to rank B at position k" and rank B says "receive at
        // local cg L from rank A at position k", positions match.
        // Sort claim INDICES (4-byte) instead of copying 40-byte
        // PhysClaim structs: avoids a ~total*40-byte allocation+copy and
        // cuts the sort's memory traffic ~10x. The tn* fields the old
        // struct carried are unused in this section (the winner pick
        // already happened above). Determinism is preserved: all* arrays
        // are byte-identical across ranks (allgathered) and the
        // comparator is identical, so std::sort yields the same
        // permutation on every rank. rankOf[i] = which rank advertised
        // global claim i (claims are contiguous per rank in the all*
        // arrays).
        std::vector<unsigned int> rankOf(total);
        for (int p = 0; p < m_uiActiveNpes; p++)
            for (int i = offs[p]; i < offs[p] + counts[p]; i++)
                rankOf[i] = (unsigned int)p;
        std::vector<unsigned int> sidx(total);

        // Fast path: pack (x, y, z, rank) into one uint64 sort key when
        // the phys coords fit. Phys coord max ≈ 2*eOrd*2^maxDepth, so
        // 19 bits/coord covers maxDepth≲15; rank in 7 bits covers ≤128
        // ranks. Sorting uint64 keys (single integer compare,
        // cache-local) is ~5x faster than the 3-way indirect comparator
        // on allX/Y/Z. Falls back to the comparator sort if anything
        // doesn't fit. Both produce the identical (x,y,z,rank)-lex order.
        const unsigned long long COORD_BITS = 19;
        const unsigned long long COORD_MAX = (1ULL << COORD_BITS) - 1;
        bool canPack = (m_uiActiveNpes <= 127);
        if (canPack) {
            for (int t = 0; t < total; t++) {
                if (allX[t] > COORD_MAX || allY[t] > COORD_MAX
                    || allZ[t] > COORD_MAX) { canPack = false; break; }
            }
        }
        if (canPack) {
            std::vector<std::pair<unsigned long long, unsigned int>> keyed(
                total);
            for (int t = 0; t < total; t++) {
                unsigned long long key =
                    (allX[t] << (COORD_BITS * 2 + 7))
                    | (allY[t] << (COORD_BITS + 7))
                    | (allZ[t] << 7)
                    | (unsigned long long)rankOf[t];
                keyed[t] = {key, (unsigned int)t};
            }
            std::sort(keyed.begin(), keyed.end(),
                      [](const std::pair<unsigned long long, unsigned int>& a,
                         const std::pair<unsigned long long, unsigned int>& b) {
                          return a.first < b.first;
                      });
            for (int t = 0; t < total; t++) sidx[t] = keyed[t].second;
        } else {
            std::iota(sidx.begin(), sidx.end(), 0u);
            std::sort(sidx.begin(), sidx.end(),
                      [&](unsigned int a, unsigned int b) {
                          if (allX[a] != allX[b]) return allX[a] < allX[b];
                          if (allY[a] != allY[b]) return allY[a] < allY[b];
                          if (allZ[a] != allZ[b]) return allZ[a] < allZ[b];
                          return rankOf[a] < rankOf[b];
                      });
        }

        // First pass: count send/recv per rank.
        // Walk consecutive entries with same (x, y, z) to identify
        // claim groups per phys_pos. (sidx[k] indexes the all* arrays.)
        // cache multi-claim group (begin, end, winner_rank, winner_cg)
        // so pass 2 doesn't have to redo the group-finding while-loop
        // or the winners.find() lookup.
        struct McGroup {
            size_t b;
            size_t e;
            int p_rank;
            unsigned int p_cg;
        };
        std::vector<McGroup> mcGroups;
        size_t i = 0;
        while (i < sidx.size()) {
            size_t j = i;
            while (j < sidx.size()
                   && allX[sidx[j]] == allX[sidx[i]]
                   && allY[sidx[j]] == allY[sidx[i]]
                   && allZ[sidx[j]] == allZ[sidx[i]])
                ++j;
            // [i, j) is the claim group for one phys_pos.
            if (j - i > 1) {
                // Multi-claim phys_pos: find primary, count sync.
                auto wit = winners.find(PhysKey3{
                    allX[sidx[i]], allY[sidx[i]], allZ[sidx[i]]});
                if (wit != winners.end()) {
                    const int p_rank = wit->second.rank;
                    mcGroups.push_back({i, j, p_rank, wit->second.cg});
                    for (size_t k = i; k < j; k++) {
                        const int r = (int)rankOf[sidx[k]];
                        if (r == p_rank) continue;  // skip primary itself
                        if (m_uiActiveRank == p_rank)
                            m_uiZipSyncSendCounts[r]++;
                        if (m_uiActiveRank == r)
                            m_uiZipSyncRecvCounts[p_rank]++;
                    }
                }
            }
            i = j;
        }
        for (int p = 1; p < m_uiActiveNpes; p++) {
            m_uiZipSyncSendOffsets[p] = m_uiZipSyncSendOffsets[p - 1]
                                        + m_uiZipSyncSendCounts[p - 1];
            m_uiZipSyncRecvOffsets[p] = m_uiZipSyncRecvOffsets[p - 1]
                                        + m_uiZipSyncRecvCounts[p - 1];
        }
        const int totalSend = m_uiZipSyncSendOffsets[m_uiActiveNpes - 1]
                              + m_uiZipSyncSendCounts[m_uiActiveNpes - 1];
        const int totalRecv = m_uiZipSyncRecvOffsets[m_uiActiveNpes - 1]
                              + m_uiZipSyncRecvCounts[m_uiActiveNpes - 1];
        m_uiZipSyncSendCg.resize(totalSend);
        m_uiZipSyncRecvCg.resize(totalRecv);

        // Second pass: fill send/recv cg lists.
        // reuses mcGroups from pass 1 — no re-walking sidx, no winners
        // re-lookup. saves one O(total_claims) scan + as many hash
        // lookups as multi-claim phys positions.
        std::vector<int> sendPos(m_uiActiveNpes), recvPos(m_uiActiveNpes);
        for (int p = 0; p < m_uiActiveNpes; p++) {
            sendPos[p] = m_uiZipSyncSendOffsets[p];
            recvPos[p] = m_uiZipSyncRecvOffsets[p];
        }
        for (const auto& g : mcGroups) {
            const int p_rank        = g.p_rank;
            const unsigned int p_cg = g.p_cg;
            for (size_t k = g.b; k < g.e; k++) {
                const int r = (int)rankOf[sidx[k]];
                const unsigned int kcg = allCg[sidx[k]];
                if (r == p_rank) {
                    // intra-rank duplicate: same rank has the
                    // primary AND another cg at the same phys.
                    // sync's Alltoallv handles cross-rank
                    // entries only; for intra-rank we record a
                    // direct (src=winner_cg, dst=this_cg) pair
                    // and copy locally after every sync.
                    if (m_uiActiveRank == r && kcg != p_cg) {
                        m_uiZipLocalDupSrc.push_back(p_cg);
                        m_uiZipLocalDupDst.push_back(kcg);
                    }
                    continue;
                }
                if (m_uiActiveRank == p_rank)
                    m_uiZipSyncSendCg[sendPos[r]++] = p_cg;
                if (m_uiActiveRank == r)
                    m_uiZipSyncRecvCg[recvPos[p_rank]++] = kcg;
            }
        }
        // probe: dump multi-claim phys positions to file. one file per
        // rank per remesh. set DENDRO_DUP_PROBE_DIR=/path to enable.
        static int s_dup_probe_call_id = 0;
        const char* dup_probe_dir = DENDRO_PROBE_GETENV("DENDRO_DUP_PROBE_DIR");
        if (dup_probe_dir) {
            // also write a sidecar with the syncRecvCg / syncSendCg arrays
            // so we can verify whether a given non-primary local cg is in
            // the recv list.
            char fn2[1024];
            std::snprintf(fn2, sizeof(fn2),
                          "%s/sync_arrays_call%d_r%d.txt",
                          dup_probe_dir, s_dup_probe_call_id,
                          (int)m_uiActiveRank);
            FILE* fp2 = std::fopen(fn2, "w");
            if (fp2) {
                std::fprintf(fp2, "# rank=%d call=%d sendN=%zu recvN=%zu planN=%zu\n",
                             (int)m_uiActiveRank,
                             s_dup_probe_call_id,
                             m_uiZipSyncSendCg.size(),
                             m_uiZipSyncRecvCg.size(),
                             m_uiZipPlanCg.size());
                std::fprintf(fp2, "# send_cgs:\n");
                for (size_t kk = 0; kk < m_uiZipSyncSendCg.size(); kk++)
                    std::fprintf(fp2, "S %u\n", m_uiZipSyncSendCg[kk]);
                std::fprintf(fp2, "# recv_cgs:\n");
                for (size_t kk = 0; kk < m_uiZipSyncRecvCg.size(); kk++)
                    std::fprintf(fp2, "R %u\n", m_uiZipSyncRecvCg[kk]);
                std::fprintf(fp2, "# plan_cgs:\n");
                for (size_t kk = 0; kk < m_uiZipPlanCg.size(); kk++)
                    std::fprintf(fp2, "P %u\n", m_uiZipPlanCg[kk]);
                std::fclose(fp2);
            }
            char fn[1024];
            std::snprintf(fn, sizeof(fn),
                          "%s/dup_probe_call%d_r%d.txt",
                          dup_probe_dir, s_dup_probe_call_id,
                          (int)m_uiActiveRank);
            FILE* fp = std::fopen(fn, "w");
            if (fp) {
                std::fprintf(fp,
                    "# rank=%d call=%d\n"
                    "# phys_x phys_y phys_z n_claims winner_rank "
                    "winner_cg winner_tnLev winner_interior claims...\n",
                    (int)m_uiActiveRank, s_dup_probe_call_id);
                size_t ii = 0;
                while (ii < sidx.size()) {
                    size_t jj = ii;
                    while (jj < sidx.size()
                           && allX[sidx[jj]] == allX[sidx[ii]]
                           && allY[sidx[jj]] == allY[sidx[ii]]
                           && allZ[sidx[jj]] == allZ[sidx[ii]])
                        ++jj;
                    if (jj - ii > 1) {
                        auto wit2 = winners.find(PhysKey3{
                            allX[sidx[ii]], allY[sidx[ii]], allZ[sidx[ii]]});
                        std::fprintf(fp,
                            "%llu %llu %llu %zu",
                            (unsigned long long)allX[sidx[ii]],
                            (unsigned long long)allY[sidx[ii]],
                            (unsigned long long)allZ[sidx[ii]],
                            jj - ii);
                        if (wit2 != winners.end()) {
                            std::fprintf(fp, " %d %u %u %u",
                                wit2->second.rank,
                                wit2->second.cg,
                                wit2->second.tnLev,
                                wit2->second.interior);
                        } else {
                            std::fprintf(fp, " -1 0 0 0");
                        }
                        for (size_t k = ii; k < jj; k++) {
                            std::fprintf(fp, " | r=%d cg=%u tnLev=%u",
                                (int)rankOf[sidx[k]],
                                allCg[sidx[k]],
                                allTNLev[sidx[k]]);
                        }
                        std::fprintf(fp, "\n");
                    }
                    ii = jj;
                }
                std::fclose(fp);
            }
            s_dup_probe_call_id++;
        }
    }

    // ---- Stage 3: build inverse scatter map for non-primary sync ----
    //
    // Each entry m_uiScatterMapActualNodeRecv[k] is a ghost cg on this
    // rank, populated at recv buffer position k by the source rank.
    // The source rank is determined by k's range: k in
    // [m_uiRecvNodeOffset[p], +m_uiRecvNodeCount[p]) means source = p.
    // The source CG is the value at the SAME buffer position in the
    // sender's m_uiScatterMapActualNodeSend.
    //
    // We exchange the send-cg index lists via Alltoallv so each rank
    // can build (source_rank, source_cg) -> ghost_cg.
    m_uiZipNonPrimaryToGhostCg.clear();
    std::unordered_map<uint64_t, unsigned int> invScatter;
    if (m_uiActiveNpes > 1
        && !m_uiScatterMapActualNodeRecv.empty()) {
        const size_t sendN = m_uiScatterMapActualNodeSend.size();
        const size_t recvN = m_uiScatterMapActualNodeRecv.size();
        std::vector<unsigned int> sendBuf(sendN);
        std::vector<unsigned int> recvBuf(recvN);
        for (size_t k = 0; k < sendN; k++)
            sendBuf[k] = m_uiScatterMapActualNodeSend[k];
        std::vector<int> sCounts(m_uiActiveNpes), rCounts(m_uiActiveNpes),
            sOffs(m_uiActiveNpes), rOffs(m_uiActiveNpes);
        for (int p = 0; p < m_uiActiveNpes; p++) {
            sCounts[p] = (int)m_uiSendNodeCount[p];
            rCounts[p] = (int)m_uiRecvNodeCount[p];
            sOffs[p]   = (int)m_uiSendNodeOffset[p];
            rOffs[p]   = (int)m_uiRecvNodeOffset[p];
        }
        MPI_Alltoallv(sendBuf.data(), sCounts.data(), sOffs.data(),
                      MPI_UNSIGNED, recvBuf.data(), rCounts.data(),
                      rOffs.data(), MPI_UNSIGNED, m_uiCommActive);
        invScatter.reserve(recvN);
        for (int p = 0; p < m_uiActiveNpes; p++) {
            const unsigned int base = m_uiRecvNodeOffset[p];
            const unsigned int cnt  = m_uiRecvNodeCount[p];
            for (unsigned int k = 0; k < cnt; k++) {
                const unsigned int src_cg   = recvBuf[base + k];
                const unsigned int ghost_cg =
                    m_uiScatterMapActualNodeRecv[base + k];
                const uint64_t key =
                    (uint64_t(p) << 32) | uint64_t(src_cg);
                // first occurrence wins (stable choice); duplicates
                // shouldn't matter for correctness.
                if (invScatter.find(key) == invScatter.end())
                    invScatter[key] = ghost_cg;
            }
        }
    }

    // For each LOCAL cg where I'm the primary, build a plan entry.
    // The writer (elem, sub) is the smallest-TN local elem at this
    // phys_pos (lookup via myMin).
    // For each LOCAL cg where I'm NOT the primary, look up the
    // ghost cg whose source is the primary's local cg on the primary's
    // rank (via the inverse scatter map). Stash in
    // m_uiZipNonPrimaryToGhostCg for the post-ghost-exchange sync.
    for (unsigned int cg = nLB; cg < nLE; cg++) {
        unsigned int dg = m_uiCG2DG[cg];
        if (dg == LOOK_UP_TABLE_DEFAULT) continue;
        unsigned int e = dg / npe;
        unsigned int n = dg % npe;
        if (e >= nElTot) continue;
        unsigned long long x, y, z;
        encodeKey(e, n, x, y, z);
        PhysKey3 k{x, y, z};
        auto wit = winners.find(k);
        if (wit == winners.end()) continue;

        // Non-primary case: someone else (or my different local cg)
        // is the primary. Look up the ghost cg on this rank whose
        // scatter source is the primary's local cg on the primary's
        // rank, and stash for sync.
        const bool i_am_primary_rank = (wit->second.rank == m_uiActiveRank);
        const bool this_is_primary_cg = i_am_primary_rank
            && (wit->second.cg == cg);
        if (!this_is_primary_cg) {
            const uint64_t key =
                (uint64_t(wit->second.rank) << 32)
                | uint64_t(wit->second.cg);
            auto iit = invScatter.find(key);
            if (iit != invScatter.end())
                m_uiZipNonPrimaryToGhostCg[cg] = iit->second;
            continue;
        }
        // Primary path falls through to plan-entry construction.
        auto mit = myMin.find(k);
        if (mit == myMin.end()) continue;  // I'm primary but have no
                                            // local elem? shouldn't
                                            // happen since we
                                            // advertised based on
                                            // myMin.

        const unsigned int writer_elem = mit->second.elem;
        const unsigned int writer_sub  = mit->second.sub;

        // Look up writer_elem's block to compute the unzipped buffer
        // index. m_uiE2BlkMap is indexed by (elem - eLB).
        if (writer_elem < m_uiElementLocalBegin
            || writer_elem >= m_uiElementLocalEnd) continue;
        const unsigned int blk =
            m_uiE2BlkMap[writer_elem - m_uiElementLocalBegin];
        if (blk == LOOK_UP_TABLE_DEFAULT) continue;
        const auto& block = m_uiLocalBlockList[blk];
        const ot::TreeNode blkNode = block.getBlockNode();
        const unsigned int regLev  = block.getRegularGridLev();
        const unsigned int lx      = block.getAllocationSzX();
        const unsigned int ly      = block.getAllocationSzY();
        const unsigned int offset    = block.getOffset();
        const unsigned int paddWidth = block.get1DPadWidth();
        const unsigned int ei =
            (pNodes[writer_elem].getX() - blkNode.getX()) >>
            (m_uiMaxDepth - regLev);
        const unsigned int ej =
            (pNodes[writer_elem].getY() - blkNode.getY()) >>
            (m_uiMaxDepth - regLev);
        const unsigned int ek =
            (pNodes[writer_elem].getZ() - blkNode.getZ()) >>
            (m_uiMaxDepth - regLev);
        const unsigned int i = writer_sub % (eOrd + 1);
        const unsigned int j = (writer_sub / (eOrd + 1)) % (eOrd + 1);
        const unsigned int kk =
            writer_sub / ((eOrd + 1) * (eOrd + 1));
        const unsigned int unzip_idx =
            offset
            + (ek * eOrd + kk + paddWidth) * (ly * lx)
            + (ej * eOrd + j + paddWidth) * lx
            + (ei * eOrd + i + paddWidth);
        m_uiZipPlanCg.push_back(cg);
        m_uiZipPlanUnzipIdx.push_back(unzip_idx);
    }

    // ---- Cleanup pass 2: PassD's heuristic mirror is shadowed by
    // (a) the precise Stage 3 mirror m_uiZipNonPrimaryToGhostCg, and
    // (b) the side-channel Alltoallv syncZipNonPrimary which delivers
    // the primary's value directly into local non-primary cgs in
    // m_uiZipSyncRecvCg.
    //
    // PassD can pick a wrong ghost source ("smallest-indexed ghost cg
    // at phys_pos" walk) and OVERWRITE the value that (a) or (b)
    // placed. Erase any cg from PassDDemotedToGhostCg that is covered
    // by either (a) or (b). What remains in PassDDemotedToGhostCg is
    // only those cgs that NEITHER mechanism handles, where PassD's
    // heuristic mirror is the only sync available.
    size_t passDOverlapErased = 0;
    for (auto& kv : m_uiZipNonPrimaryToGhostCg) {
        if (m_uiPassDDemotedToGhostCg.erase(kv.first)) passDOverlapErased++;
        m_uiPassDDemotedLocalCgs.erase(kv.first);
    }
    for (unsigned int recv_cg : m_uiZipSyncRecvCg) {
        if (m_uiPassDDemotedToGhostCg.erase(recv_cg)) passDOverlapErased++;
        m_uiPassDDemotedLocalCgs.erase(recv_cg);
    }
    if (dbgWinnerCleanup) {
        std::cout << "[passd-cleanup r" << m_uiActiveRank
                  << "] erased " << passDOverlapErased
                  << " overlap cgs from PassDDemotedToGhostCg"
                  << " (Stage3+SyncZipNonPrimary coverage)"
                  << std::endl;
    }

    // ---- Build element-read fixup map ----
    //
    // The unzip / getElementNodalValues path reads vec[E2N_CG[e*npe+sub]]
    // for every local element's sub-node. At level-transition consensus
    // phys_pos, cascade redirects can route different elements' subs to
    // different cgs (some are ghosts on this rank with stale values
    // because the rebuilt scatter map sources them from a wrong rank
    // or doesn't cover them). The standard ghost exchange + Stage 3
    // mirror + syncZipNonPrimary handle some cgs but not all.
    //
    // For each local element, every sub at a consensus phys_pos: get
    // cgIdx = E2N_CG[e*npe+sub]. The cgIdx MUST hold the primary's
    // post-sync value (otherwise unzip/getElementNodalValues read
    // stale data, which propagates through the RHS stencil).
    //
    // Build cgIdx → local_primary_or_synced_cg on this rank for every
    // such cgIdx (LOCAL or GHOST on this rank). Apply post-sync to
    // ensure cgIdx holds the consensus value.
    m_uiZipGhostToLocalAtConsensus.clear();
    {
        // Map phys_pos → a local cg on this rank that holds the
        // primary's post-sync value. Prefer the cascade winner if
        // it's me; else any local at the phys_pos that's in the
        // syncZipNonPrimary recv list (so it gets the primary's
        // value via direct alltoallv).
        std::unordered_set<unsigned int> recvCgSet;
        recvCgSet.reserve(m_uiZipSyncRecvCg.size());
        for (unsigned int recv_cg : m_uiZipSyncRecvCg)
            recvCgSet.insert(recv_cg);

        std::unordered_map<PhysKey3, unsigned int, PhysKey3Hash>
            posToLocalCg;
        posToLocalCg.reserve(winners.size());
        for (unsigned int cg = nLB; cg < nLE; cg++) {
            unsigned int dg = m_uiCG2DG[cg];
            if (dg == LOOK_UP_TABLE_DEFAULT) continue;
            unsigned int e = dg / npe;
            unsigned int n = dg % npe;
            if (e >= nElTot) continue;
            unsigned long long x, y, z;
            encodeKey(e, n, x, y, z);
            PhysKey3 k{x, y, z};
            auto wit = winners.find(k);
            if (wit == winners.end()) continue;
            const bool i_am_primary =
                (wit->second.rank == m_uiActiveRank
                 && wit->second.cg == cg);
            const bool i_will_get_synced =
                recvCgSet.count(cg) > 0;
            // Only consider cgs that will hold the consensus value
            // post-sync chain.
            if (!i_am_primary && !i_will_get_synced) continue;
            auto pit = posToLocalCg.find(k);
            if (pit == posToLocalCg.end() || i_am_primary)
                posToLocalCg[k] = cg;
        }

        // Walk every local element. For each sub, get cgIdx via E2N_CG.
        // If cgIdx's phys_pos has a winner AND we have a local mirror
        // for that phys_pos AND cgIdx isn't already that local mirror,
        // add cgIdx → local_mirror.
        for (unsigned int e = m_uiElementLocalBegin;
             e < m_uiElementLocalEnd; e++) {
            for (unsigned int n = 0; n < npe; n++) {
                unsigned int cgIdx = m_uiE2NMapping_CG[e * npe + n];
                if (cgIdx >= m_uiNumActualNodes) continue;
                if (cgIdx >= m_uiCG2DG.size()) continue;
                unsigned int dg = m_uiCG2DG[cgIdx];
                if (dg == LOOK_UP_TABLE_DEFAULT) continue;
                unsigned int oe = dg / npe;
                unsigned int os = dg % npe;
                if (oe >= nElTot) continue;
                unsigned long long x, y, z;
                encodeKey(oe, os, x, y, z);
                PhysKey3 k{x, y, z};
                auto pit = posToLocalCg.find(k);
                if (pit == posToLocalCg.end()) continue;
                if (cgIdx == pit->second) continue;  // already correct
                // Don't overwrite cgs that the primary's cleanup
                // categories already manage correctly.
                auto wfind = winners.find(k);
                if (wfind != winners.end()
                    && cgIdx == wfind->second.cg
                    && wfind->second.rank == m_uiActiveRank) continue;
                m_uiZipGhostToLocalAtConsensus[cgIdx] = pit->second;
            }
        }
    }
    if (dbgWinnerCleanup) {
        std::cout << "[passd-cleanup r" << m_uiActiveRank
                  << "] built " << m_uiZipGhostToLocalAtConsensus.size()
                  << " element-read fixup entries"
                  << std::endl;
    }

    // ---- E2N_CG cross-instance unification ----
    //
    // When the mesh has multiple instances of the same TreeNode in
    // m_uiAllElements (deeper ghost layers from R2/R3 fetch), each
    // instance gets its own cascade walk producing E2N_CG entries.
    // For instances with incomplete neighbor sets (deeper ghosts), the
    // walk lands at wrong cgs — observed at level-transition edges
    // where elem 1056 sub (0,6,3) routes to a cg whose phys_pos is
    // off by 8 grid units in Z.
    //
    // Cascade is deterministic: any two instances of the same TN
    // SHOULD produce identical E2N_CG. When they don't, the one whose
    // routed cg's phys_pos matches the (e, sub) phys_pos is correct;
    // others are wrong. Override wrong instances to use the correct.
    {
        struct TNKey {
            unsigned int x, y, z, lev;
            bool operator==(const TNKey& o) const {
                return x == o.x && y == o.y && z == o.z && lev == o.lev;
            }
        };
        struct TNKeyHash {
            size_t operator()(const TNKey& k) const {
                size_t h = std::hash<unsigned int>()(k.x);
                h ^= std::hash<unsigned int>()(k.y) << 1;
                h ^= std::hash<unsigned int>()(k.z) << 2;
                h ^= std::hash<unsigned int>()(k.lev) << 3;
                return h;
            }
        };
        std::unordered_map<TNKey, std::vector<unsigned int>, TNKeyHash>
            tnGroups;
        for (unsigned int e = 0; e < nElTot; e++) {
            TNKey k{pNodes[e].getX(), pNodes[e].getY(),
                    pNodes[e].getZ(), pNodes[e].getLevel()};
            tnGroups[k].push_back(e);
        }
        size_t e2n_cross_fixed = 0;
        for (auto& kv : tnGroups) {
            if (kv.second.size() <= 1) continue;
            for (unsigned int sub = 0; sub < npe; sub++) {
                unsigned long long ex, ey, ez;
                encodeKey(kv.second[0], sub, ex, ey, ez);
                unsigned int best_cg = LOOK_UP_TABLE_DEFAULT;
                for (unsigned int e : kv.second) {
                    unsigned int cg = m_uiE2NMapping_CG[e * npe + sub];
                    if (cg >= m_uiCG2DG.size()) continue;
                    unsigned int dg = m_uiCG2DG[cg];
                    if (dg == LOOK_UP_TABLE_DEFAULT) continue;
                    unsigned int oe = dg / npe;
                    unsigned int os = dg % npe;
                    if (oe >= nElTot) continue;
                    unsigned long long rx, ry, rz;
                    encodeKey(oe, os, rx, ry, rz);
                    if (rx == ex && ry == ey && rz == ez) {
                        best_cg = cg;
                        break;
                    }
                }
                if (best_cg == LOOK_UP_TABLE_DEFAULT) continue;
                for (unsigned int e : kv.second) {
                    unsigned int cg = m_uiE2NMapping_CG[e * npe + sub];
                    if (cg == best_cg) continue;
                    bool wrong = (cg >= m_uiCG2DG.size());
                    if (!wrong) {
                        unsigned int dg = m_uiCG2DG[cg];
                        if (dg == LOOK_UP_TABLE_DEFAULT) {
                            wrong = true;
                        } else {
                            unsigned int oe = dg / npe;
                            unsigned int os = dg % npe;
                            if (oe >= nElTot) wrong = true;
                            else {
                                unsigned long long rx, ry, rz;
                                encodeKey(oe, os, rx, ry, rz);
                                if (rx != ex || ry != ey || rz != ez)
                                    wrong = true;
                            }
                        }
                    }
                    if (wrong) {
                        m_uiE2NMapping_CG[e * npe + sub] = best_cg;
                        e2n_cross_fixed++;
                    }
                }
            }
        }
        if (dbgWinnerCleanup) {
            std::cout << "[passd-cleanup r" << m_uiActiveRank
                      << "] E2N_CG cross-TN unify: fixed="
                      << e2n_cross_fixed << std::endl;
        }
    }
}

void Mesh::computeNodalScatterMapDG(MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    dendro::logger::debug(dendro::logger::Scope{"MESH"},
                          "Now computing the nodal scattermap");

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (npes <= 1)
        return;  // nothing to do in the sequential case. (No scatter map
                 // required.)

    m_uiSendNodeCount.resize(npes);
    m_uiRecvNodeCount.resize(npes);
    m_uiSendNodeOffset.resize(npes);
    m_uiRecvNodeOffset.resize(npes);

    for (unsigned int p = 0; p < npes; p++) {
        m_uiSendNodeCount[p]  = m_uiSendEleCount[p] * m_uiNpE;
        m_uiRecvNodeCount[p]  = m_uiRecvEleCount[p] * m_uiNpE;

        m_uiSendNodeOffset[p] = m_uiSendEleOffset[p] * m_uiNpE;
        m_uiRecvNodeOffset[p] = m_uiRecvNodeOffset[p] * m_uiNpE;
    }

    if ((m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]) !=
        m_uiGhostElementRound1Index.size() * m_uiNpE) {
        std::cout << "Error: " << __func__ << " line: " << __LINE__
                  << " send and recv DG node mismatch " << std::endl;
        MPI_Abort(comm, 0);
    }

    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();

    m_uiScatterMapActualNodeSend.resize(
        (m_uiSendNodeOffset[npes - 1] + m_uiSendNodeCount[npes - 1]));
    m_uiScatterMapActualNodeRecv.resize(
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]));

    unsigned int nCount = 0;
    // note that we don't need all the only surface points are enough.
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int k = m_uiSendEleOffset[p];
             k < (m_uiSendEleOffset[p] + m_uiSendEleCount[p]); k++) {
            for (unsigned int n = 0; n < m_uiNpE; n++, nCount++)
                m_uiScatterMapActualNodeSend[nCount] =
                    m_uiE2NMapping_CG[(m_uiScatterMapElementRound1[k] +
                                       m_uiElementLocalBegin) *
                                          m_uiNpE +
                                      n];
        }
    }

    nCount = 0;
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int k = m_uiRecvEleOffset[p];
             k < (m_uiRecvEleOffset[p] + m_uiRecvEleCount[p]); k++) {
            for (unsigned int n = 0; n < m_uiNpE; n++, nCount++)
                m_uiScatterMapActualNodeRecv[nCount] =
                    m_uiE2NMapping_CG[m_uiGhostElementRound1Index[k] * m_uiNpE +
                                      n];
        }
    }

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished building the nodal scatter map!");

    return;
}

void Mesh::computeNodeScatterMaps(MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    ot::TreeNode minMaxLocalNode[2];

    ot::TreeNode rootNode(0, 0, 0, 0, m_uiDim, m_uiMaxDepth);
    std::vector<ot::TreeNode> tmpNode;

    unsigned int x, y, z, sz;  // x y z and size of an octant.
    unsigned int ownerID, ii_x, jj_y,
        kk_z;  // DG index to ownerID and ijk decomposition variable.
    std::set<unsigned int>
        nodeIndexVisited;  // To keep track of the nodes already include in the
                           // sendNode maps.
    std::pair<std::set<unsigned int>::iterator, bool> setHintUint;

    unsigned int nodeIndex;
    unsigned int nodeIndex_DG;
    unsigned int elementLookUp;

#ifdef DEBUG_E2N_MAPPING_SM
    std::vector<ot::TreeNode> cusEleCheck;
    /*  for(unsigned int
   ele=m_uiElementLocalBegin;ele<m_uiElementLocalEnd;ele++)
   {

       for(unsigned int k=0;k<(m_uiElementOrder+1);k++)
           for(unsigned int j=0;j<(m_uiElementOrder+1);j++)
               for(unsigned int i=0;i<(m_uiElementOrder+1);i++) {

                   if(m_uiActiveRank==0)
                   {
                       dg2eijk(m_uiE2NMapping_DG[ele * m_uiNpE +
                                                 k * (m_uiElementOrder + 1) *
   (m_uiElementOrder + 1) + j * (m_uiElementOrder + 1) + i], ownerID, ii_x,
   jj_y, kk_z); x = m_uiAllElements[ownerID].getX(); y =
   m_uiAllElements[ownerID].getY(); z = m_uiAllElements[ownerID].getZ(); sz = 1u
   << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());

                       cusEleCheck.push_back(ot::TreeNode((x + ii_x *
   sz/m_uiElementOrder), (y + jj_y * sz/m_uiElementOrder), (z + kk_z *
   sz/m_uiElementOrder), m_uiMaxDepth,m_uiDim, m_uiMaxDepth));

                   }


               }

       if(m_uiActiveRank==0){
           cusEleCheck.push_back(m_uiAllElements[ele]);
           treeNodesTovtk(cusEleCheck,ele,"cusEleCheck");
           cusEleCheck.clear();
       }


   }*/
#endif

    // 1. compute the local nodes.
    ///@todo We need not to generate all the nodes (which is expensive). we just
    /// need to find min and the max nodes.

    for (unsigned int ele = m_uiElementLocalBegin; ele < m_uiElementLocalEnd;
         ele++) {
        for (unsigned int i = 0; i < m_uiElementOrder + 1; i++)
            for (unsigned int j = 0; j < m_uiElementOrder + 1; j++)
                for (unsigned int k = 0; k < m_uiElementOrder + 1; k++) {
                    if (((i > 1) && (i < (m_uiElementOrder - 1))) ||
                        ((j > 1) && (j < (m_uiElementOrder - 1))) ||
                        ((k > 1) && (k < (m_uiElementOrder - 1))))
                        continue;
                    nodeIndex =
                        m_uiE2NMapping_CG[ele * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i];

                    if (nodeIndex >= m_uiNodeLocalBegin &&
                        nodeIndex < m_uiNodeLocalEnd) {
                        setHintUint = nodeIndexVisited.emplace(nodeIndex);
                        if (setHintUint.second) {
                            nodeIndex_DG =
                                m_uiE2NMapping_DG[ele * m_uiNpE +
                                                  k * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  j * (m_uiElementOrder + 1) +
                                                  i];
                            dg2eijk(nodeIndex_DG, ownerID, ii_x, jj_y, kk_z);
                            x  = m_uiAllElements[ownerID].getX();
                            y  = m_uiAllElements[ownerID].getY();
                            z  = m_uiAllElements[ownerID].getZ();
                            sz = 1u << (m_uiMaxDepth -
                                        m_uiAllElements[ownerID].getLevel());
                            assert(sz % m_uiElementOrder == 0);
                            m_uiAllLocalNode.push_back(ot::TreeNode(
                                (x + ii_x * sz / m_uiElementOrder),
                                (y + jj_y * sz / m_uiElementOrder),
                                (z + kk_z * sz / m_uiElementOrder),
                                m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                        }
                    }
                }
    }

    // 2. Find the local max of the local nodes to compute local splitters.
    ///@todo: Can use optimal Local node computation to find the min max.
    // NOTE: We need to use m_uiMaxDepth+1 to sort because this generates the
    // all the possible nodes. Hence it contains 1u<<m_uiMaxDepth for (x y z)
    // values.
    // SFC::seqSort::SFC_treeSort(&(*(m_uiAllLocalNode.begin())),m_uiAllLocalNode.size(),tmpNode,tmpNode,tmpNode,m_uiMaxDepth+1,m_uiMaxDepth+1,rootNode,ROOT_ROTATION,1,TS_SORT_ONLY);
    // SFC::seqSort::SFC_treeSortLocalOptimal(&(*(m_uiAllLocalNode.begin())),m_uiAllLocalNode.size(),m_uiMaxDepth+1,m_uiMaxDepth+1,rootNode,ROOT_ROTATION,true,minMaxLocalNode[0]);
    // SFC::seqSort::SFC_treeSortLocalOptimal(&(*(m_uiAllLocalNode.begin())),m_uiAllLocalNode.size(),m_uiMaxDepth+1,m_uiMaxDepth+1,rootNode,ROOT_ROTATION,false,minMaxLocalNode[1]);
    SFC::seqSort::SFC_treeSortLocalOptimal(
        &(*(m_uiAllLocalNode.begin())), m_uiAllLocalNode.size(),
        m_uiMaxDepth + 1, m_uiMaxDepth + 1, rootNode, ROOT_ROTATION,
        minMaxLocalNode[0], minMaxLocalNode[1]);
    // minMaxLocalNode[0] = m_uiAllLocalNode.front();
    // minMaxLocalNode[1] = m_uiAllLocalNode.back();

#ifdef DEBUG_E2N_MAPPING_SM
    treeNodesTovtk(m_uiAllLocalNode, m_uiActiveRank, "m_uiAllLocalNode");
#endif

    // 3. Gather the splitter max
    m_uiSplitterNodes = new ot::TreeNode[2 * npes];  // both min and max
    std::vector<unsigned int> minMaxIDs;
    minMaxIDs.resize(2 * npes);

    par::Mpi_Allgather(minMaxLocalNode, m_uiSplitterNodes, 2, comm);
#ifdef DEBUG_E2N_MAPPING_SM
    std::vector<ot::TreeNode> splitterNodes;
    std::vector<ot::TreeNode> splitterElements;

    for (unsigned int p = 0; p < npes; p++) {
        splitterNodes.push_back(m_uiSplitterNodes[p]);
        splitterElements.push_back(m_uiLocalSplitterElements[p]);
    }

    if (!rank) treeNodesTovtk(splitterNodes, rank, "splitterNodes");
    if (!rank) treeNodesTovtk(splitterElements, rank, "splitterElements");

    assert(seq::test::isUniqueAndSorted(splitterElements));
#endif

    m_uiScatterMapActualNodeSend.clear();
    m_uiSendNodeCount.resize(npes);   //=new unsigned int [npes];
    m_uiRecvNodeCount.resize(npes);   //=new unsigned int [npes];
    m_uiSendNodeOffset.resize(npes);  //=new unsigned int [npes];
    m_uiRecvNodeOffset.resize(npes);  //=new unsigned int [npes];

    std::set<unsigned int> *scatterMapNodeSet =
        new std::set<unsigned int>[npes];  // To keep track of the nodes to send
                                           // to each processor.
    std::vector<ot::Key> *sendNodeOctants = new std::vector<ot::Key>[npes];
    std::vector<SearchKey> allocatedGhostNodes;

    // assert((m_uiNumPreGhostElements+m_uiNumPostGhostElements)==(m_uiRecvOctOffsetRound1[npes-1]+m_uiRecvOctCountRound1[npes-1]));

#ifdef DEBUG_E2N_MAPPING_SM
    std::vector<ot::TreeNode> customElements;
    std::vector<ot::TreeNode> sendNodes[npes];

    /*if(!m_uiActiveRank) customElements.push_back(m_uiAllElements[206]);
    if(!m_uiActiveRank) customElements.push_back(m_uiAllElements[57]);
    if(!m_uiActiveRank)
    treeNodesTovtk(customElements,m_uiActiveRank,"customElement");*/

    std::vector<ot::TreeNode> allocatedGNodes;

#endif

    // 3a. Compute send nodes based on the send elements that processor p has
    // sent to other processors in the round 1 ghost communication.
    for (unsigned int p = 0; p < npes; p++) {
        m_uiSendNodeCount[p] = 0;
        scatterMapNodeSet[p] = std::set<unsigned int>();
        sendNodeOctants[p]   = std::vector<ot::Key>();
    }

    std::vector<ot::TreeNode> neighbourElement;
    std::vector<ot::SearchKey> *gEleChained =
        new std::vector<ot::SearchKey>[npes];
    std::vector<ot::SearchKey>::iterator hintSK;
    std::vector<Key> ghostElementChained;
    std::vector<Key> tmpKeys;
    Key rootKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth);
    SearchKey rootSKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth);
    unsigned int nodeFlag  = 0;
    unsigned int tmpEleLev = 0;
    ot::TreeNode tmpElement;
    ot::SearchKey tmpSKey;
    ot::Key tmpKey;

    unsigned int *sendChainedGCount    = new unsigned int[npes];
    unsigned int *recvChainedGCount    = new unsigned int[npes];
    unsigned int *sendChainedGOffset   = new unsigned int[npes];
    unsigned int *recvChainedGOffset   = new unsigned int[npes];
    unsigned int *recvChainedKeyCount  = new unsigned int[npes];
    unsigned int *recvChainedKeyOffset = new unsigned int[npes];

    std::vector<unsigned int> sendChainedGBuffer;
    std::set<unsigned int> *sendChainedGhostIDSet =
        new std::set<unsigned int>[npes];

    // 4. Allocation of the node corresponding to pre & post level 1 ghost
    // elements.

    nodeIndexVisited.clear();

    for (unsigned int ele = 0; ele < m_uiGhostElementRound1Index.size();
         ele++) {
        for (unsigned int node = 0; node < m_uiNpE; node++) {
            nodeIndex =
                m_uiE2NMapping_CG[m_uiGhostElementRound1Index[ele] * m_uiNpE +
                                  node];
            if (!(nodeIndex >= m_uiNodeLocalBegin &&
                  nodeIndex < m_uiNodeLocalEnd)) {
                assert(nodeIndex != LOOK_UP_TABLE_DEFAULT);
                setHintUint = nodeIndexVisited.emplace(nodeIndex);
                if (setHintUint.second) {
                    nodeIndex_DG =
                        m_uiE2NMapping_DG[m_uiGhostElementRound1Index[ele] *
                                              m_uiNpE +
                                          node];
                    dg2eijk(nodeIndex_DG, ownerID, ii_x, jj_y, kk_z);

                    x  = m_uiAllElements[ownerID].getX();
                    y  = m_uiAllElements[ownerID].getY();
                    z  = m_uiAllElements[ownerID].getZ();
                    sz = 1u << (m_uiMaxDepth -
                                m_uiAllElements[ownerID].getLevel());
                    assert(sz % m_uiElementOrder == 0);
                    tmpSKey =
                        SearchKey((x + ii_x * sz / m_uiElementOrder),
                                  (y + jj_y * sz / m_uiElementOrder),
                                  (z + kk_z * sz / m_uiElementOrder),
                                  m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1);
                    tmpSKey.addOwner((*(setHintUint.first)));

                    allocatedGhostNodes.push_back(tmpSKey);
                    assert(ownerID < m_uiAllElements.size());

                    tmpElement = m_uiAllElements[ownerID];
                    assert(tmpElement.getLevel() >=
                           m_uiAllElements[ownerID].getLevel());
                    nodeFlag = getDIROfANode(ii_x, jj_y, kk_z);

                    if (nodeFlag == OCT_DIR_INTERNAL)
                        assert(ownerID == m_uiGhostElementRound1Index[ele]);

                    tmpEleLev = tmpElement.getLevel();
                    tmpElement.setFlag(
                        (tmpEleLev) |
                        (1u << (nodeFlag + CHAINED_GHOST_OFFSET)));
                    assert(tmpElement.getFlag() >> (CHAINED_GHOST_OFFSET) &
                           (1u << nodeFlag));

                    hintSK = gEleChained[m_uiActiveRank].emplace(
                        gEleChained[m_uiActiveRank].end(),
                        SearchKey(tmpElement));
                    hintSK->addOwner(m_uiActiveRank);
                }
            }
        }
    }

#ifdef DEBUG_E2N_MAPPING_SM
    treeNodesTovtk(allocatedGhostNodes, rank, "allocatedGhostNodes");
#endif

    ghostElementChained.clear();
    std::vector<SearchKey> tmpSearchKeyVec;
    unsigned int skip = 1;
    for (unsigned int p = 0; p < npes; p++) {
        SFC::seqSort::SFC_treeSort(
            &(*(gEleChained[p].begin())), gEleChained[p].size(),
            tmpSearchKeyVec, tmpSearchKeyVec, tmpSearchKeyVec, m_uiMaxDepth,
            m_uiMaxDepth, rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);
        for (unsigned int e = 0; e < (gEleChained[p].size()); e++) {
            tmpElement = gEleChained[p][e];
            skip       = 1;
            while (((e + skip) < gEleChained[p].size()) &&
                   (gEleChained[p][e] == gEleChained[p][e + skip])) {
                tmpElement.setFlag((tmpElement.getFlag()) |
                                   (gEleChained[p][e + skip].getFlag()));
                skip++;
            }
            e += (skip - 1);

            tmpKey = ot::Key(tmpElement);
            tmpKey.addOwner(p);
            tmpKey.setFlag(tmpElement.getFlag());
            ghostElementChained.push_back(tmpKey);
        }
    }

#ifdef DEBUG_E2N_MAPPING_SM
    std::cout << "m_uiActiveRank " << m_uiActiveRank
              << " gElementChained.size(): " << ghostElementChained.size()
              << std::endl;
    treeNodesTovtk(ghostElementChained, m_uiActiveRank, "ghostElementChained");
#endif

    std::vector<unsigned int> recvGhostChainedFlag;
    std::vector<Key> recvGhostChained;
    std::vector<unsigned int> recvChainedGBuffer;
    std::vector<Key> recvGhostChainedSearchKeys;
    std::vector<unsigned int> *ownerList;
    unsigned int result;

    unsigned int tmpCornerIndex;
    std::vector<unsigned int> tmpCornerIndexVec;

    std::vector<Key> missingNodes;
    std::vector<SearchKey> missingNodesSkey;

    while (ghostElementChained.size()) {
        sendChainedGBuffer.clear();
        recvGhostChainedFlag.clear();
        recvGhostChained.clear();
        recvChainedGBuffer.clear();
        recvGhostChainedSearchKeys.clear();
        missingNodes.clear();
        missingNodesSkey.clear();

        for (unsigned int p = 0; p < npes; p++) {
            gEleChained[p].clear();
            sendChainedGhostIDSet[p].clear();
            sendChainedGCount[p]   = 0;
            recvChainedKeyCount[p] = 0;
        }

        unsigned int myX, myY, myZ, mySz;

        for (unsigned int e = 0; e < ghostElementChained.size(); e++) {
            myX      = ghostElementChained[e].getX();
            myY      = ghostElementChained[e].getY();
            myZ      = ghostElementChained[e].getZ();
            mySz     = 1u << (m_uiMaxDepth - ghostElementChained[e].getLevel());

            // assert(mySz%m_uiElementOrder ==0);

            nodeFlag = ghostElementChained[e].getFlag();
            nodeFlag = nodeFlag >> (CHAINED_GHOST_OFFSET);

            // assert(mySz%m_uiElementOrder==0);
            //  Corner nodes.
            if ((nodeFlag & (1u << (OCT_DIR_LEFT_DOWN_BACK)))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 0 * mySz), (myY + 0 * mySz),
                              (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            if ((nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_BACK))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 1 * mySz), (myY + 0 * mySz),
                              (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            if ((nodeFlag & (1u << OCT_DIR_LEFT_UP_BACK))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 0 * mySz), (myY + 1 * mySz),
                              (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            if ((nodeFlag & (1u << OCT_DIR_RIGHT_UP_BACK))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 1 * mySz), (myY + 1 * mySz),
                              (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            if ((nodeFlag & (1u << OCT_DIR_LEFT_DOWN_FRONT))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 0 * mySz), (myY + 0 * mySz),
                              (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            if ((nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_FRONT))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 1 * mySz), (myY + 0 * mySz),
                              (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            if ((nodeFlag & (1u << OCT_DIR_LEFT_UP_FRONT))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 0 * mySz), (myY + 1 * mySz),
                              (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            if ((nodeFlag & (1u << OCT_DIR_RIGHT_UP_FRONT))) {
                hintSK = missingNodesSkey.emplace(
                    missingNodesSkey.end(),
                    SearchKey((myX + 1 * mySz), (myY + 1 * mySz),
                              (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                              m_uiMaxDepth + 1));
                hintSK->addOwner(e);
            }

            // if(!m_uiActiveRank) std::cout<<"m_uiActiveRank:
            // "<<m_uiActiveRank<<" corner missing nodes:
            // "<<missingNodesSet.size()<<std::endl;

            if (m_uiElementOrder > 1) {
                // internal Nodes;
                if (nodeFlag & (1u << OCT_DIR_INTERNAL)) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + (mySz / 2)),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);

                    /*if(m_uiActiveRank==0 && setHintKey.second)
                    {
                        std::cout<<" internal node added for element e: "<<e<<"
                    val: "<<ghostElementChained[e]<<std::endl;
                        std::vector<ot::TreeNode> flagedElements;
                        flagedElements.push_back(ghostElementChained[e]);
                        flagedElements.push_back(*setHintKey.first);

                        treeNodesTovtk(flagedElements,e,"flagedKeys");
                    }*/
                }

                // internal edges.  can happen if only order is >1
                if ((nodeFlag & (1u << OCT_DIR_LEFT_DOWN))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 0 * mySz), (myY + 0 * mySz),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_LEFT_UP))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 0 * mySz), (myY + 1 * mySz),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_LEFT_BACK))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 0 * mySz), (myY + (mySz / 2)),
                                  (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_LEFT_FRONT))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 0 * mySz), (myY + (mySz / 2)),
                                  (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_RIGHT_DOWN))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 1 * mySz), (myY + 0 * mySz),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_RIGHT_UP))) {
                    // if(!m_uiActiveRank) std::cout<<" m_uiActiveRank:
                    // "<<m_uiActiveRank<<" RIGHT_UP MISSING NODE ADDED
                    // "<<std::endl;
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 1 * mySz), (myY + 1 * mySz),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_RIGHT_BACK))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 1 * mySz), (myY + (mySz / 2)),
                                  (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_RIGHT_FRONT))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 1 * mySz), (myY + (mySz / 2)),
                                  (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_DOWN_BACK))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + 0 * mySz),
                                  (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_DOWN_FRONT))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + 0 * mySz),
                                  (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_UP_BACK))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + 1 * mySz),
                                  (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_UP_FRONT))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + 1 * mySz),
                                  (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }

                // internal faces.

                if ((nodeFlag & (1u << OCT_DIR_LEFT))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 0 * mySz), (myY + (mySz / 2)),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_RIGHT))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + 1 * mySz), (myY + (mySz / 2)),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_DOWN))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + 0 * mySz),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_UP))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + 1 * mySz),
                                  (myZ + (mySz / 2)), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_BACK))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + (mySz / 2)),
                                  (myZ + 0 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }
                if ((nodeFlag & (1u << OCT_DIR_FRONT))) {
                    hintSK = missingNodesSkey.emplace(
                        missingNodesSkey.end(),
                        SearchKey((myX + (mySz / 2)), (myY + (mySz / 2)),
                                  (myZ + 1 * mySz), m_uiMaxDepth + 1, m_uiDim,
                                  m_uiMaxDepth + 1));
                    hintSK->addOwner(e);
                }

                // if(!m_uiActiveRank) std::cout<<"m_uiActiveRank:
                // "<<m_uiActiveRank<<" internal missing nodes:
                // "<<missingNodesSet.size()<<" missing Node:
                // "<<*setHintKey.first<<std::endl;
            }
        }

#ifdef DEBUG_E2N_MAPPING_SM
        missingNodes.clear();
        missingNodes.insert(missingNodes.end(), missingNodesSet.begin(),
                            missingNodesSet.end());
        std::cout << "m_uiActiveRank: " << m_uiActiveRank
                  << " missing node size: " << missingNodes.size() << std::endl;
        treeNodesTovtk(missingNodes, m_uiActiveRank, "missingNodes");
        treeNodesTovtk(nonLocalNodes, rank, "nonLocal");
        missingNodes.clear();
#endif

        for (unsigned int p = 0; p < 2 * npes; p++) {
            missingNodesSkey.emplace(missingNodesSkey.end(),
                                     SearchKey(m_uiSplitterNodes[p]));
        }

        missingNodes.clear();
        SFC::seqSort::SFC_treeSort(
            &(*(missingNodesSkey.begin())), missingNodesSkey.size(),
            tmpSearchKeyVec, tmpSearchKeyVec, tmpSearchKeyVec, m_uiMaxDepth + 1,
            m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

        for (unsigned int e = 0; e < (missingNodesSkey.size()); e++) {
            skip = 1;
            tmpKey =
                Key(missingNodesSkey[e].getX(), missingNodesSkey[e].getY(),
                    missingNodesSkey[e].getZ(), missingNodesSkey[e].getLevel(),
                    m_uiDim, m_uiMaxDepth + 1);
            if (missingNodesSkey[e].getOwner() >= 0)
                tmpKey.addOwner(missingNodesSkey[e].getOwner());
            while (((e + skip) < missingNodesSkey.size()) &&
                   (missingNodesSkey[e] == missingNodesSkey[e + skip])) {
                if (missingNodesSkey[e + skip].getOwner() >= 0)
                    tmpKey.addOwner(missingNodesSkey[e + skip].getOwner());
                skip++;
            }
            missingNodes.push_back(tmpKey);
            e += (skip - 1);
        }

        /*missingNodes.insert(missingNodes.end(),missingNodesSet.begin(),missingNodesSet.end());

        //NOTE: We need to use m_uiMaxDepth+1 to sort because this generates the
        all the possible nodes. Hence it contains 1u<<m_uiMaxDepth for (x y z)
        values.
        SFC::seqSort::SFC_treeSort(&(*(missingNodes.begin())),missingNodes.size(),tmpKeys,tmpKeys,tmpKeys,m_uiMaxDepth+1,m_uiMaxDepth+1,rootKey,ROOT_ROTATION,1,TS_SORT_ONLY);*/

        std::vector<ot::Key> splitterNode_keys;
        splitterNode_keys.resize(2 * npes);
        for (unsigned int p = 0; p < 2 * npes; p++)
            splitterNode_keys[p] = ot::Key(m_uiSplitterNodes[p]);

        m_uiMaxDepth++;
        searchKeys(splitterNode_keys, missingNodes);
        m_uiMaxDepth--;

        for (unsigned int p = 0; p < 2 * npes; p++) {
            assert(splitterNode_keys[p].getFlag() & OCT_FOUND);
            minMaxIDs[p] = splitterNode_keys[p].getSearchResult();
            /*minMaxIDs[p]=(std::find(missingNodes.begin(),missingNodes.end(),m_uiSplitterNodes[p])-missingNodes.begin());
            if(minMaxIDs[p]!=splitterNode_keys[p].getSearchResult())
                std::cout<<" minMax ID: "<<minMaxIDs[p]<<" sfcTSearch:
            "<<splitterNode_keys[p].getSearchResult()<<std::endl;*/
            assert(minMaxIDs[p] < missingNodes.size());
        }

        Key *missingNodesPtr = &(*(missingNodes.begin()));
        unsigned int sBegin, sEnd;
        for (unsigned int p = 0; p < npes; p++) {
            if (p == m_uiActiveRank) continue;
            sBegin = minMaxIDs[2 * p];
            sEnd   = minMaxIDs[2 * p + 1] + 1;
            for (unsigned int e = sBegin; e < sEnd; e++) {
                for (unsigned int w = 0;
                     w < missingNodesPtr[e].getOwnerList()->size(); w++) {
                    setHintUint = sendChainedGhostIDSet[p].emplace(
                        (*(missingNodesPtr[e].getOwnerList()))[w]);
                }
            }
        }

        // std::cout<<"m_uiActiveRank: "<<m_uiActiveRank<<" splitterNodeCount:
        // "<<splitterNodeCount<<std::endl;
        for (unsigned int p = 0; p < npes; p++) {
            for (auto it = sendChainedGhostIDSet[p].begin();
                 it != sendChainedGhostIDSet[p].end(); ++it) {
                sendChainedGBuffer.push_back(ghostElementChained[*it].getX());
                sendChainedGBuffer.push_back(ghostElementChained[*it].getY());
                sendChainedGBuffer.push_back(ghostElementChained[*it].getZ());
                sendChainedGBuffer.push_back(
                    ghostElementChained[*it].getFlag());
                sendChainedGBuffer.push_back(
                    ghostElementChained[*it].getOwnerList()->size());
                sendChainedGCount[p] +=
                    5 + ghostElementChained[*it].getOwnerList()->size();
                for (unsigned int w = 0;
                     w < ghostElementChained[*it].getOwnerList()->size(); w++)
                    sendChainedGBuffer.push_back(
                        (*(ghostElementChained[*it].getOwnerList()))[w]);
            }
        }

        par::Mpi_Alltoall(sendChainedGCount, recvChainedGCount, 1, comm);

        sendChainedGOffset[0] = 0;
        recvChainedGOffset[0] = 0;

        omp_par::scan(sendChainedGCount, sendChainedGOffset, npes);
        omp_par::scan(recvChainedGCount, recvChainedGOffset, npes);

        // std::cout<<" m_uiActiveRank: "<<m_uiActiveRank<<" sendBuf ID size:
        // "<<sendChainedGBuffer.size()<<" sendCount Size:
        // "<<(sendChainedGOffset[npes-1]+sendChainedGCount[npes-1])<<std::endl;
        assert(sendChainedGBuffer.size() ==
               (sendChainedGOffset[npes - 1] + sendChainedGCount[npes - 1]));

        recvChainedGBuffer.resize(recvChainedGOffset[npes - 1] +
                                  recvChainedGCount[npes - 1]);
        par::Mpi_Alltoallv(
            &(*(sendChainedGBuffer.begin())), (int *)sendChainedGCount,
            (int *)sendChainedGOffset, &(*(recvChainedGBuffer.begin())),
            (int *)recvChainedGCount, (int *)recvChainedGOffset, comm);

        unsigned int recvGindex   = 0;
        unsigned int pCount       = 0;
        unsigned int recvKeyCount = 0;

        for (unsigned int p = 0; p < npes; p++) {
            recvKeyCount = 0;
            while (recvGindex <
                   (recvChainedGOffset[p] + recvChainedGCount[p])) {
                tmpKey = Key(recvChainedGBuffer[recvGindex],
                             recvChainedGBuffer[recvGindex + 1],
                             recvChainedGBuffer[recvGindex + 2],
                             (recvChainedGBuffer[recvGindex + 3] &
                              ot::TreeNode::MAX_LEVEL),
                             m_uiDim, m_uiMaxDepth);
                recvGhostChainedFlag.push_back(
                    recvChainedGBuffer[recvGindex + 3]);
                tmpKey.getOwnerList()->resize(
                    recvChainedGBuffer[recvGindex + 4]);
                tmpKey.getOwnerList()->assign(
                    (recvChainedGBuffer.begin() + (recvGindex + 5)),
                    (recvChainedGBuffer.begin() + (recvGindex + 5) +
                     recvChainedGBuffer[recvGindex + 4]));
                recvGindex =
                    recvGindex + 5 + recvChainedGBuffer[recvGindex + 4];
                recvGhostChained.push_back(tmpKey);
                recvKeyCount++;
            }

            recvChainedKeyCount[p] = recvKeyCount;
        }

        recvChainedKeyOffset[0] = 0;
        omp_par::scan(recvChainedKeyCount, recvChainedKeyOffset, npes);

#ifdef DEBUG_E2N_MAPPING_SM
        treeNodesTovtk(recvGhostChained, m_uiActiveRank, "recvGChained");
#endif

        recvGhostChainedSearchKeys.resize(recvGhostChained.size());
        for (unsigned int ele = 0; ele < recvGhostChained.size(); ele++) {
            recvGhostChainedSearchKeys[ele] =
                Key(recvGhostChained[ele].getX(), recvGhostChained[ele].getY(),
                    recvGhostChained[ele].getZ(),
                    recvGhostChained[ele].getLevel(), m_uiDim, m_uiMaxDepth);
            recvGhostChainedSearchKeys[ele].addOwner(ele);
            recvGhostChained[ele].setSearchResult(LOOK_UP_TABLE_DEFAULT);
        }

        /*std::cout<<"m_uiActiveRank "<<m_uiActiveRank<<" recvKeySize:
         * "<<recvGhostChained.size()<<std::endl;*/
        /* if(m_uiActiveRank==12)
         for(unsigned int p=0;p<npes;p++)
         {   //std::cout<<"recvKeyGCount p : "<<p<<" : "<<
         (recvChainedGOffset[p]+recvChainedGCount[p])<<std::endl;
             std::cout<<"recvKeyCount p: "<<p<<" :
         "<<recvChainedKeyCount[p]<<std::endl;
         }*/

        SFC::seqSearch::SFC_treeSearch(&(*(recvGhostChainedSearchKeys.begin())),
                                       &(*(m_uiAllElements.begin())), 0,
                                       recvGhostChainedSearchKeys.size(), 0,
                                       m_uiAllElements.size(), m_uiMaxDepth,
                                       m_uiMaxDepth, ROOT_ROTATION);
        for (unsigned int ele = 0; ele < recvGhostChained.size(); ele++) {
            assert(recvGhostChained[ele].getOwnerList()->size() == 1);
            if (recvGhostChainedSearchKeys[ele].getFlag() & OCT_FOUND) {
                recvGhostChained[(*(recvGhostChainedSearchKeys[ele]
                                        .getOwnerList()))[0]]
                    .setSearchResult(
                        recvGhostChainedSearchKeys[ele].getSearchResult());
            }
        }

        ghostElementChained.clear();
        std::vector<unsigned int> edgeInternalIndex;
        std::vector<unsigned int> faceInternalIndex;
        std::vector<unsigned int> elementInternalIndex;

        for (unsigned int ele = 0; ele < recvGhostChained.size(); ele++) {
            if (recvGhostChained[ele].getSearchResult() !=
                LOOK_UP_TABLE_DEFAULT) {  // implies that current chainedGhost
                                          // is one of my local node.

                tmpCornerIndexVec.clear();
                ownerList = recvGhostChained[ele].getOwnerList();
                result    = recvGhostChained[ele].getSearchResult();
                nodeFlag  = recvGhostChainedFlag[ele];
                nodeFlag  = nodeFlag >> (CHAINED_GHOST_OFFSET);

                /*if((m_uiActiveRank==12 && ele==615) || ( m_uiActiveRank==12 &&
                (recvGhostChained[ele]==recvGhostChained[615])))
                {

                    std::cout<<"ele: "<<ele<<" Key: "<<recvGhostChained[ele]<<"
                found: "<<m_uiAllElements[result]<<" nodeFlag:
                "<<recvGhostChainedFlag[ele]<<std::endl; for(unsigned int
                node=0;node<m_uiNpE;node++)
                    {
                        nodeIndex=m_uiE2NMapping_CG[result*m_uiNpE+node];
                        if(nodeIndex>=m_uiNodeLocalBegin &&
                nodeIndex<m_uiNodeLocalEnd)
                        {
                            std::cout<<"E2N: "<<ele<<" found: "<<result<<" node:
                "<<"node: "<<node;
                        }
                        std::cout<<std::endl;


                    }
                    std::vector<ot::TreeNode> cusElement;
                    cusElement.push_back(recvGhostChained[ele]);
                    treeNodesTovtk(cusElement,ele,"cusElement");
                    cusElement.clear();


                }
                */

                if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN_BACK)) {
                    cornerNodeIndex(result, 0, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_BACK)) {
                    cornerNodeIndex(result, 1, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_UP_BACK)) {
                    cornerNodeIndex(result, 2, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_UP_BACK)) {
                    cornerNodeIndex(result, 3, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN_FRONT)) {
                    cornerNodeIndex(result, 4, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_FRONT)) {
                    cornerNodeIndex(result, 5, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_UP_FRONT)) {
                    cornerNodeIndex(result, 6, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_UP_FRONT)) {
                    cornerNodeIndex(result, 7, tmpCornerIndex);
                    tmpCornerIndexVec.push_back(tmpCornerIndex);
                }

                if (m_uiElementOrder > 1) {
                    if (nodeFlag & (1u << OCT_DIR_INTERNAL)) {
                        elementNodeIndex(result, elementInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 elementInternalIndex.begin(),
                                                 elementInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN)) {
                        edgeNodeIndex(result, OCT_DIR_LEFT, OCT_DIR_DOWN,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_LEFT_UP)) {
                        edgeNodeIndex(result, OCT_DIR_LEFT, OCT_DIR_UP,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_LEFT_BACK)) {
                        edgeNodeIndex(result, OCT_DIR_LEFT, OCT_DIR_BACK,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_LEFT_FRONT)) {
                        edgeNodeIndex(result, OCT_DIR_LEFT, OCT_DIR_FRONT,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN)) {
                        edgeNodeIndex(result, OCT_DIR_RIGHT, OCT_DIR_DOWN,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_RIGHT_UP)) {
                        edgeNodeIndex(result, OCT_DIR_RIGHT, OCT_DIR_UP,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_RIGHT_BACK)) {
                        edgeNodeIndex(result, OCT_DIR_RIGHT, OCT_DIR_BACK,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_RIGHT_FRONT)) {
                        edgeNodeIndex(result, OCT_DIR_RIGHT, OCT_DIR_FRONT,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_DOWN_BACK)) {
                        edgeNodeIndex(result, OCT_DIR_DOWN, OCT_DIR_BACK,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_DOWN_FRONT)) {
                        edgeNodeIndex(result, OCT_DIR_DOWN, OCT_DIR_FRONT,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_UP_BACK)) {
                        edgeNodeIndex(result, OCT_DIR_UP, OCT_DIR_BACK,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_UP_FRONT)) {
                        edgeNodeIndex(result, OCT_DIR_UP, OCT_DIR_FRONT,
                                      edgeInternalIndex, true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 edgeInternalIndex.begin(),
                                                 edgeInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_LEFT)) {
                        faceNodesIndex(result, OCT_DIR_LEFT, faceInternalIndex,
                                       true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 faceInternalIndex.begin(),
                                                 faceInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_RIGHT)) {
                        faceNodesIndex(result, OCT_DIR_RIGHT, faceInternalIndex,
                                       true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 faceInternalIndex.begin(),
                                                 faceInternalIndex.end());
                    }
                    if (nodeFlag & (1u << OCT_DIR_DOWN)) {
                        faceNodesIndex(result, OCT_DIR_DOWN, faceInternalIndex,
                                       true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 faceInternalIndex.begin(),
                                                 faceInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_UP)) {
                        faceNodesIndex(result, OCT_DIR_UP, faceInternalIndex,
                                       true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 faceInternalIndex.begin(),
                                                 faceInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_BACK)) {
                        faceNodesIndex(result, OCT_DIR_BACK, faceInternalIndex,
                                       true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 faceInternalIndex.begin(),
                                                 faceInternalIndex.end());
                    }

                    if (nodeFlag & (1u << OCT_DIR_FRONT)) {
                        faceNodesIndex(result, OCT_DIR_FRONT, faceInternalIndex,
                                       true);
                        tmpCornerIndexVec.insert(tmpCornerIndexVec.end(),
                                                 faceInternalIndex.begin(),
                                                 faceInternalIndex.end());
                    }
                }

                for (unsigned int cornerNodeIndex = 0;
                     cornerNodeIndex < tmpCornerIndexVec.size();
                     cornerNodeIndex++) {
                    nodeIndex =
                        m_uiE2NMapping_CG[tmpCornerIndexVec[cornerNodeIndex]];
                    nodeIndex_DG =
                        m_uiE2NMapping_DG[tmpCornerIndexVec[cornerNodeIndex]];
                    dg2eijk(nodeIndex_DG, ownerID, ii_x, jj_y, kk_z);
                    // if(!m_uiActiveRank &&
                    // getDIROfANode(ii_x,jj_y,kk_z)==OCT_DIR_LEFT_DOWN)
                    // std::cout<<"m_uiActiveRank: "<<m_uiActiveRank<<" OwnerID:
                    // "<<ownerID<<" ii_x: "<<ii_x<<" jj_y: "<<jj_y<<" kk_z:
                    // "<<kk_z<<std::endl;
                    nodeFlag = getDIROfANode(ii_x, jj_y, kk_z);

                    for (unsigned int w = 0; w < ownerList->size(); w++) {
                        if ((nodeIndex >= m_uiNodeLocalBegin) &&
                            (nodeIndex < m_uiNodeLocalEnd)) {
                            assert((*ownerList)[w] != m_uiActiveRank);
                            setHintUint =
                                scatterMapNodeSet[(*ownerList)[w]].emplace(
                                    nodeIndex);
                            if (setHintUint.second) {
                                dg2eijk(nodeIndex_DG, ownerID, ii_x, jj_y,
                                        kk_z);
                                x  = m_uiAllElements[ownerID].getX();
                                y  = m_uiAllElements[ownerID].getY();
                                z  = m_uiAllElements[ownerID].getZ();
                                sz = 1u
                                     << (m_uiMaxDepth -
                                         m_uiAllElements[ownerID].getLevel());
                                assert(sz % m_uiElementOrder == 0);
                                tmpKey =
                                    ot::Key((x + ii_x * sz / m_uiElementOrder),
                                            (y + jj_y * sz / m_uiElementOrder),
                                            (z + kk_z * sz / m_uiElementOrder),
                                            m_uiMaxDepth + 1, m_uiDim,
                                            m_uiMaxDepth + 1);
                                tmpKey.addOwner((*setHintUint.first));
                                assert((*setHintUint.first) == nodeIndex);
                                sendNodeOctants[(*ownerList)[w]].push_back(
                                    tmpKey);
#ifdef DEBUG_E2N_MAPPING_SM
                                x  = m_uiAllElements[ownerID].getX();
                                y  = m_uiAllElements[ownerID].getY();
                                z  = m_uiAllElements[ownerID].getZ();
                                sz = 1u
                                     << (m_uiMaxDepth -
                                         m_uiAllElements[ownerID].getLevel());
                                sendNodes[(*ownerList)[w]].push_back(
                                    ot::TreeNode(
                                        (x + ii_x * sz / m_uiElementOrder),
                                        (y + jj_y * sz / m_uiElementOrder),
                                        (z + kk_z * sz / m_uiElementOrder),
                                        m_uiMaxDepth, m_uiDim, m_uiMaxDepth));
                                // if((*ownerList)[w]==2) std::cout<<"
                                // m_uiActiveRank: "<<m_uiActiveRank<<"
                                // SendNode:
                                // "<<sendNodes[(*ownerList)[w]].back()<<" to
                                // rank 2"<<std::endl;
#endif
                                m_uiSendNodeCount[(*ownerList)[w]]++;
                                // std::cout<<"m_uiActiveRank:
                                // "<<m_uiActiveRank<<" R2 SendNodes Executed.
                                // "<<std::endl;
                            }

                        } else {
                            assert(!(nodeIndex >= m_uiNodeLocalBegin &&
                                     nodeIndex < m_uiNodeLocalEnd));
                            assert(ownerID < m_uiAllElements.size());
                            tmpElement = m_uiAllElements[ownerID];
                            assert(tmpElement.getLevel() >=
                                   m_uiAllElements[ownerID].getLevel());
                            nodeFlag  = getDIROfANode(ii_x, jj_y, kk_z);
                            // if(!m_uiActiveRank &&
                            // nodeFlag==(OCT_DIR_LEFT_DOWN)) std::cout<<RED<<"
                            // nodeFlag: "<<nodeFlag<<NRM<<std::endl;
                            // assert(nodeFlag!=OCT_DIR_INTERNAL);
                            tmpEleLev = tmpElement.getLevel();
                            tmpElement.setFlag(
                                (tmpEleLev) |
                                (1u << (nodeFlag + CHAINED_GHOST_OFFSET)));
                            assert(tmpElement.getFlag() >>
                                       (CHAINED_GHOST_OFFSET) &
                                   (1u << nodeFlag));

                            if ((ownerID >= m_uiElementLocalBegin &&
                                 ownerID < m_uiElementLocalEnd)) {
                                hintSK = gEleChained[(*ownerList)[w]].emplace(
                                    gEleChained[(*ownerList)[w]].end(),
                                    SearchKey(tmpElement));
                                hintSK->addOwner((*ownerList)[w]);

                            } else {
                                assert(ownerID != LOOK_UP_TABLE_DEFAULT);
                            }
                        }
                    }
                }
            }
        }

        unsigned int skip = 1;
        for (unsigned int p = 0; p < npes; p++) {
            SFC::seqSort::SFC_treeSort(
                &(*(gEleChained[p].begin())), gEleChained[p].size(),
                tmpSearchKeyVec, tmpSearchKeyVec, tmpSearchKeyVec, m_uiMaxDepth,
                m_uiMaxDepth, rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);
            for (unsigned int e = 0; e < (gEleChained[p].size()); e++) {
                tmpElement = gEleChained[p][e];
                skip       = 1;
                while (((e + skip) < gEleChained[p].size()) &&
                       (gEleChained[p][e] == gEleChained[p][e + skip])) {
                    tmpElement.setFlag((tmpElement.getFlag()) |
                                       (gEleChained[p][e + skip].getFlag()));
                    skip++;
                }
                e += (skip - 1);

                tmpKey = ot::Key(tmpElement);
                tmpKey.addOwner(p);
                tmpKey.setFlag(tmpElement.getFlag());
                ghostElementChained.push_back(tmpKey);
            }
        }
    }

#ifdef DEBUG_E2N_MAPPING_SM

    for (unsigned int p = 0; p < npes; p++) {
        tmpNode.clear();
        SFC::seqSort::SFC_treeSort(&(*(sendNodes[p].begin())),
                                   sendNodes[p].size(), tmpNode, tmpNode,
                                   tmpNode, m_uiMaxDepth, m_uiMaxDepth,
                                   rootNode, ROOT_ROTATION, 1, TS_SORT_ONLY);
        assert(seq::test::isUniqueAndSorted(sendNodes[p]));
    }

    unsigned int *sendNodeCount  = new unsigned int[npes];
    unsigned int *recvNodeCount  = new unsigned int[npes];
    unsigned int *sendNodeOffset = new unsigned int[npes];
    unsigned int *recvNodeOffset = new unsigned int[npes];

    std::vector<ot::TreeNode> sendNodeBuffer;
    std::vector<ot::TreeNode> recvNodeBuffer;

    for (unsigned int p = 0; p < npes; p++) {
        sendNodeCount[p] = scatterMapNodeSet[p].size();
        sendNodeBuffer.insert(sendNodeBuffer.end(), sendNodes[p].begin(),
                              sendNodes[p].end());
    }

    par::Mpi_Alltoall(sendNodeCount, recvNodeCount, 1, comm);

    sendNodeOffset[0] = 0;
    recvNodeOffset[0] = 0;

    omp_par::scan(sendNodeCount, sendNodeOffset, npes);
    omp_par::scan(recvNodeCount, recvNodeOffset, npes);

    recvNodeBuffer.resize(recvNodeCount[npes - 1] + recvNodeOffset[npes - 1]);
    par::Mpi_Alltoallv(&(*(sendNodeBuffer.begin())), (int *)sendNodeCount,
                       (int *)sendNodeOffset, &(*(recvNodeBuffer.begin())),
                       (int *)recvNodeCount, (int *)recvNodeOffset, comm);

    tmpNode.clear();
    SFC::seqSort::SFC_treeSort(&(*(allocatedGNodes.begin())),
                               allocatedGNodes.size(), tmpNode, tmpNode,
                               tmpNode, m_uiMaxDepth, m_uiMaxDepth, rootNode,
                               ROOT_ROTATION, 1, TS_SORT_ONLY);
    assert(seq::test::isUniqueAndSorted(allocatedGNodes));

    /* if(m_uiActiveRank==2)
        {
            for(unsigned int ele=0;ele<recvNodeBuffer.size();ele++)
            {
                std::cout<<"recvNode: "<<recvNodeBuffer[ele]<<std::endl;
            }
        }*/

    tmpNode.clear();
    SFC::seqSort::SFC_treeSort(&(*(recvNodeBuffer.begin())),
                               recvNodeBuffer.size(), tmpNode, tmpNode, tmpNode,
                               m_uiMaxDepth, m_uiMaxDepth, rootNode,
                               ROOT_ROTATION, 1, TS_SORT_ONLY);
    assert(seq::test::isUniqueAndSorted(recvNodeBuffer));

    if (recvNodeBuffer.size() != allocatedGNodes.size()) {
        std::vector<Key> recvNodeKeys;
        recvNodeKeys.resize(recvNodeBuffer.size());
        std::vector<ot::TreeNode> missmatchedNodes;

        for (unsigned int e = 0; e < recvNodeBuffer.size(); e++) {
            unsigned int findIndex =
                std::find(allocatedGNodes.begin(), allocatedGNodes.end(),
                          recvNodeBuffer[e]) -
                allocatedGNodes.begin();
            if (findIndex >= allocatedGNodes.size())
                missmatchedNodes.push_back(recvNodeBuffer[e]);
        }

        treeNodesTovtk(allocatedGNodes, rank, "allGNode");
        treeNodesTovtk(recvNodeBuffer, rank, "recvNodeBuffer");

        treeNodesTovtk(missmatchedNodes, rank, "missmatchedNodes");
    }

    delete[] sendNodeCount;
    delete[] recvNodeCount;
    delete[] sendNodeOffset;
    delete[] recvNodeOffset;

    for (unsigned int p = 0; p < npes; p++) {
        char filename[256];
        sprintf(filename, "sendNode_R2%d", p);
        treeNodesTovtk(sendNodes[p], rank, filename);
        sendNodes[p].clear();
    }
#endif

    delete[] sendChainedGCount;
    delete[] recvChainedGCount;
    delete[] sendChainedGOffset;
    delete[] recvChainedGOffset;
    delete[] recvChainedKeyCount;
    delete[] recvChainedKeyOffset;
    delete[] gEleChained;

#ifdef DEBUG_E2N_MAPPING
    treeNodesTovtk(neighbourElement, m_uiActiveRank, "neighBourElement");
    std::cout << "======== m_uiActiveRank: " << m_uiActiveRank << " : "
              << allocatedGNodes.size() << std::endl;
    treeNodesTovtk(ownerElement1, rank, "owner1Elements");
    treeNodesTovtk(ownerElement2, rank, "owner2Elements");
    treeNodesTovtk(nonLocalNodes, rank, "nonLocal");
    treeNodesTovtk(LocalNodes, rank, "Local");
#endif

    // prepare send recv scatter maps.
    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  (&(*(m_uiRecvNodeOffset.begin()))), npes);

    std::vector<ot::TreeNode> sendNodeBuffer;
    std::vector<ot::TreeNode> recvNodeBuffer;

    recvNodeBuffer.resize(m_uiRecvNodeOffset[npes - 1] +
                          m_uiRecvNodeCount[npes - 1]);
    // sendNodeBuffer.resize(m_uiSendNodeOffset[npes-1]+m_uiSendNodeCount[npes-1]);
    for (unsigned int p = 0; p < npes; p++) {
        /*m_uiScatterMapActualNodeSend.insert(m_uiScatterMapActualNodeSend.end(),
           scatterMapNodeSet[p].begin(), scatterMapNodeSet[p].end());*/
        for (unsigned int k = 0; k < sendNodeOctants[p].size(); k++) {
            assert((sendNodeOctants[p][k].getOwnerList()->size() == 1));
            m_uiScatterMapActualNodeSend.push_back(
                (sendNodeOctants[p][k].getOwnerList()->front()));
            sendNodeBuffer.push_back(sendNodeOctants[p][k]);
        }
        assert(m_uiSendNodeCount[p] == scatterMapNodeSet[p].size());
        assert(scatterMapNodeSet[p].size() == sendNodeOctants[p].size());
        scatterMapNodeSet[p].clear();
        sendNodeOctants[p].clear();
    }

    delete[] scatterMapNodeSet;
    delete[] sendChainedGhostIDSet;
    delete[] sendNodeOctants;

    par::Mpi_Alltoallv(
        &(*(sendNodeBuffer.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))),
        (&(*(recvNodeBuffer.begin()))),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), m_uiCommActive);

    std::vector<ot::Key> recvNodeKeys;
    recvNodeKeys.resize(recvNodeBuffer.size());

    for (unsigned int k = 0; k < recvNodeBuffer.size(); k++) {
        tmpKey = ot::Key(recvNodeBuffer[k]);
        tmpKey.addOwner(k);
        recvNodeKeys[k] = tmpKey;
    }

#ifdef DEBUG_MESH_GENERATION
    treeNodesTovtk(allocatedGhostNodes, m_uiActiveRank, "allocatedNodes");
    treeNodesTovtk(recvNodeBuffer, m_uiActiveRank, "recvNodes");
#endif
    if ((m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]) !=
        allocatedGhostNodes.size()) {
        std::cout << RED << "[Error]"
                  << " m_uiActiveRank: " << m_uiActiveRank
                  << " Total Ghost Elements allocated: "
                  << allocatedGhostNodes.size()
                  << " Number of elements will get recieved: "
                  << (m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1])
                  << NRM << std::endl;
        assert(false);
    }

    /* std::sort(allocatedGhostNodes.begin(),allocatedGhostNodes.end(),OctreeComp<ot::Key>());
     std::sort(recvNodeKeys.begin(),recvNodeKeys.end(),OctreeComp<ot::Key>());*/

    SFC::seqSort::SFC_treeSort(
        &(*(allocatedGhostNodes.begin())), allocatedGhostNodes.size(),
        tmpSearchKeyVec, tmpSearchKeyVec, tmpSearchKeyVec, m_uiMaxDepth + 1,
        m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);
    SFC::seqSort::SFC_treeSort(&(*(recvNodeKeys.begin())), recvNodeKeys.size(),
                               tmpKeys, tmpKeys, tmpKeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

    m_uiScatterMapActualNodeRecv.clear();
    m_uiScatterMapActualNodeRecv.resize(
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]));
    assert(allocatedGhostNodes.size() == recvNodeKeys.size());
    for (unsigned int k = 0; k < allocatedGhostNodes.size(); k++) {
        if (allocatedGhostNodes[k] != recvNodeKeys[k]) {
            std::cout << RED << "[ERROR]: "
                      << "m_uiActiveRank : " << m_uiActiveRank << " allocated ["
                      << k << "]: " << allocatedGhostNodes[k] << " recieved["
                      << k << "]: " << recvNodeKeys[k] << std::endl;
            assert(false);
        }

        // m_uiScatterMapActualNodeRecv[(*(recvNodeKeys[k].getOwnerList()))[0]]=(*(allocatedGhostNodes[k].getOwnerList()))[0];
        // assert(allocatedGhostNodes[k].getOwnerList()->size()==1);
        m_uiScatterMapActualNodeRecv[(*(recvNodeKeys[k].getOwnerList()))[0]] =
            allocatedGhostNodes[k].getOwner();
    }

#ifdef DEBUG_E2N_MAPPING
    MPI_Barrier(comm);
    if (!rank)
        for (unsigned int p = 0; p < npes; p++)
            std::cout << "rank: " << rank << " recv nodes from : [" << p
                      << "]: begin:  " << m_uiRecvNodeOffset[p] << " end: "
                      << (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p])
                      << std::endl;

    MPI_Barrier(comm);
    if (rank == 1)
        for (unsigned int p = 0; p < npes; p++)
            std::cout << "rank: " << rank << " recv nodes from : [" << p
                      << "]: begin:  " << m_uiRecvNodeOffset[p] << " end: "
                      << (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p])
                      << std::endl;

    MPI_Barrier(comm);
#endif
}

void Mesh::computeNodalScatterMap(MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (npes <= 1)
        return;  // nothing to do in the sequential case. (No scatter map
                 // required.)

    unsigned int x, y, z, sz;  // x y z and size of an octant.
    unsigned int ownerID, ii_x, jj_y,
        kk_z;  // DG index to ownerID and ijk decomposition variable.
    unsigned int nodeIndex;

    std::vector<SearchKey> allocatedNodes;
    std::vector<SearchKey> localNodes;
    std::vector<SearchKey>::iterator it;
    std::vector<SearchKey> tmpSkeys;
    std::vector<Key> tmpKeys;

    SearchKey rootSKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);
    Key rootKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);

    ot::TreeNode minMaxLocalNode[2];
    m_uiSplitterNodes = new ot::TreeNode[2 * npes];  // both min and max (nodal)
    std::vector<unsigned int> minMaxIDs;
    minMaxIDs.resize(2 * npes);

    for (unsigned int e = m_uiNodeLocalBegin; e < m_uiNodeLocalEnd; e++) {
        dg2eijk(m_uiCG2DG[e], ownerID, ii_x, jj_y, kk_z);
        x  = m_uiAllElements[ownerID].getX();
        y  = m_uiAllElements[ownerID].getY();
        z  = m_uiAllElements[ownerID].getZ();
        sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
        assert(sz % m_uiElementOrder == 0);
        it = localNodes.emplace(
            localNodes.end(),
            SearchKey((x + ii_x * sz / m_uiElementOrder),
                      (y + jj_y * sz / m_uiElementOrder),
                      (z + kk_z * sz / m_uiElementOrder), m_uiMaxDepth + 1,
                      m_uiDim, m_uiMaxDepth + 1));
        it->addOwner(e);
    }

    SFC::seqSort::SFC_treeSort(&(*(localNodes.begin())), localNodes.size(),
                               tmpSkeys, tmpSkeys, tmpSkeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(localNodes));
    m_uiMaxDepth--;

    minMaxLocalNode[0] = localNodes.front();
    minMaxLocalNode[1] = localNodes.back();

    par::Mpi_Allgather(minMaxLocalNode, m_uiSplitterNodes, 2, comm);

    std::vector<bool> g1Visited;
    g1Visited.resize(m_uiCG2DG.size(), false);

    for (unsigned int ele = 0; ele < m_uiGhostElementRound1Index.size();
         ele++) {
        for (unsigned int node = 0; node < m_uiNpE; node++) {
            nodeIndex =
                m_uiE2NMapping_CG[m_uiGhostElementRound1Index[ele] * m_uiNpE +
                                  node];
            if ((!(nodeIndex >= m_uiNodeLocalBegin &&
                   nodeIndex < m_uiNodeLocalEnd)) &&
                (!g1Visited[nodeIndex])) {
                assert(nodeIndex < g1Visited.size());
                dg2eijk(m_uiCG2DG[nodeIndex], ownerID, ii_x, jj_y, kk_z);
                x  = m_uiAllElements[ownerID].getX();
                y  = m_uiAllElements[ownerID].getY();
                z  = m_uiAllElements[ownerID].getZ();
                sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
                it = allocatedNodes.emplace(
                    allocatedNodes.end(),
                    SearchKey((x + ii_x * sz / m_uiElementOrder),
                              (y + jj_y * sz / m_uiElementOrder),
                              (z + kk_z * sz / m_uiElementOrder),
                              m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                it->addOwner(nodeIndex);
                g1Visited[nodeIndex] = true;
            }
        }
    }

    // number of allocated nodes without splitters. (By construction this should
    // be unique)
    unsigned int allocatedNodeSz = allocatedNodes.size();

    for (unsigned int p = 0; p < 2 * npes; p++)
        allocatedNodes.emplace(allocatedNodes.end(), m_uiSplitterNodes[p]);

    SFC::seqSort::SFC_treeSort(&(*(allocatedNodes.begin())),
                               allocatedNodes.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

    tmpSkeys.clear();
    SearchKey tmpSkey;
    unsigned int skip;
    for (unsigned int e = 0; e < (allocatedNodes.size()); e++) {
        skip    = 1;
        tmpSkey = allocatedNodes[e];
        while (((e + skip) < allocatedNodes.size()) &&
               (allocatedNodes[e] == allocatedNodes[e + skip])) {
            if (allocatedNodes[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(allocatedNodes[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(allocatedNodes, tmpSkeys);
    tmpSkeys.clear();

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(allocatedNodes));
    m_uiMaxDepth--;

    std::vector<ot::Key> splitterNode_keys;
    splitterNode_keys.resize(2 * npes);
    for (unsigned int p = 0; p < 2 * npes; p++)
        splitterNode_keys[p] = ot::Key(m_uiSplitterNodes[p]);

    m_uiMaxDepth++;
    searchKeys(splitterNode_keys, allocatedNodes);
    m_uiMaxDepth--;

    for (unsigned int p = 0; p < 2 * npes; p++) {
        assert(splitterNode_keys[p].getFlag() & OCT_FOUND);
        minMaxIDs[p] = splitterNode_keys[p].getSearchResult();
        assert(minMaxIDs[p] < allocatedNodes.size());
    }

    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();

    m_uiSendNodeCount.resize(npes);
    m_uiRecvNodeCount.resize(npes);
    m_uiSendNodeOffset.resize(npes);
    m_uiRecvNodeOffset.resize(npes);

    for (unsigned int p = 0; p < npes; p++) m_uiSendNodeCount[p] = 0;

    std::vector<ot::TreeNode> sendNodes;
    std::vector<ot::TreeNode> recvNodes;

    for (unsigned int p = 0; p < npes; p++) {
        if (p == rank) continue;
        for (unsigned int e = minMaxIDs[2 * p]; e < (minMaxIDs[2 * p + 1] + 1);
             e++) {
            if (allocatedNodes[e].getOwner() >= 0) {
                sendNodes.push_back(allocatedNodes[e]);
                m_uiSendNodeCount[p]++;
            }
        }
    }

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    assert(sendNodes.size() ==
           (m_uiSendNodeOffset[npes - 1] + m_uiSendNodeCount[npes - 1]));
    recvNodes.resize(m_uiRecvNodeOffset[npes - 1] +
                     m_uiRecvNodeCount[npes - 1]);

    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNodes.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);

    std::vector<Key> recvNodekeys;
    std::vector<Key>::iterator itKey;
    unsigned int sResult;

    /* m_uiMaxDepth++;
     treeNodesTovtk(allocatedNodes,rank,"allocatedNodes");
     treeNodesTovtk(recvNodes,rank,"recvNodes");
     treeNodesTovtk(localNodes,rank,"localNodes");
     m_uiMaxDepth--;*/

    for (unsigned int p = 0; p < npes; p++) m_uiSendNodeCount[p] = 0;

    sendNodes.clear();

    for (unsigned int p = 0; p < npes; p++) {
        recvNodekeys.clear();

        for (unsigned int e = m_uiRecvNodeOffset[p];
             e < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); e++) {
            itKey = recvNodekeys.emplace(recvNodekeys.end(), Key(recvNodes[e]));
            itKey->addOwner(p);
        }

        SFC::seqSearch::SFC_treeSearch(
            &(*(recvNodekeys.begin())), &(*(localNodes.begin())), 0,
            recvNodekeys.size(), 0, localNodes.size(), m_uiMaxDepth + 1,
            m_uiMaxDepth + 1, ROOT_ROTATION);

        for (unsigned int e = 0; e < (recvNodekeys.size()); e++) {
            // NOTE: recvNodes can contain duplicates but recvNodeKeys cannot
            // contain duplicates since we traverse by p.
            if ((recvNodekeys[e].getFlag() & OCT_FOUND)) {
                sResult = recvNodekeys[e].getSearchResult();
                assert(sResult >= 0 && sResult < localNodes.size());
                m_uiScatterMapActualNodeSend.push_back(
                    localNodes[sResult].getOwner());
                sendNodes.push_back(localNodes[sResult]);
                m_uiSendNodeCount[p]++;
            }
        }
    }

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    if (allocatedNodeSz !=
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]))
        std::cout << "rank: " << rank
                  << "[SM Error]: allocated nodes: " << allocatedNodeSz
                  << " received nodes: "
                  << (m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1])
                  << std::endl;

    recvNodes.clear();
    recvNodes.resize(
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]));

    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNodes.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);

    recvNodekeys.clear();
    std::vector<SearchKey> recvNodeSKeys;
    recvNodeSKeys.resize(recvNodes.size());
    for (unsigned int e = 0; e < recvNodes.size(); e++) {
        recvNodeSKeys[e] = SearchKey(recvNodes[e]);
        recvNodeSKeys[e].addOwner(e);
    }

    SFC::seqSort::SFC_treeSort(&(*(recvNodeSKeys.begin())),
                               recvNodeSKeys.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(recvNodeSKeys));
    m_uiMaxDepth--;

    m_uiScatterMapActualNodeRecv.resize(recvNodes.size());
    unsigned int alCount = 0;
    for (int e = 0; e < recvNodeSKeys.size(); e++) {
        if (allocatedNodes[alCount].getOwner() < 0) {
            e--;
            alCount++;
            continue;
        }

        if (allocatedNodes[alCount] != recvNodeSKeys[e]) {
            std::cout << "rank: " << rank << " allocated[" << alCount
                      << "]: " << allocatedNodes[alCount] << " received[" << e
                      << "]: " << recvNodeSKeys[e] << std::endl;
            exit(0);
        }

        m_uiScatterMapActualNodeRecv[recvNodeSKeys[e].getOwner()] =
            allocatedNodes[alCount].getOwner();
        alCount++;
    }

    m_uiCG2DG.clear();
    m_uiDG2CG.clear();
    localNodes.clear();
    allocatedNodes.clear();
}

void Mesh::computeNodalScatterMap1(MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (npes <= 1)
        return;  // nothing to do in the sequential case. (No scatter map
                 // required.)

    unsigned int x, y, z, sz;  // x y z and size of an octant.
    unsigned int ownerID, ii_x, jj_y,
        kk_z;  // DG index to ownerID and ijk decomposition variable.
    unsigned int nodeIndex;

    std::vector<SearchKey> allocatedNodes;
    std::vector<SearchKey> localNodes;
    std::vector<SearchKey>::iterator it;
    std::vector<SearchKey> tmpSkeys;
    std::vector<Key> tmpKeys;

    SearchKey rootSKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);
    Key rootKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);

    ot::TreeNode minMaxLocalNode[2];
    m_uiSplitterNodes = new ot::TreeNode[2 * npes];  // both min and max (nodal)
    std::vector<unsigned int> minMaxIDs;
    minMaxIDs.resize(2 * npes);
    unsigned int domain_max = 1u << (m_uiMaxDepth);

    for (unsigned int e = m_uiNodeLocalBegin; e < m_uiNodeLocalEnd; e++) {
        dg2eijk(m_uiCG2DG[e], ownerID, ii_x, jj_y, kk_z);
        x  = m_uiAllElements[ownerID].getX();
        y  = m_uiAllElements[ownerID].getY();
        z  = m_uiAllElements[ownerID].getZ();
        sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
        assert(sz % m_uiElementOrder == 0);
        it = localNodes.emplace(
            localNodes.end(),
            SearchKey((x + ii_x * sz / m_uiElementOrder),
                      (y + jj_y * sz / m_uiElementOrder),
                      (z + kk_z * sz / m_uiElementOrder), m_uiMaxDepth + 1,
                      m_uiDim, m_uiMaxDepth + 1));
        it->addOwner(e);
    }

    SFC::seqSort::SFC_treeSort(&(*(localNodes.begin())), localNodes.size(),
                               tmpSkeys, tmpSkeys, tmpSkeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(localNodes));
    m_uiMaxDepth--;

    minMaxLocalNode[0] = localNodes.front();
    minMaxLocalNode[1] = localNodes.back();

    par::Mpi_Allgather(minMaxLocalNode, m_uiSplitterNodes, 2, comm);

    std::vector<bool> g1Visited;
    g1Visited.resize(m_uiCG2DG.size(), false);

    for (unsigned int ele = 0; ele < m_uiGhostElementRound1Index.size();
         ele++) {
        for (unsigned int node = 0; node < m_uiNpE; node++) {
            nodeIndex =
                m_uiE2NMapping_CG[m_uiGhostElementRound1Index[ele] * m_uiNpE +
                                  node];
            if ((!(nodeIndex >= m_uiNodeLocalBegin &&
                   nodeIndex < m_uiNodeLocalEnd)) &&
                (!g1Visited[nodeIndex])) {
                assert(nodeIndex < g1Visited.size());
                dg2eijk(m_uiCG2DG[nodeIndex], ownerID, ii_x, jj_y, kk_z);
                x  = m_uiAllElements[ownerID].getX();
                y  = m_uiAllElements[ownerID].getY();
                z  = m_uiAllElements[ownerID].getZ();
                sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
                it = allocatedNodes.emplace(
                    allocatedNodes.end(),
                    SearchKey((x + ii_x * sz / m_uiElementOrder),
                              (y + jj_y * sz / m_uiElementOrder),
                              (z + kk_z * sz / m_uiElementOrder),
                              m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                it->addOwner(nodeIndex);
                g1Visited[nodeIndex] = true;
            }
        }
    }

    // number of allocated nodes without splitters. (By construction this should
    // be unique)
    unsigned int allocatedNodeSz = allocatedNodes.size();

    for (unsigned int p = 0; p < 2 * npes; p++)
        allocatedNodes.emplace(allocatedNodes.end(), m_uiSplitterNodes[p]);

    SFC::seqSort::SFC_treeSort(&(*(allocatedNodes.begin())),
                               allocatedNodes.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

    tmpSkeys.clear();
    SearchKey tmpSkey;
    unsigned int skip;
    for (unsigned int e = 0; e < (allocatedNodes.size()); e++) {
        skip    = 1;
        tmpSkey = allocatedNodes[e];
        while (((e + skip) < allocatedNodes.size()) &&
               (allocatedNodes[e] == allocatedNodes[e + skip])) {
            if (allocatedNodes[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(allocatedNodes[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(allocatedNodes, tmpSkeys);
    tmpSkeys.clear();

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(allocatedNodes));
    m_uiMaxDepth--;

    std::vector<ot::Key> splitterNode_keys;
    splitterNode_keys.resize(2 * npes);
    for (unsigned int p = 0; p < 2 * npes; p++)
        splitterNode_keys[p] = ot::Key(m_uiSplitterNodes[p]);

    m_uiMaxDepth++;
    searchKeys(splitterNode_keys, allocatedNodes);
    m_uiMaxDepth--;

    for (unsigned int p = 0; p < 2 * npes; p++) {
        assert(splitterNode_keys[p].getFlag() & OCT_FOUND);
        minMaxIDs[p] = splitterNode_keys[p].getSearchResult();
        assert(minMaxIDs[p] < allocatedNodes.size());
    }

    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();

    m_uiSendNodeCount.resize(npes);
    m_uiRecvNodeCount.resize(npes);
    m_uiSendNodeOffset.resize(npes);
    m_uiRecvNodeOffset.resize(npes);

    for (unsigned int p = 0; p < npes; p++) m_uiSendNodeCount[p] = 0;

    std::vector<ot::TreeNode> sendNodes;
    std::vector<ot::TreeNode> recvNodes;

    std::vector<ot::TreeNode> sendElements;
    std::vector<ot::TreeNode> recvElements;
    std::vector<ot::TreeNode>::iterator itTN;
    std::vector<ot::TreeNode> tmpOcts;
    ot::TreeNode tmpOct;
    ot::TreeNode rootOct(m_uiDim, m_uiMaxDepth);
    unsigned int nodeFlag;

    for (unsigned int p = 0; p < npes; p++) {
        if (p == rank) continue;
        sendElements.clear();
        for (unsigned int e = minMaxIDs[2 * p]; e < (minMaxIDs[2 * p + 1] + 1);
             e++) {
            if (allocatedNodes[e].getOwner() >= 0) {
                dg2eijk(m_uiCG2DG[allocatedNodes[e].getOwner()], ownerID, ii_x,
                        jj_y, kk_z);
                nodeFlag = getDIROfANode(ii_x, jj_y, kk_z);
                itTN     = sendElements.emplace(sendElements.end(),
                                                m_uiAllElements[ownerID]);
                itTN->setFlag((itTN->getLevel()) |
                              (1u << (nodeFlag + CHAINED_GHOST_OFFSET)));
            }
        }

        SFC::seqSort::SFC_treeSort(&(*(sendElements.begin())),
                                   sendElements.size(), tmpOcts, tmpOcts,
                                   tmpOcts, m_uiMaxDepth, m_uiMaxDepth, rootOct,
                                   ROOT_ROTATION, 1, TS_SORT_ONLY);
        for (unsigned int e = 0; e < (sendElements.size()); e++) {
            itTN = sendNodes.emplace(sendNodes.end(), sendElements[e]);
            skip = 1;
            while (((e + skip) < sendElements.size()) &&
                   (sendElements[e] == sendElements[e + skip])) {
                itTN->setFlag((itTN->getFlag()) |
                              (sendElements[e + skip].getFlag()));
                skip++;
            }
            m_uiSendNodeCount[p]++;

            e += (skip - 1);
        }
    }

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    assert(sendNodes.size() ==
           (m_uiSendNodeOffset[npes - 1] + m_uiSendNodeCount[npes - 1]));
    recvElements.resize(m_uiRecvNodeOffset[npes - 1] +
                        m_uiRecvNodeCount[npes - 1]);
    double t1, t2, t_stat, t_stat_g;
    DendroIntL localSz;
    DendroIntL stat_sz[3];

    localSz = sendNodes.size();

    t1      = MPI_Wtime();
    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvElements.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);

    par::Mpi_Reduce(&localSz, &stat_sz[0], 1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[1], 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[2], 1, MPI_MAX, 0, comm);
    stat_sz[1] = stat_sz[1] / npes;

    if (!rank) std::cout << "a2a_ 1 time max: " << t_stat_g << std::endl;
    if (!rank)
        std::cout << "a2a_ 1 sz  min mean max : " << stat_sz[0] << ", "
                  << stat_sz[1] << ", " << stat_sz[2] << std::endl;

    recvNodes.clear();
    unsigned int hSz;
    unsigned int *recvNodeCount = new unsigned int[npes];
    for (unsigned int p = 0; p < npes; p++) recvNodeCount[p] = 0;

    t1 = MPI_Wtime();
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = m_uiRecvNodeOffset[p];
             e < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); e++) {
            nodeFlag = recvElements[e].getFlag();
            nodeFlag = nodeFlag >> (CHAINED_GHOST_OFFSET);
            sz       = 1u << (m_uiMaxDepth - recvElements[e].getLevel());
            assert((sz % m_uiElementOrder) == 0);
            hSz = sz / m_uiElementOrder;
            x   = recvElements[e].getX();
            y   = recvElements[e].getY();
            z   = recvElements[e].getZ();

            if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN_BACK)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x, y, z, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_BACK)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x + sz, y, z, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_LEFT_UP_BACK)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x, y + sz, z, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_UP_BACK)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x + sz, y + sz, z, m_uiMaxDepth + 1, m_uiDim,
                                 m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN_FRONT)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x, y, z + sz, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_FRONT)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x + sz, y, z + sz, m_uiMaxDepth + 1, m_uiDim,
                                 m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_LEFT_UP_FRONT)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x, y + sz, z + sz, m_uiMaxDepth + 1, m_uiDim,
                                 m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_UP_FRONT)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x + sz, y + sz, z + sz, m_uiMaxDepth + 1,
                                 m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            // face internal.
            if (m_uiElementOrder > 1) {
                if (nodeFlag & (1u << OCT_DIR_LEFT)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int j = 1; j < m_uiElementOrder; j++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x, y + j * hSz, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int j = 1; j < m_uiElementOrder; j++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + sz, y + j * hSz, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }
                if (nodeFlag & (1u << OCT_DIR_DOWN)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_UP)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y + sz, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_BACK)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y + j * hSz, z,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_FRONT)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y + j * hSz, z + sz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y, z + k * hSz, m_uiMaxDepth + 1,
                                         m_uiDim, m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_UP)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y + sz, z + k * hSz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_BACK)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y + j * hSz, z, m_uiMaxDepth + 1,
                                         m_uiDim, m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_FRONT)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y + j * hSz, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y, z + k * hSz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_UP)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y + sz, z + k * hSz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_BACK)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y + j * hSz, z,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_FRONT)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y + j * hSz, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_DOWN_BACK)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y, z, m_uiMaxDepth + 1,
                                         m_uiDim, m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_DOWN_FRONT)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_UP_BACK)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y + sz, z,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_UP_FRONT)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y + sz, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_INTERNAL)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int j = 1; j < m_uiElementOrder; j++)
                            for (unsigned int i = 1; i < m_uiElementOrder; i++)
                                recvNodes.emplace(
                                    recvNodes.end(),
                                    ot::TreeNode(x + i * hSz, y + j * hSz,
                                                 z + k * hSz, m_uiMaxDepth + 1,
                                                 m_uiDim, m_uiMaxDepth + 1));

                    recvNodeCount[p] +=
                        ((m_uiElementOrder - 1) * (m_uiElementOrder - 1) *
                         (m_uiElementOrder - 1));
                }
            }
        }
    }

    for (unsigned int p = 0; p < npes; p++)
        m_uiRecvNodeCount[p] = recvNodeCount[p];

    delete[] recvNodeCount;
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " unzip time: " << t_stat_g << std::endl;

    m_uiRecvNodeOffset[0] = 0;
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    std::vector<Key> recvNodekeys;
    std::vector<Key>::iterator itKey;
    unsigned int sResult;

    /* m_uiMaxDepth++;
     treeNodesTovtk(allocatedNodes,rank,"allocatedNodes");
     treeNodesTovtk(recvNodes,rank,"recvNodes");
     treeNodesTovtk(localNodes,rank,"localNodes");
     m_uiMaxDepth--;*/

    for (unsigned int p = 0; p < npes; p++) m_uiSendNodeCount[p] = 0;

    sendNodes.clear();

    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = m_uiRecvNodeOffset[p];
             e < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); e++) {
            itKey = recvNodekeys.emplace(recvNodekeys.end(), Key(recvNodes[e]));
            itKey->addOwner(p);
        }
    }

    t1 = MPI_Wtime();
    SFC::seqSearch::SFC_treeSearch(
        &(*(recvNodekeys.begin())), &(*(localNodes.begin())), 0,
        recvNodekeys.size(), 0, localNodes.size(), m_uiMaxDepth + 1,
        m_uiMaxDepth + 1, ROOT_ROTATION);
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " search time: " << t_stat_g << std::endl;

    t1 = MPI_Wtime();
    std::vector<unsigned int> *sendResultID =
        new std::vector<unsigned int>[npes];
    for (unsigned int e = 0; e < (recvNodekeys.size()); e++) {
        // NOTE: recvNodes can contain duplicates but recvNodeKeys cannot
        // contain duplicates since we traverse by p.
        if ((recvNodekeys[e].getFlag() & OCT_FOUND)) {
            sResult = recvNodekeys[e].getSearchResult();
            assert(sResult >= 0 && sResult < localNodes.size());
            sendResultID[recvNodekeys[e].getOwnerList()->front()].push_back(
                sResult);
        }
    }

    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = 0; e < sendResultID[p].size(); e++) {
            m_uiScatterMapActualNodeSend.push_back(
                localNodes[sendResultID[p][e]].getOwner());
            sendNodes.push_back(localNodes[sendResultID[p][e]]);
        }
        m_uiSendNodeCount[p] = sendResultID[p].size();
        sendResultID[p].clear();
    }

    delete[] sendResultID;
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " send sm time: " << t_stat_g << std::endl;

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    if (allocatedNodeSz !=
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]))
        std::cout << "rank: " << rank
                  << "[SM Error]: allocated nodes: " << allocatedNodeSz
                  << " received nodes: "
                  << (m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1])
                  << std::endl;

    recvNodes.clear();
    recvNodes.resize(
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]));
    t1 = MPI_Wtime();
    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNodes.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);
    t2      = MPI_Wtime();
    t_stat  = t2 - t1;

    localSz = sendNodes.size();
    par::Mpi_Reduce(&localSz, &stat_sz[0], 1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[1], 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[2], 1, MPI_MAX, 0, comm);
    stat_sz[1] = stat_sz[1] / npes;

    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << "a2a_ 2 time max: " << t_stat_g << std::endl;
    if (!rank)
        std::cout << "a2a_ 2 sz  min mean max : " << stat_sz[0] << ", "
                  << stat_sz[1] << ", " << stat_sz[2] << std::endl;

    recvNodekeys.clear();
    std::vector<SearchKey> recvNodeSKeys;
    recvNodeSKeys.resize(recvNodes.size());
    for (unsigned int e = 0; e < recvNodes.size(); e++) {
        recvNodeSKeys[e] = SearchKey(recvNodes[e]);
        recvNodeSKeys[e].addOwner(e);
    }

    t1 = MPI_Wtime();
    SFC::seqSort::SFC_treeSort(&(*(recvNodeSKeys.begin())),
                               recvNodeSKeys.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " sort time: " << t_stat_g << std::endl;

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(recvNodeSKeys));
    m_uiMaxDepth--;

    t1 = MPI_Wtime();
    m_uiScatterMapActualNodeRecv.resize(recvNodes.size());
    unsigned int alCount = 0;
    for (int e = 0; e < recvNodeSKeys.size(); e++) {
        if (allocatedNodes[alCount].getOwner() < 0) {
            e--;
            alCount++;
            continue;
        }

        if (allocatedNodes[alCount] != recvNodeSKeys[e]) {
            std::cout << "rank: " << rank << " allocated[" << alCount
                      << "]: " << allocatedNodes[alCount] << " received[" << e
                      << "]: " << recvNodeSKeys[e] << std::endl;
            exit(0);
        }

        m_uiScatterMapActualNodeRecv[recvNodeSKeys[e].getOwner()] =
            allocatedNodes[alCount].getOwner();
        alCount++;
    }

    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " sm recv update time: " << t_stat_g << std::endl;

    m_uiCG2DG.clear();
    m_uiDG2CG.clear();
    localNodes.clear();
    allocatedNodes.clear();
}

void Mesh::computeNodalScatterMap2(MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (npes <= 1)
        return;  // nothing to do in the sequential case. (No scatter map
                 // required.)

    unsigned int x, y, z, sz;  // x y z and size of an octant.
    unsigned int ownerID, ii_x, jj_y,
        kk_z;  // DG index to ownerID and ijk decomposition variable.
    unsigned int nodeIndex;
    unsigned int lookUp;

    std::vector<SearchKey> localNodes;
    std::vector<SearchKey>::iterator it;
    std::vector<SearchKey> tmpSkeys;
    std::vector<Key> tmpKeys;

    SearchKey rootSKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);
    Key rootKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);
    const unsigned int domain_max = 1u << (m_uiMaxDepth);

    // 1. generate the local & sort the local nodes. (this should be unique and
    // sorted)
    for (unsigned int e = m_uiNodeLocalBegin; e < m_uiNodeLocalEnd; e++) {
        dg2eijk(m_uiCG2DG[e], ownerID, ii_x, jj_y, kk_z);
        x  = m_uiAllElements[ownerID].getX();
        y  = m_uiAllElements[ownerID].getY();
        z  = m_uiAllElements[ownerID].getZ();
        sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
        assert(sz % m_uiElementOrder == 0);
        it = localNodes.emplace(
            localNodes.end(),
            SearchKey((x + ii_x * sz / m_uiElementOrder),
                      (y + jj_y * sz / m_uiElementOrder),
                      (z + kk_z * sz / m_uiElementOrder), m_uiMaxDepth + 1,
                      m_uiDim, m_uiMaxDepth + 1));
        it->addOwner(e);
    }

    SFC::seqSort::SFC_treeSort(&(*(localNodes.begin())), localNodes.size(),
                               tmpSkeys, tmpSkeys, tmpSkeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(localNodes));
    m_uiMaxDepth--;

    // 2. compute the local splitters. We compute 3 splitters. min max(in
    // m_uiMaxDepth domain), & max (in m_uiMaxDepth +1)
    ot::TreeNode localNodeSpliters[3];
    assert((localNodes[0].getX() < domain_max) &&
           (localNodes[0].getY() < domain_max) &&
           (localNodes[0].getZ() < domain_max));
    localNodeSpliters[0] = localNodes.front();
    localNodeSpliters[2] = localNodes.back();

    for (int i = (localNodes.size() - 1); i >= 0; i--) {
        if ((localNodes[i].getX() < domain_max) &&
            (localNodes[i].getY() < domain_max) &&
            (localNodes[i].getZ() < domain_max)) {
            localNodeSpliters[1] = localNodes[i];
            break;
        }
    }

    // 3. gather all the local splitters.
    m_uiSplitterNodes = new ot::TreeNode[3 * npes];
    par::Mpi_Allgather(localNodeSpliters, m_uiSplitterNodes, 3, comm);

    // 4. compute the ownership (which processor it belongs to) all the ghost
    // elements.
    std::vector<unsigned int> elementOwner;
    elementOwner.resize(m_uiAllElements.size(), rank);

    std::vector<ot::SearchKey> ghostElements;
    std::vector<ot::SearchKey>::iterator itSKey;
    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int e = m_uiElementPostGhostBegin;
         e < m_uiElementPostGhostEnd; e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int p = 0; p < npes; p++)
        ghostElements.emplace(
            ghostElements.end(),
            ot::SearchKey(m_uiLocalSplitterElements[2 * p + 1]));

    SFC::seqSort::SFC_treeSort(&(*(ghostElements.begin())),
                               ghostElements.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth, m_uiMaxDepth, rootSKey,
                               ROOT_ROTATION, 1, TS_SORT_ONLY);

    tmpSkeys.clear();
    SearchKey tmpSkey;
    unsigned int skip;
    for (unsigned int e = 0; e < (ghostElements.size()); e++) {
        skip    = 1;
        tmpSkey = ghostElements[e];
        while (((e + skip) < ghostElements.size()) &&
               (ghostElements[e] == ghostElements[e + skip])) {
            if (ghostElements[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(ghostElements[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(ghostElements, tmpSkeys);
    tmpSkeys.clear();

    unsigned int gCount = 0;
    for (unsigned int p = 0; p < npes; p++) {
        while (
            gCount < ghostElements.size() &&
            (ghostElements[gCount] != m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;

            gCount++;
        }

        if (gCount < ghostElements.size() &&
            (ghostElements[gCount] == m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;
            gCount++;
        }
    }

    std::vector<SearchKey> *allocated_p = new std::vector<SearchKey>[npes];

    std::vector<unsigned int> lookUps;
    lookUps.resize(NUM_CHILDREN);
    std::set<unsigned int> ownerPoc;

    unsigned int nodeFlag;
    std::vector<unsigned int> internalIndex;
    std::vector<unsigned int> faceIndex;    // face internal
    std::vector<unsigned int> edgeIndex;    // edge internal
    std::vector<unsigned int> vertexIndex;  // vertex internal

    unsigned int edge_dir[] = {
        OCT_DIR_LEFT_BACK,  OCT_DIR_LEFT_FRONT,  OCT_DIR_LEFT_DOWN,
        OCT_DIR_LEFT_FRONT, OCT_DIR_RIGHT_BACK,  OCT_DIR_RIGHT_FRONT,
        OCT_DIR_RIGHT_DOWN, OCT_DIR_RIGHT_FRONT, OCT_DIR_UP_BACK,
        OCT_DIR_UP_FRONT,   OCT_DIR_DOWN_BACK,   OCT_DIR_DOWN_FRONT};
    unsigned int face_dir[]   = {OCT_DIR_LEFT, OCT_DIR_RIGHT, OCT_DIR_DOWN,
                                 OCT_DIR_UP,   OCT_DIR_BACK,  OCT_DIR_FRONT};
    unsigned int vertex_dir[] = {
        OCT_DIR_LEFT_DOWN_BACK,  OCT_DIR_RIGHT_DOWN_BACK,
        OCT_DIR_LEFT_UP_BACK,    OCT_DIR_RIGHT_UP_BACK,
        OCT_DIR_LEFT_DOWN_FRONT, OCT_DIR_RIGHT_DOWN_FRONT,
        OCT_DIR_LEFT_UP_FRONT,   OCT_DIR_RIGHT_UP_FRONT};

    std::vector<bool> g1Visited;
    g1Visited.resize(m_uiCG2DG.size(), false);

    std::vector<SearchKey> allocated1;  // allocated nodes where the owner of
                                        // the nodes are undecided.
    std::vector<SearchKey> allocatedNodes;  // actual allocated nodes.

    for (unsigned int ele = 0; ele < m_uiGhostElementRound1Index.size();
         ele++) {
        for (unsigned int node = 0; node < m_uiNpE; node++) {
            nodeIndex =
                m_uiE2NMapping_CG[m_uiGhostElementRound1Index[ele] * m_uiNpE +
                                  node];
            if ((!(nodeIndex >= m_uiNodeLocalBegin &&
                   nodeIndex < m_uiNodeLocalEnd)) &&
                (!g1Visited[nodeIndex])) {
                assert(nodeIndex < g1Visited.size());
                dg2eijk(m_uiCG2DG[nodeIndex], ownerID, ii_x, jj_y, kk_z);
                nodeFlag = getDIROfANode(ii_x, jj_y, kk_z);
                x        = m_uiAllElements[ownerID].getX();
                y        = m_uiAllElements[ownerID].getY();
                z        = m_uiAllElements[ownerID].getZ();
                sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());

                if (nodeFlag ==
                    OCT_DIR_INTERNAL) {  // for internal nodes we can directly
                                         // determine the ownership.
                    it = allocated_p[elementOwner[ownerID]].emplace(
                        allocated_p[elementOwner[ownerID]].end(),
                        SearchKey((x + ii_x * sz / m_uiElementOrder),
                                  (y + jj_y * sz / m_uiElementOrder),
                                  (z + kk_z * sz / m_uiElementOrder),
                                  m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                    it->addOwner(nodeIndex);
                } else {  // for other nodes we use the modified splitter
                          // approach.
                    it = allocated1.emplace(
                        allocated1.end(),
                        SearchKey((x + ii_x * sz / m_uiElementOrder),
                                  (y + jj_y * sz / m_uiElementOrder),
                                  (z + kk_z * sz / m_uiElementOrder),
                                  m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                    it->addOwner(nodeIndex);
                }
                g1Visited[nodeIndex] = true;
            }
        }
    }

    allocatedNodes.clear();
    allocatedNodes.resize(allocated1.size());
    allocatedNodes.assign(allocated1.begin(), allocated1.end());

    for (unsigned int p = 0; p < npes; p++)
        allocatedNodes.insert(allocatedNodes.end(), allocated_p[p].begin(),
                              allocated_p[p].end());

    unsigned int allocatedNodeSz = allocatedNodes.size();

    for (unsigned int p = 0; p < npes; p++) {
        allocated1.emplace(allocated1.end(),
                           SearchKey(m_uiSplitterNodes[3 * p]));
        allocated1.emplace(allocated1.end(),
                           SearchKey(m_uiSplitterNodes[3 * p + 1]));
        allocated1.emplace(allocated1.end(),
                           SearchKey(m_uiSplitterNodes[3 * p + 2]));
    }

    SFC::seqSort::SFC_treeSort(&(*(allocated1.begin())), allocated1.size(),
                               tmpSkeys, tmpSkeys, tmpSkeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

    tmpSkeys.clear();
    for (unsigned int e = 0; e < (allocated1.size()); e++) {
        skip    = 1;
        tmpSkey = allocated1[e];
        while (((e + skip) < allocated1.size()) &&
               (allocated1[e] == allocated1[e + skip])) {
            if (allocated1[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(allocated1[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(allocated1, tmpSkeys);
    tmpSkeys.clear();

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(allocated1));
    m_uiMaxDepth--;

    std::vector<ot::Key> splitterNode_keys;
    splitterNode_keys.resize(3 * npes);
    std::vector<unsigned int> nodeSplitterID;
    nodeSplitterID.resize(3 * npes);

    for (unsigned int p = 0; p < 3 * npes; p++)
        splitterNode_keys[p] = ot::Key(m_uiSplitterNodes[p]);

    m_uiMaxDepth++;
    searchKeys(splitterNode_keys, allocated1);
    m_uiMaxDepth--;

    for (unsigned int p = 0; p < 3 * npes; p++) {
        assert(splitterNode_keys[p].getFlag() & OCT_FOUND);
        nodeSplitterID[p] = splitterNode_keys[p].getSearchResult();
        assert(nodeSplitterID[p] < allocated1.size());
    }

    for (unsigned int p = 0; p < npes; p++) {
        if (p == rank) continue;
        for (unsigned int e = nodeSplitterID[3 * p];
             e < (nodeSplitterID[3 * p + 1] + 1); e++) {
            if (allocated1[e].getOwner() >= 0) {
                assert((allocated1[e].getX() < domain_max) &&
                       (allocated1[e].getY() < domain_max) &&
                       (allocated1[e].getZ() < domain_max));
                allocated_p[p].push_back(allocated1[e]);
            }
        }

        for (unsigned int e = nodeSplitterID[3 * p + 1] + 1;
             e < (nodeSplitterID[3 * p + 2] + 1); e++) {
            if (allocated1[e].getOwner() >= 0 &&
                (!((allocated1[e].getX() < domain_max) &&
                   (allocated1[e].getY() < domain_max) &&
                   (allocated1[e].getZ() < domain_max)))) {
                allocated_p[p].push_back(allocated1[e]);
            }
        }
    }

    allocated1.clear();

    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();

    m_uiSendNodeCount.resize(npes);
    m_uiRecvNodeCount.resize(npes);
    m_uiSendNodeOffset.resize(npes);
    m_uiRecvNodeOffset.resize(npes);

    std::vector<ot::TreeNode> sendNodes;
    std::vector<ot::TreeNode> recvNodes;

    double t1, t2, t_stat, t_stat_g;
    DendroIntL localSz;
    DendroIntL stat_sz[3];

    for (unsigned int p = 0; p < npes; p++) {
        m_uiSendNodeCount[p] = allocated_p[p].size();
        sendNodes.insert(sendNodes.end(), allocated_p[p].begin(),
                         allocated_p[p].end());
        allocated_p[p].clear();
    }

    delete[] allocated_p;

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    assert(sendNodes.size() ==
           (m_uiSendNodeOffset[npes - 1] + m_uiSendNodeCount[npes - 1]));
    recvNodes.resize(m_uiRecvNodeOffset[npes - 1] +
                     m_uiRecvNodeCount[npes - 1]);

    localSz = sendNodes.size();
    t1      = MPI_Wtime();
    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNodes.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);

    par::Mpi_Reduce(&localSz, &stat_sz[0], 1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[1], 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[2], 1, MPI_MAX, 0, comm);
    stat_sz[1] = stat_sz[1] / npes;

    if (!rank) std::cout << "a2a_ 1 time max: " << t_stat_g << std::endl;
    if (!rank)
        std::cout << "a2a_ 1 sz  min mean max : " << stat_sz[0] << ", "
                  << stat_sz[1] << ", " << stat_sz[2] << std::endl;

    std::vector<Key> recvNodekeys;
    std::vector<Key>::iterator itKey;
    unsigned int sResult;

    /* m_uiMaxDepth++;
     treeNodesTovtk(allocatedNodes,rank,"allocatedNodes");
     treeNodesTovtk(recvNodes,rank,"recvNodes");
     treeNodesTovtk(localNodes,rank,"localNodes");
     m_uiMaxDepth--;*/

    for (unsigned int p = 0; p < npes; p++) m_uiSendNodeCount[p] = 0;

    sendNodes.clear();

    for (unsigned int p = 0; p < npes; p++) {
        recvNodekeys.clear();

        for (unsigned int e = m_uiRecvNodeOffset[p];
             e < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); e++) {
            itKey = recvNodekeys.emplace(recvNodekeys.end(), Key(recvNodes[e]));
            itKey->addOwner(p);
        }

        SFC::seqSearch::SFC_treeSearch(
            &(*(recvNodekeys.begin())), &(*(localNodes.begin())), 0,
            recvNodekeys.size(), 0, localNodes.size(), m_uiMaxDepth + 1,
            m_uiMaxDepth + 1, ROOT_ROTATION);

        for (unsigned int e = 0; e < (recvNodekeys.size()); e++) {
            // NOTE: recvNodes can contain duplicates but recvNodeKeys cannot
            // contain duplicates since we traverse by p.
            if ((recvNodekeys[e].getFlag() & OCT_FOUND)) {
                sResult = recvNodekeys[e].getSearchResult();
                assert(sResult >= 0 && sResult < localNodes.size());
                m_uiScatterMapActualNodeSend.push_back(
                    localNodes[sResult].getOwner());
                sendNodes.push_back(localNodes[sResult]);
                m_uiSendNodeCount[p]++;
            } /*else
             {
                 std::cout<<" key: recv form "<<p<<" to rank: "<<rank<<" key:
             "<<recvNodekeys[e]<<" not found "<<std::endl;
             }*/
        }
    }

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    if (allocatedNodeSz !=
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]))
        std::cout << "rank: " << rank
                  << "[SM Error]: allocated nodes: " << allocatedNodeSz
                  << " received nodes: "
                  << (m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1])
                  << std::endl;

    recvNodes.clear();
    recvNodes.resize(
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]));

    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNodes.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);

    recvNodekeys.clear();
    std::vector<SearchKey> recvNodeSKeys;
    recvNodeSKeys.resize(recvNodes.size());
    for (unsigned int e = 0; e < recvNodes.size(); e++) {
        recvNodeSKeys[e] = SearchKey(recvNodes[e]);
        recvNodeSKeys[e].addOwner(e);
    }

    SFC::seqSort::SFC_treeSort(&(*(recvNodeSKeys.begin())),
                               recvNodeSKeys.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);
    SFC::seqSort::SFC_treeSort(&(*(allocatedNodes.begin())),
                               allocatedNodes.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(recvNodeSKeys));
    m_uiMaxDepth--;

    m_uiScatterMapActualNodeRecv.resize(recvNodes.size());
    unsigned int alCount = 0;
    for (int e = 0; e < recvNodeSKeys.size(); e++) {
        if (allocatedNodes[alCount].getOwner() < 0) {
            e--;
            alCount++;
            continue;
        }

        if (allocatedNodes[alCount] != recvNodeSKeys[e]) {
            std::cout << "rank: " << rank << " allocated[" << alCount
                      << "]: " << allocatedNodes[alCount] << " received[" << e
                      << "]: " << recvNodeSKeys[e] << std::endl;
            exit(0);
        }

        m_uiScatterMapActualNodeRecv[recvNodeSKeys[e].getOwner()] =
            allocatedNodes[alCount].getOwner();
        alCount++;
    }

    m_uiCG2DG.clear();
    m_uiDG2CG.clear();
    localNodes.clear();
    allocatedNodes.clear();
}

void Mesh::computeNodalScatterMap3(MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (npes <= 1)
        return;  // nothing to do in the sequential case. (No scatter map
                 // required.)

    unsigned int x, y, z, sz;  // x y z and size of an octant.
    unsigned int ownerID, ii_x, jj_y,
        kk_z;  // DG index to ownerID and ijk decomposition variable.
    unsigned int ownerID1;
    unsigned int nodeIndex;
    unsigned int lookUp;

    double t1, t2, t_stat, t_stat_g;
    DendroIntL localSz;
    DendroIntL stat_sz[3];

    std::vector<SearchKey> localNodes;
    std::vector<SearchKey>::iterator it;
    std::vector<SearchKey> tmpSkeys;
    std::vector<Key> tmpKeys;

    SearchKey rootSKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);
    Key rootKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth + 1);
    const unsigned int domain_max = 1u << (m_uiMaxDepth);

    t1                            = MPI_Wtime();
    // 1. generate the local & sort the local nodes. (this should be unique and
    // sorted)
    for (unsigned int e = m_uiNodeLocalBegin; e < m_uiNodeLocalEnd; e++) {
        dg2eijk(m_uiCG2DG[e], ownerID, ii_x, jj_y, kk_z);
        x  = m_uiAllElements[ownerID].getX();
        y  = m_uiAllElements[ownerID].getY();
        z  = m_uiAllElements[ownerID].getZ();
        sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
        assert(sz % m_uiElementOrder == 0);
        it = localNodes.emplace(
            localNodes.end(),
            SearchKey((x + ii_x * sz / m_uiElementOrder),
                      (y + jj_y * sz / m_uiElementOrder),
                      (z + kk_z * sz / m_uiElementOrder), m_uiMaxDepth + 1,
                      m_uiDim, m_uiMaxDepth + 1));
        it->addOwner(e);
    }

    SFC::seqSort::SFC_treeSort(&(*(localNodes.begin())), localNodes.size(),
                               tmpSkeys, tmpSkeys, tmpSkeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " local node generation + sort time (max) (s): "
                  << t_stat_g << std::endl;

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(localNodes));
    m_uiMaxDepth--;

    // 2. compute the local splitters. We compute 3 splitters. min max(in
    // m_uiMaxDepth domain), & max (in m_uiMaxDepth +1)
    ot::TreeNode localNodeSpliters[3];
    assert((localNodes[0].getX() < domain_max) &&
           (localNodes[0].getY() < domain_max) &&
           (localNodes[0].getZ() < domain_max));
    localNodeSpliters[0] = localNodes.front();
    localNodeSpliters[2] = localNodes.back();

    for (int i = (localNodes.size() - 1); i >= 0; i--) {
        if ((localNodes[i].getX() < domain_max) &&
            (localNodes[i].getY() < domain_max) &&
            (localNodes[i].getZ() < domain_max)) {
            localNodeSpliters[1] = localNodes[i];
            break;
        }
    }

    // 3. gather all the local splitters.
    m_uiSplitterNodes = new ot::TreeNode[3 * npes];
    par::Mpi_Allgather(localNodeSpliters, m_uiSplitterNodes, 3, comm);

    t1 = MPI_Wtime();
    // 4. compute the ownership (which processor it belongs to) all the ghost
    // elements.
    std::vector<unsigned int> elementOwner;
    elementOwner.resize(m_uiAllElements.size(), rank);

    std::vector<ot::SearchKey> ghostElements;
    std::vector<ot::SearchKey>::iterator itSKey;
    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int e = m_uiElementPostGhostBegin;
         e < m_uiElementPostGhostEnd; e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int p = 0; p < npes; p++)
        ghostElements.emplace(
            ghostElements.end(),
            ot::SearchKey(m_uiLocalSplitterElements[2 * p + 1]));

    SFC::seqSort::SFC_treeSort(&(*(ghostElements.begin())),
                               ghostElements.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth, m_uiMaxDepth, rootSKey,
                               ROOT_ROTATION, 1, TS_SORT_ONLY);

    tmpSkeys.clear();
    SearchKey tmpSkey;
    unsigned int skip;
    for (unsigned int e = 0; e < (ghostElements.size()); e++) {
        skip    = 1;
        tmpSkey = ghostElements[e];
        while (((e + skip) < ghostElements.size()) &&
               (ghostElements[e] == ghostElements[e + skip])) {
            if (ghostElements[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(ghostElements[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(ghostElements, tmpSkeys);
    tmpSkeys.clear();

    unsigned int gCount = 0;
    for (unsigned int p = 0; p < npes; p++) {
        while (
            gCount < ghostElements.size() &&
            (ghostElements[gCount] != m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;

            gCount++;
        }

        if (gCount < ghostElements.size() &&
            (ghostElements[gCount] == m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;
            gCount++;
        }
    }
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " ghost element ownership build time (max) (s): "
                  << t_stat_g << std::endl;

    std::vector<SearchKey> *allocated_p = new std::vector<SearchKey>[npes];

    std::vector<unsigned int> lookUps;
    lookUps.resize(NUM_CHILDREN);
    std::set<unsigned int> ownerPoc;

    unsigned int nodeFlag;
    std::vector<unsigned int> internalIndex;
    std::vector<unsigned int> faceIndex;    // face internal
    std::vector<unsigned int> edgeIndex;    // edge internal
    std::vector<unsigned int> vertexIndex;  // vertex internal

    /*unsigned int
    edge_dir[]={OCT_DIR_LEFT_BACK,OCT_DIR_LEFT_FRONT,OCT_DIR_LEFT_DOWN,OCT_DIR_LEFT_FRONT,OCT_DIR_RIGHT_BACK,OCT_DIR_RIGHT_FRONT,OCT_DIR_RIGHT_DOWN,OCT_DIR_RIGHT_FRONT,OCT_DIR_UP_BACK,OCT_DIR_UP_FRONT,OCT_DIR_DOWN_BACK,OCT_DIR_DOWN_FRONT};
    unsigned int
    face_dir[]={OCT_DIR_LEFT,OCT_DIR_RIGHT,OCT_DIR_DOWN,OCT_DIR_UP,OCT_DIR_BACK,OCT_DIR_FRONT};
    unsigned int
    vertex_dir[]={OCT_DIR_LEFT_DOWN_BACK,OCT_DIR_RIGHT_DOWN_BACK,OCT_DIR_LEFT_UP_BACK,OCT_DIR_RIGHT_UP_BACK,
    OCT_DIR_LEFT_DOWN_FRONT,OCT_DIR_RIGHT_DOWN_FRONT,OCT_DIR_LEFT_UP_FRONT,OCT_DIR_RIGHT_UP_FRONT};*/

    std::vector<bool> g1Visited;
    g1Visited.resize(m_uiCG2DG.size(), false);

    std::vector<SearchKey> allocated1;  // allocated nodes where the owner of
                                        // the nodes are undecided.
    std::vector<SearchKey> allocatedNodes;  // actual allocated nodes.
    t1 = MPI_Wtime();
    for (unsigned int ele = 0; ele < m_uiGhostElementRound1Index.size();
         ele++) {
        for (unsigned int node = 0; node < m_uiNpE; node++) {
            nodeIndex =
                m_uiE2NMapping_CG[m_uiGhostElementRound1Index[ele] * m_uiNpE +
                                  node];
            if ((!(nodeIndex >= m_uiNodeLocalBegin &&
                   nodeIndex < m_uiNodeLocalEnd)) &&
                (!g1Visited[nodeIndex])) {
                assert(nodeIndex < g1Visited.size());
                dg2eijk(m_uiCG2DG[nodeIndex], ownerID, ii_x, jj_y, kk_z);
                nodeFlag = getDIROfANode(ii_x, jj_y, kk_z);
                x        = m_uiAllElements[ownerID].getX();
                y        = m_uiAllElements[ownerID].getY();
                z        = m_uiAllElements[ownerID].getZ();
                sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());

                if (nodeFlag ==
                    OCT_DIR_INTERNAL) {  // for internal nodes we can directly
                                         // determine the ownership.
                    it = allocated_p[elementOwner[ownerID]].emplace(
                        allocated_p[elementOwner[ownerID]].end(),
                        SearchKey((x + ii_x * sz / m_uiElementOrder),
                                  (y + jj_y * sz / m_uiElementOrder),
                                  (z + kk_z * sz / m_uiElementOrder),
                                  m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                    it->addOwner(nodeIndex);
                } else {  // for other nodes we use the modified splitter
                          // approach.
                    it = allocated1.emplace(
                        allocated1.end(),
                        SearchKey((x + ii_x * sz / m_uiElementOrder),
                                  (y + jj_y * sz / m_uiElementOrder),
                                  (z + kk_z * sz / m_uiElementOrder),
                                  m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                    it->addOwner(nodeIndex);
                }
                g1Visited[nodeIndex] = true;
            }
        }
    }

    allocatedNodes.clear();
    allocatedNodes.resize(allocated1.size());
    allocatedNodes.assign(allocated1.begin(), allocated1.end());

    for (unsigned int p = 0; p < npes; p++)
        allocatedNodes.insert(allocatedNodes.end(), allocated_p[p].begin(),
                              allocated_p[p].end());

    SFC::seqSort::SFC_treeSort(&(*(allocatedNodes.begin())),
                               allocatedNodes.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

    unsigned int allocatedNodeSz = allocatedNodes.size();
    t2                           = MPI_Wtime();

    t_stat                       = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " allocated node generation + sort time (max) (s): "
                  << t_stat_g << std::endl;

    for (unsigned int p = 0; p < npes; p++) {
        allocated1.emplace(allocated1.end(),
                           SearchKey(m_uiSplitterNodes[3 * p]));
        allocated1.emplace(allocated1.end(),
                           SearchKey(m_uiSplitterNodes[3 * p + 1]));
        allocated1.emplace(allocated1.end(),
                           SearchKey(m_uiSplitterNodes[3 * p + 2]));
    }

    t1 = MPI_Wtime();

    SFC::seqSort::SFC_treeSort(&(*(allocated1.begin())), allocated1.size(),
                               tmpSkeys, tmpSkeys, tmpSkeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

    tmpSkeys.clear();
    for (unsigned int e = 0; e < (allocated1.size()); e++) {
        skip    = 1;
        tmpSkey = allocated1[e];
        while (((e + skip) < allocated1.size()) &&
               (allocated1[e] == allocated1[e + skip])) {
            if (allocated1[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(allocated1[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(allocated1, tmpSkeys);
    tmpSkeys.clear();

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(allocated1));
    m_uiMaxDepth--;

    std::vector<ot::Key> splitterNode_keys;
    splitterNode_keys.resize(3 * npes);
    std::vector<unsigned int> nodeSplitterID;
    nodeSplitterID.resize(3 * npes);

    for (unsigned int p = 0; p < 3 * npes; p++)
        splitterNode_keys[p] = ot::Key(m_uiSplitterNodes[p]);

    m_uiMaxDepth++;
    searchKeys(splitterNode_keys, allocated1);
    m_uiMaxDepth--;

    for (unsigned int p = 0; p < 3 * npes; p++) {
        assert(splitterNode_keys[p].getFlag() & OCT_FOUND);
        nodeSplitterID[p] = splitterNode_keys[p].getSearchResult();
        assert(nodeSplitterID[p] < allocated1.size());
    }

    for (unsigned int p = 0; p < npes; p++) {
        if (p == rank) continue;
        for (unsigned int e = nodeSplitterID[3 * p];
             e < (nodeSplitterID[3 * p + 1] + 1); e++) {
            if (allocated1[e].getOwner() >= 0) {
                assert((allocated1[e].getX() < domain_max) &&
                       (allocated1[e].getY() < domain_max) &&
                       (allocated1[e].getZ() < domain_max));
                allocated_p[p].push_back(allocated1[e]);
            }
        }

        for (unsigned int e = nodeSplitterID[3 * p + 1] + 1;
             e < (nodeSplitterID[3 * p + 2] + 1); e++) {
            if (allocated1[e].getOwner() >= 0 &&
                (!((allocated1[e].getX() < domain_max) &&
                   (allocated1[e].getY() < domain_max) &&
                   (allocated1[e].getZ() < domain_max)))) {
                allocated_p[p].push_back(allocated1[e]);
            }
        }
    }

    allocated1.clear();
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " allocated1 ownership computation (max) (s): " << t_stat_g
                  << std::endl;

    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();

    m_uiSendNodeCount.resize(npes);
    m_uiRecvNodeCount.resize(npes);
    m_uiSendNodeOffset.resize(npes);
    m_uiRecvNodeOffset.resize(npes);

    std::vector<ot::TreeNode> sendNodes;
    std::vector<ot::TreeNode> recvNodes;

    std::vector<ot::TreeNode> sendElements;
    std::vector<ot::TreeNode> recvElements;
    std::vector<ot::TreeNode> tmpOcts;
    ot::TreeNode rootOct(m_uiDim, m_uiMaxDepth);

    std::vector<ot::TreeNode>::iterator itTN;

    assert(allocated_p[rank].size() == 0);

    t1 = MPI_Wtime();
    for (unsigned int p = 0; p < npes; p++) {
        m_uiSendNodeCount[p] = 0;
        if (p == rank) continue;

        sendElements.clear();
        for (unsigned int e = 0; e < allocated_p[p].size(); e++) {
            assert(allocated_p[p][e].getOwner() >= 0);
            dg2eijk(m_uiCG2DG[allocated_p[p][e].getOwner()], ownerID, ii_x,
                    jj_y, kk_z);
            nodeFlag = getDIROfANode(ii_x, jj_y, kk_z);
            itTN     = sendElements.emplace(sendElements.end(),
                                            m_uiAllElements[ownerID]);
            itTN->setFlag((itTN->getLevel()) |
                          (1u << (nodeFlag + CHAINED_GHOST_OFFSET)));
        }

        allocated_p[p].clear();

        SFC::seqSort::SFC_treeSort(&(*(sendElements.begin())),
                                   sendElements.size(), tmpOcts, tmpOcts,
                                   tmpOcts, m_uiMaxDepth, m_uiMaxDepth, rootOct,
                                   ROOT_ROTATION, 1, TS_SORT_ONLY);
        for (unsigned int e = 0; e < (sendElements.size()); e++) {
            itTN = sendNodes.emplace(sendNodes.end(), sendElements[e]);
            skip = 1;
            while (((e + skip) < sendElements.size()) &&
                   (sendElements[e] == sendElements[e + skip])) {
                itTN->setFlag((itTN->getFlag()) |
                              (sendElements[e + skip].getFlag()));
                skip++;
            }
            m_uiSendNodeCount[p]++;
            e += (skip - 1);
        }
    }

    delete[] allocated_p;

    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " allocated node zip time (max) (s): " << t_stat_g
                  << std::endl;

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    assert(sendNodes.size() ==
           (m_uiSendNodeOffset[npes - 1] + m_uiSendNodeCount[npes - 1]));
    recvElements.resize(m_uiRecvNodeOffset[npes - 1] +
                        m_uiRecvNodeCount[npes - 1]);

    localSz = sendNodes.size();
    t1      = MPI_Wtime();
    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvElements.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);

    par::Mpi_Reduce(&localSz, &stat_sz[0], 1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[1], 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[2], 1, MPI_MAX, 0, comm);
    stat_sz[1] = stat_sz[1] / npes;

    if (!rank) std::cout << "a2a_ 1 time max: " << t_stat_g << std::endl;
    if (!rank)
        std::cout << "a2a_ 1 sz  min mean max : " << stat_sz[0] << ", "
                  << stat_sz[1] << ", " << stat_sz[2] << std::endl;

    recvNodes.clear();
    unsigned int hSz;
    unsigned int *recvNodeCount = new unsigned int[npes];
    for (unsigned int p = 0; p < npes; p++) recvNodeCount[p] = 0;

    t1 = MPI_Wtime();
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = m_uiRecvNodeOffset[p];
             e < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); e++) {
            nodeFlag = recvElements[e].getFlag();
            nodeFlag = nodeFlag >> (CHAINED_GHOST_OFFSET);
            sz       = 1u << (m_uiMaxDepth - recvElements[e].getLevel());
            assert((sz % m_uiElementOrder) == 0);
            hSz = sz / m_uiElementOrder;
            x   = recvElements[e].getX();
            y   = recvElements[e].getY();
            z   = recvElements[e].getZ();

            if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN_BACK)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x, y, z, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_BACK)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x + sz, y, z, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_LEFT_UP_BACK)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x, y + sz, z, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_UP_BACK)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x + sz, y + sz, z, m_uiMaxDepth + 1, m_uiDim,
                                 m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN_FRONT)) {
                recvNodes.emplace(recvNodes.end(),
                                  ot::TreeNode(x, y, z + sz, m_uiMaxDepth + 1,
                                               m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN_FRONT)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x + sz, y, z + sz, m_uiMaxDepth + 1, m_uiDim,
                                 m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_LEFT_UP_FRONT)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x, y + sz, z + sz, m_uiMaxDepth + 1, m_uiDim,
                                 m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            if (nodeFlag & (1u << OCT_DIR_RIGHT_UP_FRONT)) {
                recvNodes.emplace(
                    recvNodes.end(),
                    ot::TreeNode(x + sz, y + sz, z + sz, m_uiMaxDepth + 1,
                                 m_uiDim, m_uiMaxDepth + 1));
                recvNodeCount[p]++;
            }

            // face internal.
            if (m_uiElementOrder > 1) {
                if (nodeFlag & (1u << OCT_DIR_LEFT)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int j = 1; j < m_uiElementOrder; j++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x, y + j * hSz, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int j = 1; j < m_uiElementOrder; j++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + sz, y + j * hSz, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }
                if (nodeFlag & (1u << OCT_DIR_DOWN)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_UP)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y + sz, z + k * hSz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_BACK)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y + j * hSz, z,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_FRONT)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        for (unsigned int i = 1; i < m_uiElementOrder; i++)
                            recvNodes.emplace(
                                recvNodes.end(),
                                ot::TreeNode(x + i * hSz, y + j * hSz, z + sz,
                                             m_uiMaxDepth + 1, m_uiDim,
                                             m_uiMaxDepth + 1));
                    recvNodeCount[p] +=
                        (m_uiElementOrder - 1) * (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_DOWN)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y, z + k * hSz, m_uiMaxDepth + 1,
                                         m_uiDim, m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_UP)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y + sz, z + k * hSz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_BACK)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y + j * hSz, z, m_uiMaxDepth + 1,
                                         m_uiDim, m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_LEFT_FRONT)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x, y + j * hSz, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_DOWN)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y, z + k * hSz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_UP)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y + sz, z + k * hSz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_BACK)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y + j * hSz, z,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_RIGHT_FRONT)) {
                    for (unsigned int j = 1; j < m_uiElementOrder; j++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + sz, y + j * hSz, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_DOWN_BACK)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y, z, m_uiMaxDepth + 1,
                                         m_uiDim, m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_DOWN_FRONT)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_UP_BACK)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y + sz, z,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_UP_FRONT)) {
                    for (unsigned int i = 1; i < m_uiElementOrder; i++)
                        recvNodes.emplace(
                            recvNodes.end(),
                            ot::TreeNode(x + i * hSz, y + sz, z + sz,
                                         m_uiMaxDepth + 1, m_uiDim,
                                         m_uiMaxDepth + 1));
                    recvNodeCount[p] += (m_uiElementOrder - 1);
                }

                if (nodeFlag & (1u << OCT_DIR_INTERNAL)) {
                    for (unsigned int k = 1; k < m_uiElementOrder; k++)
                        for (unsigned int j = 1; j < m_uiElementOrder; j++)
                            for (unsigned int i = 1; i < m_uiElementOrder; i++)
                                recvNodes.emplace(
                                    recvNodes.end(),
                                    ot::TreeNode(x + i * hSz, y + j * hSz,
                                                 z + k * hSz, m_uiMaxDepth + 1,
                                                 m_uiDim, m_uiMaxDepth + 1));

                    recvNodeCount[p] +=
                        ((m_uiElementOrder - 1) * (m_uiElementOrder - 1) *
                         (m_uiElementOrder - 1));
                }
            }
        }
    }

    for (unsigned int p = 0; p < npes; p++)
        m_uiRecvNodeCount[p] = recvNodeCount[p];

    delete[] recvNodeCount;

    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " allocated node unzip time (max) (s): " << t_stat_g
                  << std::endl;

    m_uiRecvNodeOffset[0] = 0;
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    std::vector<Key> recvNodekeys;
    std::vector<Key>::iterator itKey;
    unsigned int sResult;

    for (unsigned int p = 0; p < npes; p++) m_uiSendNodeCount[p] = 0;

    sendNodes.clear();

    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = m_uiRecvNodeOffset[p];
             e < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); e++) {
            itKey = recvNodekeys.emplace(recvNodekeys.end(), Key(recvNodes[e]));
            itKey->addOwner(p);
        }
    }

    t1 = MPI_Wtime();
    SFC::seqSearch::SFC_treeSearch(
        &(*(recvNodekeys.begin())), &(*(localNodes.begin())), 0,
        recvNodekeys.size(), 0, localNodes.size(), m_uiMaxDepth + 1,
        m_uiMaxDepth + 1, ROOT_ROTATION);
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " search time: " << t_stat_g << std::endl;

    t1 = MPI_Wtime();
    std::vector<unsigned int> *sendResultID =
        new std::vector<unsigned int>[npes];

    for (unsigned int e = 0; e < (recvNodekeys.size()); e++) {
        // NOTE: recvNodes can contain duplicates but recvNodeKeys cannot
        // contain duplicates since we traverse by p.
        if ((recvNodekeys[e].getFlag() & OCT_FOUND)) {
            sResult = recvNodekeys[e].getSearchResult();
            assert(sResult >= 0 && sResult < localNodes.size());
            sendResultID[recvNodekeys[e].getOwnerList()->front()].push_back(
                sResult);
        }
    }

    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = 0; e < sendResultID[p].size(); e++) {
            m_uiScatterMapActualNodeSend.push_back(
                localNodes[sendResultID[p][e]].getOwner());
            sendNodes.push_back(localNodes[sendResultID[p][e]]);
        }
        m_uiSendNodeCount[p] = sendResultID[p].size();
        sendResultID[p].clear();
    }

    delete[] sendResultID;
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " send sm time: " << t_stat_g << std::endl;

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    if (allocatedNodeSz !=
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]))
        std::cout << "rank: " << rank
                  << "[SM Error]: allocated nodes: " << allocatedNodeSz
                  << " received nodes: "
                  << (m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1])
                  << std::endl;

    recvNodes.clear();
    recvNodes.resize(
        (m_uiRecvNodeOffset[npes - 1] + m_uiRecvNodeCount[npes - 1]));

    localSz = sendNodes.size();

    t1      = MPI_Wtime();
    par::Mpi_Alltoallv(
        &(*(sendNodes.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNodes.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);

    par::Mpi_Reduce(&localSz, &stat_sz[0], 1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[1], 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(&localSz, &stat_sz[2], 1, MPI_MAX, 0, comm);
    stat_sz[1] = stat_sz[1] / npes;

    if (!rank)
        std::cout << "a2a_ 2 (nodal) time max: " << t_stat_g << std::endl;
    if (!rank)
        std::cout << "a2a_ 2 (nodal) sz  min mean max : " << stat_sz[0] << ", "
                  << stat_sz[1] << ", " << stat_sz[2] << std::endl;

    recvNodekeys.clear();
    std::vector<SearchKey> recvNodeSKeys;
    recvNodeSKeys.resize(recvNodes.size());
    for (unsigned int e = 0; e < recvNodes.size(); e++) {
        recvNodeSKeys[e] = SearchKey(recvNodes[e]);
        recvNodeSKeys[e].addOwner(e);
    }

    t1 = MPI_Wtime();
    SFC::seqSort::SFC_treeSort(&(*(recvNodeSKeys.begin())),
                               recvNodeSKeys.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);
    t2     = MPI_Wtime();

    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << "recvnodes sort time  max (s): " << t_stat_g << std::endl;

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(recvNodeSKeys));
    m_uiMaxDepth--;

    m_uiScatterMapActualNodeRecv.resize(recvNodes.size());
    t1                   = MPI_Wtime();
    unsigned int alCount = 0;
    for (int e = 0; e < recvNodeSKeys.size(); e++) {
        if (allocatedNodes[alCount].getOwner() < 0) {
            e--;
            alCount++;
            continue;
        }

        if (allocatedNodes[alCount] != recvNodeSKeys[e]) {
            std::cout << "rank: " << rank << " allocated[" << alCount
                      << "]: " << allocatedNodes[alCount] << " received[" << e
                      << "]: " << recvNodeSKeys[e] << std::endl;
            exit(0);
        }

        m_uiScatterMapActualNodeRecv[recvNodeSKeys[e].getOwner()] =
            allocatedNodes[alCount].getOwner();
        alCount++;
    }

    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << "recv sm time  max (s): " << t_stat_g << std::endl;

    m_uiCG2DG.clear();
    m_uiDG2CG.clear();
    localNodes.clear();
    allocatedNodes.clear();
}

void Mesh::computeNodalScatterMap4(MPI_Comm comm) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    if (npes <= 1)
        return;  // nothing to do in the sequential case. (No scatter map
                 // required.)

#ifdef PROFILE_SM
    double t1, t2, t_stat, t_stat_g;
    DendroIntL localSz;
    DendroIntL stat_sz[3];
#endif

    unsigned int x, y, z, sz;  // x y z and size of an octant.
    unsigned int ownerID, ii_x, jj_y,
        kk_z;  // DG index to ownerID and ijk decomposition variable.
    unsigned int lookUp;
    unsigned int nodeIndex;
    unsigned int nodeFlag;

    std::vector<ot::TreeNode> sendTNElements;
    std::vector<ot::TreeNode> recvTNElements;
    std::vector<ot::Node> sendNNodal;
    std::vector<ot::Node> recvNNodal;

    ot::TreeNode rootTN(m_uiDim, m_uiMaxDepth);
    ot::Node rootNN(m_uiDim, m_uiMaxDepth);
    ot::SearchKey rootSKey(m_uiDim, m_uiMaxDepth);

    std::vector<ot::TreeNode>::iterator itTN;
    std::vector<ot::Node>::iterator itNN;
    std::vector<ot::SearchKey>::iterator itSK;
    std::vector<ot::Key>::iterator itKK;

    std::vector<SearchKey> tmpSkeys;
    std::vector<ot::Node> tmpNN;

    ot::Node tmpNode;

    m_uiSendNodeCount.resize(npes);
    m_uiRecvNodeCount.resize(npes);
    m_uiSendNodeOffset.resize(npes);
    m_uiRecvNodeOffset.resize(npes);

// 1. compute the ownership (which processor it belongs to) all the ghost
// elements.
#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    std::vector<unsigned int> elementOwner;
    elementOwner.resize(m_uiAllElements.size(), rank);

    std::vector<ot::SearchKey> ghostElements;
    std::vector<ot::SearchKey>::iterator itSKey;
    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int e = m_uiElementPostGhostBegin;
         e < m_uiElementPostGhostEnd; e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int p = 0; p < npes; p++)
        ghostElements.emplace(
            ghostElements.end(),
            ot::SearchKey(m_uiLocalSplitterElements[2 * p + 1]));

    SFC::seqSort::SFC_treeSort(&(*(ghostElements.begin())),
                               ghostElements.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth, m_uiMaxDepth, rootSKey,
                               ROOT_ROTATION, 1, TS_SORT_ONLY);

    tmpSkeys.clear();
    SearchKey tmpSkey;
    unsigned int skip;
    for (unsigned int e = 0; e < (ghostElements.size()); e++) {
        skip    = 1;
        tmpSkey = ghostElements[e];
        while (((e + skip) < ghostElements.size()) &&
               (ghostElements[e] == ghostElements[e + skip])) {
            if (ghostElements[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(ghostElements[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(ghostElements, tmpSkeys);
    tmpSkeys.clear();

    unsigned int gCount = 0;
    for (unsigned int p = 0; p < npes; p++) {
        while (
            gCount < ghostElements.size() &&
            (ghostElements[gCount] != m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;

            gCount++;
        }

        if (gCount < ghostElements.size() &&
            (ghostElements[gCount] == m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;
            gCount++;
        }
    }

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " ghost element ownership build time (max) (s): "
                  << t_stat_g << std::endl;
#endif

    std::vector<unsigned int> *sendIDs = new std::vector<unsigned int>[npes];

    for (unsigned int e = 0; e < m_uiGhostElementRound1Index.size(); e++)
        sendIDs[elementOwner[m_uiGhostElementRound1Index[e]]].push_back(
            m_uiGhostElementRound1Index[e]);

    sendTNElements.clear();
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = 0; e < sendIDs[p].size(); e++)
            sendTNElements.push_back(m_uiAllElements[sendIDs[p][e]]);

        m_uiSendNodeCount[p] = sendIDs[p].size();
        sendIDs[p].clear();
    }

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    recvTNElements.resize(m_uiRecvNodeOffset[npes - 1] +
                          m_uiRecvNodeCount[npes - 1]);

    par::Mpi_Alltoallv(
        &(*(sendTNElements.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvTNElements.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);

    // 2. generate recvTNElement keys and send the local nodes to the owner.
    std::vector<ot::Key> recvTNElem_keys;
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = m_uiRecvNodeOffset[p];
             e < (m_uiRecvNodeOffset[p] + m_uiRecvNodeCount[p]); e++) {
            itKK = recvTNElem_keys.emplace(recvTNElem_keys.end(),
                                           ot::Key(recvTNElements[e]));
            itKK->addOwner(p);
        }
    }

    SFC::seqSearch::SFC_treeSearch(
        &(*(recvTNElem_keys.begin())), &(*(m_uiAllElements.begin())), 0,
        recvTNElem_keys.size(), m_uiElementLocalBegin, m_uiElementLocalEnd,
        m_uiMaxDepth, m_uiMaxDepth, ROOT_ROTATION);
    // local nodes is done.

    // for bdy nodes.
    // 3. create local bdy nodes.
    std::vector<ot::Node> localBdy;
    std::vector<ot::SearchKey> localBdy1;  // original local Bdy1.

    for (unsigned int e = m_uiNodeLocalBegin; e < m_uiNodeLocalEnd; e++) {
        dg2eijk(m_uiCG2DG[e], ownerID, ii_x, jj_y, kk_z);
        if (getDIROfANode(ii_x, jj_y, kk_z) != OCT_DIR_INTERNAL) {
            x  = m_uiAllElements[ownerID].getX();
            y  = m_uiAllElements[ownerID].getY();
            z  = m_uiAllElements[ownerID].getZ();
            sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
            assert(sz % m_uiElementOrder == 0);

            itSK = localBdy1.emplace(
                localBdy1.end(),
                SearchKey((x + ii_x * sz / m_uiElementOrder),
                          (y + jj_y * sz / m_uiElementOrder),
                          (z + kk_z * sz / m_uiElementOrder), m_uiMaxDepth + 1,
                          m_uiDim, m_uiMaxDepth + 1));
            itSK->addOwner(e);
        }
    }

    localBdy.resize(localBdy1.size());
    for (unsigned int e = 0; e < localBdy1.size(); e++) {
        localBdy[e] = localBdy1[e];
        localBdy[e].setOwner(rank);
    }

    SFC::seqSort::SFC_treeSort(&(*(localBdy1.begin())), localBdy1.size(),
                               tmpSkeys, tmpSkeys, tmpSkeys, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootSKey, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

// 3.a par sort of local bdy nodes.
#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    SFC::parSort::SFC_treeSort(localBdy, tmpNN, tmpNN, tmpNN, 0.1,
                               m_uiMaxDepth + 1, rootNN, ROOT_ROTATION, 1,
                               TS_SORT_ONLY, 2, comm);

#ifdef DEBUG_SM
    m_uiMaxDepth++;
    treeNodesTovtk(localBdy, rank, "localBdy");
    m_uiMaxDepth--;
#endif

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " par::sort local_bdy nodes (max) (s): " << t_stat_g
                  << std::endl;
#endif

    m_uiMaxDepth++;
    assert(par::test::isUniqueAndSorted(localBdy, comm));
    m_uiMaxDepth--;

    ot::TreeNode minMax[2];
    minMax[0]         = localBdy.front();
    minMax[1]         = localBdy.back();

    m_uiSplitterNodes = new ot::TreeNode[2 * npes];
    par::Mpi_Allgather(minMax, m_uiSplitterNodes, 2, comm);

    std::vector<bool> g1Visited;
    g1Visited.resize(m_uiCG2DG.size(), false);

    // 4. Generated allocated nodes with boundary allocation.

    std::vector<ot::Node> allocatedBdy;  // allocated nodes where the owner of
                                         // the nodes are undecided.
    std::vector<SearchKey> allocatedNodes;  // actual allocated nodes.

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    for (unsigned int ele = 0; ele < m_uiGhostElementRound1Index.size();
         ele++) {
        for (unsigned int node = 0; node < m_uiNpE; node++) {
            nodeIndex =
                m_uiE2NMapping_CG[m_uiGhostElementRound1Index[ele] * m_uiNpE +
                                  node];
            if ((!(nodeIndex >= m_uiNodeLocalBegin &&
                   nodeIndex < m_uiNodeLocalEnd)) &&
                (!g1Visited[nodeIndex])) {
                assert(nodeIndex < g1Visited.size());
                dg2eijk(m_uiCG2DG[nodeIndex], ownerID, ii_x, jj_y, kk_z);
                nodeFlag = getDIROfANode(ii_x, jj_y, kk_z);
                x        = m_uiAllElements[ownerID].getX();
                y        = m_uiAllElements[ownerID].getY();
                z        = m_uiAllElements[ownerID].getZ();
                sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
                itSK = allocatedNodes.emplace(
                    allocatedNodes.end(),
                    SearchKey((x + ii_x * sz / m_uiElementOrder),
                              (y + jj_y * sz / m_uiElementOrder),
                              (z + kk_z * sz / m_uiElementOrder),
                              m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                itSK->addOwner(nodeIndex);

                if (nodeFlag !=
                    OCT_DIR_INTERNAL) {  // for internal nodes we can directly
                                         // determine the ownership.
                    itNN = allocatedBdy.emplace(
                        allocatedBdy.end(),
                        SearchKey((x + ii_x * sz / m_uiElementOrder),
                                  (y + jj_y * sz / m_uiElementOrder),
                                  (z + kk_z * sz / m_uiElementOrder),
                                  m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
                    itNN->setOwner(rank);
                }
                g1Visited[nodeIndex] = true;
            }
        }
    }

    const unsigned int totAllocated = allocatedNodes.size();
    SFC::seqSort::SFC_treeSort(&(*(allocatedNodes.begin())),
                               allocatedNodes.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

#ifdef DEBUG_SM
    m_uiMaxDepth++;
    treeNodesTovtk(allocatedBdy, rank, "allocatedBdy");
    m_uiMaxDepth--;
#endif

    for (unsigned int p = 0; p < 2 * npes; p++)
        allocatedBdy.emplace(allocatedBdy.end(), m_uiSplitterNodes[p]);

    SFC::seqSort::SFC_treeSort(&(*(allocatedBdy.begin())), allocatedBdy.size(),
                               tmpNN, tmpNN, tmpNN, m_uiMaxDepth + 1,
                               m_uiMaxDepth + 1, rootNN, ROOT_ROTATION, 1,
                               TS_SORT_ONLY);

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " allocated node generation + sort time (max) (s): "
                  << t_stat_g << std::endl;
#endif

    // remove duplicates for allocated bdy octants.
    tmpNN.clear();
    for (unsigned int e = 0; e < (allocatedBdy.size()); e++) {
        skip    = 1;
        tmpNode = allocatedBdy[e];
        while (((e + skip) < allocatedBdy.size()) &&
               (allocatedBdy[e] == allocatedBdy[e + skip])) {
            if (allocatedBdy[e + skip].getOwner() >= 0)
                tmpNode.setOwner(allocatedBdy[e + skip].getOwner());
            skip++;
        }
        tmpNN.push_back(tmpNode);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(allocatedBdy, tmpNN);
    tmpNN.clear();

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(allocatedBdy));
    m_uiMaxDepth--;

    std::vector<ot::Key> splitterNode_keys;
    splitterNode_keys.resize(2 * npes);
    std::vector<unsigned int> nodeSplitterID;
    nodeSplitterID.resize(2 * npes);

    for (unsigned int p = 0; p < 2 * npes; p++)
        splitterNode_keys[p] = ot::Key(m_uiSplitterNodes[p]);

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(splitterNode_keys));
    m_uiMaxDepth--;

    m_uiMaxDepth++;
    searchKeys(splitterNode_keys, allocatedBdy);
    m_uiMaxDepth--;

    for (unsigned int p = 0; p < 2 * npes; p++) {
        assert(splitterNode_keys[p].getFlag() & OCT_FOUND);
        assert(allocatedBdy[splitterNode_keys[p].getSearchResult()] ==
               splitterNode_keys[p]);
        nodeSplitterID[p] = splitterNode_keys[p].getSearchResult();
        assert(nodeSplitterID[p] < allocatedBdy.size());
    }

    // 5. send bdy allocated nodes to the correct processor (this is according
    // to the par::sort of localBdy).
    sendNNodal.clear();
    unsigned int sBegin;
    unsigned int sEnd;
    for (unsigned int p = 0; p < npes; p++) {
        sBegin               = nodeSplitterID[2 * p];
        sEnd                 = nodeSplitterID[2 * p + 1] + 1;
        m_uiSendNodeCount[p] = 0;
        for (unsigned int e = sBegin; e < sEnd; e++) {
            if (allocatedBdy[e].getOwner() >= 0) {
                sendNNodal.push_back(allocatedBdy[e]);
                m_uiSendNodeCount[p]++;
            }
        }
    }

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    recvNNodal.resize(m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1]);

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    par::Mpi_Alltoallv(
        &(*(sendNNodal.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNNodal.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " R1 all2all (max) (s): " << t_stat_g << std::endl;
#endif

    std::vector<ot::Key> recvNodal_keys;
    recvNodal_keys.resize(recvNNodal.size());
    for (unsigned int e = 0; e < recvNNodal.size(); e++) {
        recvNodal_keys[e] = ot::Key(recvNNodal[e]);
        recvNodal_keys[e].addOwner(e);
    }

#ifdef DEBUG_SM
    m_uiMaxDepth++;
    treeNodesTovtk(recvNodal_keys, rank, "recvNNodal");
    m_uiMaxDepth--;
#endif

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif
    SFC::seqSearch::SFC_treeSearch(
        &(*(recvNodal_keys.begin())), &(*(localBdy.begin())), 0,
        recvNodal_keys.size(), 0, localBdy.size(), m_uiMaxDepth + 1,
        m_uiMaxDepth + 1, ROOT_ROTATION);

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " R1 seq::search max (s): " << t_stat_g << std::endl;
#endif

    sendNNodal.clear();
    unsigned int result;

    for (unsigned int p = 0; p < npes; p++) sendIDs[p].clear();

// all the recv nodes should be found in localBdy
#ifdef DEBUG_SM
    std::vector<ot::TreeNode> missingKeys;
#endif

    for (unsigned int e = 0; e < recvNodal_keys.size(); e++) {
        if (!(recvNodal_keys[e].getFlag() & OCT_FOUND)) {
#ifdef DEBUG_SM
            unsigned int found =
                std::find(localBdy.begin(), localBdy.end(), recvNodal_keys[e]) -
                localBdy.begin();
            missingKeys.push_back(recvNodal_keys[e]);
            m_uiMaxDepth++;
            std::cout << "rank: " << rank
                      << " recvNodalKey : " << recvNodal_keys[e]
                      << " not found: status: "
                      << (recvNodal_keys[e] >= m_uiSplitterNodes[32] &&
                          recvNodal_keys[e] <= m_uiSplitterNodes[33])
                      << "found: " << found << " of " << localBdy.size()
                      << std::endl;
            m_uiMaxDepth--;
            continue;
#endif

            std::cout << "rank: " << m_uiActiveRank
                      << " SM4 Error: Allocated key<" << recvNodal_keys[e]
                      << " not found. " << std::endl;
            exit(0);
        }

        assert(recvNodal_keys[e].getFlag() & OCT_FOUND);
        result  = recvNodal_keys[e].getSearchResult();
        ownerID = recvNodal_keys[e].getOwnerList()->front();
        sendIDs[localBdy[result].getOwner()].push_back(ownerID);
    }

#ifdef DEBUG_SM
    m_uiMaxDepth++;
    if (missingKeys.size()) treeNodesTovtk(missingKeys, rank, "missingKeys");
    m_uiMaxDepth--;
#endif

    sendNNodal.clear();
    // now send the recv nodes to actual correct proceesor (based on localBdy1)
    for (unsigned int p = 0; p < npes; p++) {
        for (unsigned int e = 0; e < sendIDs[p].size(); e++)
            sendNNodal.push_back(recvNNodal[sendIDs[p][e]]);

        m_uiSendNodeCount[p] = sendIDs[p].size();
        sendIDs[p].clear();
    }

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    recvNNodal.resize(m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1]);
#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    par::Mpi_Alltoallv(
        &(*(sendNNodal.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvNNodal.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " R2 all2all max (s): " << t_stat_g << std::endl;
#endif

    recvNodal_keys.resize(recvNNodal.size());
    for (unsigned int e = 0; e < recvNNodal.size(); e++) {
        recvNodal_keys[e] = ot::Key(recvNNodal[e]);
        recvNodal_keys[e].addOwner(e);
    }

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    SFC::seqSearch::SFC_treeSearch(
        &(*(recvNodal_keys.begin())), &(*(localBdy1.begin())), 0,
        recvNodal_keys.size(), 0, localBdy1.size(), m_uiMaxDepth + 1,
        m_uiMaxDepth + 1, ROOT_ROTATION);

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " R2 seq::search max (s): " << t_stat_g << std::endl;
#endif

    sendNNodal.clear();

    m_uiScatterMapActualNodeSend.clear();
    m_uiScatterMapActualNodeRecv.clear();
    sendTNElements.clear();

    for (unsigned int p = 0; p < npes; p++) sendIDs[p].clear();

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif
    // put internal nodes to the scattermap.
    for (unsigned int e = 0; e < recvTNElem_keys.size(); e++) {
        assert(recvTNElem_keys[e].getFlag() & OCT_FOUND);
        result  = recvTNElem_keys[e].getSearchResult();
        ownerID = recvTNElem_keys[e].getOwnerList()->front();
        assert(ownerID != rank);
        assert(result >= m_uiElementLocalBegin &&
               result <= m_uiElementLocalEnd);

        for (unsigned int k = 1; k < m_uiElementOrder; k++)
            for (unsigned int j = 1; j < m_uiElementOrder; j++)
                for (unsigned int i = 1; i < m_uiElementOrder; i++)
                    sendIDs[ownerID].push_back(
                        m_uiE2NMapping_CG[result * m_uiNpE +
                                          k * (m_uiElementOrder + 1) *
                                              (m_uiElementOrder + 1) +
                                          j * (m_uiElementOrder + 1) + i]);
    }

    // all the recv nodal should be found in localBdy1
    for (unsigned int e = 0; e < recvNodal_keys.size(); e++) {
        assert(recvNodal_keys[e].getFlag() & OCT_FOUND);
        result = recvNodal_keys[e].getSearchResult();
        ownerID =
            recvNNodal[recvNodal_keys[e].getOwnerList()->front()].getOwner();
        assert(ownerID != rank);
        sendIDs[ownerID].push_back(localBdy1[result].getOwner());
    }

    for (unsigned int p = 0; p < npes; p++) {
        // Note: This should be unique.
        std::sort(sendIDs[p].begin(), sendIDs[p].end());
        /*if(!rank) std::cout<<" rank : "<<rank<<" p: "<<p<<" bf rmd :
        "<<sendIDs[p].size()<<std::endl;
        sendIDs[p].erase(std::unique(sendIDs[p].begin(),sendIDs[p].end()),sendIDs[p].end());
        if(!rank) std::cout<<" rank : "<<rank<<" p: "<<p<<" af rmd :
        "<<sendIDs[p].size()<<std::endl;*/
        for (unsigned int e = 0; e < sendIDs[p].size(); e++) {
            // if(!rank) std::cout<<" e: "<<e<<" val:
            // "<<sendIDs[p][e]<<std::endl;
            dg2eijk(m_uiCG2DG[sendIDs[p][e]], ownerID, ii_x, jj_y, kk_z);
            x  = m_uiAllElements[ownerID].getX();
            y  = m_uiAllElements[ownerID].getY();
            z  = m_uiAllElements[ownerID].getZ();
            sz = 1u << (m_uiMaxDepth - m_uiAllElements[ownerID].getLevel());
            sendTNElements.emplace(
                sendTNElements.end(),
                ot::TreeNode((x + ii_x * sz / m_uiElementOrder),
                             (y + jj_y * sz / m_uiElementOrder),
                             (z + kk_z * sz / m_uiElementOrder),
                             m_uiMaxDepth + 1, m_uiDim, m_uiMaxDepth + 1));
            m_uiScatterMapActualNodeSend.push_back(sendIDs[p][e]);
        }

        m_uiSendNodeCount[p] = sendIDs[p].size();
        sendIDs[p].clear();
    }

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " send sm setup max (s): " << t_stat_g << std::endl;
#endif

    par::Mpi_Alltoall(&(*(m_uiSendNodeCount.begin())),
                      &(*(m_uiRecvNodeCount.begin())), 1, comm);

    m_uiSendNodeOffset[0] = 0;
    m_uiRecvNodeOffset[0] = 0;

    omp_par::scan(&(*(m_uiSendNodeCount.begin())),
                  &(*(m_uiSendNodeOffset.begin())), npes);
    omp_par::scan(&(*(m_uiRecvNodeCount.begin())),
                  &(*(m_uiRecvNodeOffset.begin())), npes);

    recvTNElements.resize(m_uiRecvNodeOffset[npes - 1] +
                          m_uiRecvNodeCount[npes - 1]);

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    par::Mpi_Alltoallv(
        &(*(sendTNElements.begin())), (int *)(&(*(m_uiSendNodeCount.begin()))),
        (int *)(&(*(m_uiSendNodeOffset.begin()))), &(*(recvTNElements.begin())),
        (int *)(&(*(m_uiRecvNodeCount.begin()))),
        (int *)(&(*(m_uiRecvNodeOffset.begin()))), comm);

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << " final all2all max (s): " << t_stat_g << std::endl;
#endif

    if (totAllocated != recvTNElements.size()) {
        std::cout << "rank: " << rank
                  << "[SM Error]: allocated nodes: " << totAllocated
                  << " received nodes: "
                  << (m_uiRecvNodeOffset[npes - 1] +
                      m_uiRecvNodeCount[npes - 1])
                  << std::endl;
        exit(0);
    }

    std::vector<SearchKey> recvNodeSKeys;
    recvNodeSKeys.resize(recvTNElements.size());
    for (unsigned int e = 0; e < recvTNElements.size(); e++) {
        recvNodeSKeys[e] = SearchKey(recvTNElements[e]);
        recvNodeSKeys[e].addOwner(e);
    }

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    SFC::seqSort::SFC_treeSort(&(*(recvNodeSKeys.begin())),
                               recvNodeSKeys.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth + 1, m_uiMaxDepth + 1,
                               rootSKey, ROOT_ROTATION, 1, TS_SORT_ONLY);

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank)
        std::cout << " recv node seq::sort max (s): " << t_stat_g << std::endl;
#endif

    m_uiMaxDepth++;
    assert(seq::test::isUniqueAndSorted(recvNodeSKeys));
    m_uiMaxDepth--;

#ifdef PROFILE_SM
    t1 = MPI_Wtime();
#endif

    m_uiScatterMapActualNodeRecv.resize(recvTNElements.size());
    unsigned int alCount = 0;
    for (int e = 0; e < recvNodeSKeys.size(); e++) {
        if (allocatedNodes[alCount].getOwner() < 0) {
            e--;
            alCount++;
            continue;
        }

        if (allocatedNodes[alCount] != recvNodeSKeys[e]) {
            std::cout << "rank: " << rank << " allocated[" << alCount
                      << "]: " << allocatedNodes[alCount] << " received[" << e
                      << "]: " << recvNodeSKeys[e] << std::endl;
            exit(0);
        }

        m_uiScatterMapActualNodeRecv[recvNodeSKeys[e].getOwner()] =
            allocatedNodes[alCount].getOwner();
        alCount++;
    }

#ifdef PROFILE_SM
    t2     = MPI_Wtime();
    t_stat = t2 - t1;
    par::Mpi_Reduce(&t_stat, &t_stat_g, 1, MPI_MAX, 0, comm);
    if (!rank) std::cout << "recv sm time  max (s): " << t_stat_g << std::endl;
#endif

    delete[] sendIDs;
}

void Mesh::flagBlockGhostDependancies() {
    if (m_uiIsActive) {
        const ot::TreeNode *const pNodes = m_uiAllElements.data();

        for (unsigned int blk = 0; blk < m_uiLocalBlockList.size(); blk++) {
            const ot::TreeNode blkNode = m_uiLocalBlockList[blk].getBlockNode();
            const unsigned int regLev =
                m_uiLocalBlockList[blk].getRegularGridLev();
            BlockType btype;
            bool is_blk_internal_independent = true;
            for (unsigned int elem : m_uiLocalBlockList[blk]) {
                const unsigned int ei =
                    (pNodes[elem].getX() - blkNode.getX()) >>
                    (m_uiMaxDepth - regLev);
                const unsigned int ej =
                    (pNodes[elem].getY() - blkNode.getY()) >>
                    (m_uiMaxDepth - regLev);
                const unsigned int ek =
                    (pNodes[elem].getZ() - blkNode.getZ()) >>
                    (m_uiMaxDepth - regLev);

                const unsigned int emin = 0;
                const unsigned int emax =
                    (1u << (regLev - blkNode.getLevel())) - 1;

                if (!((ei == emin) || (ej == emin) || (ek == emin) ||
                      (ei == emax) || (ej == emax) || (ek == emax))) {
                    // this should be true since internal elements should be
                    // independent.
                    assert(this->getElementType(elem) == EType::INDEPENDENT);
                    continue;
                }

                if (this->getElementType(elem) == EType::W_DEPENDENT) {
                    is_blk_internal_independent = false;
                    break;
                }
            }

            bool is_blk_independent;
            unsigned int lookup;
            if (is_blk_internal_independent) {
                is_blk_independent = true;
                for (unsigned int elem : m_uiLocalBlockList[blk]) {
                    const unsigned int ei =
                        (pNodes[elem].getX() - blkNode.getX()) >>
                        (m_uiMaxDepth - regLev);
                    const unsigned int ej =
                        (pNodes[elem].getY() - blkNode.getY()) >>
                        (m_uiMaxDepth - regLev);
                    const unsigned int ek =
                        (pNodes[elem].getZ() - blkNode.getZ()) >>
                        (m_uiMaxDepth - regLev);

                    const unsigned int emin = 0;
                    const unsigned int emax =
                        (1u << (regLev - blkNode.getLevel())) - 1;

                    if (ei == emin) {
                        // OCT_DIR_LEFT
                        lookup =
                            m_uiE2EMapping[elem * NUM_FACES + OCT_DIR_LEFT];
                        if (lookup != LOOK_UP_TABLE_DEFAULT &&
                            (this->getElementType(lookup) ==
                             EType::W_DEPENDENT))
                            is_blk_independent = false;
                    }

                    if (ei == emax) {  // OCT_DIR_RIGHT
                        lookup =
                            m_uiE2EMapping[elem * NUM_FACES + OCT_DIR_RIGHT];
                        if (lookup != LOOK_UP_TABLE_DEFAULT &&
                            (this->getElementType(lookup) ==
                             EType::W_DEPENDENT))
                            is_blk_independent = false;
                    }

                    if (ej == emin) {  // OCT_DIR_DOWN
                        lookup =
                            m_uiE2EMapping[elem * NUM_FACES + OCT_DIR_DOWN];
                        if (lookup != LOOK_UP_TABLE_DEFAULT &&
                            (this->getElementType(lookup) ==
                             EType::W_DEPENDENT))
                            is_blk_independent = false;
                    }

                    if (ej == emax) {  // OCT_DIR_UP
                        lookup = m_uiE2EMapping[elem * NUM_FACES + OCT_DIR_UP];
                        if (lookup != LOOK_UP_TABLE_DEFAULT &&
                            (this->getElementType(lookup) ==
                             EType::W_DEPENDENT))
                            is_blk_independent = false;
                    }

                    if (ek == emin) {  // OCT_DIR_BACK
                        lookup =
                            m_uiE2EMapping[elem * NUM_FACES + OCT_DIR_BACK];
                        if (lookup != LOOK_UP_TABLE_DEFAULT &&
                            (this->getElementType(lookup) ==
                             EType::W_DEPENDENT))
                            is_blk_independent = false;
                    }

                    if (ek == emax) {  // OCT_DIR_FRONT
                        lookup =
                            m_uiE2EMapping[elem * NUM_FACES + OCT_DIR_FRONT];
                        if (lookup != LOOK_UP_TABLE_DEFAULT &&
                            (this->getElementType(lookup) ==
                             EType::W_DEPENDENT))
                            is_blk_independent = false;
                    }

                    if (!is_blk_independent) break;
                }

                if (is_blk_independent) {
                    // check diagonal edges.
                    const std::vector<unsigned int> blkDiagMap =
                        m_uiLocalBlockList[blk].getBlk2DiagMap_vec();
                    const std::vector<unsigned int> blkVertMap =
                        m_uiLocalBlockList[blk].getBlk2VertexMap_vec();
                    const unsigned int blk_ele_1d =
                        m_uiLocalBlockList[blk].getElemSz1D();

                    for (unsigned int dir = 0; dir < NUM_EDGES; dir++) {
                        for (unsigned int k = 0; k < blk_ele_1d; k++) {
                            if (blkDiagMap[dir * (2 * blk_ele_1d) + 2 * k +
                                           0] !=
                                blkDiagMap[dir * (2 * blk_ele_1d) + 2 * k +
                                           1]) {
                                if ((this->getElementType(
                                         blkDiagMap[2 * k + 0]) ==
                                     EType::W_DEPENDENT) ||
                                    (this->getElementType(
                                         blkDiagMap[2 * k + 1]) ==
                                     EType::W_DEPENDENT))
                                    is_blk_independent = false;

                            } else if (blkDiagMap[dir * (2 * blk_ele_1d) +
                                                  2 * k + 0] !=
                                       LOOK_UP_TABLE_DEFAULT) {
                                if ((this->getElementType(
                                         blkDiagMap[2 * k + 0]) ==
                                     EType::W_DEPENDENT))
                                    is_blk_independent = false;
                            }
                        }

                        if (!is_blk_independent) break;
                    }

                    if (is_blk_independent) {
                        // check vertices.
                        for (unsigned int k = 0; k < blkVertMap.size(); k++) {
                            if ((blkVertMap[k] != LOOK_UP_TABLE_DEFAULT) &&
                                (this->getElementType(blkVertMap[k]) ==
                                 EType::W_DEPENDENT)) {
                                is_blk_independent = false;
                                break;
                            }
                        }
                    }
                }

            } else
                is_blk_independent = false;

            if (is_blk_independent)
                m_uiLocalBlockList[blk].setBlkType(
                    BlockType::UNZIP_INDEPENDENT);
            else
                m_uiLocalBlockList[blk].setBlkType(BlockType::UNZIP_DEPENDENT);

            // std::cout<<" blk: "<<blk<<" node :
            // "<<m_uiLocalBlockList[blk].getBlockNode()<<" independent :
            // "<<is_blk_independent<<std::endl;
        }
    }
}

void Mesh::performBlocksSetup(unsigned int cLev, unsigned int *tag,
                              unsigned int tsz) {
    dendro::logger::debug(dendro::logger::Scope{"MESH"},
                          "Now building the blocks from the adaptive mesh");
    m_uiIsBlockSetup  = true;
    m_uiCoarsetBlkLev = cLev;
    m_uiLocalBlockList.clear();

    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    // assumes that E2E and E2N mapping is done and m_uiAllElements should be
    // sorted otherwise this will chnage the order of elements in
    // m_uiAllElements.
    assert(seq::test::isUniqueAndSorted(m_uiAllElements));
    octree2BlockDecomposition(m_uiAllElements, m_uiLocalBlockList, m_uiMaxDepth,
                              m_uiDmin, m_uiDmax, m_uiElementLocalBegin,
                              m_uiElementLocalEnd, m_uiElementOrder,
                              m_uiCoarsetBlkLev, tag, tsz);
    assert(ot::test::isBlockListValid(m_uiAllElements, m_uiLocalBlockList,
                                      m_uiDmin, m_uiDmax, m_uiElementLocalBegin,
                                      m_uiElementLocalEnd));

    std::vector<DendroIntL> blkSz;
    std::vector<DendroIntL> blkSzOffset;

    // construct element to block map.
    m_uiE2BlkMap.resize(m_uiNumLocalElements, LOOK_UP_TABLE_DEFAULT);

    blkSz.resize(m_uiLocalBlockList.size());
    blkSzOffset.resize(m_uiLocalBlockList.size());

    for (unsigned int k = 0; k < m_uiLocalBlockList.size(); k++)
        blkSz[k] =
            m_uiLocalBlockList[k]
                .getAlignedBlockSz();  // m_uiLocalBlockList[k].get1DArraySize()*m_uiLocalBlockList[k].get1DArraySize()*m_uiLocalBlockList[k].get1DArraySize();

    blkSzOffset[0] = 0;
    omp_par::scan(&(*(blkSz.begin())), &(*(blkSzOffset.begin())),
                  m_uiLocalBlockList.size());

    for (unsigned int k = 0; k < m_uiLocalBlockList.size(); k++)
        m_uiLocalBlockList[k].setOffset(blkSzOffset[k]);

    m_uiUnZippedVecSz =
        blkSzOffset[m_uiLocalBlockList.size() - 1] + blkSz.back();

    std::cout << m_uiGlobalRank
              << ": ORIGINAL UNZIPPED VEC SIZE: " << m_uiUnZippedVecSz
              << std::endl;

    const unsigned int dmin = 0;
    const unsigned int dmax = 1u << (m_uiMaxDepth);
    ot::TreeNode blkNode;

    std::vector<ot::Key> blkKeys;
    std::vector<ot::SearchKey> blkSkeys;

    unsigned int sz;
    unsigned int regLev;
    unsigned int blkElem_1D;

    std::vector<unsigned int> *ownerList;
    std::vector<unsigned int> *directionList;
    unsigned int result;

    for (unsigned int e = 0; e < m_uiLocalBlockList.size(); e++) {
        blkNode = m_uiLocalBlockList[e].getBlockNode();

        // update the element to block map.
        for (unsigned int m = m_uiLocalBlockList[e].getLocalElementBegin();
             m < m_uiLocalBlockList[e].getLocalElementEnd(); m++)
            m_uiE2BlkMap[(m - m_uiElementLocalBegin)] = e;

        if (blkNode.minX() == dmin) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_LEFT + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_LEFT));
        }

        if (blkNode.minY() == dmin) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_DOWN + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_DOWN));
        }

        if (blkNode.minZ() == dmin) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_BACK + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_BACK));
        }

        if (blkNode.maxX() == dmax) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_RIGHT + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_RIGHT));
        }

        if (blkNode.maxY() == dmax) {
            blkNode.setFlag(
                ((blkNode.getFlag()) |
                 ((1u << (OCT_DIR_UP + NUM_LEVEL_BITS)) | blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) & (1u << OCT_DIR_UP));
        }

        if (blkNode.maxZ() == dmax) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_FRONT + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_FRONT));
        }

        assert(blkNode.getLevel() ==
               m_uiLocalBlockList[e].getBlockNode().getLevel());
        m_uiLocalBlockList[e].setBlkNodeFlag(blkNode.getFlag());

        regLev = m_uiLocalBlockList[e].getRegularGridLev();
        sz     = 1u << (m_uiMaxDepth - regLev);

        m_uiLocalBlockList[e].initializeBlkDiagMap(LOOK_UP_TABLE_DEFAULT);
        m_uiLocalBlockList[e].initializeBlkVertexMap(LOOK_UP_TABLE_DEFAULT);

        blkSkeys.clear();
        blkKeys.clear();
        generateBlkEdgeSKeys(m_uiLocalBlockList[e], blkSkeys);
        generateBlkVertexSKeys(m_uiLocalBlockList[e], blkSkeys);
        mergeKeys(blkSkeys, blkKeys);
        blkSkeys.clear();
        SFC::seqSearch::SFC_treeSearch(
            &(*(blkKeys.begin())), &(*(m_uiAllElements.begin())), 0,
            blkKeys.size(), 0, m_uiAllElements.size(), m_uiMaxDepth,
            m_uiMaxDepth, ROOT_ROTATION);

        std::cout << m_uiGlobalRank << ": SEARCHING FOR: " << blkKeys.size()
                  << " blocks" << std::endl;

        for (unsigned int i = 0; i < blkKeys.size(); i++) {
            assert(blkKeys[i].getFlag() & OCT_FOUND);
            if (!(blkKeys[i].getFlag() & OCT_FOUND)) {
                std::cout << RED << "block diagonal key not found" << NRM
                          << std::endl;
            }
            ownerList     = blkKeys[i].getOwnerList();
            directionList = blkKeys[i].getStencilIndexDirectionList();
            result        = blkKeys[i].getSearchResult();

            assert(ownerList->size() == directionList->size());
            for (unsigned int w = 0; w < ownerList->size(); w++) {
                if ((*directionList)[w] < VERTEX_OFFSET) {
                    assert((*directionList)[w] >= EDGE_OFFSET);
                    m_uiLocalBlockList[e].setBlk2DiagMap(
                        (*ownerList)[w], ((*directionList)[w] - EDGE_OFFSET),
                        result);
                } else {  // this is an vertex neighbour.
                    assert((*directionList)[w] >= VERTEX_OFFSET);
                    m_uiLocalBlockList[e].setBlk2VertexMap(
                        ((*directionList)[w] - VERTEX_OFFSET), result);
                }
            }
        }
    }

    this->flagBlockGhostDependancies();

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished building the blocks!");
}

void Mesh::findBlockNeighborsWithoutSFC(ot::Block &blk) {
    const unsigned int domain_max = 1u << (m_uiMaxDepth);
    const ot::TreeNode blkNode    = blk.getBlockNode();
    const unsigned int regLev     = blk.getRegularGridLev();

    const unsigned int blkElem_1D = (1u << (regLev - blkNode.getLevel())) * 2;

    const unsigned int myX        = blkNode.getX();
    const unsigned int myY        = blkNode.getY();
    const unsigned int myZ        = blkNode.getZ();
    const unsigned int mySz       = 1u << (m_uiMaxDepth - blkNode.getLevel());
    const unsigned int hsz        = 1u << (m_uiMaxDepth - regLev - 1);  // hx/2

    // potential neighbor coordinates
    std::vector<std::tuple<unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int>>
        neighborCoordsDirAndOwner;

    // then we get the edge neighbors (all 12!)
    if (myX > 0 && myY > 0) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX - 1, myY - 1, myZ + k * hsz, OCT_DIR_LEFT_DOWN, k);
        }
    }

    if (myX > 0 && (myY + mySz) < domain_max) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX - 1, myY + mySz, myZ + k * hsz, OCT_DIR_LEFT_UP, k);
        }
    }

    if (myX > 0 && myZ > 0) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX - 1, myY + k * hsz, myZ - 1, OCT_DIR_LEFT_BACK, k);
        }
    }

    if (myX > 0 && (myZ + mySz) < domain_max) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX - 1, myY + k * hsz, myZ + mySz, OCT_DIR_LEFT_FRONT, k);
        }
    }

    if ((myX + mySz) < domain_max && myY > 0) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX + mySz, myY - 1, myZ + k * hsz, OCT_DIR_RIGHT_DOWN, k);
        }
    }

    if ((myX + mySz) < domain_max && (myY + mySz) < domain_max) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX + mySz, myY + mySz, myZ + k * hsz, OCT_DIR_RIGHT_UP, k);
        }
    }

    if ((myX + mySz) < domain_max && myZ > 0) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX + mySz, myY + k * hsz, myZ - 1, OCT_DIR_RIGHT_BACK, k);
        }
    }

    if ((myX + mySz) < domain_max && (myZ + mySz) < domain_max) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX + mySz, myY + k * hsz, myZ + mySz, OCT_DIR_RIGHT_FRONT, k);
        }
    }

    if (myY > 0 && myZ > 0) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX + k * hsz, myY - 1, myZ - 1, OCT_DIR_DOWN_BACK, k);
        }
    }

    if (myY > 0 && (myZ + mySz) < domain_max) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX + k * hsz, myY - 1, myZ + mySz, OCT_DIR_DOWN_FRONT, k);
        }
    }

    if ((myY + mySz) < domain_max && myZ > 0) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(myX + k * hsz, myY + mySz,
                                                   myZ - 1, OCT_DIR_UP_BACK, k);
        }
    }

    if ((myY + mySz) < domain_max && (myZ + mySz) < domain_max) {
        for (unsigned int k = 0; k < blkElem_1D; k++) {
            neighborCoordsDirAndOwner.emplace_back(
                myX + k * hsz, myY + mySz, myZ + mySz, OCT_DIR_UP_FRONT, k);
        }
    }

    // NOW we add in the vertex neighbors, which is 8 directions!
    if ((myX > 0) && (myY > 0) && (myZ > 0)) {
        neighborCoordsDirAndOwner.emplace_back(myX - 1, myY - 1, myZ - 1,
                                               OCT_DIR_LEFT_DOWN_BACK, 0);
    }
    if (((myX + mySz) < domain_max) && (myY > 0) && (myZ > 0)) {
        neighborCoordsDirAndOwner.emplace_back(myX + mySz, myY - 1, myZ - 1,
                                               OCT_DIR_RIGHT_DOWN_BACK, 1);
    }
    if ((myX > 0) && ((myY + mySz) < domain_max) && (myZ > 0)) {
        neighborCoordsDirAndOwner.emplace_back(myX - 1, myY + mySz, myZ - 1,
                                               OCT_DIR_LEFT_UP_BACK, 2);
    }
    if (((myX + mySz) < domain_max) && ((myY + mySz) < domain_max) &&
        (myZ > 0)) {
        neighborCoordsDirAndOwner.emplace_back(myX + mySz, myY + mySz, myZ - 1,
                                               OCT_DIR_RIGHT_UP_BACK, 3);
    }
    if ((myX > 0) && (myY > 0) && ((myZ + mySz) < domain_max)) {
        neighborCoordsDirAndOwner.emplace_back(myX - 1, myY - 1, myZ + mySz,
                                               OCT_DIR_LEFT_DOWN_FRONT, 4);
    }
    if (((myX + mySz) < domain_max) && (myY > 0) &&
        ((myZ + mySz) < domain_max)) {
        neighborCoordsDirAndOwner.emplace_back(myX + mySz, myY - 1, myZ + mySz,
                                               OCT_DIR_RIGHT_DOWN_FRONT, 5);
    }
    if ((myX > 0) && ((myY + mySz) < domain_max) &&
        ((myZ + mySz) < domain_max)) {
        neighborCoordsDirAndOwner.emplace_back(myX - 1, myY + mySz, myZ + mySz,
                                               OCT_DIR_LEFT_UP_FRONT, 6);
    }
    if (((myX + mySz) < domain_max) && ((myY + mySz) < domain_max) &&
        ((myZ + mySz) < domain_max)) {
        neighborCoordsDirAndOwner.emplace_back(
            myX + mySz, myY + mySz, myZ + mySz, OCT_DIR_RIGHT_UP_FRONT, 7);
    }

    unsigned int totalFound = 0;

    // now find the actual neighbors using the E2E map
    for (const auto &coord : neighborCoordsDirAndOwner) {
        unsigned int x_new     = std::get<0>(coord);
        unsigned int y_new     = std::get<1>(coord);
        unsigned int z_new     = std::get<2>(coord);
        unsigned int dir_new   = std::get<3>(coord);
        unsigned int owner_new = std::get<4>(coord);

        // we can create a temporary node for searching (this isn't as fast as
        // sfcPartitioning... but it'll do for now)
        // make sure the level input is m_uiMaxDepth
        ot::TreeNode searchNode(x_new, y_new, z_new, m_uiMaxDepth, m_uiDim,
                                m_uiMaxDepth);
        // so now we find the element that contains this point...
        unsigned int result = LOOK_UP_TABLE_DEFAULT;
        bool found = findContainingElementInAllNodes(searchNode, result);

        if (found) {
            if (dir_new < VERTEX_OFFSET) {
                // we have an edge neighbor!
                unsigned int edgeIdx = dir_new - EDGE_OFFSET;
                blk.setBlk2DiagMap(owner_new, edgeIdx, result);
            } else {
                // otherwise this is a vertex neighbor
                unsigned int vertexIdx = dir_new - VERTEX_OFFSET;
                blk.setBlk2VertexMap(vertexIdx, result);
            }
            totalFound++;
        } else {
            // std::cout << m_uiGlobalRank
            //           << ": WARNING: couldn't find vertex/edge offset "
            //           << searchNode.getX() << " " << searchNode.getY() << " "
            //           << searchNode.getZ() << std::endl;
        }
    }

    // std::cout << m_uiGlobalRank << ": found " << totalFound << "/"
    //           << neighborCoordsDirAndOwner.size() << " vertex/edges"
    //           << std::endl;
}

bool Mesh::findContainingElementInAllNodes(const ot::TreeNode &searchNode,
                                           unsigned int &result) {
    // Fast path: spatial hash lookup. Walk levels from finest (max depth)
    // to coarsest (level 1), rounding the search coordinate to each
    // level's cell origin and probing the hash. The first hit is the
    // containing element, since the octree is a disjoint partition of
    // space. This is O(maxDepth) vs the O(N) linear fallback.
    if (!m_uiAllElementsSpatialHash.empty()) {
        const unsigned int sx = searchNode.getX();
        const unsigned int sy = searchNode.getY();
        const unsigned int sz = searchNode.getZ();
        for (unsigned int L = m_uiMaxDepth; L >= 1; L--) {
            const unsigned int cellSz = 1u << (m_uiMaxDepth - L);
            const uint32_t mask       = ~(cellSz - 1);
            const uint32_t cx         = sx & mask;
            const uint32_t cy         = sy & mask;
            const uint32_t cz         = sz & mask;
            const uint64_t key        =
                (uint64_t)L |
                ((uint64_t)cx << 6) |
                ((uint64_t)cy << 28) |
                ((uint64_t)cz << 50);
            auto it = m_uiAllElementsSpatialHash.find(key);
            if (it != m_uiAllElementsSpatialHash.end()) {
                result = it->second;
                return true;
            }
        }
        result = LOOK_UP_TABLE_DEFAULT;
        return false;
    }

    // first search the local elements, because it's more likely our target will
    // be there
    for (unsigned int e = m_uiElementLocalBegin; e < m_uiElementLocalEnd; e++) {
        if (m_uiAllElements[e].contains(searchNode)) {
            result = e;
            return true;
        }
    }

    // now search through pre-ghosts
    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++) {
        if (m_uiAllElements[e].contains(searchNode)) {
            result = e;
            return true;
        }
    }

    // finally search through post-ghosts
    for (unsigned int e = m_uiElementPostGhostBegin;
         e < m_uiElementPostGhostEnd; e++) {
        if (m_uiAllElements[e].contains(searchNode)) {
            result = e;
            return true;
        }
    }

    // search failed, returns lookup_table_default
    result = LOOK_UP_TABLE_DEFAULT;
    return false;
}

void Mesh::performBlocksSetupRepartitioned(unsigned int cLev, unsigned int *tag,
                                           unsigned int tsz) {
    m_uiIsBlockSetup  = true;
    m_uiCoarsetBlkLev = cLev;

    if (!m_uiIsActive) return;

    // Build spatial hash for fast findContainingElementInAllNodes
    // lookups. Without this, block-neighbor discovery on graph-
    // partitioned meshes degrades to O(num_blocks * num_coords * N),
    // which was ~6.5s for a 16k-element Random partition run.
    m_uiAllElementsSpatialHash.clear();
    m_uiAllElementsSpatialHash.reserve(m_uiAllElements.size());
    for (unsigned int e = 0; e < m_uiAllElements.size(); e++) {
        const auto &tn = m_uiAllElements[e];
        const uint64_t L  = tn.getLevel();
        const uint64_t cx = tn.getX();
        const uint64_t cy = tn.getY();
        const uint64_t cz = tn.getZ();
        const uint64_t key =
            L | (cx << 6) | (cy << 28) | (cz << 50);
        m_uiAllElementsSpatialHash[key] = e;
    }

    // blkSz and offset vectors
    std::vector<DendroIntL> blkSz(m_uiLocalBlockList.size());
    std::vector<DendroIntL> blkSzOffset(m_uiLocalBlockList.size());

    // construct the element to block map
    m_uiE2BlkMap.resize(m_uiNumLocalElements, LOOK_UP_TABLE_DEFAULT);

    for (unsigned int k = 0; k < m_uiLocalBlockList.size(); k++) {
        blkSz[k] = m_uiLocalBlockList[k].getAlignedBlockSz();
    }

    // calculate the offsets of how many blocks are within a single block list
    blkSzOffset[0] = 0;
    omp_par::scan(blkSz.data(), blkSzOffset.data(), m_uiLocalBlockList.size());

    // put the offsets inside the local block list
    for (unsigned int k = 0; k < m_uiLocalBlockList.size(); k++)
        m_uiLocalBlockList[k].setOffset(blkSzOffset[k]);

    // calculate the unzipped vec size now, which is based on the offsets!
    m_uiUnZippedVecSz =
        blkSzOffset[m_uiLocalBlockList.size() - 1] + blkSz.back();

    // then we calculate a few additional things
    const unsigned int dmin = 0;
    const unsigned int dmax = 1u << (m_uiMaxDepth);

    // now iterate through the blocks, there's no need to predeclare what we
    // need, this saves space
    for (unsigned int e = 0; e < m_uiLocalBlockList.size(); e++) {
        // easier referencing into the blk
        ot::Block &blk       = m_uiLocalBlockList[e];
        ot::TreeNode blkNode = blk.getBlockNode();

        // update the element to block map (E2Blk); iterate via the
        // block iterator so non-SFC blocks with index-list storage
        // (which report getLocalElementBegin == End == 0) also map.
        for (unsigned int m : blk)
            m_uiE2BlkMap[(m - m_uiElementLocalBegin)] = e;

        // set the boundary flags!
        if (blkNode.minX() == dmin) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_LEFT + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_LEFT));
        }

        if (blkNode.minY() == dmin) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_DOWN + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_DOWN));
        }

        if (blkNode.minZ() == dmin) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_BACK + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_BACK));
        }

        if (blkNode.maxX() == dmax) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_RIGHT + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_RIGHT));
        }

        if (blkNode.maxY() == dmax) {
            blkNode.setFlag(
                ((blkNode.getFlag()) |
                 ((1u << (OCT_DIR_UP + NUM_LEVEL_BITS)) | blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) & (1u << OCT_DIR_UP));
        }

        if (blkNode.maxZ() == dmax) {
            blkNode.setFlag(((blkNode.getFlag()) |
                             ((1u << (OCT_DIR_FRONT + NUM_LEVEL_BITS)) |
                              blkNode.getLevel())));
            assert((blkNode.getFlag() >> NUM_LEVEL_BITS) &
                   (1u << OCT_DIR_FRONT));
        }

        // run the assertion that our level for block node and block list level
        // are the same
        assert(blkNode.getLevel() == blk.getBlockNode().getLevel());
        blk.setBlkNodeFlag(blkNode.getFlag());

        // I removed the regLevel and sz stuff because they aren't called
        // anywhere

        // then we initialize the block maps
        blk.initializeBlkDiagMap(LOOK_UP_TABLE_DEFAULT);
        blk.initializeBlkVertexMap(LOOK_UP_TABLE_DEFAULT);

        // split into a function to make it far easier to read
        findBlockNeighborsWithoutSFC(m_uiLocalBlockList[e]);
    }

    this->flagBlockGhostDependancies();

    // Release spatial hash memory now that block neighbor discovery
    // is complete. Any later findContainingElementInAllNodes falls
    // back to the linear scan.
    m_uiAllElementsSpatialHash.clear();
}

bool Mesh::isEdgeHanging(unsigned int elementId, unsigned int edgeId,
                         unsigned int &cnum) const {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return false;

    unsigned int nodeLookUp_DG;
    bool isHanging = false;
    cnum           = 0;
    assert(elementId >= 0 && elementId < m_uiAllElements.size());

    bool isVertexHanging[2] = {false, false};
    unsigned int owner[2];
    unsigned int ii_x[2], jj_y[2], kk_z[2];
    unsigned int mid_bit;
    unsigned int lenSz;

    switch (edgeId) {
        case OCT_DIR_LEFT_DOWN:

            isVertexHanging[0] = this->isNodeHanging(elementId, 0, 0, 0);
            isVertexHanging[1] =
                this->isNodeHanging(elementId, 0, 0, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG =
                    m_uiE2NMapping_DG[elementId * m_uiNpE +
                                      (0) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (0) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG =
                    m_uiE2NMapping_DG[elementId * m_uiNpE +
                                      (m_uiElementOrder) *
                                          (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (0) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minZ() ==
                    m_uiAllElements[owner[0]].minZ() + lenSz * kk_z[0])
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxZ() ==
                           m_uiAllElements[owner[1]].minZ() + lenSz * kk_z[1]);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG =
                    m_uiE2NMapping_DG[elementId * m_uiNpE +
                                      ((m_uiElementOrder >> 1u)) *
                                          (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (0) * (m_uiElementOrder + 1) + (0)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minZ() ==
                    m_uiAllElements[elementId].minZ())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxZ() ==
                            m_uiAllElements[elementId].maxZ()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_LEFT_UP:

            isVertexHanging[0] =
                this->isNodeHanging(elementId, 0, m_uiElementOrder, 0);
            isVertexHanging[1] = this->isNodeHanging(
                elementId, 0, m_uiElementOrder, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (0) * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) *
                         (m_uiElementOrder + 1) +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minZ() ==
                    m_uiAllElements[owner[0]].minZ() + lenSz * kk_z[0])
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxZ() ==
                           m_uiAllElements[owner[1]].minZ() + lenSz * kk_z[1]);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     ((m_uiElementOrder >> 1u)) * (m_uiElementOrder + 1) *
                         (m_uiElementOrder + 1) +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) + (0)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minZ() ==
                    m_uiAllElements[elementId].minZ())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxZ() ==
                            m_uiAllElements[elementId].maxZ()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_LEFT_BACK:

            isVertexHanging[0] = this->isNodeHanging(elementId, 0, 0, 0);
            isVertexHanging[1] =
                this->isNodeHanging(elementId, 0, m_uiElementOrder, 0);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG =
                    m_uiE2NMapping_DG[elementId * m_uiNpE +
                                      (0) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (0) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (0) * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minY() ==
                    (m_uiAllElements[owner[0]].minY() + jj_y[0] * lenSz))
                    cnum = 0;
                else {
                    assert(
                        m_uiAllElements[elementId].maxY() ==
                        (m_uiAllElements[owner[1]].minY() + jj_y[1] * lenSz));
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (0) * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                     ((m_uiElementOrder >> 1u)) * (m_uiElementOrder + 1) + (0)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minY() ==
                    m_uiAllElements[elementId].minY())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxY() ==
                            m_uiAllElements[elementId].maxY()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_LEFT_FRONT:

            isVertexHanging[0] =
                this->isNodeHanging(elementId, 0, 0, m_uiElementOrder);
            isVertexHanging[1] = this->isNodeHanging(
                elementId, 0, m_uiElementOrder, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG =
                    m_uiE2NMapping_DG[elementId * m_uiNpE +
                                      (m_uiElementOrder) *
                                          (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (0) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) *
                         (m_uiElementOrder + 1) +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minY() ==
                    (m_uiAllElements[owner[0]].minY() + jj_y[0] * lenSz))
                    cnum = 0;
                else {
                    assert(
                        m_uiAllElements[elementId].maxY() ==
                        (m_uiAllElements[owner[1]].minY() + jj_y[1] * lenSz));
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) *
                         (m_uiElementOrder + 1) +
                     ((m_uiElementOrder >> 1u)) * (m_uiElementOrder + 1) + (0)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minY() ==
                    m_uiAllElements[elementId].minY())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxY() ==
                            m_uiAllElements[elementId].maxY()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_RIGHT_DOWN:

            isVertexHanging[0] =
                this->isNodeHanging(elementId, m_uiElementOrder, 0, 0);
            isVertexHanging[1] = this->isNodeHanging(
                elementId, m_uiElementOrder, 0, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minZ() ==
                    m_uiAllElements[owner[0]].minZ() + lenSz * kk_z[0])
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxZ() ==
                           m_uiAllElements[owner[1]].minZ() + lenSz * kk_z[1]);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  ((m_uiElementOrder >> 1u)) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minZ() ==
                    m_uiAllElements[elementId].minZ())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxZ() ==
                            m_uiAllElements[elementId].maxZ()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_RIGHT_UP:

            isVertexHanging[0] = this->isNodeHanging(
                elementId, m_uiElementOrder, m_uiElementOrder, 0);
            isVertexHanging[1] =
                this->isNodeHanging(elementId, m_uiElementOrder,
                                    m_uiElementOrder, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minZ() ==
                    m_uiAllElements[owner[0]].minZ() + lenSz * kk_z[0])
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxZ() ==
                           m_uiAllElements[owner[1]].minZ() + lenSz * kk_z[1]);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  ((m_uiElementOrder >> 1u)) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minZ() ==
                    m_uiAllElements[elementId].minZ())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxZ() ==
                            m_uiAllElements[elementId].maxZ()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_RIGHT_BACK:

            isVertexHanging[0] =
                this->isNodeHanging(elementId, m_uiElementOrder, 0, 0);
            isVertexHanging[1] = this->isNodeHanging(
                elementId, m_uiElementOrder, m_uiElementOrder, 0);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minY() ==
                    (m_uiAllElements[owner[0]].minY() + jj_y[0] * lenSz))
                    cnum = 0;
                else {
                    assert(
                        m_uiAllElements[elementId].maxY() ==
                        (m_uiAllElements[owner[1]].minY() + jj_y[1] * lenSz));
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  ((m_uiElementOrder >> 1u)) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minY() ==
                    m_uiAllElements[elementId].minY())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxY() ==
                            m_uiAllElements[elementId].maxY()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_RIGHT_FRONT:

            isVertexHanging[0] = this->isNodeHanging(
                elementId, m_uiElementOrder, 0, m_uiElementOrder);
            isVertexHanging[1] =
                this->isNodeHanging(elementId, m_uiElementOrder,
                                    m_uiElementOrder, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minY() ==
                    (m_uiAllElements[owner[0]].minY() + jj_y[0] * lenSz))
                    cnum = 0;
                else {
                    assert(
                        m_uiAllElements[elementId].maxY() ==
                        (m_uiAllElements[owner[1]].minY() + jj_y[1] * lenSz));
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  ((m_uiElementOrder >> 1u)) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minY() ==
                    m_uiAllElements[elementId].minY())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxY() ==
                            m_uiAllElements[elementId].maxY()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_DOWN_BACK:

            isVertexHanging[0] = this->isNodeHanging(elementId, 0, 0, 0);
            isVertexHanging[1] =
                this->isNodeHanging(elementId, m_uiElementOrder, 0, 0);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG =
                    m_uiE2NMapping_DG[elementId * m_uiNpE +
                                      (0) * (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (0) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minX() ==
                    m_uiAllElements[owner[0]].minX() + ii_x[0] * lenSz)
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxX() ==
                           m_uiAllElements[owner[1]].minX() + ii_x[1] * lenSz);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder >> 1u)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minX() ==
                    m_uiAllElements[elementId].minX())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxX() ==
                            m_uiAllElements[elementId].maxX()));
                    cnum = 1;
                }
            }
            break;

        case OCT_DIR_DOWN_FRONT:

            isVertexHanging[0] =
                this->isNodeHanging(elementId, 0, 0, m_uiElementOrder);
            isVertexHanging[1] = this->isNodeHanging(
                elementId, m_uiElementOrder, 0, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG =
                    m_uiE2NMapping_DG[elementId * m_uiNpE +
                                      (m_uiElementOrder) *
                                          (m_uiElementOrder + 1) *
                                          (m_uiElementOrder + 1) +
                                      (0) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minX() ==
                    m_uiAllElements[owner[0]].minX() + ii_x[0] * lenSz)
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxX() ==
                           m_uiAllElements[owner[1]].minX() + ii_x[1] * lenSz);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (0) * (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder >> 1u)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minX() ==
                    m_uiAllElements[elementId].minX())
                    cnum = 0;
                else if ((m_uiElementOrder == 1) ||
                         (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxX() ==
                          m_uiAllElements[elementId].maxX())) {
                    cnum = 1;
                } else {
                    return false;
                }
            }

            break;

        case OCT_DIR_UP_BACK:

            isVertexHanging[0] =
                this->isNodeHanging(elementId, 0, m_uiElementOrder, 0);
            isVertexHanging[1] = this->isNodeHanging(
                elementId, m_uiElementOrder, m_uiElementOrder, 0);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (0) * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minX() ==
                    m_uiAllElements[owner[0]].minX() + ii_x[0] * lenSz)
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxX() ==
                           m_uiAllElements[owner[1]].minX() + ii_x[1] * lenSz);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (0) * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder >> 1u)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minX() ==
                    m_uiAllElements[elementId].minX())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxX() ==
                            m_uiAllElements[elementId].maxX()));
                    cnum = 1;
                }
            }

            break;

        case OCT_DIR_UP_FRONT:

            isVertexHanging[0] = this->isNodeHanging(
                elementId, 0, m_uiElementOrder, m_uiElementOrder);
            isVertexHanging[1] =
                this->isNodeHanging(elementId, m_uiElementOrder,
                                    m_uiElementOrder, m_uiElementOrder);

            isHanging = (isVertexHanging[0] && isVertexHanging[1]);
            if (!isHanging) return false;

            if (m_uiElementOrder == 1) {
                // special case to linear order,
                nodeLookUp_DG = m_uiE2NMapping_DG
                    [elementId * m_uiNpE +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) *
                         (m_uiElementOrder + 1) +
                     (m_uiElementOrder) * (m_uiElementOrder + 1) + 0];
                this->dg2eijk(nodeLookUp_DG, owner[0], ii_x[0], jj_y[0],
                              kk_z[0]);

                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  m_uiElementOrder];
                this->dg2eijk(nodeLookUp_DG, owner[1], ii_x[1], jj_y[1],
                              kk_z[1]);

                assert(m_uiAllElements[owner[0]].getLevel() ==
                       m_uiAllElements[owner[1]].getLevel());
                lenSz =
                    1u << (m_uiMaxDepth - m_uiAllElements[owner[0]].getLevel());

                if (m_uiAllElements[elementId].minX() ==
                    m_uiAllElements[owner[0]].minX() + ii_x[0] * lenSz)
                    cnum = 0;
                else {
                    assert(m_uiAllElements[elementId].maxX() ==
                           m_uiAllElements[owner[1]].minX() + ii_x[1] * lenSz);
                    cnum = 1;
                }

            } else {
                nodeLookUp_DG = m_uiE2NMapping_DG[elementId * m_uiNpE +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder) *
                                                      (m_uiElementOrder + 1) +
                                                  (m_uiElementOrder >> 1u)];
                if (m_uiAllElements[nodeLookUp_DG / m_uiNpE].minX() ==
                    m_uiAllElements[elementId].minX())
                    cnum = 0;
                else {
                    assert((m_uiElementOrder == 1) ||
                           (m_uiAllElements[nodeLookUp_DG / m_uiNpE].maxX() ==
                            m_uiAllElements[elementId].maxX()));
                    cnum = 1;
                }
            }

            break;
    }

    return isHanging;
}

bool Mesh::isFaceHanging(unsigned int elementId, unsigned int faceId,
                         unsigned int &cnum) const {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return false;

    unsigned int nodeLookUp_DG;
    bool isHanging = false;
    unsigned int ownerID;
    unsigned int mid_bit;
    assert(elementId >= 0 && elementId < m_uiAllElements.size());
    unsigned int pSz, cSz;
    cnum = 0;
    const unsigned int lookup =
        m_uiE2EMapping[elementId * m_uiNumDirections + faceId];
    if (lookup == LOOK_UP_TABLE_DEFAULT)
        isHanging = false;
    else
        isHanging = m_uiAllElements[lookup].getLevel() <
                    m_uiAllElements[elementId].getLevel();

    // Note: if the face is Hanging it is reliable to use the element to element
    // map.

    if (isHanging) {
        switch (faceId) {
            case OCT_DIR_LEFT:
                // nodeLookUp_DG=m_uiE2NMapping_DG[elementId*m_uiNpE+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)+(0)];
                // ownerID=nodeLookUp_DG/m_uiNpE;
                ownerID = lookup;
                mid_bit =
                    m_uiMaxDepth - m_uiAllElements[ownerID].getLevel() - 1;
                cnum = ((((((m_uiAllElements[elementId].getZ()) -
                            (m_uiAllElements[ownerID].getZ())) >>
                           mid_bit) &
                          1u)
                         << 1u) |
                        ((((m_uiAllElements[elementId].getY()) -
                           (m_uiAllElements[ownerID].getY())) >>
                          mid_bit) &
                         1u));
                break;

            case OCT_DIR_RIGHT:
                // nodeLookUp_DG=m_uiE2NMapping_DG[elementId*m_uiNpE+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)+(m_uiElementOrder)];
                // ownerID=nodeLookUp_DG/m_uiNpE;
                ownerID = lookup;
                mid_bit =
                    m_uiMaxDepth - m_uiAllElements[ownerID].getLevel() - 1;
                cnum = ((((((m_uiAllElements[elementId].getZ()) -
                            (m_uiAllElements[ownerID].getZ())) >>
                           mid_bit) &
                          1u)
                         << 1u) |
                        ((((m_uiAllElements[elementId].getY()) -
                           (m_uiAllElements[ownerID].getY())) >>
                          mid_bit) &
                         1u));

                break;

            case OCT_DIR_DOWN:
                // nodeLookUp_DG=m_uiE2NMapping_DG[elementId*m_uiNpE+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)*(m_uiElementOrder+1)+(0)*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))];
                // ownerID=nodeLookUp_DG/m_uiNpE;
                ownerID = lookup;
                mid_bit =
                    m_uiMaxDepth - m_uiAllElements[ownerID].getLevel() - 1;
                cnum = ((((((m_uiAllElements[elementId].getZ()) -
                            (m_uiAllElements[ownerID].getZ())) >>
                           mid_bit) &
                          1u)
                         << 1u) |
                        ((((m_uiAllElements[elementId].getX()) -
                           (m_uiAllElements[ownerID].getX())) >>
                          mid_bit) &
                         1u));
                break;

            case OCT_DIR_UP:
                // nodeLookUp_DG=m_uiE2NMapping_DG[elementId*m_uiNpE+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)*(m_uiElementOrder+1)+(m_uiElementOrder)*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))];
                // ownerID=nodeLookUp_DG/m_uiNpE;
                ownerID = lookup;
                mid_bit =
                    m_uiMaxDepth - m_uiAllElements[ownerID].getLevel() - 1;
                cnum = ((((((m_uiAllElements[elementId].getZ()) -
                            (m_uiAllElements[ownerID].getZ())) >>
                           mid_bit) &
                          1u)
                         << 1u) |
                        ((((m_uiAllElements[elementId].getX()) -
                           (m_uiAllElements[ownerID].getX())) >>
                          mid_bit) &
                         1u));
                break;

            case OCT_DIR_BACK:
                // nodeLookUp_DG=m_uiE2NMapping_DG[elementId*m_uiNpE+(0)*(m_uiElementOrder+1)*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))];
                // ownerID=nodeLookUp_DG/m_uiNpE;
                ownerID = lookup;

                mid_bit =
                    m_uiMaxDepth - m_uiAllElements[ownerID].getLevel() - 1;
                cnum = ((((((m_uiAllElements[elementId].getY()) -
                            (m_uiAllElements[ownerID].getY())) >>
                           mid_bit) &
                          1u)
                         << 1u) |
                        ((((m_uiAllElements[elementId].getX()) -
                           (m_uiAllElements[ownerID].getX())) >>
                          mid_bit) &
                         1u));
                // if(m_uiAllElements[lookup] !=
                // m_uiAllElements[nodeLookUp_DG/m_uiNpE]) std::cout<<"owner :
                // "<<m_uiAllElements[lookup]<<" dg:
                // "<<m_uiAllElements[nodeLookUp_DG/m_uiNpE]<<" current
                // "<<m_uiAllElements[elementId]<<" cnum: "<<cnum<<std::endl;
                break;

            case OCT_DIR_FRONT:
                // nodeLookUp_DG=m_uiE2NMapping_DG[elementId*m_uiNpE+(m_uiElementOrder)*(m_uiElementOrder+1)*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))*(m_uiElementOrder+1)+((m_uiElementOrder>>1u))];
                // ownerID=nodeLookUp_DG/m_uiNpE;
                ownerID = lookup;
                mid_bit =
                    m_uiMaxDepth - m_uiAllElements[ownerID].getLevel() - 1;
                cnum = ((((((m_uiAllElements[elementId].getY()) -
                            (m_uiAllElements[ownerID].getY())) >>
                           mid_bit) &
                          1u)
                         << 1u) |
                        ((((m_uiAllElements[elementId].getX()) -
                           (m_uiAllElements[ownerID].getX())) >>
                          mid_bit) &
                         1u));
                break;
        }
    }

    return isHanging;
}

bool Mesh::isNodeHanging(unsigned int eleID, unsigned int ix, unsigned int jy,
                         unsigned int kz) const {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return false;

    return m_uiAllElements[(m_uiE2NMapping_DG[eleID * m_uiNpE +
                                              kz * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              jy * (m_uiElementOrder + 1) +
                                              ix] /
                            m_uiNpE)]
               .getLevel() < m_uiAllElements[eleID].getLevel();
}

ot::Mesh *Mesh::ReMesh(unsigned int grainSz, double ld_tol, unsigned int sfK,
                       unsigned int (*getWeight)(const ot::TreeNode *),
                       unsigned int *blk_tags, unsigned int blk_tag_sz) {
    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Creating new mesh during remesh");
    std::vector<ot::TreeNode> balOct1;  // new balanced octree.

    if (m_uiIsActive) {
        // 1. build the unbalanced octree from the remesh flags.
        std::vector<ot::TreeNode> unBalancedOctree;
        unsigned int remeshFlag;
        unsigned int sz;
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < (m_uiElementLocalEnd); ele++) {
            remeshFlag = (m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS);
            assert(m_uiAllElements[ele].getLevel() != 0);
            if (remeshFlag == OCT_SPLIT) {
                m_uiAllElements[ele].addChildren(unBalancedOctree);

            } else if (remeshFlag == OCT_COARSE) {
                assert(m_uiAllElements[ele].getParent() ==
                       m_uiAllElements[ele + NUM_CHILDREN - 1].getParent());
                unBalancedOctree.push_back(m_uiAllElements[ele].getParent());
                ele = ele + NUM_CHILDREN - 1;

            } else {
                assert(remeshFlag == OCT_NO_CHANGE);
                unBalancedOctree.push_back(m_uiAllElements[ele]);
            }
        }

        std::vector<ot::TreeNode> unBalOctSplitters;
        unBalOctSplitters.resize(m_uiActiveNpes);

        par::Mpi_Allgather(&(unBalancedOctree.back()),
                           &(*(unBalOctSplitters.begin())), 1, m_uiCommActive);

        // std::cout<<"rank: "<<m_uiActiveRank<<" unbalnced oct size:
        // "<<unBalancedOctree.size()<<std::endl;
        // treeNodesTovtk(unBalancedOctree,m_uiActiveRank,"unbalncedOctree");
        //  unbalancedOctree should be complete. Due to 1 level coarsen and
        //  refinement.

        // 2. balanced the octree.

        ot::TreeNode rootNode(m_uiDim, m_uiMaxDepth);
        SFC::parSort::SFC_treeSort(
            unBalancedOctree, balOct1, balOct1, balOct1, ld_tol, m_uiMaxDepth,
            rootNode, ROOT_ROTATION, 1, TS_BALANCE_OCTREE, sfK, m_uiCommActive);
        par::partitionW(balOct1, getWeight, m_uiCommActive);
        assert(par::test::isUniqueAndSorted(balOct1, m_uiCommActive));
        unBalancedOctree.clear();

        if (m_uiActiveNpes == 1) {
            // sequential case to synchronize flags.

            unsigned int count2         = 0;  // element iterator for the mesh2.
            const ot::TreeNode *pNodes2 = &(*(balOct1.begin()));
            ot::TreeNode *pNodes1       = &(*(m_uiAllElements.begin()));

            assert(m_uiAllElements[m_uiElementLocalBegin] == balOct1.front() ||
                   m_uiAllElements[m_uiElementLocalBegin].getParent() ==
                       balOct1.front() ||
                   m_uiAllElements[m_uiElementLocalBegin] ==
                       balOct1.front().getParent());
            assert(m_uiAllElements[m_uiElementLocalEnd - 1] == balOct1.back() ||
                   m_uiAllElements[m_uiElementLocalEnd - 1].getParent() ==
                       balOct1.back() ||
                   m_uiAllElements[m_uiElementLocalEnd - 1] ==
                       balOct1.back().getParent());

            /* while( (count2<balOct1.size()) &&
             (pNodes1[m_uiElementLocalBegin]!=pNodes2[count2]) &&
             (pNodes1[m_uiElementLocalBegin].getParent()!=pNodes2[count2]) &&
             (pNodes1[m_uiElementLocalBegin]!=pNodes2[count2].getParent()))
             count2++; assert(count2<balOct1.size());*/

            for (unsigned int ele = m_uiElementLocalBegin;
                 ele < m_uiElementLocalEnd; ele++) {
                /* if(!(pNodes1[ele]==pNodes2[count2] ||
                 pNodes1[ele].getParent()==pNodes2[count2] ||
                 pNodes1[ele]==pNodes2[count2].getParent()))
                 {
                     std::cout<<"rank: "<<m_uiActiveRank<<" ele:
                 "<<ele<<"pNodes1: "<<pNodes1[ele]<<" pNodes2:
                 "<<pNodes2[count2]<<std::endl;
                 }*/
                assert(count2 < balOct1.size());
                assert(pNodes1[ele] == pNodes2[count2] ||
                       pNodes1[ele].getParent() == pNodes2[count2] ||
                       pNodes1[ele] == pNodes2[count2].getParent());

                if (pNodes1[ele] ==
                    pNodes2[count2]
                        .getParent()) {  // old elements have splitted.
                    assert(pNodes2[count2].getParent() ==
                           pNodes2[count2 + NUM_CHILDREN - 1].getParent());
                    pNodes1[ele].setFlag(((OCT_SPLIT << NUM_LEVEL_BITS) |
                                          pNodes1[ele].getLevel()));
                    count2 = count2 + NUM_CHILDREN;

                } else if (pNodes1[ele].getParent() ==
                           pNodes2[count2]) {  // old elements have coarsen
                    assert(pNodes1[ele].getParent() ==
                           pNodes1[ele + NUM_CHILDREN - 1].getParent());
                    pNodes1[ele].setFlag(((OCT_COARSE << NUM_LEVEL_BITS) |
                                          pNodes1[ele].getLevel()));
                    ele = ele + NUM_CHILDREN - 1;
                    count2++;

                } else {
                    assert(pNodes1[ele] == pNodes2[count2]);
                    pNodes1[ele].setFlag(((OCT_NO_CHANGE << NUM_LEVEL_BITS) |
                                          pNodes1[ele].getLevel()));
                    count2++;
                }
            }

            assert(par::test::isUniqueAndSorted(balOct1, m_uiCommActive));
            // std::cout<<"rank: "<<m_uiActiveRank<<" balanced oct size:
            // "<<balOct1.size()<<std::endl;

        } else {
            assert(par::test::isUniqueAndSorted(balOct1, m_uiCommActive));

            int *sendOctCount   = new int[m_uiActiveNpes];
            int *recvOctCount   = new int[m_uiActiveNpes];
            int *sendOctOffset  = new int[m_uiActiveNpes];
            int *recvOctOffset  = new int[m_uiActiveNpes];

            unsigned int sBegin = 0, sEnd = 0, sResult = 0;
            for (unsigned int p = 0; p < m_uiActiveNpes; p++)
                sendOctCount[p] = 0;

            unsigned int pCount = 0;
            std::vector<unsigned int> searchResultIndex;
            searchResultIndex.resize(m_uiActiveNpes, LOOK_UP_TABLE_DEFAULT);

            if (balOct1.size()) {
                while (
                    (pCount < m_uiActiveNpes) &&
                    (!unBalOctSplitters[pCount].isAncestor(balOct1.front())) &&
                    (balOct1.front() > unBalOctSplitters[pCount]))
                    pCount++;

                for (unsigned int e = 0;
                     ((e < balOct1.size()) && (pCount < m_uiActiveNpes)); e++) {
                    if (balOct1[e] == unBalOctSplitters[pCount]) {
                        searchResultIndex[pCount] = e + 1;
                        pCount++;

                    } else if (unBalOctSplitters[pCount].isAncestor(
                                   balOct1[e])) {
                        while (((e + 1) < balOct1.size()) &&
                               (unBalOctSplitters[pCount].isAncestor(
                                   balOct1[e + 1])))
                            e++;
                        searchResultIndex[pCount] = e + 1;
                        pCount++;

                    } else if (balOct1[e] ==
                               unBalOctSplitters[pCount].getParent()) {
                        searchResultIndex[pCount] = e + 1;
                        pCount++;

                    } else if ((e == (balOct1.size() - 1))) {
                        searchResultIndex[pCount] = balOct1.size();
                    }
                }

                for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
                    if (searchResultIndex[p] != LOOK_UP_TABLE_DEFAULT) {
                        sBegin = sEnd;
                        sEnd   = searchResultIndex[p];
                        assert(sBegin <= sEnd);
                        sendOctCount[p] = sEnd - sBegin;
                    }
                }
            }

            par::Mpi_Alltoall(sendOctCount, recvOctCount, 1, m_uiCommActive);

            sendOctOffset[0] = 0;
            recvOctOffset[0] = 0;

            omp_par::scan(sendOctCount, sendOctOffset, m_uiActiveNpes);
            omp_par::scan(recvOctCount, recvOctOffset, m_uiActiveNpes);

            // std::cout<<" rank: "<<m_uiActiveRank<<" balOct1 size:
            // "<<balOct1.size()<<" sendCount:
            // "<<(sendOctOffset[m_uiActiveNpes-1]+sendOctCount[m_uiActiveNpes-1])<<std::endl;
            assert(balOct1.size() == (sendOctOffset[m_uiActiveNpes - 1] +
                                      sendOctCount[m_uiActiveNpes - 1]));

            std::vector<ot::TreeNode> recvOctBuffer;
            recvOctBuffer.resize(recvOctOffset[m_uiActiveNpes - 1] +
                                 recvOctCount[m_uiActiveNpes - 1]);

            par::Mpi_Alltoallv(&(*(balOct1.begin())), sendOctCount,
                               sendOctOffset, &(*(recvOctBuffer.begin())),
                               recvOctCount, recvOctOffset, m_uiCommActive);
            std::swap(balOct1, recvOctBuffer);
            recvOctBuffer.clear();
            assert(par::test::isUniqueAndSorted(balOct1, m_uiCommActive));

            delete[] sendOctCount;
            delete[] recvOctCount;
            delete[] sendOctOffset;
            delete[] recvOctOffset;

            // 3. synchronize the wavelet remesh flags with 2:1 balanced octree.
            if (balOct1.size()) {
                unsigned int count2 = 0;  // element iterator for the mesh2.
                const ot::TreeNode *pNodes2 = &(*(balOct1.begin()));
                ot::TreeNode *pNodes1       = &(*(m_uiAllElements.begin()));

                assert(m_uiAllElements[m_uiElementLocalBegin] ==
                           balOct1.front() ||
                       m_uiAllElements[m_uiElementLocalBegin].getParent() ==
                           balOct1.front() ||
                       m_uiAllElements[m_uiElementLocalBegin] ==
                           balOct1.front().getParent());
                assert(m_uiAllElements[m_uiElementLocalEnd - 1] ==
                           balOct1.back() ||
                       m_uiAllElements[m_uiElementLocalEnd - 1].getParent() ==
                           balOct1.back() ||
                       m_uiAllElements[m_uiElementLocalEnd - 1] ==
                           balOct1.back().getParent());
                if (!(m_uiAllElements[m_uiElementLocalBegin] ==
                          balOct1.front() ||
                      m_uiAllElements[m_uiElementLocalBegin].getParent() ==
                          balOct1.front() ||
                      m_uiAllElements[m_uiElementLocalBegin] ==
                          balOct1.front().getParent())) {
                    std::cout << "[Remesh Error]: rank: " << m_uiActiveRank
                              << " M1 & M2 front alignment failed "
                              << std::endl;
                    exit(0);
                }

                if (!(m_uiAllElements[m_uiElementLocalEnd - 1] ==
                          balOct1.back() ||
                      m_uiAllElements[m_uiElementLocalEnd - 1].getParent() ==
                          balOct1.back() ||
                      m_uiAllElements[m_uiElementLocalEnd - 1] ==
                          balOct1.back().getParent())) {
                    std::cout << "[Remesh Error]: rank: " << m_uiActiveRank
                              << " M1 & M2 back alignment failed " << std::endl;
                    exit(0);
                }

                /* while( (count2<balOct1.size()) &&
                 (pNodes1[m_uiElementLocalBegin]!=pNodes2[count2]) &&
                 (pNodes1[m_uiElementLocalBegin].getParent()!=pNodes2[count2])
                 &&
                 (pNodes1[m_uiElementLocalBegin]!=pNodes2[count2].getParent()))
                 count2++; assert(count2<balOct1.size());*/

                for (unsigned int ele = m_uiElementLocalBegin;
                     ele < m_uiElementLocalEnd; ele++) {
                    if (!(pNodes1[ele] == pNodes2[count2] ||
                          pNodes1[ele].getParent() == pNodes2[count2] ||
                          pNodes1[ele] == pNodes2[count2].getParent())) {
                        std::cout
                            << "[Remesh Error]: rank: " << m_uiActiveRank
                            << " ele: " << ele << "pNodes1: " << pNodes1[ele]
                            << " count2: " << count2
                            << " pNodes2: " << pNodes2[count2] << std::endl;
                        exit(0);
                    }

                    assert(count2 < balOct1.size());
                    assert(pNodes1[ele] == pNodes2[count2] ||
                           pNodes1[ele].getParent() == pNodes2[count2] ||
                           pNodes1[ele] == pNodes2[count2].getParent());

                    if (pNodes1[ele] ==
                        pNodes2[count2]
                            .getParent()) {  // old elements have splitted.
                        assert(pNodes2[count2].getParent() ==
                               pNodes2[count2 + NUM_CHILDREN - 1].getParent());
                        pNodes1[ele].setFlag(((OCT_SPLIT << NUM_LEVEL_BITS) |
                                              pNodes1[ele].getLevel()));
                        count2 = count2 + NUM_CHILDREN;

                    } else if (pNodes1[ele].getParent() ==
                               pNodes2[count2]) {  // old elements have coarsen
                        assert(pNodes1[ele].getParent() ==
                               pNodes1[ele + NUM_CHILDREN - 1].getParent());
                        pNodes1[ele].setFlag(((OCT_COARSE << NUM_LEVEL_BITS) |
                                              pNodes1[ele].getLevel()));
                        ele = ele + NUM_CHILDREN - 1;
                        count2++;

                    } else {
                        assert(pNodes1[ele] == pNodes2[count2]);
                        pNodes1[ele].setFlag(
                            ((OCT_NO_CHANGE << NUM_LEVEL_BITS) |
                             pNodes1[ele].getLevel()));
                        count2++;
                    }
                }
            }

            std::vector<ot::TreeNode> balOct2;
            SFC::parSort::SFC_treeSort(balOct1, balOct2, balOct2, balOct2,
                                       ld_tol, m_uiMaxDepth, rootNode,
                                       ROOT_ROTATION, 1, TS_REMOVE_DUPLICATES,
                                       sfK, m_uiCommActive);
            std::swap(balOct1, balOct2);
            balOct2.clear();

            // repartition balOct1 to ensure that it is not partitioned across,
            // children of the same parent.
            enforceSiblingsAreNotPartitioned(balOct1, m_uiCommActive);
            assert(par::test::isUniqueAndSorted(balOct1, m_uiCommActive));
        }
    }

    ot::Mesh *pMesh =
        new ot::Mesh(balOct1, 1, m_uiElementOrder, m_uiCommGlobal,
                     m_uiIsBlockSetup, m_uiScatterMapType, grainSz, ld_tol, sfK,
                     getWeight, blk_tags, blk_tag_sz);
    pMesh->setDomainBounds(m_uiDMinPt, m_uiDMaxPt);

    dendro::logger::info(dendro::logger::Scope{"MESH"},
                         "Finished creating the new mesh during remesh!");
    return pMesh;
}

ot::Mesh *Mesh::ReMeshRepartitioned(unsigned int grainSz, double ld_tol,
                                    unsigned int sfK) {
    std::vector<ot::TreeNode> balOct;

    if (m_uiIsActive) {
        // 1. Build unbalanced octree from refinement flags (local, no comm)
        std::vector<ot::TreeNode> unBalanced;
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ele++) {
            unsigned int remeshFlag =
                (m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS);
            assert(m_uiAllElements[ele].getLevel() != 0);

            if (remeshFlag == OCT_SPLIT) {
                m_uiAllElements[ele].addChildren(unBalanced);
            } else if (remeshFlag == OCT_COARSE) {
                assert(m_uiAllElements[ele].getParent() ==
                       m_uiAllElements[ele + NUM_CHILDREN - 1].getParent());
                unBalanced.push_back(m_uiAllElements[ele].getParent());
                ele = ele + NUM_CHILDREN - 1;
            } else {
                assert(remeshFlag == OCT_NO_CHANGE);
                unBalanced.push_back(m_uiAllElements[ele]);
            }
        }

        // 2. Global SFC sort + 2:1 balance
        ot::TreeNode rootNode(m_uiDim, m_uiMaxDepth);
        SFC::parSort::SFC_treeSort(unBalanced, balOct, balOct, balOct, ld_tol,
                                   m_uiMaxDepth, rootNode, ROOT_ROTATION, 1,
                                   TS_BALANCE_OCTREE, sfK, m_uiCommActive);
        unBalanced.clear();
    }

    // 3. Build a lightweight SFC mesh (E2E only) from the balanced octants.
    //    repartitionMeshGlobal only needs m_uiAllElements + m_uiE2EMapping
    //    + element-ghost scatter map — all provided by E2E_ONLY. The full
    //    E2N / node scatter map / block decomposition gets built once on
    //    the GRAPH partition inside repartitionMeshGlobal, avoiding a
    //    duplicate build on the intermediate SFC mesh.
    ot::Mesh *sfcMesh =
        new ot::Mesh(balOct, 1, m_uiElementOrder, m_uiCommGlobal,
                     m_uiIsBlockSetup, SM_TYPE::E2E_ONLY, grainSz,
                     ld_tol, sfK);
    sfcMesh->setDomainBounds(m_uiDMinPt, m_uiDMaxPt);

    // 4. Apply graph partitioning. After this, sfcMesh is a fully-built
    //    graph-partitioned mesh (E2N, scatter map, blocks all built once
    //    on the new partition).
    sfcMesh->setPartitioningMethod(m_partitionOption);
    sfcMesh->m_uiScatterMapType = m_uiScatterMapType;
    sfcMesh->repartitionMeshGlobal();

    return sfcMesh;
}

void ot::Mesh::getElementalFaceNeighbors(const unsigned int eID,
                                         const unsigned int dir,
                                         unsigned int *lookup) const {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    lookup[0]       = eID;
    lookup[1]       = LOOK_UP_TABLE_DEFAULT;

    unsigned int lk = m_uiE2EMapping[eID * m_uiNumDirections + dir];
    if (lk != LOOK_UP_TABLE_DEFAULT &&
        m_uiAllElements[lk].getLevel() <= m_uiAllElements[eID].getLevel())
        lookup[1] = lk;

    return;
}

void ot::Mesh::getElementalEdgeNeighbors(const unsigned int eID,
                                         const unsigned int dir,
                                         unsigned int *lookup) const {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    lookup[0]      = eID;
    lookup[1]      = LOOK_UP_TABLE_DEFAULT;
    lookup[2]      = LOOK_UP_TABLE_DEFAULT;
    lookup[3]      = LOOK_UP_TABLE_DEFAULT;
    unsigned level = m_uiAllElements[eID].getLevel();

    unsigned int dir1, dir2;
    unsigned int lk = LOOK_UP_TABLE_DEFAULT;

    if (dir == OCT_DIR_LEFT_DOWN) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_DOWN;

    } else if (dir == OCT_DIR_LEFT_UP) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_UP;

    } else if (dir == OCT_DIR_LEFT_FRONT) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_FRONT;

    } else if (dir == OCT_DIR_LEFT_BACK) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_RIGHT_DOWN) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_DOWN;

    } else if (dir == OCT_DIR_RIGHT_UP) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_UP;

    } else if (dir == OCT_DIR_RIGHT_BACK) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_RIGHT_FRONT) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_FRONT;

    } else if (dir == OCT_DIR_UP_BACK) {
        dir1 = OCT_DIR_UP;
        dir2 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_UP_FRONT) {
        dir1 = OCT_DIR_UP;
        dir2 = OCT_DIR_FRONT;

    } else if (dir == OCT_DIR_DOWN_BACK) {
        dir1 = OCT_DIR_DOWN;
        dir2 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_DOWN_FRONT) {
        dir1 = OCT_DIR_DOWN;
        dir2 = OCT_DIR_FRONT;
    }

    lookup[1] = m_uiE2EMapping[lookup[0] * m_uiNumDirections + dir1];

    for (unsigned int i = 0; i < 2; i++) {
        if (lookup[i] != LOOK_UP_TABLE_DEFAULT)
            lookup[i + 2] =
                m_uiE2EMapping[lookup[i] * m_uiNumDirections + dir2];
    }

    for (unsigned int i = 1; i < 4; i++) {
        if (lookup[i] != LOOK_UP_TABLE_DEFAULT &&
            (m_uiAllElements[lookup[i]].getLevel() > level))
            lookup[i] = LOOK_UP_TABLE_DEFAULT;
    }

    return;
}

void ot::Mesh::getElementalVertexNeighbors(const unsigned int eID,
                                           const unsigned int dir,
                                           unsigned int *lookup) const {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    lookup[0] = eID;
    for (unsigned int i = 1; i < NUM_CHILDREN; i++)
        lookup[i] = LOOK_UP_TABLE_DEFAULT;

    unsigned int level = m_uiAllElements[eID].getLevel();
    unsigned int dir1, dir2, dir3;

    if (dir == OCT_DIR_LEFT_DOWN_BACK) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_DOWN;
        dir3 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_RIGHT_DOWN_BACK) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_DOWN;
        dir3 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_LEFT_UP_BACK) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_UP;
        dir3 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_RIGHT_UP_BACK) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_UP;
        dir3 = OCT_DIR_BACK;

    } else if (dir == OCT_DIR_LEFT_DOWN_FRONT) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_DOWN;
        dir3 = OCT_DIR_FRONT;

    } else if (dir == OCT_DIR_RIGHT_DOWN_FRONT) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_DOWN;
        dir3 = OCT_DIR_FRONT;

    } else if (dir == OCT_DIR_LEFT_UP_FRONT) {
        dir1 = OCT_DIR_LEFT;
        dir2 = OCT_DIR_UP;
        dir3 = OCT_DIR_FRONT;

    } else if (dir == OCT_DIR_RIGHT_UP_FRONT) {
        dir1 = OCT_DIR_RIGHT;
        dir2 = OCT_DIR_UP;
        dir3 = OCT_DIR_FRONT;
    }

    lookup[1] = m_uiE2EMapping[lookup[0] * m_uiNumDirections + dir1];

    for (unsigned int i = 0; i < 2; i++) {
        if (lookup[i] != LOOK_UP_TABLE_DEFAULT)
            lookup[i + 2] =
                m_uiE2EMapping[lookup[i] * m_uiNumDirections + dir2];
    }

    for (unsigned int i = 0; i < 4; i++) {
        if (lookup[i] != LOOK_UP_TABLE_DEFAULT)
            lookup[i + 4] =
                m_uiE2EMapping[lookup[i] * m_uiNumDirections + dir3];
    }

    for (unsigned int i = 1; i < NUM_CHILDREN; i++) {
        if (lookup[i] != LOOK_UP_TABLE_DEFAULT &&
            (m_uiAllElements[lookup[i]].getLevel() > level))
            lookup[i] = LOOK_UP_TABLE_DEFAULT;
    }

    return;
}

void ot::Mesh::getElementQMat(unsigned int currentId, double *&qMat,
                              bool isAllocated) const {
    if (!m_uiIsActive) return;

    if (!isAllocated) qMat = new double[m_uiNpE];

    assert(qMat != NULL);

    const unsigned int eleOrder = m_uiElementOrder;
    const unsigned int npe_1d   = eleOrder + 1;
    const unsigned int npe_2d   = (eleOrder + 1) * (eleOrder + 1);
    const unsigned int nPe = (eleOrder + 1) * (eleOrder + 1) * (eleOrder + 1);

    // note that this is because in the reference element interpolation
    // operators are transposed to support on the fly interpolations.
    const DendroScalar *I0 = m_uiRefEl.getIMTChild0();
    const DendroScalar *I1 = m_uiRefEl.getIMTChild1();

    // set qMat to be identity.
    for (unsigned int i = 0; i < m_uiNpE; i++)
        for (unsigned int j = 0; j < m_uiNpE; j++) qMat[i * m_uiNpE + j] = 0.0;

    for (unsigned int i = 0; i < m_uiNpE; i++) qMat[i * m_uiNpE + i] = 1.0;

    bool faceHang[NUM_FACES];
    bool edgeHang[NUM_EDGES];
    unsigned int cnumFace[NUM_FACES];
    unsigned int cnumEdge[NUM_EDGES];

    double *im2D_00 = new double[npe_2d * npe_2d];
    double *im2D_01 = new double[npe_2d * npe_2d];
    double *im2D_10 = new double[npe_2d * npe_2d];
    double *im2D_11 = new double[npe_2d * npe_2d];

    faceHang[0]     = this->isFaceHanging(currentId, OCT_DIR_LEFT, cnumFace[0]);
    faceHang[1] = this->isFaceHanging(currentId, OCT_DIR_RIGHT, cnumFace[1]);
    faceHang[2] = this->isFaceHanging(currentId, OCT_DIR_DOWN, cnumFace[2]);
    faceHang[3] = this->isFaceHanging(currentId, OCT_DIR_UP, cnumFace[3]);
    faceHang[4] = this->isFaceHanging(currentId, OCT_DIR_BACK, cnumFace[4]);
    faceHang[5] = this->isFaceHanging(currentId, OCT_DIR_FRONT, cnumFace[5]);

    edgeHang[0] =
        this->isEdgeHanging(currentId, OCT_DIR_LEFT_DOWN, cnumEdge[0]);
    edgeHang[1] = this->isEdgeHanging(currentId, OCT_DIR_LEFT_UP, cnumEdge[1]);
    edgeHang[2] =
        this->isEdgeHanging(currentId, OCT_DIR_LEFT_BACK, cnumEdge[2]);
    edgeHang[3] =
        this->isEdgeHanging(currentId, OCT_DIR_LEFT_FRONT, cnumEdge[3]);

    edgeHang[4] =
        this->isEdgeHanging(currentId, OCT_DIR_RIGHT_DOWN, cnumEdge[4]);
    edgeHang[5] = this->isEdgeHanging(currentId, OCT_DIR_RIGHT_UP, cnumEdge[5]);
    edgeHang[6] =
        this->isEdgeHanging(currentId, OCT_DIR_RIGHT_BACK, cnumEdge[6]);
    edgeHang[7] =
        this->isEdgeHanging(currentId, OCT_DIR_RIGHT_FRONT, cnumEdge[7]);

    edgeHang[8] =
        this->isEdgeHanging(currentId, OCT_DIR_DOWN_BACK, cnumEdge[8]);
    edgeHang[9] =
        this->isEdgeHanging(currentId, OCT_DIR_DOWN_FRONT, cnumEdge[9]);
    edgeHang[10] =
        this->isEdgeHanging(currentId, OCT_DIR_UP_BACK, cnumEdge[10]);
    edgeHang[11] =
        this->isEdgeHanging(currentId, OCT_DIR_UP_FRONT, cnumEdge[11]);

    const ot::TreeNode *allElements = &(*(m_uiAllElements.begin()));

    // compute the 2d operators.
    kron(I0, I0, im2D_00, npe_1d, npe_1d, npe_1d, npe_1d);
    kron(I0, I1, im2D_01, npe_1d, npe_1d, npe_1d, npe_1d);
    kron(I1, I0, im2D_10, npe_1d, npe_1d, npe_1d, npe_1d);
    kron(I1, I1, im2D_11, npe_1d, npe_1d, npe_1d, npe_1d);

    double *im2D;

    // left
    if (faceHang[0]) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++)
            for (unsigned int j = 0; j < npe_1d; j++) {
                const unsigned int rowId = k * npe_1d * npe_1d + j * npe_1d + 0;
                entry.push_back(rowId);
            }

        if (cnumFace[0] == 0)
            im2D = im2D_00;
        else if (cnumFace[0] == 1)
            im2D = im2D_01;
        else if (cnumFace[0] == 2)
            im2D = im2D_10;
        else {
            assert(cnumFace[0] == 3);
            im2D = im2D_11;
        }

        for (unsigned int i = 0; i < entry.size(); i++)
            for (unsigned int j = 0; j < entry.size(); j++) {
                const unsigned int rid = entry[i];
                const unsigned int cid = entry[j];

                qMat[rid * nPe + cid]  = im2D[i * npe_2d + j];
            }
    }

    // right
    if (faceHang[1]) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++)
            for (unsigned int j = 0; j < npe_1d; j++) {
                const unsigned int rowId =
                    k * npe_1d * npe_1d + j * npe_1d + eleOrder;
                entry.push_back(rowId);
            }

        if (cnumFace[1] == 0)
            im2D = im2D_00;
        else if (cnumFace[1] == 1)
            im2D = im2D_01;
        else if (cnumFace[1] == 2)
            im2D = im2D_10;
        else {
            assert(cnumFace[1] == 3);
            im2D = im2D_11;
        }

        for (unsigned int i = 0; i < entry.size(); i++)
            for (unsigned int j = 0; j < entry.size(); j++) {
                const unsigned int rid = entry[i];
                const unsigned int cid = entry[j];
                qMat[rid * nPe + cid]  = im2D[i * npe_2d + j];
            }
    }

    // down
    if (faceHang[2]) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++)
            for (unsigned int i = 0; i < npe_1d; i++) {
                const unsigned int rowId = k * npe_1d * npe_1d + 0 * npe_1d + i;
                entry.push_back(rowId);
            }

        if (cnumFace[2] == 0)
            im2D = im2D_00;
        else if (cnumFace[2] == 1)
            im2D = im2D_01;
        else if (cnumFace[2] == 2)
            im2D = im2D_10;
        else {
            assert(cnumFace[2] == 3);
            im2D = im2D_11;
        }

        for (unsigned int i = 0; i < entry.size(); i++)
            for (unsigned int j = 0; j < entry.size(); j++) {
                const unsigned int rid = entry[i];
                const unsigned int cid = entry[j];
                qMat[rid * nPe + cid]  = im2D[i * npe_2d + j];
            }
    }

    // up
    if (faceHang[3]) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++)
            for (unsigned int i = 0; i < npe_1d; i++) {
                const unsigned int rowId =
                    k * npe_1d * npe_1d + (eleOrder)*npe_1d + i;
                entry.push_back(rowId);
            }

        if (cnumFace[3] == 0)
            im2D = im2D_00;
        else if (cnumFace[3] == 1)
            im2D = im2D_01;
        else if (cnumFace[3] == 2)
            im2D = im2D_10;
        else {
            assert(cnumFace[3] == 3);
            im2D = im2D_11;
        }

        for (unsigned int i = 0; i < entry.size(); i++)
            for (unsigned int j = 0; j < entry.size(); j++) {
                const unsigned int rid = entry[i];
                const unsigned int cid = entry[j];
                qMat[rid * nPe + cid]  = im2D[i * npe_2d + j];
            }
    }

    // back
    if (faceHang[4]) {
        // std::cout<<" current: "<<allElements[currentId]<<" back face hanging:
        // "<<cnumFace[4]<<std::endl;
        std::vector<unsigned int> entry;
        for (unsigned int j = 0; j < npe_1d; j++)
            for (unsigned int i = 0; i < npe_1d; i++) {
                const unsigned int rowId = 0 * npe_1d * npe_1d + j * npe_1d + i;
                entry.push_back(rowId);
            }

        if (cnumFace[4] == 0)
            im2D = im2D_00;
        else if (cnumFace[4] == 1)
            im2D = im2D_01;
        else if (cnumFace[4] == 2)
            im2D = im2D_10;
        else {
            assert(cnumFace[4] == 3);
            im2D = im2D_11;
        }

        for (unsigned int i = 0; i < entry.size(); i++)
            for (unsigned int j = 0; j < entry.size(); j++) {
                const unsigned int rid = entry[i];
                const unsigned int cid = entry[j];
                qMat[rid * nPe + cid]  = im2D[i * npe_2d + j];
            }
    }

    // front
    if (faceHang[5]) {
        std::vector<unsigned int> entry;
        for (unsigned int j = 0; j < npe_1d; j++)
            for (unsigned int i = 0; i < npe_1d; i++) {
                const unsigned int rowId =
                    (eleOrder)*npe_1d * npe_1d + j * npe_1d + i;
                entry.push_back(rowId);
            }

        if (cnumFace[5] == 0)
            im2D = im2D_00;
        else if (cnumFace[5] == 1)
            im2D = im2D_01;
        else if (cnumFace[5] == 2)
            im2D = im2D_10;
        else {
            assert(cnumFace[5] == 3);
            im2D = im2D_11;
        }

        for (unsigned int i = 0; i < entry.size(); i++)
            for (unsigned int j = 0; j < entry.size(); j++) {
                const unsigned int rid = entry[i];
                const unsigned int cid = entry[j];
                qMat[rid * nPe + cid]  = im2D[i * npe_2d + j];
            }
    }

    // OCT_DIR_LEFT_DOWN
    if (edgeHang[0] && (!faceHang[0] && !faceHang[2])) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++) {
            const unsigned int rowId = k * npe_1d * npe_1d + 0 * npe_1d + 0;
            entry.push_back(rowId);
        }

        if (cnumEdge[0] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[0] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_LEFT_UP
    if (edgeHang[1] && (!faceHang[0] && !faceHang[3])) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++) {
            const unsigned int rowId =
                k * npe_1d * npe_1d + eleOrder * npe_1d + 0;
            entry.push_back(rowId);
        }

        if (cnumEdge[1] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[1] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_LEFT_BACK
    if (edgeHang[2] && (!faceHang[0] && !faceHang[4])) {
        std::vector<unsigned int> entry;
        for (unsigned int j = 0; j < npe_1d; j++) {
            const unsigned int rowId = 0 * npe_1d * npe_1d + j * npe_1d + 0;
            entry.push_back(rowId);
        }
        if (cnumEdge[2] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[2] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_LEFT_FRONT
    if (edgeHang[3] && (!faceHang[0] && !faceHang[5])) {
        std::vector<unsigned int> entry;
        for (unsigned int j = 0; j < npe_1d; j++) {
            const unsigned int rowId =
                eleOrder * npe_1d * npe_1d + j * npe_1d + 0;
            entry.push_back(rowId);
        }

        if (cnumEdge[3] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[3] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_RIGHT_DOWN
    if (edgeHang[4] && (!faceHang[1] && !faceHang[2])) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++) {
            const unsigned int rowId =
                k * npe_1d * npe_1d + 0 * npe_1d + eleOrder;
            entry.push_back(rowId);
        }

        if (cnumEdge[4] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[4] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_RIGHT_UP
    if (edgeHang[5] && (!faceHang[1] && !faceHang[3])) {
        std::vector<unsigned int> entry;
        for (unsigned int k = 0; k < npe_1d; k++) {
            const unsigned int rowId =
                k * npe_1d * npe_1d + eleOrder * npe_1d + eleOrder;
            entry.push_back(rowId);
        }

        if (cnumEdge[5] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[5] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_RIGHT_BACK
    if (edgeHang[6] && (!faceHang[1] && !faceHang[4])) {
        std::vector<unsigned int> entry;
        for (unsigned int j = 0; j < npe_1d; j++) {
            const unsigned int rowId =
                0 * npe_1d * npe_1d + j * npe_1d + eleOrder;
            entry.push_back(rowId);
        }

        if (cnumEdge[6] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[6] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_RIGHT_FRONT
    if (edgeHang[7] && (!faceHang[1] && !faceHang[5])) {
        std::vector<unsigned int> entry;
        for (unsigned int j = 0; j < npe_1d; j++) {
            const unsigned int rowId =
                eleOrder * npe_1d * npe_1d + j * npe_1d + eleOrder;
            entry.push_back(rowId);
        }

        if (cnumEdge[7] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[7] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_DOWN_BACK
    if (edgeHang[8] && (!faceHang[2] && !faceHang[4])) {
        std::vector<unsigned int> entry;
        for (unsigned int i = 0; i < npe_1d; i++) {
            const unsigned int rowId = 0 * npe_1d * npe_1d + 0 * npe_1d + i;
            entry.push_back(rowId);
        }

        if (cnumEdge[8] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[8] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_DOWN_FRONT
    if (edgeHang[9] && (!faceHang[2] && !faceHang[5])) {
        std::vector<unsigned int> entry;
        for (unsigned int i = 0; i < npe_1d; i++) {
            const unsigned int rowId =
                eleOrder * npe_1d * npe_1d + 0 * npe_1d + i;
            entry.push_back(rowId);
        }

        if (cnumEdge[9] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[9] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_UP_BACK
    if (edgeHang[10] && (!faceHang[3] && !faceHang[4])) {
        std::vector<unsigned int> entry;
        for (unsigned int i = 0; i < npe_1d; i++) {
            const unsigned int rowId =
                0 * npe_1d * npe_1d + eleOrder * npe_1d + i;
            entry.push_back(rowId);
        }

        if (cnumEdge[10] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[10] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // OCT_DIR_UP_FRONT
    if (edgeHang[11] && (!faceHang[3] && !faceHang[5])) {
        std::vector<unsigned int> entry;
        for (unsigned int i = 0; i < npe_1d; i++) {
            const unsigned int rowId =
                eleOrder * npe_1d * npe_1d + eleOrder * npe_1d + i;
            entry.push_back(rowId);
        }

        if (cnumEdge[11] == 0) {
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I0[i * npe_1d + j];
                }

        } else {
            assert(cnumEdge[11] == 1);
            for (unsigned int i = 0; i < entry.size(); i++)
                for (unsigned int j = 0; j < entry.size(); j++) {
                    const unsigned int rid = entry[i];
                    const unsigned int cid = entry[j];

                    qMat[rid * nPe + cid]  = I1[i * npe_1d + j];
                }
        }
    }

    // if(allElements[currentId].getLevel()>3)
    // {
    //     const char* face_dir[]
    //     ={"OCT_DIR_LEFT","OCT_DIR_RIGHT","OCT_DIR_DOWN","OCT_DIR_UP","OCT_DIR_BACK","OCT_DIR_FRONT"};
    //     const char* edge_dir[]
    //     ={"OCT_DIR_LEFT_DOWN","OCT_DIR_LEFT_UP","OCT_DIR_LEFT_BACK","OCT_DIR_LEFT_FRONT","OCT_DIR_RIGHT_DOWN","OCT_DIR_RIGHT_UP","OCT_DIR_RIGHT_BACK","OCT_DIR_RIGHT_FRONT","OCT_DIR_DOWN_BACK","OCT_DIR_DOWN_FRONT","OCT_DIR_UP_BACK","OCT_DIR_UP_FRONT"};

    //     for(unsigned int dir = 0; dir<NUM_FACES; dir++)
    //     {
    //         if(faceHang[dir])
    //         std::cout<<" current : "<<allElements[currentId]<<"face
    //         "<<face_dir[dir]<<" is hanging: "<<faceHang[dir]<<" cnum:
    //         "<<cnumFace[dir]<<" owner :
    //         "<<allElements[m_uiE2EMapping[currentId*NUM_FACES+dir]]<<std::endl;
    //     }
    //     std::cout<<"RD Mat\n";
    //     for(unsigned int i=0;i<nPe;i++)
    //     {
    //         for(unsigned int j=0;j<nPe;j++)
    //         {
    //             std::cout<<" "<<qMat[i*nPe+j];
    //         }
    //         std::cout<<"\n";
    //     }

    // }

    delete[] im2D_00;
    delete[] im2D_01;
    delete[] im2D_10;
    delete[] im2D_11;
}

void ot::Mesh::computeElementOwnerRanks(
    std::vector<unsigned int> &elementOwner) {
    // should not be called if the mesh is not active
    if (!m_uiIsActive) return;

    elementOwner.resize(m_uiAllElements.size());
    for (unsigned int e = 0; e < m_uiAllElements.size(); e++)
        elementOwner[e] = m_uiActiveRank;

    std::vector<ot::SearchKey> ghostElements;
    std::vector<ot::SearchKey>::iterator itSKey;
    for (unsigned int e = m_uiElementPreGhostBegin; e < m_uiElementPreGhostEnd;
         e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int e = m_uiElementPostGhostBegin;
         e < m_uiElementPostGhostEnd; e++) {
        itSKey = ghostElements.emplace(ghostElements.end(),
                                       ot::SearchKey(m_uiAllElements[e]));
        itSKey->addOwner(e);
    }

    for (unsigned int p = 0; p < m_uiActiveNpes; p++)
        ghostElements.emplace(
            ghostElements.end(),
            ot::SearchKey(m_uiLocalSplitterElements[2 * p + 1]));

    std::vector<ot::SearchKey> tmpSkeys;
    ot::SearchKey rootSKey(m_uiDim, m_uiMaxDepth);

    SFC::seqSort::SFC_treeSort(&(*(ghostElements.begin())),
                               ghostElements.size(), tmpSkeys, tmpSkeys,
                               tmpSkeys, m_uiMaxDepth, m_uiMaxDepth, rootSKey,
                               ROOT_ROTATION, 1, TS_SORT_ONLY);

    tmpSkeys.clear();
    SearchKey tmpSkey;
    unsigned int skip;
    for (unsigned int e = 0; e < (ghostElements.size()); e++) {
        skip    = 1;
        tmpSkey = ghostElements[e];
        while (((e + skip) < ghostElements.size()) &&
               (ghostElements[e] == ghostElements[e + skip])) {
            if (ghostElements[e + skip].getOwner() >= 0)
                tmpSkey.addOwner(ghostElements[e + skip].getOwner());
            skip++;
        }
        tmpSkeys.push_back(tmpSkey);
        assert(skip <= 2);
        e += (skip - 1);
    }

    std::swap(ghostElements, tmpSkeys);
    tmpSkeys.clear();

    unsigned int gCount = 0;
    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        while (
            gCount < ghostElements.size() &&
            (ghostElements[gCount] != m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;

            gCount++;
        }

        if (gCount < ghostElements.size() &&
            (ghostElements[gCount] == m_uiLocalSplitterElements[2 * p + 1])) {
            if (ghostElements[gCount].getOwner() >= 0)
                elementOwner[ghostElements[gCount].getOwner()] = p;
            gCount++;
        }
    }
}

void ot::Mesh::getElementCoordinates(unsigned int eleID, double *coords) const {
    ot::TreeNode tmpNode = m_uiAllElements[eleID];
    double x, y, z, dx, dy, dz, sz;

    sz = (double)(tmpNode.maxX() - tmpNode.minX());

    x  = tmpNode.minX();
    y  = tmpNode.minY();
    z  = tmpNode.minZ();

    dx = sz / (m_uiElementOrder);
    dy = sz / (m_uiElementOrder);
    dz = sz / (m_uiElementOrder);

    for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++)
        for (unsigned int j = 0; j < (m_uiElementOrder + 1); j++)
            for (unsigned int i = 0; i < (m_uiElementOrder + 1); i++) {
                coords[(k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i) *
                           m_uiDim +
                       0] = x + i * dx;
                coords[(k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i) *
                           m_uiDim +
                       1] = y + j * dy;
                coords[(k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
                        j * (m_uiElementOrder + 1) + i) *
                           m_uiDim +
                       2] = z + k * dz;
            }
}

void ot::Mesh::setOctreeRefineFlags(unsigned int *flags, unsigned int sz) {
    if (!m_uiIsActive) return;

    assert(sz == (m_uiElementLocalEnd - m_uiElementLocalBegin));
    for (unsigned int ele = m_uiElementLocalBegin; ele < m_uiElementLocalEnd;
         ele++) {
        m_uiAllElements[ele].setFlag(
            ((flags[(ele - m_uiElementLocalBegin)] << NUM_LEVEL_BITS) |
             m_uiAllElements[ele].getLevel()));
        assert((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
               flags[(ele - m_uiElementLocalBegin)]);
    }
}

void ot::Mesh::SET_FACE_TO_ELEMENT_MAP(unsigned int ele, unsigned int dir,
                                       unsigned int dirOp, unsigned int dir1,
                                       unsigned int dir2) {
    const unsigned int l1     = m_uiAllElements[ele].getLevel();
    const unsigned int lookup = m_uiE2EMapping[ele * NUM_FACES + dir];
    if (lookup != LOOK_UP_TABLE_DEFAULT) {
        const unsigned int l2 = m_uiAllElements[lookup].getLevel();
        if (l1 == l2) {
            if (ele < lookup) {
                m_uiF2EMap[(ele * NUM_FACES + dir) + 0] =
                    (dir << ODA_FLAGS_TOTAL);
                m_uiF2EMap[(ele * NUM_FACES + dir) + 1] = ele;
                m_uiF2EMap[(ele * NUM_FACES + dir) + 2] = lookup;
                // unique_faces.push_back((ele*NUM_FACES+dir));

            } else {
                m_uiF2EMap[(ele * NUM_FACES + dir) + 0] =
                    (dirOp << ODA_FLAGS_TOTAL);
                m_uiF2EMap[(lookup * NUM_FACES + dirOp) + 1] = lookup;
                m_uiF2EMap[(lookup * NUM_FACES + dirOp) + 2] = ele;
                // unique_faces.push_back((lookup*NUM_FACES+dirOp));
            }

        } else if (l1 > l2) {
            // winner, is lookup octant.
            m_uiF2EMap[(ele * NUM_FACES + dir) + 0] =
                (dirOp << ODA_FLAGS_TOTAL);
            m_uiF2EMap[(lookup * NUM_FACES + dirOp) + 1] = lookup;
            // unique_faces.push_back((lookup*NUM_FACES+dirOp));

            // in the morton ordering.
            const unsigned int lookup0 =
                m_uiE2EMapping[lookup * NUM_FACES + dirOp];
            const unsigned int lookup1 =
                m_uiE2EMapping[lookup0 * NUM_FACES + dir2];
            const unsigned int lookup2 =
                m_uiE2EMapping[lookup0 * NUM_FACES + dir1];
            const unsigned int lookup3 =
                m_uiE2EMapping[lookup2 * NUM_FACES + dir2];

            assert(lookup0 != LOOK_UP_TABLE_DEFAULT);
            assert(lookup1 != LOOK_UP_TABLE_DEFAULT);
            assert(lookup2 != LOOK_UP_TABLE_DEFAULT);
            assert(lookup3 != LOOK_UP_TABLE_DEFAULT);

            m_uiF2EMap[(lookup * NUM_FACES + dirOp) + 2] = lookup0;
            m_uiF2EMap[(lookup * NUM_FACES + dirOp) + 3] = lookup1;
            m_uiF2EMap[(lookup * NUM_FACES + dirOp) + 4] = lookup2;
            m_uiF2EMap[(lookup * NUM_FACES + dirOp) + 5] = lookup3;

        } else {
            assert(l1 < l2);

            // winner, is ele octant.
            m_uiF2EMap[(ele * NUM_FACES + dir) + 0] = (dir << ODA_FLAGS_TOTAL);
            m_uiF2EMap[(ele * NUM_FACES + dir) + 1] = ele;
            // unique_faces.push_back((ele*NUM_FACES+dir));

            // in the morton ordering.
            const unsigned int lookup0              = lookup;
            const unsigned int lookup1 =
                m_uiE2EMapping[lookup0 * NUM_FACES + dir2];
            const unsigned int lookup2 =
                m_uiE2EMapping[lookup0 * NUM_FACES + dir1];
            const unsigned int lookup3 =
                m_uiE2EMapping[lookup2 * NUM_FACES + dir2];

            assert(lookup0 != LOOK_UP_TABLE_DEFAULT);
            assert(lookup1 != LOOK_UP_TABLE_DEFAULT);
            assert(lookup2 != LOOK_UP_TABLE_DEFAULT);
            assert(lookup3 != LOOK_UP_TABLE_DEFAULT);

            m_uiF2EMap[(ele * NUM_FACES + dir) + 2] = lookup0;
            m_uiF2EMap[(ele * NUM_FACES + dir) + 3] = lookup1;
            m_uiF2EMap[(ele * NUM_FACES + dir) + 4] = lookup2;
            m_uiF2EMap[(ele * NUM_FACES + dir) + 5] = lookup3;
        }

    } else {
        m_uiF2EMap[(ele * NUM_FACES + dir) + 0] = (dir << ODA_FLAGS_TOTAL);
        m_uiF2EMap[(ele * NUM_FACES + dir) + 1] = ele;
        // unique_faces.push_back((ele*NUM_FACES+dir));
    }
}

void ot::Mesh::buildF2EMap() {
    // assumes we have build the e2e and e2n mapping.
    m_uiF2EMap.clear();
    //
    // m_uiF2EMap structure.
    // m_uiF2EMap[fid][0] face dir relative to the owner element.
    // m_uiF2EMap[fid][1] face owner element.
    // m_uiF2EMap[fid][2] and beyond shared elements [2,5] is the elements are
    // face hanging.
    //
    //

    m_uiF2EMap.resize((m_uiNumTotalElements * NUM_FACES) * F2E_MAP_OFFSET,
                      LOOK_UP_TABLE_DEFAULT);

    std::vector<unsigned int> unique_faces;
    unsigned int dir, dirOp, dir1, dir2, lookup;

    unsigned int g1Count = 0;
    for (unsigned int ele = m_uiElementPreGhostBegin;
         ele < m_uiElementPostGhostEnd; ele++) {
        if ((ele < m_uiElementLocalBegin || ele >= m_uiElementLocalEnd) &&
            (ele != m_uiGhostElementRound1Index[g1Count]))
            continue;
        else
            g1Count++;

        // 1. OCT_DIR_LEFT

        dir   = OCT_DIR_LEFT;
        dirOp = OCT_DIR_RIGHT;
        dir1  = OCT_DIR_FRONT;
        dir2  = OCT_DIR_UP;

        SET_FACE_TO_ELEMENT_MAP(ele, dir, dirOp, dir1, dir2);

        // 2. OCT_DIR_RIGHT
        dir   = OCT_DIR_RIGHT;
        dirOp = OCT_DIR_LEFT;

        dir1  = OCT_DIR_FRONT;
        dir2  = OCT_DIR_UP;

        SET_FACE_TO_ELEMENT_MAP(ele, dir, dirOp, dir1, dir2);

        // 3. OCT_DIR_DOWN
        dir   = OCT_DIR_DOWN;
        dirOp = OCT_DIR_UP;

        dir1  = OCT_DIR_FRONT;
        dir2  = OCT_DIR_RIGHT;

        SET_FACE_TO_ELEMENT_MAP(ele, dir, dirOp, dir1, dir2);

        // 4. OCT_DIR_UP
        dir   = OCT_DIR_UP;
        dirOp = OCT_DIR_DOWN;

        dir1  = OCT_DIR_FRONT;
        dir2  = OCT_DIR_RIGHT;

        SET_FACE_TO_ELEMENT_MAP(ele, dir, dirOp, dir1, dir2);

        // 5. OCT_DIR_BACK
        dir   = OCT_DIR_BACK;
        dirOp = OCT_DIR_FRONT;

        dir1  = OCT_DIR_UP;
        dir2  = OCT_DIR_RIGHT;

        SET_FACE_TO_ELEMENT_MAP(ele, dir, dirOp, dir1, dir2);

        // 6. OCT_DIR_FRONT
        dir   = OCT_DIR_FRONT;
        dirOp = OCT_DIR_BACK;

        dir1  = OCT_DIR_UP;
        dir2  = OCT_DIR_RIGHT;

        SET_FACE_TO_ELEMENT_MAP(ele, dir, dirOp, dir1, dir2);
    }

    DendroIntL uniqueFaceCount = 0;
    for (unsigned int ele = m_uiElementPreGhostBegin;
         ele < m_uiElementPostGhostEnd; ele++) {
        if (m_uiF2EMap[ele * NUM_FACES + 0] != LOOK_UP_TABLE_DEFAULT)
            uniqueFaceCount++;
    }

    std::vector<unsigned int> tmpMap;
    std::swap(tmpMap, m_uiF2EMap);

    m_uiF2EMap.clear();
    m_uiF2EMap.resize(uniqueFaceCount, LOOK_UP_TABLE_DEFAULT);

    uniqueFaceCount    = 0;
    unsigned int owner = 0;
    bool isIndependent = true;
    bool isWdependent  = true;
    bool isBoundary    = false;
    for (unsigned int ele = m_uiElementPreGhostBegin;
         ele < m_uiElementPostGhostEnd; ele++) {
        if (tmpMap[ele * NUM_FACES + 0] != LOOK_UP_TABLE_DEFAULT) {
            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0] =
                tmpMap[ele * NUM_FACES + 0];
            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 1] =
                tmpMap[ele * NUM_FACES + 1];
            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 2] =
                tmpMap[ele * NUM_FACES + 2];
            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 3] =
                tmpMap[ele * NUM_FACES + 3];
            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 4] =
                tmpMap[ele * NUM_FACES + 4];
            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 5] =
                tmpMap[ele * NUM_FACES + 5];

            dir = (m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0] >>
                   ODA_FLAGS_TOTAL);

            switch (dir) {
                case OCT_DIR_LEFT:

                    owner = m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 1];
                    isBoundary    = false;
                    isIndependent = true;
                    isWdependent  = false;

                    if ((m_uiAllElements[owner].minX() == 0)) {
                        isBoundary = true;
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          2] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          3] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          4] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          5] == LOOK_UP_TABLE_DEFAULT);
                    }

                    if ((owner >= m_uiElementLocalBegin &&
                         owner < m_uiElementLocalEnd)) {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++) {
                                if (!isNodeLocal(owner, 0, j, k)) {
                                    isIndependent = false;
                                    break;
                                }
                            }

                        if (isIndependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_INDEPENDENT_FLAG_BIT);
                        else
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    } else {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++) {
                                if (isNodeLocal(owner, 0, j, k)) {
                                    isWdependent = true;
                                    break;
                                }
                            }
                        if (isWdependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    }

                    if ((isIndependent || isWdependent) && (isBoundary))
                        binOp::setBit(
                            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0],
                            ODA_W_BOUNDARY_FLAG_BIT);

                    break;

                case OCT_DIR_RIGHT:

                    owner = m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 1];
                    isBoundary    = false;
                    isIndependent = true;
                    isWdependent  = false;

                    if ((m_uiAllElements[owner].maxX() ==
                         (1u << m_uiMaxDepth))) {
                        isBoundary = true;
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          2] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          3] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          4] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          5] == LOOK_UP_TABLE_DEFAULT);
                    }

                    if ((owner >= m_uiElementLocalBegin &&
                         owner < m_uiElementLocalEnd)) {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++) {
                                if (!isNodeLocal(owner, m_uiElementOrder, j,
                                                 k)) {
                                    isIndependent = false;
                                    break;
                                }
                            }

                        if (isIndependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_INDEPENDENT_FLAG_BIT);
                        else
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    } else {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                                 j++) {
                                if (isNodeLocal(owner, m_uiElementOrder, j,
                                                k)) {
                                    isWdependent = true;
                                    break;
                                }
                            }
                        if (isWdependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    }

                    if ((isIndependent || isWdependent) && (isBoundary))
                        binOp::setBit(
                            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0],
                            ODA_W_BOUNDARY_FLAG_BIT);

                    break;

                case OCT_DIR_DOWN:

                    owner = m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 1];
                    isBoundary    = false;
                    isIndependent = true;
                    isWdependent  = false;

                    if ((m_uiAllElements[owner].minY() == 0)) {
                        isBoundary = true;
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          2] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          3] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          4] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          5] == LOOK_UP_TABLE_DEFAULT);
                    }

                    if ((owner >= m_uiElementLocalBegin &&
                         owner < m_uiElementLocalEnd)) {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (!isNodeLocal(owner, i, 0, k)) {
                                    isIndependent = false;
                                    break;
                                }
                            }

                        if (isIndependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_INDEPENDENT_FLAG_BIT);
                        else
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    } else {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (isNodeLocal(owner, i, 0, k)) {
                                    isWdependent = true;
                                    break;
                                }
                            }
                        if (isWdependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    }

                    if ((isIndependent || isWdependent) && (isBoundary))
                        binOp::setBit(
                            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0],
                            ODA_W_BOUNDARY_FLAG_BIT);

                    break;

                case OCT_DIR_UP:

                    owner = m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 1];
                    isBoundary    = false;
                    isIndependent = true;
                    isWdependent  = false;

                    if ((m_uiAllElements[owner].maxY() ==
                         (1u << m_uiMaxDepth))) {
                        isBoundary = true;
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          2] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          3] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          4] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          5] == LOOK_UP_TABLE_DEFAULT);
                    }

                    if ((owner >= m_uiElementLocalBegin &&
                         owner < m_uiElementLocalEnd)) {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (!isNodeLocal(owner, i, m_uiElementOrder,
                                                 k)) {
                                    isIndependent = false;
                                    break;
                                }
                            }

                        if (isIndependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_INDEPENDENT_FLAG_BIT);
                        else
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    } else {
                        for (unsigned int k = 0; k < (m_uiElementOrder + 1);
                             k++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (isNodeLocal(owner, i, m_uiElementOrder,
                                                k)) {
                                    isWdependent = true;
                                    break;
                                }
                            }
                        if (isWdependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    }

                    if ((isIndependent || isWdependent) && (isBoundary))
                        binOp::setBit(
                            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0],
                            ODA_W_BOUNDARY_FLAG_BIT);

                    break;

                case OCT_DIR_BACK:

                    owner = m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 1];
                    isBoundary    = false;
                    isIndependent = true;
                    isWdependent  = false;

                    if ((m_uiAllElements[owner].minZ() == 0)) {
                        isBoundary = true;
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          2] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          3] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          4] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          5] == LOOK_UP_TABLE_DEFAULT);
                    }

                    if ((owner >= m_uiElementLocalBegin &&
                         owner < m_uiElementLocalEnd)) {
                        for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                             j++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (!isNodeLocal(owner, i, j, 0)) {
                                    isIndependent = false;
                                    break;
                                }
                            }

                        if (isIndependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_INDEPENDENT_FLAG_BIT);
                        else
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    } else {
                        for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                             j++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (isNodeLocal(owner, i, j, 0)) {
                                    isWdependent = true;
                                    break;
                                }
                            }
                        if (isWdependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    }

                    if ((isIndependent || isWdependent) && (isBoundary))
                        binOp::setBit(
                            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0],
                            ODA_W_BOUNDARY_FLAG_BIT);

                    break;

                case OCT_DIR_FRONT:

                    owner = m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 1];
                    isBoundary    = false;
                    isIndependent = true;
                    isWdependent  = false;

                    if ((m_uiAllElements[owner].maxZ() == 0)) {
                        isBoundary = true;
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          2] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          3] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          4] == LOOK_UP_TABLE_DEFAULT);
                        assert(m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                          5] == LOOK_UP_TABLE_DEFAULT);
                    }

                    if ((owner >= m_uiElementLocalBegin &&
                         owner < m_uiElementLocalEnd)) {
                        for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                             j++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (!isNodeLocal(owner, i, j,
                                                 m_uiElementOrder)) {
                                    isIndependent = false;
                                    break;
                                }
                            }

                        if (isIndependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_INDEPENDENT_FLAG_BIT);
                        else
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    } else {
                        for (unsigned int j = 0; j < (m_uiElementOrder + 1);
                             j++)
                            for (unsigned int i = 0; i < (m_uiElementOrder + 1);
                                 i++) {
                                if (isNodeLocal(owner, i, j,
                                                m_uiElementOrder)) {
                                    isWdependent = true;
                                    break;
                                }
                            }
                        if (isWdependent)
                            binOp::setBit(
                                m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) +
                                           0],
                                ODA_W_DEPENDENT_FLAG_BIT);
                    }

                    if ((isIndependent || isWdependent) && (isBoundary))
                        binOp::setBit(
                            m_uiF2EMap[uniqueFaceCount * (F2E_MAP_OFFSET) + 0],
                            ODA_W_BOUNDARY_FLAG_BIT);

                    break;

                default:
                    std::cout << "m_uiRank: " << m_uiGlobalRank
                              << " f2e map falg set error in " << __func__
                              << std::endl;
                    break;
            }

            uniqueFaceCount++;
        }
    }

    m_uiIsF2ESetup = true;

    if (!m_uiActiveRank) std::cout << "F2E Ended " << std::endl;
}

EType ot::Mesh::getElementType(unsigned int eleID) {
    const unsigned int nx  = m_uiElementOrder + 1;
    const unsigned int ny  = m_uiElementOrder + 1;
    const unsigned int nz  = m_uiElementOrder + 1;
    const unsigned int nPe = nx * ny * nz;

    bool isIndependent     = true;
    bool isWritable        = false;

    if (eleID < m_uiAllElements.size()) {
        for (unsigned int node = 0; node < nPe; node++) {
            if ((m_uiE2NMapping_CG[eleID * nPe + node] >= m_uiNodeLocalBegin) &&
                (m_uiE2NMapping_CG[eleID * nPe + node] < m_uiNodeLocalEnd)) {
                isWritable = true;
                break;
            }
        }

        for (unsigned int node = 0; node < nPe; node++) {
            if ((m_uiE2NMapping_CG[eleID * nPe + node] < m_uiNodeLocalBegin) ||
                (m_uiE2NMapping_CG[eleID * nPe + node] >= m_uiNodeLocalEnd)) {
                isIndependent = false;
                break;
            }
        }

        if (isIndependent)
            return EType::INDEPENDENT;
        else if (!isIndependent && isWritable)
            return EType::W_DEPENDENT;
        else
            return EType::UNKWON;
    }

    return EType::UNKWON;
}

#if 0
    void Mesh::computeSMSpecialPts()
    {
         // Note: this function is specifically written to find the last point for 4th order elements in finite differencing. 
         if(!m_uiIsActive || m_uiElementOrder!=4 ) return;
        
         ot::TreeNode blkNode;
         unsigned int sz,lx,ly,lz,regLev,ei,ej,ek,eleIndexMax,eleIndexMin,offset,paddWidth,lookUp,uzip_1d,lookup1,bflag;
         const ot::TreeNode * pNodes = &(*(m_uiAllElements.begin()));
         std::vector<ot::SearchKey> m_uiUnzip_3pt;
         ot::SearchKey tmpSKey;   

         unsigned int child[NUM_CHILDREN];

         std::vector<ot::TreeNode> localPart;
         for(unsigned int i=m_uiElementLocalBegin;i<m_uiElementLocalEnd;i++)
            localPart.push_back(m_uiAllElements[i]);

         //treeNodesTovtk(localPart,m_uiActiveRank,"LocalPart",false);   
         //MPI_Barrier(m_uiCommActive); if(!m_uiActiveRank) std::cout<<"3pt SM begin"<<std::endl;

         for(unsigned int blk=0;blk<m_uiLocalBlockList.size();blk++)
         {
            blkNode = m_uiLocalBlockList[blk].getBlockNode();
            assert(blkNode.maxX()<=m_uiMeshDomain_max && blkNode.minX()>=m_uiMeshDomain_min);
            const unsigned regLev = m_uiLocalBlockList[blk].getRegularGridLev();
            //blkNpe_1D=m_uiElementOrder*(1u<<(regLev-blkNode.getLevel()))+1+2*GHOST_WIDTH;
            //std::cout<<"rank: "<<m_uiActiveRank<<" -- blkNpw_1D: "<<blkNpe_1D<<" blkNode: "<<blkNode<<" regLev: "<<regLev<<std::endl;

            sz=1u<<(m_uiMaxDepth-regLev);
            eleIndexMax=(1u<<(regLev-blkNode.getLevel()))-1;
            eleIndexMin=0;
            assert(eleIndexMax>=eleIndexMin);

            lx=m_uiLocalBlockList[blk].getAllocationSzX();
            ly=m_uiLocalBlockList[blk].getAllocationSzY();
            lz=m_uiLocalBlockList[blk].getAllocationSzZ();
            offset=m_uiLocalBlockList[blk].getOffset();
            paddWidth=m_uiLocalBlockList[blk].get1DPadWidth();
            uzip_1d = m_uiLocalBlockList[blk].get1DArraySize();

            const unsigned int nx = m_uiElementOrder+1;
            const unsigned int ny = m_uiElementOrder+1;
            const unsigned int nz = m_uiElementOrder+1;
            const unsigned int N = nx;

            unsigned int x,y,z,hx;
            bflag = m_uiLocalBlockList[blk].getBlkNodeFlag();

            // visit each block and compute the missing 3rd point if it is actually missing. 
            for(unsigned int elem=m_uiLocalBlockList[blk].getLocalElementBegin();elem<m_uiLocalBlockList[blk].getLocalElementEnd();elem++)
            {
                ei=(pNodes[elem].getX()-blkNode.getX())>>(m_uiMaxDepth-regLev);
                ej=(pNodes[elem].getY()-blkNode.getY())>>(m_uiMaxDepth-regLev);
                ek=(pNodes[elem].getZ()-blkNode.getZ())>>(m_uiMaxDepth-regLev);

                //std::cout<<"blk: "<<blk<<" : "<<blkNode<<" ek: "<<(ek)<<" ej: "<<(ej)<<" ei: "<<(ei)<<" elem: "<<m_uiAllElements[elem]<<std::endl;
                assert(pNodes[elem].getLevel()==regLev); // this is enforced by block construction
            	const unsigned int lsz = 1u << (m_uiMaxDepth -pNodes[elem].getLevel());
	            const unsigned int lszby2 = lsz>>1u;
                
                if((pNodes[elem].minX()==blkNode.minX()))
                {
                    const unsigned int dir = OCT_DIR_LEFT;
                    assert(ei==eleIndexMin);
                    lookUp=m_uiE2EMapping[elem*m_uiNumDirections+dir];
                    if(lookUp!=LOOK_UP_TABLE_DEFAULT && pNodes[lookUp].getLevel() > pNodes[elem].getLevel())
                    {  
                       
                        child[1]=lookUp;
                        child[3]=m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_UP];
                        assert(child[3]!=LOOK_UP_TABLE_DEFAULT);
                        child[5]=m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[5]!=LOOK_UP_TABLE_DEFAULT);
                        child[7]=m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[7]!=LOOK_UP_TABLE_DEFAULT);

                        child[0]=m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_LEFT];
                        child[2]=m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_LEFT];
                        child[4]=m_uiE2EMapping[child[5]*m_uiNumDirections+OCT_DIR_LEFT];
                        child[6]=m_uiE2EMapping[child[7]*m_uiNumDirections+OCT_DIR_LEFT];

                        
                        
                        bool missed_child =false;
                        for(unsigned int c=0; c<NUM_CHILDREN;c++)
                        {
                            if(child[c] == LOOK_UP_TABLE_DEFAULT || pNodes[child[c]].getLevel()!=pNodes[lookUp].getLevel() ||  !m_uiIsNodalMapValid[child[c]])
                            {
                                missed_child = true;
                                break;
                            }
                        }

                        if(missed_child)
                        {
                            x = pNodes[elem].minX()-((3*lsz)>>2u);  
                            for(unsigned int d2=0; d2 < N; d2+=1)
                            {
                                z= pNodes[elem].minZ() + d2*(lsz/m_uiElementOrder);
                                for(unsigned int d1=0; d1 < N; d1+=1)
                                {
                                    y= pNodes[elem].minY() + d1*(lsz/m_uiElementOrder);
                                    tmpSKey = ot::SearchKey(x , y , z , m_uiMaxDepth +1 ,m_uiDim, m_uiMaxDepth +1  );
                                    tmpSKey.addOwner(offset + ( ek*m_uiElementOrder + d2 +paddWidth )*ly*lx + (ej*m_uiElementOrder + paddWidth + d1) *lx + 0);
                                    //std::cout<<"ele: "<<pNodes[elem]<<" gen : "<<tmpSKey<<" with owner : "<<tmpSKey.getOwner()<<std::endl;
                                    m_uiUnzip_3pt.push_back(tmpSKey);
                                }
                            }
                        }
                        

                        
                    }
                    

                }

                if((pNodes[elem].minY()==blkNode.minY()))
                {
                    const unsigned int dir = OCT_DIR_DOWN;
                    assert(ej==eleIndexMin);
                    lookUp=m_uiE2EMapping[elem*m_uiNumDirections+dir];
                    if(lookUp!=LOOK_UP_TABLE_DEFAULT && pNodes[lookUp].getLevel() > pNodes[elem].getLevel())
                    { 
                        child[2]=lookUp;
                        child[3]=m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_RIGHT];
                        assert(child[3]!=LOOK_UP_TABLE_DEFAULT);
                        child[6]=m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[6]!=LOOK_UP_TABLE_DEFAULT);
                        child[7]=m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[7]!=LOOK_UP_TABLE_DEFAULT);

                        child[0]=m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_DOWN];
                        child[1]=m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_DOWN];
                        child[4]=m_uiE2EMapping[child[6]*m_uiNumDirections+OCT_DIR_DOWN];
                        child[5]=m_uiE2EMapping[child[7]*m_uiNumDirections+OCT_DIR_DOWN];

                        bool missed_child =false;
                        for(unsigned int c=0; c<NUM_CHILDREN;c++)
                        {
                            if(child[c] == LOOK_UP_TABLE_DEFAULT || pNodes[child[c]].getLevel()!=pNodes[lookUp].getLevel() ||  !m_uiIsNodalMapValid[child[c]])
                            {
                                missed_child = true;
                                break;
                            }
                        }

                        if(missed_child)
                        {
                            y = pNodes[elem].minY()-((3*lsz)>>2u);  
                            
                            for(unsigned int d2=0; d2 < N; d2+=1)
                            {
                                z= pNodes[elem].minZ() + d2*(lsz/m_uiElementOrder);
                                for(unsigned int d1=0; d1 < N; d1+=1)
                                {
                                    x= pNodes[elem].minX() + d1*(lsz/m_uiElementOrder);
                                    tmpSKey = ot::SearchKey(x , y , z , m_uiMaxDepth +1 ,m_uiDim, m_uiMaxDepth +1 );
                                    tmpSKey.addOwner(offset + (ek*m_uiElementOrder + paddWidth + d2)*ly*lx + 0*lx + (ei*m_uiElementOrder + paddWidth + d1));
                                    //std::cout<<"ele: "<<pNodes[elem]<<" gen : "<<tmpSKey<<" with owner : "<<tmpSKey.getOwner()<<std::endl;
                                    m_uiUnzip_3pt.push_back(tmpSKey);

                                }
                            }

                        }
                        
                        

                    }
                    

                }


                if((pNodes[elem].minZ()==blkNode.minZ()))
                {
                    const unsigned int dir = OCT_DIR_BACK;
                    assert(ek==eleIndexMin);
                    lookUp=m_uiE2EMapping[elem*m_uiNumDirections+dir];
                    if(lookUp!=LOOK_UP_TABLE_DEFAULT && pNodes[lookUp].getLevel() > pNodes[elem].getLevel())
                    { // this is the case the 3rd point might be missing on local proc. 


                        child[4]=lookUp;
                        child[5]=m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_RIGHT];
                        assert(child[5]!=LOOK_UP_TABLE_DEFAULT);
                        child[6]=m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_UP];
                        assert(child[6]!=LOOK_UP_TABLE_DEFAULT);
                        child[7]=m_uiE2EMapping[child[5]*m_uiNumDirections+OCT_DIR_UP];
                        assert(child[7]!=LOOK_UP_TABLE_DEFAULT);
            
                        child[0]=m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_BACK];
                        child[1]=m_uiE2EMapping[child[5]*m_uiNumDirections+OCT_DIR_BACK];
                        child[2]=m_uiE2EMapping[child[6]*m_uiNumDirections+OCT_DIR_BACK];
                        child[3]=m_uiE2EMapping[child[7]*m_uiNumDirections+OCT_DIR_BACK];

                        bool missed_child =false;
                        for(unsigned int c=0; c<NUM_CHILDREN;c++)
                        {
                            if(child[c] == LOOK_UP_TABLE_DEFAULT || pNodes[child[c]].getLevel()!=pNodes[lookUp].getLevel() ||  !m_uiIsNodalMapValid[child[c]])
                            {
                                missed_child = true;
                                break;
                            }
                        }

                        if(missed_child)
                        {
                            z = pNodes[elem].minZ()-((3*lsz)>>2u);  
                            for(unsigned int d2=0; d2 < N; d2+=1)
                            {
                                y= pNodes[elem].minY() + d2*(lsz/m_uiElementOrder);
                                for(unsigned int d1=0; d1 < N; d1+=1)
                                {
                                    x= pNodes[elem].minX() + d1*(lsz/m_uiElementOrder);
                                    tmpSKey = ot::SearchKey(x , y , z , m_uiMaxDepth +1 ,m_uiDim, m_uiMaxDepth +1  );
                                    tmpSKey.addOwner(offset + 0*ly*lx + (ej*m_uiElementOrder + paddWidth + d2)*lx + (ei*m_uiElementOrder + paddWidth + d1));
                                    m_uiUnzip_3pt.push_back(tmpSKey);

                                }
                            }
                        }


                    }
                    

                }


                if((pNodes[elem].maxX()==blkNode.maxX()))
                {
                    const unsigned int dir = OCT_DIR_RIGHT;
                    assert(ei==eleIndexMax);
                    lookUp=m_uiE2EMapping[elem*m_uiNumDirections+dir];
                    if(lookUp!=LOOK_UP_TABLE_DEFAULT && pNodes[lookUp].getLevel() > pNodes[elem].getLevel())
                    { // this is the case the 3rd point might be missing on local proc. 

                        child[0]=lookUp;
                        child[2]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_UP];
                        assert(child[2]!=LOOK_UP_TABLE_DEFAULT);
                        child[4]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[4]!=LOOK_UP_TABLE_DEFAULT);
                        child[6]=m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[6]!=LOOK_UP_TABLE_DEFAULT);

                        child[1]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_RIGHT];
                        child[3]=m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_RIGHT];
                        child[5]=m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_RIGHT];
                        child[7]=m_uiE2EMapping[child[6]*m_uiNumDirections+OCT_DIR_RIGHT];

                        bool missed_child =false;
                        for(unsigned int c=0; c<NUM_CHILDREN;c++)
                        {
                            if(child[c] == LOOK_UP_TABLE_DEFAULT || pNodes[child[c]].getLevel()!=pNodes[lookUp].getLevel() ||  !m_uiIsNodalMapValid[child[c]])
                            {
                                missed_child = true;
                                break;
                            }
                        }

                        if(missed_child)
                        {
                            x = pNodes[elem].maxX() + ((3*lsz)>>2u);  
                            for(unsigned int d2=0; d2 < N; d2+=1)
                            {
                                z= pNodes[elem].minZ() + d2*(lsz/m_uiElementOrder);
                                for(unsigned int d1=0; d1 < N; d1+=1)
                                {
                                    y= pNodes[elem].minY() + d1*(lsz/m_uiElementOrder);
                                    tmpSKey = ot::SearchKey(x , y , z , m_uiMaxDepth +1 ,m_uiDim, m_uiMaxDepth +1 );
                                    tmpSKey.addOwner(offset + (ek*m_uiElementOrder + paddWidth + d2)*ly*lx + (ej*m_uiElementOrder + paddWidth + d1)*lx + (uzip_1d-1));
                                    m_uiUnzip_3pt.push_back(tmpSKey);

                                }
                            }

                        }
                        
                        
                    }
                    

                }


                if((pNodes[elem].maxY()==blkNode.maxY()))
                {
                    const unsigned int dir = OCT_DIR_UP;
                    assert(ej==eleIndexMax);
                    lookUp=m_uiE2EMapping[elem*m_uiNumDirections+dir];
                    if(lookUp!=LOOK_UP_TABLE_DEFAULT && pNodes[lookUp].getLevel() > pNodes[elem].getLevel())
                    { // this is the case the 3rd point might be missing on local proc. 

                        child[0]=lookUp;
                        child[1]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_RIGHT];
                        assert(child[1]!=LOOK_UP_TABLE_DEFAULT);
                        child[4]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[4]!=LOOK_UP_TABLE_DEFAULT);
                        child[5]=m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_FRONT];
                        assert(child[5]!=LOOK_UP_TABLE_DEFAULT);

                        child[2]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_UP];
                        child[3]=m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_UP];
                        child[6]=m_uiE2EMapping[child[4]*m_uiNumDirections+OCT_DIR_UP];
                        child[7]=m_uiE2EMapping[child[5]*m_uiNumDirections+OCT_DIR_UP];

                        bool missed_child =false;
                        for(unsigned int c=0; c<NUM_CHILDREN;c++)
                        {
                            if(child[c] == LOOK_UP_TABLE_DEFAULT || pNodes[child[c]].getLevel()!=pNodes[lookUp].getLevel() ||  !m_uiIsNodalMapValid[child[c]])
                            {
                                missed_child = true;
                                break;
                            }
                        }

                        if(missed_child)
                        {
                            y = pNodes[elem].maxY() + ((3*lsz)>>2u);  
                            for(unsigned int d2=0; d2 < N; d2+=1)
                            {
                                z = pNodes[elem].minZ() + d2*(lsz/m_uiElementOrder);
                                for(unsigned int d1=0; d1 < N; d1+=1)
                                {
                                    x = pNodes[elem].minX() + d1*(lsz/m_uiElementOrder);
                                    tmpSKey = ot::SearchKey(x , y , z , m_uiMaxDepth +1 ,m_uiDim, m_uiMaxDepth +1 );
                                    tmpSKey.addOwner(offset + (ek*m_uiElementOrder + paddWidth + d2)*ly*lx + (uzip_1d-1)*lx + (ei*m_uiElementOrder + paddWidth + d1));
                                    m_uiUnzip_3pt.push_back(tmpSKey);

                                }
                            }

                        }

                        
    
                    }
                    

                }


                if((pNodes[elem].maxZ()==blkNode.maxZ()))
                {
                    const unsigned int dir = OCT_DIR_FRONT;
                    assert(ek==eleIndexMax);
                    lookUp=m_uiE2EMapping[elem*m_uiNumDirections+dir];
                    if(lookUp!=LOOK_UP_TABLE_DEFAULT && pNodes[lookUp].getLevel() > pNodes[elem].getLevel())
                    { // this is the case the 3rd point might be missing on local proc. 

                        child[0]=lookUp;
                        child[1]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_RIGHT];
                        assert(child[1]!=LOOK_UP_TABLE_DEFAULT);
                        child[2]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_UP];
                        assert(child[2]!=LOOK_UP_TABLE_DEFAULT);
                        child[3]=m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_UP];
                        assert(child[3]!=LOOK_UP_TABLE_DEFAULT);

                        child[4]=m_uiE2EMapping[child[0]*m_uiNumDirections+OCT_DIR_FRONT];
                        child[5]=m_uiE2EMapping[child[1]*m_uiNumDirections+OCT_DIR_FRONT];
                        child[6]=m_uiE2EMapping[child[2]*m_uiNumDirections+OCT_DIR_FRONT];
                        child[7]=m_uiE2EMapping[child[3]*m_uiNumDirections+OCT_DIR_FRONT];

                        bool missed_child =false;
                        for(unsigned int c=0; c<NUM_CHILDREN;c++)
                        {
                            if(child[c] == LOOK_UP_TABLE_DEFAULT || pNodes[child[c]].getLevel()!=pNodes[lookUp].getLevel() ||  !m_uiIsNodalMapValid[child[c]])
                            {
                                missed_child = true;
                                break;
                            }
                        }

                        if(missed_child)
                        {
                            z = pNodes[elem].maxZ() + ((3*lsz)>>2u);  
                            for(unsigned int d2=0; d2 < N; d2+=1)
                            {
                                y = pNodes[elem].minY() + d2*(lsz/m_uiElementOrder);
                                for(unsigned int d1=0; d1 < N; d1+=1)
                                {
                                    x = pNodes[elem].minX() + d1*(lsz/m_uiElementOrder);
                                    tmpSKey = ot::SearchKey(x , y , z , m_uiMaxDepth +1 ,m_uiDim, m_uiMaxDepth +1 );
                                    tmpSKey.addOwner(offset + (uzip_1d-1)*ly*lx + (ej*m_uiElementOrder + paddWidth + d2 )*lx + (ei*m_uiElementOrder + paddWidth +d1));
                                    m_uiUnzip_3pt.push_back(tmpSKey);

                                }
                            }

                        }
                        

                        
    
                    }
                }




            }
         }

         //std::cout<<" rank: "<<m_uiActiveRank<<"missing pts dup: "<<m_uiUnzip_3pt.size()<<std::endl;
         m_uiMaxDepth++;
         mergeKeys(m_uiUnzip_3pt,m_uiUnzip_3pt_keys);
         assert(seq::test::isUniqueAndSorted(m_uiUnzip_3pt_keys));
         m_uiMaxDepth--;

         std::vector<ot::Key> dboundary_keys;
         for(unsigned int i=0; i < m_uiUnzip_3pt_keys.size(); i++ )
         {
            unsigned int x =  m_uiUnzip_3pt_keys[i].minX();
            unsigned int y =  m_uiUnzip_3pt_keys[i].minY();
            unsigned int z =  m_uiUnzip_3pt_keys[i].minZ();

            if(x == (1u<<m_uiMaxDepth))
                x=x-1;
            
            if(y == (1u<<m_uiMaxDepth))
                y=y-1;
            
            if(z == (1u<<m_uiMaxDepth))
                z=z-1;

            dboundary_keys.push_back(ot::Key(x,y,z,m_uiMaxDepth,m_uiDim, m_uiMaxDepth));
            dboundary_keys.back().addOwner(i);
         }


         for(unsigned int p = 0; p < m_uiActiveNpes ;p++)
         {
           dboundary_keys.push_back(ot::Key(m_uiLocalSplitterElements[2*p]));
         }
         
         std::vector<ot::Key> sEleKeys;
         sEleKeys.resize(m_uiActiveNpes);
         for(unsigned int p = 0; p < m_uiActiveNpes ;p++)
            sEleKeys[p]=Key(m_uiLocalSplitterElements[2*p]);
            
         SFC::seqSearch::SFC_treeSearch(&(*(sEleKeys.begin())),&(*(dboundary_keys.begin())),0,sEleKeys.size(),0,dboundary_keys.size(),m_uiMaxDepth, m_uiMaxDepth,ROOT_ROTATION);
         
         // compute the owner rank(process) of the missing points. 
         std::vector<unsigned int> ownerrank;
         ownerrank.resize(m_uiUnzip_3pt_keys.size(),LOOK_UP_TABLE_DEFAULT);
         
         unsigned int sBegin=0;
         unsigned int sEnd;
         for(unsigned int p=0;p<m_uiActiveNpes;p++)
         {
            assert((sEleKeys[p].getFlag() & OCT_FOUND));
            assert(dboundary_keys[sEleKeys[p].getSearchResult()]==sEleKeys[p]);
            sBegin=sEleKeys[p].getSearchResult();
            (p<(m_uiActiveNpes-1))? sEnd=sEleKeys[p+1].getSearchResult()+1: sEnd=dboundary_keys.size();
            
            for(unsigned int k=sBegin;k<sEnd;k++)
            {
                // if true it implies, that this key is a splitter element. 
                if(dboundary_keys[k].getOwnerListSize()<1)
                {
                   assert(sEleKeys[p] ==  dboundary_keys[k] || (p<(m_uiActiveNpes-1) && sEleKeys[p+1] ==  dboundary_keys[k] ));
                   continue; 
                } 
                
                ownerrank[dboundary_keys[k].getOwnerList()->front()] = p;
            }
         }

         dboundary_keys.clear();

        
 
         // 2. Communicate the keys based on the computed owner ranks. 
         
         m_uiSendCountRePt.resize(m_uiActiveNpes);
         m_uiSendOffsetRePt.resize(m_uiActiveNpes);
         m_uiRecvCountRePt.resize(m_uiActiveNpes);
         m_uiRecvOffsetRePt.resize(m_uiActiveNpes);
         
         for(unsigned int i=0; i< m_uiActiveNpes; i++)
            m_uiSendCountRePt[i] = 0;    

         for(unsigned int i=0;i<ownerrank.size();i++)
         {
            if( (ownerrank[i] == LOOK_UP_TABLE_DEFAULT ))
            {
                std::cout<<"error: "<<__func__<<" sending key : "<<m_uiUnzip_3pt_keys[i]<<" to proc: "<<ownerrank[i]<<std::endl;
                MPI_Abort(m_uiCommActive,0);
            }
            
            m_uiSendCountRePt[ownerrank[i]]++;
             
                
         }            
 
         par::Mpi_Alltoall(&(*(m_uiSendCountRePt.begin())), &(*(m_uiRecvCountRePt.begin())),1,m_uiCommActive);
 
         m_uiSendOffsetRePt[0] = 0;
         m_uiRecvOffsetRePt[0] = 0;
 
         omp_par::scan(&(*(m_uiSendCountRePt.begin())),&(*(m_uiSendOffsetRePt.begin())),m_uiActiveNpes);
         omp_par::scan(&(*(m_uiRecvCountRePt.begin())),&(*(m_uiRecvOffsetRePt.begin())),m_uiActiveNpes);
 
 
         std::vector<ot::TreeNode> sBuf;
         std::vector<ot::TreeNode> rBuf;
         
         sBuf.resize((m_uiSendOffsetRePt[m_uiActiveNpes-1] + m_uiSendCountRePt[m_uiActiveNpes-1] ));
         rBuf.resize((m_uiRecvOffsetRePt[m_uiActiveNpes-1] + m_uiRecvCountRePt[m_uiActiveNpes-1] ));
 
         /*if(m_uiActiveRank==0)
         {
            for(unsigned int p=0;p<m_uiActiveNpes;p++)
                std::cout<<"rank:"<<m_uiActiveRank<<" send to "<<p<<" count : "<<m_uiSendCountRePt[p]<<std::endl;

            for(unsigned int p=0;p<m_uiActiveNpes;p++)
                std::cout<<"rank:"<<m_uiActiveRank<<" recv from "<<p<<" count : "<<m_uiRecvCountRePt[p]<<std::endl;
         }*/
 
         for(unsigned int i=0; i< m_uiActiveNpes; i++)
            m_uiSendCountRePt[i] = 0;

         // Note: this is important to do so to match the send node order with recv order. 
         std::vector<ot::Key> tmpSendKey;
         tmpSendKey.resize(sBuf.size());
         
         for(unsigned int i=0;i<ownerrank.size();i++)
         {
            sBuf[m_uiSendOffsetRePt[ownerrank[i]] + m_uiSendCountRePt[ownerrank[i]]] = ot::TreeNode(m_uiUnzip_3pt_keys[i].minX(),m_uiUnzip_3pt_keys[i].minY(),m_uiUnzip_3pt_keys[i].minZ(),m_uiMaxDepth+1,m_uiDim,m_uiMaxDepth+1);
            tmpSendKey[m_uiSendOffsetRePt[ownerrank[i]] + m_uiSendCountRePt[ownerrank[i]]]=(m_uiUnzip_3pt_keys[i]);
            m_uiSendCountRePt[ownerrank[i]]++;
         }

         std::swap(tmpSendKey,m_uiUnzip_3pt_keys);
         tmpSendKey.clear();

            //  if(m_uiActiveRank==0)
            //  {
            //      for(unsigned  int p=1; p<m_uiActiveNpes;p++ )
            //      {
            //          for(unsigned int i=m_uiSendOffsetRePt[p]; i< m_uiSendOffsetRePt[p] + m_uiSendCountRePt[p];i++)
            //          {
            //              std::cout<<"rnk: "<<m_uiActiveRank<<" sBuf key["<<i<<"] : "<<sBuf[i]<<" key:  "<<m_uiUnzip_3pt_keys[i]<<"to proc: "<<p<<" dist from offset : "<<(i-m_uiSendOffsetRePt[p])<<std::endl;
            //          }

            //      }
                
            //  }


         par::Mpi_Alltoallv(&(*(sBuf.begin())), (int *)(&(*(m_uiSendCountRePt.begin()))), (int *) (&(*(m_uiSendOffsetRePt.begin()))), &(*(rBuf.begin())), (int  *) (&(*(m_uiRecvCountRePt.begin()))), (int *) (&(*(m_uiRecvOffsetRePt.begin()))), m_uiCommActive);
        
         std::vector<ot::SearchKey> keys;
         keys.reserve(rBuf.size());
 
         for(unsigned int p=0; p< m_uiActiveNpes; p++)
         {
            for(unsigned int i=m_uiRecvOffsetRePt[p]; i< (m_uiRecvOffsetRePt[p] + m_uiRecvCountRePt[p]); i++)
            {   
                keys.push_back(ot::SearchKey(rBuf[i].minX(),rBuf[i].minY(),rBuf[i].minZ(), m_uiMaxDepth+1, m_uiDim, m_uiMaxDepth+1));
                keys.back().addOwner(i);
            }
         }
         
 
         m_uiMaxDepth++;
         mergeKeys(keys,m_uiUnzip_3pt_recv_keys);
         m_uiMaxDepth--;
         keys.clear();

         //std::cout<<"rank : "<<m_uiActiveRank<<" recv keys : "<<m_uiUnzip_3pt_recv_keys.size()<<std::endl;

         std::vector<ot::Key> rkey_merged;
         for(unsigned int i=0; i < m_uiUnzip_3pt_recv_keys.size(); i++ )
         {
            unsigned int x =  m_uiUnzip_3pt_recv_keys[i].minX();
            unsigned int y =  m_uiUnzip_3pt_recv_keys[i].minY();
            unsigned int z =  m_uiUnzip_3pt_recv_keys[i].minZ();

            if(x == (1u<<m_uiMaxDepth))
                x=x-1;
            
            if(y == (1u<<m_uiMaxDepth))
                y=y-1;
            
            if(z == (1u<<m_uiMaxDepth))
                z=z-1;
            
            rkey_merged.push_back(ot::Key(x,y,z,m_uiMaxDepth,m_uiDim, m_uiMaxDepth));
            rkey_merged.back().addOwner(i);
         }

         
         SFC::seqSearch::SFC_treeSearch(&(*(rkey_merged.begin())),&(*(m_uiAllElements.begin())),0,rkey_merged.size(),m_uiElementLocalBegin,m_uiElementLocalEnd, m_uiMaxDepth,m_uiMaxDepth,ROOT_ROTATION);
         //MPI_Barrier(m_uiCommActive); std::cout<<"search 2 pass "<<std::endl;
         
         std::vector<SearchKey> eKeys; 
         for(unsigned int i=0;i<rkey_merged.size();i++)
         {
            if( !(rkey_merged[i].getFlag() & OCT_FOUND) )
            {
                std::cout<<"Error["<<m_uiActiveRank<<"] : "<<__func__<<" requested key : "<<rkey_merged[i]<<" node is not found at any local partition "<<std::endl;
                MPI_Abort(m_uiCommActive,0);
            }

            const unsigned eleID = rkey_merged[i].getSearchResult();
            eKeys.push_back(ot::SearchKey(m_uiAllElements[eleID]));
            eKeys.back().addOwner(rkey_merged[i].getOwnerList()->front());
            
         }
 
         rkey_merged.clear();
         mergeKeys(eKeys,m_uiUnzip_3pt_ele);

         //std::cout<<"rank: "<<m_uiActiveRank<<" ele: "<<m_uiUnzip_3pt_ele.size()<<std::endl;
         SFC::seqSearch::SFC_treeSearch(&(*(m_uiUnzip_3pt_ele.begin())),&(*(m_uiAllElements.begin())),0,m_uiUnzip_3pt_ele.size(),m_uiElementLocalBegin, m_uiElementLocalEnd,m_uiMaxDepth,m_uiMaxDepth,ROOT_ROTATION);
 
         // swap the send nodes counts with recv node counts since the communication needs to be done in the other direction.
         std::swap(m_uiSendCountRePt,m_uiRecvCountRePt);
         std::swap(m_uiSendOffsetRePt,m_uiRecvOffsetRePt);
 
         for(unsigned int i=0;i<m_uiActiveNpes;i++)
         {
            if(m_uiSendCountRePt[i]>0)
                m_uiReqSendProcList.push_back(i);

            if(m_uiRecvCountRePt[i]>0)
                m_uiReqRecvProcList.push_back(i);
         }

         //MPI_Barrier(m_uiCommActive); if(!m_uiActiveRank) std::cout<<"3rd pt sm build"<<std::endl;
 
 
                  
 
                     
                     
    }
#endif

int Mesh::getBlkBdyParentCNums(unsigned int blkId, unsigned int eleId,
                               unsigned int dir, unsigned int *child,
                               unsigned int *fid, unsigned int *cid) {
    // return -1 if the invalid call for the function.
    if ((!m_uiIsBlockSetup) || (!m_uiIsActive)) return -1;

    for (unsigned int i = 0; i < NUM_CHILDREN; i++)
        child[i] = LOOK_UP_TABLE_DEFAULT;

    const unsigned int lookup = m_uiE2EMapping[eleId * m_uiNumDirections + dir];

    if (lookup == LOOK_UP_TABLE_DEFAULT) return -1;

    unsigned int cnum;
    const bool isHanging = this->isFaceHanging(eleId, dir, cnum);

    if ((!isHanging)) return -1;

    unsigned char bit[3];
    const unsigned int eorder_by2 = (m_uiElementOrder >> 1u);

    if (dir == OCT_DIR_LEFT) {
        fid[0] = 1;
        fid[1] = 3;
        fid[2] = 5;
        fid[3] = 7;
        cid[0] = 0;
        cid[1] = 2;
        cid[2] = 4;
        cid[3] = 6;

        child[fid[0]] =
            m_uiE2EMapping[lookup * m_uiNumDirections + OCT_DIR_RIGHT];
        child[fid[1]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_UP];

        child[fid[2]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_FRONT];
        child[fid[3]] =
            m_uiE2EMapping[child[fid[1]] * m_uiNumDirections + OCT_DIR_FRONT];

        assert((child[fid[0]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[1]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[2]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[3]] != LOOK_UP_TABLE_DEFAULT));

    } else if (dir == OCT_DIR_RIGHT) {
        fid[0] = 0;
        fid[1] = 2;
        fid[2] = 4;
        fid[3] = 6;
        cid[0] = 1;
        cid[1] = 3;
        cid[2] = 5;
        cid[3] = 7;

        child[fid[0]] =
            m_uiE2EMapping[lookup * m_uiNumDirections + OCT_DIR_LEFT];
        child[fid[1]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_UP];

        child[fid[2]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_FRONT];
        child[fid[3]] =
            m_uiE2EMapping[child[fid[1]] * m_uiNumDirections + OCT_DIR_FRONT];

        assert((child[fid[0]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[1]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[2]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[3]] != LOOK_UP_TABLE_DEFAULT));

    } else if (dir == OCT_DIR_DOWN) {
        fid[0]        = 2;
        fid[1]        = 3;
        fid[2]        = 6;
        fid[3]        = 7;
        cid[0]        = 0;
        cid[1]        = 1;
        cid[2]        = 4;
        cid[3]        = 5;

        child[fid[0]] = m_uiE2EMapping[lookup * m_uiNumDirections + OCT_DIR_UP];
        child[fid[1]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_RIGHT];

        child[fid[2]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_FRONT];
        child[fid[3]] =
            m_uiE2EMapping[child[fid[1]] * m_uiNumDirections + OCT_DIR_FRONT];

        assert((child[fid[0]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[1]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[2]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[3]] != LOOK_UP_TABLE_DEFAULT));

    } else if (dir == OCT_DIR_UP) {
        fid[0] = 0;
        fid[1] = 1;
        fid[2] = 4;
        fid[3] = 5;
        cid[0] = 2;
        cid[1] = 3;
        cid[2] = 6;
        cid[3] = 7;

        child[fid[0]] =
            m_uiE2EMapping[lookup * m_uiNumDirections + OCT_DIR_DOWN];
        child[fid[1]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_RIGHT];

        child[fid[2]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_FRONT];
        child[fid[3]] =
            m_uiE2EMapping[child[fid[1]] * m_uiNumDirections + OCT_DIR_FRONT];

        assert((child[fid[0]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[1]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[2]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[3]] != LOOK_UP_TABLE_DEFAULT));

    } else if (dir == OCT_DIR_BACK) {
        fid[0] = 4;
        fid[1] = 5;
        fid[2] = 6;
        fid[3] = 7;
        cid[0] = 0;
        cid[1] = 1;
        cid[2] = 2;
        cid[3] = 3;

        child[fid[0]] =
            m_uiE2EMapping[lookup * m_uiNumDirections + OCT_DIR_FRONT];
        child[fid[1]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_RIGHT];

        child[fid[2]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_UP];
        child[fid[3]] =
            m_uiE2EMapping[child[fid[1]] * m_uiNumDirections + OCT_DIR_UP];

        assert((child[fid[0]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[1]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[2]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[3]] != LOOK_UP_TABLE_DEFAULT));

    } else if (dir == OCT_DIR_FRONT) {
        fid[0] = 0;
        fid[1] = 1;
        fid[2] = 2;
        fid[3] = 3;
        cid[0] = 4;
        cid[1] = 5;
        cid[2] = 6;
        cid[3] = 7;

        child[fid[0]] =
            m_uiE2EMapping[lookup * m_uiNumDirections + OCT_DIR_BACK];
        child[fid[1]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_RIGHT];

        child[fid[2]] =
            m_uiE2EMapping[child[fid[0]] * m_uiNumDirections + OCT_DIR_UP];
        child[fid[3]] =
            m_uiE2EMapping[child[fid[1]] * m_uiNumDirections + OCT_DIR_UP];

        assert((child[fid[0]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[1]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[2]] != LOOK_UP_TABLE_DEFAULT) &&
               (child[fid[3]] != LOOK_UP_TABLE_DEFAULT));

    } else {
        return -1;
    }

    if (eleId == child[fid[0]])
        return 1;
    else if (child[fid[0]] < m_uiElementLocalBegin ||
             child[fid[0]] >= m_uiElementLocalEnd)
        return 1;
    else
        return -1;  // child[fid[0]] is not eleID and it is local, hence we
                    // don't need to return 1 to overwirte the same data.
}

void Mesh::computeMinMaxLevel(unsigned int &lmin, unsigned int &lmax) const {
    if (m_uiIsActive) {
        unsigned int lmin_l = m_uiAllElements[m_uiElementLocalBegin].getLevel();
        unsigned int lmax_l = m_uiAllElements[m_uiElementLocalBegin].getLevel();
        for (unsigned int e = m_uiElementLocalBegin + 1;
             e < m_uiElementLocalEnd; e++) {
            if (m_uiAllElements[e].getLevel() < lmin_l)
                lmin_l = m_uiAllElements[e].getLevel();

            if (m_uiAllElements[e].getLevel() > lmax_l)
                lmax_l = m_uiAllElements[e].getLevel();
        }

        par::Mpi_Reduce(&lmin_l, &lmin, 1, MPI_MIN, 0, m_uiCommActive);
        par::Mpi_Reduce(&lmax_l, &lmax, 1, MPI_MAX, 0, m_uiCommActive);
    }

    par::Mpi_Bcast(&lmin, 1, 0, m_uiCommGlobal);
    par::Mpi_Bcast(&lmax, 1, 0, m_uiCommGlobal);

    return;
}

void Mesh::getFinerFaceNeighbors(unsigned int ele, unsigned int dir,
                                 unsigned int *child) const {
    const unsigned int lookup = m_uiE2EMapping[ele * NUM_FACES + dir];

    if (lookup == LOOK_UP_TABLE_DEFAULT) {
        child[0] = LOOK_UP_TABLE_DEFAULT;
        child[1] = LOOK_UP_TABLE_DEFAULT;
        child[2] = LOOK_UP_TABLE_DEFAULT;
        child[3] = LOOK_UP_TABLE_DEFAULT;

        return;
    }

    if (m_uiAllElements[lookup].getLevel() <= m_uiAllElements[ele].getLevel()) {
        child[0] = lookup;
        child[1] = lookup;
        child[2] = lookup;
        child[3] = lookup;

        return;
    }

    const unsigned int *e2e = m_uiE2EMapping.data();

    switch (dir) {
        case OCT_DIR_LEFT:
            child[0] = lookup;
            child[1] = e2e[child[0] * NUM_FACES + OCT_DIR_UP];
            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
            child[2] = e2e[child[0] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
            child[3] = e2e[child[1] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[3] != LOOK_UP_TABLE_DEFAULT);

            assert(ele == e2e[child[0] * NUM_FACES + OCT_DIR_RIGHT]);
            assert(ele == e2e[child[1] * NUM_FACES + OCT_DIR_RIGHT]);
            assert(ele == e2e[child[2] * NUM_FACES + OCT_DIR_RIGHT]);
            assert(ele == e2e[child[3] * NUM_FACES + OCT_DIR_RIGHT]);

            break;

        case OCT_DIR_RIGHT:
            child[0] = lookup;
            child[1] = e2e[child[0] * NUM_FACES + OCT_DIR_UP];
            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
            child[2] = e2e[child[0] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
            child[3] = e2e[child[1] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[3] != LOOK_UP_TABLE_DEFAULT);

            assert(ele == e2e[child[0] * NUM_FACES + OCT_DIR_LEFT]);
            assert(ele == e2e[child[1] * NUM_FACES + OCT_DIR_LEFT]);
            assert(ele == e2e[child[2] * NUM_FACES + OCT_DIR_LEFT]);
            assert(ele == e2e[child[3] * NUM_FACES + OCT_DIR_LEFT]);

            break;

        case OCT_DIR_DOWN:

            child[0] = lookup;
            child[1] = e2e[child[0] * NUM_FACES + OCT_DIR_RIGHT];
            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
            child[2] = e2e[child[0] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
            child[3] = e2e[child[1] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[3] != LOOK_UP_TABLE_DEFAULT);

            assert(ele == e2e[child[0] * NUM_FACES + OCT_DIR_UP]);
            assert(ele == e2e[child[1] * NUM_FACES + OCT_DIR_UP]);
            assert(ele == e2e[child[2] * NUM_FACES + OCT_DIR_UP]);
            assert(ele == e2e[child[3] * NUM_FACES + OCT_DIR_UP]);

            break;

        case OCT_DIR_UP:
            child[0] = lookup;
            child[1] = e2e[child[0] * NUM_FACES + OCT_DIR_RIGHT];
            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
            child[2] = e2e[child[0] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
            child[3] = e2e[child[1] * NUM_FACES + OCT_DIR_FRONT];
            assert(child[3] != LOOK_UP_TABLE_DEFAULT);

            assert(ele == e2e[child[0] * NUM_FACES + OCT_DIR_DOWN]);
            assert(ele == e2e[child[1] * NUM_FACES + OCT_DIR_DOWN]);
            assert(ele == e2e[child[2] * NUM_FACES + OCT_DIR_DOWN]);
            assert(ele == e2e[child[3] * NUM_FACES + OCT_DIR_DOWN]);

            break;

        case OCT_DIR_BACK:
            child[0] = lookup;
            child[1] = e2e[child[0] * NUM_FACES + OCT_DIR_RIGHT];
            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
            child[2] = e2e[child[0] * NUM_FACES + OCT_DIR_UP];
            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
            child[3] = e2e[child[1] * NUM_FACES + OCT_DIR_UP];
            assert(child[3] != LOOK_UP_TABLE_DEFAULT);

            assert(ele == e2e[child[0] * NUM_FACES + OCT_DIR_FRONT]);
            assert(ele == e2e[child[1] * NUM_FACES + OCT_DIR_FRONT]);
            assert(ele == e2e[child[2] * NUM_FACES + OCT_DIR_FRONT]);
            assert(ele == e2e[child[3] * NUM_FACES + OCT_DIR_FRONT]);

            break;

        case OCT_DIR_FRONT:
            child[0] = lookup;
            child[1] = e2e[child[0] * NUM_FACES + OCT_DIR_RIGHT];
            assert(child[1] != LOOK_UP_TABLE_DEFAULT);
            child[2] = e2e[child[0] * NUM_FACES + OCT_DIR_UP];
            assert(child[2] != LOOK_UP_TABLE_DEFAULT);
            child[3] = e2e[child[1] * NUM_FACES + OCT_DIR_UP];
            assert(child[3] != LOOK_UP_TABLE_DEFAULT);

            assert(ele == e2e[child[0] * NUM_FACES + OCT_DIR_BACK]);
            assert(ele == e2e[child[1] * NUM_FACES + OCT_DIR_BACK]);
            assert(ele == e2e[child[2] * NUM_FACES + OCT_DIR_BACK]);
            assert(ele == e2e[child[3] * NUM_FACES + OCT_DIR_BACK]);

            break;

        default:
            break;
    }
}

void Mesh::interGridTransferSendRecvCompute(const ot::Mesh *pMesh) {
    if (m_uiIsIGTSetup) return;

    MPI_Comm comm = m_uiCommGlobal;
    int rank, npes;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    m_uiIGTSendC.clear();
    m_uiIGTRecvC.clear();
    m_uiIGTSendOfst.clear();
    m_uiIGTRecvOfst.clear();
    m_uiM2Prime.clear();

    m_uiIGTSendC.resize(npes, 0);
    m_uiIGTRecvC.resize(npes, 0);
    m_uiIGTSendOfst.resize(npes, 0);
    m_uiIGTRecvOfst.resize(npes, 0);

    if (m_uiIsActive) {
        MPI_Comm comm1            = m_uiCommActive;
        const int rank1           = m_uiActiveRank;
        const int npes1           = m_uiActiveNpes;

        // 1. compute the number of m2 octants (based of m1 splitters)
        unsigned int m2primeCount = 0;
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ele++) {
            if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) == OCT_SPLIT)
                m2primeCount += NUM_CHILDREN;
            else if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                     OCT_COARSE) {
                assert(m_uiAllElements[ele].getParent() ==
                       m_uiAllElements[ele + NUM_CHILDREN - 1].getParent());
                m2primeCount += 1;
                ele += (NUM_CHILDREN - 1);
            } else {
                assert((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                       OCT_NO_CHANGE);
                m2primeCount += 1;
            }
        }

        const unsigned int numM2PrimeElems = m2primeCount;
        m_uiM2Prime.clear();
        m_uiM2Prime.reserve(numM2PrimeElems);

        m2primeCount = 0;
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ele++) {
            if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                OCT_SPLIT) {
                m_uiAllElements[ele].addChildren(m_uiM2Prime);
                m2primeCount += NUM_CHILDREN;

            } else if ((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                       OCT_COARSE) {
                assert(m_uiAllElements[ele].getParent() ==
                       m_uiAllElements[ele + NUM_CHILDREN - 1].getParent());
                m_uiM2Prime.push_back(m_uiAllElements[ele].getParent());

                ele += (NUM_CHILDREN - 1);
                m2primeCount += 1;
            } else {
                assert((m_uiAllElements[ele].getFlag() >> NUM_LEVEL_BITS) ==
                       OCT_NO_CHANGE);
                m_uiM2Prime.push_back(m_uiAllElements[ele]);
                m2primeCount += 1;
            }
        }

        assert(seq::test::isUniqueAndSorted(m_uiM2Prime));

        if (npes == 1)  // m2_prime is equivalent to m2, hence no need to
                        // compute the send/ recv counts.
        {
            for (unsigned int p = 0; p < npes; p++) {
                m_uiIGTSendC[p]    = 0;
                m_uiIGTRecvC[p]    = 0;
                m_uiIGTSendOfst[p] = 0;
                m_uiIGTRecvOfst[p] = 0;
            }
            m_uiIsIGTSetup = true;
            return;
        }

        int npes2 = 0;
        int rank2 = 0;
        std::vector<ot::TreeNode> m2_splitters;
        // note : assumes that global rank 0 is going to be active always.
        if (pMesh->isActive()) {
            npes2 = pMesh->getMPICommSize();
            rank2 = pMesh->getMPIRank();
            const std::vector<ot::TreeNode> &m2_splitters_root =
                pMesh->getSplitterElements();
            m2_splitters.resize(2 * npes2);
            for (unsigned int w = 0; w < m2_splitters_root.size(); w++)
                m2_splitters[w] = m2_splitters_root[w];
        }

        par::Mpi_Bcast(&npes2, 1, 0, comm1);
        par::Mpi_Bcast(&rank2, 1, 0, comm1);
        m2_splitters.resize(2 * npes2);
        par::Mpi_Bcast(&(*(m2_splitters.begin())), 2 * npes2, 0, comm1);
        assert(seq::test::isUniqueAndSorted(m2_splitters));

        std::vector<ot::SearchKey> m2primeSK;
        m2primeSK.resize(m_uiM2Prime.size());

        for (unsigned int e = 0; e < m_uiM2Prime.size(); e++) {
            m2primeSK[e] = ot::SearchKey(m_uiM2Prime[e]);
            m2primeSK[e].addOwner(
                rank1);  // note that this is the rank in comm1.
        }

        std::vector<ot::Key> m2_splitterKeys;
        m2_splitterKeys.resize(2 * npes2);

        for (unsigned int p = 0; p < npes2; p++) {
            m2_splitterKeys[2 * p] = ot::Key(m2_splitters[2 * p]);
            m2_splitterKeys[2 * p].addOwner(p);

            m2_splitterKeys[2 * p + 1] = ot::Key(m2_splitters[2 * p + 1]);
            m2_splitterKeys[2 * p + 1].addOwner(p);

            m2primeSK.push_back(ot::SearchKey(m2_splitters[2 * p]));
            m2primeSK.push_back(ot::SearchKey(m2_splitters[2 * p + 1]));
        }

        ot::SearchKey rootSK(m_uiDim, m_uiMaxDepth);
        std::vector<ot::SearchKey> tmpNodes;

        SFC::seqSort::SFC_treeSort(&(*(m2primeSK.begin())), m2primeSK.size(),
                                   tmpNodes, tmpNodes, tmpNodes, m_uiMaxDepth,
                                   m_uiMaxDepth, rootSK, ROOT_ROTATION, 1,
                                   TS_SORT_ONLY);

        unsigned int skip = 0;
        ot::SearchKey tmpSK;
        std::vector<ot::SearchKey> tmpSKVec;

        for (unsigned int e = 0; e < (m2primeSK.size()); e++) {
            tmpSK = m2primeSK[e];
            skip  = 1;
            while (((e + skip) < m2primeSK.size()) &&
                   (m2primeSK[e] == m2primeSK[e + skip])) {
                if (m2primeSK[e + skip].getOwner() >= 0) {
                    tmpSK.addOwner(m2primeSK[e + skip].getOwner());
                }
                skip++;
            }

            tmpSKVec.push_back(tmpSK);
            e += (skip - 1);
        }

        std::swap(m2primeSK, tmpSKVec);
        tmpSKVec.clear();

        assert(seq::test::isUniqueAndSorted(m2primeSK));
        assert(seq::test::isUniqueAndSorted(m2_splitterKeys));

        ot::Key rootKey(0, 0, 0, 0, m_uiDim, m_uiMaxDepth);
        SFC::seqSearch::SFC_treeSearch(
            &(*(m2_splitterKeys.begin())), &(*(m2primeSK.begin())), 0,
            m2_splitterKeys.size(), 0, m2primeSK.size(), m_uiMaxDepth,
            m_uiMaxDepth, ROOT_ROTATION);

        unsigned int sBegin, sEnd, selectedRank;
        for (unsigned int p = 0; p < npes2; p++) {
            assert(m2_splitterKeys[2 * p].getFlag() & OCT_FOUND);
            assert(m2_splitterKeys[2 * p + 1].getFlag() & OCT_FOUND);

            sBegin = m2_splitterKeys[2 * p].getSearchResult();
            sEnd   = m2_splitterKeys[2 * p + 1].getSearchResult();
            assert(sBegin < sEnd);
            selectedRank =
                rankSelectRule(m_uiGlobalNpes, m_uiGlobalRank, npes2, p);
            m_uiIGTSendC[selectedRank] = sEnd - sBegin - 1;

            if (m2primeSK[sBegin].getOwner() >= 0) m_uiIGTSendC[selectedRank]++;
            if (m2primeSK[sEnd].getOwner() >= 0) m_uiIGTSendC[selectedRank]++;
        }

        // we don't need below for intergrid transfer, but these can be help
        // full for debugging.
        m2primeSK.clear();
    }

    par::Mpi_Alltoall(m_uiIGTSendC.data(), m_uiIGTRecvC.data(), 1, comm);

    m_uiIGTSendOfst[0] = 0;
    m_uiIGTRecvOfst[0] = 0;

    omp_par::scan(m_uiIGTSendC.data(), m_uiIGTSendOfst.data(), npes);
    omp_par::scan(m_uiIGTRecvC.data(), m_uiIGTRecvOfst.data(), npes);

    const unsigned int total_recv_elements =
        m_uiIGTRecvOfst[npes - 1] + m_uiIGTRecvC[npes - 1];

    if (total_recv_elements != pMesh->getNumLocalMeshElements()) {
        std::cout << "rank: " << rank
                  << " [Inter-grid Transfer error ]: Recvn M2' elements: "
                  << total_recv_elements << " m2 num local elements "
                  << pMesh->getNumLocalMeshElements() << std::endl;
        MPI_Abort(comm, 0);
    }

    m_uiIsIGTSetup = true;
    return;
}

std::vector<unsigned int> Mesh::getAllRefinementFlags() {
    // create the starting vector with oct no change based on the element IDs
    std::vector<unsigned int> refine_flags(m_uiNumLocalElements, OCT_NO_CHANGE);

    for (unsigned int ele = m_uiElementLocalBegin; ele < m_uiElementLocalEnd;
         ele++) {
        // the relative ID is based on the element we traverse over
        const unsigned int rel_id = ele - m_uiElementLocalBegin;

        // then get the flag
        const unsigned int flag   = m_uiAllElements[ele].getFlag();

        // don't forget to strip it from it's level bits
        refine_flags[rel_id]      = flag >> NUM_LEVEL_BITS;
    }

    // resulting vector should just have OCT_COARSE, OCT_SPLIT, or OCT_NO_CHANGE
    return refine_flags;
}

bool Mesh::setMeshRefinementFlags(
    const std::vector<unsigned int> &refine_flags) {
    // explicitly set the refinement flags,
    assert(refine_flags.size() == m_uiNumLocalElements);

    // set all the elements to no change.
    for (unsigned int ele = m_uiElementLocalBegin; ele < m_uiElementLocalEnd;
         ele++)
        m_uiAllElements[ele].setFlag(((OCT_NO_CHANGE << NUM_LEVEL_BITS) |
                                      m_uiAllElements[ele].getLevel()));

    bool isMeshChangeLocal = false;

    for (unsigned int ele = m_uiElementLocalBegin; ele < m_uiElementLocalEnd;
         ele++) {
        const unsigned int rid = ele - m_uiElementLocalBegin;

        if (refine_flags[rid] == OCT_COARSE) {
            bool isCoarse = true;
            if (((ele + NUM_CHILDREN - 1) < m_uiElementLocalEnd) &&
                m_uiAllElements[ele + NUM_CHILDREN - 1].getParent() ==
                    m_uiAllElements[ele]
                        .getParent()) {  // all the 8 children are in the same
                                         // level,

                if (m_uiAllElements[ele].getLevel() ==
                    0)  // current element is the root cannnot coarsen anymore.
                    isCoarse = false;

                for (unsigned int child = 0; child < NUM_CHILDREN;
                     child++) {  // to check if all the children agrees to
                                 // coarsen.

                    if (refine_flags[rid + child] != OCT_COARSE) {
                        isCoarse = false;
                        break;
                    }
                }

            } else {  // all the 8 children are not in the same level.
                isCoarse = false;
            }

            if (isCoarse) {
                assert(((ele + NUM_CHILDREN - 1) < m_uiElementLocalEnd) &&
                       m_uiAllElements[ele + NUM_CHILDREN - 1].getParent() ==
                           m_uiAllElements[ele].getParent());

                for (unsigned int child = 0; child < NUM_CHILDREN; child++)
                    m_uiAllElements[ele + child].setFlag(
                        ((OCT_COARSE << NUM_LEVEL_BITS) |
                         m_uiAllElements[ele + child].getLevel()));

                isMeshChangeLocal = true;
                ele += (NUM_CHILDREN - 1);

            } else {
                m_uiAllElements[ele].setFlag(
                    ((OCT_NO_CHANGE << NUM_LEVEL_BITS) |
                     m_uiAllElements[ele].getLevel()));
            }

        } else if (refine_flags[rid] == OCT_SPLIT) {
            if ((m_uiAllElements[ele].getLevel() + MAXDEAPTH_LEVEL_DIFF + 1) <
                m_uiMaxDepth) {
                m_uiAllElements[ele].setFlag(((OCT_SPLIT << NUM_LEVEL_BITS) |
                                              m_uiAllElements[ele].getLevel()));
                isMeshChangeLocal = true;
            } else
                m_uiAllElements[ele].setFlag(
                    ((OCT_NO_CHANGE << NUM_LEVEL_BITS) |
                     m_uiAllElements[ele].getLevel()));

        } else {
            assert(refine_flags[rid] == OCT_NO_CHANGE);
            m_uiAllElements[ele].setFlag(((OCT_NO_CHANGE << NUM_LEVEL_BITS) |
                                          m_uiAllElements[ele].getLevel()));
        }
    }

    return isMeshChangeLocal;
}

void Mesh::octCoordToDomainCoord(const Point &oct_pt, Point &domain_pt) const {
    const double RgX   = (m_uiDMaxPt.x() - m_uiDMinPt.x());
    const double RgY   = (m_uiDMaxPt.y() - m_uiDMinPt.y());
    const double RgZ   = (m_uiDMaxPt.z() - m_uiDMinPt.z());

    const double octRg = (1u << (m_uiMaxDepth));

    double x           = (((oct_pt.x() - 0) * RgX) / octRg) + m_uiDMinPt.x();
    double y           = (((oct_pt.y() - 0) * RgY) / octRg) + m_uiDMinPt.y();
    double z           = (((oct_pt.z() - 0) * RgZ) / octRg) + m_uiDMinPt.z();

    domain_pt          = Point(x, y, z);
    return;
}

void Mesh::domainCoordToOctCoord(const Point &domain_pt, Point &oct_pt) const {
    const double RgX   = (m_uiDMaxPt.x() - m_uiDMinPt.x());
    const double RgY   = (m_uiDMaxPt.y() - m_uiDMinPt.y());
    const double RgZ   = (m_uiDMaxPt.z() - m_uiDMinPt.z());

    const double octRg = (1u << (m_uiMaxDepth));

    double x           = (((domain_pt.x() - m_uiDMinPt.x()) * octRg) / RgX);
    double y           = (((domain_pt.y() - m_uiDMinPt.y()) * octRg) / RgY);
    double z           = (((domain_pt.z() - m_uiDMinPt.z()) * octRg) / RgZ);

    oct_pt             = Point(x, y, z);
    return;
}

void Mesh::computeTreeNodeOwnerProc(const ot::TreeNode *pNodes, unsigned int n,
                                    int *ownerranks) const {
    if (m_uiIsActive) {
        std::vector<ot::SearchKey> keys;
        keys.resize(n);

        for (unsigned int i = 0; i < n; i++) {
            keys[i] = ot::SearchKey(pNodes[i]);
            keys[i].addOwner(i);
            ownerranks[i] = -1;
        }

        const unsigned int npes = this->getMPICommSize();

        const std::vector<ot::TreeNode> &sElements =
            this->getSplitterElements();
        for (unsigned int p = 0; p < npes; p++) {
            keys.push_back(ot::SearchKey(sElements[2 * p]));
            keys.back().addOwner(-1);
        }

        std::vector<ot::SearchKey> tmp;
        ot::SearchKey root(ot::TreeNode(0, 0, 0, 0, m_uiDim, m_uiMaxDepth));
        SFC::seqSort::SFC_treeSort(&(*(keys.begin())), keys.size(), tmp, tmp,
                                   tmp, m_uiMaxDepth, m_uiMaxDepth, root,
                                   ROOT_ROTATION, 1, TS_SORT_ONLY);

        std::vector<ot::Key> key_merged;
        mergeKeys(keys, key_merged);

        std::vector<ot::Key> sEleKeys;
        sEleKeys.resize(npes);
        for (unsigned int p = 0; p < npes; p++) {
            sEleKeys[p] = Key(sElements[2 * p]);
        }

        SFC::seqSearch::SFC_treeSearch(
            &(*(sEleKeys.begin())), &(*(key_merged.begin())), 0,
            sEleKeys.size(), 0, key_merged.size(), m_uiMaxDepth, m_uiMaxDepth,
            ROOT_ROTATION);
        unsigned int sBegin = 0;
        unsigned int sEnd;

        for (unsigned int p = 0; p < npes; p++) {
            assert((sEleKeys[p].getFlag() & OCT_FOUND));
            assert(key_merged[sEleKeys[p].getSearchResult()] == sEleKeys[p]);
            sBegin = sEleKeys[p].getSearchResult();
            (p < (npes - 1)) ? sEnd = sEleKeys[p + 1].getSearchResult() + 1
                             : sEnd = key_merged.size();

            for (unsigned int k = sBegin; k < sEnd; k++) {
                for (unsigned int w = 0;
                     w < key_merged[k].getOwnerList()->size(); w++) {
                    const unsigned kowner =
                        (*(key_merged[k].getOwnerList()))[w];
                    if (kowner >= 0) {
                        ownerranks[kowner] = p;
                    }
                }
            }
        }
    }

    return;
}

void Mesh::blkUnzipElementIDs(unsigned int blk,
                              std::vector<unsigned int> &eid) const {
    eid.clear();
    if (this->isActive()) {
        const ot::TreeNode *pNodes        = m_uiAllElements.data();

        const unsigned int nodeLocalBegin = this->getNodeLocalBegin();
        const unsigned int nodeLocalEnd   = this->getNodeLocalEnd();

        const unsigned int *e2n_cg        = &(*(this->getE2NMapping().begin()));
        const unsigned int *e2e           = &(*(this->getE2EMapping().begin()));

        const std::vector<ot::Block> &blkList = this->getLocalBlockList();
        const unsigned int nPe                = this->getNumNodesPerElement();

        unsigned int lookup, node_cg;
        unsigned int child[NUM_CHILDREN];

        if (blk > blkList.size()) return;

        const unsigned int pWidth   = blkList[blk].get1DPadWidth();
        const ot::TreeNode blkNode  = blkList[blk].getBlockNode();
        const unsigned int regLevel = blkList[blk].getRegularGridLev();

        eid.clear();
        unsigned int fchild[4];

        // Iterate via the block's range-based iterator so this works
        // for both SFC blocks (which use the begin/end range) and
        // non-SFC blocks from octree2BlockDecompositionRepartitioned
        // (which carry an explicit index list). A plain
        // getLocalElementBegin/End loop would visit no elements on
        // non-SFC blocks since begin == end == 0 there, silently
        // producing an empty eid and a zeroed unzip padding.
        for (unsigned int elem : blkList[blk]) {
            const unsigned int ei = (pNodes[elem].getX() - blkNode.getX()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ej = (pNodes[elem].getY() - blkNode.getY()) >>
                                    (m_uiMaxDepth - regLevel);
            const unsigned int ek = (pNodes[elem].getZ() - blkNode.getZ()) >>
                                    (m_uiMaxDepth - regLevel);

            const unsigned int emin = 0;
            const unsigned int emax =
                (1u << (regLevel - blkNode.getLevel())) - 1;

            if (pWidth > 0) {
                // we need to look for the boundary neigbours only when the
                // padding width is > 0 .
                if (ei == emin) {
                    // OCT_DIR_LEFT
                    const unsigned int dir = OCT_DIR_LEFT;
                    lookup                 = e2e[elem * NUM_FACES + dir];
                    if (lookup != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookup].getLevel() > regLevel) {
                            this->getFinerFaceNeighbors(elem, dir, fchild);
                            eid.push_back(fchild[0]);
                            eid.push_back(fchild[1]);
                            eid.push_back(fchild[2]);
                            eid.push_back(fchild[3]);

                        } else {
                            // neighbour octant is same lev or coarser
                            assert(pNodes[lookup].getLevel() <= regLevel);
                            eid.push_back(lookup);
                        }
                    }
                }

                if (ei == emax) {
                    // OCT_DIR_RIGHT
                    const unsigned int dir = OCT_DIR_RIGHT;
                    lookup                 = e2e[elem * NUM_FACES + dir];

                    if (lookup != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookup].getLevel() > regLevel) {
                            this->getFinerFaceNeighbors(elem, dir, fchild);
                            eid.push_back(fchild[0]);
                            eid.push_back(fchild[1]);
                            eid.push_back(fchild[2]);
                            eid.push_back(fchild[3]);

                        } else {
                            // neighbour octant is same lev or coarser
                            assert(pNodes[lookup].getLevel() <= regLevel);
                            eid.push_back(lookup);
                        }
                    }
                }

                if (ej == emin) {
                    // OCT_DIR_DOWN
                    const unsigned int dir = OCT_DIR_DOWN;
                    lookup                 = e2e[elem * NUM_FACES + dir];

                    if (lookup != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookup].getLevel() > regLevel) {
                            this->getFinerFaceNeighbors(elem, dir, fchild);
                            eid.push_back(fchild[0]);
                            eid.push_back(fchild[1]);
                            eid.push_back(fchild[2]);
                            eid.push_back(fchild[3]);

                        } else {
                            // neighbour octant is same lev or coarser
                            assert(pNodes[lookup].getLevel() <= regLevel);
                            eid.push_back(lookup);
                        }
                    }
                }

                if (ej == emax) {
                    // OCT_DIR_UP
                    const unsigned int dir = OCT_DIR_UP;
                    lookup                 = e2e[elem * NUM_FACES + dir];
                    if (lookup != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookup].getLevel() > regLevel) {
                            this->getFinerFaceNeighbors(elem, dir, fchild);
                            eid.push_back(fchild[0]);
                            eid.push_back(fchild[1]);
                            eid.push_back(fchild[2]);
                            eid.push_back(fchild[3]);

                        } else {
                            // neighbour octant is same lev or coarser
                            assert(pNodes[lookup].getLevel() <= regLevel);
                            eid.push_back(lookup);
                        }
                    }
                }

                if (ek == emin) {
                    // OCT_DIR_BACK
                    const unsigned int dir = OCT_DIR_BACK;
                    lookup                 = e2e[elem * NUM_FACES + dir];
                    if (lookup != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookup].getLevel() > regLevel) {
                            this->getFinerFaceNeighbors(elem, dir, fchild);
                            eid.push_back(fchild[0]);
                            eid.push_back(fchild[1]);
                            eid.push_back(fchild[2]);
                            eid.push_back(fchild[3]);

                        } else {
                            // neighbour octant is same lev or coarser
                            assert(pNodes[lookup].getLevel() <= regLevel);
                            eid.push_back(lookup);
                        }
                    }
                }

                if (ek == emax) {
                    // OCT_DIR_FRONT
                    const unsigned int dir = OCT_DIR_FRONT;
                    lookup                 = e2e[elem * NUM_FACES + dir];
                    if (lookup != LOOK_UP_TABLE_DEFAULT) {
                        if (pNodes[lookup].getLevel() > regLevel) {
                            this->getFinerFaceNeighbors(elem, dir, fchild);
                            eid.push_back(fchild[0]);
                            eid.push_back(fchild[1]);
                            eid.push_back(fchild[2]);
                            eid.push_back(fchild[3]);

                        } else {
                            // neighbour octant is same lev or coarser
                            assert(pNodes[lookup].getLevel() <= regLevel);
                            eid.push_back(lookup);
                        }
                    }
                }
            }
        }

        // now look for edge neighbors and vertex neighbors of the block, this
        // is only needed when the padding width is >0
        if (pWidth > 0) {
            const std::vector<unsigned int> blk2Edge_map =
                blkList[blk].getBlk2DiagMap_vec();
            const std::vector<unsigned int> blk2Vert_map =
                blkList[blk].getBlk2VertexMap_vec();
            const unsigned int blk_ele_1D = blkList[blk].getElemSz1D();

            for (unsigned int edir = 0; edir < NUM_EDGES; edir++) {
                for (unsigned int k = 0; k < blk_ele_1D; k++) {
                    if (blk2Edge_map[edir * (2 * blk_ele_1D) + 2 * k] !=
                        LOOK_UP_TABLE_DEFAULT) {
                        if (blk2Edge_map[edir * (2 * blk_ele_1D) + 2 * k + 0] ==
                            blk2Edge_map[edir * (2 * blk_ele_1D) + 2 * k + 1]) {
                            lookup = blk2Edge_map[edir * (2 * blk_ele_1D) +
                                                  2 * k + 0];
                            eid.push_back(lookup);

                        } else {
                            // slot[0] non-default, slot[1] may be default
                            // (only half the edge is covered); skip defaults.
                            lookup = blk2Edge_map[edir * (2 * blk_ele_1D) +
                                                  2 * k + 0];
                            eid.push_back(lookup);

                            lookup = blk2Edge_map[edir * (2 * blk_ele_1D) +
                                                  2 * k + 1];
                            if (lookup != LOOK_UP_TABLE_DEFAULT)
                                eid.push_back(lookup);
                        }
                    }
                }
            }

            for (unsigned int k = 0; k < blk2Vert_map.size(); k++) {
                lookup = blk2Vert_map[k];

                if (lookup != LOOK_UP_TABLE_DEFAULT) eid.push_back(lookup);
            }
        }

        std::sort(eid.begin(), eid.end());
        eid.erase(std::unique(eid.begin(), eid.end()), eid.end());
    }

    return;
}

void Mesh::repartitionMeshGlobal(bool do_block_creation,
                                 bool do_fastpart_filesave,
                                 std::string fileprefix) {
    if (!m_uiIsActive) return;

    if (m_partitionOption == PartitioningOptions::NoPartition) {
        return;
    }

    constexpr size_t RANK_TEST = 2;

    typedef unsigned int D_INT_L;
    // first generate the oct_connectivity_map, this gives us connectivity
    // in global IDs and also provides a local-to-global mapping

    int rank            = this->getMPIRank();
    int npes            = this->getMPICommSize();
    MPI_Comm commActive = this->getMPICommunicator();

    // -------- PHASE TIMING --------
    auto __t_start = MPI_Wtime();
    auto __t_prev  = __t_start;
    auto __phase   = [&](const char *name) {
        MPI_Barrier(commActive);
        double now = MPI_Wtime();
        if (rank == 0)
            std::cout << "[repart] phase '" << name << "' : "
                      << (now - __t_prev) * 1e3 << " ms" << std::endl;
        __t_prev = now;
    };

    auto [oct_connectivity_map, local_to_global, ele_offsets, ele_counts] =
        buildOctantConnectivityMap<D_INT_L>();
    __phase("buildOctantConnectivityMap");

    if (rank == 0)
        std::cout << rank << ":  LOCAL_TO_GLOBAL size=" << local_to_global.size()
                  << std::endl;

    // ele_offsets and ele_counts will help us figure out which process
    // belongs to which

    // then figure out node global IDs to get a local-to-global mapping for
    // them

    // Note: createLocalToGlobalE2N populated oct.e2n_dg with global DG
    // indices from the original mesh. That field is no longer needed:
    // the keyset-based ghost fetch uses only oct.e2e/edgeNeighbors/
    // vertexNeighbors, and the scatter map is built from the rebuilt
    // m_uiE2NMapping_DG (not oct.e2n_dg). Skipping it saves ~20 ms on
    // a 16k-element mesh.

    std::vector<D_INT_L> my_partition;

    if (m_partitionOption == PartitioningOptions::OriginalPartition) {
        my_partition = noPartitionChange(oct_connectivity_map);
    } else if (m_partitionOption == PartitioningOptions::RandomPartition) {
        my_partition = randomPartitioningSimple(oct_connectivity_map);
    } else if (m_partitionOption == PartitioningOptions::fastpart) {
        // quick conversion of oct_connectivity_map to wanted format

        // oct_element is from fastpart.h
        std::vector<oct_element> temp_oct_data(oct_connectivity_map.size());

        for (unsigned int i = 0; i < oct_connectivity_map.size(); ++i) {
            const auto &oct          = oct_connectivity_map[i];
            temp_oct_data[i].rank    = oct.rank;
            temp_oct_data[i].trank   = oct.trank;
            temp_oct_data[i].eid     = oct.eid;
            temp_oct_data[i].localid = oct.localid;
            for (unsigned int j = 0; j < 3; ++j) {
                temp_oct_data[i].coord[j] = oct.coord[j];
            }
            for (unsigned int j = 0; j < 6; ++j) {
                temp_oct_data[i].e2e[j] = oct.e2e[j];
            }
            temp_oct_data[i].level = oct.level;
#if 0
                for (unsigned int j = 0; j < 12; ++j) {
                    temp_oct_data[i].edgeNeighbors[j] = oct.edgeNeighbors[j];
                }
                for (unsigned int j = 0; j < 8; ++j) {
                    temp_oct_data[i].vertexNeighbors[j] =
                        oct.vertexNeighbors[j];
                }
#endif
        }

        // vtx_dist is a prefix scan of the element count for each MPI node
        fastpart_uint_t *vtx_dist = static_cast<fastpart_uint_t *>(
            malloc(ele_offsets.size() * sizeof(fastpart_uint_t)));

        for (unsigned int rk = 0; rk < ele_offsets.size(); rk++) {
            vtx_dist[rk] = ele_offsets[rk];
        }

        // and convert the number in each rank
        fastpart_uint_t *ele_counts_fp = static_cast<fastpart_uint_t *>(
            malloc(npes * sizeof(fastpart_uint_t)));
        for (unsigned int rk = 0; rk < npes; ++rk) {
            ele_counts_fp[rk] = ele_counts[rk];
        }

        if (do_fastpart_filesave) {
            dumpOctDataParallel(temp_oct_data, vtx_dist, ele_counts_fp,
                                oct_connectivity_map.size(),
                                fileprefix.c_str());
        }

        // TEMP: skip doing all of this, because we just need to save
        // everything
#ifndef _ONLY_DUMP_OCT_DATA_

        fastpart_uint_t *parts = static_cast<fastpart_uint_t *>(
            malloc(oct_connectivity_map.size() * sizeof(fastpart_uint_t)));

        fastpart_partgraph_octree(vtx_dist, temp_oct_data.data(), parts,
                                  &commActive);

        // BLOCK-ATOMIC POST-PROCESSING:
        // for each canonical SFC block on this source rank, override
        // fastpart's per-element decisions so all elements of that
        // block land on a single target rank. without this, ~16% of
        // canonical blocks get split across target ranks, which leaves
        // partial blocks with missing block-interior data on each
        // target rank → corrupted FD stencils → long-haul divergence.
        //
        // vote rule: target rank that already received the most
        // elements of this block wins. ties broken by smallest rank.
        // load-imbalance impact is bounded by max-block-size /
        // mean-elements-per-rank (≤ 1% in typical configurations).
        //
        // gated by DENDRO_DISABLE_BLOCK_VOTE=1 for diagnostic A/B.
        const char* disableVote = std::getenv("DENDRO_DISABLE_BLOCK_VOTE");
        if (!disableVote || disableVote[0] != '1') {
            struct AnchorKey {
                uint32_t x, y, z, lev;
                bool operator<(const AnchorKey& o) const {
                    if (x != o.x) return x < o.x;
                    if (y != o.y) return y < o.y;
                    if (z != o.z) return z < o.z;
                    return lev < o.lev;
                }
            };
            std::map<AnchorKey, std::vector<size_t>> blockGroups;
            for (size_t i = 0; i < oct_connectivity_map.size(); ++i) {
                const auto& oct = oct_connectivity_map[i];
                // skip elements without valid block info — leave them
                // on their fastpart-assigned target.
                if ((oct.blkMeta & (1u << 31)) == 0u) continue;
                AnchorKey k{oct.blkAnchorX, oct.blkAnchorY,
                            oct.blkAnchorZ, oct.blkAnchorLevel};
                blockGroups[k].push_back(i);
            }

            size_t reassigned = 0;
            for (auto& kv : blockGroups) {
                const auto& indices = kv.second;
                if (indices.size() <= 1) continue;
                // tally votes per target rank
                std::map<unsigned int, unsigned int> votes;
                for (size_t i : indices) votes[parts[i]]++;
                // pick winner: highest count, then lowest rank
                unsigned int winnerRank = 0;
                unsigned int winnerCount = 0;
                for (const auto& vk : votes) {
                    if (vk.second > winnerCount ||
                        (vk.second == winnerCount && vk.first < winnerRank)) {
                        winnerRank  = vk.first;
                        winnerCount = vk.second;
                    }
                }
                // reassign all block elements to winner
                for (size_t i : indices) {
                    if (parts[i] != winnerRank) {
                        parts[i] = winnerRank;
                        reassigned++;
                    }
                }
            }
            if (rank == 0) {
                std::cout << "[block-vote] rank " << rank
                          << " reassigned " << reassigned
                          << " elements via block-atomic vote (out of "
                          << oct_connectivity_map.size() << ")"
                          << std::endl;
            }
        }

        // with the target_locations in mind, we need to figure out our "new
        // partition"
        std::vector<unsigned int> to_send[m_uiActiveNpes];
        unsigned int total_send = 0;
        for (unsigned int i = 0; i < oct_connectivity_map.size(); ++i) {
            if (parts[i] == rank) {
                my_partition.push_back(oct_connectivity_map[i].eid);
                continue;
            }

            to_send[parts[i]].push_back(oct_connectivity_map[i].eid);
            total_send++;
        }
        // then flatten it
        std::vector<unsigned int> flatten_send(total_send);
        std::vector<int> nsend(npes, 0);
        unsigned int counter = 0;
        for (unsigned int i = 0; i < npes; i++) {
            for (const unsigned int &snd : to_send[i]) {
                flatten_send[counter++] = snd;
            }
            nsend[i] = to_send[i].size();
        }
        std::vector<int> nrecv(npes);

        MPI_Alltoall(nsend.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT,
                     commActive);

        std::vector<int> sendOffset(npes);
        std::vector<int> recvOffset(npes);
        sendOffset[0] = 0;
        for (int i = 1; i < npes; ++i) {
            sendOffset[i] = sendOffset[i - 1] + nsend[i - 1];
            recvOffset[i] = recvOffset[i - 1] + nrecv[i - 1];
        }
        unsigned int nTotalRecv = recvOffset[npes - 1] + nrecv[npes - 1];
        std::vector<unsigned int> flatten_recv(nTotalRecv);

        MPI_Alltoallv(flatten_send.data(), nsend.data(), sendOffset.data(),
                      MPI_UNSIGNED, flatten_recv.data(), nrecv.data(),
                      recvOffset.data(), MPI_UNSIGNED, commActive);

        my_partition.insert(my_partition.end(), flatten_recv.begin(),
                            flatten_recv.end());
        std::sort(my_partition.begin(), my_partition.end());
        free(parts);
#endif

        free(vtx_dist);
        free(ele_counts_fp);
    }
    __phase("partition-decision");

#ifdef _ONLY_DUMP_OCT_DATA_
    // NOTE: early return, we just want to save the .oct files
    return;
#endif

    if (rank == 0)
        std::cout << rank << ": ORIGINAL PARTITION SIZE - "
                  << oct_connectivity_map.size() << " NEW PARTITION SIZE - "
                  << my_partition.size() << std::endl;

    auto new_oct_connectivity_map = getOctDataFromOtherProcesses(
        oct_connectivity_map, ele_offsets, ele_counts, my_partition);
    __phase("fetch-my-partition");

    // now that we have all of this information, we just need to get our new
    // assignment and then probe the rest of the mesh to get the necessary
    // information

    // assign back in the retained elements
    for (auto &ele_id : my_partition) {
        if (ele_id >= ele_offsets[rank] && ele_id < ele_offsets[rank + 1]) {
            // make sure the target is properly set here too for sending
            // information later
            oct_connectivity_map[ele_id - ele_offsets[rank]].trank = rank;

            new_oct_connectivity_map.push_back(
                oct_connectivity_map[ele_id - ele_offsets[rank]]);
        }
    }
    // then sort by global ID
    std::sort(new_oct_connectivity_map.begin(), new_oct_connectivity_map.end(),
              [](const oct_data<D_INT_L> &o1, const oct_data<D_INT_L> &o2) {
                  return o1.trank < o2.trank;
              });

    size_t newLocalBegin = LOOK_UP_TABLE_DEFAULT;
    size_t newNumEle;
    std::vector<D_INT_L> post_first_round_comms_ids;


    // ================= KEYSET-BASED GHOST FETCH =================
    // Replaces the old 7-round BFS + E2N-fixup (21+ collectives, O(N^2)
    // worst case) with a single R1 + R2 keyset fetch (4 collectives).
    // Works because oct_data already stores each element's 26-neighbor
    // global IDs (e2e[6] + edgeNeighbors[12] + vertexNeighbors[8]),
    // populated by buildOctantConnectivityMap on the original mesh.
    // Those neighbor lists are invariant under repartitioning, so we
    // can compute the ghost keyset purely from local data.

    auto collectNeighborKeys = [](const std::vector<oct_data<D_INT_L>> &src,
                                  std::set<D_INT_L> &out) {
        for (const auto &oct : src) {
            for (unsigned int i = 0; i < 6; ++i)
                if (oct.e2e[i] != LOOK_UP_TABLE_DEFAULT)
                    out.insert(oct.e2e[i]);
            for (unsigned int i = 0; i < 12; ++i)
                if (oct.edgeNeighbors[i] != LOOK_UP_TABLE_DEFAULT)
                    out.insert(oct.edgeNeighbors[i]);
            for (unsigned int i = 0; i < 8; ++i)
                if (oct.vertexNeighbors[i] != LOOK_UP_TABLE_DEFAULT)
                    out.insert(oct.vertexNeighbors[i]);
        }
    };

    auto splitLocalRemote =
        [&](std::set<D_INT_L> &keys,
            std::vector<oct_data<D_INT_L>> &local_out,
            std::vector<D_INT_L> &remote_out) {
            for (auto id : keys) {
                if (id >= ele_offsets[rank] &&
                    id < ele_offsets[rank + 1]) {
                    local_out.push_back(
                        oct_connectivity_map[id - ele_offsets[rank]]);
                } else {
                    remote_out.push_back(id);
                }
            }
        };

    std::set<D_INT_L> present;
    for (const auto &oct : new_oct_connectivity_map) present.insert(oct.eid);

    // ---- R1 ----
    std::set<D_INT_L> r1_keys;
    collectNeighborKeys(new_oct_connectivity_map, r1_keys);
    for (auto id : present) r1_keys.erase(id);

    std::vector<oct_data<D_INT_L>> r1_local;
    std::vector<D_INT_L> r1_remote;
    splitLocalRemote(r1_keys, r1_local, r1_remote);

    auto r1_fetched = getOctDataFromOtherProcesses(
        oct_connectivity_map, ele_offsets, ele_counts, r1_remote, false);

    new_oct_connectivity_map.insert(new_oct_connectivity_map.end(),
                                    r1_fetched.begin(), r1_fetched.end());
    new_oct_connectivity_map.insert(new_oct_connectivity_map.end(),
                                    r1_local.begin(), r1_local.end());
    for (const auto &oct : r1_fetched) present.insert(oct.eid);
    for (const auto &oct : r1_local) present.insert(oct.eid);

    // post_first_round_comms_ids = all non-ghostTwo eids (local + R1),
    // used later by the element scatter map.
    for (const auto &oct : new_oct_connectivity_map)
        post_first_round_comms_ids.push_back(oct.eid);

    // ---- R2 (marked ghostTwo) ----
    std::set<D_INT_L> r2_keys;
    collectNeighborKeys(r1_fetched, r2_keys);
    collectNeighborKeys(r1_local, r2_keys);
    for (auto id : present) r2_keys.erase(id);

    std::vector<oct_data<D_INT_L>> r2_local;
    std::vector<D_INT_L> r2_remote;
    splitLocalRemote(r2_keys, r2_local, r2_remote);

    auto r2_fetched = getOctDataFromOtherProcesses(
        oct_connectivity_map, ele_offsets, ele_counts, r2_remote, false);

    for (auto &oct : r2_fetched) oct.isGhostTwo = true;
    for (auto &oct : r2_local) oct.isGhostTwo = true;
    new_oct_connectivity_map.insert(new_oct_connectivity_map.end(),
                                    r2_fetched.begin(), r2_fetched.end());
    new_oct_connectivity_map.insert(new_oct_connectivity_map.end(),
                                    r2_local.begin(), r2_local.end());
    for (const auto &oct : r2_fetched) present.insert(oct.eid);
    for (const auto &oct : r2_local) present.insert(oct.eid);

    // ---- R3 (also marked ghostTwo, for R2 boundary candidate completeness) ----
    // R2 elements need their own 26-neighborhood available so my canonical
    // E2N build on this rank produces the same owner selection for R2's
    // face/edge/vertex nodes as the home rank would. Without R3, R2's
    // candidate set is truncated and we get rank-inconsistent canonical
    // ownership, which later makes createVector on different ranks
    // write different f(physical_pos) values at the same CG slot.
    std::set<D_INT_L> r3_keys;
    collectNeighborKeys(r2_fetched, r3_keys);
    collectNeighborKeys(r2_local, r3_keys);
    for (auto id : present) r3_keys.erase(id);

    std::vector<oct_data<D_INT_L>> r3_local;
    std::vector<D_INT_L> r3_remote;
    splitLocalRemote(r3_keys, r3_local, r3_remote);

    auto r3_fetched = getOctDataFromOtherProcesses(
        oct_connectivity_map, ele_offsets, ele_counts, r3_remote, false);

    for (auto &oct : r3_fetched) oct.isGhostTwo = true;
    for (auto &oct : r3_local) oct.isGhostTwo = true;
    new_oct_connectivity_map.insert(new_oct_connectivity_map.end(),
                                    r3_fetched.begin(), r3_fetched.end());
    new_oct_connectivity_map.insert(new_oct_connectivity_map.end(),
                                    r3_local.begin(), r3_local.end());

    std::sort(new_oct_connectivity_map.begin(),
              new_oct_connectivity_map.end(),
              [](const oct_data<D_INT_L> &a, const oct_data<D_INT_L> &b) {
                  if (a.trank != b.trank) return a.trank < b.trank;
                  return a.eid < b.eid;
              });

    {
        size_t ctr = 0;
        for (auto &o : new_oct_connectivity_map) {
            if (o.trank == rank) break;
            ctr++;
        }
        newLocalBegin = ctr;
        newNumEle     = new_oct_connectivity_map.size();
    }
    size_t newLocalEnd = newLocalBegin + my_partition.size();
    __phase("keyset ghost fetch (R1+R2)");

    // now we can rebuild the E2E map based on this data!
    std::vector<D_INT_L> newE2EMap(newNumEle * this->getNumDirections(),
                                   LOOK_UP_TABLE_DEFAULT);

    // Create global-to-local map. Prefer LOCAL element indices over
    // ghost indices so that scatter map lookups find CG values in the
    // local range (populated by createVector).
    std::map<D_INT_L, D_INT_L> globaltoNewLocal;
    // First pass: all elements
    for (size_t eid_local = 0; eid_local < newNumEle; eid_local++) {
        globaltoNewLocal[new_oct_connectivity_map[eid_local].eid] = eid_local;
    }
    // Second pass: overwrite with LOCAL elements (guaranteed to have
    // CG values in the local range after createVector)
    for (size_t eid_local = newLocalBegin; eid_local < newLocalEnd;
         eid_local++) {
        globaltoNewLocal[new_oct_connectivity_map[eid_local].eid] = eid_local;
    }

    // CREATE FULL E2E MAP BASED ON DATA
    for (size_t eid_local = 0; eid_local < newNumEle; eid_local++) {
        for (int faceid = 0; faceid < this->getNumDirections(); faceid++) {
            if (new_oct_connectivity_map[eid_local].e2e[faceid] !=
                LOOK_UP_TABLE_DEFAULT) {
                D_INT_L tempVal =
                    new_oct_connectivity_map[eid_local].e2e[faceid];
                if (globaltoNewLocal.find(tempVal) != globaltoNewLocal.end()) {
                    newE2EMap[eid_local * this->getNumDirections() + faceid] =
                        globaltoNewLocal[new_oct_connectivity_map[eid_local]
                                             .e2e[faceid]];
                } else {
                    newE2EMap[eid_local * this->getNumDirections() + faceid] =
                        LOOK_UP_TABLE_DEFAULT;
                }
            }
        }
    }
    // done with E2E map creation!

    // --------------------
    // BUILD E2N USING buildE2NMap
    // Use the same E2E-based algorithm as the original mesh construction.
    // This correctly determines hanging node relationships from element
    // levels via the E2E mapping, avoiding E2N ownership mismatches.
    //
    // buildE2NMap operates on member variables, so we must first swap
    // in the new E2E and AllElements.

    std::swap(m_uiE2EMapping, newE2EMap);

    // Reconstruct m_uiAllElements from oct_data coordinates AND
    // copy received ownerMasks into m_uiOwnerMask in lockstep — the
    // mask rides with each element across the partition exchange.
    // canonical block info rides along the same way so post-partition
    // block setup can reconstruct the source rank's SFC blocks.
    std::vector<ot::TreeNode> newAllElements;
    newAllElements.reserve(new_oct_connectivity_map.size());
    std::vector<uint32_t> newOwnerMask;
    newOwnerMask.reserve(new_oct_connectivity_map.size());
    std::vector<CanonicalBlockInfo> newBlockInfo;
    newBlockInfo.reserve(new_oct_connectivity_map.size());
    for (const auto &oct : new_oct_connectivity_map) {
        unsigned int psz = 1u << (m_uiMaxDepth - oct.level - 1);
        ot::TreeNode temp(oct.coord[0] - psz, oct.coord[1] - psz,
                          oct.coord[2] - psz, oct.level, 3, m_uiMaxDepth);
        temp.setFlag(oct.flag);
        newAllElements.push_back(temp);
        newOwnerMask.push_back(oct.ownerMask);

        CanonicalBlockInfo bi;
        bi.anchorX     = oct.blkAnchorX;
        bi.anchorY     = oct.blkAnchorY;
        bi.anchorZ     = oct.blkAnchorZ;
        bi.anchorLevel = oct.blkAnchorLevel;
        bi.meta        = oct.blkMeta;
        newBlockInfo.push_back(bi);
    }
    std::swap(m_uiAllElements, newAllElements);
    std::swap(m_uiOwnerMask, newOwnerMask);
    std::swap(m_uiBlockInfo, newBlockInfo);

    // Update element ranges and counts
    m_uiElementPreGhostBegin  = 0;
    m_uiElementPreGhostEnd    = newLocalBegin;
    m_uiElementLocalBegin     = newLocalBegin;
    m_uiElementLocalEnd       = newLocalEnd;
    m_uiElementPostGhostBegin = newLocalEnd;
    m_uiElementPostGhostEnd   = new_oct_connectivity_map.size();

    m_uiNumLocalElements     = newLocalEnd - newLocalBegin;
    m_uiNumPreGhostElements  = newLocalBegin;
    m_uiNumPostGhostElements =
        m_uiElementPostGhostEnd - m_uiElementPostGhostBegin;
    m_uiNumTotalElements = m_uiNumPreGhostElements + m_uiNumLocalElements +
                           m_uiNumPostGhostElements;

    // Rebuild nodal map validity
    m_uiIsNodalMapValid.clear();
    m_uiIsNodalMapValid.resize(m_uiNumTotalElements, true);
    for (unsigned int e = 0; e < m_uiNumTotalElements; e++) {
        if (new_oct_connectivity_map[e].isGhostTwo)
            m_uiIsNodalMapValid[e] = false;
    }

    // Build order-2 E2N from E2E (handles hanging nodes correctly)
    {
        const unsigned int eleOrder = m_uiElementOrder;
        const unsigned int pp       = 2;
        m_uiElementOrder            = pp;
        if (m_uiDim == 2)
            m_uiNpE = (pp + 1) * (pp + 1);
        else if (m_uiDim == 3)
            m_uiNpE = (pp + 1) * (pp + 1) * (pp + 1);

        buildE2NMap();
        __phase("buildE2NMap(order=2)");


        // Zero out scatter maps so the expansion code's scatter map
        // section produces empty results (we build our own below)
        m_uiSendNodeCount.assign(m_uiActiveNpes, 0);
        m_uiRecvNodeCount.assign(m_uiActiveNpes, 0);
        m_uiSendNodeOffset.assign(m_uiActiveNpes, 0);
        m_uiRecvNodeOffset.assign(m_uiActiveNpes, 0);
        m_uiScatterMapActualNodeSend.clear();
        m_uiScatterMapActualNodeRecv.clear();

        // Expand to full order and rebuild CG/DG + node ranges
        buildE2NWithSMRepartitioned(eleOrder);
        __phase("buildE2NWithSMRepartitioned(expand)");

        // Diagnostic: how many representative-level ownership decisions
        // does the post-partition cascade disagree on, vs the
        // pre-partition mask transported with each element via oct_data?
        static const char* mask_dbg_env =
            std::getenv("DENDRO_VALIDATE_MASK");
        if (mask_dbg_env && mask_dbg_env[0] == '1'
            && mask_dbg_env[1] == '\0') {
            const size_t nDisagree =
                this->validateOwnerMasksAgainstCurrentCascade();
            std::cout << "[mask-validate r" << m_uiActiveRank
                      << "] post-repartition cascade vs mask"
                      << " disagreements=" << nDisagree
                      << " (out of " << (m_uiAllElements.size() * 27)
                      << " representative-level decisions)"
                      << std::endl;
        }

        // Stage 3: patch over-claim disagreements using mask-driven
        // ownership. Gated by DENDRO_USE_MASK_OWNERSHIP=1.
        static const char* mask_use_env =
            std::getenv("DENDRO_USE_MASK_OWNERSHIP");
        if (mask_use_env && mask_use_env[0] == '1'
            && mask_use_env[1] == '\0') {
            this->patchE2NCgFromMasks();
        }

        // Stage 4: phys_pos audit + repair. Walks local elements;
        // for each sub, checks E2N_CG[e*npe+sub]'s actual phys_pos
        // against expected. Repairs same-level wrong-routings;
        // preserves legitimate hanging-edge p2c (owner_level<elem_level).
        // Must run BEFORE the scatter-map rebuild + buildZipPlan so
        // the corrected E2N_CG flows into both. Default ON.
        static const char* audit_env = std::getenv("DENDRO_E2N_AUDIT");
        const bool audit_on = !audit_env
            || (audit_env[0] == '1' && audit_env[1] == '\0');
        // Pre-audit TN-based canonicalization (Option A — tested,
        // NOT USEFUL at current ghost-layer config). Octree parents
        // are not leaves (they're replaced by 8 children in the leaf
        // tree), so pNodes[e].getParent() returns a TreeNode that's
        // not in m_uiAllElements (parent_local=0 in practice). The
        // audit's phys-pos lookup finds canonical CGs at SHARING
        // NEIGHBOR leaves at the right physical position, which IS
        // reliably in m_uiAllElements. Default OFF; kept for future
        // revisit if the ghost layer expands to include octree
        // parents. See docs/findings_2026-05-14e.md follow-up.
        static const char* canontn_env =
            std::getenv("DENDRO_E2N_CANON_TN");
        const bool canontn_on =
            canontn_env && canontn_env[0] == '1'
            && canontn_env[1] == '\0';
        if (canontn_on) {
            this->canonicalizeHangingFaceRoutingTN();
        }
        __phase("e2n-canon-tn");

        if (audit_on) {
            this->auditAndRepairE2NCgPhysPos();
        }
        __phase("e2n-audit");
    }

    // --------------------
    // REBUILD NODAL SCATTER MAPS
    // The expansion code produced empty scatter maps (we zeroed the
    // inputs). Build them from the now-correct member E2N by iterating
    // over ghost element nodes and checking DG ownership.
    {
        std::vector<unsigned int> recvNodeSM_r[npes];
        std::vector<unsigned int> recvNodeDGG[npes];
        std::vector<unsigned char> recvNodeIsDG[npes];
        std::vector<int> recvCount(npes, 0);

        // CG path for local + R1 + R2 interior: route via canonical
        // owner decoded from E2N_DG.
        //
        // DG path for R2/R3 boundary: receiver and home rank may
        // disagree on canonical owner (candidate-set incomplete even
        // after R3 fetch in some cascade orders), so route (R2_gid, n)
        // to R2's home rank with explicit DG flag so sender reads from
        // m_uiLocalNodalDG (= f at R2's own sub physical). This keeps
        // the delivered value consistent with what receiver's E2N
        // cgIdx represents when the chain is truncated. For collision
        // cases (same cgIdx also referenced by a CG-path entry from
        // an R1 element via canonical chain), master's 1:1 mirror
        // convention ensures the physical positions coincide and
        // values match.
        const unsigned int eOrd_s = m_uiElementOrder;
        for (unsigned int ele_id = 0; ele_id < m_uiNumTotalElements;
             ele_id++) {
            const bool isR2 = new_oct_connectivity_map[ele_id].isGhostTwo;

            for (unsigned int n = 0; n < m_uiNpE; ++n) {
                unsigned int ni = n % (eOrd_s + 1);
                unsigned int nj = (n / (eOrd_s + 1)) % (eOrd_s + 1);
                unsigned int nk = n / ((eOrd_s + 1) * (eOrd_s + 1));
                bool onBoundary = (ni == 0) || (ni == eOrd_s) ||
                                  (nj == 0) || (nj == eOrd_s) ||
                                  (nk == 0) || (nk == eOrd_s);

                unsigned int cgIdx = m_uiE2NMapping_CG[ele_id * m_uiNpE + n];

                if (cgIdx >= m_uiNodeLocalBegin &&
                    cgIdx < m_uiNodeLocalEnd)
                    continue;

                unsigned int ownerGid;
                unsigned int ownerTrank;
                unsigned int sub;
                unsigned char isDG;

                unsigned int ownerDG =
                    m_uiE2NMapping_DG[ele_id * m_uiNpE + n];
                unsigned int ownerLocal = ownerDG / m_uiNpE;
                unsigned int ownerSub   = ownerDG % m_uiNpE;
                const bool selfOwned =
                    (ownerLocal == ele_id && ownerSub == n);

                if (isR2 && onBoundary) {
                    // R2/R3 boundary: always DG path. The DG buffer on
                    // the owner's rank holds f at that element's own
                    // sub physical — rank-independent and matches what
                    // receiver's cgIdx represents (via 1:1 mirror
                    // convention, owner's sub physical coincides with
                    // child's at shared physical positions for
                    // same-level, or is the hanging-interpolation basis
                    // for coarser-level cases which master's cascade
                    // would read the same way).
                    if (selfOwned) {
                        ownerGid   = new_oct_connectivity_map[ele_id].eid;
                        ownerTrank = new_oct_connectivity_map[ele_id].trank;
                        sub        = n;
                    } else {
                        ownerGid =
                            new_oct_connectivity_map[ownerLocal].eid;
                        ownerTrank =
                            new_oct_connectivity_map[ownerLocal].trank;
                        sub = ownerSub;
                    }
                    isDG = 1;
                } else {
                    sub      = ownerSub;
                    ownerGid = new_oct_connectivity_map[ownerLocal].eid;
                    ownerTrank =
                        new_oct_connectivity_map[ownerLocal].trank;
                    isDG = 0;
                }

                if (ownerTrank == rank) continue;

                recvNodeSM_r[ownerTrank].push_back(cgIdx);
                recvNodeDGG[ownerTrank].push_back(
                    ownerGid * m_uiNpE + sub);
                recvNodeIsDG[ownerTrank].push_back(isDG);
            }
        }

        // Dedup per-rank: prefer CG entry over DG entry for the same
        // cgIdx. CG path goes via the canonical owner (which is the
        // physical "source of truth" for the CG slot under master's
        // cascade + 1:1 mirror convention). DG path delivers
        // f(element's own sub) — only correct when receiver's E2N
        // self-owns this CG, otherwise it'd deliver the wrong physical
        // value for hanging cases.
        for (unsigned int p = 0; p < npes; p++) {
            std::vector<unsigned int> outSM;
            std::vector<unsigned int> outDGG;
            std::vector<unsigned char> outIsDG;
            outSM.reserve(recvNodeSM_r[p].size());
            outDGG.reserve(recvNodeSM_r[p].size());
            outIsDG.reserve(recvNodeSM_r[p].size());
            std::set<unsigned int> seen;
            // First pass: keep CG entries
            for (size_t i = 0; i < recvNodeSM_r[p].size(); i++) {
                if (!recvNodeIsDG[p][i] &&
                    seen.insert(recvNodeSM_r[p][i]).second) {
                    outSM.push_back(recvNodeSM_r[p][i]);
                    outDGG.push_back(recvNodeDGG[p][i]);
                    outIsDG.push_back(recvNodeIsDG[p][i]);
                }
            }
            // Second pass: keep DG entries for cgIdx not already seen
            for (size_t i = 0; i < recvNodeSM_r[p].size(); i++) {
                if (recvNodeIsDG[p][i] &&
                    seen.insert(recvNodeSM_r[p][i]).second) {
                    outSM.push_back(recvNodeSM_r[p][i]);
                    outDGG.push_back(recvNodeDGG[p][i]);
                    outIsDG.push_back(recvNodeIsDG[p][i]);
                }
            }
            std::swap(recvNodeSM_r[p], outSM);
            std::swap(recvNodeDGG[p], outDGG);
            std::swap(recvNodeIsDG[p], outIsDG);
            recvCount[p] = recvNodeSM_r[p].size();
        }

        // Offsets
        std::vector<int> recvOff(npes, 0);
        for (int p = 1; p < npes; p++)
            recvOff[p] = recvOff[p - 1] + recvCount[p - 1];
        int totalRecv = recvOff[npes - 1] + recvCount[npes - 1];

        // Exchange counts to get send counts
        std::vector<int> sendCount(npes);
        MPI_Alltoall(recvCount.data(), 1, MPI_INT, sendCount.data(), 1,
                     MPI_INT, commActive);

        std::vector<int> sendOff(npes, 0);
        for (int p = 1; p < npes; p++)
            sendOff[p] = sendOff[p - 1] + sendCount[p - 1];
        int totalSend = sendOff[npes - 1] + sendCount[npes - 1];

        // Exchange DG globals
        std::vector<long unsigned int> sendBuf(totalSend);
        std::vector<long unsigned int> recvBuf(totalRecv);
        {
            int off = 0;
            for (int p = 0; p < npes; p++)
                for (const auto &v : recvNodeDGG[p])
                    recvBuf[off++] = v;
        }
        MPI_Alltoallv(recvBuf.data(), recvCount.data(), recvOff.data(),
                      MPI_UNSIGNED_LONG, sendBuf.data(), sendCount.data(),
                      sendOff.data(), MPI_UNSIGNED_LONG, commActive);

        // Exchange the DG tags in parallel so the sender knows which
        // entries should read from m_uiLocalNodalDG vs vec.
        std::vector<unsigned char> sendIsDGBuf(totalSend);
        std::vector<unsigned char> recvIsDGBuf(totalRecv);
        {
            int off = 0;
            for (int p = 0; p < npes; p++)
                for (const auto &v : recvNodeIsDG[p])
                    recvIsDGBuf[off++] = v;
        }
        MPI_Alltoallv(recvIsDGBuf.data(), recvCount.data(), recvOff.data(),
                      MPI_UNSIGNED_CHAR, sendIsDGBuf.data(), sendCount.data(),
                      sendOff.data(), MPI_UNSIGNED_CHAR, commActive);

        // Decode (ownerGid, sub) on sender. Three modes:
        //  1. Receiver tagged isDG=1 (R2 boundary): use DG path,
        //     sendSM encodes (localEle - localBegin) * NpE + sub.
        //  2. Receiver tagged isDG=0 and sender's E2N_CG is LOCAL:
        //     CG path, sendSM = E2N_CG value (populated by createVector).
        //  3. Receiver tagged isDG=0 but sender's E2N_CG is GHOST:
        //     sender has gid local but its cascade lands on a ghost
        //     cg slot. Fall back to DG: send vec[ghost_cg] via the
        //     receiver-side readFromGhostBegin translation. This is
        //     only a fallback for the downstream orphan-cg issue in
        //     graph-partitioned meshes; Mesh::orphanPreGather in the
        //     user code is the actual workaround.
        std::vector<unsigned int> sendSM(totalSend);
        std::vector<unsigned char> sendIsDG(totalSend, 0);
        for (int i = 0; i < totalSend; i++) {
            unsigned int gid =
                static_cast<unsigned int>(sendBuf[i] / m_uiNpE);
            unsigned int nid =
                static_cast<unsigned int>(sendBuf[i] % m_uiNpE);
            unsigned int localEle = globaltoNewLocal.count(gid)
                                        ? globaltoNewLocal[gid]
                                        : m_uiElementLocalBegin;
            const bool inLocalRange =
                (localEle >= m_uiElementLocalBegin &&
                 localEle < m_uiElementLocalEnd);

            if (sendIsDGBuf[i] && inLocalRange) {
                sendSM[i] =
                    (localEle - m_uiElementLocalBegin) * m_uiNpE + nid;
                sendIsDG[i] = 1;
                continue;
            }

            unsigned int cg = m_uiE2NMapping_CG[localEle * m_uiNpE + nid];
            bool cgLocal    = (cg >= m_uiNodeLocalBegin &&
                            cg < m_uiNodeLocalEnd);

            if (cgLocal) {
                sendSM[i]   = cg;
                sendIsDG[i] = 0;
            } else if (inLocalRange) {
                sendSM[i] =
                    (localEle - m_uiElementLocalBegin) * m_uiNpE + nid;
                sendIsDG[i] = 1;
            } else {
                sendSM[i]   = cg;
                sendIsDG[i] = 0;
            }
        }

        // Flatten recv scatter map
        std::vector<unsigned int> recvSM(totalRecv);
        {
            int off = 0;
            for (int p = 0; p < npes; p++)
                for (const auto &v : recvNodeSM_r[p])
                    recvSM[off++] = v;
        }

        // Swap into member variables
        m_uiSendNodeCount  = convertVectorType<int, unsigned int>(sendCount);
        m_uiRecvNodeCount  = convertVectorType<int, unsigned int>(recvCount);
        m_uiSendNodeOffset = convertVectorType<int, unsigned int>(sendOff);
        m_uiRecvNodeOffset = convertVectorType<int, unsigned int>(recvOff);
        std::swap(m_uiScatterMapActualNodeSend, sendSM);
        std::swap(m_uiScatterMapActualNodeRecv, recvSM);
        std::swap(m_uiScatterMapSendIsDG, sendIsDG);

        m_uiSendBufferNodes.resize(totalSend);
        m_uiRecvBufferNodes.resize(totalRecv);

        m_uiSendProcList.clear();
        m_uiRecvProcList.clear();
        for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
            if (m_uiSendNodeCount[p] != 0) m_uiSendProcList.push_back(p);
            if (m_uiRecvNodeCount[p] != 0) m_uiRecvProcList.push_back(p);
        }
    }
    __phase("nodal-scatter-map");

    // -----
    // ELEMENT SCATTERMAP
    std::set<unsigned int> scatterMapSend_R1[npes];
    // get the element scattermap as well

    // post_first_round_comms_ids are *all* the elements by global ID that
    // *actually* work, need to convert them to locals, then sort
    std::vector<unsigned int> post_first_round_comms_vec_local;
    post_first_round_comms_vec_local.reserve(post_first_round_comms_ids.size());
    for (const auto &ele_id_global : post_first_round_comms_ids) {
        // fortunately, since they were fetched, they're all captured
        const unsigned int local_id = globaltoNewLocal[ele_id_global];
        post_first_round_comms_vec_local.push_back(local_id);
    }

    // then build up ele_scattermap
    std::sort(post_first_round_comms_vec_local.begin(),
              post_first_round_comms_vec_local.end());
    for (const auto &ele_id : post_first_round_comms_vec_local) {
        if (ele_id >= newLocalBegin && ele_id < newLocalEnd) {
            // don't process local stuff for this map
            continue;
        }

        const auto &oct         = new_oct_connectivity_map[ele_id];
        const D_INT_L procOwner = oct.trank;

        unsigned int lookup[NUM_CHILDREN];

        for (const unsigned int dir :
             {OCT_DIR_LEFT, OCT_DIR_RIGHT, OCT_DIR_DOWN, OCT_DIR_UP,
              OCT_DIR_BACK, OCT_DIR_FRONT}) {
            getElementalFaceNeighbors(ele_id, dir, lookup);
            if (lookup[1] != LOOK_UP_TABLE_DEFAULT) {
                if (lookup[1] >= newLocalBegin && lookup[1] < newLocalEnd) {
                    scatterMapSend_R1[procOwner].insert(lookup[1] -
                                                        newLocalBegin);
                }
            }
        }

        for (const unsigned int dir :
             {OCT_DIR_LEFT_DOWN, OCT_DIR_LEFT_UP, OCT_DIR_LEFT_BACK,
              OCT_DIR_LEFT_FRONT, OCT_DIR_RIGHT_DOWN, OCT_DIR_RIGHT_UP,
              OCT_DIR_RIGHT_BACK, OCT_DIR_RIGHT_FRONT, OCT_DIR_DOWN_BACK,
              OCT_DIR_DOWN_FRONT, OCT_DIR_UP_BACK, OCT_DIR_UP_FRONT}) {
            getElementalEdgeNeighbors(ele_id, dir, lookup);
            for (unsigned int lookup_id = 1; lookup_id < 4; ++lookup_id) {
                if (lookup[lookup_id] != LOOK_UP_TABLE_DEFAULT) {
                    if (lookup[lookup_id] >= newLocalBegin &&
                        lookup[lookup_id] < newLocalEnd) {
                        scatterMapSend_R1[procOwner].insert(lookup[lookup_id] -
                                                            newLocalBegin);
                    }
                }
            }
        }

        for (const unsigned int dir :
             {OCT_DIR_LEFT_DOWN_BACK, OCT_DIR_RIGHT_DOWN_BACK,
              OCT_DIR_LEFT_UP_BACK, OCT_DIR_RIGHT_UP_BACK,
              OCT_DIR_LEFT_DOWN_FRONT, OCT_DIR_RIGHT_DOWN_FRONT,
              OCT_DIR_LEFT_UP_FRONT, OCT_DIR_RIGHT_UP_FRONT}) {
            getElementalVertexNeighbors(ele_id, dir, lookup);
            for (unsigned int lookup_id = 1; lookup_id < NUM_CHILDREN;
                 ++lookup_id) {
                if (lookup[lookup_id] != LOOK_UP_TABLE_DEFAULT) {
                    if (lookup[lookup_id] >= newLocalBegin &&
                        lookup[lookup_id] < newLocalEnd) {
                        scatterMapSend_R1[procOwner].insert(lookup[lookup_id] -
                                                            newLocalBegin);
                    }
                }
            }
        }
    }

    // now we can flatten the data
    m_uiScatterMapElementRound1.clear();
    std::fill(m_uiSendEleCount.begin(), m_uiSendEleCount.end(), 0);
    std::fill(m_uiRecvEleCount.begin(), m_uiRecvEleCount.end(), 0);
    std::fill(m_uiSendEleOffset.begin(), m_uiSendEleOffset.end(), 0);
    std::fill(m_uiRecvEleOffset.begin(), m_uiRecvEleOffset.end(), 0);
    for (unsigned int p = 0; p < npes; ++p) {
        m_uiScatterMapElementRound1.insert(m_uiScatterMapElementRound1.end(),
                                           scatterMapSend_R1[p].begin(),
                                           scatterMapSend_R1[p].end());
        m_uiSendEleCount[p] = scatterMapSend_R1[p].size();
    }

    par::Mpi_Alltoall(m_uiSendEleCount.data(), m_uiRecvEleCount.data(), 1,
                      m_uiCommActive);
    m_uiSendEleOffset[0] = 0;
    m_uiRecvEleOffset[0] = 0;

    omp_par::scan(m_uiSendEleCount.data(), m_uiSendEleOffset.data(), npes);
    omp_par::scan(m_uiRecvEleCount.data(), m_uiRecvEleOffset.data(), npes);

    m_uiElementSendProcList.clear();
    m_uiElementRecvProcList.clear();

    for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
        if (m_uiSendEleCount[p] > 0) m_uiElementSendProcList.push_back(p);
        if (m_uiRecvEleCount[p] > 0) m_uiElementRecvProcList.push_back(p);
    }

    // END ELEMENT SCATTERMAP

    // Build m_uiGhostElementRound1Index: maps recv buffer positions
    // to element indices in m_uiAllElements for element ghost exchange.
    {
        // convert to int for MPI
        std::vector<int> sendEleCounts_i(npes), sendEleOffsets_i(npes);
        std::vector<int> recvEleCounts_i(npes), recvEleOffsets_i(npes);
        for (int p = 0; p < npes; p++) {
            sendEleCounts_i[p]  = static_cast<int>(m_uiSendEleCount[p]);
            sendEleOffsets_i[p] = static_cast<int>(m_uiSendEleOffset[p]);
            recvEleCounts_i[p]  = static_cast<int>(m_uiRecvEleCount[p]);
            recvEleOffsets_i[p] = static_cast<int>(m_uiRecvEleOffset[p]);
        }
        int totalEleSend = sendEleOffsets_i[npes - 1] + sendEleCounts_i[npes - 1];
        int totalEleRecv = recvEleOffsets_i[npes - 1] + recvEleCounts_i[npes - 1];

        // build send buffer: global IDs of elements we're sending
        std::vector<unsigned int> sendEleGlobalIDs(totalEleSend);
        for (int k = 0; k < totalEleSend; k++) {
            unsigned int localEleIdx = m_uiScatterMapElementRound1[k];
            sendEleGlobalIDs[k] =
                new_oct_connectivity_map[newLocalBegin + localEleIdx].eid;
        }

        // exchange: other ranks tell us which elements they're sending
        std::vector<unsigned int> recvEleGlobalIDs(totalEleRecv);
        MPI_Alltoallv(sendEleGlobalIDs.data(), sendEleCounts_i.data(),
                      sendEleOffsets_i.data(), MPI_UNSIGNED,
                      recvEleGlobalIDs.data(), recvEleCounts_i.data(),
                      recvEleOffsets_i.data(), MPI_UNSIGNED, commActive);

        // now map received global IDs to positions in m_uiAllElements
        m_uiGhostElementRound1Index.resize(totalEleRecv);
        for (int k = 0; k < totalEleRecv; k++) {
            auto it = globaltoNewLocal.find(recvEleGlobalIDs[k]);
            if (it != globaltoNewLocal.end()) {
                m_uiGhostElementRound1Index[k] = it->second;
            } else {
                m_uiGhostElementRound1Index[k] = 0;
                std::cerr << rank
                          << ": ERROR building GhostElementRound1Index: "
                             "couldn't find global ID "
                          << recvEleGlobalIDs[k] << std::endl;
            }
        }
    }

    // -----
    // SPLITTER NODES
    m_uiSplitterNodes = new ot::TreeNode[2 * npes];
    {
        ot::TreeNode minMaxLocal[2];
        if (m_uiElementLocalBegin < m_uiElementLocalEnd) {
            minMaxLocal[0] = m_uiAllElements[m_uiElementLocalBegin];
            minMaxLocal[1] = m_uiAllElements[m_uiElementLocalEnd - 1];
        }
        // else: default-constructed TreeNodes (rank has no elements)
        par::Mpi_Allgather(minMaxLocal, m_uiSplitterNodes, 2, commActive);
    }

    // SEND BUFFER FOR ELEMENTS
    {
        unsigned int eleBufSz =
            m_uiSendEleOffset[npes - 1] + m_uiSendEleCount[npes - 1];
        m_uiSendBufferElement.resize(eleBufSz);
    }

    // data we don't need to update:
    // m_uiSendKeyCount, m_uiSendKeyOffset, m_uiSendOct[Count/Offset]Round1/2
    // m_ui[Send/Recv]KeyDiag[Count/Offset], m_ui[Send/Recv]OctRound1Diag
    // m_uiRecvKey[Count/Offset], m_uiRecvOct[Count/Offset]Round1/2
    // m_uiGhostElementIDsToBe[Sent/Recv]
    // m_uiFElement*, m_uiMeshDomain_min/max, m_uiNumFakeNodes
    // m_ui[Pre/Post]GhostHangingNodeCGID
    // m_uiNpE, m_uiElementOrder, m_uiStensilSz, m_uiNumDirections, m_uiRefEl
    // m_uiF2EMap, m_ui[Send/Recv][Count/Offset]RePt, intergrid transfer
    // unzip map, unzip offset, unzip counts

    if (do_block_creation) {
        m_uiIsBlockSetup = false;
        m_uiLocalBlockList.clear();
        if (!rank) {
            std::cout << rank << ": Now preparing to set up blocks..."
                      << std::endl;
        }

        if (m_uiElementLocalBegin < m_uiElementLocalEnd) {
            // canonical block path: each element carries its source-rank
            // SFC block anchor + meta in m_uiBlockInfo (transported via
            // oct_data). bucket local elements by anchor → reconstruct
            // partial blocks. this preserves the SFC block decomposition
            // exactly, so one-sided FD stencils at block boundaries
            // sample the same padding regardless of partition.
            //
            // legacy path: octree2BlockDecompositionRepartitioned does a
            // local-only re-decomposition that depends on which subset
            // of elements landed on this rank — different partitions
            // produce different blocks, breaking long-haul bit-identity.
            const char* legacyEnv = std::getenv("DENDRO_USE_LEGACY_BLOCKS");
            const bool useLegacyBlocks = (legacyEnv && legacyEnv[0] == '1');

            size_t nProduced = 0;
            if (!useLegacyBlocks) {
                nProduced = buildBlocksFromCanonicalInfo();
                if (nProduced > 0 && !rank) {
                    std::cout << "[canonical-blocks] rank " << rank
                              << " reconstructed " << nProduced
                              << " blocks from transported block info"
                              << std::endl;
                }
            }
            if (nProduced == 0) {
                // fall back to legacy local re-decomposition (no
                // canonical info shipped, or env gate forces it)
                if (!rank) {
                    std::cout << "[canonical-blocks] rank " << rank
                              << " falling back to legacy local "
                                 "re-decomposition"
                              << std::endl;
                }
                octree2BlockDecompositionRepartitioned(
                    m_uiAllElements, m_uiLocalBlockList, m_uiMaxDepth,
                    m_uiDmin, m_uiDmax, m_uiElementLocalBegin,
                    m_uiElementLocalEnd, m_uiElementOrder, m_uiE2EMapping,
                    m_uiCoarsetBlkLev, NULL, 0);
            }

            performBlocksSetupRepartitioned(m_uiCoarsetBlkLev, NULL, 0);

            buildE2BlockMap();
            buildUnzipCanonicalWriterTable();
            buildZipPlan();

            // post-buildZipPlan E2N_CG dump for target TN slots. used to
            // detect whether buildZipPlan's cross-TN unify or any other
            // later phase re-routes slots that the audit had already
            // canonicalized. gate: DENDRO_E2N_POSTZIP_DIR=<dir>
            // + DENDRO_E2N_POSTZIP_TN="lev,x,y,z" + optional
            // DENDRO_E2N_POSTZIP_SUB="n1,n2,...". writes
            // <dir>/postzip_call<N>_r<R>.txt per repartition call.
            {
                static const char* pdir = DENDRO_PROBE_GETENV("DENDRO_E2N_POSTZIP_DIR");
                static const char* ptn_env =
                    DENDRO_PROBE_GETENV("DENDRO_E2N_POSTZIP_TN");
                static const char* psub_env =
                    DENDRO_PROBE_GETENV("DENDRO_E2N_POSTZIP_SUB");
                static int pz_call = 0;
                if (pdir && ptn_env) {
                    unsigned int tlev = 0, tx = 0, ty = 0, tz = 0;
                    std::sscanf(ptn_env, "%u,%u,%u,%u",
                                &tlev, &tx, &ty, &tz);
                    std::vector<unsigned int> tsubs;
                    if (psub_env) {
                        std::string ss(psub_env);
                        size_t pp = 0;
                        while (pp < ss.size()) {
                            unsigned int sn;
                            if (std::sscanf(ss.c_str() + pp, "%u", &sn) == 1)
                                tsubs.push_back(sn);
                            size_t nn = ss.find(',', pp);
                            if (nn == std::string::npos) break;
                            pp = nn + 1;
                        }
                    }
                    char fn[1024];
                    std::snprintf(fn, sizeof(fn),
                                  "%s/postzip_call%d_r%d.txt",
                                  pdir, pz_call, (int)m_uiActiveRank);
                    FILE* fp = std::fopen(fn, "w");
                    if (fp) {
                        std::fprintf(fp,
                            "# call=%d rank=%d TN=(lev%u,%u,%u,%u)\n",
                            pz_call, (int)m_uiActiveRank, tlev, tx, ty, tz);
                        const unsigned int npe   = m_uiNpE;
                        const unsigned int eOrd  = m_uiElementOrder;
                        const unsigned int LB    = m_uiElementLocalBegin;
                        const unsigned int LE    = m_uiElementLocalEnd;
                        const auto* pN_ = m_uiAllElements.data();
                        for (unsigned int e = 0;
                             e < m_uiAllElements.size(); e++) {
                            if (pN_[e].getLevel() != tlev) continue;
                            if (pN_[e].getX() != tx) continue;
                            if (pN_[e].getY() != ty) continue;
                            if (pN_[e].getZ() != tz) continue;
                            const bool eLoc = (e >= LB && e < LE);
                            for (unsigned int n = 0; n < npe; n++) {
                                if (!tsubs.empty()) {
                                    bool match = false;
                                    for (auto s : tsubs)
                                        if (s == n) { match = true; break; }
                                    if (!match) continue;
                                }
                                const unsigned int slot = e * npe + n;
                                const unsigned int cg =
                                    m_uiE2NMapping_CG[slot];
                                unsigned long long rx_ = 0, ry_ = 0, rz_ = 0;
                                if (cg < m_uiCG2DG.size()) {
                                    const unsigned int dg = m_uiCG2DG[cg];
                                    if (dg != LOOK_UP_TABLE_DEFAULT) {
                                        const unsigned int oe = dg / npe;
                                        const unsigned int on = dg % npe;
                                        if (oe < m_uiAllElements.size()) {
                                            const unsigned int olev =
                                                m_uiAllElements[oe].getLevel();
                                            if (olev <= m_uiMaxDepth) {
                                                const unsigned long long olen =
                                                    1ULL << (m_uiMaxDepth - olev);
                                                const unsigned int oni =
                                                    on % (eOrd + 1);
                                                const unsigned int onj =
                                                    (on / (eOrd + 1))
                                                    % (eOrd + 1);
                                                const unsigned int onk =
                                                    on / ((eOrd + 1)
                                                          * (eOrd + 1));
                                                rx_ = (unsigned long long)
                                                    m_uiAllElements[oe].getX()
                                                    * eOrd + (unsigned long long)
                                                    oni * olen;
                                                ry_ = (unsigned long long)
                                                    m_uiAllElements[oe].getY()
                                                    * eOrd + (unsigned long long)
                                                    onj * olen;
                                                rz_ = (unsigned long long)
                                                    m_uiAllElements[oe].getZ()
                                                    * eOrd + (unsigned long long)
                                                    onk * olen;
                                            }
                                        }
                                    }
                                }
                                std::fprintf(fp,
                                    "elem=%u %s sub=%u cg=%u "
                                    "resolved=(%llu,%llu,%llu)\n",
                                    e, eLoc ? "LOCAL" : "GHOST", n, cg,
                                    rx_, ry_, rz_);
                            }
                        }
                        std::fclose(fp);
                    }
                    pz_call++;
                }
            }
        }

        // Note: send/recv proc lists and buffer sizes were already
        // set up above (outside do_block_creation), no need to
        // duplicate here.

        __phase("block-decomposition");
    }
    if (!rank) {
        std::cout << rank << ": Now finished with the repartitioning scheme!"
                  << std::endl;
        std::cout << "[repart] TOTAL: "
                  << (MPI_Wtime() - __t_start) * 1e3 << " ms" << std::endl;
    }
}

template <typename T>
std::vector<oct_data<T>> Mesh::getOctDataFromOtherProcesses(
    std::vector<oct_data<T>> &oct_connectivity_map,
    const std::vector<T> &ele_offsets, const std::vector<T> &ele_counts,
    std::vector<T> &data_to_fetch, bool set_target_rank) {
    int rank            = this->getMPIRank();
    int npes            = this->getMPICommSize();
    MPI_Comm commActive = this->getMPICommunicator();

    // go through the partition to figure out what needs to be sent, since
    // that's the info we have
    std::vector<int> send_counts(npes, 0);
    std::vector<int> recv_counts(npes, 0);
    std::vector<int> send_offsets(npes, 0);
    std::vector<int> recv_offsets(npes, 0);

    std::vector<oct_data<T>> send_buffer;
    std::vector<oct_data<T>> data_keep;

    std::vector<unsigned long int> send_requests[npes];
    std::vector<unsigned long int> recv_requests[npes];

    for (const auto &ele_id : data_to_fetch) {
        if (ele_id < ele_offsets[rank] || ele_id >= ele_offsets[rank + 1]) {
            // find the original owner of this data
            for (size_t r_id = 0; r_id < npes; ++r_id) {
                if (ele_id >= ele_offsets[r_id] &&
                    ele_id < ele_offsets[r_id + 1]) {
                    recv_requests[r_id].push_back(ele_id);
                    send_counts[r_id]++;
                    break;
                }
            }
        }
    }

    send_offsets[0] = 0;
    for (int i = 1; i < npes; ++i) {
        send_offsets[i] = send_offsets[i - 1] + send_counts[i - 1];
    }
    int total_send_size = send_offsets[npes - 1] + send_counts[npes - 1];

    // then exchange the send counts to get "receive counts", which is how
    // many we'll actually need to send to each process
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 commActive);

    // calculate receive displacements
    recv_offsets[0] = 0;
    for (int i = 1; i < npes; ++i) {
        recv_offsets[i] = recv_offsets[i - 1] + recv_counts[i - 1];
    }
    int total_recv_size = recv_offsets[npes - 1] + recv_counts[npes - 1];

    // flattened buffers to get which elements need to be fetched
    std::vector<long unsigned int> flattened_send_buffer(total_send_size);
    std::vector<long unsigned int> flattened_recv_buffer(total_recv_size);

    int send_offset = 0;
    for (int i = 0; i < npes; ++i) {
        for (long unsigned int ele_id : recv_requests[i]) {
            flattened_send_buffer[send_offset++] = ele_id;
        }
    }

    // perform all-to-all-v which lets us know what we need
    MPI_Alltoallv(flattened_send_buffer.data(), send_counts.data(),
                  send_offsets.data(), MPI_UNSIGNED_LONG,
                  flattened_recv_buffer.data(), recv_counts.data(),
                  recv_offsets.data(), MPI_UNSIGNED_LONG, commActive);

    // flattened_recv_buffer is basically how we'll know what we need to
    // **SEND** from our original data structures, so recv from above means
    // "to send off"
    std::vector<oct_data<T>> flattened_send_full_data(total_recv_size);
    std::vector<oct_data<T>> flattened_recv_full_data(total_send_size);

    // build up the array of data that we need to send
    uint32_t counter = 0;
    for (uint32_t i = 0; i < npes; ++i) {
        for (uint32_t j = 0; j < recv_counts[i]; ++j) {
            T requested = flattened_recv_buffer[counter];
            requested -= ele_offsets[rank];

            // update the target rank, this will be used as the new home
            if (set_target_rank) oct_connectivity_map[requested].trank = i;

            flattened_send_full_data[counter] = oct_connectivity_map[requested];

            counter++;
        }
    }

    MPI_Datatype octdata_mpi_type = create_octdata_mpi_type<T>();

    // perform full communication, flattened_recv_full_data now has what was
    // requested
    MPI_Alltoallv(flattened_send_full_data.data(), recv_counts.data(),
                  recv_offsets.data(), octdata_mpi_type,
                  flattened_recv_full_data.data(), send_counts.data(),
                  send_offsets.data(), octdata_mpi_type, commActive);

    MPI_Type_free(&octdata_mpi_type);

    return flattened_recv_full_data;
}

}  // namespace ot
