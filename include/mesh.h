//
// Created by milinda on 9/2/16.

/**
 * @author: Milinda Shayamal Fernando
 *
 * School of Computing , University of Utah
 *
 * @breif Contains the functions to generate the mesh data structure from the
 * 2:1 balanced linear octree.
 * @details
 * Assumptions:
 * 1). Assumes that octree is balanced and sorted.
 * 2). Assumes that there is no duplicate nodes.
 * 3). Inorder to make the implementation much simpler we use the Morton
 * ordering to number the each vertex of a given particular element. Note that,
 * this doesn't cause any dependency between SFC curve used and the mesh
 * generation part.
 *
 * */

//

#pragma once

// comment this out if you want to run fastpart, having this defined will make
// it so it only outputs the .oct files
// #define _ONLY_DUMP_OCT_DATA_

#include <fdCoefficient.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <random>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "TreeNode.h"
#include "asyncExchangeContex.h"
#include "block.h"
#include "dendro.h"
#include "dendroProfileParams.h"  // only need to profile unzip_asyn for bssn. remove this header file later.
#include "key.h"
#include "logger.h"
#include "mpi.h"
#include "node.h"
#include "octUtils.h"
#include "parUtils.h"
#include "point.h"
#include "refel.h"
#include "sfcSearch.h"
#include "sfcSort.h"
#include "skey.h"
#include "stencil.h"
#include "testUtils.h"
#include "treenode2vtk.h"
#include "wavelet.h"

#if 1
#include "fastpart.h"
#endif

extern double t_e2e;  // e2e map generation time
extern double t_e2n;  // e2n map generation time
extern double t_sm;   // sm map generation time
extern double t_blk;

extern double t_e2e_g[3];
extern double t_e2n_g[3];
extern double t_sm_g[3];
extern double t_blk_g[3];

#define NUM_PTS_ELE 343

#include "dendroProfileParams.h"
#include "waveletRefEl.h"

// two small helpers to check if something is a vector, and they're the same
// (hopefully)
template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

struct E2EUpdateData {
    uint64_t sourceGlobalID;
    uint64_t targetGlobalID;
    int face;
};

// oct_data used for storing some information about the global map
template <typename T>
struct oct_data {
    uint32_t rank;
    uint32_t trank = UINT32_MAX;
    T eid;
    T localid;
    T coord[3];
    T e2e[6];
    T level;
    unsigned int flag;

    // information about the e2n maps based on global IDs, only need dg
    T e2n_dg[343];

    // a boolean that is used for identification in the scattermap regeneration
    bool isGhostTwo = false;

    T edgeNeighbors[12];
    T vertexNeighbors[8];

    // 27-bit ownership mask in the eOrd=2 representative layout. Bit r
    // ∈ [0, 27) is set iff this element OWNS the sub-node at
    // representative r = nk*9 + nj*3 + ni for ni,nj,nk ∈ {0,1,2}.
    // Computed pre-partition from a working cascade and travels with
    // the element through any partition exchange.
    uint32_t ownerMask = 0u;

    // canonical SFC block membership. populated pre-partition from the
    // source mesh's m_uiLocalBlockList and travels with the element
    // through partition exchange. on the receiver, elements that share
    // the same (blkAnchorX,Y,Z,Level) are grouped into a partial block
    // matching the source rank's canonical decomposition. blkMeta packs
    // bits 0-7 = regGridLev, 8-15 = rotID, 31 = valid.
    uint32_t blkAnchorX     = 0u;
    uint32_t blkAnchorY     = 0u;
    uint32_t blkAnchorZ     = 0u;
    uint32_t blkAnchorLevel = 0u;
    uint32_t blkMeta        = 0u;

    oct_data() {
        e2e[0] = LOOK_UP_TABLE_DEFAULT;
        e2e[1] = LOOK_UP_TABLE_DEFAULT;
        e2e[2] = LOOK_UP_TABLE_DEFAULT;
        e2e[3] = LOOK_UP_TABLE_DEFAULT;
        e2e[4] = LOOK_UP_TABLE_DEFAULT;
        e2e[5] = LOOK_UP_TABLE_DEFAULT;

        for (unsigned int i = 0; i < 12; ++i) {
            edgeNeighbors[i] = LOOK_UP_TABLE_DEFAULT;
        }
        for (unsigned int i = 0; i < 8; ++i) {
            vertexNeighbors[i] = LOOK_UP_TABLE_DEFAULT;
        }
    };

    oct_data(T eid, uint32_t rank, T coord[3], T e2e[6]) {
        this->eid      = eid;
        this->rank     = rank;

        this->coord[0] = coord[0];
        this->coord[1] = coord[1];
        this->coord[2] = coord[2];

        this->e2e[0]   = e2e[0];
        this->e2e[1]   = e2e[1];
        this->e2e[2]   = e2e[2];

        this->e2e[3]   = e2e[3];
        this->e2e[4]   = e2e[4];
        this->e2e[5]   = e2e[5];
    }
};

#if 0
typedef struct {
    fastpart_uint_t rank;
    fastpart_uint_t trank;
    fastpart_uint_t eid;
    fastpart_uint_t localid;
    fastpart_uint_t coord[3];
    fastpart_uint_t e2e[6];
    fastpart_uint_t level;
    fastpart_uint_t edgeNeighbors[12];
    fastpart_uint_t vertexNeighbors[8];
} oct_element_MINE;
#endif

template <typename T>
void print_octdata_vector(std::vector<oct_data<T>> &oct, int procrank = 0) {
    std::cout << ": OCT DATA:\n" << std::endl;
    for (auto &o : oct) {
        std::cout << procrank << " \tEID: " << std::setw(5) << std::right
                  << o.eid << " LID: " << std::setw(5) << std::right
                  << o.localid << " rank: " << std::setw(3) << std::right
                  << o.rank << " level: " << std::setw(2) << std::right
                  << o.level << " coord(" << std::setw(4) << std::right
                  << o.coord[0] << ", " << std::setw(4) << std::right
                  << o.coord[1] << ", " << std::setw(4) << std::right
                  << o.coord[2] << ") : " << "e2e(" << std::setw(10)
                  << std::right << o.e2e[0] << ", " << std::setw(10)
                  << std::right << o.e2e[1] << ", " << std::setw(10)
                  << std::right << o.e2e[2] << ", " << std::setw(10)
                  << std::right << o.e2e[3] << ", " << std::setw(10)
                  << std::right << o.e2e[4] << ", " << std::setw(10)
                  << std::right << o.e2e[5] << ") "
                  << "\t\tTARGET: " << std::setw(3) << std::right << o.trank

                  << " \te2n_cg[0:5]=";
        // for (int i = 0; i < 5; i++) {
        //     std::cout << o.e2n_cg[i] << " ";
        // }
        std::cout << "\te2n_dg[0:5]=";
        for (int i = 0; i < 5; i++) {
            std::cout << o.e2n_dg[i] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
struct node_data_ele {
    uint32_t rank;
    T ele_global_id;
#if 0
    std::unique_ptr<T[]> e2n_cg;
    std::unique_ptr<T[]> e2n_dg;
#else
    // T e2n_cg[343];
    T e2n_dg[343];
#endif
    uint32_t pts = 343;

    node_data_ele() {}

#if 0
    node_data_ele(uint32_t pts_in)
        : pts(pts_in), e2n_cg(new T[pts_in]), e2n_dg(new T[pts_in]) {}

    ~node_data_ele() = default;

    void set_pts(uint32_t pts_in) {
        if (pts == pts_in) {
            return;
        }

        pts = pts_in;

        std::unique_ptr<T[]> new_e2n_cg(new T[pts_in]);
        if (e2n_cg) {
            std::copy(e2n_cg.get(), e2n_cg.get() + std::min(pts, pts_in),
                      new_e2n_cg);
        }
        e2n_cg = std::move(new_e2n_cg);

        std::unique_ptr<T[]> new_e2n_dg(new T[pts_in]);
        if (e2n_dg) {
            std::copy(e2n_dg.get(), e2n_dg.get() + std::min(pts, pts_in),
                      new_e2n_dg);
        }
        e2n_cg = std::move(new_e2n_dg);
    }

    node_data_ele(node_data_ele &&other) noexcept
        : rank(other.rank),
          ele_global_id(other.ele_global_id),
          e2n_cg(std::move(other.e2n_dg)),
          e2n_dg(std::move(other.e2n_dg)),
          pts(other.pts) {}

    node_data_ele &operator=(node_data_ele &&other) noexcept {
        if (this != &other) {
            rank          = other.rank;
            ele_global_id = other.ele_global_id;
            e2n_cg        = std::move(other.e2n_cg);
            e2n_dg        = std::move(other.e2n_dg);
            pts           = other.pts;
        }
        return *this;
    }
#endif
};

template <typename T>
void print_node_data_vector(std::vector<node_data_ele<T>> &node_data,
                            int procrank = 0) {
    std::cout << ": ELEMENT NODE DATA:\n" << std::endl;
    for (auto &o : node_data) {
        std::cout << procrank << " \tEGID: " << o.ele_global_id
                  << " rank: " << o.rank << " npoints: " << o.pts
                  << " e2n_cg[0]=" << o.e2n_cg[0]
                  << " e2n_cg[end]=" << o.e2n_cg[o.pts - 1]
                  << " e2n_dg[0]=" << o.e2n_dg[0]
                  << " e2n_dg[end]=" << o.e2n_dg[o.pts - 1] << std::endl;
    }
}

template <typename T>
MPI_Datatype get_mpi_type() {
    if (std::is_same<T, int8_t>::value) return MPI_INT8_T;
    if (std::is_same<T, int16_t>::value) return MPI_INT16_T;
    if (std::is_same<T, int32_t>::value) return MPI_INT32_T;
    if (std::is_same<T, int64_t>::value) return MPI_INT64_T;
    if (std::is_same<T, uint8_t>::value) return MPI_UINT8_T;
    if (std::is_same<T, uint16_t>::value) return MPI_UINT16_T;
    if (std::is_same<T, uint32_t>::value) return MPI_UINT32_T;
    if (std::is_same<T, uint64_t>::value) return MPI_UINT64_T;
    if (std::is_same<T, int>::value) return MPI_INT;
    if (std::is_same<T, long>::value) return MPI_LONG;
    if (std::is_same<T, long long>::value) return MPI_LONG_LONG_INT;
    if (std::is_same<T, float>::value) return MPI_FLOAT;
    if (std::is_same<T, double>::value) return MPI_DOUBLE;
    throw std::runtime_error("Unsupported MPI type for typename T");
}

template <typename T>
MPI_Datatype create_octdata_mpi_type() {
    MPI_Datatype octdata_mpi_datatype;
    int blocklengths[] = {1, 1, 1, 1, 3, 6, 1, 1, NUM_PTS_ELE, 1, 12, 8,
                          1, 1, 1, 1, 1, 1};
    MPI_Aint offsets[18];
    MPI_Datatype types[18];

    types[0] = MPI_UINT32_T;  // 'rank' is always uint32_t
    types[1] = MPI_UINT32_T;  // 'target_rank' is always uint32_t
    types[2] = types[3] = types[4] = types[5] = types[6] = get_mpi_type<T>();
    types[7]                                             = MPI_UNSIGNED;
    types[8]                                             = get_mpi_type<T>();
    types[9]                                             = MPI_CXX_BOOL;
    types[10]                                            = get_mpi_type<T>();
    types[11]                                            = get_mpi_type<T>();
    types[12]                                            = MPI_UINT32_T;  // ownerMask
    types[13]                                            = MPI_UINT32_T;  // blkAnchorX
    types[14]                                            = MPI_UINT32_T;  // blkAnchorY
    types[15]                                            = MPI_UINT32_T;  // blkAnchorZ
    types[16]                                            = MPI_UINT32_T;  // blkAnchorLevel
    types[17]                                            = MPI_UINT32_T;  // blkMeta

    // get the offsets for each member of the struct
    offsets[0]  = offsetof(oct_data<T>, rank);
    offsets[1]  = offsetof(oct_data<T>, trank);
    offsets[2]  = offsetof(oct_data<T>, eid);
    offsets[3]  = offsetof(oct_data<T>, localid);
    offsets[4]  = offsetof(oct_data<T>, coord);
    offsets[5]  = offsetof(oct_data<T>, e2e);
    offsets[6]  = offsetof(oct_data<T>, level);
    offsets[7]  = offsetof(oct_data<T>, flag);
    offsets[8]  = offsetof(oct_data<T>, e2n_dg);
    offsets[9]  = offsetof(oct_data<T>, isGhostTwo);
    offsets[10] = offsetof(oct_data<T>, edgeNeighbors);
    offsets[11] = offsetof(oct_data<T>, vertexNeighbors);
    offsets[12] = offsetof(oct_data<T>, ownerMask);
    offsets[13] = offsetof(oct_data<T>, blkAnchorX);
    offsets[14] = offsetof(oct_data<T>, blkAnchorY);
    offsets[15] = offsetof(oct_data<T>, blkAnchorZ);
    offsets[16] = offsetof(oct_data<T>, blkAnchorLevel);
    offsets[17] = offsetof(oct_data<T>, blkMeta);

    MPI_Type_create_struct(18, blocklengths, offsets, types,
                           &octdata_mpi_datatype);
    MPI_Type_commit(&octdata_mpi_datatype);

    return octdata_mpi_datatype;
}

enum PartitioningOptions {
    NoPartition = 0,
    OriginalPartition,
    RandomPartition,
    fastpart
};

/**
 * How the oct flags are used in the mesh generation part.
 *
 * bit 0 - 4 : Used to store the level.
 * bit 5: 1- if the key was found in the tree 0- otherwise.
 * bit 6: Key type NONE
 * bit 7: Key type SPLITTER
 * bit 8: Key type UP
 * bit 9: Key type DOWN
 * bit 10: Key type FRONT
 * bit 11: Key type BACK
 * bit 12: Key type LEFT
 * bit 13: key type RIGHT
 *
 * */

enum KeyType { NONE, SPLITTER, UP, DOWN, FRONT, BACK, LEFT, RIGHT };

/**
 * @brief contains the level of neighbour.
 * COARSE: the neighbour octant is at lower level (coarser) than current level.
 * REFINE: the neighbour octant is  (refined) than current level.
 * SAME: the neighbour octant is at same  level than current level.
 * **/
enum NeighbourLevel { COARSE, SAME, REFINE };

// #define DEBUG_MESH_GENERATION

#define KS_MAX \
    31  // maximum number of points in any direction that wavelet DA supports.
#define KEY_DIR_OFFSET       7
#define CHAINED_GHOST_OFFSET 5u

#define OCT_NO_CHANGE 0u
#define OCT_SPLIT     1u
#define OCT_COARSE    2u

namespace ot {

/**@brief type of the scatter map, based on numerical computation method*/
enum SM_TYPE {
    FDM = 0,  // Finite Difference Method
    FEM_CG,   // Continous Galerkin  methods
    FEM_DG,   // Discontinous Galerkin methods
    E2E_ONLY  // only builds the e2e maps.
};

enum EType {
    INDEPENDENT,  // all the elemental nodal values,  are local.
    W_DEPENDENT,  // element is writable but has ghost nodal values as well.
                  // (note that the writable would be Independent U W_dependent)
    UNKWON
};

/**@brief ghost write modes. */
enum GWMode {
    OVERWRITE,   // over write the ghost write,
    ACCUMILATE,  // accumilate the ghost write

};

namespace WaveletDA {
enum LoopType { ALL, INDEPENDENT, DEPENDENT };
}

/**
 * @brief intergrid transfer mode.
 * We assume we have refinement and coarsen or no change during the intergird
 * transfer.
 *
 * INJECTION : when coarsening finer grid point injected to the coarser grids.
 * (this does not guranteed to preserve integrals across paren and child) P2CT :
 * when coarsening, use parent to child transpose (child nodes computed
 * contributed back to the parent node. ) This will preserve integrals iff, the
 * child node value coarser to the value interpolated from the parent.
 * CELLVEC_CPY : cell vec copy without any interpolation.
 */
enum INTERGRID_TRANSFER_MODE { INJECTION = 0, P2CT, CELLVEC_CPY };

/**
 * @brief vector types supported by the mesh class.
 * CG_NODAL  : Nodal CG vector.
 * DG_NODAL  : Nodal DG vector.
 * ELEMENTAL : Elemental Vector.
 */
enum VEC_TYPE { CG_NODAL, DG_NODAL, ELEMENTAL };

}  // namespace ot

namespace ot {
/** Structure to order the send recv nodes based on the (e,i,j,k ) ordering. */
struct NodeTuple {
   private:
    /** i index of the node*/
    unsigned int i;
    /** j index of the node*/
    unsigned int j;
    /** k index of the node*/
    unsigned int k;
    /** element id of the node*/
    unsigned int e;

   public:
    NodeTuple(unsigned int xi, unsigned int yj, unsigned int zk,
              unsigned int owner) {
        i = xi;
        j = yj;
        k = zk;
        e = owner;
    }
    inline unsigned int getX() const { return i; }
    inline unsigned int getY() const { return j; }
    inline unsigned int getZ() const { return k; }
    inline unsigned int getLevel() const { return e; }
};

}  // namespace ot

namespace ot {

/**
 * @breif Contains all the information needed to build neighbourhood information
 * for the balanced octree based on treeSearch.
 * */

class Mesh {
   private:
    /** Element to Element mapping data. Array size:
     * [m_uiAllNodes.size()*m_uiStensilSz*m_uiNumDirections];  But this is done
     * for m_uiStencilSz=1*/
    std::vector<unsigned int> m_uiE2EMapping;
    /** Element ot Node mapping data for continous Galerkin methods. Array size:
     * [m_uiAllNodes.size()*m_uiNpE];*/
    std::vector<unsigned int> m_uiE2NMapping_CG;
    /** Element to Node mapping with DG indexing after removing duplicates. This
     * is used for debugging. */
    std::vector<unsigned int> m_uiE2NMapping_DG;
    /** cg to dg mapping*/
    std::vector<unsigned int> m_uiCG2DG;
    /** dg to cg mapping*/
    std::vector<unsigned int> m_uiDG2CG;
    /** Local CG slots (sorted) where Pass A (in
     *  buildE2NWithSMRepartitioned) found no local element with
     *  E2N_DG self-owned and E2N_CG = cg. These are
     *  duplicate-local-ownership candidates for cross-rank
     *  reconciliation in reconcileCrossRankDuplicateCGs. */
    std::vector<unsigned int> m_uiPassACgsWithoutLocalCanonical;

    /** Local CG indices that Pass D (cross-rank consolidation in
     *  buildE2NWithSMRepartitioned) demoted on this rank because
     *  another (winning) rank claimed the same phys_pos. Pass E
     *  must NOT promote these back to local writers — leaving them
     *  unwritten is the correct outcome (their value flows from
     *  the winner via standard ghost exchange to a ghost cg that
     *  this rank's elements have been redirected to). */
    std::unordered_set<unsigned int> m_uiPassDDemotedLocalCgs;

    /** Map: demoted local cg → its corresponding ghost cg at the
     *  same phys_pos (sourced from the winner rank). After ghost
     *  exchange, performGhostExchange/syncDemotedLocalCgs uses
     *  this to mirror chi[ghost_cg] → chi[demoted_cg] so the
     *  demoted slot reflects the correct value (otherwise it would
     *  hold whatever stale data was last written, leading to
     *  partition-dependent VTU output and analytical-diff
     *  artifacts even though the actual evolution is correct). */
    std::unordered_map<unsigned int, unsigned int>
        m_uiPassDDemotedToGhostCg;

    /** Explicit zip plan: list of (target_cg, unzip_idx) pairs telling
     *  zip exactly which slot in the unzipped block buffer to read
     *  and which local cg to write. Replaces the implicit
     *  "E2N_DG self-owned scan" inside zip with an O(N) data-driven
     *  loop. Built by buildZipPlan() after the cascade and
     *  E2BlkMap are finalized; rebuilt on every remesh. */
    std::vector<unsigned int> m_uiZipPlanCg;
    std::vector<unsigned int> m_uiZipPlanUnzipIdx;

    /** Per-element 27-bit ownership mask (eOrd=2 representative
     *  layout). Bit r ∈ [0, 27) is set iff this element owns the
     *  sub-node at representative r = nk*9 + nj*3 + ni for ni,nj,nk
     *  in {0,1,2}. Computed once after buildE2NMap on the SFC mesh
     *  (where the cascade is canonically correct), then travels
     *  with the element through partition exchanges via oct_data.
     *
     *  At eOrd=2 this maps directly to all 27 sub-nodes. At higher
     *  eOrd, an actual sub n at (ni, nj, nk) maps to representative
     *  r where each axis is collapsed: 0 → 0 (boundary low), eOrd
     *  → 2 (boundary high), interior → 1. All higher-order
     *  sub-nodes in the same sharing category as a representative
     *  inherit ownership from that representative.
     *
     *  Sized to match m_uiAllElements (local + ghost). Indexed by
     *  the element's local index in m_uiAllElements. */
    std::vector<uint32_t> m_uiOwnerMask;

    /** Per-element canonical SFC block info. populated pre-partition
     *  from m_uiLocalBlockList (which is correct on the SFC mesh) and
     *  travels through partition exchange via oct_data. on the
     *  receiver, elements with the same anchor are grouped into a
     *  partial-block matching the source rank's canonical SFC
     *  decomposition — this is what makes block boundaries (and
     *  therefore one-sided FD stencils) partition-invariant.
     *
     *  meta packs: bits 0-7 regGridLev, 8-15 rotID, bit 31 valid.
     *
     *  Sized to match m_uiAllElements (local + ghost). */
   public:
    struct CanonicalBlockInfo {
        uint32_t anchorX     = 0u;
        uint32_t anchorY     = 0u;
        uint32_t anchorZ     = 0u;
        uint32_t anchorLevel = 0u;
        uint32_t meta        = 0u;
        bool isValid() const { return (meta & (1u << 31)) != 0u; }
        unsigned int regGridLev() const { return meta & 0xFFu; }
        unsigned int rotID() const { return (meta >> 8) & 0xFFu; }
    };
   private:
    std::vector<CanonicalBlockInfo> m_uiBlockInfo;

    /** Map: non-primary local cg → ghost cg on this rank whose
     *  scatter source is the primary's local cg on the primary's
     *  rank. Built by buildZipPlan() via an inverse-scatter-map
     *  lookup. Useful when the standard scatter map happens to have
     *  the right path; otherwise the m_uiZipSync* side-channel below
     *  delivers primary values directly. */
    std::unordered_map<unsigned int, unsigned int>
        m_uiZipNonPrimaryToGhostCg;

    /** Forward-direction post-sync fixup: ghost cgs at consensus
     *  phys_pos → a local cg holding the primary's value (on this
     *  rank). Built by buildZipPlan(): for each ghost cg whose
     *  cg2dg maps to a phys_pos that has a winner in `winners`, find
     *  a local cg at the same phys_pos and store ghost_cg → local_cg.
     *  After the standard ghost exchange + syncZipNonPrimary, the
     *  local cg holds the primary's post-axpy value (or got it via
     *  Alltoallv). Copying from local_cg into ghost_cg makes the
     *  ghost reflect the consensus value. Without this, ghost cgs at
     *  level-transition corners stay at the IC values (the standard
     *  scatter map's source for some ghost cgs doesn't reach the
     *  primary's local through the post-fastpart consolidation). */
    std::unordered_map<unsigned int, unsigned int>
        m_uiZipGhostToLocalAtConsensus;

    /** Side-channel non-primary sync via direct MPI_Alltoallv.
     *
     *  The standard ghost-exchange scatter maps were built before
     *  Pass A/D/E rewrote E2N_CG (rescue path adds new
     *  primary_local_cg references that weren't in the original
     *  scatter map). To deliver primary cg values to non-primary
     *  local cgs on other ranks, we maintain a separate "sync
     *  scatter map" that runs an Alltoallv after each standard
     *  ghost exchange.
     *
     *  Layout (analogous to m_uiSendNodeCount/Offset etc):
     *    Send: this rank's primary cgs that need to go to others.
     *      m_uiZipSyncSendCg[k] = local cg index to pack into
     *        send buffer at flat position k (for recipient rank
     *        determined by m_uiZipSyncSendOffsets).
     *    Recv: this rank's non-primary cgs that receive primary
     *      values.
     *      m_uiZipSyncRecvCg[k] = local cg index to write recv
     *        buffer value into.
     *
     *  Built each remesh by buildZipPlan() from the global primary
     *  picks. Each rank computes its own lists deterministically
     *  from the allgathered (phys_pos, rank, cg) claims. No
     *  additional handshake exchange needed. */
    std::vector<int> m_uiZipSyncSendCounts;
    std::vector<int> m_uiZipSyncSendOffsets;
    std::vector<int> m_uiZipSyncRecvCounts;
    std::vector<int> m_uiZipSyncRecvOffsets;
    std::vector<unsigned int> m_uiZipSyncSendCg;
    std::vector<unsigned int> m_uiZipSyncRecvCg;

    /** Intra-rank duplicate sync: for phys_pos with multiple LOCAL cgs
     *  on this rank, after the cross-rank Alltoallv copies the primary
     *  value into our recv targets, we still have duplicate LOCAL cgs
     *  on the SAME rank (e.g., two writer slots at one phys due to
     *  hanging-face neighborhood). The cross-rank sync skips entries
     *  where (r == p_rank), so the secondary local cg is never reached.
     *  m_uiZipLocalDupSrc[k] -> m_uiZipLocalDupDst[k]: copy src into
     *  dst on this rank after every cross-rank sync. */
    std::vector<unsigned int> m_uiZipLocalDupSrc;
    std::vector<unsigned int> m_uiZipLocalDupDst;

    // m_uiCanonicalOverride was an old experiment in routing zip
    // writes; superseded by the explicit zip plan
    // (m_uiZipPlanCg / m_uiZipPlanUnzipIdx) and removed.

    /** splitter element for each processor. */
    std::vector<ot::TreeNode>
        m_uiLocalSplitterElements;  // used to spit the keys to the correct
                                    // nodes.

    /**Splitter Node for each processor*/
    ot::TreeNode *m_uiSplitterNodes;

    // Pre  and Post ghost octants.

    /** pre ghost elements (will be freed after use)*/
    std::vector<ot::TreeNode> m_uiPreGhostOctants;
    /**post ghost elements (will be freed after use)*/
    std::vector<ot::TreeNode> m_uiPostGhostOctants;

    // Keys Related attributes
    /**search keys generated for local elements. */
    std::vector<Key> m_uiKeys;

    /**search keys generated for ghost elements*/
    std::vector<Key> m_uiGhostKeys;

    /**search keys generated for missing diagonal elements in the ghost*/
    std::vector<Key> m_uiKeysDiag;

    /**input to the mesh generation (will be freed after use)*/
    std::vector<ot::TreeNode> m_uiEmbeddedOctree;

    /** Ghost Elements. (will be freed after use)*/
    std::vector<ot::TreeNode> m_uiGhostOctants;

    /** store the indices of round 1 communication.
     *
     * Note: !! You cannot build a m_uiGhostElementRound2Index, because at the
     * round 2 exchange of ghost elements we might need to send a round1 ghost
     * element to another processor.
     *
     * */
    std::vector<unsigned int> m_uiGhostElementRound1Index;

    /** stores all the pre + local + post ELEMENTS. */
    std::vector<ot::TreeNode> m_uiAllElements;
    /** Spatial hash (x,y,z,level) -> local idx into m_uiAllElements,
     *  used by findContainingElementInAllNodes to avoid O(N) linear
     *  scans during block-neighbor discovery on repartitioned meshes.
     *  Built lazily when needed and cleared whenever m_uiAllElements
     *  changes. */
    std::unordered_map<uint64_t, unsigned int> m_uiAllElementsSpatialHash;
    /** stores the local nodes */
    std::vector<ot::TreeNode> m_uiAllLocalNode;

    /** block list for the current local elements in order to apply the stencil.
     */
    std::vector<ot::Block> m_uiLocalBlockList;

    /**minimum depth of the octree*/
    unsigned int m_uiDmin;
    /**maximum depth of the octree*/
    unsigned int m_uiDmax;

    /**begin of the pre ghost elements*/
    unsigned int m_uiElementPreGhostBegin = 0;  // Not mandatory to store this.
    /** end of pre ghost elements*/
    unsigned int m_uiElementPreGhostEnd;
    /** begin of locat octants (INDEPENDENT)*/
    unsigned int m_uiElementLocalBegin;
    /** end of the local octants (INDEPENDET)*/
    unsigned int m_uiElementLocalEnd;
    /**begin location for the post ghost octants*/
    unsigned int m_uiElementPostGhostBegin;
    /** end location for the post ghost octants*/
    unsigned int m_uiElementPostGhostEnd;
    /** explicit list of LOCAL element indices into m_uiAllElements. when
     * empty, the iterator falls back to enumerating [LocalBegin, LocalEnd)
     * in order (the historical layout assumption). when populated (by
     * Phase 2's SFC-sort experiment), the iterator follows this list,
     * which may interleave with ghost positions in m_uiAllElements. */
    std::vector<unsigned int> m_uiLocalElementIds;
    /**begin of the pre ghost elements of  fake elements*/
    unsigned int m_uiFElementPreGhostBegin;
    /**end of the pre ghost elements of fake elements. */
    unsigned int m_uiFElementPreGhostEnd;
    /**begin of the local fake elements*/
    unsigned int m_uiFElementLocalBegin;
    /**end of the local fake elements*/
    unsigned int m_uiFElementLocalEnd;
    /**begin of the fake post ghost elements*/
    unsigned int m_uiFElementPostGhostBegin;
    /**end of the fake post ghost elements*/
    unsigned int m_uiFElementPostGhostEnd;

    /**denotes the mesh domain min*/
    unsigned int m_uiMeshDomain_min;
    /**denotes the mesh domain max*/
    unsigned int m_uiMeshDomain_max;

    /**Number of local elements. */
    unsigned int m_uiNumLocalElements;
    /**Number of pre ghost elements*/
    unsigned int m_uiNumPreGhostElements;
    /**Number of post ghost elements*/
    unsigned int m_uiNumPostGhostElements;

    /*** Total number of actual elements in the mesh (preG+ local + postG ) */
    unsigned int m_uiNumTotalElements;

    /** Total number of actual nodes. Defn: Actual nodes are the ones that are
     * owned by actual elememts. */
    unsigned int m_uiNumActualNodes;

    /** unzipped vector size. */
    DendroIntL m_uiUnZippedVecSz;

    /** Total number of fake nodes. Defn: these nodes are owned my the fake
     * elements. */
    unsigned int m_uiNumFakeNodes;

    /** begin location  of the pre ghost nodes in CG indexing*/
    unsigned int m_uiNodePreGhostBegin;
    /** end location of the pre ghost nodes in CG indexing*/
    unsigned int m_uiNodePreGhostEnd;
    /** begin location of the local nodes in CG indexing*/
    unsigned int m_uiNodeLocalBegin;
    /** end location of the local nodes in CG indexing*/
    unsigned int m_uiNodeLocalEnd;
    /** begin location of the post ghost nodes in CG indexing*/
    unsigned int m_uiNodePostGhostBegin;
    /** end location of the post ghost nodes in CG indexing*/
    unsigned int m_uiNodePostGhostEnd;

    // Stuff for communication and synchronization ...
    //-------------------------------------------------------------------------

    /**Rank of the current process*/
    int m_uiActiveRank;
    /** size of the active  Comm */
    int m_uiActiveNpes;

    /** MPI active communicator. (this is a subcomm of the m_uiCommGlobal) */
    MPI_Comm m_uiCommActive;

    /** MPI global communicator*/
    MPI_Comm m_uiCommGlobal;

    /**Rank of the current process*/
    int m_uiGlobalRank;
    /** size of the active  Comm */
    int m_uiGlobalNpes;

    /** indicates whether the current mesh is active under the m_uiCommActive*/
    bool m_uiIsActive;

    /**Number of keys should be sent to each processor. (Note that instead of
     * keys we are sending the owners of keys to the required processor)*/
    unsigned int *m_uiSendKeyCount;
    /**Number of keys received by each processor*/
    unsigned int *m_uiRecvKeyCount;
    /** send key offset array*/
    unsigned int *m_uiSendKeyOffset;
    /**receive key offset array*/
    unsigned int *m_uiRecvKeyOffset;

    /**send diagonal key count*/
    unsigned int *m_uiSendKeyDiagCount;
    /**recv diagonal key count*/
    unsigned int *m_uiRecvKeyDiagCount;
    /** send key diagonal offset*/
    unsigned int *m_uiSendKeyDiagOffset;
    /** recv key diagonal offset*/
    unsigned int *m_uiRecvKeyDiagOffset;

    /**SendOct counts related to round 1 of ghost exchange. */
    /**Number of ghost octants(elements) that needed to be sent to each
     * processor */
    unsigned int *m_uiSendOctCountRound1;
    /**Number of ghost elemets recieved from each processor. */
    unsigned int *m_uiRecvOctCountRound1;
    /**Ghost send count offset, used in all2allv*/
    unsigned int *m_uiSendOctOffsetRound1;
    /**Ghost recieve count offset, used in all2allv*/
    unsigned int *m_uiRecvOctOffsetRound1;

    unsigned int *m_uiSendOctCountRound1Diag;
    /**Number of ghost elemets recieved from each processor. */
    unsigned int *m_uiRecvOctCountRound1Diag;
    /**Ghost send count offset, used in all2allv*/
    unsigned int *m_uiSendOctOffsetRound1Diag;
    /**Ghost recieve count offset, used in all2allv*/
    unsigned int *m_uiRecvOctOffsetRound1Diag;

    /**SendOct counts related to round 2 of ghost exchange. */
    /**Number of ghost octants(elements) that needed to be sent to each
     * processor */
    unsigned int *m_uiSendOctCountRound2;
    /**Number of ghost elemets recieved from each processor. */
    unsigned int *m_uiRecvOctCountRound2;
    /**Ghost send count offset, used in all2allv*/
    unsigned int *m_uiSendOctOffsetRound2;
    /**Ghost recieve count offset, used in all2allv*/
    unsigned int *m_uiRecvOctOffsetRound2;

    /**Number of nodes that needed to be sent to each processor*/
    std::vector<unsigned int> m_uiSendNodeCount;
    /**Number of nodes that recieved from each processor*/
    std::vector<unsigned int> m_uiRecvNodeCount;
    /**Send node count offset*/
    std::vector<unsigned int> m_uiSendNodeOffset;
    /**Recv node count offset*/
    std::vector<unsigned int> m_uiRecvNodeOffset;

    /**@brief : number of elements that needed to be sent to each processor*/
    std::vector<unsigned int> m_uiSendEleCount;
    /**@brief : number of elements that recieved from each processor*/
    std::vector<unsigned int> m_uiRecvEleCount;

    /**@brief : Send element count offset*/
    std::vector<unsigned int> m_uiSendEleOffset;

    /**@brief : Recv element count offset*/
    std::vector<unsigned int> m_uiRecvEleOffset;

    /**Send processor list for ghost exchange (nodal ghost exchange)*/
    std::vector<unsigned int> m_uiSendProcList;

    /**recv processor list for the ghost exchange (nodal ghost exchange)*/
    std::vector<unsigned int> m_uiRecvProcList;

    /**@brief: Send proc list for elemental (cell) ghost sync*/
    std::vector<unsigned int> m_uiElementSendProcList;

    /**@brief: Send proc list for elemental (cell) ghost sync*/
    std::vector<unsigned int> m_uiElementRecvProcList;

    /** local element ids that needs to be send for other processors. */
    std::vector<unsigned int> m_uiGhostElementIDsToBeSent;
    /** local element ids that recieved from other processors after ghost
     * element exchange*/
    std::vector<unsigned int> m_uiGhostElementIDsToBeRecv;

    /**stores the CG indecies of pre ghost nodes which are hanging, */
    std::vector<unsigned int> m_uiPreGhostHangingNodeCGID;

    /**stores the CG indices of post ghost nodes which are hanging.*/
    std::vector<unsigned int> m_uiPostGhostHangingNodeCGID;

    /**Send buffer of ghost elements (octants)*/
    std::vector<ot::TreeNode> m_uiSendBufferElement;

    /**Actual exchange of node values. This is the actual exhange that happens
     * in element looping*/
    std::vector<double> m_uiSendBufferNodes;
    /**recv buffer for the ghost node values receiveing from others. */
    std::vector<double> m_uiRecvBufferNodes;

    /**Scatter map for the elements. Keeps track of which local elements need to
     * be sent to which processor. (This has to be derived with m_uiSendOctCount
     * values) Element ID's that is been exchanged at round 1 of ghost exchange.
     * Property: For a given processor rank p the scatter map elements that is
     * set to processor p is sorted and unique.
     * */
    std::vector<unsigned int> m_uiScatterMapElementRound1;

    /**Scatter map for the actual nodes. Keeps track of which local node needs
     * to be sent to which processor. */
    std::vector<unsigned int> m_uiScatterMapActualNodeSend;

    /** Scatter map for the actual nodes, recieving from other processors. */
    std::vector<unsigned int> m_uiScatterMapActualNodeRecv;

    /** Repartitioned meshes only: for R2 ghost boundary nodes, the owner
     * candidate set is incomplete on the receiver (R2's 26-neighborhood
     * isn't ghost-fetched), so the receiver can't agree with the home rank
     * on the canonical CG index. We instead deliver the value by routing
     * (R2_gid, sub) to R2's home rank and reading from a per-local-element
     * nodal buffer populated directly by createVector(vec, func). The
     * tag below marks the send-side scatter-map entries that read from
     * m_uiLocalNodalDG instead of vec. */
    std::vector<unsigned char> m_uiScatterMapSendIsDG;

    /** Per-local-element nodal values at each element's own sub physical
     * position. Size = numLocalElements * m_uiNpE. Populated by
     * createVector(vec, func); used by performGhostExchange to deliver
     * correct values to R2 ghost boundary slots. Not maintained for
     * vectors produced by arithmetic — only freshly created via func.
     * mutable so it can be populated by const createVector overloads. */
    mutable std::vector<double> m_uiLocalNodalDG;

   public:
    /** Mutable accessor to the per-local-element DG buffer. External
     * code (e.g., partition redistribution helpers) needs to refresh
     * this when a vec is moved across partitions outside of
     * createVector(vec, func). */
    inline std::vector<double>& getLocalNodalDGRef() {
        return m_uiLocalNodalDG;
    }

    /** Populate m_uiLocalNodalDG from a CG vector, for use after external
     * code writes CG values directly (e.g., initial-condition loops that
     * iterate local elements and write into vec via E2N_CG) without
     * going through createVector(vec, func). Required before
     * performGhostExchange on graph-partitioned meshes where some R2
     * boundary sends route through the DG path and read from
     * m_uiLocalNodalDG. Idempotent. */
    template <typename T>
    void syncLocalNodalDGFromCG(const T* vec);

    /** Fix "orphan" local CG slots — slots in the local CG range that
     * are NOT referenced by any LOCAL element's E2N_CG on this rank,
     * so the normal zip / RK update path never writes to them. Such
     * slots can arise in graph-partitioned meshes where
     * buildE2NMap/buildE2NWithSMRepartitioned allocates local-range
     * CGs at positions only referenced by ghost elements.
     *
     * This pulls the current value from a rank that DOES have a
     * LOCAL element at the same physical position (where RK DID
     * update that rank's ghost-cg slot). Call BEFORE
     * performGhostExchange so the gathered values are then broadcast
     * correctly; or after an RK step to sync orphan slots with RK
     * progress made elsewhere.
     *
     * No-op if the mesh has zero orphans (expected for SFC
     * partitions). */
    template <typename T>
    void orphanPreGather(T* vec);

   protected:

    // variables to manage loop access over elements.
    /**counter for the current element*/
    unsigned int m_uiEL_i;

    /**order of the octant elements. */
    unsigned int m_uiElementOrder;

    /**Number of nodes per element*/
    unsigned int m_uiNpE;

    /**Number of neighbours that need to be searched in each direction. */
    unsigned int m_uiStensilSz;

    /**number of total directions that the search is performed*/
    unsigned int m_uiNumDirections;

    /**Reference element to perform interpolations. */
    RefElement m_uiRefEl;

    /**@brief: ghost elements needed for FEM computation*/
    std::vector<unsigned int> m_uiFEMGhostLev1IDs;

    //===== maps needed for DG computations.

    /**@brief: stores the face to element map. */
    std::vector<unsigned int> m_uiF2EMap;

    /**@brief: indicate octree to block decomposition performed or not*/
    bool m_uiIsBlockSetup;

    /** type of the build scatter map.*/
    SM_TYPE m_uiScatterMapType;

    /**@brief indicates whether the f2e map has build or not*/
    bool m_uiIsF2ESetup;

    /**@brief: async comunication context to support async ghost exchange*/
    std::vector<AsyncExchangeContex> m_uiMPIContexts;

    /**@brief: communicator tag used for async communication*/
    unsigned int m_uiCommTag = 0;

    /**bool vector for elementy ID, of size m_uiAllElements*/
    std::vector<bool> m_uiIsNodalMapValid;

    // --
    // Note : These are special data stored to search the 3rd point in Dendro-GR
    // unzip with 4th order.
    //

    /**@brief: missing unzip keys*/
    std::vector<ot::Key> m_uiUnzip_3pt_keys;

    std::vector<ot::Key> m_uiUnzip_3pt_ele;

    std::vector<ot::Key> m_uiUnzip_3pt_recv_keys;

    /**@brief: send node count for req. keys*/
    std::vector<unsigned int> m_uiSendCountRePt;

    /**@brief: send node offset for req. keys*/
    std::vector<unsigned int> m_uiSendOffsetRePt;

    /**@brief: recv node count for req. keys*/
    std::vector<unsigned int> m_uiRecvCountRePt;

    /**@brief: recv node offset for req. keys*/
    std::vector<unsigned int> m_uiRecvOffsetRePt;

    /**@brief: req pts send proc list*/
    std::vector<unsigned int> m_uiReqSendProcList;

    /**@brief: req pts recv proc list*/
    std::vector<unsigned int> m_uiReqRecvProcList;

    /**@brief: send node req pt SM*/
    std::vector<unsigned int> m_uiSendNodeReqPtSM;

    /**@brief: element to block map */
    std::vector<unsigned int> m_uiE2BlkMap;

    /**@brief: coarset block level allowed. (this is used in perform block set
     * up) */
    unsigned int m_uiCoarsetBlkLev = OCT2BLK_COARSEST_LEV;

    /**@brief: domain min point*/
    Point m_uiDMinPt               = Point(0, 0, 0);

    /**@brief: domain max point. */
    Point m_uiDMaxPt =
        Point((1u << m_uiMaxDepth), (1u << m_uiMaxDepth), (1u << m_uiMaxDepth));

    /**@brief: send counts in elements, for inter-grid transfer*/
    std::vector<unsigned int> m_uiIGTSendC;

    /**@brief: recv counts in elements, for inter-grid transfer*/
    std::vector<unsigned int> m_uiIGTRecvC;

    /**@brief: send offsets in elements, for inter-grid transfer*/
    std::vector<unsigned int> m_uiIGTSendOfst;

    /**@brief: recv offsets in elements, for inter-grid transfer*/
    std::vector<unsigned int> m_uiIGTRecvOfst;

    /**@brief: Mesh 2 partitioned with M1 splitters (Needed for IGT)*/
    std::vector<ot::TreeNode> m_uiM2Prime;

    /**@brief: true if IGT data strucutures are setup. false otherwise*/
    bool m_uiIsIGTSetup = false;

    /**@brief element to block map for element wise scatter for unzip operation.
     */
    std::vector<unsigned int> m_e2b_unzip_map;

    /**@brief element to block map offset, look at the  buildE2BlockMap function
     * to see how this is used. */
    std::vector<unsigned int> m_e2b_unzip_offset;

    /**@brief element to block map count, if ele has no dependence then count
     * will be zero. */
    std::vector<unsigned int> m_e2b_unzip_counts;

    /**@brief Canonical writer table for unzip multi-writer block-padding
     * slots. Indexed by absolute unzipped-buffer offset (block.getOffset() +
     * kkz*lx*ly + jjy*lx + iix). Value = element index of canonical writer
     * (in m_uiAllElements). LOOK_UP_TABLE_DEFAULT means "no override"; every
     * writer is allowed to write (single-writer slots, or no contention).
     * Built once per mesh by buildUnzipCanonicalWriterTable. Used to make
     * unzip output partition-invariant by enforcing the same canonical
     * writer's value at every multi-writer slot regardless of which rank
     * (or which partition) is doing the unzip. */
    std::vector<unsigned int> m_uiUnzipCanonWriter;
    bool m_uiUnzipCanonWriterBuilt = false;

    PartitioningOptions m_partitionOption = PartitioningOptions::fastpart;

   private:
    /**@brief build E2N map for FEM computation*/
    void buildFEM_E2N();

    /**
     * @author Milinda Fernando
     * @brief generates search key elements for local elements.
     * */

    void generateSearchKeys();

    /**
     * @author Milinda Fernando
     * @brief generates search keys for ghost elements, inorder to build the E2E
     * mapping between ghost elements.
     * */

    void generateGhostElementSearchKeys();

    /**
     * @brief generates diagonal keys for ghost if it is not already in the
     * current processor.
     * */

    void generateBdyElementDiagonalSearchKeys();

    /**
     * @author Milinda Fernando
     * @brief Builds the E2E mapping, in the mesh
     * @param [in] in: 2:1 balanced octree (assumes that the input is 2:1
     * balanced unique and sorted)
     * @param [in] k_s : Stencil size, how many elements to search in each
     * direction
     * @param [in] comm: MPI Communicator.
     * */
    void buildE2EMap(std::vector<ot::TreeNode> &in, MPI_Comm comm);

    void computeElementOwnerRanks(std::vector<unsigned int> &elementOwner);

    /**
     * @author Milinda Fernando
     * @brief Builds the E2E mapping, in the mesh (sequential case. No ghost
     * nodes. )
     * @param [in] in: 2:1 balanced octree (assumes that the input is 2:1
     * balanced unique and sorted)
     * @param [in] k_s : Stencil size, how many elements to search in each
     * direction
     * */
    void buildE2EMap(std::vector<ot::TreeNode> &in);

    /**
     * @brief
     *
     */
    void buildE2NWithSM();

    /**
     *
     * @author Milinda Fernando
     * @brief Builds the Element to Node (E2N) mapping to enforce the continuity
     * of the solution. (Needed in continous Galerkin methods. ) This function
     * assumes that E2E mapping is already built.
     *
     * */

    void buildE2NMap();

    /**
     * @brief: Expands the order-2 E2N to the specified element order,
     * rebuilds CG/DG mappings, node ranges, and scatter maps.
     * @param eleOrder: the target element order
     */
    void buildE2NWithSMRepartitioned(unsigned int eleOrder);

    /**
     * @brief: Builds the Element to nodal mapping for DG computations.
     * (No hanging node consideration)
     */
    void buildE2N_DG();

    /**@brief : elements contribution to the unzip block representation.
     * for i in e2b_count[e]:
     *    b=e2block_map[offset[e] + i]
     *    block b has some unzip nodes coming from elemental nodes of e.
     */
    void buildE2BlockMap();

    /**@brief Build the per-slot canonical writer table for unzip multi-writer
     * block-padding positions. Replays the unzip_scatter scatter logic in a
     * dry mode (no writes) to enumerate all (block, slot) → list-of-writers
     * triples. For each multi-writer slot, picks ONE canonical writer using a
     * partition-invariant rule: prefer the writer that the global canonical
     * iteration order (canon_mode=2: level desc, x asc, y asc, z asc) would
     * iterate LAST. Stores per-slot canonical winner in m_uiUnzipCanonWriter.
     *
     * Must run AFTER buildE2BlockMap (which populates m_e2b_unzip_map) and
     * AFTER m_uiLocalBlockList is finalized.
     *
     * Cost: one full unzip-scatter dry pass + O(slots) selection. Modest
     * one-time setup. */
    void buildUnzipCanonicalWriterTable();

    /**@brief Build the explicit zip plan (m_uiZipPlanCg /
     * m_uiZipPlanUnzipIdx). Replaces the implicit "E2N_DG self-owned
     * scan" inside zip with an O(N) data-driven loop over the plan.
     * Must be called after buildE2NWithSM and buildE2BlockMap (the
     * cascade and E2BlkMap must be finalized). Rebuilt on every
     * mesh change. */
    void buildZipPlan();

    /**@brief Derive the per-element 27-bit ownership mask
     * (m_uiOwnerMask) from the current cascade output. For each
     * element e, walk the 27 representative sub-nodes (eOrd=2
     * layout) and set bit r iff E2N_DG[e*npe+sub_for_r]/npe == e.
     *
     * Must be called AFTER buildE2NMap completes the cascade
     * (i.e., after the SFC mesh is built where the cascade is
     * canonically correct). The result is a compact, transportable
     * representation of "this element owns these sub-nodes" that
     * can ride along oct_data through partition exchanges.
     *
     * Cost: O(27) per element. Cheap. */
    void deriveOwnerMasksFromCascade();

    /**@brief Compare m_uiOwnerMask (transported from pre-partition
     * cascade) with what the current cascade output says, per
     * element + per representative. Logs disagreements and returns
     * the count. Useful for diagnosing where post-partition cascade
     * disagrees with the canonical pre-partition decision.
     *
     * @return number of (elem, representative) disagreements on this rank. */
    size_t validateOwnerMasksAgainstCurrentCascade() const;

    /**@brief Patch the post-partition cascade output using
     * authoritative ownership info from m_uiOwnerMask. For each
     * local (elem, sub) where cascade marks elem self-owned but the
     * mask disagrees ("over-claim"), find the true owner via E2EMap
     * neighbor walk and mask check. Redirect E2N_CG / E2N_DG to
     * point at the true owner's slot, and stash the now-orphan local
     * cg into m_uiPassDDemotedToGhostCg so the existing sync helper
     * mirrors values correctly post-exchange.
     *
     * Must run AFTER buildE2NWithSMRepartitioned and AFTER masks are
     * populated (either from cascade derivation on initial SFC build
     * or from oct_data transport on repartition).
     *
     * @return number of slots patched. */
    size_t patchE2NCgFromMasks();

    /**@brief Audit E2N_CG entries and repair wrong-cascade routings using
     *  phys_pos comparison. For each local element (e, sub), checks whether
     *  E2N_CG[e*npe+sub]'s actual phys_pos matches the expected phys_pos for
     *  that sub. Distinguishes legitimate hanging-face/edge p2c interpolation
     *  (cg's owner element is at COARSER level) from same-level cascade bugs.
     *  Repairs same-level mismatches by routing to the local cg at the
     *  expected phys_pos (which is guaranteed present given the R3 ghost
     *  layer).
     *
     *  Must run AFTER buildE2NWithSMRepartitioned + Pass A/D/E + mask-patch,
     *  and BEFORE REBUILD NODAL SCATTER MAPS / buildZipPlan so the corrected
     *  E2N_CG is reflected in scatter maps and zip-plan.
     *
     *  Honors PassD's intentional demotions (skips cgs in
     *  m_uiPassDDemotedLocalCgs and m_uiPassDDemotedToGhostCg).
     *
     *  @return number of E2N_CG entries patched. */
    size_t auditAndRepairE2NCgPhysPos();

    /**@brief structural canonicalization of hanging-face/edge/corner
     *  routing in E2N_CG/DG. For any (e, n) where the order=eOrder
     *  routing decodes to an owner whose level != e's level (hanging
     *  case), redirect to the Morton-tree parent's slot at the SAME
     *  sub-index n. Uses a TN→local_idx hash map over m_uiAllElements
     *  for fast lookup — no per-cg phys-pos hash needed.
     *
     *  When the Morton-tree parent is on this rank (local or ghost),
     *  the override is direct and partition-invariant. When the
     *  parent is not on this rank, the slot is left untouched and
     *  the slower phys-pos audit acts as a fallback.
     *
     *  Should be called BEFORE auditAndRepairE2NCgPhysPos to reduce
     *  the audit's work to a fast verification pass.
     *
     *  @return number of E2N_CG/DG entries canonicalized. */
    size_t canonicalizeHangingFaceRoutingTN();

    /**@brief derive m_uiBlockInfo from the current m_uiLocalBlockList
     *  on the SFC mesh. for each block, walks its (contiguous)
     *  element range and stamps each element with the block's
     *  anchor TreeNode + regGridLev + rotID. ghost elements get
     *  zero-valued (invalid) entries. must run after performBlocksSetup
     *  on the SFC mesh — this is where the block decomposition is
     *  canonically correct. */
    void deriveBlockInfoFromBlocks();

    /**@brief reconstruct m_uiLocalBlockList from per-element
     *  m_uiBlockInfo (transported from pre-partition SFC mesh).
     *  buckets local elements by (anchor, anchorLevel) and emits
     *  one ot::Block per bucket using the non-SFC constructor.
     *  blocks may be partial (fewer elements than the source-rank
     *  block had) — this is fine because each rank only computes
     *  stencils on its local elements; padding comes from ghost
     *  exchange.
     *
     *  @return number of blocks produced; 0 if no valid block info
     *  was transported (caller can fall back to legacy path). */
    size_t buildBlocksFromCanonicalInfo();

    /**@brief Side-channel non-primary sync. Runs an Alltoallv that
     * delivers primary cg values to non-primary local cg slots on
     * other ranks, using the lists built by buildZipPlan(). Called
     * after each ghost exchange. */
    template <typename T>
    void syncZipNonPrimary(T* vec, unsigned int dof);

    /**@brief Position-keyed cross-rank broadcast for vec values.
     * For every phys position with cgs on multiple ranks (incl ghost-only
     * positions), picks ONE canonical value via global allgather and
     * broadcasts it to every cg at that position globally. Stronger than
     * syncZipNonPrimary because it does not rely on the cascade winner pick
     * (which can leave dangling/forgotten ghosts un-synced when no rank has
     * a LOCAL writer at a phys position). Canonical rule: lowest rank
     * with a LOCAL cg wins; among lowest-rank locals, lowest cg index.
     * If no rank has a LOCAL cg, lowest rank with any (ghost) cg wins. */
    template <typename T>
    void broadcastCgValuesByPhysPos(T* vec, unsigned int dof);


    /**
     * @author Milinda Fernando
     * @brief inorder to make the E2N consistent across all processors we have
     * exchanged additional layer of ghost elements. This procedure remove those
     * additional elements( layer 2 ghost) from global octant array and update
     * E2E and E2N accordingly.
     *
     * */
    void shrinkE2EAndE2N();

    /**
     *
     * @author: Milinda Fernando.
     * @def Fake Elements(Local) are teh fake elements where all it's nodal lies
     * in the nodal local and nodal L1 ghost layer.
     * @brief: Generates local fake elements and update E2E and E2N mapping.
     * (This is not used. )
     * */

    void generateFakeElementsAndUpdateE2EAndE2N();

    /**
     * @author  Milinda Fernando
     * @brief Computes the overlapping nodes between given two elements. Note
     * that idx ,idy, idz should be integer array size of (m_uiElementOrder +
     * 1).
     * @param [in] Parent : denotes the larger(ancestor) element of the two
     * considering elements.
     * @param [in] child : denotes the desendent from the parent element.
     * @param [out] idx: mapping between x nodes indecies of parent element to x
     * node indecies for the child element.
     * @param [out] idy: mapping between y nodes indecies of parent element to y
     * node indecies for the child element.
     * @param [out] idz: mapping between z nodes indecies of parent element to z
     * node indecies for the child element.
     * @example idx[0]= 2 implies that the parent element x index 0 is given
     * from the child x index 2.
     *
     * */
    inline bool computeOveralppingNodes(const ot::TreeNode &parent,
                                        const ot::TreeNode &child, int *idx,
                                        int *idy, int *idz);

    /**
     *  @breif Computes the ScatterMap and the send node counts for ghost node
     * exchage. We do the scatter map exchange in two different steps. Hence we
     * need to compute 2 different scatter maps.
     *      1. Scatter map for actual nodes.
     *      2. Scatter map for fake nodes.
     *  @param[in] MPI_Communicator for scatter map node exchange.
     *  @note: depreciated remove this method later.
     *
     */
    void computeNodeScatterMaps(MPI_Comm comm);

    /**
     *  @breif Computes the ScatterMap and the send node counts for ghost node
     * exchage.
     *  @param[in] comm : MPI communicator.
     * */
    void computeNodalScatterMap(MPI_Comm comm);

    /**
     *  @breif Computes the ScatterMap and the send node counts for ghost node
     * exchage. (uses the compression of the all to all data exchange to reduce
     * the total number of nodes get exchanged. )
     *  @param[in] comm : MPI communicator.
     *
     * */
    void computeNodalScatterMap1(MPI_Comm comm);

    /**
     *@breif Computes the ScatterMap and the send node counts for ghost node
     *exchage. (uses the compression of the all to all data exchange to reduce
     *the total number of nodes get exchanged. )
     *@param[in] comm : MPI communicator.
     *
     **/
    void computeNodalScatterMap2(MPI_Comm comm);

    /**
     * @breif Computes the ScatterMap and the send node counts for ghost node
     *exchage. (uses the compression of the all to all data exchange to reduce
     *the total number of nodes get exchanged. )
     * @param[in] comm : MPI communicator. scattermap2 with compression.
     *
     **/
    void computeNodalScatterMap3(MPI_Comm comm);

    /**
     *@breif Computes the ScatterMap and the send node counts for ghost node
     *exchage. (uses the compression of the all to all data exchange to reduce
     *the total number of nodes get exchanged. )
     *@param[in] comm : MPI communicator. scattermap2 with compression.
     *
     **/
    void computeNodalScatterMap4(MPI_Comm comm);

    void computeNodalScatterMapDG(MPI_Comm comm);

    /**
     * childf1 and childf2 denotes the edge that happens ar the intersection of
     * those two planes.
     * */

    /*
     * NOTE: These private functions does not check for m_uiIsActive
     * sicne they are called in buildE2E and buildE2N mappings.
     *
     * */

    /**
     * @breif This function is to map internal edge nodes in the LEFT face.
     * @param[in] child element in cordieration for mapping.
     * @param[in] parent or the LEFT neighbour of child
     * @param[in] parentChildLevEqual specifies whether the parent child levels
     * are equal (true) if their equal.
     * @param[in] edgeChildIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * @param[in] edgeOwnerIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * */
    inline void OCT_DIR_LEFT_INTERNAL_EDGE_MAP(
        unsigned int child, unsigned int parent, bool parentChildLevEqual,
        std::vector<unsigned int> &edgeChildIndex,
        std::vector<unsigned int> &edgeOwnerIndex);
    /**
     * @breif This function is to map internal edge nodes in the RIGHT face.
     * @param[in] child element in cordieration for mapping.
     * @param[in] parent or the RIGHT neighbour of child
     * @param[in] parentChildLevEqual specifies whether the parent child levels
     * are equal (true) if their equal.
     * @param[in] edgeChildIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * @param[in] edgeOwnerIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * */
    inline void OCT_DIR_RIGHT_INTERNAL_EDGE_MAP(
        unsigned int child, unsigned int parent, bool parentChildLevEqual,
        std::vector<unsigned int> &edgeChildIndex,
        std::vector<unsigned int> &edgeOwnerIndex);
    /**
     * @breif This function is to map internal edge nodes in the DOWN face.
     * @param[in] child element in cordieration for mapping.
     * @param[in] parent or the DOWN neighbour of child
     * @param[in] parentChildLevEqual specifies whether the parent child levels
     * are equal (true) if their equal.
     * @param[in] edgeChildIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * @param[in] edgeOwnerIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * */
    inline void OCT_DIR_DOWN_INTERNAL_EDGE_MAP(
        unsigned int child, unsigned int parent, bool parentChildLevEqual,
        std::vector<unsigned int> &edgeChildIndex,
        std::vector<unsigned int> &edgeOwnerIndex);
    /**
     * @breif This function is to map internal edge nodes in the UP face.
     * @param[in] child element in cordieration for mapping.
     * @param[in] parent or the UP neighbour of child
     * @param[in] parentChildLevEqual specifies whether the parent child levels
     * are equal (true) if their equal.
     * @param[in] edgeChildIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * @param[in] edgeOwnerIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * */
    inline void OCT_DIR_UP_INTERNAL_EDGE_MAP(
        unsigned int child, unsigned int parent, bool parentChildLevEqual,
        std::vector<unsigned int> &edgeChildIndex,
        std::vector<unsigned int> &edgeOwnerIndex);
    /**
     * @breif This function is to map internal edge nodes in the FRONT face.
     * @param[in] child element in cordieration for mapping.
     * @param[in] parent or the FRONT neighbour of child
     * @param[in] parentChildLevEqual specifies whether the parent child levels
     * are equal (true) if their equal.
     * @param[in] edgeChildIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * @param[in] edgeOwnerIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * */
    inline void OCT_DIR_FRONT_INTERNAL_EDGE_MAP(
        unsigned int child, unsigned int parent, bool parentChildLevEqual,
        std::vector<unsigned int> &edgeChildIndex,
        std::vector<unsigned int> &edgeOwnerIndex);
    /**
     * @breif This function is to map internal edge nodes in the BACK face.
     * @param[in] child element in cordieration for mapping.
     * @param[in] parent or the BACK neighbour of child
     * @param[in] parentChildLevEqual specifies whether the parent child levels
     * are equal (true) if their equal.
     * @param[in] edgeChildIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * @param[in] edgeOwnerIndex std::vector (pre-allocated) for the size of the
     * internal nodes of the edge.
     * */
    inline void OCT_DIR_BACK_INTERNAL_EDGE_MAP(
        unsigned int child, unsigned int parent, bool parentChildLevEqual,
        std::vector<unsigned int> &edgeChildIndex,
        std::vector<unsigned int> &edgeOwnerIndex);

    /**
     * @brief: Computes the FACE to element map for each face direction.
     * @param[in] ele: element ID
     * @param[in] dir: primary face direction
     * @param[in] dirOp: opposite face direction for dir
     * @param[in] dir1: direction 1 (dir 1 and dir2 used to )
     * @param[in] dir2: direction 2
     * */
    void SET_FACE_TO_ELEMENT_MAP(unsigned int ele, unsigned int dir,
                                 unsigned int dirOp, unsigned int dir1,
                                 unsigned int dir2);

    /**
     * @brief maps the corner nodes of an elements to a  corresponsing elements.
     * @param[in] child element id in corsideration.
     *  */
    inline void CORNER_NODE_MAP(unsigned int child);

    /**
     * @brief Returns the diagonal element ID if considering two faces are
     * simillar in size of elementID.
     * */
    inline void OCT_DIR_DIAGONAL_E2E(unsigned int elementID, unsigned int face1,
                                     unsigned int face2,
                                     unsigned int &lookUp) const;

    /**
     * @brief Search the given set of keys in given nodes and update the search
     * result of the keys. The speciality of this method from SFC_TreeSearch()
     * is that this won't change the ordering of the Keys. Hence this is
     * expensive than the SFC_TreeSearch. So use it only when the ordering of
     * the keys matters.
     *
     * @param [in] pKeys: Keys needed to be searched.
     * @param [in] pNodes: Nodes (represents the space where keys being
     * searched)
     * */

    template <typename pKey, typename pNode>
    void searchKeys(std::vector<pKey> &pKeys, std::vector<pNode> &pNodes);

    /**
     * @brief Returns the direction of the node when i, j, k index of the node
     * given.
     * */
    inline unsigned int getDIROfANode(unsigned int ii_x, unsigned int jj_y,
                                      unsigned int kk_z);

    /**
     * @brief decompose the direction to it's ijk values.
     * */

    inline void directionToIJK(unsigned int direction,
                               std::vector<unsigned int> &ii_x,
                               std::vector<unsigned int> &jj_y,
                               std::vector<unsigned int> &kk_z);

    /**@brief: Performs block padding along the diagonal direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void blockDiagonalUnZip(const ot::Block &blk, const T *zippedVec,
                            T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the vertex direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void blockVertexUnZip(const ot::Block &blk, const T *zippedVec,
                          T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_DOWN direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_DOWN_Unzip(const ot::Block &blk, const T *zippedVec,
                                 T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_UP direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_UP_Unzip(const ot::Block &blk, const T *zippedVec,
                               T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_BACK direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                                 T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_FRONT direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_FRONT_Unzip(const ot::Block &blk, const T *zippedVec,
                                  T *unzippedVec, T *eleDGVec,
                                  bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_DOWN direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_DOWN_Unzip(const ot::Block &blk, const T *zippedVec,
                                  T *unzippedVec, T *eleDGVec,
                                  bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_UP direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_UP_Unzip(const ot::Block &blk, const T *zippedVec,
                                T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_BACK direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                                  T *unzippedVec, T *eleDGVec,
                                  bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_FRONT direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_FRONT_Unzip(const ot::Block &blk, const T *zippedVec,
                                   T *unzippedVec, T *eleDGVec,
                                   bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_UP_BACK direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_UP_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                               T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_UP_FRONT direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_UP_FRONT_Unzip(const ot::Block &blk, const T *zippedVec,
                                T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_DOWN_BACK direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_DOWN_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                                 T *unzippedVec, T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_DOWN_FRONT direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_DOWN_FRONT_Unzip(const ot::Block &blk, const T *zippedVec,
                                  T *unzippedVec, T *eleDGVec,
                                  bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_DOWN_BACK
     * direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_DOWN_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                                      T *unzippedVec, T *eleDGVec,
                                      bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_DOWN_BACK
     * direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_DOWN_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                                       T *unzippedVec, T *eleDGVec,
                                       bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_UP_BACK direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_UP_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                                    T *unzippedVec, T *eleDGVec,
                                    bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_UP_BACK direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_UP_BACK_Unzip(const ot::Block &blk, const T *zippedVec,
                                     T *unzippedVec, T *eleDGVec,
                                     bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_DOWN_FRONT
     * direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_DOWN_FRONT_Unzip(const ot::Block &blk, const T *zippedVec,
                                       T *unzippedVec, T *eleDGVec,
                                       bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_DOWN_FRONT
     * direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_DOWN_FRONT_Unzip(const ot::Block &blk,
                                        const T *zippedVec, T *unzippedVec,
                                        T *eleDGVec, bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_LEFT_UP_FRONT direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_LEFT_UP_FRONT_Unzip(const ot::Block &blk, const T *zippedVec,
                                     T *unzippedVec, T *eleDGVec,
                                     bool *eleDGValid);

    /**@brief: Performs block padding along the OCT_DIR_RIGHT_UP_FRONT
     * direction.
     * @param[in] blk: block to perform diagonal direction padding.
     * @param[in] zippedVec: zipped vector
     * @param[out] unzippedVec: updated unzip vec.
     * */
    template <typename T>
    void OCT_DIR_RIGHT_UP_FRONT_Unzip(const ot::Block &blk, const T *zippedVec,
                                      T *unzippedVec, T *eleDGVec,
                                      bool *eleDGValid);

    /***
     * @brief: computes the block boundary parent containing elements of two
     *levels of refinement.
     * @param [in] zipVec: zip vector (input vector)
     * @param [out] out: parent nodal values.
     * @param [in] lookUp: coarser elemet ID.
     * @param [in] fid: finer child numbers.
     * @param [in] cid: coarser child numbers
     * @param [in] child: blk input children.
     **/
    template <typename T>
    void getBlkBoundaryParentNodes(const T *zipVec, T *out, T *w1, T *w2,
                                   unsigned int lookUp, const unsigned int *fid,
                                   const unsigned int *cid,
                                   const unsigned int *child);

#if 0
    //---Note: These functions are specifically written for find missing 3rd block unzip points for the GR application

    /**
     * @brief: Compute the scatter maps for the 3rd point interpolation this is written only for GR application. 
     * Note that when performing communication send counts and recv counts should be interchanged. 
     */
    void computeSMSpecialPts();

    /**
     * @brief performs the 3rd pt interpolation based on the scatter mapped built
     * @tparam T type of the input and output vectors. 
     * @param in : input vector
     */
    template <typename T>
    void readSpecialPtsBegin(const T *in);

    template <typename T>
    void readSpecialPtsEnd(const T *in, T* out);
#endif

    // --- 3rd point exchange function end.

   public:
    /**@brief Public access to per-element owner-rank computation.
     * Returns elementOwner[e] = active-rank that owns element e in
     * m_uiAllElements (LOCAL on this rank → m_uiActiveRank; GHOST on
     * other rank → that rank). Used by diagnostic probes. */
    inline void computeElementOwnerRanksPublic(
        std::vector<unsigned int>& elementOwner) {
        computeElementOwnerRanks(elementOwner);
    }

    /**@brief Public access to the side-channel non-primary sync. Used
     * by ETS callers that need to harden the post-zip CG state before
     * stage-vector axpy reads it (axpy is local-only and doesn't sync).
     * Mirrors primary cg values into non-primary local cg slots at
     * multi-claim phys_pos. */
    template <typename T>
    inline void syncZipNonPrimaryPublic(T* vec, unsigned int dof) {
        this->syncZipNonPrimary(vec, dof);
    }

    /**@brief Public wrapper for broadcastCgValuesByPhysPos. */
    template <typename T>
    inline void broadcastCgValuesByPhysPosPublic(T* vec, unsigned int dof) {
        this->broadcastCgValuesByPhysPos(vec, dof);
    }

    /**@brief Read-only accessor for the per-element ownership masks. */
    inline const std::vector<uint32_t>& getOwnerMask() const {
        return m_uiOwnerMask;
    }

    /**@brief Assign external ownership masks (by move). Used to inject
     * pre-computed masks from a source mesh (where cascade is correct)
     * into a fresh repartition target before its cascade runs.
     * Caller is responsible for matching the size to m_uiAllElements. */
    inline void setOwnerMask(std::vector<uint32_t>&& mask) {
        m_uiOwnerMask = std::move(mask);
    }

    /**@brief Read-only accessor for canonical block info. */
    inline const std::vector<CanonicalBlockInfo>& getBlockInfo() const {
        return m_uiBlockInfo;
    }

    /**@brief Inject canonical block info (by move). Used to seed a
     * fresh repartition target with the source SFC mesh's block
     * decomposition before its post-partition block setup runs. */
    inline void setBlockInfo(std::vector<CanonicalBlockInfo>&& info) {
        m_uiBlockInfo = std::move(info);
    }

    /**@brief parallel mesh constructor
     * @param[in] in: complete sorted 2:1 balanced octree to generate mesh
     * @param[in] k_s: how many neighbours to check on each direction (used =1)
     * @param[in] pOrder: order of an element.
     * @param[in] commActive: MPI active communicator
     * @param[in] comm: MPI communicator (global)
     * */
    Mesh(std::vector<ot::TreeNode> &in, unsigned int k_s, unsigned int pOrder,
         unsigned int activeNpes, MPI_Comm comm, bool pBlockSetup = true,
         SM_TYPE smType       = SM_TYPE::FDM,
         unsigned int grainSz = DENDRO_DEFAULT_GRAIN_SZ,
         double ld_tol        = DENDRO_DEFAULT_LB_TOL,
         unsigned int sf_k    = DENDRO_DEFAULT_SF_K);

    /**@brief parallel mesh constructor
     * @param[in] in: complete sorted 2:1 balanced octree to generate mesh
     * @param[in] k_s: how many neighbours to check on each direction (used =1)
     * @param[in] pOrder: order of an element.
     * @param[in] comm: MPI communicator (global)
     * @param[in] grainSz: prefered grain sz. (this parameter is used to perform
     * automatic comm expansion and shrinking)
     * @param[in] ld_tol: load imbalance tolerance for comm expansion and
     * shrinking
     * @param[in] sf_k: splitter fix _k value. (Needed by SFC_partitioinng for
     * large p>=10,000)
     * */
    Mesh(std::vector<ot::TreeNode> &in, unsigned int k_s, unsigned int pOrder,
         MPI_Comm comm, bool pBlockSetup = true, SM_TYPE smType = SM_TYPE::FDM,
         unsigned int grainSz = DENDRO_DEFAULT_GRAIN_SZ,
         double ld_tol        = DENDRO_DEFAULT_LB_TOL,
         unsigned int sf_k    = DENDRO_DEFAULT_SF_K,
         unsigned int (*getWeight)(const ot::TreeNode *) = NULL,
         unsigned int *blk_tags = NULL, unsigned int blk_tags_sz = 0);

    /**@brief destructor for mesh (releases the allocated variables in the
     * class. )*/
    ~Mesh();

    /**
     * @brief Perform the blocks initialization so that we can apply the stencil
     * for the grid as a sequnce of finite number of regular grids. note that
     * this should be called after performing all E2N and E2N mapping.
     * */
    void performBlocksSetup(unsigned int cLev, unsigned int *tag,
                            unsigned int tsz);

    void performBlocksSetupRepartitioned(unsigned int cLev, unsigned int *tag,
                                         unsigned int tsz);

    void findBlockNeighborsWithoutSFC(ot::Block &blk);

    bool findContainingElementInAllNodes(const ot::TreeNode &searchNode,
                                         unsigned int &result);

    /**
     * @brief computes the face to element map.
     * needs to be called after e2e and e2n maps has built.
     **/
    void buildF2EMap();

    /*** @brief: perform block dependancy flags flag block unzip depends on
     * ghost nodes.  */
    void flagBlockGhostDependancies();

    /**
     * @brief ground-truth audit of the UNZIP_INDEPENDENT/DEPENDENT block
     * classification. poisons every ghost CG node with NaN, unzips once, and
     * checks which blocks catch a NaN in their unzip output (i.e. genuinely
     * read ghost data). any UNZIP_INDEPENDENT block that does is a
     * misclassification (unsafe for compute/comm overlap). returns the global
     * count of misclassified independent blocks (0 == correct). exercised by
     * TEST 11 in testPartitioning. requires a fully-built mesh with working
     * unzip (do NOT call mid-construction — it does not perform ghost
     * exchange, only poisons ghost slots to detect reads). */
    DendroIntL auditBlockTypeIndependence(const char *site = "");

    /**@brief: returns if the block setup has performed or not*/
    inline bool isBlockSetep() { return m_uiIsBlockSetup; }

    /**@brief: returns if the scatter map typed set*/
    inline SM_TYPE getScatterMapType() { return m_uiScatterMapType; }
    inline void setScatterMapType(SM_TYPE t) { m_uiScatterMapType = t; }

    /** Mutable access to the local-element TreeNode at internal index `ele`.
     * Needed by repartition helpers that carry refinement flags across
     * a partition change (flags live on the TreeNode itself). Safer
     * alternative to exposing m_uiAllElements wholesale. */
    inline ot::TreeNode& getMutableElement(unsigned int ele) {
        return m_uiAllElements[ele];
    }

    // Setters and getters.
    /** @breif Returns the number of local elements in the grid. (local to the
     * current considering processor)*/
    inline unsigned int getNumLocalMeshElements() const {
        return m_uiNumLocalElements;
    }
    /** @breif Returns the number of pre-ghost elements */
    inline unsigned int getNumPreGhostElements() const {
        return m_uiNumPreGhostElements;
    }
    /** @brief Returns the number of post-ghost elements */
    inline unsigned int getNumPostGhostElements() const {
        return m_uiNumPostGhostElements;
    }
    // get nodal information.

    /** @brief Returns the number of post-ghost elements */
    inline unsigned int getNumActualNodes() const { return m_uiNumActualNodes; }

    /**@brief return the number of nodes local to the mesh*/
    inline unsigned int getNumLocalMeshNodes() const {
        return (m_uiNodeLocalEnd - m_uiNodeLocalBegin);
    }
    /**@brief return the number of pre ghost mesh nodes*/
    inline unsigned int getNumPreMeshNodes() const {
        return (m_uiNodePreGhostEnd - m_uiNodePreGhostBegin);
    }
    /**@brief return the number of post ghost mesh nodes*/
    inline unsigned int getNumPostMeshNodes() const {
        return (m_uiNodePostGhostEnd - m_uiNodePostGhostBegin);
    }
    /**@brief return the begin location of element pre ghost*/
    inline unsigned int getElementPreGhostBegin() const {
        return m_uiElementPreGhostBegin;
    }
    /**@brief return the end location of element pre ghost*/
    inline unsigned int getElementPreGhostEnd() const {
        return m_uiElementPreGhostEnd;
    }
    /**@brief return the begin location of element local*/
    inline unsigned int getElementLocalBegin() const {
        return m_uiElementLocalBegin;
    }
    /**@brief return the end location of element local*/
    inline unsigned int getElementLocalEnd() const {
        return m_uiElementLocalEnd;
    }
    /**@brief return the begin location of element post ghost */
    inline unsigned int getElementPostGhostBegin() const {
        return m_uiElementPostGhostBegin;
    }
    /**@brief return the end location of element post ghost*/
    inline unsigned int getElementPostGhostEnd() const {
        return m_uiElementPostGhostEnd;
    }

    /**@brief opaque iterator-yielding range over LOCAL element indices.
     *
     * usage: `for (unsigned int e : mesh->localElements()) { ... }`
     *
     * default backing: indices [getElementLocalBegin(), getElementLocalEnd())
     * in order. once m_uiLocalElementIds is populated (Phase 2; gated by
     * DENDRO_SFC_SORT_ALL_ELEMENTS=1 after repartition) the range iterates
     * those IDs instead — which may interleave with ghost positions in
     * m_uiAllElements. consumers using this API don't need to know which
     * backing applies; consumers using the raw [LB, LE) range bake in the
     * contiguity assumption.
     */
    class LocalElementRange {
        const Mesh* m_mesh_;
    public:
        class iterator {
            const Mesh*  m_;
            std::size_t  pos_;
        public:
            iterator(const Mesh* m, std::size_t p) : m_(m), pos_(p) {}
            unsigned int operator*() const;
            iterator& operator++() {
                ++pos_;
                return *this;
            }
            bool operator!=(const iterator& o) const {
                return pos_ != o.pos_;
            }
            bool operator==(const iterator& o) const {
                return pos_ == o.pos_;
            }
        };
        explicit LocalElementRange(const Mesh* m) : m_mesh_(m) {}
        iterator begin() const;
        iterator end() const;
        std::size_t size() const;
    };
    inline LocalElementRange localElements() const {
        return LocalElementRange(this);
    }

    /**@brief return the begin location of pre ghost nodes*/
    inline unsigned int getNodePreGhostBegin() const {
        return m_uiNodePreGhostBegin;
    }
    /**@brief return the end location of pre ghost nodes*/
    inline unsigned int getNodePreGhostEnd() const {
        return m_uiNodePreGhostEnd;
    }
    /**@brief return the location of local node begin*/
    inline unsigned int getNodeLocalBegin() const { return m_uiNodeLocalBegin; }
    /**@brief return the location of local node end*/
    inline unsigned int getNodeLocalEnd() const { return m_uiNodeLocalEnd; }
    /**@brief return the location of post node begin*/
    inline unsigned int getNodePostGhostBegin() const {
        return m_uiNodePostGhostBegin;
    }
    /**@brief return the location of post node end*/
    inline unsigned int getNodePostGhostEnd() const {
        return m_uiNodePostGhostEnd;
    }

    /**@brief returns the dof for a partition (grid points) */
    inline unsigned int getDegOfFreedom() const { return m_uiNumActualNodes; }

    /**@brief returns the dof for a partition (grid points) */
    inline unsigned int getDegOfFreedomUnZip() const {
        return m_uiUnZippedVecSz;
    }

    inline unsigned int getNumTotalELements() const {
        return m_uiNumTotalElements;
    }

    /**@brief returns the dof DG for a partition. (grid points) */
    inline unsigned int getDegOfFreedomDG() const {
        return m_uiNumTotalElements * m_uiNpE;
    }

    /**@brief returns the pointer to All elements array. */
    inline const std::vector<ot::TreeNode> &getAllElements() const {
        return m_uiAllElements;
    }

    /**@brief returns the Level 1 ghost element indices. */
    inline const std::vector<unsigned int> &getLevel1GhostElementIndices()
        const {
        return m_uiGhostElementRound1Index;
    }

    /**@brief returns the splitter elements of the mesh local elements. */
    inline const std::vector<ot::TreeNode> &getSplitterElements() const {
        return m_uiLocalSplitterElements;
    }

    /**@brief returns all local nodes(vertices)*/
    inline const std::vector<ot::TreeNode> &getAllLocalNodes() const {
        return m_uiAllLocalNode;
    }

    /**@brief returns const e2e mapping instance. */
    inline const std::vector<unsigned int> &getE2EMapping() const {
        return m_uiE2EMapping;
    }

    /**@breif returns const e2n mapping instance */
    inline const std::vector<unsigned int> &getE2NMapping() const {
        return m_uiE2NMapping_CG;
    }

    /**@brief returns dg to cg map*/
    inline const std::vector<unsigned int> &getDG2CGMap() const {
        return m_uiDG2CG;
    }

    /**@brief returns cg to dg map*/
    inline const std::vector<unsigned int> &getCG2DGMap() const {
        return m_uiCG2DG;
    }

    /**@breif returns const e2n mapping instance (debuging purposes only) */
    inline const std::vector<unsigned int> &getE2NMapping_DG() const {
        return m_uiE2NMapping_DG;
    }
    /**@brief returns const list of local blocks (regular grids) for the
     * consdering mesh. */
    inline const std::vector<ot::Block> &getLocalBlockList() const {
        return m_uiLocalBlockList;
    }

    /**@brief element-to-block map: for local element e, returns the block
     * id this element belongs to. indexed by (e - elementLocalBegin). */
    inline const std::vector<unsigned int> &getE2BlkMap() const {
        return m_uiE2BlkMap;
    }

    /**@biref get element to block unzip map. */
    inline const std::vector<unsigned int> &getE2BUnzipMap() const {
        return m_e2b_unzip_map;
    }

    /**@biref get element to block unzip map counts. */
    inline const std::vector<unsigned int> &getE2BUnzipMapCounts() const {
        return m_e2b_unzip_counts;
    }

    /**@biref get element to block unzip map offsets. */
    inline const std::vector<unsigned int> &getE2BUnzipMapOffsets() const {
        return m_e2b_unzip_offset;
    }

    /**@brief return the number of directions in the E2E mapping. */
    inline unsigned int getNumDirections() const { return m_uiNumDirections; }

    /**@brief return the number of nodes per element.*/
    inline unsigned int getNumNodesPerElement() const { return m_uiNpE; }

    /**@brief returns the order of an element*/
    inline unsigned int getElementOrder() const { return m_uiElementOrder; }

    /**@brief returns the communicator (acitve) */
    inline MPI_Comm getMPICommunicator() const { return m_uiCommActive; }

    /**@brief returns the global communicator*/
    inline MPI_Comm getMPIGlobalCommunicator() const { return m_uiCommGlobal; }

    /**@brief returns the rank w.r.t. global comm*/
    inline unsigned int getMPIRankGlobal() const { return m_uiGlobalRank; }

    /** @brief returns the comm size w.r.t. global comm*/
    inline unsigned int getMPICommSizeGlobal() const { return m_uiGlobalNpes; }

    /** @brief returns the rank */
    inline unsigned int getMPIRank() const { return m_uiActiveRank; }

    /** @brief returns the comm size: */
    inline unsigned int getMPICommSize() const { return m_uiActiveNpes; }

    /** @brief returns const pointer to reference element */
    inline const RefElement *getReferenceElement() const { return &m_uiRefEl; }

    /**@brief returns the send proc list size*/
    inline unsigned int getSendProcListSize() const {
        return m_uiSendProcList.size();
    }

    /**@brief returns the recv proc list size*/
    inline unsigned int getRecvProcListSize() const {
        return m_uiRecvProcList.size();
    }

    /**@brief returns the nodal send counts*/
    inline const std::vector<unsigned int> &getNodalSendCounts() const {
        return m_uiSendNodeCount;
    }

    /**@brief returns the nodal send offsets*/
    inline const std::vector<unsigned int> &getNodalSendOffsets() const {
        return m_uiSendNodeOffset;
    }

    /**@brief returns the nodal recv counts*/
    inline const std::vector<unsigned int> &getNodalRecvCounts() const {
        return m_uiRecvNodeCount;
    }

    /**@brief returns the nodal recv offsets*/
    inline const std::vector<unsigned int> &getNodalRecvOffsets() const {
        return m_uiRecvNodeOffset;
    }

    /**@brief returns the send proc. list*/
    inline const std::vector<unsigned int> &getSendProcList() const {
        return m_uiSendProcList;
    }

    /**@brief returns the recv proc. list*/
    inline const std::vector<unsigned int> &getRecvProcList() const {
        return m_uiRecvProcList;
    }

    /**@brief return Scatter map for node send*/
    inline const std::vector<unsigned int> &getSendNodeSM() const {
        return m_uiScatterMapActualNodeSend;
    }

    /**@brief return Scatter map for node send*/
    inline const std::vector<unsigned int> &getRecvNodeSM() const {
        return m_uiScatterMapActualNodeRecv;
    }

    /**@brief returns the cell/element send counts*/
    inline const std::vector<unsigned int> &getElementSendCounts() const {
        return m_uiSendEleCount;
    }

    /**@brief returns the cell/element send offsets*/
    inline const std::vector<unsigned int> &getElementSendOffsets() const {
        return m_uiSendEleOffset;
    }

    /**@brief returns the cell/element recv counts*/
    inline const std::vector<unsigned int> &getElementRecvCounts() const {
        return m_uiRecvEleCount;
    }

    /**@brief returns the cell/element recv offsets*/
    inline const std::vector<unsigned int> &getElementRecvOffsets() const {
        return m_uiRecvEleOffset;
    }

    /**@brief returns the cell/element send proc list*/
    inline const std::vector<unsigned int> &getSendEleProcList() const {
        return m_uiElementSendProcList;
    }

    /**@brief returns the cell/element recv proc list*/
    inline const std::vector<unsigned int> &getRecvEleProcList() const {
        return m_uiElementRecvProcList;
    }

    /**@brief: returns the scatter map for send element cell/ DG computations,
     * note that this is offset by m_uiElementLocalBegin. */
    inline const std::vector<unsigned int> &getSendElementSM() const {
        return m_uiScatterMapElementRound1;
    }

    /**@breif: returns the scatter map for the recv element */
    inline const std::vector<unsigned int> &getRecvElementSM() const {
        return m_uiGhostElementRound1Index;
    }

    /**@brief: set the min and max bounds to the domain. */
    void setDomainBounds(Point dmin, Point dmax) {
        m_uiDMinPt = Point(dmin.x(), dmin.y(), dmin.z());
        m_uiDMaxPt = Point(dmax.x(), dmax.y(), dmax.z());

        dendro::logger::debug(dendro::logger::Scope{"MESH"},
                              "Domain bounds set to ({}, {}, {})->({}, {}, {})",
                              m_uiDMinPt.x(), m_uiDMinPt.y(), m_uiDMinPt.z(),
                              m_uiDMaxPt.x(), m_uiDMaxPt.y(), m_uiDMaxPt.z());
    }

    /**@brief: get the domain min point. */
    inline Point getDomainMinPt() const { return m_uiDMinPt; }

    /**@brief: get the domain max point. */
    inline Point getDomainMaxPt() const { return m_uiDMaxPt; }

    /** @brief: Decompose the DG index to element id and it's i,j,k values.*/
    inline void dg2eijk(unsigned int dg_index, unsigned int &e, unsigned int &i,
                        unsigned int &j, unsigned int &k) const {
        // a more readable version that avoids comparisons
        e                                   = dg_index / m_uiNpE;
        unsigned int local_index            = dg_index % m_uiNpE;

        unsigned int order_plus_one         = m_uiElementOrder + 1;
        unsigned int order_plus_one_squared = order_plus_one * order_plus_one;

        k = local_index / order_plus_one_squared;
        local_index %= order_plus_one_squared;

        j = local_index / order_plus_one;
        i = local_index % order_plus_one;

#if 0
        e = dg_index / m_uiNpE;
        k = 0;
        j = 0;
        i = 0;

        // std::cout<<"e: "<<e<<std::endl;
        if (dg_index > e * m_uiNpE)
            k = (dg_index - e * m_uiNpE) /
                ((m_uiElementOrder + 1) * (m_uiElementOrder + 1));
        // std::cout<<"k: "<<k<<std::endl;
        if ((dg_index + k * ((m_uiElementOrder + 1) * (m_uiElementOrder + 1))) >
            (e * m_uiNpE))
            j = (dg_index - e * m_uiNpE -
                 k * ((m_uiElementOrder + 1) * (m_uiElementOrder + 1))) /
                (m_uiElementOrder + 1);
        // std::cout<<"j: "<<j<<std::endl;
        if ((dg_index + k * ((m_uiElementOrder + 1) * (m_uiElementOrder + 1)) +
             j * (m_uiElementOrder + 1)) > (e * m_uiNpE))
            i = (dg_index - e * m_uiNpE -
                 k * ((m_uiElementOrder + 1) * (m_uiElementOrder + 1)) -
                 j * (m_uiElementOrder + 1));
        // std::cout<<"i: "<<i<<std::endl;
#endif
    }

    inline unsigned int eijk2dg(unsigned int e, unsigned int i, unsigned int j,
                                unsigned int k) {
        unsigned int order_plus_one         = m_uiElementOrder + 1;
        unsigned int order_plus_one_squared = order_plus_one * order_plus_one;

        unsigned int dg_index =
            e * m_uiNpE + k * order_plus_one_squared + j * order_plus_one + i;
        return dg_index;

#if 0
        unsigned int dg_index =
            e * m_uiNpE + k * (m_uiElementOrder + 1) * (m_uiElementOrder + 1) +
            j * (m_uiElementOrder + 1) + i;
        return dg_index;
#endif
    }

    /**@brief returns the morton child number*/
    inline unsigned int getMortonchildNum(unsigned int eleID) const {
        return m_uiAllElements[eleID].getMortonIndex();
    }

    /**@brief returns true if mesh is active*/
    inline bool isActive() const { return m_uiIsActive; }

    /**@brief waiting for all the mesh instances both active and inactive. This
     * should not be called if not needed. this is a BARRIER. */
    inline void waitAll() const { MPI_Barrier(m_uiCommGlobal); }

    /**@brief waiting for all the mesh instances both active. This should not be
     * called if not needed. this is a BARRIER. */
    inline void waitActive() const { MPI_Barrier(m_uiCommActive); }

    /**@brief: Destroy an allocated vector using the createVectorXX */
    template <typename T>
    inline void destroyVector(T *&vec) const {
        delete[] vec;
        vec = NULL;
    }

    /**@brief set refinement flags for the octree.
     * This is non const function
     *
     * @param[in] flags indicating to refine/coarsen or no change
     * @param[in] sz: size of the array flags, sz should be equivalent to number
     * of local elements.
     *
     * */
    void setOctreeRefineFlags(unsigned int *flags, unsigned int sz);

    /**
     * @brief returns the face neighours in specidied direction.
     * @param[in] eID:  Element ID.
     * @param[in] dir: direction of the face.
     * @param[out] lookUp: result if the result is a same level or lower level
     *than element ID. (only 2 neighbor)
     **/

    void getElementalFaceNeighbors(const unsigned int eID,
                                   const unsigned int dir,
                                   unsigned int *lookup) const;
    /**
     * @brief returns the edge neighours in specidied direction.
     * @param[in] eID:  Element ID.
     * @param[in] dir: direction of the edge.
     * @param[out] lookUp: result if the result is a same level or lower level
     * than element ID. (only 4 neighbor)
     * */

    void getElementalEdgeNeighbors(const unsigned int eID,
                                   const unsigned int dir,
                                   unsigned int *lookup) const;
    /**
     * @brief returns the vertex neighours in specidied direction.
     * @param[in] eID:  Element ID.
     * @param[in] dir: direction of the vertex.
     * @param[out] lookUp: result if the result is a same level or lower level
     * than element ID. (only 8 neighbor)
     * */
    void getElementalVertexNeighbors(const unsigned int eID,
                                     const unsigned int dir,
                                     unsigned int *lookup) const;

    /**
     * @brief : Compute the elemental interpolation matrix.
     * @param[in] currentId: Element ID.
     * @param[in/out] qMat: computed interpolation matrix. by default it is
     * assumed to be allocated.
     *
     */
    void getElementQMat(unsigned int currentId, double *&qMat,
                        bool isAllocated = true) const;

    /**
     * @brief Get the elemental nodal values using unzip representation of the
     * array.
     *
     * @tparam T type of the vector.
     * @param uzipVec : unzip vector
     * @param blkID : block ID the element belongs to
     * @param ele : element ID of the block
     * @param out : output values. (allocated with corresponding padding width)
     * @param isPadded : true if the we need the elemental values with padding.
     */
    template <typename T>
    void getUnzipElementalNodalValues(const T *uzipVec, unsigned int blkID,
                                      unsigned int ele, T *out,
                                      bool isPadded = true) const;

    // Wavelet Init functions
    /**
     * @brief Initilizes the wavelet DA loop depending on the WaveletDA flags
     * specified
     * */
    template <ot::WaveletDA::LoopType type>
    void init();

    /**
     * @brief Check whether the next element is available.
     * */

    template <ot::WaveletDA::LoopType type>
    bool nextAvailable();

    /**
     * @brief Increment the counters to access the next element in the mesh.
     * */

    template <ot::WaveletDA::LoopType type>
    void next();

    /**
     * @brief Return the current element as an octant.
     * */
    inline const ot::TreeNode &currentOctant();

    /**
     * @brief Returns the current element index.
     * */
    inline unsigned int currentIndex();

    /**
     * @brief Returns the current neighbour list (Element) information. Note
     * that for 3D it is 8 neighbours for each octant and for 2D it is 4.
     * */

    inline void currentElementNeighbourIndexList(unsigned int *neighList);

    /**
     * @brief: Returns the node index list belongs to current element.
     * NodeList size should be m_uiNpE;
     * */

    inline void currentElementNodeList(unsigned int *nodeList);

    /**
     * @brief Returns the node index list belogns to the currentl element in DG
     * indexing. NodeList size should be m_uiNpE;
     *
     * */
    inline void currentElementNodeList_DG(unsigned int *nodeList);

    // functions to access, faces and edges of a given element.

    /**
     * @brief Returns the index of m_uiE2NMapping (CG or DG) for a specified
     * face.
     * @param [in] elementID: element ID of the face, that the face belongs to .
     * @param [in] face : Face that you need the indexing,
     * @param [in] isInternal: return only the internal nodes if true hence the
     * index size would be ((m_uiElementOrder-1)*(m_uiElementOrder-1))
     * @param [out] index: returns the indecies of the requested face.
     * */

    inline void faceNodesIndex(unsigned int elementID, unsigned int face,
                               std::vector<unsigned int> &index,
                               bool isInternal) const;

    /**
     * @brief Returns the index of m_uiE2NMapping (CG or DG) for a specified
     * Edge.
     * @param [in] elementID: element ID of the edge, that the face belongs to .
     * @param [in] face1 : one of the face, that an edge belongs to
     * @param [in] face2: second face that an edge belongs to . Hence the edge
     * in cosideration will be intersection of the two faces, face1 and face2.
     * @param [in] isInternal: return only the internal nodes if true hence the
     * index size would be ((m_uiElementOrder-1)*(m_uiElementOrder-1))
     * @param [out] index: returns the indecies of the requested face.
     * */

    inline void edgeNodeIndex(unsigned int elementID, unsigned int face1,
                              unsigned int face2,
                              std::vector<unsigned int> &index,
                              bool isInternal) const;

    /**
     * @brief Returns the index of m_uiE2NMapping (CG or DG) for a specified
     * Edge.
     * @param [in] elementID: element ID of the edge, that the face belongs to .
     * @param [in] mortonIndex: morton ID of the corner node in the cordinate
     * change in the order of x y z.
     * @param [out] index: returns the indecies of the requested face.
     * */

    inline void cornerNodeIndex(unsigned int elementID,
                                unsigned int mortonIndex,
                                unsigned int &index) const;

    /**
     * @brief Returns the all the internal node indices of an element.
     * @param [in] elementID: element ID of the edge, that the face belongs to .
     * @param [in] isInternal: return only the internal nodes if true hence the
     * index size would be ((m_uiElementOrder-1)*(m_uiElementOrder-1))
     * @param [out] index: returns the indecies of the requested face.
     *
     *
     * */

    inline void elementNodeIndex(unsigned int elementID,
                                 std::vector<unsigned int> &index,
                                 bool isInternal) const;

    /**
     * @brief: Returns true or false (bases on specified edge is hanging or
     * not.)for a given element id , and edge id.
     * @param[in] elementId: element ID of the octant.
     * @param[in] edgeId: edge id
     * */
    bool isEdgeHanging(unsigned int elementId, unsigned int edgeId,
                       unsigned int &cnum) const;

    /**
     * @brief: Returns true or false (bases on specified edge is hanging or
     * not.)for a given element id , and edge id.
     * @param[in] elementId: element ID of the octant.
     * @param[in] faceId: face id
     * */
    bool isFaceHanging(unsigned int elementId, unsigned int faceId,
                       unsigned int &cnum) const;

    /**
     * @brief: Returns true if the specified node (e,i,j,k) is hanging.
     * @param[in] eleID: element ID
     * @param[in] ix: i-index of the node.
     * @param[in] jy: j-index of the node.
     * @param[in] kz: k-index of the node.
     * */
    bool isNodeHanging(unsigned int eleID, unsigned int ix, unsigned int jy,
                       unsigned int kz) const;

    /**
     * @brief: Returns true if the specified node (e,i,j,k) is local.
     * @param[in] eleID: element ID
     * @param[in] ix: i-index of the node.
     * @param[in] jy: j-index of the node.
     * @param[in] kz: k-index of the node.
     * */
    inline bool isNodeLocal(unsigned int eleID, unsigned int ix,
                            unsigned int jy, unsigned int kz) const {
        return ((m_uiE2NMapping_CG[eleID * m_uiNpE +
                                   kz * (m_uiElementOrder + 1) *
                                       (m_uiElementOrder + 1) +
                                   jy * (m_uiElementOrder + 1) + ix] >=
                 m_uiNodeLocalBegin) &&
                (m_uiE2NMapping_CG[eleID * m_uiNpE +
                                   kz * (m_uiElementOrder + 1) *
                                       (m_uiElementOrder + 1) +
                                   jy * (m_uiElementOrder + 1) + ix] <
                 m_uiNodeLocalEnd));
    };

    inline bool isBoundaryOctant(unsigned int ele) const {
        assert(ele < m_uiAllElements.size());
        return (m_uiAllElements[ele].minX() == 0 ||
                m_uiAllElements[ele].minY() == 0 ||
                m_uiAllElements[ele].minZ() == 0 ||
                m_uiAllElements[ele].maxX() == 1u << (m_uiMaxDepth) ||
                m_uiAllElements[ele].maxY() == 1u << (m_uiMaxDepth) ||
                m_uiAllElements[ele].maxZ() == 1u << (m_uiMaxDepth));
    }

    inline DendroIntL getGhostExcgTotalSendNodeCount() const {
        if (m_uiGlobalNpes == 1) return 0;

        if (m_uiIsActive)
            return (m_uiSendNodeOffset[m_uiActiveNpes - 1] +
                    m_uiSendNodeCount[m_uiActiveNpes - 1]);
        else
            return 0;
    }

    inline DendroIntL getGhostExcgTotalRecvNodeCount() const {
        if (m_uiGlobalNpes == 1) return 0;

        if (m_uiIsActive)
            return (m_uiRecvNodeOffset[m_uiActiveNpes - 1] +
                    m_uiRecvNodeCount[m_uiActiveNpes - 1]);
        else
            return 0;
    }

    /**@brief : returns the coarset block level allowed. */
    inline unsigned int getCoarsetBlockLevAllowed() const {
        return m_uiCoarsetBlkLev;
    }

    /**@brief get plitter nodes for each processor. */
    inline const ot::TreeNode *getNodalSplitterNodes() const {
        return m_uiSplitterNodes;
    }

    // Methods needed for PDE & ODE and other solvers.
    /**@brief allocate memory for variable array based on the adaptive mesh*/
    template <typename T>
    T *createVector() const;

    /**
     * @brief create CG nodal vector
     *
     * @tparam T : vector data type
     * @param initVal: inital value
     * @param dof : degrees of freedoms.
     * @return T*
     */
    template <typename T>
    T *createCGVector(T initVal = 0, unsigned int dof = 1) const;

    /**
     * @brief create a CG nodal vector and initialize it to func
     * @tparam T vector type
     * @param func: function f(x,y,z,val)
     * @param dof : dof for the vector
     * @return T*
     */
    template <typename T>
    T *createCGVector(std::function<void(T, T, T, T *)> func,
                      unsigned int dof = 1) const;

    /**
     * @brief Create a Element Vector
     *
     * @tparam T vector data type.
     * @param initVal initialize value
     * @return T*
     */
    template <typename T>
    T *createElementVector(T initVal = 0, unsigned int dof = 1) const;

    /**
     * @brief Create a Element DG Vector object (each element will have it's own
     * node)
     *
     * @tparam T vector type
     * @param initVal : initial value for the vector
     * @return T*
     */
    template <typename T>
    T *createDGVector(T initVal = 0, unsigned int dof = 1) const;

    /**
     * @brief create a CG nodal vector and initialize it to func
     * @tparam T vector type
     * @param func: function f(x,y,z,val)
     * @param dof : dof for the vector
     * @return T*
     */
    template <typename T>
    T *createDGVector(std::function<void(T, T, T, T *)> func,
                      unsigned int dof = 1) const;

    /**@brief allocate memory for variable array based on the adaptive mesh.*/
    template <typename T>
    void createVector(std::vector<T> &vec) const;

    /**@brief allocate memory for variable array based on the adaptive mesh.
     * @param[in] initValue: initialize the vector to the given value.
     * */
    template <typename T>
    T *createVector(const T initValue) const;

    /**@brief allocate memory for variable array based on the adaptive mesh.
     * @param[in] initValue: initialize the vector to the given value.
     * */
    template <typename T>
    T *createVector(std::function<T(T, T, T)> func) const;

    /**@brief allocate memory for variable array based on the adaptive mesh.
     * @param[in] vec: allocate memory for vec
     * @param[in] initValue: initialize the vector to the given value.
     * */
    template <typename T>
    void createVector(std::vector<T> &vec, const T initValue) const;

    /**
     * @brief create and initialize local elements based on a given function.
     * @param[in] vec: mesh varaible vector
     * @param[in] func: function to initialize with.
     * */
    template <typename T>
    void createVector(std::vector<T> &vec,
                      std::function<T(T, T, T)> func) const;

    /**
     * @brief: creates (memory allocation) for the unzipped version of the
     * vector.
     * @param[in]: vector to allocate memory for unzipped version
     * @param[out]: vector allocated the memory for unzipped version.
     * */
    template <typename T>
    void createUnZippedVector(std::vector<T> &uvec) const;

    /**
     * @brief: creates (memory allocation) for the unzipped version of the
     * vector.
     * @param[in]: vector to allocate memory for unzipped version
     * @param[out]: vector allocated the memory for unzipped version.
     * */
    template <typename T>
    T *createUnZippedVector(unsigned int dof = 1) const;

    /**@brief allocate memory for variable array based on the adaptive mesh.
     * @param[in] uvec: allocate memory for uvec (unzipped version)
     * @param[in] initValue: initialize the vector to the given value.
     * */
    template <typename T>
    void createUnZippedVector(std::vector<T> &uvec, const T initValue) const;

    /**@brief allocate memory for variable array based on the adaptive mesh.
     * @param[in] uvec: allocate memory for uvec (unzipped version)
     * @param[in] initValue: initialize the vector to the given value.
     * */
    template <typename T>
    T *createUnZippedVector(const T initValue, unsigned int dof = 1) const;

    /**
     * @brief converts a cg vector to element local DG vector.
     * @tparam T type of the vector.
     * @param cg_vec : Input cg vector
     * @param dg_vec : input dg vector
     * @param isAllocated : true if dg_vector is allocated
     * @param gsynced: true if ghost is synced.
     * @param dof : degrees of freedom.
     * @return T*
     */
    template <typename T>
    void CG2DGVec(T *cg_vec, T *dg_vec, bool gsynced, unsigned int dof = 1);

    /**
     * @brief converts a cg vector to element local DG vector.
     * note : dg to cg is well defined if and only if the element boundary nodes
     * are on agree with each other.
     * @tparam T type of the vector.
     * @param cg_vec : Input cg vector
     * @param dg_vec : input dg vector
     * @param isAllocated : true if dg_vector is allocated
     * @param gsynced: true if ghost is synced.
     * @param dof : degrees of freedom.
     * @return T*
     */
    template <typename T>
    void DG2CGVec(const T *dg_vec, T *cg_vec, unsigned int dof = 1) const;

    /**
     * @brief performs partial DG to CG vec conversion.
     *
     * @tparam T
     * @param dg_vec
     * @param cg_vec
     * @param isAllocated
     * @param int
     * @param nEle
     * @param dof
     */
    template <typename T>
    void DG2CGVec(const T *dg_vec, T *&cg_vec, bool isAllocated,
                  const unsigned int *eleIDs, unsigned int nEle,
                  unsigned int dof = 1) const;

    /**
     *
     * @brief Performs all parent to child interpolations for the m_uiEl_i
     * element in order to apply the stencil.
     * @param[in] parent: function values
     * @param[in] cnum: child number to interpolate.
     * @param[in] dim: dim of the interpolation. (dim=1 for edge interpolation ,
     * dim=2 for face interpolation, dim=3 for octant to child interpolation.)
     * @param[out] out: interpolated values.
     *
     * */
    inline void parent2ChildInterpolation(const double *in, double *out,
                                          unsigned int cnum,
                                          unsigned int dim = 3) const;

    /**
     * @brief performs the child to parent contribution (only from a single
     * child).
     * @param[in] in: child function values
     * @param[in] cnum: morton ID of the current child.
     * @param[in] dim: dim of the interpolation. (dim=1 for edge interpolation ,
     * dim=2 for face interpolation, dim=3 for octant to child interpolation.)
     * @param[out] out: interpolated values. (child to parent contribution)
     *
     * */

    inline void child2ParentInterpolation(const double *in, double *out,
                                          unsigned int cnum,
                                          unsigned int dim = 3) const;

    /**
     * @brief Performs child to parent injection.
     * @param [in] in : input vector.
     * @param [out] out : injected vector
     * @param [in] child : element IDs of the children should be at the same
     * level.
     * @param [in] lev: level of the children, all the children should be in the
     * same level otherwise they will be skipped.
     * */
    template <typename T>
    void child2ParentInjection(const T *in, T *out, unsigned int *child,
                               unsigned int lev) const;

    /**
     * @author Milinda Fernando
     * @brief Creates the decomposition of adaptive octree variables into
     * blocklist variables that we computed.
     * @param [in] in : adaptive representation of the variable array. (created
     * by createVec function)
     * @param [out] out: decomposed representation of the adaptive array.
     * @note this routine assumes that for both arrays memory has been
     * allocated. Routine is responsible only to fill up the unzipped entries.
     * */
    template <typename T>
    void unzip(const T *in, T *out, unsigned int dof = 1);

    /**
     * @brief performs unzip operation for a given block id.
     *
     * @tparam T type of the vector.
     * @param in : zipped vector
     * @param out : unzipped vector.
     * @param blk :pointer to list of block ids, for the unzip.
     * @param numblks: number of block ids specified.
     */
    template <typename T>
    void unzip(const T *in, T *out, const unsigned int *blkIDs,
               unsigned int numblks, unsigned int dof = 1);

    /**
     * @brief performs unzip operation for all the blocks, each element scatters
     * the unzip data to its corresponding blocks. Note : This cannot be used to
     * unzip only a one block, which is not supported (can be MODIFIED to get it
     * done but would be ineffient).
     * @tparam T type of the vector.
     * @param in : zipped vector
     * @param out : unzipped vector.
     * @param blk :pointer to list of block ids, for the unzip.
     * @param numblks: number of block ids specified.
     */
    template <typename T>
    void unzip_scatter(const T *in, T *out, unsigned int dof = 1);

    /**
     * @brief performs unzip operation for a given block id.
     *
     * @tparam T type of the vector.
     * @param in : DG vector
     * @param out : unzipped vector.
     * @param blk :pointer to list of block ids, for the unzip.
     * @param numblks: number of block ids specified.
     */
    template <typename T>
    void unzipDG(const T *in, T *out, const unsigned int *blkIDs,
                 unsigned int numblks, unsigned int dof = 1);

    /**
     * @brief Unzip with local scatter communication patterm, not duplicate
     * interplolations.
     * @tparam T type of the vector.
     * @param in : DG vector
     * @param out : unzipped vector.
     * @param blk :pointer to list of block ids, for the unzip.
     * @param numblks: number of block ids specified.
     *
     * Note :  this cannot be used to unzip only paticular blocks, unzips to all
     * the blocks, each element scatters its data to the corresponding block.
     */
    template <typename T>
    void unzipDG_scatter(const T *in, T *out, unsigned int dof = 1);

    /**
     * @brief Creates the decomposition of adaptive octree variables into
     * blocklist variables that we computed.
     * @param [in] in : adaptive representation of the variable array. (created
     * by createVec function)
     * @param [out] out: decomposed representation of the adaptive array.
     * @note this routine assumes that for both arrays memory has been
     * allocated. Routine is responsible only to fill up the unzipped entries.
     * */
    template <typename T>
    void unzipDG(const T *in, T *out, unsigned int dof = 1);

    /**@author Milinda Fernando
     * @brief Performs the compression frrom regular block grid varable list to
     * adaptive representation.
     * @param [in] unzippedVec decomposed version of the adaptive array
     * @param [out] compressed version of the unzippedVec.
     * */
    template <typename T>
    void zip(const T *unzippedVec, T *zippedVec);

    /**
     * @brief perform block wise zip operation.
     *
     * @tparam T type of the vector
     * @param unzippedVec : unzip vector
     * @param zippedVec : zipped vector
     * @param local_blkID : local block id.
     */
    template <typename T>
    void zip(const T *unzippedVec, T *zippedVec, const unsigned int *blkIDs,
             unsigned int numblks, unsigned int ll);

    /**
     * @brief Apply a given stencil to for provided variable array.
     * @param [in] in : vector that we need to apply the stencil on.
     * @param [in] centered: Stencil that we need to apply on vector in, this is
     * the centered stencil.
     * @param [in] backward: backward version of the centered stencil.(This is
     * used for boundary elements. )
     * @param [in] forward: foward version of the centered stencil. (This is
     * used for boundary elements. )
     * @param [out] out: output vector that after applying the stencil.
     */

    template <typename T, unsigned int length, unsigned int offsetCentered,
              unsigned int offsetBackward, unsigned int offsetForward>
    void applyStencil(const std::vector<T> &in, std::vector<T> &out,
                      const Stencil<T, length, offsetCentered> &centered,
                      const Stencil<T, length, offsetBackward> &backward,
                      const Stencil<T, length, offsetForward> &forward);

    /**
     * @brief Perform the ghost exchange for the vector vec.
     * @param [in] vec: adaptive mesh vector contaiting the values.
     * */
    template <typename T>
    void performGhostExchange(std::vector<T> &vec);

    /**
     * @brief Perform the ghost exchange for the vector vec.
     * @param [in] vec: adaptive mesh vector contaiting the values.
     * */
    template <typename T>
    void performGhostExchange(T *vec);

    /**
     * @brief Perform the ghost asynchronous send for the vector vec Note: this
     * is a non-blocking asynchronous communication. User is resposible to call
     * the (synchronous call such as MPI_WaitAll) when overlapping the
     * communication and computation.
     * @param [in] vec: adaptive mesh vector contaiting the values.
     * */
    template <typename T>
    void ghostExchangeStart(T *vec, T *sendNodeBuffer, T *recvNodeBuffer,
                            MPI_Request *send_reqs, MPI_Request *recv_reqs);

    /**
     * @brief Perform the wait on the recv requests
     * @param [in] vec: adaptive mesh vector containing the values.
     * @param [in] recv_reqs: m_uiRecvProcList.size() recv request
     * @param [in] recv_sts: m_uiRecvProcList.size() recv status
     * */
    template <typename T>
    void ghostExchangeRecvSync(T *vec, T *recvNodeBuffer,
                               MPI_Request *recv_reqs, MPI_Status *recv_sts);

    /**
     * @brief : ghost read begin.
     *
     * @tparam T
     * @param vec : vector to perform ghost syncronization. (begin)
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void readFromGhostBegin(T *vec, unsigned int dof = 1);

    /**
     * @brief:  ghost read end
     *
     * @tparam T
     * @param vec : vector to perform ghost syncronization. (begin)
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void readFromGhostEnd(T *vec, unsigned int dof = 1);

    /**
     * @brief Aysnc ghost exchange with Ctx, assumes that ctx bufferes are
     * already allocated.
     * @tparam T type of the vector
     * @param ctx async ctx
     * @param dof number of dofs to exchange
     */
    template <typename T>
    void readFromGhostBegin(AsyncExchangeContex &ctx, T *vec,
                            unsigned int dof = 1);

    /**
     * @brief Aysnc ghost exchange with Ctx, assumes that ctx bufferes are
     * already allocated.
     * @tparam T type of the vector
     * @param ctx async ctx
     * @param dof number of dofs to exchange
     */
    template <typename T>
    void readFromGhostEnd(AsyncExchangeContex &ctx, T *vec,
                          unsigned int dof = 1);

    /**
     * @brief : ghost read begin.
     *
     * @tparam T
     * @param vec : vector to perform ghost syncronization. (begin) for
     * elemental vector (cell vector)
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void readFromGhostBeginElementVec(T *vec, unsigned int dof = 1);

    /**
     * @brief:  ghost read end
     *
     * @tparam T
     * @param vec : vector to perform ghost syncronization. (begin) for
     * elemental vector (cell vector)
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void readFromGhostEndElementVec(T *vec, unsigned int dof = 1);

    /**
     * @brief : ghost read begin for a DG vector space with CG vector, but
     * communication occurs in the element DG format.
     *
     * @tparam T
     * @param vec : vector to perform ghost syncronization. (begin) for
     * elemental vector (element local vector)
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void readFromGhostBeginEleDGVec(T *vec, unsigned int dof = 1);

    /**
     * @brief : ghost read end for a DG vector space with CG vector, but
     * communication occurs in the element DG format.
     *
     * @tparam T
     * @param vec : vector to perform ghost syncronization. (begin) for
     * elemental vector (element local vector)
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void readFromGhostEndEleDGVec(T *vec, unsigned int dof = 1);

    /**
     * @brief : begin of the write from ghost.
     *
     * @tparam T
     * @param vec : vector to perform ghost syncronization. (begin)
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void writeFromGhostBegin(T *vec, unsigned int dof = 1);

    /**
     * @brief:  end of write from ghost.
     *
     * @tparam T
     * @param vec : vector to perform write from ghost. (begin)
     * @param mode: write mode.
     * @param dof : degrees of freedoms
     */
    template <typename T>
    void writeFromGhostEnd(T *vec, ot::GWMode mode, unsigned int dof = 1);

    /**
     * @brief
     * @tparam T
     * @param vec
     * @param dof
     */
    template <typename T>
    void gatherFromGhostBegin(T *vec, unsigned int dof = 1);

    /**
     * @brief
     *
     * @tparam T
     * @param vec
     * @param gatherV
     * @param dof
     */
    template <typename T>
    void gatherFromGhostEnd(T *vec, std::vector<std::vector<T>> &gatherV,
                            unsigned int dof = 1);

    /**
     * @brief Perform the wait on the recv requests
     * @param [in] vec: adaptive mesh vector containing the values.
     * @param [in] send_reqs: m_uiSendProcList.size() send request
     * @param [in] send_sts: m_uiSendProcList.size() send status
     * */
    inline void ghostExchangeSendSync(MPI_Request *send_reqs,
                                      MPI_Status *send_sts) {
        MPI_Waitall(m_uiSendProcList.size(), send_reqs, send_sts);
    }

    /**
     * @brief write out function values to a vtk file.
     * @param[in] vec: variable vector that needs to be written as a vtk file.
     * @param[in] fprefix: prefix of the output vtk file name.
     * */
    template <typename T>
    void vectorToVTK(const std::vector<T> &vec, char *fprefix,
                     double pTime = 0.0, unsigned int nCycle = 0) const;

    /**
     * @param[in] vec: sequence of varaibles to check
     * @param[in] varIds: variable ids to check. (var ids to index the vec,
     * vec[i] is a T* pointintg to one of the variable in vec. )
     * @param[in] numVars: number of variables to check
     * @param[in] tol: wavelet tolerance
     * Returns true if specified variable violates the specified wavelet
     * toerlance.
     * @note: this method will flag every element in the mesh with
     * OCT_NO_CHANGE, OCT_SPLIT, OCT_COARSE.
     *
     * */

    template <typename T>
    bool isReMeshUnzip(
        const T **unzippedVec, const unsigned int *varIds,
        const unsigned int numVars,
        std::function<double(double, double, double, double *)> wavelet_tol,
        double amr_coarse_fac = DENDRO_AMR_COARSEN_FAC,
        double coarsen_hx     = DENDRO_REMESH_UNZIP_SCALE_FAC);

    /**
     * @brief
     *
     * @tparam T
     * @param blkID
     * @param unzippedVec
     * @param varIds
     * @param numVars
     * @param wavelet_tol
     * @param amr_coarse_fac
     * @param coarsen_hx
     * @return true
     * @return false
     */
    template <typename T>
    bool isReMeshBlk(unsigned int blkID, const T **unzippedVec,
                     const unsigned int *varIds, const unsigned int numVars,
                     std::function<double(double, double, double)> wavelet_tol,
                     double amr_coarse_fac = DENDRO_AMR_COARSEN_FAC,
                     double coarsen_hx     = DENDRO_REMESH_UNZIP_SCALE_FAC);

    /**
     * @brief: Remesh the mesh with the new computed elements.
     * @note assumes that refinedIDs and corasenIDs are sorted. (This is
     * automatically done by the isRemesh Fucntion)
     * @param[in] refinedIDs: element IDs need to be refined. (computed by
     * isReMesh function)
     * @param[in] coarsenIDs: element IDs need to be coarsened. (computed by
     * isReMesh function)
     * @param[in] ld_tol: tolerance value used for flexible partitioning
     * @param[in] sfK: spliiter fix parameter (need to specify larger value when
     * run in super large scale)
     * @param[in] getWeight: function pointer which returns a uint weight values
     * for an given octant
     * */
    ot::Mesh *ReMesh(unsigned int grainSz = DENDRO_DEFAULT_GRAIN_SZ,
                     double ld_tol        = DENDRO_DEFAULT_LB_TOL,
                     unsigned int sfK     = DENDRO_DEFAULT_SF_K,
                     unsigned int (*getWeight)(const ot::TreeNode *) = NULL,
                     unsigned int *blk_tags                          = NULL,
                     unsigned int blk_tag_sz                         = 0);

    /**
     * @brief: Remeshes and repartitions using the current partitioning
     * method.  For graph-partitioned meshes, this builds a temporary
     * SFC mesh via ReMesh, then calls repartitionMeshGlobal to apply
     * the graph partition.  Returns the new mesh (caller owns it).
     * @param grainSz, ld_tol, sfK: passed to ReMesh
     */
    ot::Mesh *ReMeshRepartitioned(
        unsigned int grainSz = DENDRO_DEFAULT_GRAIN_SZ,
        double ld_tol        = DENDRO_DEFAULT_LB_TOL,
        unsigned int sfK     = DENDRO_DEFAULT_SF_K);

    /**
     * @brief: Computes the all to all v communication parameters interms of
     * element counts. Let M1 be the current mesh, M2 be the new mesh (pMesh),
     * then we compute M2' auxiliary mesh, where, M2' is partitioned w.r.t
     * splitters, of the M1. Computed communication parameters, tells us how to
     * perform data transfers from, M2' to M2. Also note that the allocated
     * send/recv counts parameters should be in global counts.
     * @param pMesh : new mesh M2.
     */
    void interGridTransferSendRecvCompute(const ot::Mesh *pMesh);

    /**
     * @brief transfer a variable vector form old grid to new grid. Assumes the
     * ghost is synchronized in the old vector
     * @param[in] vec: variable vector needs to be transfered.
     * @param[out] vec: transfered varaible vector
     * @param[in] pMesh: Mesh that we need to transfer the old varaible.
     * */
    template <typename T>
    void interGridTransfer(
        std::vector<T> &vec, const ot::Mesh *pMesh,
        INTERGRID_TRANSFER_MODE mode = INTERGRID_TRANSFER_MODE::INJECTION);

    /**
     * @brief transfer a variable vector form old grid to new grid. Assumes the
     * ghost is synchronized in the old vector
     * @param[in] vec: variable vector needs to be transfered.
     * @param[out] vec: transfered varaible vector
     * @param[in] pMesh: Mesh that we need to transfer the old varaible.
     * @param mode : intergrid transfer mode.
     * @param dof : number of dof.
     * */
    template <typename T>
    void interGridTransfer(
        T *&vec, const ot::Mesh *pMesh,
        INTERGRID_TRANSFER_MODE mode = INTERGRID_TRANSFER_MODE::INJECTION,
        unsigned int dof             = 1);

    /**
     * @brief Performs intergrid transfer without deallocating the existing
     * vector. Assumes the ghost is synchronized in the old vector
     * @tparam T : data type of the vector.
     * @param vec : input vector (vector corresponding to the old mesh)
     * @param vecOut : output vector (new vector consresponding to the new
     * vector)
     * @param pMesh : pointer to the new mesh object.
     * @param isAlloc : True if out vector is allocated with ghost, false
     * otherwise.
     * @param mode: mode of intergrid transfer defined by
     * INTERGRID_TRANSFER_MODE
     * @param dof : number of dof.
     */
    template <typename T>
    void interGridTransfer(
        T *vecIn, T *vecOut, const ot::Mesh *pMesh,
        INTERGRID_TRANSFER_MODE mode = INTERGRID_TRANSFER_MODE::INJECTION,
        unsigned int dof             = 1);

    /**
     * @brief Intergrid transfer for the 2D vector.
     * @tparam T
     * @param vec : input vector (allocated in the current mesh)
     * @param vecOut : output vector (allocated in the new mesh)
     * @param dof : size of different variables.
     * @param pMesh : new mesh
     * @param mode : intergrid transfer mode.
     */
    template <typename T>
    void intergridTransfer(
        T **vecIn, T **vecOut, unsigned int dof, const ot::Mesh *pMesh,
        INTERGRID_TRANSFER_MODE mode = INTERGRID_TRANSFER_MODE::INJECTION);

    /**
     * @brief: intre-grid transfer for DG vector.
     * currently only inmplemented for the strong form, intergrid transfers,
     * (i.e. child to parent happens with injection, not p2c^T)
     * @tparam T vector type.
     * @param vec : Input vector.
     * @param vecOut : Output vector
     * @param pMesh : pointer to the new mesh object
     * @param isAlloc : True if out vector is allocated with ghost, false
     * otherwise.
     */
    template <typename T>
    void interGridTransfer_DG(T *vecIn, T *vecOut, const ot::Mesh *pMesh,
                              unsigned int dof = 1);

    /**
     * @brief performs intergrid transfer for a cell vector.
     * @tparam T type of the vector.
     * @param vec : input cell vector.
     * @param vecOut : allocated new cell vector.
     * @param pMesh : new mesh.
     */
    template <typename T>
    void interGridTransferCellVec(
        T *vecIn, T *vecOut, const ot::Mesh *pMesh, unsigned int dof = 1,
        INTERGRID_TRANSFER_MODE mode = INTERGRID_TRANSFER_MODE::CELLVEC_CPY);

    /**
     * @brief Redistribute a CG vector across a partition change.
     *
     * Assumes `this` (source mesh) and `dstMesh` contain the same global
     * element set, only distributed differently across ranks (e.g. this
     * mesh is SFC-partitioned and dstMesh is graph-partitioned from a
     * subsequent `repartitionMeshGlobal`). Works only for the NO_CHANGE
     * refinement case — if elements differ (SPLIT/COARSE), use
     * `interGridTransfer` first to a same-partition mesh, then this
     * to the graph-partitioned twin.
     *
     * Uses a TreeNode-keyed Allgatherv (to build a global TreeNode→rank
     * map for dstMesh) + Alltoallv of per-element DG values. Writes into
     * `vecOut` via dstMesh's E2N_CG, and refreshes dstMesh's internal
     * m_uiLocalNodalDG buffer (used by the graph-partition DG ghost path).
     *
     * Pass dof>1 to redistribute several fields at once: vecIn/vecOut are
     * laid out [dof][CG size] (field v at vecIn + v*this->getDegOfFreedom()
     * and vecOut + v*dstMesh->getDegOfFreedom()). The scatter plan is built
     * once and all fields move in a single batched set of collectives,
     * instead of one Allgatherv+Alltoallv set per field.
     *
     * Caller responsibilities:
     *   - vecIn sized as this->createVector<T>(dof)
     *   - vecOut sized as dstMesh->createVector<T>(dof)
     *   - vecIn ghosts should be synchronized (performGhostExchange) before
     *     calling, otherwise DG values read from ghost elements' CG slots
     *     will be stale.
     */
    template <typename T>
    void redistributeVec(ot::Mesh *dstMesh, const T *vecIn, T *vecOut,
                         unsigned int dof = 1) const;

    /**
     *@brief : Returns the nodal values of a given element for a given variable
     *vector.
     *@param[in] vec: variable vector that we want to get the nodal values.
     *@param[in] elementID: element ID that we need to get the nodal values.
     *@param[in] isDGVec: true if the vec is elemental dg vec.
     *@param[out] nodalValues: nodal values of the specified element ID
     *
     * */
    template <typename T>
    void getElementNodalValues(const T *vec, T *nodalValues,
                               unsigned int elementID,
                               bool isDGVec = false) const;

    /**
     * @assumption: input is the elemental nodal values.
     * @brief: Computes the contribution of elemental nodal values to the parent
     * elements if it is hanging. Note: internal nodes for the elements cannnot
     * be hagging. Only the face edge nodes are possible for hanging.
     *
     * @param[in] vec: child var vector (nPe)
     * @param[in] elementID: element ID of the current element (or child octant)
     * @param[out] out: add the contributions to the current vector accordingly.
     *
     * Usage: This is needed when performing matrix-free matvec for FEM method.
     *
     * */
    template <typename T>
    void computeElementalContribution(const T *in, T *out,
                                      unsigned int elementID) const;

    /**@brief computes the elementCoordinates (based on the nodal placement)
     * @param[in] eleID : element ID
     * @param[in/out] coords: computed coords (note: assumes memory is allocated
     * allocated) coords are stored by p0,p1,p2... each pi \in R^dim where pi
     * are ordered in along x axis y axis and z coors size m_uiDim*m_uiNpE
     * */

    void getElementCoordinates(unsigned int eleID, double *coords) const;

    /**
     * @brief computes the face neighbor points for additional computations for
     * a specified direction.
     * @param [in] eleID: element ID
     * @param [in] in: inpute vector
     * @param [out] out: output vector values are in the order of the x,y,z size
     * : 4*NodesPerElement
     * @param [out] coords: get the corresponding coordinates size:
     * 4*NodesPerElement*m_uiDim;
     * @param [out] neighID: face neighbor octant IDs,
     * @param [in] face: face direction in
     * {OCT_DIR_LEFT,OCT_IDR_RIGHT,OCT_DIR_DOWN,
     * OCT_DIR_UP,OCT_DIR_BACK,OCT_DIR_FRONT}
     * @param [out] level: the level of the neighbour octant with respect to the
     * current octant. returns  the number of face neighbours 1/4 for 3D.
     * */
    template <typename T>
    int getFaceNeighborValues(unsigned int eleID, const T *in, T *out,
                              T *coords, unsigned int *neighID,
                              unsigned int face, NeighbourLevel &level) const;

    /** @brief returns the type of the element.
     * */
    EType getElementType(unsigned int eleID);

    /**
     * @brief compute the block boundary parent nodal locations.
     * @param[in] blkId : block id.
     * @param[in] eleId : element id of the block
     * @param[in] dir: face direction.
     * @param[out] child: element id for the block boudary faces, (elements
     * containing inside the block)
     * @param[out] fid: reference pointer to the child array for the finer
     * elements.
     * @param[out] cid: reference pointer to the child array for the coarser
     * elements. (if it was refined. (cnumbers reference to the coarser
     * elements))
     */
    int getBlkBdyParentCNums(unsigned int blkId, unsigned int eleId,
                             unsigned int dir, unsigned int *child,
                             unsigned int *fid, unsigned int *cid);

    /**
     * @brief computes the min and the maximum level of refinement.
     *
     * @param lmin : min refinement accross all procs.
     * @param lmax : max refinement acrross all procs.
     */
    void computeMinMaxLevel(unsigned int &lmin, unsigned int &lmax) const;

    /**
     * @brief Get the Finer Face Neighbors of the current element.
     *
     * @param ele : element ID
     * @param dir : face direction
     * @param child : neighbor ids (array of size 4).
     */
    void getFinerFaceNeighbors(unsigned int ele, unsigned int dir,
                               unsigned int *child) const;

    /**
     * @brief Extract the raw refinement flags from all local elements
     */
    std::vector<unsigned int> getAllRefinementFlags();

    /**
     * @brief Set the Mesh Refinement flags, for the local portion of the mesh.
     Note that coarsening happens if all the children are
     * have the same parent and all the children should be in the same processor
     as local elements.
     * In this method, mesh class ignore the wavelet refinement, and trust the
     user, and select the user specified refinement flags.
     * To perform Intergrid-transfers and other operations it is important to
     decide, refine and coarsening based on some proper,
     * basis error capture crieteria, (look at the RefEl Class, to see how
     Dendro uses the basis representation)

     * @param refine_flags : refinement flags, OCT_SPLIT, OCT_COARSE,
     OCT_NO_CHANGE
     * return true if the local partition is chnaged.
     *
     */
    bool setMeshRefinementFlags(const std::vector<unsigned int> &refine_flags);

    /**
     * @brief Perform linear transformation from octree coordinate to domain
     * coordinates
     * @param oct_pt : Octree point
     * @param domain_pt : doamin point
     */
    void octCoordToDomainCoord(const Point &oct_pt, Point &domain_pt) const;

    /**
     * @brief Perform linear coord. transformation from domain points to octree
     * coords.
     * @param domain_pt : domain point.
     * @param oct_pt : octree point.
     */
    void domainCoordToOctCoord(const Point &domain_pt, Point &oct_pt) const;

    /**
     * @brief computes tree node owner processor
     * @param pNodes List of pNodes.
     * @param n : number of pNodes.
     * @param ownerranks Owner rank size allocated (n)
     */
    void computeTreeNodeOwnerProc(const ot::TreeNode *pNodes, unsigned int n,
                                  int *ownerranks) const;

    /**
     * @brief computes the element ids of padding elements in all directions for
     * a given block id.
     *
     * @param blk block local id
     * @param eid : vector of element ids.
     */
    void blkUnzipElementIDs(unsigned int blk,
                            std::vector<unsigned int> &eid) const;

    template <typename T>
    std::vector<T> noPartitionChange(
        std::vector<oct_data<T>> &oct_connectivity_map) {
        std::vector<T> my_partition;

        for (const auto &o : oct_connectivity_map) {
            my_partition.push_back(o.eid);
        }

        std::sort(my_partition.begin(), my_partition.end());

        return my_partition;
    }

    template <typename T>
    std::vector<T> randomPartitioningSimple(
        std::vector<oct_data<T>> oct_connectivity_map) {
        int rank            = this->getMPIRank();
        int npes            = this->getMPICommSize();
        MPI_Comm commActive = this->getMPICommunicator();

        // number of local elements on the grid
        T localSz = (this->getElementLocalEnd() - this->getElementLocalBegin());

        std::vector<int> counts(npes);
        int local_size_int = localSz;

        MPI_Gather(&local_size_int, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                   commActive);

        // calculate displacements
        std::vector<int> displs;
        int total_size = 0;
        if (!rank) {
            displs.resize(npes);
            displs[0] = 0;
            for (int i = 1; i < npes; ++i) {
                displs[i] = displs[i - 1] + counts[i - 1];
            }

            total_size = std::accumulate(counts.begin(), counts.end(), 0);
        }

        // now gather all data to rank 0

        std::vector<T> global_ids;

        // randomness
        std::random_device rd;
        std::mt19937 gen(12345);

        if (rank == 0) {
            global_ids.resize(total_size);

            for (unsigned int i = 0; i < total_size; i++) {
                global_ids[i] = i;
            }

            // just shuffle the global ids
            std::shuffle(global_ids.begin(), global_ids.end(), gen);

            // and then we want to iterate through the counts and offsets to
            // sort the arrays
            for (int i = 0; i < npes - 1; ++i) {
                std::sort(global_ids.begin() + displs[i],
                          global_ids.begin() + displs[i + 1]);
            }
            std::sort(global_ids.begin() + displs[npes - 1], global_ids.end());
        }

        std::vector<T> my_partition(localSz);

        MPI_Scatterv(global_ids.data(), counts.data(), displs.data(),
                     get_mpi_type<T>(), my_partition.data(), localSz,
                     get_mpi_type<T>(), 0, commActive);

        return my_partition;
    }

    // this builds up the global connectivity map, but for each individual
    // processor we'll need to communicate the map to synchronize it, probably
    void buildOctreeConnectivity() {
        if (!m_uiIsActive) return;

        typedef uint64_t D_INT_L;

        int rank                         = this->getMPIRank();
        int npes                         = this->getMPICommSize();
        MPI_Comm commActive              = this->getMPICommunicator();

        // helpers for the nodes and e2e_map
        const ot::TreeNode *const pNodes = this->getAllElements().data();
        const unsigned int *e2e_map      = this->getE2EMapping().data();
        const unsigned int lb            = this->getElementLocalBegin();
        const unsigned int le            = this->getElementLocalEnd();

        // number of local elements on the grid
        D_INT_L localSz =
            (this->getElementLocalEnd() - this->getElementLocalBegin());

        // used to store the element counts and offsets for the global case
        std::vector<D_INT_L> ele_counts;
        std::vector<D_INT_L> ele_offsets;
        ele_offsets.resize(npes);
        ele_counts.resize(npes);

        // gather counts from all processes to build global map
        par::Mpi_Allgather(&localSz, ele_counts.data(), 1, commActive);
        ele_offsets[0] = 0;
        // this computes the offsets for each process
        for (unsigned int i = 1; i < npes; i++)
            ele_offsets[i] = ele_offsets[i - 1] + ele_counts[i - 1];

        // final total number of elements across the entire global mesh
        D_INT_L num_ele_global = ele_offsets[npes - 1] + ele_counts[npes - 1];

        // element ID storage
        D_INT_L *eid_vec       = this->createElementVector<D_INT_L>(0L, 1);
        // populate element ID vector with global IDs for the **local** elements
        for (unsigned int ele = this->getElementLocalBegin();
             ele < this->getElementLocalEnd(); ele++) {
            unsigned int lid_ele = ele - this->getElementLocalBegin();
            eid_vec[ele]         = lid_ele + ele_offsets[rank];
        }

        // read from ghost element vectors
        this->readFromGhostBeginElementVec(eid_vec, 1);
        this->readFromGhostEndElementVec(eid_vec, 1);

        // then build up and create the oct_connectivity_map
        std::vector<oct_data<D_INT_L>> oct_connectivity_map;
        oct_connectivity_map.resize(localSz);

        // for each local element, populate the map
        for (unsigned int ele = this->getElementLocalBegin();
             ele < this->getElementLocalEnd(); ele++) {
            // level
            unsigned int pl      = pNodes[ele].getLevel();
            // size of the cell
            unsigned int psz     = 1u << (m_uiMaxDepth - pl - 1);
            // local index of current element
            unsigned int lid_ele = ele - this->getElementLocalBegin();
            // number of directions
            unsigned int ndir    = this->getNumDirections();

            // global ID is then just the element ID (at 0) + global offsets
            // for our rank
            D_INT_L gid_ele =
                (ele - this->getElementLocalBegin()) + ele_offsets[rank];
            // sub-cell coordinate
            D_INT_L coord[3] = {pNodes[ele].minX() + psz,
                                pNodes[ele].minY() + psz,
                                pNodes[ele].minZ() + psz};

            if (gid_ele > num_ele_global)
                printf("ele = %06ld x = %0ld y = %06ld z = %06ld\n", gid_ele,
                       coord[0], coord[1], coord[2]);

            oct_connectivity_map[lid_ele].eid      = gid_ele;
            oct_connectivity_map[lid_ele].localid  = lid_ele;
            oct_connectivity_map[lid_ele].coord[0] = coord[0];
            oct_connectivity_map[lid_ele].coord[1] = coord[1];
            oct_connectivity_map[lid_ele].coord[2] = coord[2];
            // and set the rank to the initial rank
            oct_connectivity_map[lid_ele].rank     = rank;
            oct_connectivity_map[lid_ele].level    = pNodes[ele].getLevel();

            // this calculates which element-to-element connections it has
            for (unsigned int k = 0; k < ndir; k++) {
                D_INT_L e2e_use = e2e_map[ele * ndir + k];

                // check if it is a valid mapping
                if (e2e_use != LOOK_UP_TABLE_DEFAULT) {
                    // convert to global ID using element ID vector
                    e2e_use = eid_vec[e2e_map[ele * ndir + k]];
                    oct_connectivity_map[lid_ele].e2e[k] = e2e_use;
                } else {
                    oct_connectivity_map[lid_ele].e2e[k] =
                        LOOK_UP_TABLE_DEFAULT;
                }
            }
        }
    }

    // quick and dirty function that calculates a unique global node ID based on
    // the global element ID
    inline unsigned int globalNodeID(unsigned int global_element_id,
                                     unsigned int node_internal_pos) {
        return (global_element_id * m_uiNpE) + node_internal_pos;
    }

    std::tuple<unsigned int, unsigned int> globalAndNodeFromGlobalNodeID(
        unsigned int global_node_id) {
        return std::make_tuple(global_node_id / m_uiNpE,
                               global_node_id % m_uiNpE);
    }

    /**
     * Build Octant Connectivity on Global Scale
     *
     * Additionally builds up a local-to-global map object for easy conversion
     *
     */
    template <typename T>
    std::tuple<std::vector<oct_data<T>>, std::vector<T>, std::vector<T>,
               std::vector<T>>
    buildOctantConnectivityMap() {
        int rank                         = this->getMPIRank();
        int npes                         = this->getMPICommSize();
        MPI_Comm commActive              = this->getMPICommunicator();

        // helpers for the nodes and e2e_map
        const ot::TreeNode *const pNodes = this->getAllElements().data();
        const unsigned int *e2e_map      = this->getE2EMapping().data();
        const unsigned int lb            = this->getElementLocalBegin();
        const unsigned int le            = this->getElementLocalEnd();

        // BUILD UP OUTPUTS
        std::vector<oct_data<T>> oct_connectivity_map;
        std::vector<T> local_to_global(m_uiNumTotalElements,
                                       LOOK_UP_TABLE_DEFAULT);

        // number of local elements on the grid
        T localSz = (this->getElementLocalEnd() - this->getElementLocalBegin());

        // used to store the element counts and offsets for the global case
        std::vector<T> ele_counts;
        std::vector<T> ele_offsets;
        ele_offsets.resize(npes);
        ele_counts.resize(npes);

        // gather counts from all processes to build global map
        par::Mpi_Allgather(&localSz, ele_counts.data(), 1, commActive);
        ele_offsets[0] = 0;
        // this computes the offsets for each process
        for (unsigned int i = 1; i < npes; i++)
            ele_offsets[i] = ele_offsets[i - 1] + ele_counts[i - 1];

        // final total number of elements across the entire global mesh
        T num_ele_global = ele_offsets[npes - 1] + ele_counts[npes - 1];

        // element ID storage
        T *eid_vec = this->createElementVector<T>(LOOK_UP_TABLE_DEFAULT, 1);
        // populate element ID vector with global IDs for the **local** elements
        for (unsigned int ele = this->getElementLocalBegin();
             ele < this->getElementLocalEnd(); ele++) {
            unsigned int lid_ele = ele - this->getElementLocalBegin();
            eid_vec[ele]         = lid_ele + ele_offsets[rank];
        }

        // read from ghost element vectors (covers round-1 ghosts only)
        this->readFromGhostBeginElementVec(eid_vec, 1);
        this->readFromGhostEndElementVec(eid_vec, 1);

        // Resolve global IDs for round-2 ghost elements not covered
        // by the element ghost exchange.  Uses an Allgatherv of local
        // element coordinates to build a global lookup table.
        {
            unsigned int numMissing = 0;
            for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
                if (eid_vec[ele] == LOOK_UP_TABLE_DEFAULT) numMissing++;
            }

            if (numMissing > 0) {
                // gather all local element descriptors globally
                // each entry: (x, y, z, level, global_id) packed as
                // 5 unsigned ints
                const unsigned int PACK_SZ = 5;
                std::vector<int> allLocalSizes(npes);
                int myLocalSz = static_cast<int>(localSz);
                MPI_Allgather(&myLocalSz, 1, MPI_INT, allLocalSizes.data(),
                              1, MPI_INT, commActive);

                std::vector<int> allLocalDispls(npes, 0);
                for (int p = 1; p < npes; p++)
                    allLocalDispls[p] =
                        allLocalDispls[p - 1] + allLocalSizes[p - 1];
                int totalGlobal =
                    allLocalDispls[npes - 1] + allLocalSizes[npes - 1];

                // pack local elements
                std::vector<unsigned int> myPack(localSz * PACK_SZ);
                for (unsigned int e = this->getElementLocalBegin();
                     e < this->getElementLocalEnd(); e++) {
                    unsigned int li = e - this->getElementLocalBegin();
                    myPack[li * PACK_SZ + 0] = pNodes[e].getX();
                    myPack[li * PACK_SZ + 1] = pNodes[e].getY();
                    myPack[li * PACK_SZ + 2] = pNodes[e].getZ();
                    myPack[li * PACK_SZ + 3] = pNodes[e].getLevel();
                    myPack[li * PACK_SZ + 4] = eid_vec[e];
                }

                // scale counts and displacements for packed data
                std::vector<int> sendCounts(npes), sendDispls(npes);
                for (int p = 0; p < npes; p++) {
                    sendCounts[p] = allLocalSizes[p] * PACK_SZ;
                    sendDispls[p] = allLocalDispls[p] * PACK_SZ;
                }

                std::vector<unsigned int> allPack(totalGlobal * PACK_SZ);
                MPI_Allgatherv(myPack.data(), localSz * PACK_SZ,
                               MPI_UNSIGNED, allPack.data(),
                               sendCounts.data(), sendDispls.data(),
                               MPI_UNSIGNED, commActive);

                // build coordinate lookup: (x,y,z,level) -> global_id
                std::map<std::tuple<unsigned int, unsigned int,
                                    unsigned int, unsigned int>,
                         T>
                    coordToGlobal;
                for (int g = 0; g < totalGlobal; g++) {
                    auto key = std::make_tuple(
                        allPack[g * PACK_SZ + 0], allPack[g * PACK_SZ + 1],
                        allPack[g * PACK_SZ + 2], allPack[g * PACK_SZ + 3]);
                    coordToGlobal[key] = allPack[g * PACK_SZ + 4];
                }

                // resolve missing ghost element global IDs
                for (unsigned int ele = 0; ele < m_uiNumTotalElements;
                     ele++) {
                    if (eid_vec[ele] != LOOK_UP_TABLE_DEFAULT) continue;
                    auto key = std::make_tuple(
                        pNodes[ele].getX(), pNodes[ele].getY(),
                        pNodes[ele].getZ(), pNodes[ele].getLevel());
                    auto it = coordToGlobal.find(key);
                    if (it != coordToGlobal.end()) {
                        eid_vec[ele] = it->second;
                    }
                }
            }
        }

        oct_connectivity_map.resize(localSz);

        for (unsigned int ele = 0; ele < m_uiNumTotalElements; ele++) {
            if (eid_vec[ele] != LOOK_UP_TABLE_DEFAULT) {
                local_to_global[ele] = eid_vec[ele];
            }
        }

        // for each local element, populate the map
        for (unsigned int ele = this->getElementLocalBegin();
             ele < this->getElementLocalEnd(); ele++) {
            // level
            unsigned int pl      = pNodes[ele].getLevel();
            // size of the cell
            unsigned int psz     = 1u << (m_uiMaxDepth - pl - 1);
            // local index of current element
            unsigned int lid_ele = ele - this->getElementLocalBegin();
            // number of directions
            unsigned int ndir    = this->getNumDirections();

            // global ID is then just the element ID (at 0) + global offsets
            // for our rank
            T gid_ele =
                (ele - this->getElementLocalBegin()) + ele_offsets[rank];
            // sub-cell coordinate
            T coord[3] = {pNodes[ele].minX() + psz, pNodes[ele].minY() + psz,
                          pNodes[ele].minZ() + psz};

            if (gid_ele > num_ele_global)
                printf("ele = %06ld x = %0ld y = %06ld z = %06ld\n", gid_ele,
                       coord[0], coord[1], coord[2]);

            oct_connectivity_map[lid_ele].eid      = gid_ele;
            oct_connectivity_map[lid_ele].localid  = ele;
            oct_connectivity_map[lid_ele].coord[0] = coord[0];
            oct_connectivity_map[lid_ele].coord[1] = coord[1];
            oct_connectivity_map[lid_ele].coord[2] = coord[2];
            // and set the rank to the initial rank
            oct_connectivity_map[lid_ele].rank     = rank;
            oct_connectivity_map[lid_ele].level    = pNodes[ele].getLevel();
            oct_connectivity_map[lid_ele].flag     = pNodes[ele].getFlag();
            // 27-bit ownership mask in eOrd=2 representative layout.
            // Pre-partition value (cascade-correct on SFC mesh). Empty
            // (0u) before deriveOwnerMasksFromCascade has run; that's
            // ok because the post-partition mask path detects this and
            // falls back to cascade.
            oct_connectivity_map[lid_ele].ownerMask =
                (ele < m_uiOwnerMask.size()) ? m_uiOwnerMask[ele] : 0u;

            // canonical SFC block membership. populated pre-partition
            // from the source mesh's m_uiLocalBlockList. zero (invalid
            // bit unset) when source didn't carry block info — the
            // post-partition path detects this and falls back to the
            // legacy octree2BlockDecompositionRepartitioned algorithm.
            if (ele < m_uiBlockInfo.size()) {
                const auto& bi                                 = m_uiBlockInfo[ele];
                oct_connectivity_map[lid_ele].blkAnchorX       = bi.anchorX;
                oct_connectivity_map[lid_ele].blkAnchorY       = bi.anchorY;
                oct_connectivity_map[lid_ele].blkAnchorZ       = bi.anchorZ;
                oct_connectivity_map[lid_ele].blkAnchorLevel   = bi.anchorLevel;
                oct_connectivity_map[lid_ele].blkMeta          = bi.meta;
            } else {
                oct_connectivity_map[lid_ele].blkAnchorX       = 0u;
                oct_connectivity_map[lid_ele].blkAnchorY       = 0u;
                oct_connectivity_map[lid_ele].blkAnchorZ       = 0u;
                oct_connectivity_map[lid_ele].blkAnchorLevel   = 0u;
                oct_connectivity_map[lid_ele].blkMeta          = 0u;
            }

            // then local-to-global map updates
            // local_to_global[ele]                   = gid_ele;

            // this calculates which element-to-element connections it has
            for (unsigned int k = 0; k < ndir; k++) {
                T local_id_temp = e2e_map[ele * ndir + k];

                // check if it is a valid mapping
                if (local_id_temp != LOOK_UP_TABLE_DEFAULT) {
                    // convert to global ID using element ID vector
                    T e2e_use = eid_vec[local_id_temp];

                    oct_connectivity_map[lid_ele].e2e[k] = e2e_use;
                } else {
                    oct_connectivity_map[lid_ele].e2e[k] =
                        LOOK_UP_TABLE_DEFAULT;
                }
            }

            unsigned int lookup[NUM_CHILDREN];

            // then we use this element ID to find edge and vertex neighbors
            unsigned int counter = 0;
            for (const unsigned int dir :
                 {OCT_DIR_LEFT_DOWN, OCT_DIR_LEFT_UP, OCT_DIR_LEFT_BACK,
                  OCT_DIR_LEFT_FRONT, OCT_DIR_RIGHT_DOWN, OCT_DIR_RIGHT_UP,
                  OCT_DIR_RIGHT_BACK, OCT_DIR_RIGHT_FRONT, OCT_DIR_DOWN_BACK,
                  OCT_DIR_DOWN_FRONT, OCT_DIR_UP_BACK, OCT_DIR_UP_FRONT}) {
                getElementalEdgeNeighbors(ele, dir, lookup);
                T temp_local = lookup[3];
                if (temp_local != LOOK_UP_TABLE_DEFAULT) {
                    // find in the e2e vector for global ID
                    oct_connectivity_map[lid_ele].edgeNeighbors[counter] =
                        eid_vec[temp_local];
                }

                counter++;
            }

            counter = 0;
            for (const unsigned int dir :
                 {OCT_DIR_LEFT_DOWN_BACK, OCT_DIR_RIGHT_DOWN_BACK,
                  OCT_DIR_LEFT_UP_BACK, OCT_DIR_RIGHT_UP_BACK,
                  OCT_DIR_LEFT_DOWN_FRONT, OCT_DIR_RIGHT_DOWN_FRONT,
                  OCT_DIR_LEFT_UP_FRONT, OCT_DIR_RIGHT_UP_FRONT}) {
                getElementalVertexNeighbors(ele, dir, lookup);
                T temp_local = lookup[7];
                if (temp_local != LOOK_UP_TABLE_DEFAULT) {
                    // find in the e2e vector for global ID
                    oct_connectivity_map[lid_ele].vertexNeighbors[counter] =
                        eid_vec[temp_local];
                }
            }
        }
        ele_offsets.push_back(num_ele_global);

        return std::make_tuple(oct_connectivity_map, local_to_global,
                               ele_offsets, ele_counts);
    }


    template <typename T>
    inline oct_data<T> *findOctDataByGlobalID(
        std::vector<oct_data<T>> &oct_connectivity_data,
        const unsigned int eID) {
        oct_data<T> *oct = nullptr;

        for (auto &o : oct_connectivity_data) {
            if (o.eid == eID) {
                oct = &o;
                break;
            }
        }

        if (oct == nullptr) {
            std::cerr << "ERROR: couldn't find requested data in getting "
                         "element edge neighbors new partition! EID= "
                      << eID << std::endl;
            exit(-1);
        }

        return oct;
    }

    template <typename T>
    void getElementFaceNeighborsNewPartition(
        std::vector<oct_data<T>> &oct_connectivity_data, const unsigned int eID,
        const unsigned int dir, unsigned int *lookup) {
        lookup[0]        = eID;
        lookup[1]        = LOOK_UP_TABLE_DEFAULT;
        oct_data<T> *oct = findOctDataByGlobalID(oct_connectivity_data, eID);
        unsigned level   = oct->level;

        lookup[1]        = oct->e2e[dir];

#if 0
        // included in the original function, but since this function is only used to *identify* faces we need information from, it can be ignored
        if (lookup[1] != LOOK_UP_TABLE_DEFAULT) {
            oct_data<T> *oct2 =
                findOctDataByGlobalID(oct_connectivity_data, lookup[i]);
            if (oct2->level > level) {
                lookup[1] = LOOK_UP_TABLE_DEFAULT;
            }
        }
#endif
    }

    template <typename T>
    void getElementEdgeNeighborsNewPartition(
        std::vector<oct_data<T>> &oct_connectivity_data, const unsigned int eID,
        const unsigned int dir, unsigned int *lookup) {
        // oct_connectivity_data should be sorted
        // remember though that this is based on GLOBALs
        lookup[0]        = eID;
        lookup[1]        = LOOK_UP_TABLE_DEFAULT;
        lookup[2]        = LOOK_UP_TABLE_DEFAULT;
        lookup[3]        = LOOK_UP_TABLE_DEFAULT;

        oct_data<T> *oct = findOctDataByGlobalID(oct_connectivity_data, eID);
        unsigned level   = oct->level;

        // now we can actually find the data
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

        // we should be good here

        lookup[1] = oct->e2e[dir1];

        for (unsigned int i = 0; i < 2; i++) {
            if (lookup[i] != LOOK_UP_TABLE_DEFAULT) {
                // need to look up based on eid
                oct_data<T> *oct2 =
                    findOctDataByGlobalID(oct_connectivity_data, lookup[i]);
                lookup[i + 2] = oct2->e2e[dir2];
            }
        }

#if 0
        // included in the original function, but since this function is only used to *identify* edges we need information from, it can be ignored
        for (unsigned int i = 1; i < 4; i++) {
            if (lookup[i] != LOOK_UP_TABLE_DEFAULT) {
                oct_data<T> *oct2 =
                    findOctDataByGlobalID(oct_connectivity_data, lookup[i]);
                if (oct2->level > level) {
                    lookup[i] = LOOK_UP_TABLE_DEFAULT;
                }
            }
        }
#endif
    }

    template <typename T>
    void getElementVertexNeighborsNewPartition(
        std::vector<oct_data<T>> &oct_connectivity_data, const unsigned int eID,
        const unsigned int dir, unsigned int *lookup) {
        // expects lookup to be of size 8

        lookup[0] = eID;
        for (unsigned int i = 1; i < NUM_CHILDREN; i++)
            lookup[i] = LOOK_UP_TABLE_DEFAULT;

        unsigned int dir1, dir2, dir3;
        oct_data<T> *oct = findOctDataByGlobalID(oct_connectivity_data, eID);
        unsigned level   = oct->level;

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

        lookup[1] = oct->e2e[dir1];

        // then go for other areas
        for (unsigned int i = 0; i < 2; i++) {
            if (lookup[i] != LOOK_UP_TABLE_DEFAULT) {
                // need to look up based on eid
                oct_data<T> *oct2 =
                    findOctDataByGlobalID(oct_connectivity_data, lookup[i]);
                lookup[i + 2] = oct2->e2e[dir2];
            }
        }

        // then update the last two
        for (unsigned int i = 0; i < 4; i++) {
            if (lookup[i] != LOOK_UP_TABLE_DEFAULT) {
                oct_data<T> *oct2 =
                    findOctDataByGlobalID(oct_connectivity_data, lookup[i]);
                lookup[i + 4] = oct2->e2e[dir3];
            }
        }

#if 0
        // checks based on level, not necessary for this function due to it only
        // being used to find potential elements needed for communication
        for (unsigned int i = 1; i < NUM_CHILDREN; i++) {
            if (lookup[i] != LOOK_UP_TABLE_DEFAULT) {
                oct_data<T> *oct2 =
                    findOctDataByGlobalID(oct_connectivity_data, lookup[i]);
                if (oct2->level > level) {
                    lookup[i] = LOOK_UP_TABLE_DEFAULT;
                }
            }
        }
#endif
    }

    void setPartitioningMethod(PartitioningOptions part_method) {
        m_partitionOption = part_method;
    }
    PartitioningOptions getPartitioningMethod() { return m_partitionOption; }

    void dumpOctDataParallel(std::vector<oct_element> my_oct,
                             fastpart_uint_t *ele_offsets,
                             fastpart_uint_t *ele_counts,
                             fastpart_uint_t localSz, const char *fprefix) {
        int rank            = this->getMPIRank();
        int npes            = this->getMPICommSize();
        MPI_Comm commActive = this->getMPICommunicator();

        fastpart_uint_t num_ele_global =
            ele_offsets[npes - 1] + ele_counts[npes - 1];

        MPI_Datatype arraytype;
        MPI_Offset disp         = sizeof(oct_element) * ele_offsets[rank];
        fastpart_uint_t N_BYTES = sizeof(oct_element) * (int)localSz;

        MPI_Type_contiguous(N_BYTES, MPI_BYTE, &arraytype);
        MPI_Type_commit(&arraytype);

        char fname[256];
        sprintf(fname, "%s_%ld.oct", fprefix, num_ele_global);

        MPI_File fh;
        MPI_File_open(commActive, fname, MPI_MODE_CREATE | MPI_MODE_RDWR,
                      MPI_INFO_NULL, &fh);
        MPI_File_set_view(fh, disp, MPI_BYTE, arraytype, "native",
                          MPI_INFO_NULL);
        MPI_File_write(fh, my_oct.data(), N_BYTES, MPI_BYTE, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);

        this->waitActive();
    }

    template <typename T>
    std::vector<T> createGlobalE2EMap(
        std::vector<oct_data<T>> &oct_connectivity_map,
        const std::vector<T> &ele_offsets, const std::vector<T> &ele_counts) {
        int rank            = this->getMPIRank();
        int npes            = this->getMPICommSize();
        MPI_Comm commActive = this->getMPICommunicator();

        int local_size      = m_uiNumLocalElements * 6;
        std::vector<T> local_e2e_slice(local_size);

        unsigned int counter = 0;
        for (const auto &oct : oct_connectivity_map) {
            for (int i = 0; i < 6; ++i) {
                local_e2e_slice[counter * 6 + i] = oct.e2e[i];
            }
            counter++;
        }

        std::vector<int> counts(ele_counts.size());
        for (int i = 0; i < ele_counts.size(); ++i) {
            counts[i] = ele_counts[i] * 6;
        }
        // REMEMBER: ele_offsets is one larger, and the last one is the total
        // amount
        std::vector<int> displ(ele_offsets.size());
        for (int i = 0; i < ele_offsets.size(); ++i) {
            displ[i] = ele_offsets[i] * 6;
        }

        // global size is just the last one here
        int global_size = displ.back();
        std::vector<T> global_e2e(global_size);

        par::Mpi_Allgatherv(local_e2e_slice.data(), local_size,
                            global_e2e.data(), counts.data(), displ.data(),
                            commActive);

        return global_e2e;
    }



    void repartitionMeshGlobal(bool do_block_creation    = true,
                               bool do_fastpart_filesave = false,
                               std::string fileprefix    = "octtree");

    template <typename T>
    std::vector<oct_data<T>> getOctDataFromOtherProcesses(
        std::vector<oct_data<T>> &oct_connectivity_map,
        const std::vector<T> &ele_offsets, const std::vector<T> &ele_counts,
        std::vector<T> &data_to_fetch, bool set_target_rank = true);

    template <typename T>
    std::tuple<std::vector<T>, std::vector<T>> removeDuplicatesSameOrder(
        std::vector<T> &vec, std::vector<T> &vec2) {
        // I don't like this, but it gets the job done...
        std::vector<T> result;
        std::vector<T> result2;
        std::unordered_set<T> seen;

        size_t counter = 0;
        for (T element : vec) {
            if (seen.find(element) == seen.end()) {
                // Element not seen before
                result.push_back(element);
                result2.push_back(vec2[counter]);
                seen.insert(element);
            }
            counter++;
        }
        return std::make_tuple(result, result2);
    }

    template <typename T, typename N>
    std::vector<N> convertVectorType(const std::vector<T> &input) {
        // static assertion helps make sure the compiler never tries to use this
        // with non-arithmetic values, i.e. ints (signed/unsigned) and
        // floats/doubles
        static_assert(std::is_arithmetic_v<T>,
                      "Input type in vector conversion must be integer or "
                      "floating point.");

        static_assert(std::is_arithmetic_v<N>,
                      "Output type in vector conversion must be integer or "
                      "floating point.");

        // NOTE: floating point to integers uses std::round! it doesn't just
        // truncate

        std::vector<N> output(input.size());

        std::transform(input.begin(), input.end(), output.begin(), [](T val) {
            if constexpr (std::is_integral_v<N>) {
                // float to integer rounding
                if constexpr (std::is_floating_point_v<T>) {
                    val = std::round(val);
                }
                if constexpr (std::is_unsigned_v<N> && std::is_signed_v<T>) {
                    return static_cast<N>(val < 0 ? std::abs(val) : val);
                }
            }
            return static_cast<N>(val);
        });
        return output;
    }

    template <typename T>
    inline int checkMismatch(const T &self_val, const T &other_val,
                             const std::string &field) {
        if constexpr (is_vector<T>::value) {
            // check size comparison first
            if (self_val.size() != other_val.size()) {
                std::cerr << "ERROR (rank " << this->m_uiGlobalRank
                          << "): " << field
                          << " size mismatch! self: " << self_val.size()
                          << " vs other: " << other_val.size() << std::endl;
                return -1;
            }
            for (size_t i = 0; i < self_val.size(); ++i) {
                if (self_val[i] != other_val[i]) {
                    std::cerr << "ERROR (rank " << this->m_uiGlobalRank
                              << "): " << field << " mismatch at index " << i
                              << "! self: " << self_val[i]
                              << " vs other: " << other_val[i] << std::endl;
                    return -1;
                }
            }
        } else {
            if (self_val != other_val) {
                std::cerr << "ERROR (rank " << this->m_uiGlobalRank
                          << "): " << field << " mismatch! self: " << self_val
                          << " vs other: " << other_val << std::endl;
                return -1;
            }
        }
        return 0;
    }

    /**
     * @brief This is a "debug" test function that lets us directly compare
     * various objects together
     */
    int compareCriticalMeshObjects(const Mesh *other) {
        if (!this->m_uiIsActive) return 0;

        bool failed = false;

        // test the m_uiE2EMapping, CG_DG, etc
        if (checkMismatch(this->m_uiE2EMapping, other->getE2EMapping(),
                          "E2EMapping (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiE2NMapping_CG, other->getE2NMapping(),
                          "E2NMapping_CG (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiE2NMapping_DG, other->getE2NMapping_DG(),
                          "E2NMapping_DG (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiCG2DG, other->getCG2DGMap(),
                          "CG2DGMap (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiDG2CG, other->getDG2CGMap(),
                          "DG2CGMap (vec)") != 0) {
            failed = true;
        }

        // number of elements stored
        if (checkMismatch(this->m_uiNumLocalElements,
                          other->getNumLocalMeshElements(),
                          "NumLocalElements") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNumPreGhostElements,
                          other->getNumPreGhostElements(),
                          "NumPreGhostElements") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNumPostGhostElements,
                          other->getNumPostGhostElements(),
                          "NumPostGhostElements") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNumTotalElements,
                          other->getNumTotalELements(),
                          "NumTotalElements") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNumActualNodes, other->getNumActualNodes(),
                          "NumActualNodes") != 0) {
            failed = true;
        }
        // mesh element offsets...
        if (checkMismatch(this->m_uiElementPreGhostBegin,
                          other->getElementPreGhostBegin(),
                          "ElementPreGhostBegin") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiElementPreGhostEnd,
                          other->getElementPreGhostEnd(),
                          "ElementPreGhostEnd") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiElementLocalBegin,
                          other->getElementLocalBegin(),
                          "ElementLocalBegin") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiElementLocalEnd,
                          other->getElementLocalEnd(),
                          "ElementLocalEnd") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiElementPostGhostBegin,
                          other->getElementPostGhostBegin(),
                          "ElementPostGhostBegin") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiElementPostGhostEnd,
                          other->getElementPostGhostEnd(),
                          "ElementPostGhostEnd") != 0) {
            failed = true;
        }

        // node element offsets
        if (checkMismatch(this->m_uiNodePreGhostBegin,
                          other->getNodePreGhostBegin(),
                          "NodePreGhostBegin") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNodePreGhostEnd,
                          other->getNodePreGhostEnd(),
                          "NodePreGhostEnd") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNodeLocalBegin, other->getNodeLocalBegin(),
                          "NodeLocalBegin") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNodeLocalEnd, other->getNodeLocalEnd(),
                          "NodeLocalEnd") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNodePostGhostBegin,
                          other->getNodePostGhostBegin(),
                          "NodePostGhostBegin") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiNodePostGhostEnd,
                          other->getNodePostGhostEnd(),
                          "NodePostGhostEnd") != 0) {
            failed = true;
        }

        // send counts
        if (checkMismatch(this->m_uiSendNodeCount, other->getNodalSendCounts(),
                          "SendNodeCount (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiRecvNodeCount, other->getNodalRecvCounts(),
                          "RecvNodeCount (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiSendNodeOffset,
                          other->getNodalSendOffsets(),
                          "SendNodeOffset (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiRecvNodeOffset,
                          other->getNodalRecvOffsets(),
                          "RecvNodeOffset (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiScatterMapActualNodeSend,
                          other->getSendNodeSM(),
                          "SendNodeScatterMap (vec)") != 0) {
            failed = true;
        }
        if (checkMismatch(this->m_uiScatterMapActualNodeRecv,
                          other->getRecvNodeSM(),
                          "RecvNodeScatterMap (vec)") != 0) {
            failed = true;
        }

        // TODO: scattermap data!

        if (failed) return -1;
        return 0;
    }
};

// LocalElementRange iterator definitions. dereferences via the explicit
// ID list when populated, otherwise reads the [LB, LE) range identity.
inline unsigned int Mesh::LocalElementRange::iterator::operator*() const {
    const auto& ids = m_->m_uiLocalElementIds;
    if (!ids.empty()) {
        return ids[pos_];
    }
    return m_->m_uiElementLocalBegin + (unsigned int)pos_;
}
inline Mesh::LocalElementRange::iterator
Mesh::LocalElementRange::begin() const {
    return iterator(m_mesh_, 0);
}
inline Mesh::LocalElementRange::iterator
Mesh::LocalElementRange::end() const {
    return iterator(m_mesh_, size());
}
inline std::size_t Mesh::LocalElementRange::size() const {
    const auto& ids = m_mesh_->m_uiLocalElementIds;
    if (!ids.empty()) return ids.size();
    if (m_mesh_->m_uiElementLocalEnd < m_mesh_->m_uiElementLocalBegin) return 0;
    return (std::size_t)(m_mesh_->m_uiElementLocalEnd
                         - m_mesh_->m_uiElementLocalBegin);
}

template <>
inline void Mesh::init<WaveletDA::LoopType ::ALL>() {
    m_uiEL_i = m_uiElementPreGhostBegin;
}

template <>
inline void Mesh::init<WaveletDA::INDEPENDENT>() {
    m_uiEL_i = m_uiElementLocalBegin;
}

template <>
inline void Mesh::init<WaveletDA::DEPENDENT>() {
    m_uiEL_i = m_uiElementPreGhostBegin;
}

template <>
inline bool Mesh::nextAvailable<WaveletDA::ALL>() {
    return (m_uiEL_i < m_uiElementPostGhostEnd);
}

template <>
inline bool Mesh::nextAvailable<WaveletDA::INDEPENDENT>() {
    return (m_uiEL_i < m_uiElementLocalEnd);
}
template <>

inline bool Mesh::nextAvailable<WaveletDA::DEPENDENT>() {
    return (m_uiEL_i < m_uiElementPreGhostEnd) ||
           ((m_uiEL_i > m_uiElementLocalEnd) &&
            m_uiEL_i < m_uiElementPostGhostEnd);
}

template <>
inline void Mesh::next<WaveletDA::ALL>() {
    // interpolation should come here.
    m_uiEL_i++;
}

template <>
inline void Mesh::next<WaveletDA::INDEPENDENT>() {
    // interpolation should come here.
    m_uiEL_i++;
}

template <>
inline void Mesh::next<WaveletDA::DEPENDENT>() {
    // interpolation should come here.
    m_uiEL_i++;
    if (m_uiEL_i == m_uiElementPreGhostEnd)
        m_uiEL_i = m_uiElementPostGhostBegin;
}

inline const ot::TreeNode &Mesh::currentOctant() {
    return m_uiAllElements[m_uiEL_i];
}

inline unsigned int Mesh::currentIndex() { return m_uiEL_i; }

inline void Mesh::currentElementNeighbourIndexList(unsigned int *neighList) {
    if (!m_uiIsActive) return;

    for (unsigned int k = 0; k < m_uiNumDirections; k++)
        neighList[k] = m_uiE2EMapping[m_uiEL_i * m_uiNumDirections + k];
}

inline void Mesh::currentElementNodeList(unsigned int *nodeList) {
    if (!m_uiIsActive) return;

    for (unsigned int k = 0; k < m_uiNpE; k++) {
        nodeList[k] = m_uiE2NMapping_CG[m_uiEL_i * m_uiNpE + k];
    }
}

inline void Mesh::currentElementNodeList_DG(unsigned int *nodeList) {
    if (!m_uiIsActive) return;

    for (unsigned int k = 0; k < m_uiNpE; k++) {
        nodeList[k] = m_uiE2NMapping_DG[m_uiEL_i * m_uiNpE + k];
    }
}

inline void Mesh::parent2ChildInterpolation(const double *in, double *out,
                                            unsigned int cnum,
                                            unsigned int dim) const {
    dendro::timer::t_unzip_p2c.start();
    if (dim == 3)
        m_uiRefEl.I3D_Parent2Child(in, out, cnum);
    else if (dim == 2)
        m_uiRefEl.I2D_Parent2Child(in, out, cnum);
    else if (dim == 1)
        m_uiRefEl.I1D_Parent2Child(in, out, cnum);
    dendro::timer::t_unzip_p2c.stop();
}

inline void Mesh::child2ParentInterpolation(const double *in, double *out,
                                            unsigned int cnum,
                                            unsigned int dim) const {
    if (dim == 3)
        m_uiRefEl.I3D_Child2Parent(in, out, cnum);
    else if (dim == 2)
        m_uiRefEl.I2D_Child2Parent(in, out, cnum);
    else if (dim == 1)
        m_uiRefEl.I1D_Child2Parent(in, out, cnum);
}

inline bool Mesh::computeOveralppingNodes(const ot::TreeNode &parent,
                                          const ot::TreeNode &child, int *idx,
                                          int *idy, int *idz) {
    unsigned int Lp = 1u << (m_uiMaxDepth - parent.getLevel());
    unsigned int Lc = 1u << (m_uiMaxDepth - child.getLevel());
    // intilize the mapping to -1. -1 denotes that mapping is not defined
    // for given k value.

    unsigned int dp, dc;
    dp = (m_uiElementOrder);
    dc = m_uiElementOrder;

    assert(Lp % dp == 0);
    assert(Lc % dc == 0);

    for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++) {
        idx[k] = -1;
        idy[k] = -1;
        idz[k] = -1;
    }
    bool state  = false;
    bool stateX = false;
    bool stateY = false;
    bool stateZ = false;
    if (parent == child) {
        for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++) {
            idx[k] = k;
            idy[k] = k;
            idz[k] = k;
        }
        return true;
    } else if (parent.isAncestor(child)) {
        /*if((((child.getX()-parent.getX())*m_uiElementOrder)%Lp) ||
           (((child.getY()-parent.getY())*m_uiElementOrder)%Lp) ||
           (((child.getZ()-parent.getZ())*m_uiElementOrder)%Lp)) return
           false; else*/
        {
            unsigned int index[3];
            for (unsigned int k = 0; k < (m_uiElementOrder + 1); k++) {
                index[0] = (m_uiElementOrder + 1);
                index[1] = (m_uiElementOrder + 1);
                index[2] = (m_uiElementOrder + 1);

                if (!(((child.getX() - parent.getX()) * dp * dc + k * Lc * dp) %
                      (Lp * dc)))
                    index[0] =
                        ((child.getX() - parent.getX()) * dp * dc +
                         k * Lc * dp) /
                        (Lp *
                         dc);  //((child.getX()-parent.getX())*m_uiElementOrder
                               //+ k*Lc)/Lp;

                if (!(((child.getY() - parent.getY()) * dp * dc + k * Lc * dp) %
                      (Lp * dc)))
                    index[1] =
                        ((child.getY() - parent.getY()) * dp * dc +
                         k * Lc * dp) /
                        (Lp *
                         dc);  //((child.getY()-parent.getY())*m_uiElementOrder
                               //+ k*Lc)/Lp;

                if (!(((child.getZ() - parent.getZ()) * dp * dc + k * Lc * dp) %
                      (Lp * dc)))
                    index[2] =
                        ((child.getZ() - parent.getZ()) * dp * dc +
                         k * Lc * dp) /
                        (Lp *
                         dc);  //((child.getZ()-parent.getZ())*m_uiElementOrder
                               //+ k*Lc)/Lp;

                if (!stateX && index[0] < (m_uiElementOrder + 1)) stateX = true;

                if (!stateY && index[1] < (m_uiElementOrder + 1)) stateY = true;

                if (!stateZ && index[2] < (m_uiElementOrder + 1)) stateZ = true;

                if (index[0] < (m_uiElementOrder + 1)) {
                    idx[k] = index[0];
                    assert((parent.getX() + idx[k] * Lp / dp) ==
                           (child.getX() + k * Lc / dc));
                }
                if (index[1] < (m_uiElementOrder + 1)) {
                    idy[k] = index[1];
                    assert((parent.getY() + idy[k] * Lp / dp) ==
                           (child.getY() + k * Lc / dc));
                }
                if (index[2] < (m_uiElementOrder + 1)) {
                    idz[k] = index[2];
                    assert((parent.getZ() + idz[k] * Lp / dp) ==
                           (child.getZ() + k * Lc / dc));
                }
            }
            state = stateX & stateY & stateZ;
            return state;
        }
    } else {
        return false;
    }
}

}  // namespace ot

#include "mesh.tcc"
#include "meshE2NUtils.tcc"
