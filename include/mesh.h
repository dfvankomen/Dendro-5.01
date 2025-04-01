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
#include <fdCoefficient.h>
#include <sys/types.h>

#include <cstdint>
#include <numeric>
#include <ostream>
#include <random>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "TreeNode.h"
#include "asyncExchangeContex.h"
#include "block.h"
#include "dendro.h"
#include "dendroProfileParams.h"  // only need to profile unzip_asyn for bssn. remove this header file later.
#include "key.h"
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

    oct_data() {
        e2e[0] = LOOK_UP_TABLE_DEFAULT;
        e2e[1] = LOOK_UP_TABLE_DEFAULT;
        e2e[2] = LOOK_UP_TABLE_DEFAULT;
        e2e[3] = LOOK_UP_TABLE_DEFAULT;
        e2e[4] = LOOK_UP_TABLE_DEFAULT;
        e2e[5] = LOOK_UP_TABLE_DEFAULT;
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
    int blocklengths[] = {1, 1, 1, 1, 3, 6, 1, 1, NUM_PTS_ELE, 1};
    MPI_Aint offsets[10];
    MPI_Datatype types[10];

    types[0] = MPI_UINT32_T;  // 'rank' is always uint32_t
    types[1] = MPI_UINT32_T;  // 'target_rank' is always uint32_t
    types[2] = types[3] = types[4] = types[5] = types[6] = get_mpi_type<T>();
    types[7]                                             = MPI_UNSIGNED;
    types[8]                                             = get_mpi_type<T>();
    types[9]                                             = MPI_CXX_BOOL;

    // get the offsets for each member of the struct
    offsets[0] = offsetof(oct_data<T>, rank);
    offsets[1] = offsetof(oct_data<T>, trank);
    offsets[2] = offsetof(oct_data<T>, eid);
    offsets[3] = offsetof(oct_data<T>, localid);
    offsets[4] = offsetof(oct_data<T>, coord);
    offsets[5] = offsetof(oct_data<T>, e2e);
    offsets[6] = offsetof(oct_data<T>, level);
    offsets[7] = offsetof(oct_data<T>, flag);
    offsets[8] = offsetof(oct_data<T>, e2n_dg);
    offsets[9] = offsetof(oct_data<T>, isGhostTwo);

    MPI_Type_create_struct(10, blocklengths, offsets, types,
                           &octdata_mpi_datatype);
    MPI_Type_commit(&octdata_mpi_datatype);

    return octdata_mpi_datatype;
}

enum PartitioningOptions {
    NoPartition,
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

    /**
     * @brief computes the face to element map.
     * needs to be called after e2e and e2n maps has built.
     **/
    void buildF2EMap();

    /*** @brief: perform block dependancy flags flag block unzip depends on
     * ghost nodes.  */
    void flagBlockGhostDependancies();

    /**@brief: returns if the block setup has performed or not*/
    inline bool isBlockSetep() { return m_uiIsBlockSetup; }

    /**@brief: returns if the scatter map typed set*/
    inline SM_TYPE getScatterMapType() { return m_uiScatterMapType; }

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

        // read from ghost element vectors
        this->readFromGhostBeginElementVec(eid_vec, 1);
        this->readFromGhostEndElementVec(eid_vec, 1);

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
        }
        ele_offsets.push_back(num_ele_global);

        return std::make_tuple(oct_connectivity_map, local_to_global,
                               ele_offsets, ele_counts);
    }

    template <typename T>
    void createLocalToGlobalE2N(std::vector<oct_data<T>> &oct_connectivity_map,
                                std::vector<T> &ele_local_to_global) {
        int rank            = this->getMPIRank();
        int npes            = this->getMPICommSize();
        MPI_Comm commActive = this->getMPICommunicator();
        // m_uiEL_i is just a counter to manage loop access over elements

        // the function currentElementNeighbourIndexList iterates from 0 to
        // m_uiNumDirections accessed by m_uiEL_i * m_uiNumDirections + k

        // the m_uiE2NMapping_CG vector has an indexing pattern of m_uiEL_i *
        // m_uiNpE + k (where k is 0-m_uiNpE from currentLementNodeList() ) the
        // m_uiE2NMapping_DG vector does the same!

        // size of m_uiE2NMapping_CG is m_uiAllNodes.size() * m_uiNpE

        // -----------------
        // CONTINOUS GALERKIN

        // cg_local_to_global
        std::map<T, T> cg_local_to_global;

        // calculate how many there will be for this particular array
        T localSzCG =
            m_uiNpE * m_uiElementLocalEnd - m_uiNpE * m_uiElementLocalBegin;

        // store element counts and offsets for global case, NOTE: these will be
        // the same for DG and CG
        std::vector<T> node_counts_CG;
        std::vector<T> node_offsets_CG;
        node_counts_CG.resize(npes);
        node_offsets_CG.resize(npes);

        // gather counts from all processes to build global map
        par::Mpi_Allgather(&localSzCG, node_counts_CG.data(), 1, commActive);
        node_offsets_CG[0] = 0;
        for (unsigned int i = 1; i < npes; i++) {
            node_offsets_CG[i] = node_offsets_CG[i - 1] + node_counts_CG[i - 1];
        }

        T num_node_CG_global =
            node_offsets_CG[npes - 1] + node_counts_CG[npes - 1];
        T *nodeid_vec_CG = this->createCGVector<T>((T)(0), 1);

        for (unsigned int node_i = m_uiNodeLocalBegin;
             node_i < m_uiNodeLocalEnd; node_i++) {
            unsigned int lid_node = node_i - m_uiNodeLocalBegin;
            // calculate a global value
            nodeid_vec_CG[node_i] = lid_node + node_offsets_CG[rank];
        }

        // create a vector the same way as OCT_SHARED_NODES on
        // DVec.create_vector, this will allow us to determine global IDs based
        // on values requested
        size_t vec_size = 1 * this->getDegOfFreedom();
        std::vector<T> nodeid_vec_CG_use(vec_size, LOOK_UP_TABLE_DEFAULT);

        // okay, determine unique local nodes
        unsigned int nodeLookUp_CG;
        unsigned int nodeLookUp_DG;
        double x, y, z, len;
        unsigned int ownerID, ii_x, jj_y, kk_z;
        unsigned int extract_id;
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ++ele) {
            for (unsigned int k = 0; k < (m_uiElementOrder + 1); ++k) {
                for (unsigned int j = 0; j < (m_uiElementOrder + 1); ++j) {
                    for (unsigned int i = 0; (i < m_uiElementOrder + 1); ++i) {
                        extract_id = ele * m_uiNpE +
                                     k * (m_uiElementOrder + 1) *
                                         (m_uiElementOrder + 1) +
                                     j * (m_uiElementOrder + 1) + i;
                        nodeLookUp_CG = m_uiE2NMapping_CG[extract_id];

                        if (nodeLookUp_CG >= m_uiNodeLocalBegin &&
                            nodeLookUp_CG < m_uiNodeLocalEnd) {
                            nodeLookUp_DG = m_uiE2NMapping_DG[extract_id];

                            // we can also calculate the X, Y, Z position here
                            // too!
                            dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);

                            len = 1u << (m_uiMaxDepth -
                                         m_uiAllElements[ele].getLevel());

                            x   = m_uiAllElements[ele].getX() +
                                ii_x * (len / m_uiElementOrder);
                            y = m_uiAllElements[ele].getY() +
                                jj_y * (len / m_uiElementOrder);
                            z = m_uiAllElements[ele].getZ() +
                                kk_z * (len / m_uiElementOrder);

                            nodeid_vec_CG_use[nodeLookUp_CG] =
                                extract_id - (m_uiElementLocalBegin * m_uiNpE) +
                                node_offsets_CG[rank];

                            if (nodeLookUp_DG < m_uiNodeLocalBegin &&
                                nodeLookUp_DG >= m_uiNodeLocalEnd) {
                                std::cout
                                    << rank
                                    << ": NODE_LOOKUP_DG value out of bounds??"
                                    << std::endl;
                            }
#if 0
                            if (nodeLookUp_DG > vec_size) {
                                std::cout << rank
                                          << ": NODE_LOOKUP_DG value TOO BIG"
                                          << std::endl;
                            }
#endif
                        }
                    }
                }
            }
        }

        this->readFromGhostBegin(nodeid_vec_CG_use.data(), 1);
        this->readFromGhostEnd(nodeid_vec_CG_use.data(), 1);

        // so now we should be able to create a new E2N CG:
        std::vector<T> e2n_cg_new(m_uiE2NMapping_CG.size(),
                                  LOOK_UP_TABLE_DEFAULT);

        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ++ele) {
            for (unsigned int k = 0; k < (m_uiElementOrder + 1); ++k) {
                for (unsigned int j = 0; j < (m_uiElementOrder + 1); ++j) {
                    for (unsigned int i = 0; (i < m_uiElementOrder + 1); ++i) {
                        extract_id = ele * m_uiNpE +
                                     k * (m_uiElementOrder + 1) *
                                         (m_uiElementOrder + 1) +
                                     j * (m_uiElementOrder + 1) + i;
                        e2n_cg_new[extract_id] =
                            nodeid_vec_CG_use[m_uiE2NMapping_CG[extract_id]];

                        if (e2n_cg_new[extract_id] == LOOK_UP_TABLE_DEFAULT) {
                            std::cout << rank << ": ERROR: INVALID VALUE FOUND!"
                                      << std::endl;
                        }
                    }
                }
            }
        }

        // now get the global values for e2n_dg, for our m_uiElementLocal values
        std::vector<T> e2n_dg_new(m_uiE2NMapping_DG.size(),
                                  LOOK_UP_TABLE_DEFAULT);

        // NOW THE E2N_CG MAP IS UPDATED WITH GLOBAL IDS
        unsigned int new_dg_val;
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ++ele) {
            long int id_for_global = ele - m_uiElementLocalBegin;
            for (unsigned int k = 0; k < (m_uiElementOrder + 1); ++k) {
                for (unsigned int j = 0; j < (m_uiElementOrder + 1); ++j) {
                    for (unsigned int i = 0; (i < m_uiElementOrder + 1); ++i) {
                        extract_id = ele * m_uiNpE +
                                     k * (m_uiElementOrder + 1) *
                                         (m_uiElementOrder + 1) +
                                     j * (m_uiElementOrder + 1) + i;

                        nodeLookUp_DG = m_uiE2NMapping_DG[extract_id];
                        dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);

                        // convert the ownerID to its global ID
                        if (ele_local_to_global[ownerID] !=
                            LOOK_UP_TABLE_DEFAULT) {
                            id_for_global = ele_local_to_global[ownerID];
                        } else {
#if 0
                            std::cout << rank << ": COULDN'T FIND LOCAL ID "
                                      << ownerID << " in map!" << std::endl;
#endif
                            continue;
                        }

                        new_dg_val = eijk2dg(id_for_global, ii_x, jj_y, kk_z);

                        e2n_dg_new[extract_id] = new_dg_val;
                    }
                }
            }
        }

        // with all of this information now, we can build up a vector of data
        for (unsigned int ele = m_uiElementLocalBegin;
             ele < m_uiElementLocalEnd; ++ele) {
            // update the data in oct_connectivity_map's e2n_cg and e2n_dg

            for (unsigned int i = 0; i < m_uiNpE; ++i) {
                // oct_connectivity_map[ele - m_uiElementLocalBegin].e2n_cg[i] =
                //     e2n_cg_new[ele * m_uiNpE + i];
                oct_connectivity_map[ele - m_uiElementLocalBegin].e2n_dg[i] =
                    e2n_dg_new[ele * m_uiNpE + i];
            }
        }

        // return node_data_by_ele;
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

    void repartitionMeshGlobal() {
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

        auto [oct_connectivity_map, local_to_global, ele_offsets, ele_counts] =
            buildOctantConnectivityMap<D_INT_L>();

        // ele_offsets and ele_counts will help us figure out which process
        // belongs to which

        // then figure out node global IDs to get a local-to-global mapping for
        // them

        // E2E, E2E_DG, and E2E_CG have mesh connectivity and hanging node
        // information
        createLocalToGlobalE2N<D_INT_L>(oct_connectivity_map, local_to_global);

        std::vector<D_INT_L> my_partition;

        // auto my_partition = randomPartitioningSimple(oct_connectivity_map);
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
            }

            // vtx_dist is a prefix scan of the element count for each MPI node
            fastpart_uint_t *vtx_dist = static_cast<fastpart_uint_t *>(
                malloc(ele_offsets.size() * sizeof(fastpart_uint_t)));

            for (unsigned int rk = 0; rk < ele_offsets.size(); rk++) {
                vtx_dist[rk] = ele_offsets[rk];
            }

            fastpart_uint_t *parts = static_cast<fastpart_uint_t *>(
                malloc(oct_connectivity_map.size() * sizeof(fastpart_uint_t)));

            fastpart_partgraph_octree(vtx_dist, temp_oct_data.data(), parts,
                                      &commActive);

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
            free(vtx_dist);
        }

        std::cout << rank << ": ORIGINAL PARTITION SIZE - "
                  << oct_connectivity_map.size() << " NEW PARTITION SIZE - "
                  << my_partition.size() << std::endl;

        auto new_oct_connectivity_map = getOctDataFromOtherProcesses(
            oct_connectivity_map, ele_offsets, ele_counts, my_partition);

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
        std::sort(new_oct_connectivity_map.begin(),
                  new_oct_connectivity_map.end(),
                  [](const oct_data<D_INT_L> &o1, const oct_data<D_INT_L> &o2) {
                      return o1.trank < o2.trank;
                  });

        size_t newLocalBegin = LOOK_UP_TABLE_DEFAULT;
        size_t newNumEle;

        std::set<D_INT_L> global_ids_needed_for_e2e;
        std::vector<D_INT_L> global_ids_needed_for_e2e_vec;
        std::vector<oct_data<D_INT_L>> collected_ghost_elements;
        std::vector<oct_data<D_INT_L>> remaining_data;
        std::vector<D_INT_L> remove_these_ids;
        std::vector<D_INT_L> post_first_round_comms_ids;
        // 6 rounds of communication are what we want:
        // 1- identify face neighbors of our partition
        // 2- identify edge neighbors of our partition
        // 3- identify vertex neighbors of our partition
        // 4- identify face neighbors for level-2 ghosts (ownership concerns!)
        // 5- identify edge neighbors for level-2 ghosts (ownership concerns!)
        // 6- identify vertex neighbors for level-2 ghosts (ownership concerns!)
        // 7- find remaining owners of the points that are needed in the
        // original partition and level-1 ghosts
        static const unsigned int N_COMM_ROUNDS = 7;
        for (unsigned int comm_round = 0; comm_round < N_COMM_ROUNDS;
             ++comm_round) {
            std::vector<D_INT_L> &partition_search =
                (comm_round < 3) ? my_partition : post_first_round_comms_ids;

            // first round is to get just the face neighbors of everything
            if (comm_round == 0 || comm_round == 3) {
                unsigned int lookup[2];
                for (const unsigned int dir :
                     {OCT_DIR_LEFT, OCT_DIR_RIGHT, OCT_DIR_DOWN, OCT_DIR_UP,
                      OCT_DIR_BACK, OCT_DIR_FRONT}) {
                    for (const auto &ele_id : partition_search) {
                        getElementFaceNeighborsNewPartition(
                            new_oct_connectivity_map, ele_id, dir, lookup);
                        if (lookup[1] != LOOK_UP_TABLE_DEFAULT) {
                            global_ids_needed_for_e2e.insert(lookup[1]);
                        }
                    }
                }
            } else if (comm_round == 1 || comm_round == 4) {
                // second round should be to get edge neighbors
                unsigned int lookup[4];

                for (const unsigned int dir :
                     {OCT_DIR_LEFT_DOWN, OCT_DIR_LEFT_UP, OCT_DIR_LEFT_BACK,
                      OCT_DIR_LEFT_FRONT, OCT_DIR_RIGHT_DOWN, OCT_DIR_RIGHT_UP,
                      OCT_DIR_RIGHT_BACK, OCT_DIR_RIGHT_FRONT,
                      OCT_DIR_DOWN_BACK, OCT_DIR_DOWN_FRONT, OCT_DIR_UP_BACK,
                      OCT_DIR_UP_FRONT}) {
                    for (const auto &ele_id : partition_search) {
                        getElementEdgeNeighborsNewPartition(
                            new_oct_connectivity_map, ele_id, dir, lookup);

                        // then add all of the lookup values if they're not
                        // defaults
                        for (unsigned int lookup_id = 1; lookup_id < 4;
                             ++lookup_id) {
                            if (lookup[lookup_id] != LOOK_UP_TABLE_DEFAULT)
                                global_ids_needed_for_e2e.insert(
                                    lookup[lookup_id]);
                        }
                    }
                }
            } else if (comm_round == 2 || comm_round == 5) {
                // third round should be to get vertex neighbors
                unsigned int lookup[NUM_CHILDREN];

                for (const unsigned int dir :
                     {OCT_DIR_LEFT_DOWN_BACK, OCT_DIR_RIGHT_DOWN_BACK,
                      OCT_DIR_LEFT_UP_BACK, OCT_DIR_RIGHT_UP_BACK,
                      OCT_DIR_LEFT_DOWN_FRONT, OCT_DIR_RIGHT_DOWN_FRONT,
                      OCT_DIR_LEFT_UP_FRONT, OCT_DIR_RIGHT_UP_FRONT}) {
                    for (const auto &ele_id : partition_search) {
                        getElementVertexNeighborsNewPartition(
                            new_oct_connectivity_map, ele_id, dir, lookup);

                        // then add all of the lookup values if they're not
                        // defaults
                        for (unsigned int lookup_id = 1;
                             lookup_id < NUM_CHILDREN; ++lookup_id) {
                            if (lookup[lookup_id] != LOOK_UP_TABLE_DEFAULT)
                                global_ids_needed_for_e2e.insert(
                                    lookup[lookup_id]);
                        }
                    }
                }
            } else if (comm_round == 6) {
                // make sure we have everything we need based on the E2N global
                // mapping
                unsigned int ownerID, ii_x, jj_y, kk_z;
                for (const auto &od : new_oct_connectivity_map) {
                    if (od.isGhostTwo) {
                        continue;
                    }
                    for (unsigned int i = 0; i < m_uiNpE; ++i) {
                        dg2eijk(od.e2n_dg[i], ownerID, ii_x, jj_y, kk_z);
                        global_ids_needed_for_e2e.insert(ownerID);
                    }
                }
            }

            // -- CLEAN UP
            // remove those that are already contained inside the
            // oct_connectivity_map
            for (const auto &od : new_oct_connectivity_map) {
                global_ids_needed_for_e2e.erase(od.eid);
            }

            // find the ones that might already still be part of
            // oct_connectivity_map, then we can grab those directly
            for (auto &ele_id : global_ids_needed_for_e2e) {
                if (ele_id >= ele_offsets[rank] &&
                    ele_id < ele_offsets[rank + 1]) {
                    collected_ghost_elements.push_back(
                        oct_connectivity_map[ele_id - ele_offsets[rank]]);
                    remove_these_ids.push_back(ele_id);
                }
            }
            // clear these overlapped elements, no need to communicate
            for (auto &ele_id : remove_these_ids) {
                global_ids_needed_for_e2e.erase(ele_id);
            }

            // convert to vector
            std::copy(global_ids_needed_for_e2e.begin(),
                      global_ids_needed_for_e2e.end(),
                      std::back_inserter(global_ids_needed_for_e2e_vec));

            // then fetch the data, make sure we don't update our target
            remaining_data = getOctDataFromOtherProcesses(
                oct_connectivity_map, ele_offsets, ele_counts,
                global_ids_needed_for_e2e_vec, false);

            // then make sure we insert the chunks we stored in
            // collected_ghost_elements
            remaining_data.insert(remaining_data.end(),
                                  collected_ghost_elements.begin(),
                                  collected_ghost_elements.end());

            if (comm_round >= 3) {
                // if we're on round 3 or more, we mark each one as "ghostTwo"
                for (unsigned int i = 0; i < remaining_data.size(); i++) {
                    remaining_data[i].isGhostTwo = true;
                }
            }

            // then make sure we push it into the new oct_connectivity_map
            new_oct_connectivity_map.insert(new_oct_connectivity_map.end(),
                                            remaining_data.begin(),
                                            remaining_data.end());

            // clear the maps of interest
            global_ids_needed_for_e2e.clear();
            global_ids_needed_for_e2e_vec.clear();
            collected_ghost_elements.clear();
            remove_these_ids.clear();
            remaining_data.clear();

            // finally, make sure we sort

            std::sort(
                new_oct_connectivity_map.begin(),
                new_oct_connectivity_map.end(),
                [](const oct_data<D_INT_L> &a, const oct_data<D_INT_L> &b) {
                    if (a.trank != b.trank) {
                        return a.trank < b.trank;
                    } else {
                        return a.eid < b.eid;
                    }
                });

            size_t counter = 0;
            for (auto &o : new_oct_connectivity_map) {
                if (o.trank == rank) {
                    break;
                }
                counter++;
            }

            newLocalBegin = counter;
            newNumEle     = new_oct_connectivity_map.size();

            if (comm_round == 2) {
                // just finished first round of ghost fetching
                for (const auto &ot : new_oct_connectivity_map) {
                    // fill out the vector of comm ids, this is what we'll use
                    post_first_round_comms_ids.push_back(ot.eid);
                }
            }
        }

        size_t newLocalEnd = newLocalBegin + my_partition.size();
        // finished all of the fetching rounds, should now have ghosts and
        // "second" ghosts

        // TODO: do we need to check for remaining global IDs? i.e. do we need
        // to do one more round of communication for the level 2 ghost
        // neighbors, which would include edge and corner neighbors?

        // now we can rebuild the E2E map based on this data!
        std::vector<D_INT_L> newE2EMap(newNumEle * this->getNumDirections(),
                                       LOOK_UP_TABLE_DEFAULT);

        // quickly create a global to local map
        std::map<D_INT_L, D_INT_L> globaltoNewLocal;
        for (size_t eid_local = 0; eid_local < newNumEle; eid_local++) {
            globaltoNewLocal[new_oct_connectivity_map[eid_local].eid] =
                eid_local;
        }

        // CREATE FULL E2E MAP BASED ON DATA
        for (size_t eid_local = 0; eid_local < newNumEle; eid_local++) {
            for (int faceid = 0; faceid < this->getNumDirections(); faceid++) {
                if (new_oct_connectivity_map[eid_local].e2e[faceid] !=
                    LOOK_UP_TABLE_DEFAULT) {
                    D_INT_L tempVal =
                        new_oct_connectivity_map[eid_local].e2e[faceid];
                    if (globaltoNewLocal.find(tempVal) !=
                        globaltoNewLocal.end()) {
                        newE2EMap[eid_local * this->getNumDirections() +
                                  faceid] =
                            globaltoNewLocal[new_oct_connectivity_map[eid_local]
                                                 .e2e[faceid]];
                    } else {
                        newE2EMap[eid_local * this->getNumDirections() +
                                  faceid] = LOOK_UP_TABLE_DEFAULT / 2;
                    }
                }
            }
        }
        // done with E2E map creation!

        // --------------------
        // CREATE THE E2N MAP

        // start with the E2N_DG, because it's easiest
        std::vector<D_INT_L> newE2N_dg(newNumEle * m_uiNpE,
                                       LOOK_UP_TABLE_DEFAULT);
        std::vector<D_INT_L> newE2N_cg(newNumEle * m_uiNpE,
                                       LOOK_UP_TABLE_DEFAULT);
        unsigned int nodeLookUp_DG;
        unsigned int nodeLookUp_CG;
        unsigned int inject_id, sub_extract_id;
        unsigned int ownerID, ii_x, jj_y, kk_z;
        unsigned int newOwnerID;

        // construction of the e2n_dg based on the methods in create e2n
        // function
        for (size_t ele_id = 0; ele_id < newNumEle; ++ele_id) {
            // then through the number of dimensions
            D_INT_L currGlobal = new_oct_connectivity_map[ele_id].eid;

            // ele_id is the current local id

            // so the math is pretty simple
            for (unsigned int k = 0; k < (m_uiElementOrder + 1); ++k) {
                for (unsigned int j = 0; j < (m_uiElementOrder + 1); ++j) {
                    for (unsigned int i = 0; (i < m_uiElementOrder + 1); ++i) {
                        sub_extract_id = k * (m_uiElementOrder + 1) *
                                             (m_uiElementOrder + 1) +
                                         j * (m_uiElementOrder + 1) + i;
                        inject_id     = ele_id * m_uiNpE + sub_extract_id;

                        // the "global" representation
                        nodeLookUp_DG = new_oct_connectivity_map[ele_id]
                                            .e2n_dg[sub_extract_id];

                        // get the k, j, i, values through dg2eijk based on this
                        // global value
                        dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);

                        // so, ownerID is the global ID, we need to match it
                        // back to new global
                        if (globaltoNewLocal.find(ownerID) !=
                            globaltoNewLocal.end()) {
                            newOwnerID = globaltoNewLocal[ownerID];
                        } else {
                            // if we don't have a match, then it's technically
                            // owned by another value, but at this point if we
                            // have the level 1 and level 2 ghosts, we pretty
                            // much don't need it, but to avoid errors we'll set
                            // it to the current local element
                            newOwnerID = ele_id;
                        }

                        // now that we have the ii_x, jj_y, and kk_z, we can
                        // recreate this with our local ele_id

                        newE2N_dg[inject_id] =
                            eijk2dg(newOwnerID, ii_x, jj_y, kk_z);
                    }
                }
            }
        }

        // following the logic from build e2N w/ SM
        // this lets us create the E2N CG really easily
        newE2N_cg = newE2N_dg;
        std::sort(newE2N_dg.begin(), newE2N_dg.end());
        newE2N_dg.erase(std::unique(newE2N_dg.begin(), newE2N_dg.end()),
                        newE2N_dg.end());

        // then cg2dg
        std::vector<D_INT_L> cg2dg;
        cg2dg.resize(newE2N_dg.size());
        cg2dg = newE2N_dg;

        newE2N_dg.resize(newNumEle * m_uiNpE);
        newE2N_dg = newE2N_cg;

        std::vector<unsigned int> dg2cg;
        dg2cg.resize(m_uiNpE * newNumEle, LOOK_UP_TABLE_DEFAULT);

        for (unsigned int i = 0; i < cg2dg.size(); i++) dg2cg[cg2dg[i]] = i;

        for (unsigned int i = 0; i < newE2N_dg.size(); i++)
            newE2N_cg[i] = dg2cg[newE2N_cg[i]];

        const unsigned int numCGNodes     = cg2dg.size();

        const unsigned int newEleGhostEnd = new_oct_connectivity_map.size();
        unsigned int tmpIndex;
        unsigned int newNodePreGhostBegin  = UINT_MAX;
        unsigned int newNodeLocalBegin     = UINT_MAX;
        unsigned int newNodePostGhostBegin = UINT_MAX;
        unsigned int newNodePreGhostEnd, newNodeLocalEnd, newNodePostGhostEnd;
        for (unsigned int e = 0; e < newNumEle; ++e) {
            for (unsigned int k = 0; k < m_uiNpE; k++) {
                tmpIndex = (newE2N_dg[e * m_uiNpE + k] / m_uiNpE);
                // if we're within "preghost begin or end" and our newNode value
                // is greater than what's found...
                if ((tmpIndex >= 0) && (tmpIndex < newLocalBegin) &&
                    (newNodePreGhostBegin > newE2N_dg[e * m_uiNpE + k])) {
                    newNodePreGhostBegin = newE2N_dg[e * m_uiNpE + k];
                }

                // if we're within "local begin and end" and our newNode value
                // is greater than what's found...
                if ((tmpIndex >= newLocalBegin) && (tmpIndex < newLocalEnd) &&
                    (newNodeLocalBegin > newE2N_dg[e * m_uiNpE + k])) {
                    newNodeLocalBegin = newE2N_dg[e * m_uiNpE + k];
                }

                // if we're within "postghost begin and end" and our newNode
                // value is greater than what's found...
                if ((tmpIndex >= newLocalEnd) && (tmpIndex < newEleGhostEnd) &&
                    (newNodePostGhostBegin > newE2N_dg[e * m_uiNpE + k])) {
                    newNodePostGhostBegin = newE2N_dg[e * m_uiNpE + k];
                }
            }
        }

        assert(newNodeLocalBegin != UINT_MAX);
        assert(dg2cg[newNodeLocalBegin] != LOOK_UP_TABLE_DEFAULT);
        newNodeLocalBegin = dg2cg[newNodeLocalBegin];
        if (newNodePreGhostBegin == UINT_MAX) {
            newNodePreGhostBegin = 0;
            newNodePreGhostEnd   = 0;
            assert(newNodeLocalBegin == 0);
        } else {
            assert(dg2cg[newNodePreGhostBegin] != LOOK_UP_TABLE_DEFAULT);
            newNodePreGhostBegin = dg2cg[newNodePreGhostBegin];
            newNodePreGhostEnd   = newNodeLocalBegin;
        }

        if (newNodePostGhostBegin == UINT_MAX) {
            newNodeLocalEnd       = cg2dg.size();  // E2N_DG_Sorted.size();
            newNodePostGhostBegin = newNodeLocalEnd;
            newNodePostGhostEnd   = newNodeLocalEnd;
        } else {
            assert(dg2cg[newNodePostGhostBegin] != LOOK_UP_TABLE_DEFAULT);
            newNodePostGhostBegin = dg2cg[newNodePostGhostBegin];
            newNodeLocalEnd       = newNodePostGhostBegin;
            newNodePostGhostEnd   = cg2dg.size();  // E2N_DG_Sorted.size();
        }

        // --------------------
        // RECREATE THE SCATTERMAPS
        std::vector<int> sendNodeCount;
        std::vector<int> recvNodeCount;
        std::vector<unsigned int> sendNodeSM[npes];
        std::vector<unsigned int> recvNodeSM[npes];
        std::vector<unsigned int> recvNodeDGGlobals[npes];
        sendNodeCount.resize(m_uiActiveNpes, 0);
        recvNodeCount.resize(m_uiActiveNpes, 0);
        D_INT_L nodeIndex;

        unsigned int n_dg;
        unsigned int dir;
        unsigned int ib, ie, jb, je, kb, ke;
        unsigned int procIDEle;

        // create receive maps
        for (unsigned int ele_id = 0; ele_id < newNumEle; ele_id++) {
            const auto &ele      = new_oct_connectivity_map[ele_id];
            const D_INT_L procId = ele.trank;

            if (procId == rank) {
                // no need to parse through the whole block here, the
                // "ownership" of the DG and CG map helps
                continue;
            }
            if (ele.isGhostTwo) {
                continue;
            }

            // go through the "corners" and middles of each of these, which
            // tells us enough information about the number we need in each
            // direction
            for (const unsigned int &kk :
                 {0U, m_uiElementOrder / 2, m_uiElementOrder}) {
                for (const unsigned int &jj :
                     {0U, m_uiElementOrder / 2, m_uiElementOrder}) {
                    for (const unsigned int &ii :
                         {0U, m_uiElementOrder / 2, m_uiElementOrder}) {
                        // now we can where it actually goes in the DG mapping
                        nodeIndex = newE2N_dg[ele_id * m_uiNpE +
                                              kk * (m_uiElementOrder + 1) *
                                                  (m_uiElementOrder + 1) +
                                              jj * (m_uiElementOrder + 1) + ii];
                        dg2eijk(nodeIndex, ownerID, ii_x, jj_y, kk_z);

                        // check if the ownerID is within the new localBegin and
                        // localEnd, if it's not then we need to receive it
                        if (ownerID >= newLocalBegin && ownerID < newLocalEnd) {
                            continue;
                        }

                        // calculate the procID based on the owner ID **here**
                        procIDEle = new_oct_connectivity_map[ownerID].trank;

                        // std::cout << rank << ": ii, jj, kk: " << ii << ", "
                        //           << jj << ", " << kk
                        //           << " - ii_x, jj_y, kk_z: " << ii_x << ", "
                        //           << jj_y << ", " << kk_z << std::endl;

                        // otherwise...
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

                        if (ii_x == m_uiElementOrder / 2) {
                            ib = 1;
                            ie = m_uiElementOrder - 1;
                        }
                        if (jj_y == m_uiElementOrder / 2) {
                            jb = 1;
                            je = m_uiElementOrder - 1;
                        }
                        if (kk_z == m_uiElementOrder / 2) {
                            kb = 1;
                            ke = m_uiElementOrder - 1;
                        }

                        if (ii_x == m_uiElementOrder) {
                            ib = m_uiElementOrder;
                            ie = m_uiElementOrder;
                        }
                        if (jj_y == m_uiElementOrder) {
                            jb = m_uiElementOrder;
                            je = m_uiElementOrder;
                        }
                        if (kk_z == m_uiElementOrder) {
                            kb = m_uiElementOrder;
                            ke = m_uiElementOrder;
                        }

                        for (unsigned int k = kb; k <= ke; k++)
                            for (unsigned int j = jb; j <= je; j++)
                                for (unsigned int i = ib; i <= ie; i++) {
                                    recvNodeCount[procIDEle]++;
                                    recvNodeSM[procIDEle].push_back(
                                        newE2N_cg[ownerID * m_uiNpE +
                                                  k * (m_uiElementOrder + 1) *
                                                      (m_uiElementOrder + 1) +
                                                  j * (m_uiElementOrder + 1) +
                                                  i]);
                                    recvNodeDGGlobals[procIDEle].push_back(
                                        new_oct_connectivity_map[ownerID]
                                            .e2n_dg[k * (m_uiElementOrder + 1) *
                                                        (m_uiElementOrder + 1) +
                                                    j * (m_uiElementOrder + 1) +
                                                    i]);
                                }
                    }
                }
            }
        }

        for (unsigned int i = 0; i < npes; i++) {
            auto [recv_tmp, recv_global_tmp] =
                removeDuplicatesSameOrder(recvNodeSM[i], recvNodeDGGlobals[i]);
            std::swap(recvNodeSM[i], recv_tmp);
            std::swap(recvNodeDGGlobals[i], recv_global_tmp);
            recvNodeCount[i] = recvNodeSM[i].size();
        }

        // calculate offsets
        std::vector<int> recvOffsets(npes);
        recvOffsets[0] = 0;
        for (int i = 1; i < npes; ++i) {
            recvOffsets[i] = recvOffsets[i - 1] + recvNodeCount[i - 1];
        }
        int total_recv_size = recvOffsets[npes - 1] + recvNodeCount[npes - 1];

        // exchange all of the send counts
        MPI_Alltoall(recvNodeCount.data(), 1, MPI_INT, sendNodeCount.data(), 1,
                     MPI_INT, commActive);

        std::vector<int> sendOffsets(npes);
        sendOffsets[0] = 0;
        for (int i = 1; i < npes; ++i) {
            sendOffsets[i] = sendOffsets[i - 1] + sendNodeCount[i - 1];
        }
        int total_send_size = sendOffsets[npes - 1] + sendNodeCount[npes - 1];

        // create a flattened recv and send buffer
        std::vector<long unsigned int> flattened_send_buffer(total_send_size);
        std::vector<long unsigned int> flattened_recv_buffer(total_recv_size);

        // NOTE: it's important to remember that we're "sending" receive data
        // and vice versa
        int send_offset = 0;
        for (int i = 0; i < npes; ++i) {
            for (const unsigned int &rcv_data : recvNodeDGGlobals[i]) {
                flattened_recv_buffer[send_offset++] = rcv_data;
            }
        }

        /// now we can do an alltoallv
        MPI_Alltoallv(flattened_recv_buffer.data(), recvNodeCount.data(),
                      recvOffsets.data(), MPI_UNSIGNED_LONG,
                      flattened_send_buffer.data(), sendNodeCount.data(),
                      sendOffsets.data(), MPI_UNSIGNED_LONG, commActive);

        // with the communication now, we can create the "send" scattermap

        std::vector<unsigned int> convertedSendSM(total_send_size);
        std::vector<unsigned int> convertedRecvSM(total_recv_size);
        for (size_t i = 0; i < flattened_send_buffer.size(); ++i) {
            // convert the value to local ID
            dg2eijk(flattened_send_buffer[i], ownerID, ii_x, jj_y, kk_z);

            if (globaltoNewLocal.find(ownerID) != globaltoNewLocal.end()) {
                newOwnerID = globaltoNewLocal[ownerID];
                if (newOwnerID < newLocalBegin || newOwnerID >= newLocalEnd) {
                    // TODO: some kind of error handling, perhaps? this
                    // shouldn't trigger ever
                } else {
                    // std::cout << " I FOUND A TRUE OWNER! " << std::endl;
                }
            } else {
                std::cout << "ERROR: Unknown new Owner ID, original was: "
                          << ownerID << std::endl;
            }

            unsigned int newDGVal = eijk2dg(newOwnerID, ii_x, jj_y, kk_z);

            // then we update the scattermap with the "found" ID
            convertedSendSM[i] =
                newE2N_cg[newOwnerID * m_uiNpE +
                          kk_z * (m_uiElementOrder + 1) *
                              (m_uiElementOrder + 1) +
                          jj_y * (m_uiElementOrder + 1) + ii_x];
        }

        // flatten the receive scattermap
        unsigned int recv_offset = 0;
        for (int i = 0; i < npes; ++i) {
            for (unsigned int j = 0; j < recvNodeSM[i].size(); ++j) {
                convertedRecvSM[recv_offset++] = recvNodeSM[i][j];
            }
        }

        // --------------------
        // UPDATE INTERNAL MESH DATASTRUCTURES

        std::swap(m_uiE2EMapping, newE2EMap);
        std::swap(m_uiE2NMapping_CG, newE2N_cg);
        std::swap(m_uiE2NMapping_DG, newE2N_dg);
        std::swap(m_uiCG2DG, cg2dg);
        std::swap(m_uiDG2CG, dg2cg);

        // create m_uiAllElements, which is a vector of treeNodes
        std::vector<ot::TreeNode> newAllElements;
        for (const auto &oct : new_oct_connectivity_map) {
            unsigned int psz = 1u << (m_uiMaxDepth - oct.level - 1);
            ot::TreeNode temp(oct.coord[0] - psz, oct.coord[1] - psz,
                              oct.coord[2] - psz, oct.level, 3, m_uiMaxDepth);
            // make sure flag is set, though I don't think it's used often, so
            // might not be necessary
            temp.setFlag(oct.flag);

            newAllElements.push_back(temp);
        }
        std::swap(m_uiAllElements, newAllElements);
        // m_uiAllLocalNode doesn't need to be updated here

        // update locations
        m_uiElementPreGhostEnd    = 0;
        m_uiElementPreGhostEnd    = newLocalBegin;
        m_uiElementLocalBegin     = newLocalBegin;
        m_uiElementLocalEnd       = newLocalEnd;
        m_uiElementPostGhostBegin = newLocalEnd;
        m_uiElementPostGhostEnd   = new_oct_connectivity_map.size();
        // update counts
        m_uiNumLocalElements      = newLocalEnd - newLocalBegin;
        m_uiNumPreGhostElements   = newLocalBegin;
        m_uiNumPostGhostElements =
            m_uiElementPostGhostEnd - m_uiElementPostGhostBegin;
        m_uiNumActualNodes = m_uiNumPreGhostElements + m_uiNumLocalElements +
                             m_uiNumPostGhostElements;

        // update preghost begin/end in CG indexing
        m_uiNodePreGhostBegin  = newNodePreGhostBegin;
        m_uiNodePreGhostEnd    = newNodePreGhostEnd;
        m_uiNodeLocalBegin     = newNodeLocalBegin;
        m_uiNodeLocalEnd       = newNodeLocalEnd;
        m_uiNodePostGhostBegin = newNodePostGhostBegin;
        m_uiNodePostGhostEnd   = newNodePostGhostEnd;

        // then update the nodal scattermap stuff
        std::vector<unsigned int> sendNodeCount_modified =
            convertVectorType<int, unsigned int>(sendNodeCount);
        std::vector<unsigned int> recvNodeCount_modified =
            convertVectorType<int, unsigned int>(recvNodeCount);
        std::vector<unsigned int> sendOffsets_modified =
            convertVectorType<int, unsigned int>(sendOffsets);
        std::vector<unsigned int> recvOffsets_modified =
            convertVectorType<int, unsigned int>(recvOffsets);
        std::swap(m_uiSendNodeCount, sendNodeCount_modified);
        std::swap(m_uiRecvNodeCount, recvNodeCount_modified);
        std::swap(m_uiSendNodeOffset, sendOffsets_modified);
        std::swap(m_uiRecvNodeOffset, recvOffsets_modified);
        std::swap(m_uiScatterMapActualNodeSend, convertedSendSM);
        std::swap(m_uiScatterMapActualNodeRecv, convertedRecvSM);
        m_uiSendBufferNodes.resize(std::accumulate(m_uiSendNodeCount.begin(),
                                                   m_uiSendNodeCount.end(), 0));
        m_uiRecvBufferNodes.resize(std::accumulate(m_uiRecvNodeCount.begin(),
                                                   m_uiRecvNodeCount.end(), 0));

        // then update the information
        if (m_uiActiveNpes > 1) {
            m_uiSendProcList.clear();
            m_uiRecvProcList.clear();
            for (unsigned int p = 0; p < m_uiActiveNpes; p++) {
                if (m_uiSendNodeCount[p] != 0) m_uiSendProcList.push_back(p);
                if (m_uiRecvNodeCount[p] != 0) m_uiRecvProcList.push_back(p);
            }

            // then resize the bufferNodes
            m_uiSendBufferNodes.resize(m_uiSendNodeOffset[m_uiActiveNpes - 1] +
                                       m_uiSendNodeCount[m_uiActiveNpes - 1]);
            m_uiRecvBufferNodes.resize(m_uiRecvNodeOffset[m_uiActiveNpes - 1] +
                                       m_uiRecvNodeCount[m_uiActiveNpes - 1]);
        }

        // -----
        // ELEMENT SCATTERMAP
        std::set<unsigned int> scatterMapSend_R1[npes];
        // get the element scattermap as well

        // post_first_round_comms_ids are *all* the elements by global ID that
        // *actually* work, need to convert them to locals, then sort
        std::vector<unsigned int> post_first_round_comms_vec_local;
        post_first_round_comms_vec_local.reserve(
            post_first_round_comms_ids.size());
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
                            scatterMapSend_R1[procOwner].insert(
                                lookup[lookup_id] - newLocalBegin);
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
                            scatterMapSend_R1[procOwner].insert(
                                lookup[lookup_id] - newLocalBegin);
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
            m_uiScatterMapElementRound1.insert(
                m_uiScatterMapElementRound1.end(), scatterMapSend_R1[p].begin(),
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

        // data we don't need:
        // m_uiSplitterNodes, m_uiSendKeyCount, m_uiSendKeyOffset,
        // m_uiSendOct[Count/Offset]Round1
        // m_uiSendOct[Count/Offset]Round2
        // m_ui[Send/Recv]KeyDiag[Count/Offset]
        // m_ui[Send/Recv]Oct[Count/Offset]Round1Diag
        // m_uiRecvKey[Count/Offset]
        // m_uiRecvOct[Count/Offset]Round1
        // m_uiRecvOct[Count/Offset]Round2
        // m_uiGhostElementIDsToBe[Sent/Recv]
        // m_uiSendBufferElement - only used in construction of initial E2E
        // don't need to update the FE element stuff
        // don't need to update mesh domain min/max
        // don't need to update num fake nodes
        // don't need m_uiGhostElementIDsToBe[Sent/Recv]
        // don't need m_ui[Pre/Post]GhostHangingNodeCGID
        // m_uiNpE, m_uiElementOrder, M_uiStencilSz, m_uiNumDirections,
        // m_uiRefEl
        // m_uiF2EMap stuff, since we're not FME
        // m_ui[Send/Recv][Count/Offset]RePt and the lists
        // none of the intergrid transfer stuff
        // unzip map, unzip offset, unzip counts

        // TODO: BLOCKS, includes m_uiUnzippedVecSz
        // TODO: BLOCKS ARE VERY BROKEN

#if 0
        m_uiIsBlockSetup = false;
        m_uiLocalBlockList.clear();
        if (!rank) {
            std::cout << rank << ": Now preparing to set up blocks..."
                      << std::endl;
        }
        performBlocksSetup(m_uiCoarsetBlkLev, NULL, 0);
        if (!rank) {
            std::cout << rank << ": Finished blocksetup after repartitioning..."
                      << std::endl;
        }
        // this sets up m_uiUnZippedVecSz

        buildE2BlockMap();
        if (!rank) {
            std::cout << rank
                      << ": Now finished with the repartitioning scheme!"
                      << std::endl;
        }
#endif
    }

    template <typename T>
    std::vector<oct_data<T>> getOctDataFromOtherProcesses(
        std::vector<oct_data<T>> &oct_connectivity_map,
        std::vector<T> &ele_offsets, std::vector<T> &ele_counts,
        std::vector<T> &data_to_fetch, bool set_target_rank = true) {
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
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                     MPI_INT, commActive);

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

                flattened_send_full_data[counter] =
                    oct_connectivity_map[requested];

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
};

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
