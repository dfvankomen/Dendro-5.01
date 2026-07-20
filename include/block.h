//
// Created by milinda on 4/21/17.
/**
 *@author Milinda Fernando
 *School of Computing, University of Utah
 *@brief contains the class for the block. In order to perform a stencil on the
 *adaptive grid, we treat the adaptive grid as a collection of regular blocks.
 *  Let \f$ \tau \f$ be a 2:1 balance complete sorted (according SFC ordering)
 *octree then we can decompose \f$\tau =\{b_i^l\}\f$   sequence of finite number
 *of regular blocks.
 *
 *  This class contains the block coordinates level of the regular grid
 *embedded, stencil ghost width and number of local points available.
 */

#ifndef SFCSORTBENCH_BLOCK_H
#define SFCSORTBENCH_BLOCK_H

#include <assert.h>
#include <treenode2vtk.h>

#include "TreeNode.h"
#include "dendro.h"

namespace ot {
/**
 * @brief Block type
 * UNSPECIFIED : block flag type is not set
 * UNZIP_INDEPENDENT: unzip operation does not depend on ghost nodes
 * UNZIP_DEPENDENT: unzip operation does depend on at least one ghost node
 *
 */
enum BlockType { UNSPECIFIED = 0, UNZIP_INDEPENDENT, UNZIP_DEPENDENT };

class Block {
   private:
    /**Coordinates of the block. */
    ot::TreeNode m_uiBlockNode;

    /**rotation id of the block*/
    unsigned int m_uiRotID;

    /** size of the regular grid inside the block. */
    unsigned int m_uiRegGridLev;

    /**regular grid local element begin*/
    DendroIntL m_uiLocalElementBegin;

    /** regular grid local element end. Note that element ids in
     * [localBegin,localEnd] is continous and all those elements are inside the
     * current block. */
    DendroIntL m_uiLocalElementEnd;

    /** offset used for local memory allocation.*/
    DendroIntL m_uiOffset;

    /** array size (1D for the current block. */
    unsigned int m_uiSize1D;

    /**padding width (1D) used for pad the block for neighbour blocks. */
    unsigned int m_uiPaddingWidth;

    /**element order */
    unsigned int m_uiEleOrder;

    /** allocation length on X direction*/
    unsigned int m_uiSzX;

    /** allocation length on Y direction*/
    unsigned int m_uiSzY;

    /** allocation length on Z direction*/
    unsigned int m_uiSzZ;

    /** indecies of the 12 negihbour elems*/
    std::vector<unsigned int> m_uiBLK2DIAG;

    /** indecies of the 8 vertex neighbor elems.*/
    std::vector<unsigned int> m_uiBLKVERTX;

    /** number of elements per block. **/
    unsigned int m_uiBlkElem_1D;

    /** set true after the perform block setpup if the block doesn't depend on
     * the ghost region*/
    bool m_uiIsInternal;

    /** block type*/
    BlockType m_uiBlkType;

   public:
    /**@brief Default constructor*/
    Block();

    /**
     * @brief constructor to initialize and create a block.
     * @param [in] pNode ot::TreeNode for the block.
     * @param [in] rotID rotation ID for the block.
     * @param [in] regLev level of the regular grid embedded by the block.
     * @param [in] regEleBegin Local element begin location for the for the
     * octree embedded by the block.
     * @param [in] regEleEnd Local element end location for the octree embedded
     * by the block .
     * @param [in] eleorder: element order of the mesh.
     * */
    Block(ot::TreeNode pNode, unsigned int rotID, unsigned int regLev,
          unsigned int regEleBegin, unsigned int regEleEnd,
          unsigned int eleOrder);

    ~Block();

    /**
     * @brief Return the block node
     * */
    inline ot::TreeNode getBlockNode() const { return m_uiBlockNode; }

    /**
     * @brief returns the regular grid lev (m_uiRegGridLev) value.
     * note: In octree2BlockDecomposition m_uiRegGridLev is used to store the
     * rotation id of the block.
     *  */
    inline unsigned int getRegularGridLev() const { return m_uiRegGridLev; }

    /**@brief returns the rotation id of the block*/
    inline unsigned int getRotationID() const { return m_uiRotID; }

    /**
     * @brief returns the local element begin for the block.
     * */
    inline DendroIntL getLocalElementBegin() const {
        return m_uiLocalElementBegin;
    }

    /**
     * @brief returns the local element end for the block.
     * */

    inline DendroIntL getLocalElementEnd() const { return m_uiLocalElementEnd; }

    /** @brief returns 1D padding width */
    inline unsigned int get1DPadWidth() const { return m_uiPaddingWidth; }

    /**@brief returns the element order*/
    inline unsigned int getElementOrder() const { return m_uiEleOrder; }

    /**@brief set the block offset*/
    void setOffset(DendroIntL offset);

    inline void setBlk2DiagMap(unsigned int owner, unsigned int dir,
                               unsigned int id) {
        m_uiBLK2DIAG[dir * (2 * m_uiBlkElem_1D) + owner] = id;
    }

    inline void setBlk2VertexMap(unsigned int dir, unsigned int id) {
        m_uiBLKVERTX[dir] = id;
    }

    inline void setIsInternal(bool isInternal) { m_uiIsInternal = isInternal; }

    inline void setBlkType(BlockType btype) { m_uiBlkType = btype; }

    inline BlockType getBlockType() const { return m_uiBlkType; }

    inline bool isInternal() { return m_uiIsInternal; }

    /**@brief set the blkFlag with the correct bdy*/
    inline void setBlkNodeFlag(unsigned int flag) {
        m_uiBlockNode.setFlag(flag);
    };

    /**@brief set the blkFlag with the correct bdy*/
    inline unsigned int getBlkNodeFlag() const {
        return (m_uiBlockNode.getFlag() >> NUM_LEVEL_BITS);
    };

    /** @brief get offset*/
    inline DendroIntL getOffset() const { return m_uiOffset; }

    /** @brief returns the 1D array size*/
    inline unsigned int get1DArraySize() const { return m_uiSize1D; }

    /**@brief allocation length on X direction*/
    inline unsigned int getAllocationSzX() const { return m_uiSzX; }

    /**@brief allocation length on Y direction*/
    inline unsigned int getAllocationSzY() const { return m_uiSzY; }

    /**@brief allocation length on Z direction*/
    inline unsigned int getAllocationSzZ() const { return m_uiSzZ; }

    /**@brief align the total block size*/
    inline unsigned int getAlignedBlockSz() const {
        // unsigned int tmp;
        // ((m_uiSzX & ((1u<<DENDRO_BLOCK_ALIGN_FACTOR_LOG)-1))==0)? tmp=m_uiSzX
        // :
        // tmp=((m_uiSzX/(1u<<DENDRO_BLOCK_ALIGN_FACTOR_LOG))+1)*(1u<<DENDRO_BLOCK_ALIGN_FACTOR_LOG);
        return m_uiSzX * m_uiSzY * m_uiSzZ;
        // unsigned int ax=binOp::getNextHighestPowerOfTwo(m_uiSzX);
        // unsigned int ay=binOp::getNextHighestPowerOfTwo(m_uiSzY);
        // unsigned int az=binOp::getNextHighestPowerOfTwo(m_uiSzZ);
        // return ax * ay * az;
    }

    inline const unsigned int* getBlk2DiagMap() const {
        return &(*(m_uiBLK2DIAG.begin()));
    }

    inline const unsigned int* getBlk2VertexMap() const {
        return &(*(m_uiBLKVERTX.begin()));
    }

    inline void setAllocationSzX(unsigned int sz) { m_uiSzX = sz; }
    inline void setAllocationSzY(unsigned int sz) { m_uiSzY = sz; }
    inline void setAllocationSzZ(unsigned int sz) { m_uiSzZ = sz; }
    inline void setSiz1D(unsigned int sz) { m_uiSize1D = sz; }

    inline unsigned int getElemSz1D() const { return m_uiBlkElem_1D; }

    inline const std::vector<unsigned int>& getBlk2DiagMap_vec() const {
        return m_uiBLK2DIAG;
    }
    inline const std::vector<unsigned int>& getBlk2VertexMap_vec() const {
        return m_uiBLKVERTX;
    }

    /**@brief computes and returns the space discretization (grid domain) */
    double computeGridDx() const;

    /**@brief computes and returns the space discretization (grid domain) */
    double computeGridDy() const;

    /**@brief computes and returns the space discretization (grid domain) */
    double computeGridDz() const;

    /**@brief computes and returns the space discretization on x direction
     * (problem domain)*/
    double computeDx(const Point& d_min, const Point& d_max) const;
    /**@brief computes and returns the space discretization on x direction
     * (problem domain)*/
    double computeDy(const Point& d_min, const Point& d_max) const;
    /**@brief computes and returns the space discretization on x direction
     * (problem domain)*/
    double computeDz(const Point& d_min, const Point& d_max) const;

    /*** @brief initialize the block diagonal map. */
    void initializeBlkDiagMap(const unsigned int value);

    /*** @brief initialize the block vertex neighbour map. */
    void initializeBlkVertexMap(const unsigned int value);

    /**@brief compute the eijk for an element inside the block.  */
    void computeEleIJK(ot::TreeNode pNode, unsigned int* eijk) const;

    /**@brief: returns true if the pNode is inside the current block*/
    bool isBlockInternalEle(ot::TreeNode pNode) const;
};

/**@brief When true, the OCT_LOCAL_WITH_PADDING unzip buffers are first-touched
 * block-major (each thread faults the pages of its own contiguous block range)
 * instead of the default flat static fill, so the RHS -- which consumes the same
 * cost-balanced partition -- streams NUMA-local on multi-socket nodes. Set by the
 * BSSN layer (bssn::set_rhs_omp_schedule) from BSSN_HYBRID_RHS_SCHEDULE=="balanced"
 * BEFORE the buffers are allocated. Default false => original behavior. Only
 * consulted under DENDRO_HYBRID_OMP. See DVector::create_vector and the RHS NUMA
 * Tax. */
inline bool g_padded_numa_first_touch = false;

/**@brief Cost-balanced CONTIGUOUS partition of the block index range
 * [0,numBlocks) across `nthreads`, balancing the summed block allocation volume
 * (SzX*SzY*SzZ, a cheap RHS-cost proxy). Writes nthreads+1 ascending block-index
 * boundaries into out[] with out[0]=0 and out[nthreads]=numBlocks; thread t owns
 * blocks [out[t], out[t+1]). Contiguous ranges keep each thread's unzip-buffer
 * footprint contiguous, so a matching block-major first-touch lands its pages on
 * the thread's node. Deterministic and side-effect-free: the first-touch
 * (DVector::create_vector) and the consume (bssnRHS) call it on the SAME block
 * list and therefore get the SAME partition -- the invariant that makes the two
 * agree. `out` must have room for nthreads+1 entries. */
inline void computeBalancedBlockPartition(const ot::Block* blkList,
                                          unsigned int numBlocks,
                                          unsigned int nthreads,
                                          unsigned int* out) {
    if (nthreads == 0) return;
    out[0] = 0;
    if (nthreads == 1) {
        out[1] = numBlocks;
        return;
    }
    // total cost proxy over all blocks
    unsigned long long total = 0;
    for (unsigned int b = 0; b < numBlocks; b++)
        total += (unsigned long long)blkList[b].getAllocationSzX() *
                 blkList[b].getAllocationSzY() * blkList[b].getAllocationSzZ();

    if (numBlocks == 0 || total == 0) {
        // degenerate: fall back to an even split by block COUNT
        for (unsigned int t = 1; t < nthreads; t++)
            out[t] = (unsigned int)(((unsigned long long)t * numBlocks) / nthreads);
        out[nthreads] = numBlocks;
        for (unsigned int t = 1; t <= nthreads; t++)
            if (out[t] < out[t - 1]) out[t] = out[t - 1];
        return;
    }

    // greedy contiguous split: advance the block cursor until the cumulative
    // cost reaches each thread's proportional target.
    unsigned long long cum = 0;
    unsigned int b         = 0;
    for (unsigned int t = 1; t < nthreads; t++) {
        // target = total * t / nthreads (integer math, no overflow: total fits
        // in u64 and t < nthreads)
        const unsigned long long target = (total * t) / nthreads;
        while (b < numBlocks && cum < target) {
            cum += (unsigned long long)blkList[b].getAllocationSzX() *
                   blkList[b].getAllocationSzY() * blkList[b].getAllocationSzZ();
            b++;
        }
        out[t] = b;
    }
    out[nthreads] = numBlocks;
    // enforce monotonic, non-decreasing boundaries (greedy already is, but keep
    // the invariant explicit so empty ranges are well-defined, not reversed).
    for (unsigned int t = 1; t <= nthreads; t++)
        if (out[t] < out[t - 1]) out[t] = out[t - 1];
}

}  // end of namespace ot

#endif  // SFCSORTBENCH_BLOCK_H
