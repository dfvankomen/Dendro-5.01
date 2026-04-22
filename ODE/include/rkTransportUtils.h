//
// Created by milinda on 7/26/17.
/**
 *@author Milinda Fernando
 *School of Computing, University of Utah
 *@brief Contains gradient computation functions for the advection equation
 */
//

#ifndef SFCSORTBENCH_RKTRANSPORTUTILS_H
#define SFCSORTBENCH_RKTRANSPORTUTILS_H

#include <iostream>
#include <vector>

#include "block.h"
#include "dendro_omp.h"
#include "fdCoefficient.h"
#include "mesh.h"

namespace adv_param {
static const Point domain_min(-M_PI, -M_PI, -M_PI);
static const Point domain_max(M_PI, M_PI, M_PI);
}  // namespace adv_param

/**
 * @param [in] blk: target block to compute the gradient.
 * @param [in] blkID: index of the block.
 * @param [in] uZipIn: unzipped version of the input vector.
 * @param [out] uZipOut: unzigned version of the output vector.
 *
 * */
template <typename T>
void grad(const ot::Mesh* pMesh, unsigned int dir, const T* uZipIn, T* uZipOut);

template <typename T>
void grad(const ot::Mesh* pMesh, unsigned int dir, const T* uZipIn,
          T* uZipOut) {
    const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();

    // grid bounds are read-only shared across threads; per-block state
    // (blkNode, h, blkNpe_1D, paddWidth, lx/ly/lz, offset) is declared
    // inside each parallel loop body so every thread gets its own copy
    const unsigned int grid_min  = 0;
    const unsigned int grid_max  = (1u << m_uiMaxDepth);
    const unsigned int stencilSz = 5;

    if (dir == 0) {  // compute the derivative in w.r.t x direction

        // per-block work is fully independent (each block writes to a
        // disjoint region of uZipOut via its own offset). scheduling is
        // dynamic so heterogeneous boundary work doesn't cause imbalance
        DENDRO_OMP_PARALLEL_FOR_DYNAMIC(4)
        for (int blk = 0; blk < (int)blkList.size(); blk++) {
            ot::TreeNode blkNode = blkList[blk].getBlockNode();
            unsigned int paddWidth = blkList[blk].get1DPadWidth();
            unsigned int blkNpe_1D = blkList[blk].get1DArraySize();
            double h = 1.0 / (blkList[blk].computeDx(adv_param::domain_min,
                                                     adv_param::domain_max));

            unsigned int lx = blkList[blk].getAllocationSzX();
            unsigned int ly = blkList[blk].getAllocationSzY();
            unsigned int lz = blkList[blk].getAllocationSzZ();

            unsigned int offset = blkList[blk].getOffset();

            assert(blkNpe_1D > paddWidth);

            if (blkNode.minX() ==
                grid_min) {  // std::cout<<"rank: "<<m_uiActiveRank<<" applying
                             // forward difference difference: "<<std::endl;

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++) {
                        uZipOut[offset + k * (ly * lx) + j * (lx) + paddWidth] =
                            0;
                        for (unsigned int index = 0; index < stencilSz; index++)
                            uZipOut[offset + k * (ly * lx) + j * (lx) +
                                    paddWidth] +=
                                fd::D1_ORDER_4_FORWARD[index] *
                                uZipIn[offset + k * (ly * lx) + j * (lx) +
                                       paddWidth + index] *
                                h;
                    }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = (paddWidth + 1);
                             i < (2 * paddWidth); i++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] += fd::D1_ORDER_4_UPWIND[index] *
                                              uZipIn[offset + k * (ly * lx) +
                                                     j * (lx) + i + index - 1] *
                                              h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = 2 * paddWidth;
                             i < (blkNpe_1D - 2 * paddWidth); i++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] += fd::D1_ORDER_4_CENTERED[index] *
                                              uZipIn[offset + k * (ly * lx) +
                                                     j * (lx) + i + index - 2] *
                                              h;
                        }

                if (blkNode.maxX() == grid_max) {
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - paddWidth); j++)
                            for (unsigned int i = (blkNpe_1D - 2 * paddWidth);
                                 i < (blkNpe_1D - paddWidth - 1); i++) {
                                uZipOut[offset + k * (ly * lx) + j * (lx) + i] =
                                    0;
                                for (unsigned int index = 0; index < stencilSz;
                                     index++)
                                    uZipOut[offset + k * (ly * lx) + j * (lx) +
                                            i] +=
                                        fd::D1_ORDER_4_DOWNWIND[index] *
                                        uZipIn[offset + k * (ly * lx) +
                                               j * (lx) + i + index - 3] *
                                        h;
                            }

                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - paddWidth); j++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) +
                                    (blkNpe_1D - paddWidth - 1)] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        (blkNpe_1D - paddWidth - 1)] +=
                                    fd::D1_ORDER_4_BACKWARD[index] *
                                    uZipIn[offset + k * (ly * lx) + j * (lx) +
                                           (blkNpe_1D - paddWidth - 1) + index -
                                           4] *
                                    h;
                        }

                } else {
                    assert(blkNode.maxX() < grid_max);
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - paddWidth); j++)
                            for (unsigned int i = (blkNpe_1D - 2 * paddWidth);
                                 i < (blkNpe_1D - paddWidth); i++) {
                                uZipOut[offset + k * (ly * lx) + j * (lx) + i] =
                                    0;
                                for (unsigned int index = 0; index < stencilSz;
                                     index++)
                                    uZipOut[offset + k * (ly * lx) + j * (lx) +
                                            i] +=
                                        fd::D1_ORDER_4_CENTERED[index] *
                                        uZipIn[offset + k * (ly * lx) +
                                               j * (lx) + i + index - 2] *
                                        h;
                            }
                }

            } else if (blkNode.maxX() == grid_max) {
                assert(blkNode.minX() > grid_min);
                assert((blkNpe_1D - 2 * paddWidth));
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying backward
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - 2 * paddWidth); i++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] += fd::D1_ORDER_4_CENTERED[index] *
                                              uZipIn[offset + k * (ly * lx) +
                                                     j * (lx) + i + index - 2] *
                                              h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = (blkNpe_1D - 2 * paddWidth);
                             i < (blkNpe_1D - paddWidth - 1); i++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] += fd::D1_ORDER_4_DOWNWIND[index] *
                                              uZipIn[offset + k * (ly * lx) +
                                                     j * (lx) + i + index - 3] *
                                              h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++) {
                        uZipOut[offset + k * (ly * lx) + j * (lx) +
                                (blkNpe_1D - paddWidth - 1)] = 0;
                        for (unsigned int index = 0; index < stencilSz; index++)
                            uZipOut[offset + k * (ly * lx) + j * (lx) +
                                    (blkNpe_1D - paddWidth - 1)] +=
                                fd::D1_ORDER_4_BACKWARD[index] *
                                uZipIn[offset + k * (ly * lx) + j * (lx) +
                                       (blkNpe_1D - paddWidth - 1) + index -
                                       4] *
                                h;
                    }

            } else {
                assert(blkNode.minX() > grid_min && blkNode.maxX() < grid_max);

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_CENTERED[index] *
                                    uZipIn[offset + k * (ly * lx) + j * (lx) +
                                           (i + index - 2)] *
                                    h;
                        }
            }
        }
    } else if (dir == 1) {  // compute the derivative in w.r.t y direction

        // per-block work is fully independent (each block writes to a
        // disjoint region of uZipOut via its own offset). scheduling is
        // dynamic so heterogeneous boundary work doesn't cause imbalance
        DENDRO_OMP_PARALLEL_FOR_DYNAMIC(4)
        for (int blk = 0; blk < (int)blkList.size(); blk++) {
            ot::TreeNode blkNode = blkList[blk].getBlockNode();
            unsigned int paddWidth = blkList[blk].get1DPadWidth();
            unsigned int blkNpe_1D = blkList[blk].get1DArraySize();
            double h = 1.0 / (blkList[blk].computeDy(adv_param::domain_min,
                                                     adv_param::domain_max));

            unsigned int lx = blkList[blk].getAllocationSzX();
            unsigned int ly = blkList[blk].getAllocationSzY();
            unsigned int lz = blkList[blk].getAllocationSzZ();

            unsigned int offset = blkList[blk].getOffset();

            assert(blkNpe_1D > paddWidth);

            if (blkNode.minY() ==
                grid_min) {  // std::cout<<"rank: "<<m_uiActiveRank<<" applying
                             // forward difference difference: "<<std::endl;

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++) {
                        uZipOut[offset + k * (ly * lx) + (paddWidth) * (lx) +
                                i] = 0;
                        for (unsigned int index = 0; index < stencilSz; index++)
                            uZipOut[offset + k * (ly * lx) +
                                    (paddWidth) * (lx) + i] +=
                                fd::D1_ORDER_4_FORWARD[index] *
                                uZipIn[offset + k * (ly * lx) +
                                       (paddWidth + index) * (lx) + i] *
                                h;
                    }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = (paddWidth + 1);
                             j < (2 * paddWidth); j++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_UPWIND[index] *
                                    uZipIn[offset + k * (ly * lx) +
                                           (j + index - 1) * (lx) + i] *
                                    h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = 2 * paddWidth;
                             j < (blkNpe_1D - 2 * paddWidth); j++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_CENTERED[index] *
                                    uZipIn[offset + k * (ly * lx) +
                                           (j + index - 2) * (lx) + i] *
                                    h;
                        }

                if (blkNode.maxY() == grid_max) {
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int j = (blkNpe_1D - 2 * paddWidth);
                                 j < (blkNpe_1D - paddWidth - 1); j++) {
                                uZipOut[offset + k * (ly * lx) + j * (lx) + i] =
                                    0;
                                for (unsigned int index = 0; index < stencilSz;
                                     index++)
                                    uZipOut[offset + k * (ly * lx) + j * (lx) +
                                            i] +=
                                        fd::D1_ORDER_4_DOWNWIND[index] *
                                        uZipIn[offset + k * (ly * lx) +
                                               (j + index - 3) * (lx) + i] *
                                        h;
                            }

                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++) {
                            uZipOut[offset + k * (ly * lx) +
                                    (blkNpe_1D - paddWidth - 1) * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) +
                                        (blkNpe_1D - paddWidth - 1) * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_BACKWARD[index] *
                                    uZipIn[offset + k * (ly * lx) +
                                           ((blkNpe_1D - paddWidth - 1) +
                                            index - 4) *
                                               (lx) +
                                           i] *
                                    h;
                        }

                } else {
                    assert(blkNode.maxY() < grid_max);
                    for (unsigned int k = paddWidth;
                         k < (blkNpe_1D - paddWidth); k++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int j = (blkNpe_1D - 2 * paddWidth);
                                 j < (blkNpe_1D - paddWidth); j++) {
                                uZipOut[offset + k * (ly * lx) + j * (lx) + i] =
                                    0;
                                for (unsigned int index = 0; index < stencilSz;
                                     index++)
                                    uZipOut[offset + k * (ly * lx) + j * (lx) +
                                            i] +=
                                        fd::D1_ORDER_4_CENTERED[index] *
                                        uZipIn[offset + k * (ly * lx) +
                                               (j + index - 2) * (lx) + i] *
                                        h;
                            }
                }

            } else if (blkNode.maxY() == grid_max) {
                assert(blkNode.minY() > grid_min);
                assert((blkNpe_1D - 2 * paddWidth));
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying backward
                // difference difference: "<<std::endl;
                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - 2 * paddWidth); j++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_CENTERED[index] *
                                    uZipIn[offset + k * (ly * lx) +
                                           (j + index - 2) * (lx) + i] *
                                    h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = (blkNpe_1D - 2 * paddWidth);
                             j < (blkNpe_1D - paddWidth - 1); j++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_DOWNWIND[index] *
                                    uZipIn[offset + k * (ly * lx) +
                                           (j + index - 3) * (lx) + i] *
                                    h;
                        }

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++) {
                        uZipOut[offset + k * (ly * lx) +
                                (blkNpe_1D - paddWidth - 1) * (lx) + i] = 0;
                        for (unsigned int index = 0; index < stencilSz; index++)
                            uZipOut[offset + k * (ly * lx) +
                                    (blkNpe_1D - paddWidth - 1) * (lx) + i] +=
                                fd::D1_ORDER_4_BACKWARD[index] *
                                uZipIn[offset + k * (ly * lx) +
                                       ((blkNpe_1D - paddWidth - 1) + index -
                                        4) *
                                           (lx) +
                                       i] *
                                h;
                    }

            } else {
                assert(blkNode.minY() > grid_min && blkNode.maxY() < grid_max);

                for (unsigned int k = paddWidth; k < (blkNpe_1D - paddWidth);
                     k++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int j = paddWidth;
                             j < (blkNpe_1D - paddWidth); j++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_CENTERED[index] *
                                    uZipIn[offset + k * (ly * lx) +
                                           (j + index - 2) * (lx) + i] *
                                    h;
                        }
            }
        }

    } else if (dir == 2) {  // compute the derivative in w.r.t z direction

        // per-block work is fully independent (each block writes to a
        // disjoint region of uZipOut via its own offset). scheduling is
        // dynamic so heterogeneous boundary work doesn't cause imbalance
        DENDRO_OMP_PARALLEL_FOR_DYNAMIC(4)
        for (int blk = 0; blk < (int)blkList.size(); blk++) {
            ot::TreeNode blkNode = blkList[blk].getBlockNode();
            unsigned int paddWidth = blkList[blk].get1DPadWidth();
            unsigned int blkNpe_1D = blkList[blk].get1DArraySize();
            double h = 1.0 / (blkList[blk].computeDz(adv_param::domain_min,
                                                     adv_param::domain_max));

            unsigned int lx = blkList[blk].getAllocationSzX();
            unsigned int ly = blkList[blk].getAllocationSzY();
            unsigned int lz = blkList[blk].getAllocationSzZ();

            unsigned int offset = blkList[blk].getOffset();

            assert(blkNpe_1D > paddWidth);

            if (blkNode.minZ() ==
                grid_min) {  // std::cout<<"rank: "<<m_uiActiveRank<<" applying
                             // forward difference difference: "<<std::endl;

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++) {
                        uZipOut[offset + (paddWidth) * (ly * lx) + (j) * (lx) +
                                i] = 0;
                        for (unsigned int index = 0; index < stencilSz; index++)
                            uZipOut[offset + (paddWidth) * (ly * lx) +
                                    (j) * (lx) + i] +=
                                fd::D1_ORDER_4_FORWARD[index] *
                                uZipIn[offset +
                                       (paddWidth + index) * (ly * lx) +
                                       j * (lx) + i] *
                                h;
                    }

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = (paddWidth + 1);
                             k < (2 * paddWidth); k++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_UPWIND[index] *
                                    uZipIn[offset +
                                           (k + index - 1) * (ly * lx) +
                                           j * (lx) + i] *
                                    h;
                        }

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = 2 * paddWidth;
                             k < (blkNpe_1D - 2 * paddWidth); k++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_CENTERED[index] *
                                    uZipIn[offset +
                                           (k + index - 2) * (ly * lx) +
                                           j * (lx) + i] *
                                    h;
                        }

                if (blkNode.maxZ() == grid_max) {
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int k = (blkNpe_1D - 2 * paddWidth);
                                 k < (blkNpe_1D - paddWidth - 1); k++) {
                                uZipOut[offset + k * (ly * lx) + j * (lx) + i] =
                                    0;
                                for (unsigned int index = 0; index < stencilSz;
                                     index++)
                                    uZipOut[offset + k * (ly * lx) + j * (lx) +
                                            i] +=
                                        fd::D1_ORDER_4_DOWNWIND[index] *
                                        uZipIn[offset +
                                               (k + index - 3) * (ly * lx) +
                                               j * (lx) + i] *
                                        h;
                            }

                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++) {
                            uZipOut[offset +
                                    (blkNpe_1D - paddWidth - 1) * (ly * lx) +
                                    j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset +
                                        (blkNpe_1D - paddWidth - 1) *
                                            (ly * lx) +
                                        j * (lx) + i] +=
                                    fd::D1_ORDER_4_BACKWARD[index] *
                                    uZipIn[offset +
                                           ((blkNpe_1D - paddWidth - 1) +
                                            index - 4) *
                                               (ly * lx) +
                                           j * (lx) + i] *
                                    h;
                        }

                } else {
                    assert(blkNode.maxY() < grid_max);
                    for (unsigned int j = paddWidth;
                         j < (blkNpe_1D - paddWidth); j++)
                        for (unsigned int i = paddWidth;
                             i < (blkNpe_1D - paddWidth); i++)
                            for (unsigned int k = (blkNpe_1D - 2 * paddWidth);
                                 k < (blkNpe_1D - paddWidth); k++) {
                                uZipOut[offset + k * (ly * lx) + j * (lx) + i] =
                                    0;
                                for (unsigned int index = 0; index < stencilSz;
                                     index++)
                                    uZipOut[offset + k * (ly * lx) + j * (lx) +
                                            i] +=
                                        fd::D1_ORDER_4_CENTERED[index] *
                                        uZipIn[offset +
                                               (k + index - 2) * (ly * lx) +
                                               j * (lx) + i] *
                                        h;
                            }
                }

            } else if (blkNode.maxZ() == grid_max) {
                assert(blkNode.minZ() > grid_min);
                assert((blkNpe_1D - 2 * paddWidth));
                // std::cout<<"rank: "<<m_uiActiveRank<<" applying backward
                // difference difference: "<<std::endl;
                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = paddWidth;
                             k < (blkNpe_1D - 2 * paddWidth); k++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_CENTERED[index] *
                                    uZipIn[offset +
                                           (k + index - 2) * (ly * lx) +
                                           j * (lx) + i] *
                                    h;
                        }

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = (blkNpe_1D - 2 * paddWidth);
                             k < (blkNpe_1D - paddWidth - 1); k++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_DOWNWIND[index] *
                                    uZipIn[offset +
                                           (k + index - 3) * (ly * lx) +
                                           j * (lx) + i] *
                                    h;
                        }

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++) {
                        uZipOut[offset +
                                (blkNpe_1D - paddWidth - 1) * (ly * lx) +
                                j * (lx) + i] = 0;
                        for (unsigned int index = 0; index < stencilSz; index++)
                            uZipOut[offset +
                                    (blkNpe_1D - paddWidth - 1) * (ly * lx) +
                                    j * (lx) + i] +=
                                fd::D1_ORDER_4_BACKWARD[index] *
                                uZipIn[offset +
                                       ((blkNpe_1D - paddWidth - 1) + index -
                                        4) *
                                           (ly * lx) +
                                       j * (lx) + i] *
                                h;
                    }

            } else {
                assert(blkNode.minZ() > grid_min && blkNode.maxZ() < grid_max);

                for (unsigned int j = paddWidth; j < (blkNpe_1D - paddWidth);
                     j++)
                    for (unsigned int i = paddWidth;
                         i < (blkNpe_1D - paddWidth); i++)
                        for (unsigned int k = paddWidth;
                             k < (blkNpe_1D - paddWidth); k++) {
                            uZipOut[offset + k * (ly * lx) + j * (lx) + i] = 0;
                            for (unsigned int index = 0; index < stencilSz;
                                 index++)
                                uZipOut[offset + k * (ly * lx) + j * (lx) +
                                        i] +=
                                    fd::D1_ORDER_4_CENTERED[index] *
                                    uZipIn[offset +
                                           (k + index - 2) * (ly * lx) +
                                           j * (lx) + i] *
                                    h;
                        }
            }
        }

    } else {
        std::cout << " unknown stencil direction " << std::endl;
    }
}

#endif  // SFCSORTBENCH_RKTRANSPORTUTILS_H
