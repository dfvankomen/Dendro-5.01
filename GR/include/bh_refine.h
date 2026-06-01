/**
 * @file bh_refine.h
 * @brief Black-hole-located AMR refinement.
 *
 * Stateless helper used by GR solvers (BSSN, CCZ4, EMDA, ...) to anchor
 * refinement on per-BH "onion" shells in addition to or instead of wavelet
 * refinement. Lives in dendrolib so all formulations share one impl.
 *
 * The orbit-tracking / GW-band / ringdown logic from upstream BSSN's
 * isRemeshBH() is intentionally NOT included here -- those depend on solver
 * state (current time, BH trajectories) that doesn't belong in dendrolib.
 * Solvers needing that can wrap this with their own logic on top.
 */

#ifndef DENDRO_GR_BH_REFINE_H
#define DENDRO_GR_BH_REFINE_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

#include "dendro.h"
#include "mesh.h"
#include "mpi.h"
#include "point.h"

namespace dendro_gr {

struct BHLocCfg {
    /**@brief per-BH coordinate (physical, not octree) */
    std::vector<Point> bh_locations;
    /**@brief inner refinement radius for each BH (cells within refine to
     * max_level) */
    std::vector<double> bh_amr_radii;
    /**@brief per-BH max target level */
    std::vector<unsigned int> bh_max_levels;
    /**@brief concentric-shell ratio: each outer shell has r *= ratio and
     * level -= 1. golden ratio (1.618...) matches upstream BSSN. */
    double ratio = 1.618033988749895;
    /**@brief don't coarsen below this level (set to MINDEPTH) */
    unsigned int min_level = 0;
    /**@brief subtract from max_level to account for 2:1 balance padding */
    unsigned int level_offset = MAXDEAPTH_LEVEL_DIFF + 1;
    /**@brief if false, only promote existing flags (never coarsen below the
     * current flag). Use for BH_WAMR where WAMR set flags first. */
    bool overwrite = true;
    /**@brief default flag for cells outside ALL shells when overwrite=true.
     * OCT_COARSE (default, matches upstream BSSN) actively coarsens outside
     * the BH region; OCT_NO_CHANGE preserves existing structure. */
    unsigned int default_flag = OCT_COARSE;
};

/**@brief Compute the desired refinement level for a point at distance `r`
 * from a BH whose inner shell has radius `r_inner` and target `max_lev`.
 * Returns 0 if the point is outside all shells (no anchor). */
inline int onion_level(double r, double r_inner, int max_lev, double ratio) {
    double cur_r  = r_inner;
    int cur_lev   = max_lev;
    while (cur_lev > 0) {
        if (r <= cur_r) return cur_lev;
        cur_r *= ratio;
        cur_lev--;
    }
    return 0;
}

/**@brief Refine the mesh based on BH locations + onion shells.
 *
 * For each local element: compute min distance from its 8 corners to each
 * BH; raise its flag to OCT_SPLIT/OCT_NO_CHANGE based on the highest target
 * level demanded by any BH's onion shells; default to OCT_COARSE outside
 * all shells (but never below cfg.min_level).
 *
 * @return true if the mesh changed (any rank).
 */
inline bool isRemeshBH(ot::Mesh* pMesh, const BHLocCfg& cfg) {
    const unsigned int n_bh = cfg.bh_locations.size();
    assert(n_bh == cfg.bh_amr_radii.size());
    assert(n_bh == cfg.bh_max_levels.size());

    bool oct_change = false;

    if (pMesh->isActive()) {
        const unsigned int ele_begin = pMesh->getElementLocalBegin();
        const unsigned int ele_end   = pMesh->getElementLocalEnd();
        const ot::TreeNode* nodes    = pMesh->getAllElements().data();

        std::vector<unsigned int> refine_flags;
        if (cfg.overwrite) {
            refine_flags.assign(pMesh->getNumLocalMeshElements(),
                                OCT_NO_CHANGE);
        } else {
            refine_flags = pMesh->getAllRefinementFlags();
        }

        for (unsigned int ele = ele_begin; ele < ele_end; ele++) {
            const unsigned int ln =
                1u << (m_uiMaxDepth - nodes[ele].getLevel());

            // min distance from element corners to each BH
            std::vector<double> r_min(n_bh, std::numeric_limits<double>::max());
            for (unsigned int kk = 0; kk < 2; kk++)
                for (unsigned int jj = 0; jj < 2; jj++)
                    for (unsigned int ii = 0; ii < 2; ii++) {
                        Point oct(
                            nodes[ele].minX() + ii * ln,
                            nodes[ele].minY() + jj * ln,
                            nodes[ele].minZ() + kk * ln);
                        Point phys;
                        pMesh->octCoordToDomainCoord(oct, phys);
                        for (unsigned int b = 0; b < n_bh; b++) {
                            const double d = (phys - cfg.bh_locations[b]).abs();
                            if (d < r_min[b]) r_min[b] = d;
                        }
                    }

            // overwrite: start at cfg.default_flag (NO_CHANGE by default),
            // promote inside shells. promote-only: keep WAMR's flag, only
            // promote.
            unsigned int flag = cfg.overwrite
                                    ? cfg.default_flag
                                    : refine_flags[ele - ele_begin];
            const int current_level = nodes[ele].getLevel();

            auto set_level_floor = [&](int l_target) {
                if (l_target <= 0) return;
                if (current_level < l_target) {
                    flag = OCT_SPLIT;
                } else if (current_level == l_target && flag == OCT_COARSE) {
                    flag = OCT_NO_CHANGE;
                }
            };

            for (unsigned int b = 0; b < n_bh; b++) {
                const int max_target =
                    static_cast<int>(cfg.bh_max_levels[b]) -
                    static_cast<int>(cfg.level_offset);
                if (max_target <= 0) continue;
                set_level_floor(onion_level(r_min[b], cfg.bh_amr_radii[b],
                                            max_target, cfg.ratio));
            }

            // never coarsen below min_level
            set_level_floor(static_cast<int>(cfg.min_level));

            refine_flags[ele - ele_begin] = flag;
        }

        oct_change = pMesh->setMeshRefinementFlags(refine_flags);
    }

    bool oct_change_g = false;
    MPI_Allreduce(&oct_change, &oct_change_g, 1, MPI_CXX_BOOL, MPI_LOR,
                  pMesh->getMPIGlobalCommunicator());
    return oct_change_g;
}

}  // namespace dendro_gr

#endif  // DENDRO_GR_BH_REFINE_H
