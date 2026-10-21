/// @file lts.h
/// The solver contract of local time stepping (ts::ExplicitNUTS): the options
/// that change the solution, the helpers for a context's block hooks, and the
/// driver helpers for mesh changes and event cadence.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include "block.h"
#include "mesh.h"
#include "parUtils.h"
#include "point.h"
#include "ts.h"

namespace ts {

/// Settings that change the LTS solution; everything else is fixed library
/// behavior. Given to the ExplicitNUTS constructor.
struct LtsOptions {
    /// Interface stage conversion: 3 (third order) or 15 (fourth order).
    unsigned int iface     = 3;
    /// With iface 15: exchange the stored stage vector g, so the result does
    /// not depend on the rank count.
    bool g_exchange        = false;
    /// Largest coarse-to-fine step ratio, snapped down to a power of two;
    /// 0 = no cap, 1 = every block takes the finest step.
    unsigned int ratio_cap = 0;
};

/// True when `opts` is a setting ExplicitNUTS accepts (iface 3 or 15).
/// Collective: no. Thread-safe: yes.
inline bool lts_options_valid(const LtsOptions& opts) {
    return opts.iface == 3 || opts.iface == 15;
}

/// True for the tableaus ExplicitNUTS steps: RK3 and RK4, the ones its
/// interface stage conversion is derived for. Collective: no. Thread-safe: yes.
inline bool lts_tableau_supported(ETSType type) {
    return type == ETSType::RK3 || type == ETSType::RK4;
}

/// Ratio cap snapped down to a power of two; 0 stays 0 (no cap).
/// Collective: no. Thread-safe: yes.
inline unsigned int snap_ratio_cap(unsigned int cap) {
    if (cap <= 1) return cap;
    unsigned int p = 1u;
    while ((p << 1) <= cap) p <<= 1;
    return p;
}

/// Step of a block at level `blev` in units of the finest step: 1 on the
/// finest `offset + 1` levels, doubling per level above them.
/// Collective: no. Thread-safe: yes.
/// @param lmax  finest level of the mesh
/// @param offset  level offset (see level_offset)
/// @param cap  snapped ratio cap; 0 = no cap
/// @return power of two in [1, cap]
inline unsigned int step_ratio(unsigned int blev, unsigned int lmax,
                               unsigned int offset, unsigned int cap) {
    unsigned int fac =
        ((lmax - blev) <= offset) ? 1u : (1u << (lmax - blev - offset));
    if (cap != 0u && fac > cap) fac = cap;
    return fac;
}

/// True when the context declares `static constexpr bool lts_thread_safe =
/// true;`: its block hooks may run concurrently on different blocks. Without
/// it the stepper's block loops run on one thread.
template <class Ctx, class = void>
struct lts_thread_safe : std::false_type {};
template <class Ctx>
struct lts_thread_safe<Ctx, std::void_t<decltype(Ctx::lts_thread_safe)>>
    : std::bool_constant<Ctx::lts_thread_safe> {};
template <class Ctx>
inline constexpr bool lts_thread_safe_v = lts_thread_safe<Ctx>::value;

/// Geometry of one padded block in the solver's coordinates.
struct BlockView {
    unsigned int sz[3];  ///< allocation size (padded points per direction)
    unsigned int nn;     ///< points per variable, sz[0] * sz[1] * sz[2]
    unsigned int pw;     ///< the block's padding width
    unsigned int bflag;  ///< domain-boundary flag of the block
    double dx[3];        ///< grid spacing
    double lo[3];        ///< lower corner of the block (unpadded)
    double hi[3];        ///< upper corner of the block (unpadded)
    double ptmin[3];     ///< lower corner of the padded box
    double ptmax[3];     ///< upper corner of the padded box

    /// Geometry of block `b`; grid coordinates map linearly from the octree
    /// box onto the domain box, in the solver's own expression order.
    /// @param pad  the solver's padding width, used for the padded box
    BlockView(const ot::Block& b, const Point& oct_min, const Point& oct_max,
              const Point& dom_min, const Point& dom_max, unsigned int pad)
        : sz{b.getAllocationSzX(), b.getAllocationSzY(), b.getAllocationSzZ()},
          nn(sz[0] * sz[1] * sz[2]),
          pw(b.get1DPadWidth()),
          bflag(b.getBlkNodeFlag()),
          dx{b.computeDx(dom_min, dom_max), b.computeDy(dom_min, dom_max),
             b.computeDz(dom_min, dom_max)} {
        const ot::TreeNode node = b.getBlockNode();
        const unsigned int gmin[3] = {node.minX(), node.minY(), node.minZ()};
        const unsigned int gmax[3] = {node.maxX(), node.maxY(), node.maxZ()};
        const double omin[3] = {oct_min.x(), oct_min.y(), oct_min.z()};
        const double omax[3] = {oct_max.x(), oct_max.y(), oct_max.z()};
        const double dmin[3] = {dom_min.x(), dom_min.y(), dom_min.z()};
        const double dmax[3] = {dom_max.x(), dom_max.y(), dom_max.z()};
        for (unsigned int d = 0; d < 3; d++) {
            lo[d]    = (((dmax[d] - dmin[d]) / (omax[d] - omin[d])) *
                     (gmin[d] - omin[d])) +
                    dmin[d];
            hi[d]    = (((dmax[d] - dmin[d]) / (omax[d] - omin[d])) *
                     (gmax[d] - omin[d])) +
                    dmin[d];
            ptmin[d] = lo[d] - pad * dx[d];
            ptmax[d] = hi[d] + pad * dx[d];
        }
    }

    /// Points `vars[v]` at variable v of a block-local buffer of `dof`
    /// variables.
    template <class T>
    void split(T* buf, T** vars, unsigned int dof) const {
        for (unsigned int v = 0; v < dof; v++) vars[v] = buf + v * nn;
    }

    /// True for a node of the block interior (outside the padding).
    bool interior(unsigned int i, unsigned int j, unsigned int k) const {
        return i >= pw && i < sz[0] - pw && j >= pw && j < sz[1] - pw &&
               k >= pw && k < sz[2] - pw;
    }
};

/// Calls `f(node, interior)` once on every node that pointwise enforcement
/// applies to: the whole padded block, or only the interior of a block on the
/// domain boundary. Collective: no. Thread-safe: if f is.
template <class F>
void for_each_enforced_node(const BlockView& b, F&& f) {
    const unsigned int pw = b.pw;
    if (!b.bflag) {
        for (unsigned int k = 0; k < b.sz[2]; k++)
            for (unsigned int j = 0; j < b.sz[1]; j++)
                for (unsigned int i = 0; i < b.sz[0]; i++)
                    f(k * b.sz[1] * b.sz[0] + j * b.sz[0] + i,
                      b.interior(i, j, k));
    } else {
        for (unsigned int k = pw; k < b.sz[2] - pw; k++)
            for (unsigned int j = pw; j < b.sz[1] - pw; j++)
                for (unsigned int i = pw; i < b.sz[0] - pw; i++)
                    f(k * b.sz[1] * b.sz[0] + j * b.sz[0] + i, true);
    }
}

/// When a driver event (output, remesh test, checkpoint) is due. Under the
/// global stepper it fires every `freq` steps; under LTS a step is a coarse
/// step, so it fires at the first step at or past each multiple of `freq`
/// fine steps in simulation time. One instance per event.
struct Cadence {
    double next = 0.0;  ///< LTS: time of the next event

    /// True if the event is due at this step; `freq` 0 = never. Advances
    /// `next` when it fires.
    bool due(unsigned int freq, long long step, double time, double dt_fine,
             bool lts) {
        if (freq == 0) return false;
        if (!lts) return (step % freq) == 0;
        if (time < next - 1e-9 * std::max(1.0, time)) return false;
        const double dt = freq * dt_fine;
        next            = (std::floor(time / dt + 1e-9) + 1.0) * dt;
        return true;
    }

    /// After a restart under LTS: the next event is the first interval boundary
    /// after `t_prev`, the start of the coarse step that just ended.
    void resume(unsigned int freq, double t_prev, double dt_fine) {
        const double dt = freq * dt_fine;
        next = freq ? (std::floor(t_prev / dt + 1e-9) + 1.0) * dt : 0.0;
    }
};

/// The driver's event rules in one place: reads the finest step through a
/// getter at every call (a remesh changes it), so no cadence keeps a stale
/// step. One per driver; one Cadence per event.
struct Clock {
    std::function<double()> dt_fine;  ///< the current finest step
    bool lts = false;                 ///< LTS: events by simulation time

    /// True if the event is due at this step (see Cadence::due).
    bool due(Cadence& c, unsigned int freq, long long step, double time) const {
        return c.due(freq, step, time, dt_fine(), lts);
    }

    /// The remesh event: its test skips step 0, so its first interval starts
    /// there (unseeded, it would fire at the first coarse step).
    void seed_remesh(Cadence& c, unsigned int freq) const { c.next = freq * dt_fine(); }

    /// After a restart under LTS: `t_prev` is the start of the coarse step that
    /// just ended (current time minus the coarsest step).
    void resume(Cadence& c, unsigned int freq, double t_prev) const {
        c.resume(freq, t_prev, dt_fine());
    }

    /// Checkpoint slot (0 or 1): under LTS the interval index, since the
    /// coarse step count need not alternate; -1 under the global stepper (the
    /// solver's own step-parity rule).
    int checkpoint_slot(unsigned int freq, double time) const {
        if (!lts) return -1;
        return (int)((long long)std::floor(time / (freq * dt_fine()) + 1e-9) % 2);
    }
};

/// Labels the mesh with the solver's partition method (once, after the
/// initial data); then the solver rebuilds its per-mesh data and the stepper
/// re-syncs. Collective: yes. Invalidates: the mesh object, the stepper's
/// per-mesh state. @return true if the mesh changed
template <class Ctx, class Stepper>
bool apply_partitioning(Ctx& ctx, Stepper& ets) {
    if (!ctx.maybe_apply_partitioning()) return false;
    ctx.on_mesh_changed(false);
    ets.sync_with_mesh();
    return true;
}

/// Remesh test and remesh on any labelling. A labelled mesh goes to SFC for
/// the refinement test and the remesh (both assume SFC element ranges) and is
/// labelled again afterwards, remeshed or not; then the solver rebuilds its
/// per-mesh data and the stepper re-syncs. Collective: yes. Invalidates: the
/// mesh object and the stepper's per-mesh state when the mesh is labelled or
/// remeshed. @return true if the octree changed
template <class Ctx, class Stepper>
bool remesh_if_needed(Ctx& ctx, Stepper& ets, unsigned int grain_sz,
                      double ld_tol, unsigned int sf_k) {
    const ot::Mesh* m  = ctx.get_mesh();
    const bool root    = !m->getMPIRankGlobal();
    const bool labelled = ctx.is_labelled();
    const double t0     = MPI_Wtime();
    if (labelled) {
        ctx.to_sfc_partition();
        ctx.on_mesh_changed(false);
    }
    const double t1    = MPI_Wtime();
    const bool remesh  = ctx.is_remesh();
    if (remesh) {
        if (root) std::cout << "[ETS] : Remesh is triggered." << std::endl;
        ctx.remesh_and_gridtransfer(grain_sz, ld_tol, sf_k);
    }
    if (labelled) {
        const double t2 = MPI_Wtime();
        ctx.maybe_apply_partitioning();
        if (root)
            std::cout << "[labelled-remesh] to SFC " << (t1 - t0)
                      << " s, relabel " << (MPI_Wtime() - t2) << " s, remesh "
                      << (remesh ? 1 : 0) << std::endl;
    }
    if (remesh || labelled) {
        ctx.on_mesh_changed(remesh);
        ets.sync_with_mesh();
    }
    return remesh;
}

/// Writes a checkpoint on SFC element ranges: a labelled mesh goes to SFC for
/// `write()` and is labelled again afterwards. Collective: yes. Invalidates:
/// the mesh object of a labelled mesh.
template <class Ctx, class Stepper, class Write>
void checkpoint(Ctx& ctx, Stepper& ets, Write&& write) {
    const bool labelled = ctx.is_labelled();
    if (labelled) ctx.to_sfc_partition();
    write();
    if (labelled) {
        ctx.maybe_apply_partitioning();
        ctx.on_mesh_changed(false);
        ets.sync_with_mesh();
    }
}

/// Level offset of the step ladder: the number of levels below the finest
/// that step at the finest step. A block at level blev may step at
/// 2^(lmax - blev) dt_min (its CFL step); `dt_stable(blk)` bounds it further
/// (return infinity for no bound). A block whose feasible step is not above
/// dt_min adds nothing: the finest step is the floor. Collective: yes, a
/// max-reduction over `comm`. Thread-safe: no.
/// @param dt_min  the finest step
/// @param dt_stable  callable, local block id -> largest stable step
template <class DtStable>
unsigned int level_offset(const ot::Mesh& mesh, unsigned int lmax,
                          double dt_min, DtStable&& dt_stable, MPI_Comm comm) {
    const std::vector<ot::Block>& blkList = mesh.getLocalBlockList();
    const std::vector<ot::TreeNode>& all  = mesh.getAllElements();
    unsigned int offset                   = 0;
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int blev =
            all[blkList[blk].firstElementID()].getLevel();
        const double dt_cfl      = (1u << (lmax - blev)) * dt_min;
        const double dt_feasible = std::min(dt_stable(blk), dt_cfl);
        if (dt_feasible > dt_min) {
            const unsigned int o =
                lmax - blev - std::floor(std::log2(dt_feasible / dt_min));
            offset = std::max(offset, o);
        }
    }
    unsigned int offset_max = 0;
    par::Mpi_Allreduce(&offset, &offset_max, 1, MPI_MAX, comm);
    return offset_max;
}

}  // namespace ts
