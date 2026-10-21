/**
 * @file testLtsContract.cpp
 * @brief Gates the LTS solver contract helpers of ODE/include/lts.h.
 *
 *   - snap_ratio_cap and step_ratio: power-of-two snapping, the cap (0, 1, 64),
 *     the level offset, and the shift when a remesh changes the finest level.
 *   - level_offset on a real multi-level mesh: the CFL clamp (no stability
 *     bound gives offset 0) and bounds of 1, 3, 4 and 16 finest steps, against
 *     an integer recomputation; collective, so the 2-rank run checks the max.
 *   - BlockView: every block's box and padded box bitwise equal to the
 *     pre-tidy GRIDX_TO_X expression BSSN used.
 *   - for_each_enforced_node: every node of an interior block once (with the
 *     right interior flag), only the interior of a domain-boundary block.
 *   - lts_options_valid (iface 3 | 15) and lts_tableau_supported (RK3, RK4;
 *     RK5 refused).
 *
 * LTS_CONTRACT_SABOTAGE=1..5 builds break one helper each (cap ignored,
 * BlockView one ulp off, boundary padding visited, no CFL clamp, RK5
 * accepted); each must fail.
 *
 * Usage: testLtsContract [maxDepth] [waveletTol]
 */

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "lts.h"
#include "mesh.h"
#include "meshUtils.h"
#include "octUtils.h"

#ifndef LTS_CONTRACT_SABOTAGE
#define LTS_CONTRACT_SABOTAGE 0
#endif

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        g_fail++;
        std::printf("  FAIL: %s\n", what);
    }
}

// the helpers under test, through one indirection so a sabotage build can
// swap in a broken one
unsigned int stepRatio(unsigned int blev, unsigned int lmax, unsigned int off,
                       unsigned int cap) {
#if LTS_CONTRACT_SABOTAGE == 1
    cap = 0;
#endif
    return ts::step_ratio(blev, lmax, off, cap);
}

ts::BlockView blockView(const ot::Block& b, const Point& omin, const Point& omax,
                        const Point& dmin, const Point& dmax, unsigned int pad) {
    ts::BlockView v(b, omin, omax, dmin, dmax, pad);
#if LTS_CONTRACT_SABOTAGE == 2
    v.lo[0] = std::nextafter(v.lo[0], 1e300);
#endif
    return v;
}

template <class F>
void forEachEnforced(const ts::BlockView& b, F&& f) {
#if LTS_CONTRACT_SABOTAGE == 3
    ts::BlockView all = b;
    all.bflag         = 0;
    ts::for_each_enforced_node(all, f);
#else
    ts::for_each_enforced_node(b, f);
#endif
}

template <class DtStable>
unsigned int levelOffset(const ot::Mesh& mesh, unsigned int lmax, double dt_min,
                         DtStable&& dt_stable, MPI_Comm comm) {
#if LTS_CONTRACT_SABOTAGE == 4
    // no CFL clamp: a block with no stability bound asks for an infinite step
    auto unclamped = [&](unsigned int blk) {
        const double s = dt_stable(blk);
        return std::isinf(s) ? 1e30 * dt_min : s;
    };
    const auto& blkList = mesh.getLocalBlockList();
    const auto& all     = mesh.getAllElements();
    unsigned int off    = 0;
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int blev = all[blkList[blk].firstElementID()].getLevel();
        const double f          = unclamped(blk);
        if (f > dt_min) {
            const double o = (double)(lmax - blev) - std::floor(std::log2(f / dt_min));
            off = std::max(off, o < 0 ? 1000u : (unsigned int)o);
        }
    }
    unsigned int g = 0;
    MPI_Allreduce(&off, &g, 1, MPI_UNSIGNED, MPI_MAX, comm);
    return g;
#else
    return ts::level_offset(mesh, lmax, dt_min, dt_stable, comm);
#endif
}

bool tableauSupported(ts::ETSType t) {
#if LTS_CONTRACT_SABOTAGE == 5
    if (t == ts::ETSType::RK5) return true;
#endif
    return ts::lts_tableau_supported(t);
}

void testRatios() {
    check(ts::snap_ratio_cap(0) == 0, "snap(0) = 0 (no cap)");
    check(ts::snap_ratio_cap(1) == 1, "snap(1) = 1");
    check(ts::snap_ratio_cap(3) == 2, "snap(3) = 2");
    check(ts::snap_ratio_cap(64) == 64, "snap(64) = 64");
    check(ts::snap_ratio_cap(100) == 64, "snap(100) = 64");

    const unsigned int lmax = 12;
    for (unsigned int blev = 2; blev <= lmax; blev++) {
        const unsigned int k = lmax - blev;
        for (unsigned int off : {0u, 2u}) {
            const unsigned int free = (k <= off) ? 1u : (1u << (k - off));
            check(stepRatio(blev, lmax, off, 0) == free, "no cap: 2^(lmax - blev - offset)");
            check(stepRatio(blev, lmax, off, 1) == 1, "cap 1: every block at the finest step");
            check(stepRatio(blev, lmax, off, 64) == std::min(free, 64u), "cap 64 clamps");
        }
    }
    // a remesh that adds a finer level doubles every coarser block's ratio
    for (unsigned int blev = 3; blev < 10; blev++)
        check(stepRatio(blev, 11, 0, 0) == 2 * stepRatio(blev, 10, 0, 0),
              "finest level + 1 doubles the ratio");
}

// offset from an integer recomputation: a bound of c finest steps, clamped by
// the block's CFL step 2^(lmax - blev)
unsigned int expectedOffset(const ot::Mesh& mesh, unsigned int lmax, double c,
                            MPI_Comm comm) {
    const auto& blkList = mesh.getLocalBlockList();
    const auto& all     = mesh.getAllElements();
    unsigned int off    = 0;
    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        const unsigned int k = lmax - all[blkList[blk].firstElementID()].getLevel();
        unsigned int p       = 0;  // largest p with 2^p <= min(c, 2^k)
        while (p < k && (double)(1u << (p + 1)) <= c) p++;
        if (std::min(c, (double)(1u << k)) > 1.0) off = std::max(off, k - p);
    }
    unsigned int g = 0;
    MPI_Allreduce(&off, &g, 1, MPI_UNSIGNED, MPI_MAX, comm);
    return g;
}

void testMesh(ot::Mesh& mesh, unsigned int eOrder, int rank) {
    unsigned int lmin, lmax;
    mesh.computeMinMaxLevel(lmin, lmax);
    MPI_Comm comm       = mesh.getMPICommunicator();
    const double dt_min = 0.125;

    const double inf = std::numeric_limits<double>::infinity();
    check(levelOffset(mesh, lmax, dt_min, [&](unsigned int) { return inf; }, comm) == 0,
          "level_offset: no stability bound -> 0 (CFL clamp)");
    for (double c : {1.0, 3.0, 4.0, 16.0}) {
        const unsigned int got =
            levelOffset(mesh, lmax, dt_min, [&](unsigned int) { return c * dt_min; }, comm);
        const unsigned int want = expectedOffset(mesh, lmax, c, comm);
        if (got != want && !rank)
            std::printf("  level_offset c=%g: %u, expected %u\n", c, got, want);
        check(got == want, "level_offset with a bound of c finest steps");
    }

    // BlockView against the pre-tidy expression ((R / Rg) * (xg - omin)) + dmin
    const double ext = (double)(1u << m_uiMaxDepth);
    const Point omin(0.0, 0.0, 0.0), omax(ext, ext, ext);
    const Point dmin(-400.0, -250.0, -100.0), dmax(400.0, 350.0, 100.0);
    const double om[3] = {0.0, 0.0, 0.0}, oM[3] = {ext, ext, ext};
    const double dm[3] = {-400.0, -250.0, -100.0}, dM[3] = {400.0, 350.0, 100.0};
    const unsigned int pad = eOrder / 2;
    unsigned int nBdy = 0, nInt = 0;
    for (const ot::Block& b : mesh.getLocalBlockList()) {
        const ts::BlockView v = blockView(b, omin, omax, dmin, dmax, pad);
        const ot::TreeNode n  = b.getBlockNode();
        const unsigned int g0[3] = {n.minX(), n.minY(), n.minZ()};
        const unsigned int g1[3] = {n.maxX(), n.maxY(), n.maxZ()};
        const double dx[3] = {b.computeDx(dmin, dmax), b.computeDy(dmin, dmax),
                              b.computeDz(dmin, dmax)};
        bool same = true;
        for (int d = 0; d < 3; d++) {
            const double lo = (((dM[d] - dm[d]) / (oM[d] - om[d])) * (g0[d] - om[d])) + dm[d];
            const double hi = (((dM[d] - dm[d]) / (oM[d] - om[d])) * (g1[d] - om[d])) + dm[d];
            same = same && v.lo[d] == lo && v.hi[d] == hi && v.dx[d] == dx[d] &&
                   v.ptmin[d] == lo - pad * dx[d] && v.ptmax[d] == hi + pad * dx[d];
        }
        check(same, "BlockView bitwise = the GRIDX_TO_X expression");

        std::vector<int> seen(v.nn, 0);
        bool flagOk = true;
        forEachEnforced(v, [&](unsigned int node, bool interior) {
            seen[node]++;
            const unsigned int i = node % v.sz[0], j = (node / v.sz[0]) % v.sz[1],
                               k = node / (v.sz[0] * v.sz[1]);
            flagOk = flagOk && interior == v.interior(i, j, k);
        });
        bool once = true;
        for (unsigned int node = 0; node < v.nn; node++) {
            const unsigned int i = node % v.sz[0], j = (node / v.sz[0]) % v.sz[1],
                               k = node / (v.sz[0] * v.sz[1]);
            const int want = (v.bflag && !v.interior(i, j, k)) ? 0 : 1;
            once = once && seen[node] == want;
        }
        check(once, "for_each_enforced_node visits each enforced node once");
        check(flagOk, "for_each_enforced_node interior flag");
        (v.bflag ? nBdy : nInt)++;
    }
    unsigned int loc[2] = {nBdy, nInt}, glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_UNSIGNED, MPI_SUM, comm);
    check(glob[0] > 0 && glob[1] > 0, "the mesh has both boundary and interior blocks");
}

void testOptions() {
    ts::LtsOptions o;
    check(ts::lts_options_valid(o), "default options valid (iface 3)");
    o.iface = 15;
    check(ts::lts_options_valid(o), "iface 15 valid");
    for (unsigned int bad : {0u, 1u, 7u, 16u}) {
        o.iface = bad;
        check(!ts::lts_options_valid(o), "iface other than 3 | 15 refused");
    }
    check(tableauSupported(ts::ETSType::RK3), "RK3 supported");
    check(tableauSupported(ts::ETSType::RK4), "RK4 supported");
    check(!tableauSupported(ts::ETSType::RK5), "RK5 refused");
    check(!tableauSupported(ts::ETSType::RK45_DORMAND_PRINCE), "embedded RK45 refused");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    _InitializeHcurve(m_uiDim);
    m_uiMaxDepth             = (argc > 1) ? std::atoi(argv[1]) : 8;
    const double wavelet_tol = (argc > 2) ? std::atof(argv[2]) : 1e-3;

    testRatios();
    testOptions();

    const unsigned int eOrder = 6;
    const double scale        = 1.0 / (double)(1u << m_uiMaxDepth);
    std::function<double(double, double, double)> refine_fn = [scale](double x, double y,
                                                                       double z) {
        const double dx = x * scale - 0.3, dy = y * scale - 0.4, dz = z * scale - 0.5;
        return std::exp(-4000.0 * (dx * dx + dy * dy + dz * dz));
    };
    std::vector<ot::TreeNode> tmpNodes;
    function2Octree(refine_fn, tmpNodes, m_uiMaxDepth, wavelet_tol, eOrder, MPI_COMM_WORLD);
    ot::Mesh* mesh = ot::createMesh(tmpNodes.data(), tmpNodes.size(), eOrder, MPI_COMM_WORLD, 1,
                                    ot::SM_TYPE::FDM, DENDRO_DEFAULT_GRAIN_SZ, 0.1,
                                    DENDRO_DEFAULT_SF_K);
    if (mesh->isActive()) {
        unsigned int lmin, lmax;
        mesh->computeMinMaxLevel(lmin, lmax);
        if (!rank) std::printf("  mesh levels %u..%u\n", lmin, lmax);
        check(lmax >= lmin + 3, "the mesh spans at least four levels");
        testMesh(*mesh, eOrder, rank);
    }
    delete mesh;

    int fails = g_fail, all = 0;
    MPI_Allreduce(&fails, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (!rank) std::printf("testLtsContract: %s (%d failed checks)\n", all ? "FAIL" : "PASS", all);
    MPI_Finalize();
    return all ? 1 : 0;
}
