// Oracle: every face-to-element record matches geometry, not E2E: a fixed
// octree (4^3 level-2 octants, the one at (1,1,1) split into 8) gives each
// face's neighbors by coordinates; owner rule, face-Morton order of hanging
// neighbors, one record per local face, and the ODA flag bits by definition.
// Catches: W_BOUNDARY set on a ghost-owned boundary face with no INDEP/W_DEP
// bit; under ASan, the old record-stride overflow in buildF2EMap.
// Usage: mpirun -np <n >= 2> testF2EMap   (the map is not built on 1 rank)

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <tuple>
#include <vector>

#include "TreeNode.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"

namespace {

struct Box {
    std::array<unsigned int, 3> lo, hi;
    unsigned int lev;
};

Box box(const ot::TreeNode& o) {
    return {{o.minX(), o.minY(), o.minZ()},
            {o.maxX(), o.maxY(), o.maxZ()},
            o.getLevel()};
}

// OCT_DIR_LEFT..FRONT = -x, +x, -y, +y, -z, +z
unsigned int axisOf(unsigned int d) { return d / 2; }
bool isMaxSide(unsigned int d) { return d % 2; }
unsigned int opposite(unsigned int d) { return d ^ 1u; }

// octants of g whose opposite face lies on o's face d and overlaps it
std::vector<Box> across(const Box& o, unsigned int d,
                        const std::vector<Box>& g) {
    const unsigned int a     = axisOf(d);
    const unsigned int plane = isMaxSide(d) ? o.hi[a] : o.lo[a];
    std::vector<Box> out;
    for (const Box& q : g) {
        if ((isMaxSide(d) ? q.lo[a] : q.hi[a]) != plane) continue;
        bool overlap = true;
        for (unsigned int b = 0; b < 3; b++)
            if (b != a && !(q.lo[b] < o.hi[b] && o.lo[b] < q.hi[b]))
                overlap = false;
        if (overlap) out.push_back(q);
    }
    // documented order of hanging neighbors: Morton on the face, x fastest
    std::sort(out.begin(), out.end(), [](const Box& p, const Box& q) {
        return std::make_tuple(p.lo[2], p.lo[1], p.lo[0]) <
               std::make_tuple(q.lo[2], q.lo[1], q.lo[0]);
    });
    return out;
}

enum Fail {
    MALFORMED,
    OWNER_RULE,
    NEIGHBORS,
    DUPLICATE,
    COVERAGE,
    FLAGS,
    NUM_FAIL
};
const char* kFailName[NUM_FAIL] = {"malformed", "owner rule", "neighbors",
                                   "duplicate", "coverage",   "flags"};

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    if (npes < 2) {
        if (!rank)
            std::fprintf(stderr, "usage: mpirun -np <n >= 2> %s\n", argv[0]);
        MPI_Abort(comm, 2);
    }

    m_uiMaxDepth              = 6;
    const unsigned int eOrder = 4;
    _InitializeHcurve(m_uiDim);

    // the whole octree, known on every rank
    const unsigned int h2 = 1u << (m_uiMaxDepth - 2), h3 = h2 / 2;
    std::vector<ot::TreeNode> global;
    for (unsigned int k = 0; k < 4; k++)
        for (unsigned int j = 0; j < 4; j++)
            for (unsigned int i = 0; i < 4; i++) {
                if (i == 1 && j == 1 && k == 1) {
                    for (unsigned int c = 0; c < 8; c++)
                        global.emplace_back(h2 + (c & 1u) * h3,
                                            h2 + ((c >> 1) & 1u) * h3,
                                            h2 + ((c >> 2) & 1u) * h3, 3,
                                            m_uiDim, m_uiMaxDepth);
                } else {
                    global.emplace_back(i * h2, j * h2, k * h2, 2, m_uiDim,
                                        m_uiMaxDepth);
                }
            }
    std::vector<Box> g;
    for (const auto& o : global) g.push_back(box(o));

    ot::Mesh* mesh = ot::createMesh(
        rank ? nullptr : global.data(), rank ? 0 : global.size(), eOrder, comm,
        0, ot::SM_TYPE::FEM_DG, 4, 0.1, DENDRO_DEFAULT_SF_K);

    unsigned long fails[NUM_FAIL] = {0};
    // records seen: same-level, hanging, boundary, ghost-owned boundary
    // (premise: all four occur)
    unsigned long kinds[4]        = {0, 0, 0, 0};
    unsigned int shown[NUM_FAIL]  = {0};
    auto fail = [&](Fail f, const char* fmt, unsigned int a, unsigned int b,
                    unsigned int c = 0, unsigned int d = 0,
                    unsigned int e = 0) {
        fails[f]++;
        if (shown[f]++ < 3) {
            std::printf("  [rank %d] %s: ", rank, kFailName[f]);
            std::printf(fmt, a, b, c, d, e);
            std::printf("\n");
        }
    };

    int premise_fail = 0;
    if (!mesh->isActive() || mesh->getMPICommSize() != (unsigned int)npes) {
        premise_fail = 1;  // every rank must be active for the 2- and 3-way cut
        std::printf("  [rank %d] active %d, active ranks %u of %d\n", rank,
                    mesh->isActive(), mesh->getMPICommSize(), npes);
    } else {
        const auto& all         = mesh->getAllElements();
        const auto& f2e         = mesh->getF2EMap();
        const unsigned int lb   = mesh->getElementLocalBegin();
        const unsigned int le   = mesh->getElementLocalEnd();
        const unsigned int D    = LOOK_UP_TABLE_DEFAULT;
        const unsigned int W    = F2E_MAP_OFFSET;
        const unsigned int nRec = f2e.size() / W;
        if (f2e.size() % W)
            fail(MALFORMED, "size %u not a multiple of %u",
                 (unsigned int)f2e.size(), W);

        std::map<std::array<unsigned int, 4>, unsigned int> indexOf;
        for (unsigned int e = 0; e < all.size(); e++) {
            const Box b                                 = box(all[e]);
            indexOf[{b.lo[0], b.lo[1], b.lo[2], b.lev}] = e;
        }
        auto local = [&](const Box& b) {
            auto it = indexOf.find({b.lo[0], b.lo[1], b.lo[2], b.lev});
            return it == indexOf.end() ? D : it->second;
        };

        const unsigned int p = eOrder;
        auto faceNodeLocal   = [&](unsigned int e, unsigned int d,
                                   bool requireAll) {
            for (unsigned int u = 0; u <= p; u++)
                for (unsigned int v = 0; v <= p; v++) {
                    unsigned int ijk[3];
                    const unsigned int a = axisOf(d);
                    ijk[a]               = isMaxSide(d) ? p : 0;
                    ijk[(a + 1) % 3]     = u;
                    ijk[(a + 2) % 3]     = v;
                    const bool loc =
                        mesh->isNodeLocal(e, ijk[0], ijk[1], ijk[2]);
                    if (requireAll && !loc) return false;
                    if (!requireAll && loc) return true;
                }
            return requireAll;
        };

        std::map<std::pair<unsigned int, unsigned int>, unsigned int> seen;
        std::vector<unsigned int> cover((le - lb) * NUM_FACES, 0);
        for (unsigned int r = 0; r < nRec; r++) {
            const unsigned int* rec = &f2e[r * W];
            const unsigned int d    = rec[0] >> ODA_FLAGS_TOTAL;
            const unsigned int o    = rec[1];
            if (d >= NUM_FACES || o >= all.size()) {
                fail(MALFORMED, "record %u has dir %u", r, d);
                continue;
            }
            if (seen[{o, d}]++) fail(DUPLICATE, "owner %u dir %u", o, d);

            const Box ob              = box(all[o]);
            const std::vector<Box> nb = across(ob, d, g);
            std::vector<unsigned int> expect(W - 2, D);
            const unsigned int a      = axisOf(d);
            const unsigned int plane  = isMaxSide(d) ? ob.hi[a] : ob.lo[a];
            const bool domainBoundary = nb.empty();
            if (domainBoundary) {
                kinds[2]++;
                if (plane != 0 && plane != (1u << m_uiMaxDepth))
                    fail(NEIGHBORS, "owner %u dir %u: no neighbor off-boundary",
                         o, d);
            } else if (nb.size() == 1 && nb[0].lev == ob.lev) {
                kinds[0]++;
                expect[0] = local(nb[0]);
                if (expect[0] != D && !(o < expect[0]))
                    fail(OWNER_RULE, "same-level face owned by %u, not %u", o,
                         expect[0]);
            } else if (nb.size() == 4 && nb[0].lev == ob.lev + 1) {
                kinds[1]++;
                for (unsigned int i = 0; i < 4; i++) expect[i] = local(nb[i]);
            } else {
                fail(OWNER_RULE, "owner %u dir %u is not the coarse side", o,
                     d);
                continue;
            }
            for (unsigned int i = 0; i < W - 2; i++)
                if (rec[2 + i] != expect[i]) {
                    fail(NEIGHBORS, "owner %u dir %u", o, d);
                    break;
                }

            // flags by definition (cf. computeODAFlags, restricted to the face)
            const bool ownerLocal = o >= lb && o < le;
            const bool indep      = ownerLocal && faceNodeLocal(o, d, true);
            const bool wdep = ownerLocal ? !indep : faceNodeLocal(o, d, false);
            const bool bdy  = domainBoundary && (indep || wdep);
            const unsigned int want = (indep << ODA_INDEPENDENT_FLAG_BIT) |
                                      (wdep << ODA_W_DEPENDENT_FLAG_BIT) |
                                      (bdy << ODA_W_BOUNDARY_FLAG_BIT);
            const unsigned int got  = rec[0] & ((1u << ODA_FLAGS_TOTAL) - 1);
            if (!ownerLocal && domainBoundary) kinds[3]++;
            if (got != want)
                fail(FLAGS, "owner %u dir %u, owner local %u: bits %u, want %u",
                     o, d, ownerLocal, got, want);

            if (ownerLocal) cover[(o - lb) * NUM_FACES + d]++;
            for (unsigned int i = 2; i < W; i++)
                if (rec[i] != D && rec[i] >= lb && rec[i] < le)
                    cover[(rec[i] - lb) * NUM_FACES + opposite(d)]++;
        }
        for (unsigned int k = 0; k < cover.size(); k++)
            if (cover[k] != 1)
                fail(COVERAGE, "local element %u face %u", lb + k / NUM_FACES,
                     k % NUM_FACES);
        std::printf(
            "  [rank %d] %u records: %lu same-level, %lu hanging, %lu "
            "boundary (%lu ghost-owned)\n",
            rank, nRec, kinds[0], kinds[1], kinds[2], kinds[3]);
    }

    unsigned long gfails[NUM_FAIL], gkinds[4];
    int gpremise = 0;
    MPI_Allreduce(fails, gfails, NUM_FAIL, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    MPI_Allreduce(kinds, gkinds, 4, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    MPI_Allreduce(&premise_fail, &gpremise, 1, MPI_INT, MPI_MAX, comm);
    unsigned long total = 0;
    for (unsigned int f = 0; f < NUM_FAIL; f++) total += gfails[f];
    // every record kind must occur, or a check could pass vacuously
    const bool vacuous = !gkinds[0] || !gkinds[1] || !gkinds[2] || !gkinds[3];
    if (!rank) {
        for (unsigned int f = 0; f < NUM_FAIL; f++)
            if (gfails[f]) std::printf("  %s: %lu\n", kFailName[f], gfails[f]);
        const bool ok = !total && !gpremise && !vacuous;
        std::printf("testF2EMap np %d: %lu failed%s%s, %s\n", npes, total,
                    gpremise ? ", premise not met" : "",
                    vacuous ? ", a record kind is missing" : "",
                    ok ? "PASS" : "FAIL");
    }
    const int rc = (total || gpremise || vacuous) ? 1 : 0;
    delete mesh;
    MPI_Finalize();
    return rc;
}
