// Oracle: each active rank's octree and a 3-variable CG vector, written with
// io::checkpoint and read back as solver restore does, round-trip bit for bit
// (octree vs. what was written; vectors vs. the function on the rebuilt mesh);
// a truncated file or a different mesh is rejected. Catches: offset or stride
// errors in multi-variable files. Usage: mpirun -np <n> testCheckpointRoundTrip

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "TreeNode.h"
#include "checkPoint.h"
#include "dendro.h"
#include "mesh.h"
#include "meshUtils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
    m_uiMaxDepth         = 6;
    const unsigned int p = 6, nv = 3;
    _InitializeHcurve(m_uiDim);
    // named by active rank, as bssnCtx names its checkpoints: the active ranks
    // of a createMesh mesh need not be global ranks 0..activeNpes-1
    auto name = [npes](int a, const char* ext) {
        return "ckpt_rt_np" + std::to_string(npes) + "_a" + std::to_string(a) +
               ext;
    };

    std::function<double(double, double, double)> fr = [](double x, double y,
                                                          double z) {
        const double s  = 1.0 / (1u << m_uiMaxDepth);
        const double dx = x * s - 0.4, dy = y * s - 0.6, dz = z * s - 0.5;
        return std::exp(-80.0 * (dx * dx + dy * dy + dz * dz));
    };
    std::vector<ot::TreeNode> tmp;
    function2Octree(fr, tmp, m_uiMaxDepth, 1e-3, p, comm);
    ot::Mesh* mesh =
        ot::createMesh(tmp.data(), tmp.size(), p, comm, 1, ot::SM_TYPE::FDM,
                       DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
    int activeNpes = mesh->isActive() ? mesh->getMPICommSize() : 0;
    MPI_Allreduce(MPI_IN_PLACE, &activeNpes, 1, MPI_INT, MPI_MAX, comm);

    // distinct values per variable and node, so a swap or shift shows
    std::function<void(double, double, double, double*)> fv =
        [](double x, double y, double z, double* v) {
            for (unsigned int d = 0; d < nv; d++)
                v[d] = std::sin(1.3 * x + d) * std::cos(0.7 * y - d) + z * d;
        };

    // per active rank: local element count and an order-free hash of the keys
    auto keys = [](const ot::Mesh* m, std::uint64_t* out) {
        const auto& e = m->getAllElements();
        out[0]        = m->getElementLocalEnd() - m->getElementLocalBegin();
        out[1]        = 0;
        for (unsigned int i = m->getElementLocalBegin();
             i < m->getElementLocalEnd(); i++)
            out[1] += (std::uint64_t(e[i].minX()) * 73856093u) ^
                      (std::uint64_t(e[i].minY()) * 19349663u) ^
                      (std::uint64_t(e[i].minZ()) * 83492791u) ^
                      e[i].getLevel();
    };
    std::vector<std::uint64_t> orig(2 * activeNpes, 0);

    int bad = 0;
    if (mesh->isActive()) {
        const int a   = mesh->getMPIRank();
        const auto& e = mesh->getAllElements();
        keys(mesh, &orig[2 * a]);
        std::vector<ot::TreeNode> local(
            e.begin() + mesh->getElementLocalBegin(),
            e.begin() + mesh->getElementLocalEnd());
        double* vec     = mesh->createCGVector<double>(fv, nv);
        const size_t nN = mesh->getDegOfFreedom();
        const double* vars[nv];
        for (unsigned int d = 0; d < nv; d++) vars[d] = vec + d * nN;
        bad += io::checkpoint::writeOctToFile(name(a, ".oct").c_str(),
                                              local.data(), local.size()) != 0;
        bad += io::checkpoint::writeVecToFile(name(a, ".v1").c_str(), mesh,
                                              vec) != 0;
        bad += io::checkpoint::writeVecToFile(name(a, ".vn").c_str(), mesh,
                                              vars, nv) != 0;
        std::vector<ot::TreeNode> back;
        bad +=
            io::checkpoint::readOctFromFile(name(a, ".oct").c_str(), back) != 0;
        if (back.size() != local.size() ||
            std::memcmp(back.data(), local.data(),
                        local.size() * sizeof(ot::TreeNode)))
            bad++;
        delete[] vec;
    }
    MPI_Allreduce(MPI_IN_PLACE, orig.data(), 2 * activeNpes, MPI_UINT64_T,
                  MPI_SUM, comm);
    MPI_Barrier(comm);  // every file is written before any rank reads one

    // restore: global ranks below activeNpes read the octree and rebuild
    std::vector<ot::TreeNode> back;
    if (rank < activeNpes)
        bad += io::checkpoint::readOctFromFile(name(rank, ".oct").c_str(),
                                               back) != 0;
    ot::Mesh* restored = new ot::Mesh(back, 1, p, activeNpes, comm);
    if (restored->isActive()) {
        const int a = restored->getMPIRank();
        std::uint64_t mine[2];
        keys(restored, mine);  // same elements as the original active rank a
        bad += mine[0] != orig[2 * a] || mine[1] != orig[2 * a + 1];
        const size_t nN = restored->getDegOfFreedom();
        double* want    = restored->createCGVector<double>(fv, nv);
        double* r       = restored->createCGVector<double>(0.0, nv);
        double* vars[nv];
        for (unsigned int d = 0; d < nv; d++) vars[d] = r + d * nN;
        const size_t b = restored->getNodeLocalBegin(),
                     n = restored->getNumLocalMeshNodes();
        bad += io::checkpoint::readVecFromFile(name(a, ".vn").c_str(), restored,
                                               vars, nv) != 0;
        for (unsigned int d = 0; d < nv; d++)
            bad += std::memcmp(r + d * nN + b, want + d * nN + b,
                               n * sizeof(double)) != 0;
        std::memset(r, 0, nN * sizeof(double));
        bad += io::checkpoint::readVecFromFile(name(a, ".v1").c_str(), restored,
                                               r) != 0;
        bad += std::memcmp(r + b, want + b, n * sizeof(double)) != 0;

        // a file cut short by one value must be rejected
        FILE* src = fopen(name(a, ".vn").c_str(), "rb");
        FILE* dst = fopen(name(a, ".cut").c_str(), "wb");
        if (!src || !dst || n == 0) {
            bad++;
        } else {
            std::vector<char> buf(3 * sizeof(unsigned int) +
                                  (nv * n - 1) * sizeof(double));
            const size_t got = fread(buf.data(), 1, buf.size(), src);
            bad += got != buf.size();  // the file holds at least this much
            fwrite(buf.data(), 1, got, dst);
        }
        if (src) fclose(src);
        if (dst) fclose(dst);
        bad += io::checkpoint::readVecFromFile(name(a, ".cut").c_str(),
                                               restored, vars, nv) == 0;
        delete[] want;
        delete[] r;
    }
    // a vector file read onto a different mesh must be rejected
    ot::Mesh* other =
        ot::createMesh(tmp.data(), tmp.size(), 4, comm, 1, ot::SM_TYPE::FDM,
                       DENDRO_DEFAULT_GRAIN_SZ, 0.1, DENDRO_DEFAULT_SF_K);
    int mismatchRan = 0;
    if (other->isActive() && other->getMPIRank() < activeNpes) {
        double* o = other->createCGVector<double>(0.0, 1);
        bad += io::checkpoint::readVecFromFile(
                   name(other->getMPIRank(), ".v1").c_str(), other, o) == 0;
        mismatchRan = 1;
        delete[] o;
    }
    MPI_Allreduce(MPI_IN_PLACE, &mismatchRan, 1, MPI_INT, MPI_MAX, comm);
    if (!rank && !mismatchRan) bad++;  // the mismatch check must run somewhere

    int bad_g = 0;
    MPI_Allreduce(&bad, &bad_g, 1, MPI_INT, MPI_SUM, comm);
    if (rank < activeNpes)
        for (const char* ext : {".oct", ".v1", ".vn", ".cut"})
            std::remove(name(rank, ext).c_str());
    if (!rank)
        std::printf("testCheckpointRoundTrip np %d: %d failed check(s): %s\n",
                    npes, bad_g, bad_g ? "FAIL" : "PASS");
    delete other;
    delete restored;
    delete mesh;
    MPI_Finalize();
    return bad_g ? 1 : 0;
}
