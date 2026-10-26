// BlockAsyncVector::createVec must hand back zeroed storage: ENUTS can read
// block padding before any sync writes it, and stale heap there turns into
// NaN through the derivative stencils. Each block is allocated where a dirty
// one was just freed, so the gate does not pass on a lucky zero page.
// Exit code 0 = pass.
#include <cstdio>
#include <limits>

#include "blkAsync.h"

static unsigned long nonzero(const double* v, size_t n) {
    unsigned long c = 0;
    for (size_t i = 0; i < n; i++)
        if (v[i] != 0.0) c++;
    return c;
}

int main() {
    const double nan  = std::numeric_limits<double>::quiet_NaN();
    unsigned long bad = 0, checked = 0;
    // cube sizes in the range of LTS blocks, 1 and 2 dof
    for (unsigned int n : {13u, 17u, 21u}) {
        for (unsigned int dof : {1u, 2u}) {
            const unsigned int sz[3] = {n, n, n};
            const size_t len         = (size_t)dof * n * n * n;

            // a raw chunk of the same size, dirtied and freed; the guard keeps
            // it from merging back into the top of the heap
            double* dirty            = new double[len];
            for (size_t i = 0; i < len; i++) dirty[i] = nan;
            double* guard = new double[1];
            delete[] dirty;

            ts::BlockAsyncVector<double> b;
            b.createVec(0, sz, false, ts::BLK_ASYNC_VEC_MODE::BLK_UNZIP, dof);
            const unsigned long c1 = nonzero(b.data(), len);

            // a block the stepper wrote, destroyed and created again (remesh)
            double* w              = const_cast<double*>(b.data());
            for (size_t i = 0; i < len; i++) w[i] = nan;
            b.destroyVec();
            b.createVec(0, sz, false, ts::BLK_ASYNC_VEC_MODE::BLK_UNZIP, dof);
            const unsigned long c2 = nonzero(b.data(), len);
            delete[] guard;

            checked += 2;
            if (c1) bad++;
            if (c2) bad++;
            std::printf(
                "  n=%2u dof=%u  nonzero after dirty free: %lu, after "
                "recreate: %lu of %zu  -> %s\n",
                n, dof, c1, c2, len, (c1 || c2) ? "NOT ZEROED" : "ok");
        }
    }
    std::printf("testBlkAsyncZeroInit: %lu of %lu checks failed\n", bad,
                checked);
    return bad ? 1 : 0;
}
