// ctx.h default block hooks: a leaf without a hook gets 0, a leaf with one gets
// its own result (before the defaults, a leaf without every hook did not compile)
#include "ctx.h"
#include <cstdio>
struct Bare : ts::Ctx<Bare, double, unsigned int> {
    int rhs_blk(const double*, double*, unsigned int, unsigned int, double) { return 1; }
};
struct Full : ts::Ctx<Full, double, unsigned int> {
    int rhs_blk(const double*, double*, unsigned int, unsigned int, double) { return 1; }
    int pre_stage_blk(double*, unsigned int, unsigned int, double) { return 7; }
    int post_timestep_blk(double*, unsigned int, unsigned int, double) { return 9; }
};
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Bare b; Full f; double x = 0;
    ts::Ctx<Bare, double, unsigned int>& cb = b;
    ts::Ctx<Full, double, unsigned int>& cf = f;
    int bad = 0;
    bad += cb.pre_stage_blk(&x, 1, 0, 0) != 0;
    bad += cb.post_stage_blk(&x, 1, 0, 0) != 0;
    bad += cb.pre_timestep_blk(&x, 1, 0, 0) != 0;
    bad += cb.post_timestep_blk(&x, 1, 0, 0) != 0;
    bad += b.pre_stage_blk(&x, 1, 0, 0) != 0;          // through the leaf type, as ENUTS calls it
    bad += cf.pre_stage_blk(&x, 1, 0, 0) != 7;
    bad += cf.post_timestep_blk(&x, 1, 0, 0) != 9;
    bad += cf.post_stage_blk(&x, 1, 0, 0) != 0;
    printf("testCtxHooks %s (%d bad)\n", bad ? "FAIL" : "PASS", bad);
    MPI_Finalize();
    return bad;
}
