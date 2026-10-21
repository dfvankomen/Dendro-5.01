/**
 * @file testLtsClock.cpp
 * @brief Gates ts::Clock (ODE/include/lts.h), the driver's event rules under LTS.
 *
 *   - the seeded remesh event fires at the coarse steps today's driver fires
 *     it at (the remesh test skips step 0; one interval = freq fine steps);
 *   - a restart resumed from the start of the last coarse step fires at the
 *     same steps as the uninterrupted run;
 *   - the step size is read at every call: after it halves, intervals follow
 *     the new step;
 *   - the checkpoint slot alternates by interval index under LTS and is -1
 *     under the global stepper; the global stepper fires every freq steps.
 *
 * LTS_CLOCK_SABOTAGE=1 builds leave the remesh event unseeded (it then fires
 * at the first coarse step); that build must fail.
 */

#include <cstdio>
#include <vector>

#include "lts.h"

#ifndef LTS_CLOCK_SABOTAGE
#define LTS_CLOCK_SABOTAGE 0
#endif

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        g_fail++;
        std::printf("  FAIL: %s\n", what);
    }
}

double g_dt = 1.0 / 64;  // the finest step, as a driver global a remesh may change

ts::Clock makeClock(bool lts) {
    ts::Clock c;
    c.dt_fine = [] { return g_dt; };
    c.lts     = lts;
    return c;
}

void seed(const ts::Clock& clk, ts::Cadence& c, unsigned int freq) {
#if LTS_CLOCK_SABOTAGE != 1
    clk.seed_remesh(c, freq);
#endif
}

// steps (1 .. nsteps) at which the remesh event fires; coarse step = ratio fine steps
std::vector<long long> remeshSteps(unsigned int freq, unsigned int ratio, long long nsteps) {
    ts::Clock clk = makeClock(true);
    ts::Cadence c;
    seed(clk, c, freq);
    std::vector<long long> out;
    for (long long step = 0; step <= nsteps; step++) {
        const double time = step * ratio * g_dt;
        if (step != 0 && clk.due(c, freq, step, time)) out.push_back(step);
    }
    return out;
}

void testSeeded() {
    // freq 10, coarse step 8: boundaries at 10, 20, 30, ... fine steps; a step
    // fires when its time reaches the next boundary at or after the last one
    const std::vector<long long> want = {2, 3, 4, 5, 7, 8, 9, 10, 12};
    check(remeshSteps(10, 8, 12) == want, "seeded remesh fires at today's steps 2 3 4 5 7 8 9 10 12");
}

void testResume() {
    const unsigned int freq = 10, ratio = 8;
    const std::vector<long long> full = remeshSteps(freq, ratio, 40);
    for (long long k : {5LL, 6LL, 13LL}) {
        // restart after step k: resume from the start of coarse step k
        ts::Clock clk = makeClock(true);
        ts::Cadence c;
        clk.resume(c, freq, k * ratio * g_dt - ratio * g_dt);
        std::vector<long long> got, want;
        for (long long step = k; step <= 40; step++)
            if (clk.due(c, freq, step, step * ratio * g_dt)) got.push_back(step);
        for (long long s : full)
            if (s >= k) want.push_back(s);
        check(got == want, "a resumed run fires at the uninterrupted run's steps");
    }
}

void testStepChange() {
    // the step halves before step 6: from then on an interval is 10 new steps
    ts::Clock clk = makeClock(true);
    ts::Cadence a, b;
    clk.seed_remesh(a, 10);
    b.next          = a.next;
    const double d0 = g_dt;
    double time     = 0.0;
    bool same       = true;
    for (long long step = 1; step <= 20; step++) {
        if (step == 6) g_dt = d0 / 2;
        time += 8 * g_dt;
        // b passes the step size the caller holds now, as Cadence::due needs
        same = same && clk.due(a, 10, step, time) == b.due(10, step, time, g_dt, true);
    }
    g_dt = d0;
    check(same, "the clock reads the current step at every call");
}

void testSlotsAndGlobal() {
    ts::Clock clk = makeClock(true);
    const unsigned int freq = 4;
    for (unsigned int i = 0; i < 6; i++)
        check(clk.checkpoint_slot(freq, (i * freq + 0.5) * g_dt) == (int)(i % 2),
              "checkpoint slot = interval index mod 2");
    check(clk.checkpoint_slot(freq, 2 * freq * g_dt) == 0, "slot at an interval boundary");
    ts::Clock glob = makeClock(false);
    check(glob.checkpoint_slot(freq, 3.0) == -1, "global stepper: slot -1 (step parity)");
    ts::Cadence c;
    std::vector<long long> got;
    for (long long step = 0; step <= 12; step++)
        if (glob.due(c, freq, step, step * g_dt)) got.push_back(step);
    check(got == std::vector<long long>({0, 4, 8, 12}), "global stepper fires every freq steps");
    check(!glob.due(c, 0, 0, 0.0) && !clk.due(c, 0, 1, 1.0), "freq 0 never fires");
}

}  // namespace

int main() {
    testSeeded();
    testResume();
    testStepChange();
    testSlotsAndGlobal();
    std::printf("testLtsClock: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
