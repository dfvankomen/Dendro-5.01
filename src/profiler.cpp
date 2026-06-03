//
// Created by milinda on 10/20/17.
/**
 *@author Milinda Fernando
 *School of Computing, University of Utah
 *@brief simple profiler based on Hari's sort_profiler for bssn application.
 */
//
#include "profiler.h"

profiler_t::profiler_t() {
    seconds       = 0.0;  // openmp wall time
    p_flpops      = 0;    // papi floating point operations
    snap          = 0.0;
    num_calls     = 0;

    _pri_seconds  = 0.0;
    _pri_p_flpops = 0;
}

profiler_t::~profiler_t() {}

void profiler_t::start() {
    // Skip recording from worker threads inside a parallel region: these
    // counters are shared, so concurrent start/stop would be a data race that
    // corrupts the timing. The master thread still records a representative
    // sample; phase timers started outside parallel regions are unaffected.
    if (omp_in_parallel() && omp_get_thread_num() != 0) return;
    _pri_seconds = omp_get_wtime();
    flops_papi();
}

void profiler_t::stop() {
    if (omp_in_parallel() && omp_get_thread_num() != 0) return;
    seconds -= _pri_seconds;
    p_flpops -= _pri_p_flpops;
    snap -= _pri_seconds;

    _pri_seconds = omp_get_wtime();
    flops_papi();

    seconds += _pri_seconds;
    p_flpops += _pri_p_flpops;
    snap += _pri_seconds;
    // num_calls++;
}

void profiler_t::snapreset() {
    snap      = 0.0;
    num_calls = 0;
}

void profiler_t::clear() {
    seconds       = 0.0;
    p_flpops      = 0;
    snap          = 0.0;
    num_calls     = 0;

    _pri_seconds  = 0.0;
    _pri_p_flpops = 0;
}

void profiler_t::flops_papi() {
#ifdef HAVE_PAPI
    int retval;
    float rtime, ptime, mflops;
    retval = PAPI_flops(&rtime, &ptime, &_pri_p_flpops, &mflops);
    // assert (retval == PAPI_OK);
#else
    _pri_p_flpops = 0;
#endif
}
