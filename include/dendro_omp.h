#pragma once

// Per-site OpenMP pragma wrapper for dendrolib. Use these macros in hot-path
// loops where you want to parallelize over blocks / elements. When
// DENDRO_USE_OMP is defined (default ON via CMake), they expand to the
// corresponding `_Pragma("omp ...")` directive. When disabled at compile
// time, they expand to nothing — useful for bisecting suspected races
// without rebuilding the whole tree.
//
// usage:
//     DENDRO_OMP_PARALLEL_FOR_DYNAMIC(4)
//     for (unsigned int blk = 0; blk < blkList.size(); blk++) { ... }
//
// OpenMP is already a REQUIRED link dependency (see CMakeLists.txt); this
// header only toggles whether the pragmas are emitted

#ifndef DENDRO_USE_OMP
#define DENDRO_USE_OMP 1
#endif

#if DENDRO_USE_OMP
#include <omp.h>
#define DENDRO_OMP_PRAGMA(x) _Pragma(#x)
#define DENDRO_OMP_PARALLEL_FOR DENDRO_OMP_PRAGMA(omp parallel for)
#define DENDRO_OMP_PARALLEL_FOR_DYNAMIC(chunk) \
    DENDRO_OMP_PRAGMA(omp parallel for schedule(dynamic, chunk))
#define DENDRO_OMP_PARALLEL_FOR_STATIC \
    DENDRO_OMP_PRAGMA(omp parallel for schedule(static))
#else
#define DENDRO_OMP_PARALLEL_FOR
#define DENDRO_OMP_PARALLEL_FOR_DYNAMIC(chunk)
#define DENDRO_OMP_PARALLEL_FOR_STATIC
#endif
