#pragma once

// Thin wrapper around MPI_Init_thread used by all dendrolib entry points.
// We request MPI_THREAD_FUNNELED (the weakest level that lets the main
// thread enter an OpenMP parallel region containing no MPI calls), and
// abort cleanly if the implementation can't provide at least that.

#include <iostream>
#include <mpi.h>

namespace dendro {

inline void mpi_init(int* argc, char*** argv,
                     int required = MPI_THREAD_FUNNELED) {
    int provided = 0;
    MPI_Init_thread(argc, argv, required, &provided);
    if (provided < required) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "dendro::mpi_init: MPI implementation provided "
                         "thread-support level "
                      << provided << " but required " << required
                      << " (MPI_THREAD_FUNNELED=" << MPI_THREAD_FUNNELED << ")"
                      << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

}  // namespace dendro

// short unqualified alias at global scope for call-site brevity. if a future
// audit adopts dendro:: broadly, replace these calls with dendro::mpi_init().
inline void dendro_mpi_init(int* argc, char*** argv,
                            int required = MPI_THREAD_FUNNELED) {
    dendro::mpi_init(argc, argv, required);
}
