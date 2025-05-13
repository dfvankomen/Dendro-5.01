
#include "testMatMul.h"

#include <chrono>
#include <cstdint>
#include <iomanip>

#include "derivatives/derivs_utils.h"
#include "derivs_explicit.h"
#include "lapac.h"

void print_3d_mat(const double* matrix, uint32_t nx, uint32_t ny, uint32_t nz) {
    // Set precision for floating-point output
    std::cout << std::fixed << std::setprecision(4);

    for (int z = 0; z < nz; ++z) {
        std::cout << "Layer " << z + 1 << ":" << std::endl;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                // Calculate the index in the 1D vector for the current (x,y,z)
                // coordinate
                int index = x + y * nx + z * nx * ny;
                std::cout << std::setw(10) << matrix[index] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_2d_mat(const double* matrix, uint32_t nx, uint32_t ny) {
    // Set precision for floating-point output
    std::cout << std::fixed << std::setprecision(4);

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            // Calculate the index in the 1D vector for the current (x,y,z)
            // coordinate
            int index = x + y * nx;
            std::cout << std::setw(10) << matrix[index] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

bool test_correctness_x() {
    static uint32_t nx = 7;
    static uint32_t ny = 8;
    static uint32_t nz = 9;

    double* fullBlock  = new double[nx * ny * nz];

    double* matrix_x   = new double[nx * nx]{};

    uint32_t tracker   = 0;
    for (uint32_t k = 0; k < nz; ++k)
        for (uint32_t j = 0; j < ny; ++j)
            for (uint32_t i = 0; i < nx; ++i) {
                fullBlock[INDEX_3D(i, j, k)] = tracker;
                tracker++;
            }

    // identity test for now
    for (uint32_t j = 0; j < nx; ++j) {
        for (uint32_t i = 0; i < nx; ++i) {
            if (i == j) {
                matrix_x[INDEX_N2D(i, j, nx)] = 1.0;
            }
        }
    }

#if 0
    print_2d_mat(matrix_x, nx, nx);
#endif

    // now we call the derivatives
    double* results             = new double[nx * ny * nz];
    double* workspace           = new double[nx * ny * nz];

    static const uint32_t sz[3] = {nx, ny, nz};

#if 0
    print_3d_mat(fullBlock, nx, ny, nz);
#endif

    dendroderivs::matmul_x_dim(matrix_x, results, fullBlock, 1.0, sz, 0);

#if 0
    std::cout << "RESULTS" << std::endl;
    print_3d_mat(results, nx, ny, nz);
#endif

    bool calc_passed = true;

    // do a check to see if they're accurate
    for (uint32_t k = 0; k < nz && calc_passed; ++k)
        for (uint32_t j = 0; j < ny && calc_passed; ++j)
            for (uint32_t i = 0; i < nx && calc_passed; ++i) {
                double diff = std::abs(fullBlock[INDEX_3D(i, j, k)] -
                                       results[INDEX_3D(i, j, k)]);

                if (diff > 1e-10) {
                    std::cout << "Value failed at (" << i << ", " << j << ", "
                              << k << "): " << diff << " ("
                              << fullBlock[INDEX_3D(i, j, k)] << ", "
                              << results[INDEX_3D(i, j, k)] << ")" << std::endl;
                    calc_passed = false;
                    break;
                }
            }

    delete[] results;
    delete[] workspace;
    delete[] fullBlock;
    delete[] matrix_x;

    return calc_passed;
}

bool test_correctness_y() {
    static uint32_t nx = 7;
    static uint32_t ny = 8;
    static uint32_t nz = 9;

    double* fullBlock  = new double[nx * ny * nz];

    double* matrix_y   = new double[ny * ny]{};

    uint32_t tracker   = 0;
    for (uint32_t k = 0; k < nz; ++k)
        for (uint32_t j = 0; j < ny; ++j)
            for (uint32_t i = 0; i < nx; ++i) {
                fullBlock[INDEX_3D(i, j, k)] = tracker;
                tracker++;
            }

    for (uint32_t j = 0; j < ny; ++j) {
        for (uint32_t i = 0; i < ny; ++i) {
            if (i == j) {
                matrix_y[INDEX_N2D(i, j, ny)] = 1.0;
            }
        }
    }

#if 0
    print_2d_mat(matrix_y, ny, ny);
#endif

    // now we call the derivatives
    double* results             = new double[nx * ny * nz];
    double* workspace           = new double[nx * ny * nz];

    static const uint32_t sz[3] = {nx, ny, nz};

#if 0
    print_3d_mat(fullBlock, nx, ny, nz);
#endif

    dendroderivs::matmul_y_dim(matrix_y, results, fullBlock, 1.0, sz, workspace,
                               0);

#if 0
    std::cout << "RESULTS" << std::endl;
    print_3d_mat(results, nx, ny, nz);
#endif

    bool calc_passed = true;

    // do a check to see if they're accurate
    for (uint32_t k = 0; k < nz && calc_passed; ++k)
        for (uint32_t j = 0; j < ny && calc_passed; ++j)
            for (uint32_t i = 0; i < nx && calc_passed; ++i) {
                double diff = std::abs(fullBlock[INDEX_3D(i, j, k)] -
                                       results[INDEX_3D(i, j, k)]);

                if (diff > 1e-10) {
                    std::cout << "Value failed at (" << i << ", " << j << ", "
                              << k << "): " << diff << " ("
                              << fullBlock[INDEX_3D(i, j, k)] << ", "
                              << results[INDEX_3D(i, j, k)] << ")" << std::endl;
                    calc_passed = false;
                    break;
                }
            }

    delete[] results;
    delete[] workspace;
    delete[] fullBlock;
    delete[] matrix_y;

    return calc_passed;
}

bool test_correctness_z() {
    static uint32_t nx = 7;
    static uint32_t ny = 8;
    static uint32_t nz = 9;

    double* fullBlock  = new double[nx * ny * nz];

    double* matrix_z   = new double[nz * nz]{};

    uint32_t tracker   = 0;
    for (uint32_t k = 0; k < nz; ++k)
        for (uint32_t j = 0; j < ny; ++j)
            for (uint32_t i = 0; i < nx; ++i) {
                fullBlock[INDEX_3D(i, j, k)] = tracker;
                tracker++;
            }

    for (uint32_t j = 0; j < nz; ++j) {
        for (uint32_t i = 0; i < nz; ++i) {
            if (i == j) {
                matrix_z[INDEX_N2D(i, j, nz)] = 1.0;
            }
        }
    }

#if 0
    print_2d_mat(matrix_z, ny, ny);
#endif

    // now we call the derivatives
    double* results             = new double[nx * ny * nz];
    double* workspace           = new double[nx * ny * nz];

    static const uint32_t sz[3] = {nx, ny, nz};

#if 1
    print_3d_mat(fullBlock, nx, ny, nz);
#endif

    dendroderivs::matmul_z_dim(matrix_z, results, fullBlock, 1.0, sz, workspace,
                               0);

#if 0
    std::cout << "RESULTS" << std::endl;
    print_3d_mat(results, nx, ny, nz);
#endif

    bool calc_passed = true;

    // do a check to see if they're accurate
    for (uint32_t k = 0; k < nz && calc_passed; ++k)
        for (uint32_t j = 0; j < ny && calc_passed; ++j)
            for (uint32_t i = 0; i < nx && calc_passed; ++i) {
                double diff = std::abs(fullBlock[INDEX_3D(i, j, k)] -
                                       results[INDEX_3D(i, j, k)]);

                if (diff > 1e-10) {
                    std::cout << "Value failed at (" << i << ", " << j << ", "
                              << k << "): " << diff << " ("
                              << fullBlock[INDEX_3D(i, j, k)] << ", "
                              << results[INDEX_3D(i, j, k)] << ")" << std::endl;
                    calc_passed = false;
                    break;
                }
            }

    delete[] results;
    delete[] workspace;
    delete[] fullBlock;
    delete[] matrix_z;

    return calc_passed;
}

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> currtime;
    double seconds     = 0.0;
    uint64_t num_calls = 0;

   public:
    Timer() : currtime(std::chrono::high_resolution_clock::now()) {}

    void start() { currtime = std::chrono::high_resolution_clock::now(); }

    void stop() {
        auto now = std::chrono::high_resolution_clock::now();

        seconds += std::chrono::duration<double>(now - currtime).count();
        currtime = now;
        num_calls++;
    }

    void reset() {
        seconds   = 0.0;
        num_calls = 0;
    }

    double get_time() { return seconds; }
    double get_calls() { return num_calls; }
};

template <typename Func>
double test_runner(Func lambda, uint32_t n_startups, uint32_t n_runs) {
    Timer timer;
    for (uint32_t i = 0; i < n_startups; i++) {
        lambda();
    }

    timer.start();
    for (uint32_t i = 0; i < n_startups; i++) {
        lambda();
    }
    timer.stop();

    return timer.get_time();
}

double test_speed_x(uint32_t n_startups, uint32_t n_runs, uint32_t nx,
                    uint32_t ny, uint32_t nz) {
    const uint32_t sz[3] = {nx, ny, nz};

    double* fullBlock    = new double[nx * ny * nz];
    double* results      = new double[nx * ny * nz];

    double* matrix_x     = new double[nx * nx]{};

    double time          = test_runner(
        [&]() {
            dendroderivs::matmul_x_dim(matrix_x, results, fullBlock, 1.0, sz,
                                                0);
        },
        n_startups, n_runs);

    delete[] fullBlock;
    delete[] matrix_x;
    delete[] results;

    return time;
}

double test_speed_y(uint32_t n_startups, uint32_t n_runs, uint32_t nx,
                    uint32_t ny, uint32_t nz) {
    const uint32_t sz[3] = {nx, ny, nz};

    double* fullBlock    = new double[nx * ny * nz];
    double* results      = new double[nx * ny * nz];
    double* workspace    = new double[nx * ny * nz];

    double* matrix_y     = new double[ny * ny]{};

    double time          = test_runner(
        [&]() {
            dendroderivs::matmul_y_dim(matrix_y, results, fullBlock, 1.0, sz,
                                                workspace, 0);
        },
        n_startups, n_runs);

    delete[] fullBlock;
    delete[] matrix_y;
    delete[] workspace;
    delete[] results;

    return time;
}

double test_speed_z(uint32_t n_startups, uint32_t n_runs, uint32_t nx,
                    uint32_t ny, uint32_t nz) {
    const uint32_t sz[3] = {nx, ny, nz};

    double* fullBlock    = new double[nx * ny * nz];
    double* results      = new double[nx * ny * nz];
    double* workspace    = new double[nx * ny * nz];

    double* matrix_z     = new double[nz * nz]{};

    double time          = test_runner(
        [&]() {
            dendroderivs::matmul_z_dim(matrix_z, results, fullBlock, 1.0, sz,
                                                workspace, 0);
        },
        n_startups, n_runs);

    delete[] fullBlock;
    delete[] matrix_z;
    delete[] workspace;
    delete[] results;

    return time;
}

double test_speed_x_original(uint32_t n_startups, uint32_t n_runs, uint32_t nx,
                             uint32_t ny, uint32_t nz, uint32_t pw) {
    const uint32_t sz[3] = {nx, ny, nz};

    double* fullBlock    = new double[nx * ny * nz];
    double* results      = new double[nx * ny * nz];

    // NOTE: ele_order is technically the input, but nx is fine, because
    // explicit doesn't care
    dendroderivs::ExplicitDerivsO4_DX deriv(nx);

    double time =
        test_runner([&]() { deriv.do_grad_x(results, fullBlock, 0.25, sz, 0); },
                    n_startups, n_runs);

    delete[] fullBlock;
    delete[] results;

    return time;
}

double test_speed_y_original(uint32_t n_startups, uint32_t n_runs, uint32_t nx,
                             uint32_t ny, uint32_t nz, uint32_t pw) {
    const uint32_t sz[3] = {nx, ny, nz};

    double* fullBlock    = new double[nx * ny * nz];
    double* results      = new double[nx * ny * nz];

    // NOTE: ele_order is technically the input, but nx is fine, because
    // explicit doesn't care
    dendroderivs::ExplicitDerivsO4_DX deriv(nx);

    double time =
        test_runner([&]() { deriv.do_grad_y(results, fullBlock, 0.25, sz, 0); },
                    n_startups, n_runs);

    delete[] fullBlock;
    delete[] results;

    return time;
}

double test_speed_z_original(uint32_t n_startups, uint32_t n_runs, uint32_t nx,
                             uint32_t ny, uint32_t nz, uint32_t pw) {
    const uint32_t sz[3] = {nx, ny, nz};

    double* fullBlock    = new double[nx * ny * nz];
    double* results      = new double[nx * ny * nz];

    // NOTE: ele_order is technically the input, but nx is fine, because
    // explicit doesn't care
    dendroderivs::ExplicitDerivsO4_DX deriv(nx);

    double time =
        test_runner([&]() { deriv.do_grad_z(results, fullBlock, 0.25, sz, 0); },
                    n_startups, n_runs);

    delete[] fullBlock;
    delete[] results;

    return time;
}

int main(int argc, char* argv[]) {
    std::cout << "NOW TESTING MATRIX MULTIPLICATION" << std::endl;
    if (test_correctness_x()) {
        std::cout << "X was numerically correct!" << std::endl;
    } else {
        std::cout << "X FAILED!" << std::endl;
    }
    if (test_correctness_y()) {
        std::cout << "Y was numerically correct!" << std::endl;
    } else {
        std::cout << "Y FAILED!" << std::endl;
    }
    if (test_correctness_z()) {
        std::cout << "Z was numerically correct!" << std::endl;
    } else {
        std::cout << "Z FAILED!" << std::endl;
    }

    // test the speed
    static const uint32_t eleOrder   = 6;

    static const uint32_t MAX_BLOCKS = 3;

    std::cout << "nx,ny,nz,derivx,derivy,derivz,derivx_orig" << std::endl;

    static const uint32_t NUM_RUNS = 1000000;

    for (uint32_t xblock = 1; xblock < MAX_BLOCKS; xblock++) {
        uint32_t nx = eleOrder * xblock + 1;
        for (uint32_t yblock = 1; yblock < MAX_BLOCKS; yblock++) {
            uint32_t ny = eleOrder * yblock + 1;
            for (uint32_t zblock = 1; zblock < MAX_BLOCKS; zblock++) {
                uint32_t nz      = eleOrder * zblock + 1;

                double dx_time   = test_speed_x(500, NUM_RUNS, nx, ny, nz);
                double dy_time   = test_speed_y(500, NUM_RUNS, nx, ny, nz);
                double dz_time   = test_speed_z(500, NUM_RUNS, nx, ny, nz);

                double dx_time_2 = test_speed_x_original(500, NUM_RUNS, nx, ny,
                                                         nz, eleOrder / 2);
                double dy_time_2 = test_speed_y_original(500, NUM_RUNS, nx, ny,
                                                         nz, eleOrder / 2);
                double dz_time_2 = test_speed_z_original(500, NUM_RUNS, nx, ny,
                                                         nz, eleOrder / 2);

                std::cout << nx << "," << ny << "," << nz << ","
                          << dx_time / NUM_RUNS << ',' << dy_time / NUM_RUNS
                          << "," << dz_time / NUM_RUNS << ","
                          << dx_time_2 / NUM_RUNS << "," << dy_time_2 / NUM_RUNS
                          << "," << dz_time_2 / NUM_RUNS << std::endl;
            }
        }
    }

    // test at the end here for matrix inversion
    //
    uint32_t N   = 3;
    double A[9]  = {4, 1, -2, 1, 3, -1, 2, -1, 5};

    double* Ainv = lapack::iterative_inverse(A, N);
    delete[] Ainv;

    return 0;
}
