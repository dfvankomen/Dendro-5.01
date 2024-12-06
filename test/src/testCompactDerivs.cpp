
#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <random>
#include <toml.hpp>
#include <tuple>
#include <vector>

#include "derivatives.h"
#include "derivatives/derivs_explicit.h"
#include "profiler.h"

#define RED "\e[1;31m"
#define BLU "\e[2;34m"
#define GRN "\e[0;32m"
#define YLW "\e[0;33m"
#define MAG "\e[0;35m"
#define CYN "\e[0;36m"
#define NRM "\e[0m"

#define UNIFORM_RAND_0_TO_X(X) ((double_t)rand() / (double_t)RAND_MAX * X)

#define STENCIL_OPTION 6

#if STENCIL_OPTION == 4
#define deriv_x  deriv42_x
#define deriv_y  deriv42_y
#define deriv_z  deriv42_z
#define deriv_xx deriv422_xx
#define deriv_yy deriv422_yy
#define deriv_zz deriv422_zz
#elif STENCIL_OPTION == 6
#define deriv_x  dendroderivs::deriv644_x
#define deriv_y  dendroderivs::deriv644_y
#define deriv_z  dendroderivs::deriv644_z
#define deriv_xx dendroderivs::deriv644_xx
#define deriv_yx dendroderivs::deriv644_yy
#define deriv_zx dendroderivs::deriv644_zz
#elif STENCIL_OPTION == 8
#define deriv_x deriv64_x
#define deriv_y deriv64_y
#define deriv_z deriv64_z
#endif

void init_data(const uint32_t init_type, double_t *u_var, const double *corner,
               const uint32_t *sz, const double *deltas,
               double_t *u_dx = nullptr, double_t *u_dy = nullptr,
               double_t *u_dz = nullptr, double_t *u_dxx = nullptr,
               double_t *u_dyy = nullptr, double_t *u_dzz = nullptr);

namespace params {

uint32_t eleorder            = 8;

std::string derivType_first  = "";
std::string derivType_second = "";

// dendro_cfd::FilterType filter_type    = dendro_cfd::FILT_NONE;
uint32_t num_tests           = 1000;
uint32_t data_init           = 2;
uint32_t num_x_blocks        = 10;
uint32_t num_y_blocks        = 10;
uint32_t num_z_blocks        = 10;

double x_start               = 0.0;
double y_start               = 0.0;
double z_start               = 0.0;

double dx                    = 0.25;
double dy                    = 0.25;
double dz                    = 0.25;

std::vector<double> coeffs_1 = {};
std::vector<double> coeffs_2 = {};

void readParams(const char *inFile) {
    auto file = toml::parse(inFile);

    if (file.contains("eleorder")) {
        eleorder = file["eleorder"].as_integer();
    }

    if (file.contains("deriv_type")) {
        derivType_first = file["deriv_type"].as_string();
    }
    if (file.contains("deriv_type_2nd")) {
        derivType_second = file["deriv_type_2nd"].as_string();
    }
    // if (file.contains("filter_type")) {
    //     filter_type = static_cast<dendro_cfd::FilterType>(
    //         file["filter_type"].as_integer());
    // }

    if (file.contains("num_tests")) {
        num_tests = file["num_tests"].as_integer();
    }
    if (file.contains("data_init")) {
        data_init = file["data_init"].as_integer();
    }

    // SIMPLE GRID SETUP
    if (file.contains("num_x_blocks")) {
        num_x_blocks = file["num_x_blocks"].as_integer();
    }
    if (file.contains("num_y_blocks")) {
        num_y_blocks = file["num_y_blocks"].as_integer();
    }
    if (file.contains("num_z_blocks")) {
        num_z_blocks = file["num_z_blocks"].as_integer();
    }
    if (file.contains("x_start")) {
        x_start = file["x_start"].as_floating();
    }
    if (file.contains("y_start")) {
        y_start = file["y_start"].as_floating();
    }
    if (file.contains("z_start")) {
        z_start = file["z_start"].as_floating();
    }
    if (file.contains("dx")) {
        dx = file["dx"].as_floating();
    }
    if (file.contains("dy")) {
        dy = file["dy"].as_floating();
    }
    if (file.contains("dz")) {
        dz = file["dz"].as_floating();
    }

    if (file.contains("deriv_X_coeffs")) {
        coeffs_1 = toml::find<std::vector<double>>(file, "deriv_X_coeffs");
    }

    if (file.contains("deriv_XX_coeffs")) {
        coeffs_2 = toml::find<std::vector<double>>(file, "deriv_XX_coeffs");
    }

    // check to make sure we're good to go...
    if ((eleorder % 2) != 0) {
        std::cerr << "Element order must be even!" << std::endl;
        exit(1);
    }

    std::cout << YLW << "PARAMETERS AS READ IN:" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "  eleorder : " << eleorder << std::endl;
    std::cout << "  deriv_type : " << derivType_first << std::endl;
    std::cout << "  deriv_type_2nd : " << derivType_second << std::endl;
    // std::cout << "  filter_type : " << filter_type << std::endl;
    std::cout << "  num_tests : " << num_tests << std::endl;
    std::cout << "  data_init : " << data_init << std::endl;
    std::cout << "  num_x_blocks : " << num_x_blocks << std::endl;
    std::cout << "  num_y_blocks : " << num_y_blocks << std::endl;
    std::cout << "  num_z_blocks : " << num_z_blocks << std::endl;
    std::cout << "  x_start : " << x_start << std::endl;
    std::cout << "  y_start : " << y_start << std::endl;
    std::cout << "  z_start : " << z_start << std::endl;
    std::cout << "  dx : " << dx << std::endl;
    std::cout << "  dy : " << dy << std::endl;
    std::cout << "  dz : " << dz << std::endl;
    std::cout << "  deriv_X_coeffs : [ ";
    for (auto &x : coeffs_1) {
        std::cout << x << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  deriv_XX_coeffs : [ ";
    for (auto &x : coeffs_2) {
        std::cout << x << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "======================" << NRM << std::endl << std::endl;
}

}  // namespace params

namespace helpers {
uint32_t padding;

profiler_t t_deriv_x;
profiler_t t_deriv_y;
profiler_t t_deriv_z;

profiler_t t_deriv_original_x;
profiler_t t_deriv_original_y;
profiler_t t_deriv_original_z;

profiler_t t_compact_deriv_x;
profiler_t t_compact_deriv_y;
profiler_t t_compact_deriv_z;

void print_profiler_results(uint64_t num_runs) {
    long double num_runs_d = (long double)num_runs;

    std::cout << YLW << "==== PROFILING RESULTS ====" << NRM << std::endl;
    std::cout << "Over " << num_runs << " total runs each" << std::endl;

    std::cout << "\t =< Original Stencils >=" << std::endl;
    std::cout << "\tx deriv: total=" << t_deriv_original_x.seconds
              << " average=" << t_deriv_original_x.seconds / num_runs_d
              << std::endl;
    std::cout << "\ty deriv: total=" << t_deriv_original_y.seconds
              << " average=" << t_deriv_original_y.seconds / num_runs_d
              << std::endl;
    std::cout << "\tz deriv: total=" << t_deriv_original_z.seconds
              << " average=" << t_deriv_original_z.seconds / num_runs_d
              << std::endl;

    std::cout << std::endl;

    std::cout << "\t =< Class-Based Stencils >=" << std::endl;
    std::cout << "\tx deriv: total=" << t_deriv_x.seconds
              << " average=" << t_deriv_x.seconds / num_runs_d << std::endl;
    std::cout << "\ty deriv: total=" << t_deriv_y.seconds
              << " average=" << t_deriv_y.seconds / num_runs_d << std::endl;
    std::cout << "\tz deriv: total=" << t_deriv_z.seconds
              << " average=" << t_deriv_z.seconds / num_runs_d << std::endl;

    std::cout << std::endl;

    std::cout << "\t =< Compact Stencils >=" << std::endl;
    std::cout << "\tx deriv: total=" << t_compact_deriv_x.seconds
              << " average=" << t_compact_deriv_x.seconds / num_runs_d
              << std::endl;
    std::cout << "\ty deriv: total=" << t_compact_deriv_y.seconds
              << " average=" << t_compact_deriv_y.seconds / num_runs_d
              << std::endl;
    std::cout << "\tz deriv: total=" << t_compact_deriv_z.seconds
              << " average=" << t_compact_deriv_z.seconds / num_runs_d
              << std::endl;
}

}  // namespace helpers

#if 0
class TestingFilters {
   public:
    TestingFilters(uint32_t nx, uint32_t ny, uint32_t nz);
    TestingFilters(TestingFilters &&)                 = default;
    TestingFilters(const TestingFilters &)            = default;
    TestingFilters &operator=(TestingFilters &&)      = default;
    TestingFilters &operator=(const TestingFilters &) = default;
    ~TestingFilters();

    double_t *get_u() { return m_u_var; }
    double_t *get_dx() { return m_u_dx_true; }
    double_t *get_dy() { return m_u_dy_true; }
    double_t *get_dz() { return m_u_dz_true; }
    double_t *get_dxx() { return m_u_dxx_true; }
    double_t *get_dyy() { return m_u_dyy_true; }
    double_t *get_dzz() { return m_u_dzz_true; }
    uint32_t get_total_points() { return m_total_points; }
    uint32_t get_num_blocks() { return m_total_blocks; }

    uint32_t get_offset(uint32_t blockno) {
        return blockno * m_points_per_block;
    }

    /**
     * @brief Returns the x, y, z coordinate for the top corner of the block.
     *
     * Do note that this does values does *not* include the padding inherit in
     * the stored "unzipped" data. x, y, and z will all need to be shifted by
     * `-du * padding`
     */
    std::tuple<double, double, double> get_coord(uint32_t blockno) {
        return m_coord_start[blockno];
    }

    std::tuple<double, double, double> get_coord_w_padding(uint32_t blockno) {
        auto [x, y, z] = m_coord_start[blockno];

        x              = x - (helpers::padding * params::dx);
        y              = y - (helpers::padding * params::dy);
        z              = z - (helpers::padding * params::dz);

        return std::make_tuple(x, y, z);
    }

    uint32_t get_bflag(uint32_t blockno) { return m_bflags[blockno]; }

    void initTestGrid();

   private:
    uint32_t m_nx;
    uint32_t m_ny;
    uint32_t m_nz;

    uint32_t m_single_dim;
    uint32_t m_total_points;
    uint32_t m_points_per_block;
    uint32_t m_total_blocks;

    double_t *m_u_var;
    double_t *m_u_dx_true;
    double_t *m_u_dy_true;
    double_t *m_u_dz_true;
    double_t *m_u_dxx_true;
    double_t *m_u_dyy_true;
    double_t *m_u_dzz_true;

    std::vector<std::tuple<double, double, double>> m_coord_start;
    std::vector<uint32_t> m_bflags;
};

TestingFilters::TestingFilters(uint32_t nx, uint32_t ny, uint32_t nz) {
    m_nx               = nx;
    m_ny               = ny;
    m_nz               = nz;

    m_single_dim       = params::eleorder * 2 + 1;
    m_points_per_block = m_single_dim * m_single_dim * m_single_dim;

    m_total_blocks     = nx * ny * nz;
    m_total_points     = m_total_blocks * m_points_per_block;

    initTestGrid();
}

void TestingFilters::initTestGrid() {
    // need to create a massive amount of memory for each block
    m_u_var      = new double_t[m_total_points]();
    m_u_dx_true  = new double_t[m_total_points]();
    m_u_dy_true  = new double_t[m_total_points]();
    m_u_dz_true  = new double_t[m_total_points]();
    m_u_dxx_true = new double_t[m_total_points]();
    m_u_dyy_true = new double_t[m_total_points]();
    m_u_dzz_true = new double_t[m_total_points]();

    // then set up the x, y, and z starting points
    m_coord_start.resize(m_total_blocks);
    m_bflags.resize(m_total_blocks);

    uint32_t idx       = 0;
    uint32_t offset    = 0;
    uint32_t size[3]   = {m_single_dim, m_single_dim, m_single_dim};
    double_t deltas[3] = {params::dx, params::dy, params::dz};
    double_t corner[3] = {0, 0, 0};
    uint32_t bflag;

    for (uint32_t kk = 0; kk < m_nz; kk++) {
        double z = (params::dz * params::eleorder) * kk + params::z_start;
        for (uint32_t jj = 0; jj < m_ny; jj++) {
            double y = (params::dy * params::eleorder) * jj + params::y_start;
            for (uint32_t ii = 0; ii < m_nx; ii++) {
                double x =
                    (params::dx * params::eleorder) * ii + params::x_start;

                bflag = 0;

                // calculate bflag based on i, j, and k;
                if (ii == 0) {
                    bflag |= 1u << OCT_DIR_LEFT;
                }
                if (ii == m_nx - 1) {
                    bflag |= 1u << OCT_DIR_RIGHT;
                }
                if (jj == 0) {
                    bflag |= 1u << OCT_DIR_DOWN;
                }
                if (jj == m_ny - 1) {
                    bflag |= 1u << OCT_DIR_UP;
                }
                if (kk == 0) {
                    bflag |= 1u << OCT_DIR_BACK;
                }
                if (kk == m_nz - 1) {
                    bflag |= 1u << OCT_DIR_FRONT;
                }

                m_bflags[idx]                 = bflag;

                m_coord_start[idx]            = std::make_tuple(x, y, z);

                // then initialize the data!

                auto [x_corn, y_corn, z_corn] = get_coord_w_padding(idx);
                offset                        = get_offset(idx);
                corner[0]                     = x_corn;
                corner[1]                     = y_corn;
                corner[2]                     = z_corn;

                // now populate the grid
                init_data(params::data_init, &m_u_var[offset], corner, size,
                          deltas, &m_u_dx_true[offset], &m_u_dy_true[offset],
                          &m_u_dz_true[offset], &m_u_dxx_true[offset],
                          &m_u_dyy_true[offset], &m_u_dzz_true[offset]);

                // std::cout << x << " " << y << " " << z << " "
                //           << std::bitset<8 * sizeof(bflag)>(bflag) <<
                //           std::endl;
                idx++;
            }
        }
    }
}

TestingFilters::~TestingFilters() {
    delete[] m_u_var;
    delete[] m_u_dx_true;
    delete[] m_u_dy_true;
    delete[] m_u_dz_true;
    delete[] m_u_dxx_true;
    delete[] m_u_dyy_true;
    delete[] m_u_dzz_true;
}
#endif

void sine_init(double_t *u_var, const uint32_t *sz, const double_t *deltas) {
    const double_t x_start   = 0.0;
    const double_t y_start   = 0.0;
    const double_t z_start   = 0.0;
    const double_t dx        = deltas[0];
    const double_t dy        = deltas[1];
    const double_t dz        = deltas[2];

    const double_t amplitude = 0.01;

    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        double_t z = z_start + k * dz;
        for (uint16_t j = 0; j < ny; j++) {
            double_t y = y_start + j * dy;
            for (uint16_t i = 0; i < nx; i++) {
                double x            = x_start + i * dx;
                u_var[IDX(i, j, k)] = 1.0 * sin(2 * x + 0.1) +
                                      2.0 * sin(3 * y - 0.1) +
                                      0.5 * sin(0.5 * z);
            }
        }
    }
}

void boris_init(double_t *u_var, const double_t *corner, const uint32_t *sz,
                const double_t *deltas, double_t *u_dx = nullptr,
                double_t *u_dy = nullptr, double_t *u_dz = nullptr,
                double_t *u_dxx = nullptr, double_t *u_dyy = nullptr,
                double_t *u_dzz = nullptr) {
    const double_t x_start   = corner[0];
    const double_t y_start   = corner[1];
    const double_t z_start   = corner[2];
    const double_t dx        = deltas[0];
    const double_t dy        = deltas[1];
    const double_t dz        = deltas[2];

    const double_t amplitude = 0.01;

    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        double_t z = z_start + k * dz;
        for (uint16_t j = 0; j < ny; j++) {
            double_t y = y_start + j * dy;
            for (uint16_t i = 0; i < nx; i++) {
                double_t x = x_start + i * dx;
                u_var[IDX(i, j, k)] =
                    0.5 * exp(-sin(2 * x) - sin(2 * y) - sin(2 * z));

                if (k == 0 && j == 0) {
                    // std::cout << x << " ";
                }
            }
            if (k == 0) {
                // std::cout << y << " ";
            }
        }
        // std::cout << z << " ";
    }
    // std::cout << std::endl;

    if (u_dx != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dx[IDX(i, j, k)] =
                        -1.0 * exp(-sin(2 * x) - sin(2 * y) - sin(2 * z)) *
                        cos(2 * x);
                }
            }
        }
    }

    if (u_dy != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dy[IDX(i, j, k)] =
                        -1.0 * exp(-sin(2 * x) - sin(2 * y) - sin(2 * z)) *
                        cos(2 * y);
                }
            }
        }
    }

    if (u_dz != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dz[IDX(i, j, k)] =
                        -1.0 * exp(-sin(2 * x) - sin(2 * y) - sin(2 * z)) *
                        cos(2 * z);
                }
            }
        }
    }

    if (u_dxx != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dxx[IDX(i, j, k)] =
                        2.0 * (sin(2 * x) + pow(cos(2 * x), 2)) *
                        exp(-sin(2 * x) - sin(2 * y) - sin(2 * z));
                }
            }
        }
    }

    if (u_dyy != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dyy[IDX(i, j, k)] =
                        2.0 * (sin(2 * y) + pow(cos(2 * y), 2)) *
                        exp(-sin(2 * x) - sin(2 * y) - sin(2 * z));
                }
            }
        }
    }

    if (u_dzz != nullptr) {
        for (uint16_t k = 0; k < nz; k++) {
            double_t z = z_start + k * dz;
            for (uint16_t j = 0; j < ny; j++) {
                double_t y = y_start + j * dy;
                for (uint16_t i = 0; i < nx; i++) {
                    double_t x = x_start + i * dx;
                    u_dzz[IDX(i, j, k)] =
                        2.0 * (sin(2 * z) + pow(cos(2 * z), 2)) *
                        exp(-sin(2 * x) - sin(2 * y) - sin(2 * z));
                }
            }
        }
    }
}

void random_init(double_t *u_var, const uint32_t *sz) {
    const double_t amplitude = 0.001;
    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                u_var[IDX(i, j, k)] =
                    amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
            }
        }
    }
}

void zero_init(double_t *u_var, const uint32_t *sz) {
    const double_t amplitude = 0.001;
    const unsigned int nx    = sz[0];
    const unsigned int ny    = sz[1];
    const unsigned int nz    = sz[2];

    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                u_var[IDX(i, j, k)] = 0.0;
            }
        }
    }
}

void init_data(const uint32_t init_type, double_t *u_var,
               const double_t *corner, const uint32_t *sz, const double *deltas,
               double_t *u_dx, double_t *u_dy, double_t *u_dz, double_t *u_dxx,
               double_t *u_dyy, double_t *u_dzz, const uint32_t bflag = 0) {
    switch (init_type) {
        case 0:
            zero_init(u_var, sz);
            break;

        case 1:
            random_init(u_var, sz);
            break;

        case 2:
            boris_init(u_var, corner, sz, deltas, u_dx, u_dy, u_dz, u_dxx,
                       u_dyy, u_dzz);
            break;

        default:
            std::cout << "UNRECOGNIZED INITIAL DATA FUNCTION... EXITING"
                      << std::endl;
            exit(0);
            break;
    }

    // depending on bflag, fill in a bunch of garbage along the "outer" walls,
    // to test for validity
    if (bflag) {
        const unsigned int nx    = sz[0];
        const unsigned int ny    = sz[1];
        const unsigned int nz    = sz[2];
        const double_t amplitude = 1e12;
        if (bflag & (1u << OCT_DIR_LEFT)) {
            std::cout << "Filling with bad garbage LEFT" << std::endl;
            // then fill the entire left face with garbage
            for (uint32_t k = 0; k < sz[2]; ++k) {
                for (uint32_t j = 0; j < sz[1]; ++j) {
                    for (uint32_t i = 0; i < helpers::padding; ++i) {
                        u_var[IDX(i, j, k)] =
                            amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
                        // std::sqrt(-1);
                    }
                }
            }
        }
        if (bflag & (1u << OCT_DIR_RIGHT)) {
            std::cout << "Filling with bad garbage RIGHT" << std::endl;
            // then fill the entire right face with garbage
            for (uint32_t k = 0; k < sz[2]; ++k) {
                for (uint32_t j = 0; j < sz[1]; ++j) {
                    for (uint32_t i = 0; i < helpers::padding; ++i) {
                        u_var[IDX(nx - i - 1, j, k)] =
                            amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
                        // std::sqrt(-1);
                    }
                }
            }
        }
        if (bflag & (1u << OCT_DIR_DOWN)) {
            std::cout << "Filling with bad garbage DOWN" << std::endl;
            // then fill the entire left Y face with garbage
            for (uint32_t k = 0; k < sz[2]; ++k) {
                for (uint32_t i = 0; i < sz[0]; ++i) {
                    for (uint32_t j = 0; j < helpers::padding; ++j) {
                        u_var[IDX(i, j, k)] =
                            amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
                        // std::sqrt(-1);
                    }
                }
            }
        }
        if (bflag & (1u << OCT_DIR_UP)) {
            std::cout << "Filling with bad garbage UP" << std::endl;
            // then fill the entire left Y face with garbage
            for (uint32_t k = 0; k < sz[2]; ++k) {
                for (uint32_t i = 0; i < sz[0]; ++i) {
                    for (uint32_t j = 0; j < helpers::padding; ++j) {
                        u_var[IDX(i, ny - j - 1, k)] =
                            amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
                        // std::sqrt(-1);
                    }
                }
            }
        }
        if (bflag & (1u << OCT_DIR_BACK)) {
            std::cout << "Filling with bad garbage BACK" << std::endl;
            // then fill the entire left Y face with garbage
            for (uint32_t j = 0; j < sz[1]; ++j) {
                for (uint32_t i = 0; i < sz[0]; ++i) {
                    for (uint32_t k = 0; k < helpers::padding; ++k) {
                        u_var[IDX(i, j, k)] =
                            amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
                        // std::sqrt(-1);
                    }
                }
            }
        }
        if (bflag & (1u << OCT_DIR_FRONT)) {
            std::cout << "Filling with bad garbage FRONT" << std::endl;
            // then fill the entire left Y face with garbage
            for (uint32_t j = 0; j < sz[1]; ++j) {
                for (uint32_t i = 0; i < sz[0]; ++i) {
                    for (uint32_t k = 0; k < helpers::padding; ++k) {
                        u_var[IDX(i, j, nz - k - 1)] =
                            amplitude * (UNIFORM_RAND_0_TO_X(2) - 1.0);
                        // std::sqrt(-1);
                    }
                }
            }
        }
    }
}

void print_3d_mat(double_t *u_var, const uint32_t *sz) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];
    for (uint16_t k = 0; k < nz; k++) {
        for (uint16_t j = 0; j < ny; j++) {
            for (uint16_t i = 0; i < nx; i++) {
                printf("%f ", u_var[IDX(i, j, k)]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

// void print_square_mat(double *m, const uint32_t n) {
//     // assumes "col" order in memory
//     // J is the row!
//     for (uint16_t i = 0; i < n; i++) {
//         printf("%3d : ", i);
//         // I is the column!
//         for (uint16_t j = 0; j < n; j++) {
//             printf("%8.3f ", m[INDEX_2D(i, j)]);
//         }
//         printf("\n");
//     }
// }

void print_square_mat_flat(double *m, const uint32_t n) {
    uint16_t j_count = 0;
    for (uint16_t i = 0; i < n * n; i++) {
        if (i % n == 0) {
            j_count++;
            printf("\n");
        }
        printf("%8.3f ", m[i]);
    }
}

std::tuple<double_t, double_t, double_t> calculate_rmse(
    const double_t *const x, const double_t *const y, const uint32_t *sz,
    bool skip_pading = true) {
    // required for IDX function...
    const unsigned int nx  = sz[0];
    const unsigned int ny  = sz[1];
    const unsigned int nz  = sz[2];

    double_t max_err       = 0.0;
    double_t min_err       = __DBL_MAX__;
    double_t rmse          = 0.0;

    const uint32_t i_start = skip_pading ? helpers::padding : 0;
    const uint32_t j_start = skip_pading ? helpers::padding : 0;
    const uint32_t k_start = skip_pading ? helpers::padding : 0;

    const uint32_t i_end   = skip_pading ? sz[0] - helpers::padding : sz[0];
    const uint32_t j_end   = skip_pading ? sz[1] - helpers::padding : sz[1];
    const uint32_t k_end   = skip_pading ? sz[2] - helpers::padding : sz[2];

    // std::cout << i_start << " " << i_end << std::endl;

    const uint32_t total_points =
        (i_end - i_start) * (j_end - j_start) * (k_end - k_start);

    // std::cout << total_points << std::endl;

    for (uint16_t k = k_start; k < k_end; k++) {
        for (uint16_t j = j_start; j < j_end; j++) {
            for (uint16_t i = i_start; i < i_end; i++) {
                double_t temp = (x[IDX(i, j, k)] - y[IDX(i, j, k)]) *
                                (x[IDX(i, j, k)] - y[IDX(i, j, k)]);

                if (temp > max_err) {
                    max_err = temp;
                }
                if (temp < min_err) {
                    min_err = temp;
                }

                rmse += temp;
            }
        }
    }

    rmse /= (total_points);

    return std::make_tuple(sqrt(rmse), min_err, max_err);
}

double_t calc_l2_norm(double_t *const u_var, double_t *const v_var,
                      const uint32_t n) {
    double_t sum = 0.0;

    for (uint32_t ii = 0; ii < n; ii++) {
        sum += (u_var[ii] - v_var[ii]) * (u_var[ii] - v_var[ii]);
    }

    return sqrt(sum);
}

double_t calc_3d_l2_without_padding(double_t *const u_var,
                                    double_t *const v_var, const uint32_t *sz) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    double_t sum          = 0.0;

    for (uint32_t k = helpers::padding; k < sz[2] - helpers::padding; k++) {
        for (uint32_t j = helpers::padding; j < sz[1] - helpers::padding; j++) {
            for (uint32_t i = helpers::padding; i < sz[0] - helpers::padding;
                 i++) {
                sum += (u_var[IDX(i, j, k)] - v_var[IDX(i, j, k)]) *
                       (u_var[IDX(i, j, k)] - v_var[IDX(i, j, k)]);
            }
        }
    }
    return sqrt(sum);
}

void calculate_all_derivs(double_t *const u_var, const uint32_t *sz,
                          const double *deltas,
                          dendroderivs::DendroDerivatives *derivs,
                          double_t *const deriv_workspace, uint32_t bflag) {
    std::cout << "Doing all derivatives with " << derivs->toString()
              << std::endl;

    const uint32_t totalSize = sz[0] * sz[1] * sz[2];

    double_t *const derivx   = deriv_workspace + 0 * totalSize;
    double_t *const derivy   = deriv_workspace + 1 * totalSize;
    double_t *const derivz   = deriv_workspace + 2 * totalSize;

    double_t *const derivxx  = deriv_workspace + 3 * totalSize;
    double_t *const derivyy  = deriv_workspace + 4 * totalSize;
    double_t *const derivzz  = deriv_workspace + 5 * totalSize;

    derivs->grad_x(derivx, u_var, deltas[0], sz, bflag);
    derivs->grad_y(derivy, u_var, deltas[1], sz, bflag);
    derivs->grad_z(derivz, u_var, deltas[2], sz, bflag);

    derivs->grad_xx(derivxx, u_var, deltas[0], sz, bflag);
    derivs->grad_yy(derivyy, u_var, deltas[1], sz, bflag);
    derivs->grad_zz(derivzz, u_var, deltas[2], sz, bflag);
}

void test_cfd_with_original_stencil(
    double_t *const u_var, const uint32_t *sz, const double *deltas,
    dendroderivs::DendroDerivatives *derivs,
    dendroderivs::DendroDerivatives *cfd_derivs, double_t *u_dx = nullptr,
    double_t *u_dy = nullptr, double_t *u_dz = nullptr,
    double_t *u_dxx = nullptr, double_t *u_dyy = nullptr,
    double_t *u_dzz = nullptr) {
    // allocate a double block of memory
    const uint32_t totalSize      = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace     = new double_t[totalSize * 3 * 4];

    double_t *const deriv_stencil = deriv_workspace;
    double_t *const deriv_cfd     = deriv_workspace + totalSize * 6;

    // const unsigned int bflag      = (1 << 6) - 1;
    // const unsigned int bflag      = 0b010101;
    const unsigned int bflag      = 0;

    // calculate the stencil derivatives
    calculate_all_derivs(u_var, sz, deltas, derivs, deriv_stencil, bflag);

    double *u_var_copy = new double[totalSize];

    std::copy_n(u_var, totalSize, u_var_copy);

    std::ofstream fileForFilter("testOutputFilter.bin",
                                std::ios::binary | std::ios::out);
    fileForFilter.write((char *)&totalSize, sizeof(totalSize));
    fileForFilter.write((char *)u_var_copy, sizeof(double) * totalSize);
    std::cout << "\nL2 before filts: "
              << calc_l2_norm(u_var, u_var_copy, totalSize)
              << " - without padding: "
              << calc_3d_l2_without_padding(u_var, u_var_copy, sz) << std::endl;

#if 0
    cfd->filter_cfd_x(u_var_copy, derivx_cfd, deltas[0], sz, bflag);
    fileForFilter.write((char *)u_var_copy, sizeof(double) * totalSize);
    std::cout << "L2 after X Filt: "
              << calc_l2_norm(u_var, u_var_copy, totalSize)
              << " - without padding: "
              << calc_3d_l2_without_padding(u_var, u_var_copy, sz) << std::endl;

    cfd->filter_cfd_y(u_var_copy, derivy_cfd, deltas[0], sz, bflag);
    fileForFilter.write((char *)u_var_copy, sizeof(double) * totalSize);
    std::cout << "L2 after Y Filt: "
              << calc_l2_norm(u_var, u_var_copy, totalSize)
              << " - without padding: "
              << calc_3d_l2_without_padding(u_var, u_var_copy, sz) << std::endl;

    cfd->filter_cfd_z(u_var_copy, derivz_cfd, deltas[0], sz, bflag);
    fileForFilter.write((char *)u_var_copy, sizeof(double) * totalSize);
    std::cout << "L2 after Z Filt: "
              << calc_l2_norm(u_var, u_var_copy, totalSize)
              << " - without padding: "
              << calc_3d_l2_without_padding(u_var, u_var_copy, sz) << std::endl;
#endif

    // DUMP FILTERED RESULTS
    fileForFilter.close();

    // calculate the stencil derivatives
    calculate_all_derivs(u_var_copy, sz, deltas, cfd_derivs, deriv_cfd, bflag);

    delete[] u_var_copy;

    const double *const derivx_stencil  = deriv_stencil;
    const double *const derivy_stencil  = deriv_stencil + 1 * totalSize;
    const double *const derivz_stencil  = deriv_stencil + 2 * totalSize;
    const double *const derivxx_stencil = deriv_stencil + 3 * totalSize;
    const double *const derivyy_stencil = deriv_stencil + 4 * totalSize;
    const double *const derivzz_stencil = deriv_stencil + 5 * totalSize;
    const double *const derivx_cfd      = deriv_cfd;
    const double *const derivy_cfd      = deriv_cfd + 1 * totalSize;
    const double *const derivz_cfd      = deriv_cfd + 2 * totalSize;
    const double *const derivxx_cfd     = deriv_cfd + 3 * totalSize;
    const double *const derivyy_cfd     = deriv_cfd + 4 * totalSize;
    const double *const derivzz_cfd     = deriv_cfd + 5 * totalSize;

    // then compute the "error" difference between the two
    double_t min_x, max_x, rmse_x, min_y, max_y, rmse_y, min_z, max_z, rmse_z;
    std::tie(rmse_x, min_x, max_x) =
        calculate_rmse(derivx_stencil, derivx_cfd, sz);
    std::tie(rmse_y, min_y, max_y) =
        calculate_rmse(derivy_stencil, derivy_cfd, sz);
    std::tie(rmse_z, min_z, max_z) =
        calculate_rmse(derivz_stencil, derivz_cfd, sz);

    std::cout << std::endl
              << GRN << "===COMPARING CFD TO STENCIL TEST RESULTS===" << NRM
              << std::endl;
    std::cout << "   deriv_x : rmse = \t" << rmse_x << "\tmin_err = \t" << min_x
              << "\tmax_err = \t" << max_x << std::endl;
    std::cout << "   deriv_y : rmse = \t" << rmse_y << "\tmin_err = \t" << min_y
              << "\tmax_err = \t" << max_y << std::endl;
    std::cout << "   deriv_z : rmse = \t" << rmse_z << "\tmin_err = \t" << min_z
              << "\tmax_err = \t" << max_z << std::endl;

    std::tie(rmse_x, min_x, max_x) =
        calculate_rmse(derivxx_stencil, derivxx_cfd, sz);
    std::tie(rmse_y, min_y, max_y) =
        calculate_rmse(derivyy_stencil, derivyy_cfd, sz);
    std::tie(rmse_z, min_z, max_z) =
        calculate_rmse(derivzz_stencil, derivzz_cfd, sz);
    std::cout << std::endl
              << GRN
              << "===COMPARING CFD TO STENCIL TEST RESULTS - 2ND ORDER==="
              << NRM << std::endl;
    std::cout << "   deriv_xx : rmse = \t" << rmse_x << "\tmin_err = \t"
              << min_x << "\tmax_err = \t" << max_x << std::endl;
    std::cout << "   deriv_yy : rmse = \t" << rmse_y << "\tmin_err = \t"
              << min_y << "\tmax_err = \t" << max_y << std::endl;
    std::cout << "   deriv_zz : rmse = \t" << rmse_z << "\tmin_err = \t"
              << min_z << "\tmax_err = \t" << max_z << std::endl;

    if (u_dx != nullptr && u_dy != nullptr && u_dz != nullptr) {
        // then compute the "error" difference between the two
        std::tie(rmse_x, min_x, max_x) =
            calculate_rmse(derivx_stencil, u_dx, sz);
        std::tie(rmse_y, min_y, max_y) =
            calculate_rmse(derivy_stencil, u_dy, sz);
        std::tie(rmse_z, min_z, max_z) =
            calculate_rmse(derivz_stencil, u_dz, sz);

        std::cout << std::endl
                  << GRN << "===COMPARING STENCIL TO TRUTH RESULTS===" << NRM
                  << std::endl;
        std::cout << "   deriv_x : rmse = \t" << rmse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_y : rmse = \t" << rmse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_z : rmse = \t" << rmse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;

        // then compute the "error" difference between the two
        std::tie(rmse_x, min_x, max_x) = calculate_rmse(derivx_cfd, u_dx, sz);
        std::tie(rmse_y, min_y, max_y) = calculate_rmse(derivy_cfd, u_dy, sz);
        std::tie(rmse_z, min_z, max_z) = calculate_rmse(derivz_cfd, u_dz, sz);

        std::cout << std::endl
                  << GRN << "===COMPARING CFD TO TRUTH RESULTS===" << NRM
                  << std::endl;
        std::cout << "   deriv_x : rmse = \t" << rmse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_y : rmse = \t" << rmse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_z : rmse = \t" << rmse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;
    }

    if (u_dxx != nullptr && u_dyy != nullptr && u_dzz != nullptr) {
        // then compute the "error" difference between the two
        std::tie(rmse_x, min_x, max_x) =
            calculate_rmse(derivxx_stencil, u_dxx, sz);
        std::tie(rmse_y, min_y, max_y) =
            calculate_rmse(derivyy_stencil, u_dyy, sz);
        std::tie(rmse_z, min_z, max_z) =
            calculate_rmse(derivzz_stencil, u_dzz, sz);

        std::cout << std::endl
                  << GRN
                  << "===COMPARING STENCIL TO TRUTH RESULTS - 2ND ORDER==="
                  << NRM << std::endl;
        std::cout << "   deriv_xx : rmse = \t" << rmse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_yy : rmse = \t" << rmse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_zz : rmse = \t" << rmse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;

        // then compute the "error" difference between the two
        std::tie(rmse_x, min_x, max_x) = calculate_rmse(derivxx_cfd, u_dxx, sz);
        std::tie(rmse_y, min_y, max_y) = calculate_rmse(derivyy_cfd, u_dyy, sz);
        std::tie(rmse_z, min_z, max_z) = calculate_rmse(derivzz_cfd, u_dzz, sz);

        std::cout << std::endl
                  << GRN
                  << "===COMPARING CFD TO TRUTH RESULTS - 2ND ORDER===" << NRM
                  << std::endl;
        std::cout << "   deriv_xx : rmse = \t" << rmse_x << "\tmin_err = \t"
                  << min_x << "\tmax_err = \t" << max_x << std::endl;
        std::cout << "   deriv_yy : rmse = \t" << rmse_y << "\tmin_err = \t"
                  << min_y << "\tmax_err = \t" << max_y << std::endl;
        std::cout << "   deriv_zz : rmse = \t" << rmse_z << "\tmin_err = \t"
                  << min_z << "\tmax_err = \t" << max_z << std::endl;
    }

#if 0
    // dump for visaliztaion
    std::ofstream file("testOutputDZZ.bin", std::ios::binary | std::ios::out);
    file.write((char *)&totalSize, sizeof(totalSize));
    file.write((char *)u_dzz, sizeof(double) * totalSize);
    file.write((char *)derivzz_stencil, sizeof(double) * totalSize);
    file.close();
#endif

    delete[] deriv_workspace;
}

void profile_compact_stencils(double_t *const u_var, const uint32_t *sz,
                              const double *deltas,
                              dendroderivs::DendroDerivatives *deriv,
                              uint32_t num_runs) {
    const uint32_t totalSize   = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace  = new double_t[totalSize * 3];

    double_t *const derivx_cfd = deriv_workspace + 0 * totalSize;
    double_t *const derivy_cfd = deriv_workspace + 1 * totalSize;
    double_t *const derivz_cfd = deriv_workspace + 2 * totalSize;

    uint32_t bflag             = 0b010101;

    const uint32_t n_warmups   = 0;

    // warmup runs
    for (uint32_t ii = 0; ii < n_warmups; ii++) {
        deriv->grad_x(derivx_cfd, u_var, deltas[0], sz, bflag);
    }

    helpers::t_compact_deriv_x.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv->grad_x(derivx_cfd, u_var, deltas[0], sz, bflag);
    }
    helpers::t_compact_deriv_x.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < n_warmups; ii++) {
        deriv->grad_y(derivy_cfd, u_var, deltas[1], sz, bflag);
    }

    helpers::t_compact_deriv_y.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv->grad_y(derivy_cfd, u_var, deltas[1], sz, bflag);
    }
    helpers::t_compact_deriv_y.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < n_warmups; ii++) {
        deriv->grad_z(derivz_cfd, u_var, deltas[2], sz, bflag);
    }

    helpers::t_compact_deriv_z.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv->grad_z(derivz_cfd, u_var, deltas[2], sz, bflag);
    }
    helpers::t_compact_deriv_z.stop();

    delete[] deriv_workspace;
}

void profile_class_based_stencils(double_t *const u_var, const uint32_t *sz,
                                  dendroderivs::DendroDerivatives *deriv,
                                  const double *deltas, uint32_t num_runs) {
    const uint32_t totalSize       = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace      = new double_t[totalSize * 3];

    double_t *const derivx_stencil = deriv_workspace + 0 * totalSize;
    double_t *const derivy_stencil = deriv_workspace + 1 * totalSize;
    double_t *const derivz_stencil = deriv_workspace + 2 * totalSize;

    uint32_t bflag                 = 0;

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv->grad_x(derivx_stencil, u_var, deltas[0], sz, bflag);
    }

    helpers::t_deriv_x.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv->grad_x(derivx_stencil, u_var, deltas[0], sz, bflag);
    }
    helpers::t_deriv_x.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv->grad_y(derivy_stencil, u_var, deltas[1], sz, bflag);
    }

    helpers::t_deriv_y.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv->grad_y(derivy_stencil, u_var, deltas[1], sz, bflag);
    }
    helpers::t_deriv_y.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv->grad_z(derivz_stencil, u_var, deltas[2], sz, bflag);
    }

    helpers::t_deriv_z.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv->grad_z(derivz_stencil, u_var, deltas[2], sz, bflag);
    }
    helpers::t_deriv_z.stop();

    delete[] deriv_workspace;
}

void profile_original_stencils(double_t *const u_var, const uint32_t *sz,
                               const double *deltas, uint32_t num_runs) {
    const uint32_t totalSize       = sz[0] * sz[1] * sz[2];
    double_t *deriv_workspace      = new double_t[totalSize * 3];

    double_t *const derivx_stencil = deriv_workspace + 0 * totalSize;
    double_t *const derivy_stencil = deriv_workspace + 1 * totalSize;
    double_t *const derivz_stencil = deriv_workspace + 2 * totalSize;

    uint32_t bflag                 = 0;

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv_x<3>(derivx_stencil, u_var, deltas[0], sz, bflag);
    }

    helpers::t_deriv_original_x.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv_x<3>(derivx_stencil, u_var, deltas[0], sz, bflag);
    }
    helpers::t_deriv_original_x.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv_y<3>(derivy_stencil, u_var, deltas[1], sz, bflag);
    }

    helpers::t_deriv_original_y.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv_y<3>(derivy_stencil, u_var, deltas[1], sz, bflag);
    }
    helpers::t_deriv_original_y.stop();

    // warmup runs
    for (uint32_t ii = 0; ii < 100; ii++) {
        deriv_z<3>(derivz_stencil, u_var, deltas[2], sz, bflag);
    }

    helpers::t_deriv_original_z.start();
    for (uint32_t ii = 0; ii < num_runs; ii++) {
        deriv_z<3>(derivz_stencil, u_var, deltas[2], sz, bflag);
    }
    helpers::t_deriv_original_z.stop();

    delete[] deriv_workspace;
}

void fill_with_non_nan_garbage(double_t *arr, uint32_t n) {
    double lower_bound = -100000000;
    double upper_bound = 10000;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    for (uint32_t ii = 0; ii < n; ii++) {
        arr[ii] = unif(re);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "ERROR: a parameter file must be specified!" << std::endl;
        return 0;
    }

    params::readParams(argv[1]);

    helpers::padding = params::eleorder / 2;
    std::cout << "Padding set to " << helpers::padding << std::endl;

    // the size in each dimension
    uint32_t fullwidth = 2 * params::eleorder + 1;
    uint32_t sz[3]     = {fullwidth, fullwidth, fullwidth};

#if 0
    TestingFilters testFilter(params::num_x_blocks, params::num_y_blocks,
                              params::num_z_blocks);
#endif

    // now we can actually build up our test block

    double_t *u_var      = new double_t[sz[0] * sz[1] * sz[2]];
    double_t *u_dx_true  = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dy_true  = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dz_true  = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dxx_true = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dyy_true = new double_t[sz[0] * sz[1] * sz[2]]();
    double_t *u_dzz_true = new double_t[sz[0] * sz[1] * sz[2]]();

    double_t deltas[3]   = {0.02, 0.01, 0.01};

    double_t corner[3]   = {0.0, 0.0, 0.0};
    init_data(params::data_init, u_var, corner, sz, deltas, u_dx_true,
              u_dy_true, u_dz_true, u_dxx_true, u_dyy_true, u_dzz_true, 0);

    // create the explicit objects

    std::cout << "Building Explicit derivatives..." << std::endl;
    dendroderivs::DendroDerivatives deriv("E6", "E6", params::eleorder);

    std::cout << "Building test derivatives..." << std::endl;
    dendroderivs::DendroDerivatives deriv_cfd(
        params::derivType_first, params::derivType_second, params::eleorder,
        params::coeffs_1, params::coeffs_2);
    std::cout << "Derivatives built!" << std::endl;

    // run a short test to see what the errors are
    test_cfd_with_original_stencil((double_t *const)u_var, sz, deltas, &deriv,
                                   &deriv_cfd, u_dx_true, u_dy_true, u_dz_true,
                                   u_dxx_true, u_dyy_true, u_dzz_true);

    std::cout << "... now profiling first set of derivs......" << std::endl;
    profile_class_based_stencils((double_t *const)u_var, sz, &deriv, deltas,
                                 params::num_tests);

    std::cout << "... now profiling second set of derivs......" << std::endl;
    profile_compact_stencils((double_t *const)u_var, sz, deltas, &deriv_cfd,
                             params::num_tests);

    std::cout << "... now profiling true original set of derivs......"
              << std::endl;
    profile_original_stencils((double_t *const)u_var, sz, deltas,
                              params::num_tests);

    // then print the profiler results
    helpers::print_profiler_results(params::num_tests);

    // now we'll do a quick test to see how well rectangular matrices doo

    uint32_t max_fused = std::max(params::num_x_blocks, params::num_y_blocks);
    max_fused          = std::max(max_fused, params::num_z_blocks);
    uint32_t largest_1d_size = (1 + max_fused) * (helpers::padding * 2) + 1;
    uint32_t largest_3d_size =
        largest_1d_size * largest_1d_size * largest_1d_size;
    // let's try with a bunch of stuff i guess
    double_t *u_larger          = new double_t[largest_3d_size];
    double_t *u_deriv_larger    = new double_t[largest_3d_size];
    double_t *u_dx_true_larger  = new double_t[largest_3d_size];
    double_t *u_dy_true_larger  = new double_t[largest_3d_size];
    double_t *u_dz_true_larger  = new double_t[largest_3d_size];
    double_t *u_dxx_true_larger = new double_t[largest_3d_size];
    double_t *u_dyy_true_larger = new double_t[largest_3d_size];
    double_t *u_dzz_true_larger = new double_t[largest_3d_size];

    // cfd.initialize_cfd_3dblock_workspace(largest_3d_size);

    std::cout
        << "\n\n================== NOW DOING RECTANGULAR MATRICES ==========="
        << std::endl;

    for (uint32_t xfused = 1; xfused < params::num_x_blocks + 1; xfused++) {
        for (uint32_t yfused = 1; yfused < params::num_y_blocks + 1; yfused++) {
            for (uint32_t zfused = 1; zfused < params::num_z_blocks + 1;
                 zfused++) {
                uint32_t x_size    = (1 + xfused) * (helpers::padding * 2) + 1;
                uint32_t y_size    = (1 + yfused) * (helpers::padding * 2) + 1;
                uint32_t z_size    = (1 + zfused) * (helpers::padding * 2) + 1;

                uint32_t new_sz[3] = {x_size, y_size, z_size};

                uint32_t bflag     = 0b101010;

                // fill our the matrix with boris init
                init_data(params::data_init, u_larger, corner, new_sz, deltas,
                          u_dx_true_larger, u_dy_true_larger, u_dz_true_larger,
                          u_dxx_true_larger, u_dyy_true_larger,
                          u_dzz_true_larger, bflag);

                double_t min_x, max_x, rmse_x, min_y, max_y, rmse_y, min_z,
                    max_z, rmse_z;

                std::cout << ":::xfused, yfused, zfused: " << xfused << ", "
                          << yfused << ", " << zfused << ":::" << std::endl;
                std::cout << " - Size: " << new_sz[0] << ", " << new_sz[1]
                          << ", " << new_sz[2] << std::endl;

                // fill with garbage so that we're sure the output is correct
                fill_with_non_nan_garbage(u_deriv_larger,
                                          x_size * y_size * z_size);
                // time for the x deriv
                deriv_cfd.grad_x(u_deriv_larger, u_larger, deltas[0], new_sz,
                                 bflag);
                // calculate the MSE
                std::tie(rmse_x, min_x, max_x) =
                    calculate_rmse(u_deriv_larger, u_dx_true_larger, new_sz);
                std::cout << "    deriv_x : rmse = \t" << rmse_x
                          << "\tmin_err = \t" << min_x << "\tmax_err\t" << max_x
                          << std::endl;

                // fill with garbage so that we're sure the output is correct
                fill_with_non_nan_garbage(u_deriv_larger,
                                          x_size * y_size * z_size);
                // time for the y deriv
                deriv_cfd.grad_y(u_deriv_larger, u_larger, deltas[1], new_sz,
                                 bflag);
                // calculate the MSE
                std::tie(rmse_y, min_y, max_y) =
                    calculate_rmse(u_deriv_larger, u_dy_true_larger, new_sz);
                std::cout << "    deriv_y : rmse = \t" << rmse_y
                          << "\tmin_err = \t" << min_y << "\tmax_err\t" << max_y
                          << std::endl;

                // fill with garbage so that we're sure the output is correct
                fill_with_non_nan_garbage(u_deriv_larger,
                                          x_size * y_size * z_size);
                // time for the z deriv
                deriv_cfd.grad_z(u_deriv_larger, u_larger, deltas[2], new_sz,
                                 bflag);
                // calculate the MSE
                std::tie(rmse_z, min_z, max_z) =
                    calculate_rmse(u_deriv_larger, u_dz_true_larger, new_sz);
                std::cout << "    deriv_z : rmse = \t" << rmse_z
                          << "\tmin_err = \t" << min_z << "\tmax_err\t" << max_z
                          << std::endl;

                // SECOND ORDER DERIVATIVES

                // fill with garbage so that we're sure the output is correct
                fill_with_non_nan_garbage(u_deriv_larger,
                                          x_size * y_size * z_size);
                // time for the x deriv
                deriv_cfd.grad_xx(u_deriv_larger, u_larger, deltas[0], new_sz,
                                  bflag);
                // calculate the MSE
                std::tie(rmse_x, min_x, max_x) =
                    calculate_rmse(u_deriv_larger, u_dxx_true_larger, new_sz);
                std::cout << "    deriv_xx : rmse = \t" << rmse_x
                          << "\tmin_err = \t" << min_x << "\tmax_err\t" << max_x
                          << std::endl;

                // fill with garbage so that we're sure the output is correct
                fill_with_non_nan_garbage(u_deriv_larger,
                                          x_size * y_size * z_size);
                // time for the y deriv
                deriv_cfd.grad_yy(u_deriv_larger, u_larger, deltas[1], new_sz,
                                  bflag);
                // calculate the MSE
                std::tie(rmse_y, min_y, max_y) =
                    calculate_rmse(u_deriv_larger, u_dyy_true_larger, new_sz);
                std::cout << "    deriv_yy : rmse = \t" << rmse_y
                          << "\tmin_err = \t" << min_y << "\tmax_err\t" << max_y
                          << std::endl;

                // fill with garbage so that we're sure the output is correct
                fill_with_non_nan_garbage(u_deriv_larger,
                                          x_size * y_size * z_size);
                // time for the z deriv
                deriv_cfd.grad_zz(u_deriv_larger, u_larger, deltas[2], new_sz,
                                  bflag);
                // calculate the MSE
                std::tie(rmse_z, min_z, max_z) =
                    calculate_rmse(u_deriv_larger, u_dzz_true_larger, new_sz);
                std::cout << "    deriv_zz : rmse = \t" << rmse_z
                          << "\tmin_err = \t" << min_z << "\tmax_err\t" << max_z
                          << std::endl;

                std::cout << std::endl;
            }
        }
    }

    uint32_t fused_x = 2;

    // var cleanup
    delete[] u_var;
    delete[] u_dx_true;
    delete[] u_dy_true;
    delete[] u_dz_true;
    delete[] u_dxx_true;
    delete[] u_dyy_true;
    delete[] u_dzz_true;

    delete[] u_larger;
    delete[] u_deriv_larger;
    delete[] u_dx_true_larger;
    delete[] u_dy_true_larger;
    delete[] u_dz_true_larger;
    delete[] u_dxx_true_larger;
    delete[] u_dyy_true_larger;
    delete[] u_dzz_true_larger;
}
