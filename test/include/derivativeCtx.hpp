/**
 * @file derivativeCtx.h
 * @author David Van Komen
 * @brief Derivative context object to do simple tests on various derivative
 * types on interesting meshes
 *
 */

#pragma once
#include <string>

#include "ctx.h"
#include "derivatives.h"
#include "mathMeshUtils.h"
#include "meshUtils.h"
#include "oct2vtk.h"

#define PI \
    3.1415926535897932384626433832795028841971693993751058209749445923078164062

#define Rx \
    (derivtest::DERIV_TEST_COMPD_MAX[0] - derivtest::DERIV_TEST_COMPD_MIN[0])
#define Ry \
    (derivtest::DERIV_TEST_COMPD_MAX[1] - derivtest::DERIV_TEST_COMPD_MIN[1])
#define Rz \
    (derivtest::DERIV_TEST_COMPD_MAX[2] - derivtest::DERIV_TEST_COMPD_MIN[2])

#define RgX \
    (derivtest::DERIV_TEST_OCTREE_MAX[0] - derivtest::DERIV_TEST_OCTREE_MIN[0])
#define RgY \
    (derivtest::DERIV_TEST_OCTREE_MAX[1] - derivtest::DERIV_TEST_OCTREE_MIN[1])
#define RgZ \
    (derivtest::DERIV_TEST_OCTREE_MAX[2] - derivtest::DERIV_TEST_OCTREE_MIN[2])

#define GRIDX_TO_X(xg)                                           \
    (((Rx / RgX) * (xg - derivtest::DERIV_TEST_OCTREE_MIN[0])) + \
     derivtest::DERIV_TEST_COMPD_MIN[0])
#define GRIDY_TO_Y(yg)                                           \
    (((Ry / RgY) * (yg - derivtest::DERIV_TEST_OCTREE_MIN[1])) + \
     derivtest::DERIV_TEST_COMPD_MIN[1])
#define GRIDZ_TO_Z(zg)                                           \
    (((Rz / RgZ) * (zg - derivtest::DERIV_TEST_OCTREE_MIN[2])) + \
     derivtest::DERIV_TEST_COMPD_MIN[2])

#define X_TO_GRIDX(xc)                                          \
    (((RgX / Rx) * (xc - derivtest::DERIV_TEST_COMPD_MIN[0])) + \
     derivtest::DERIV_TEST_OCTREE_MIN[0])
#define Y_TO_GRIDY(yc)                                          \
    (((RgY / Ry) * (yc - derivtest::DERIV_TEST_COMPD_MIN[1])) + \
     derivtest::DERIV_TEST_OCTREE_MIN[1])
#define Z_TO_GRIDZ(zc)                                          \
    (((RgZ / Rz) * (zc - derivtest::DERIV_TEST_COMPD_MIN[2])) + \
     derivtest::DERIV_TEST_OCTREE_MIN[2])

namespace derivtest {

const unsigned int DERIV_TEST_NUM_VARS       = 6;
const unsigned int DERIV_TEST_NUM_VARS_TRUE  = 1;
const unsigned int DERIV_TEST_ASYNC_COMM_K   = 1;
const unsigned int DERIV_TEST_INIT_GRID_ITER = 1;
extern unsigned int DERIV_TEST_ELE_ORDER;
extern unsigned int DERIV_TEST_PADDING_WIDTH;
extern double DERIV_TEST_GRID_MIN_X;
extern double DERIV_TEST_GRID_MAX_X;
extern double DERIV_TEST_GRID_MIN_Y;
extern double DERIV_TEST_GRID_MAX_Y;
extern double DERIV_TEST_GRID_MIN_Z;
extern double DERIV_TEST_GRID_MAX_Z;
const bool DERIV_TEST_INIT_GRID_REINIT_EACH_TIME = true;
const bool DERIV_TEST_DO_REFINE                  = true;
extern bool DERIV_TEST_ENABLE_BLOCK_ADAPTIVITY;
const unsigned int DERIV_TEST_NUM_REFINE_VARS = 1;
const unsigned int DERIV_TEST_REFINE_VARS[1]  = {0};
const double DERIV_TEST_WAVELET_TOL           = 0.0001;
const double DERIV_TEST_DENDRO_AMR_FAC        = 0.1;
const unsigned int DERIV_TEST_DENDRO_GRAIN_SZ = 1000;
const double DERIV_TEST_LOAD_IMB_TOL          = 0.1;
const unsigned int DERIV_TEST_SPLIT_FIX       = 256;
extern unsigned int DERIV_TEST_MAXDEPTH;
extern double DERIV_TEST_COMPD_MIN[3];
extern double DERIV_TEST_COMPD_MAX[3];
extern double DERIV_TEST_OCTREE_MIN[3];
extern double DERIV_TEST_OCTREE_MAX[3];
extern unsigned int DERIV_TEST_ID_TYPE;
extern std::string DERIV_TEST_VTU_FILE_PREFIX;
extern bool DERIV_TEST_VTU_Z_SLICE_ONLY;

const double DERIV_TEST_CFL_FACTOR = 0.25;
extern double DERIV_TEST_RK45_TIME_STEP_SIZE;

extern std::unique_ptr<dendroderivs::DendroDerivatives> DERIV_TEST_DERIVS;

extern std::string DERIV_TEST_DERIVTYPE_FIRST;
extern std::string DERIV_TEST_DERIVTYPE_SECOND;
extern std::vector<double> DERIV_TEST_DERIV_FIRST_COEFFS;
extern std::vector<double> DERIV_TEST_DERIV_SECOND_COEFFS;
extern unsigned int DERIV_TEST_DERIVFIRST_MATID;
extern unsigned int DERIV_TEST_DERIVSECOND_MATID;

const std::string DERIV_TEST_DERIV_NAMES[6] = {"U_DX",  "U_DY",  "U_DZ",
                                               "U_DXX", "U_DYY", "U_DZZ"};

inline double computeWTolDCoords(double x, double y, double z, double* hx) {
    const unsigned int eleOrder = DERIV_TEST_ELE_ORDER;

    return DERIV_TEST_WAVELET_TOL;
}

void initDataType0(const double xx, const double yy, const double zz,
                   double* var);

void initDataType0_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var);

void initDataType1(const double xx, const double yy, const double zz,
                   double* var);

void initDataType1_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var);

void initDataType2(const double xx, const double yy, const double zz,
                   double* var);

void initDataType2_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var);

void initDataType3(const double xx, const double yy, const double zz,
                   double* var);

void initDataType3_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var);

void initDataType4(const double xx, const double yy, const double zz,
                   double* var);

void initDataType4_AnalyticalDerivs(const double xx, const double yy,
                                    const double zz, double* var);

void inline initDataFuncToPhysCoords(const double xx1, const double yy1,
                                     const double zz1, double* var) {
    // convert "grid" values to physical values
    const double xx = GRIDX_TO_X(xx1);
    const double yy = GRIDY_TO_Y(yy1);
    const double zz = GRIDZ_TO_Z(zz1);

    switch (DERIV_TEST_ID_TYPE) {
        case 0:
            initDataType0(xx, yy, zz, var);
            break;
        case 1:
            initDataType1(xx, yy, zz, var);
            break;
        case 2:
            initDataType2(xx, yy, zz, var);
            break;
        case 3:
            initDataType3(xx, yy, zz, var);
            break;
        case 4:
            initDataType4(xx, yy, zz, var);
            break;

        default:
            std::cout << "Unknown ID type: " << DERIV_TEST_ID_TYPE << std::endl;
            exit(0);
            break;
    }
}

void inline analyticalDerivs(const double xx1, const double yy1,
                             const double zz1, double* var) {
    // convert "grid" values to physical values
    const double xx = GRIDX_TO_X(xx1);
    const double yy = GRIDY_TO_Y(yy1);
    const double zz = GRIDZ_TO_Z(zz1);

    switch (DERIV_TEST_ID_TYPE) {
        case 0:
            initDataType0_AnalyticalDerivs(xx, yy, zz, var);
            break;
        case 1:
            initDataType1_AnalyticalDerivs(xx, yy, zz, var);
            break;
        case 2:
            initDataType2_AnalyticalDerivs(xx, yy, zz, var);
            break;
        case 3:
            initDataType3_AnalyticalDerivs(xx, yy, zz, var);
            break;
        case 4:
            initDataType4_AnalyticalDerivs(xx, yy, zz, var);
            break;

        default:
            std::cout << "Unknown ID type: " << DERIV_TEST_ID_TYPE << std::endl;
            exit(0);
            break;
    }
}

/**@brief smoothing modes avail for LTS recomended for LTS time stepping. */
enum LTS_SMOOTH_MODE { KO = 0, WEIGHT_FUNC };

enum VL {
    CPU_EV = 0,
    CPU_DERIVS,
    CPU_DERIVS_CALC,
    CPU_DERIVS_DIFF,
    CPU_EV_UZ,
    CPU_DERIVS_UZ,
#if 0
    // redefine these if analytic needs to be done on a block-wise basis!
    CPU_ANALYTIC_UZ_IN,
    CPU_ANALYTIC_DIFF_UZ_IN,
#endif
    END
};

typedef ot::DVector<DendroScalar, unsigned int> DVec;

class DerivTestCtx : public ts::Ctx<DerivTestCtx, DendroScalar, unsigned int> {
    // #ifdef __PROFILE_CTX__
    //     public:
    //         using ts::Ctx<DendroScalar, DendroIntL>::m_uiCtxpt;
    // #endif

   protected:
    /**@brief: evolution var (zip)*/
    DVec m_var[VL::END];

    /** @brief: Lets us know if we need to compute the constraints for this
     * timestep */
    bool m_constraintsComputed = false;

    /** @brief: Lets us know if we need to compute the analytical for this
     * timestep */
    bool m_analyticalComputed  = false;

    DendroIntL m_uiGlobalMeshElements;
    DendroIntL m_uiGlobalGridPoints;
    bool m_uiWroteGridInfoHeader = false;

   public:
    /**@brief: default constructor*/
    DerivTestCtx(ot::Mesh* pMesh);

    /**@brief: default deconstructor*/
    ~DerivTestCtx();

    /** @brief: Any flags that need to be adjusted and updated for the next step
     *
     * This is particularly useful for if the analytical solution or constraints
     * need to be computed for multiple potential reasons, but the reasons
     * aren't necessarily triggered every time step!
     */
    void resetForNextStep() {
        // timer::resetSnapshot();
    }

    /**@brief: initial solution*/
    int initialize();

    /**@brief: initialize the grid, solution. */
    int init_grid();

    // /**
    //  * @brief sets time adaptive offset
    //  * @param tadapoffst
    //  */
    // void set_time_adap_offset(unsigned int tadapoffst) {
    //     SOLVER_LTS_TS_OFFSET = tadapoffst;
    // }
    //
    // /**@brief : returns the time adaptive offset value*/
    // unsigned int get_time_adap_offset() { return SOLVER_LTS_TS_OFFSET; }

    void doDerivCalculation();

    /**
     * @brief computes the BSSN rhs
     *
     * @param in : zipped input
     * @param out : zipped output
     * @param sz  : number of variables.
     * @param time : current time.
     * @return int : status. (0) on success.
     */
    int rhs(DVec* in, DVec* out, unsigned int sz, DendroScalar time);

    void compute_constraints();

    /**
     * @brief Compute the analytical solution to the system
     */
    void compute_analytical();

    /**
     * @brief compute the block for the rhs (used in ENUTS).
     * @param in :  blk vectot in
     * @param out : blk vector out
     * @param local_blk_id : blkid
     * @param blk_time : blk time
     * @return int
     */
    int rhs_blk(const DendroScalar* in, DendroScalar* out, unsigned int dof,
                unsigned int local_blk_id, DendroScalar blk_time);

    /**@brief : block wise pre_stage computations goes here*/
    int pre_stage_blk(DendroScalar* in, unsigned int dof,
                      unsigned int local_blk_id, DendroScalar blk_time) const {
        return 0;
    }

    /**@brief : block wise post stage computations goes here*/
    int post_stage_blk(DendroScalar* in, unsigned int dof,
                       unsigned int local_blk_id, DendroScalar blk_time) const {
        return 0;
    }

    /**@brief : block wise pre timestep computations goes here*/
    int pre_timestep_blk(DendroScalar* in, unsigned int dof,
                         unsigned int local_blk_id,
                         DendroScalar blk_time) const {
        return 0;
    }

    /**@brief : block wise post timestep computations goes here*/
    int post_timestep_blk(DendroScalar* in, unsigned int dof,
                          unsigned int local_blk_id,
                          DendroScalar blk_time) const {
        return 0;
    }

    /**@brief : compute LTS offset base on application specifics. */
    unsigned int compute_lts_ts_offset() { return 0; };

    /**@brief: function execute before each stage
     * @param sIn: stage var in.
     */
    int pre_stage(DVec sIn);

    /**@brief: function execute after each stage
     * @param sIn: stage var in.
     */
    int post_stage(DVec sIn);

    /**@brief: function execute before each step*/
    int pre_timestep(DVec sIn);

    /**@brief: function execute after each step*/
    int post_timestep(DVec sIn);

    /**@brief: function execute after each step*/
    bool is_remesh();

    /**@brief: write to vtu. */
    int write_vtu();

    /**@brief: writes checkpoint*/
    int write_checkpt();

    /**@brief: restore from check point*/
    int restore_checkpt();

    /**@brief: should be called for free up the contex memory. */
    int finalize();

    /**@brief: pack and returns the evolution variables to one DVector*/
    DVec& get_evolution_vars();

    /**@brief: pack and returns the constraint variables to one DVector*/
    DVec& get_constraint_vars();

    /**@brief: pack and returns the primitive variables to one DVector*/
    DVec& get_primitive_vars();

    /**@brief: prints any messages to the terminal output. */
    int terminal_output();

    /**@brief: returns the async communication batch size. */
    unsigned int get_async_batch_sz() { return DERIV_TEST_ASYNC_COMM_K; }

    /**@brief: returns the number of variables considered when performing
     * refinement*/
    unsigned int get_num_refine_vars() { return DERIV_TEST_NUM_REFINE_VARS; }

    /**@brief: return the pointer for containing evolution refinement variable
     * ids*/
    const unsigned int* get_refine_var_ids() { return DERIV_TEST_REFINE_VARS; }

    /**@brief return the wavelet tolerance function / value*/
    std::function<double(double, double, double, double*)> get_wtol_function() {
        double wtol = DERIV_TEST_WAVELET_TOL;
        std::function<double(double, double, double, double*)> waveletTolFunc =
            [wtol](double x, double y, double z, double* hx) {
                return computeWTolDCoords(x, y, z, hx);
            };
        return waveletTolFunc;
    }

    static unsigned int getBlkTimestepFac(unsigned int blev, unsigned int lmin,
                                          unsigned int lmax);

    int grid_transfer(const ot::Mesh* m_new);

    void calculate_full_grid_size();

    void write_grid_summary_data();

    void inline prepare_derivatives() {
        if (!m_uiMesh->isActive()) return;

        // find the largest block size
        const std::vector<ot::Block>& blkList = m_uiMesh->getLocalBlockList();
        unsigned int max_blk_sz               = 0;
        for (unsigned int i = 0; i < blkList.size(); i++) {
            unsigned int blk_sz = blkList[i].getAllocationSzX() *
                                  blkList[i].getAllocationSzY() *
                                  blkList[i].getAllocationSzZ();
            if (blk_sz > max_blk_sz) {
                max_blk_sz = blk_sz;
            }
        }

        DERIV_TEST_DERIVS->set_maximum_block_size(max_blk_sz);
    }
};

}  // namespace derivtest
