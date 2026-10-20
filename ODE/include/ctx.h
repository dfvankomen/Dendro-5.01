/**
 * @file ctx.h
 * @author Milinda Fernando
 * @brief : Application context which can be used for time stepping.
 * @version 0.1
 * @date 2019-12-17
 *
 * School of Computing, University of Utah.
 * @copyright Copyright (c) 2019
 *
 */
#pragma once
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

#include "dendro.h"
#include "dvec.h"
#include "mathUtils.h"
#include "mesh.h"
#include "ts.h"
#ifdef __CUDACC__
#include "mesh_gpu.cuh"
#endif
namespace ts {
/**@brief: different variable types. */
enum CTXVType { EVOLUTION = 0, CONSTRAINT, PRIMITIVE };

#ifdef __PROFILE_CTX__
enum CTXPROFILE {
    IS_REMESH = 0,
    REMESH,
    GRID_TRASFER,
    RHS,
    RHS_BLK,
    UNZIP_WCOMM,
    UNZIP,
    ZIP_WCOMM,
    ZIP,
    H2D,
    D2H,
    CTX_LAST
};
#endif

template <typename DerivedCtx, typename T, typename I>
class Ctx {
#ifdef __PROFILE_CTX__
   public:
    std::vector<profiler_t> m_uiCtxpt =
        std::vector<profiler_t>(static_cast<int>(CTXPROFILE::CTX_LAST));
    const char* CTXPROFILE_NAMES[static_cast<int>(CTXPROFILE::CTX_LAST)] = {
        "is_remesh", "remesh", "GT",  "rhs", "rhs_blk",
        "unzip",     "zip",    "h2d", "d2h"};

    void init_pt() {
        for (unsigned int i = 0; i < m_uiCtxpt.size(); i++)
            m_uiCtxpt[i].start();
    }

    void dump_pt(std::ostream& outfile) const {
        if (!(m_uiMesh->isActive())) return;

        int rank                       = m_uiMesh->getMPIRank();
        int npes                       = m_uiMesh->getMPICommSize();

        MPI_Comm comm                  = m_uiMesh->getMPICommunicator();
        const unsigned int currentStep = m_uiTinfo._m_uiStep;

        double t_stat;
        double t_stat_g[3];

        if (!rank) {
            // writes the header
            if (currentStep <= 1)
                outfile << "step_ctx\t act_npes\t glb_npes\t maxdepth\t "
                           "numOcts\t dof_cg\t dof_uz\t"
                        << "gele_min\t gele_mean\t gele_max\t"
                           "lele_min\t lele_mean\t lele_max\t"
                           "lnodes_min\t lnodes_mean\t lnodes_max\t"
                           "is_remesh_min\t is_remesh_mean\t is_remesh_max\t"
                        << "remesh_min\t remesh_mean\t remesh_max\t"
                        << "GT_min\t GT_mean\t GT_max\t"
                        << "rhs_min\t rhs_mean\t rhs_max\t"
                        << "rhs_blk_min\t rhs_blk_mean\t rhs_blk_max\t"
                        << "unzip_min\t unzip_mean\t unzip_max\t"
                        << "zip_min\t zip_mean\t zip_max\t"
                           "h2d_min\t h2d_mean\t h2d_max\t"
                           "d2h_min\t d2h_mean\t d2h_max\t"
                        << std::endl;
        }

        if (!rank) outfile << currentStep << "\t ";
        if (!rank) outfile << m_uiMesh->getMPICommSize() << "\t ";
        if (!rank) outfile << m_uiMesh->getMPICommSizeGlobal() << "\t ";
        if (!rank) outfile << m_uiMaxDepth << "\t ";

        DendroIntL localSz = m_uiMesh->getNumLocalMeshElements();
        DendroIntL globalSz;

        par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
        if (!rank) outfile << globalSz << "\t ";

        localSz = m_uiMesh->getNumLocalMeshNodes();
        par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
        if (!rank) outfile << globalSz << "\t ";

        localSz = m_uiMesh->getDegOfFreedomUnZip();
        par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);
        if (!rank) outfile << globalSz << "\t ";

        DendroIntL ghostElements = m_uiMesh->getNumPreGhostElements() +
                                   m_uiMesh->getNumPostGhostElements();
        DendroIntL localElements = m_uiMesh->getNumLocalMeshElements();

        t_stat                   = (double)ghostElements;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = (double)localElements;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        DendroIntL ghostNodes =
            m_uiMesh->getNumPreMeshNodes() + m_uiMesh->getNumPostMeshNodes();
        DendroIntL localNodes = m_uiMesh->getNumLocalMeshNodes();

        t_stat                = localNodes;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::IS_REMESH].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::REMESH].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::GRID_TRASFER].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::RHS].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::RHS_BLK].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::UNZIP].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::ZIP].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::H2D].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        t_stat = m_uiCtxpt[CTXPROFILE::D2H].snap;
        min_mean_max(&t_stat, t_stat_g, comm);
        if (!rank)
            outfile << t_stat_g[0] << "\t " << t_stat_g[1] << "\t "
                    << t_stat_g[2] << "\t ";

        if (!rank) outfile << std::endl;
    }

    void reset_pt() {
        for (unsigned int i = 0; i < m_uiCtxpt.size(); i++)
            m_uiCtxpt[i].snapreset();
    }

#endif

   protected:
    /**@brief : Mesh object  */
    ot::Mesh* m_uiMesh;

    /**@brief: Total memory allocated for time stepper in B */
    DendroIntL m_uiMemAlloc;

    /**@brief: Total memory deallocated for time stepper B*/
    DendroIntL m_uiMemDeAlloc;

    /**@brief: time stepper info*/
    ts::TSInfo m_uiTinfo;

    /**@brief: element order for mesh generation*/
    unsigned int m_uiElementOrder;

    /** @brief: min point of the compute domain. */
    Point m_uiMinPt;

    /**@brief: max point of the compute domain */
    Point m_uiMaxPt;

    /**@brief: is true indicates that the ETS is synced with the current mesh.
     */
    bool m_uiIsETSSynced                             = true;

    /**@brief: weight function to perform advanced weighted partitioning. */
    unsigned int (*m_getWeight)(const ot::TreeNode*) = NULL;

    /**@brief: MPI resources for host communication, i.e., buffers are allocated
     * on the host*/
    std::vector<ot::AsyncExchangeContex> m_mpi_ctx;

    /**@brief: MPI resources for device communication, i.e., buffers are
     * allocated on the device*/
    std::vector<ot::AsyncExchangeContex> m_mpi_ctx_device;

   public:
    /**@brief: default constructor*/
    Ctx() {};

    /**@brief: default destructor*/
    ~Ctx() {};

    /**@brief: derived class static cast*/
    inline DerivedCtx& asLeaf() { return static_cast<DerivedCtx&>(*this); }

    /**@brief true when the leaf declares its own member (pointer f) in place
     * of the base's forwarder (pointer g): the types differ by class*/
    template <class F, class G>
    static constexpr bool overrides(F, G) {
        return !std::is_same_v<F, G>;
    }

    /**@brief: update the mesh data strucutre. */
    void set_mesh(ot::Mesh* pMesh) {
        m_uiMesh        = pMesh;
        m_uiIsETSSynced = false;
        return;
    }

    /**@biref: set the weighted partition function.*/
    void set_wpart_function(unsigned int (*getWeight)(const ot::TreeNode*)) {
        m_getWeight = getWeight;
    }

    /**@brief: returns true if the m_getWeight function is set*/
    bool is_wpart_func_set() {
        if (m_getWeight == NULL)
            return false;
        else
            return true;
    }

    /**@brief: returns the const ot::Mesh. */
    inline ot::Mesh* get_mesh() const { return m_uiMesh; }

    /**@brief: returns the time stamp info, related to ets*/
    inline ts::TSInfo get_ts_info() const { return m_uiTinfo; }

    /**@brief: returns the time stamp info, related to ets*/
    inline void set_ts_info(ts::TSInfo ts_info) { m_uiTinfo = ts_info; }

    /**@brief: set just the timestep size (dt), leaving the rest of TSInfo. */
    inline void set_timestep(double dt) { m_uiTinfo._m_uiTh = dt; }

    /**@breif: returns the ETS synced status*/
    inline bool is_ets_synced() const { return m_uiIsETSSynced; }

    /**@brief:sets the ets sync status*/
    inline void set_ets_synced(bool s) { m_uiIsETSSynced = s; }

    /**@brief: initial solution*/
    int initialize() { return asLeaf().initialize(); }
    /**@brief: right hand side computation v= F(u,t);
     * @param [in] in : input (u vector) for the evolution variables.
     * @param [out] out: output (v vector) computed rhs.
     * @param [in] sz: number of in and out vectors (dof)
     * @param [in] time : current time at the evolution.
     */
    int rhs(ot::DVector<T, I>* in, ot::DVector<T, I>* out, unsigned int sz,
            T time) {
        return asLeaf().rhs(in, out, sz, time);
    }

    /**
     * @brief LTS: right-hand side of one block, out = F(in, blk_time).
     * @param in : block-local input, dof variables of the padded block
     * @param out : block-local output, same layout
     * @param dof : number of variables
     * @param local_blk_id : local block id
     * @param blk_time : time of the stage
     */
    int rhs_blk(const T* in, T* out, unsigned int dof,
                unsigned int local_blk_id, T blk_time) {
        return asLeaf().rhs_blk(in, out, dof, local_blk_id, blk_time);
    }

    /**
     * @brief LTS, optional: in-place pointwise operation on a block's stage
     * input before its right-hand side.
     * @param in : block-local stage input, dof variables of the padded block
     * @param dof : number of variables
     * @param local_blk_id : local block id
     * @param blk_time : time of the stage
     */
    int pre_stage_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                      T blk_time) {
        if constexpr (overrides(&DerivedCtx::pre_stage_blk, &Ctx::pre_stage_blk))
            return asLeaf().pre_stage_blk(in, dof, local_blk_id, blk_time);
        return 0;
    }

    /**
     * @brief LTS, optional: in-place operation on a block after a stage.
     * @param in : block-local data, dof variables of the padded block
     * @param dof : number of variables
     * @param local_blk_id : local block id
     * @param blk_time : time of the stage
     */
    int post_stage_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                       T blk_time) {
        if constexpr (overrides(&DerivedCtx::post_stage_blk, &Ctx::post_stage_blk))
            return asLeaf().post_stage_blk(in, dof, local_blk_id, blk_time);
        return 0;
    }

    /**
     * @brief LTS, optional: called with a block's step input y_k when the
     * block starts a step (padding synced, all of it at blk_time).
     * @param in : block-local step input, dof variables of the padded block
     * @param dof : number of variables
     * @param local_blk_id : local block id
     * @param blk_time : time of the step start
     */
    int pre_timestep_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                         T blk_time) {
        if constexpr (overrides(&DerivedCtx::pre_timestep_blk, &Ctx::pre_timestep_blk))
            return asLeaf().pre_timestep_blk(in, dof, local_blk_id, blk_time);
        return 0;
    }

    /**
     * @brief LTS, optional: in-place operation on a block's new step y_{k+1}.
     * @param in : block-local step result, dof variables of the padded block
     * @param dof : number of variables
     * @param local_blk_id : local block id
     * @param blk_time : time at the end of the step
     */
    int post_timestep_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                          T blk_time) {
        if constexpr (overrides(&DerivedCtx::post_timestep_blk, &Ctx::post_timestep_blk))
            return asLeaf().post_timestep_blk(in, dof, local_blk_id, blk_time);
        return 0;
    }

    /**@brief: function execute before each stage
     * @param sIn: stage var in.
     */
    int pre_stage(ot::DVector<T, I> sIn) { return asLeaf().pre_stage(sIn); }

    /**@brief: function execute after each stage
     * @param sIn: stage var in.
     */
    int post_stage(ot::DVector<T, I> sIn) { return asLeaf().post_stage(sIn); }

    /**@brief: function execute before each step*/
    int pre_timestep(ot::DVector<T, I> sIn) {
        return asLeaf().pre_timestep(sIn);
    }

    /**@brief: function execute after each step*/
    int post_timestep(ot::DVector<T, I> sIn) {
        return asLeaf().post_timestep(sIn);
    }

    /**@brief: RK stage input out = base + sum_j cf[j]*src[j], then
     * post_timestep(out). Override to fuse into one pass (bit-identical).*/
    int rk_stage_input(const ot::DVector<T, I>& base, unsigned int n,
                       const T* cf, const ot::DVector<T, I>* const* src,
                       ot::DVector<T, I>& out) {
        out.copy_data(base);
        if (n) ot::DVector<T, I>::axpy_multi(m_uiMesh, n, cf, src, out);
        return asLeaf().post_timestep(out);
    }

    /**@brief: RK final combine y += sum_j cf[j]*src[j], then
     * post_timestep(y). Override to fuse into one pass (bit-identical).*/
    int rk_combine(unsigned int n, const T* cf,
                   const ot::DVector<T, I>* const* src, ot::DVector<T, I>& y) {
        if (n) ot::DVector<T, I>::axpy_multi(m_uiMesh, n, cf, src, y);
        return asLeaf().post_timestep(y);
    }

    /**@brief: function execute after each step*/
    bool is_remesh() { return asLeaf().is_remesh(); };

    /**
     * @brief perform remesh and return the new mesh, (Note: mesh in the Ctx
     * class is not updated to the new mesh.) This just compute the new mesh and
     * returns it.
     * @param grain_sz : grain size for the remesh
     * @param ld_tol : load imbalance tolerance.
     * @param sf_k : splitter fix k
     * @return ot::Mesh* : pointer to the new mesh.
     */
    ot::Mesh* remesh(unsigned int grain_sz = DENDRO_DEFAULT_GRAIN_SZ,
                     double ld_tol         = DENDRO_DEFAULT_LB_TOL,
                     unsigned int sf_k     = DENDRO_DEFAULT_SF_K);

    /**
     * @brief performs remesh if the remesh flags are true.
     * @param grain_sz : grain size for the remesh
     * @param ld_tol : load imbalance tolerance.
     * @param sf_k : splitter fix value
     * @param transferEvolution : if true transform the evolution variales.
     * @param transferConstraint : if true transform the constraint variables.
     * @param transferPrimitive : if true transform the primitive variables.
     * @return int : return 0 if success.
     */
    int remesh_and_gridtransfer(unsigned int grain_sz = DENDRO_DEFAULT_GRAIN_SZ,
                                double ld_tol         = DENDRO_DEFAULT_LB_TOL,
                                unsigned int sf_k     = DENDRO_DEFAULT_SF_K);

    /**
     * @brief Refine the initial grid until the element count stabilizes.
     *
     * Repeatedly: is_remesh()? -> remesh_and_gridtransfer, stopping when
     * is_remesh() is false, the global element count stops changing, or max_iter
     * is reached. This is the init_grid_iter loop every solver's main.cpp
     * open-codes; it uses only base Ctx + mesh facilities (no solver globals).
     * Apply the initial conditions BEFORE calling (the refine reads the state).
     */
    void refine_initial_grid(MPI_Comm comm, unsigned int max_iter,
                             unsigned int grain_sz = DENDRO_DEFAULT_GRAIN_SZ,
                             double ld_tol         = DENDRO_DEFAULT_LB_TOL,
                             unsigned int sf_k     = DENDRO_DEFAULT_SF_K) {
        // comm is passed in (the global comm): remesh_and_gridtransfer frees the
        // old mesh's sub-communicator, so a comm cached off m_uiMesh would dangle.
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (!rank)
            std::cout << "Starting init_grid_iter loop (max=" << max_iter
                      << " iters)" << std::endl;
        for (unsigned int iter = 1; iter <= max_iter; iter++) {
            if (!is_remesh()) {
                if (!rank)
                    std::cout << "[init_grid_iter " << iter
                              << "] mesh converged (is_remesh=false)" << std::endl;
                break;
            }
            DendroIntL old_el_l = m_uiMesh->getNumLocalMeshElements();
            remesh_and_gridtransfer(grain_sz, ld_tol, sf_k);
            DendroIntL new_el_l = m_uiMesh->getNumLocalMeshElements();
            // Allreduce (not Reduce): the convergence check runs on every rank.
            DendroIntL old_el_g = 0, new_el_g = 0;
            par::Mpi_Allreduce(&old_el_l, &old_el_g, 1, MPI_SUM, comm);
            par::Mpi_Allreduce(&new_el_l, &new_el_g, 1, MPI_SUM, comm);
            if (!rank)
                std::cout << "[init_grid_iter " << iter << "] elements "
                          << old_el_g << " -> " << new_el_g << std::endl;
            if (new_el_g == old_el_g) {
                if (!rank)
                    std::cout << "[init_grid_iter " << iter
                              << "] mesh converged (stable element count)"
                              << std::endl;
                break;
            }
        }
    }

    /**
     * @brief performs unzip operation,
     *
     * @param in : input zip vector.
     * @param out : output unzip vector.
     * @param async_k : async communicator.
     */
    void unzip(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
               unsigned int async_k = 1);

    /**
     * @brief performs zip operation
     *
     * @param in : unzip vector
     * @param out : zip vector.
     */
    void zip(ot::DVector<T, I>& in, ot::DVector<T, I>& out);

    /**
     * @brief: only do the sync without communication sync
     * @param in : unzip vector
     * @param out : zip vector.
     * */
    // void zip_without_ghost_sync(ot::DVector<T,I>& in , ot::DVector<T,I>&
    // out);

    /**
     * @brief performs intergrid transfer for a given appCtx
     */
    inline int grid_transfer(const ot::Mesh* m_new) {
        dendro::logger::debug(dendro::logger::Scope{"CTX"},
                              "Now calling user-defined grid transfer");
        return asLeaf().grid_transfer(m_new);
    }

    /**@brief: write vtu files and output related stuff. */
    int write_vtu() { return asLeaf().write_vtu(); }

    /**@brief: writes checkpoint*/
    int write_checkpt() { return asLeaf().write_checkpt(); }

    /**@brief: restore from check point*/
    int restore_checkpt() { return asLeaf().restore_checkpt(); }

    /**@brief: should be called for free up the contex memory. */
    int finalize() { return asLeaf().finalize(); }

    /**@brief: pack and returns the evolution variables to one DVector*/
    inline ot::DVector<T, I>& get_evolution_vars() {
        return asLeaf().get_evolution_vars();
    }

    /**@brief: pack and returns the constraint variables to one DVector*/
    inline ot::DVector<T, I>& get_constraint_vars() {
        return asLeaf().get_constraint_vars();
    }

    /**@brief: pack and returns the primitive variables to one DVector*/
    inline ot::DVector<T, I>& get_primitive_vars() {
        return asLeaf().get_primitive_vars();
    }

    /**@brief: updates the time step information. */
    int increment_ts_info(T tfac = 1.0, unsigned int sfac = 1) {
        m_uiTinfo._m_uiT += (m_uiTinfo._m_uiTh * tfac);
        m_uiTinfo._m_uiStep += (1 * sfac);
        return 0;
    }

    /**@brief: prints any messages to the terminal output. */
    inline int terminal_output() { return asLeaf().terminal_output(); }

    /**@brief: returns the async communication batch size. */
    unsigned int get_async_batch_sz() { return asLeaf().get_async_batch_sz(); };

    /**@brief: returns the number of variables considered when performing
     * refinement*/
    unsigned int get_num_refine_vars() {
        return asLeaf().get_num_refine_vars();
    }

    /**@brief: return the pointer for containing evolution refinement variable
     * ids*/
    const unsigned int* get_refine_var_ids() {
        return asLeaf().get_refine_var_ids();
    }

    /**@brief return the wavelet tolerance function / value*/
    std::function<double(double, double, double)> get_wtol_function() {
        return asLeaf().get_wtol_function();
    }

    /**@brief compute ts offset value*/
    /**@brief LTS: level offset of the step ladder for the current mesh (see
     * ts::level_offset); collective over the mesh's global communicator*/
    unsigned int compute_lts_ts_offset() { return asLeaf().compute_lts_ts_offset(); }

#ifdef __CUDACC__
    device::MeshGPU*& get_meshgpu_device_ptr() {
        return asLeaf().get_meshgpu_device_ptr();
    }

    device::MeshGPU* get_meshgpu_host_handle() {
        return asLeaf().get_meshgpu_host_handle();
    }
#endif
};

/**
 * @brief CFL-stable timestep from the mesh's finest level.
 *
 * dt = cfl * domain_extent * (2^(maxDepth - lmax) / ele_order) / 2^maxDepth
 *
 * Solver-agnostic grid math (no equation content) -- previously copy-pasted into
 * every generated main.cpp at init and again on every remesh. `domain_extent` is
 * the physical width along an axis (e.g. COMPD_MAX[0] - COMPD_MIN[0]).
 */
inline double compute_cfl_dt(ot::Mesh* mesh, double cfl, double domain_extent,
                             unsigned int ele_order) {
    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);
    return cfl * (domain_extent *
                  ((1u << (m_uiMaxDepth - lmax)) / ((double)ele_order)) /
                  ((double)(1u << m_uiMaxDepth)));
}

/** @brief Largest per-block allocation (points) over the mesh's local blocks. */
inline unsigned int max_block_points(ot::Mesh* mesh) {
    const auto& blkList = mesh->getLocalBlockList();
    unsigned int m = 0;
    for (unsigned int i = 0; i < blkList.size(); i++) {
        unsigned int bsz = blkList[i].getAllocationSzX() *
                           blkList[i].getAllocationSzY() *
                           blkList[i].getAllocationSzZ();
        if (bsz > m) m = bsz;
    }
    return m;
}

/**
 * @brief (Re)allocate a derivative workspace sized to the mesh's largest block.
 *
 * Frees `existing` (delete[] nullptr is a safe no-op, so the first call passes
 * nullptr) and returns a fresh `count * max_block_points(mesh)` buffer. Replaces
 * the alloc/realloc loop that was copy-pasted into every generated main.cpp at
 * init, after initial refinement, and on every remesh. `count` is the only
 * per-solver input (the deriv-struct's count() / NUM_DERIVATIVES).
 */
inline double* realloc_deriv_workspace(double* existing, ot::Mesh* mesh,
                                       unsigned int count) {
    delete[] existing;
    return new double[(size_t)count * max_block_points(mesh)];
}

/**
 * @brief Seed a uniform block-adaptive octree covering a physical box.
 *
 * Fills `tmpNodes` with TreeNodes at `reg_level` tiling [box_min, box_max],
 * mapping physical -> octree coords by the affine transform (per axis)
 *   g = (oct_extent / phys_extent) * (c - phys_min) + oct_min.
 * This is the index-math loop every GR solver's main.cpp open-codes
 * (X_TO_GRIDX + stepSz + clamp + TreeNode push); only the box/level/domain
 * differ. `phys_*`/`oct_*` are the COMPD and OCTREE bounds (double[3]).
 */
inline void seed_block_adaptive_octree(
        std::vector<ot::TreeNode>& tmpNodes,
        const double* box_min, const double* box_max,
        const double* phys_min, const double* phys_max,
        const double* oct_min, const double* oct_max,
        unsigned int reg_level) {
    auto to_grid = [&](double c, int d) -> unsigned int {
        const double rg = oct_max[d] - oct_min[d];
        const double rp = phys_max[d] - phys_min[d];
        return (unsigned int)((rg / rp) * (c - phys_min[d]) + oct_min[d]);
    };
    const unsigned int xb = to_grid(box_min[0], 0), xe = to_grid(box_max[0], 0);
    const unsigned int yb = to_grid(box_min[1], 1), ye = to_grid(box_max[1], 1);
    const unsigned int zb = to_grid(box_min[2], 2), ze = to_grid(box_max[2], 2);
    const unsigned int stepSz = 1u << (m_uiMaxDepth - reg_level);
    const unsigned int octMax = 1u << m_uiMaxDepth;
    for (unsigned int x = xb; x < xe; x += stepSz)
        for (unsigned int y = yb; y < ye; y += stepSz)
            for (unsigned int z = zb; z < ze; z += stepSz) {
                unsigned int xc = (x >= octMax) ? x - 1 : x;
                unsigned int yc = (y >= octMax) ? y - 1 : y;
                unsigned int zc = (z >= octMax) ? z - 1 : z;
                tmpNodes.push_back(
                    ot::TreeNode(xc, yc, zc, reg_level, m_uiDim, m_uiMaxDepth));
            }
}

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::unzip(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
                                  unsigned int async_k) {
    if (!m_uiMesh->isActive()) return;
    dendro::stepprof::Scope spUzTotal(dendro::stepprof::CTX_UNZIP_TOTAL);

#ifdef __PROFILE_CTX__
    m_uiCtxpt[CTXPROFILE::UNZIP_WCOMM].start();
#endif

    // assert( (in.IsUnzip() == false) && (in.get_dof()== out.get_dof()) &&
    // (out.IsUnzip()==true) && (in.IsGhosted()==true) && async_k <=
    // in.get_dof());
    const unsigned int dof             = in.get_dof();
    T* in_ptr                          = in.get_vec_ptr();
    T* out_ptr                         = out.get_vec_ptr();

    const unsigned int sz_per_dof_zip  = in.get_size() / dof;
    const unsigned int sz_per_dof_uzip = out.get_size() / dof;

    assert(sz_per_dof_uzip == m_uiMesh->getDegOfFreedomUnZip());
    assert(sz_per_dof_zip == m_uiMesh->getDegOfFreedom());

    if (in.get_loc() == ot::DVEC_LOC::HOST) {
#ifdef DENDRO_UNZIP_OVERLAP
        // Comm/compute overlap: post all ghost exchanges, unzip the interior
        // (ghost-independent) blocks while the messages are in flight, then
        // wait for ghosts and unzip the boundary (ghost-dependent) blocks.
        // Interior blocks read only local data, so unzipping them before
        // readFromGhostEnd is correct (see flagBlockGhostDependancies).
        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);
            m_uiMesh->readFromGhostBegin(
                m_mpi_ctx[i], in_ptr + v_begin * sz_per_dof_zip, batch_sz);
        }

#ifdef __PROFILE_CTX__
        m_uiCtxpt[CTXPROFILE::UNZIP].start();
#endif
        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);
            m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip,
                            out_ptr + v_begin * sz_per_dof_uzip, batch_sz,
                            (int)ot::UNZIP_INDEPENDENT);
        }
#ifdef __PROFILE_CTX__
        m_uiCtxpt[CTXPROFILE::UNZIP].stop();
#endif

        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);
            m_uiMesh->readFromGhostEnd(
                m_mpi_ctx[i], in_ptr + v_begin * sz_per_dof_zip, batch_sz);
#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].start();
#endif
            m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip,
                            out_ptr + v_begin * sz_per_dof_uzip, batch_sz,
                            (int)ot::UNZIP_DEPENDENT);
#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].stop();
#endif
        }
#else
        // unzip on the host.
        const bool prof = ot::Mesh::meshProfOn();
        double t0 = prof ? MPI_Wtime() : 0.0;
        {
            dendro::stepprof::Scope spGb(dendro::stepprof::UZ_GHOST_BEGIN);
            for (unsigned int i = 0; i < async_k; i++) {
                const unsigned int v_begin  = ((i * dof) / async_k);
                const unsigned int v_end    = (((i + 1) * dof) / async_k);
                const unsigned int batch_sz = (v_end - v_begin);

                m_uiMesh->readFromGhostBegin(
                    m_mpi_ctx[i], in_ptr + v_begin * sz_per_dof_zip, batch_sz);
            }
        }
        if (prof) { const double t1 = MPI_Wtime(); ot::Mesh::s_dProfGhostBegin += t1 - t0; t0 = t1; }

        // DENDRO_UNZIP_SRC_SPLIT=1 (2 = also check against a plain unzip): scatter the
        // local elements that read no ghost node while the messages are in flight,
        // everything else after the wait. Each element is scattered once.
        static const int srcSplitMode = [] {
            const char* e = std::getenv("DENDRO_UNZIP_SRC_SPLIT");
            return e ? std::atoi(e) : 0;
        }();
#ifdef DENDRO_UNZIP_BATCH
        const bool srcSplit = false;
#else
        const bool srcSplit = srcSplitMode > 0 && !m_uiMesh->useLabelledUnzip();
#endif
        if (srcSplit) {
            dendro::stepprof::Scope spUz(dendro::stepprof::UZ_UNZIP_COMPUTE);
            for (unsigned int i = 0; i < async_k; i++) {
                const unsigned int v_begin  = ((i * dof) / async_k);
                const unsigned int v_end    = (((i + 1) * dof) / async_k);
                m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip,
                                out_ptr + v_begin * sz_per_dof_uzip, v_end - v_begin,
                                ot::Mesh::UNZIP_SRC_EARLY);
            }
        }

        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);

            {
                dendro::stepprof::Scope spGe(dendro::stepprof::UZ_GHOST_END);
                m_uiMesh->readFromGhostEnd(
                    m_mpi_ctx[i], in_ptr + v_begin * sz_per_dof_zip, batch_sz);
            }
            if (prof) { const double t1 = MPI_Wtime(); ot::Mesh::s_dProfGhostEnd += t1 - t0; t0 = t1; }

#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].start();
#endif

            {
                dendro::stepprof::Scope spUz(dendro::stepprof::UZ_UNZIP_COMPUTE);
#ifdef DENDRO_UNZIP_BATCH
                // Same work as Mesh::unzip below, but unzip_scatter_batch loops the
                // variables INSIDE one parallel region against a dof=1-sized all_dg,
                // instead of materializing a (numTotalElements * batch_sz * nPe)
                // scratch and zero-filling it on every call (mesh.tcc:11295). The DG
                // values, and therefore the results, are identical -- only the
                // scratch size and the fork/join count change.
                {
                    // batch_sz pointers (dof/async_k, so ~6): trivially small next
                    // to the per-element scratch this call avoids allocating.
                    std::vector<const T*> ins(batch_sz);
                    std::vector<T*> outs(batch_sz);
                    for (unsigned int v = 0; v < batch_sz; v++) {
                        ins[v]  = in_ptr + (v_begin + v) * sz_per_dof_zip;
                        outs[v] = out_ptr + (v_begin + v) * sz_per_dof_uzip;
                    }
                    m_uiMesh->unzip_scatter_batch(ins.data(), outs.data(),
                                                  batch_sz);
                }
#else
                m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip,
                                out_ptr + v_begin * sz_per_dof_uzip, batch_sz,
                                srcSplit ? ot::Mesh::UNZIP_SRC_LATE : -1);
                if (srcSplit && srcSplitMode == 2 && sizeof(T) == sizeof(double)) {
                    // check: same slots filled, values equal up to which writer won
                    static long calls = 0, bitDiff = 0;
                    static double maxAbs = 0;
                    const size_t n = (size_t)batch_sz * sz_per_dof_uzip;
                    std::vector<T> ref(n);
                    std::memcpy(ref.data(), out_ptr + v_begin * sz_per_dof_uzip, n * sizeof(T));
                    T* o = out_ptr + v_begin * sz_per_dof_uzip;
                    m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip, o, batch_sz);
                    bool bad = false;
                    double scale = 0;
                    for (size_t q = 0; q < n; q++) scale = std::max(scale, std::fabs((double)o[q]));
                    for (size_t q = 0; q < n; q++) {
                        if (std::memcmp(&o[q], &ref[q], sizeof(T)) == 0) continue;
                        bitDiff++;
                        const double d = std::fabs((double)o[q] - (double)ref[q]);
                        maxAbs = std::max(maxAbs, d);
                        if (!(d <= 1e-9 * (1.0 + scale))) bad = true;
                    }
                    if (bad || (++calls <= 3 && m_uiMesh->getMPIRank() == 0))
                        std::cout << "[unzip-src-split] check r" << m_uiMesh->getMPIRank() << ": slots=" << n
                                  << " cumulative bit-different=" << bitDiff << " max abs diff=" << maxAbs
                                  << (bad ? " FAILED" : "") << std::endl;
                    if (bad) MPI_Abort(m_uiMesh->getMPIGlobalCommunicator(), 98);
                }
#endif
            }
            if (prof) { const double t1 = MPI_Wtime(); ot::Mesh::s_dProfUnzip += t1 - t0; t0 = t1; }

#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].stop();
#endif
        }
#endif

    } else if (in.get_loc() == ot::DVEC_LOC::DEVICE) {
#ifdef __CUDACC__

        device::MeshGPU* dptr_mesh = this->get_meshgpu_device_ptr();
        device::MeshGPU* mesh_gpu  = this->get_meshgpu_host_handle();

        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);
            mesh_gpu->read_from_ghost_cg_begin<DEVICE_REAL, cudaStream_t>(
                m_mpi_ctx[i], m_mpi_ctx_device[i], m_uiMesh, dptr_mesh,
                in_ptr + v_begin * sz_per_dof_zip, batch_sz, 0);
        }

        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);
            mesh_gpu->read_from_ghost_cg_end<DEVICE_REAL, cudaStream_t>(
                m_mpi_ctx[i], m_mpi_ctx_device[i], m_uiMesh, dptr_mesh,
                in_ptr + v_begin * sz_per_dof_zip, batch_sz, 0);

#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].start();
#endif
            mesh_gpu->unzip_cg<DEVICE_REAL, cudaStream_t>(
                m_uiMesh, dptr_mesh, in_ptr + v_begin * sz_per_dof_zip,
                out_ptr + v_begin * sz_per_dof_uzip, batch_sz, 0);
            GPUDevice::device_synchronize();
#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].stop();
#endif
        }
#endif
    }

#ifdef __PROFILE_CTX__
    m_uiCtxpt[CTXPROFILE::UNZIP_WCOMM].stop();
#endif
}

/*template<typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx,T,I>::zip(ot::DVector<T,I>& in , ot::DVector<T,I>& out,
unsigned int async_k)
{

    #ifdef __PROFILE_CTX__
        m_uiCtxpt[CTXPROFILE::ZIP_WCOMM].start();
    #endif

    // assert( (in.IsUnzip() == true) && (in.get_dof()== out.get_dof()) &&
(out.IsUnzip()==false) && (out.IsGhosted()==true)); const unsigned int dof =
in.get_dof(); const unsigned int sz_per_dof_uzip = in.get_size()/dof; const
unsigned int sz_per_dof_zip = out.get_size()/dof;

    T* in_ptr = in.get_vec_ptr();
    T* out_ptr = out.get_vec_ptr();

    assert(sz_per_dof_uzip == m_uiMesh->getDegOfFreedomUnZip());
    assert(sz_per_dof_zip == m_uiMesh->getDegOfFreedom());

    for(unsigned int i=0 ; i < async_k; i++)
    {
        const unsigned int v_begin = ((i*dof)/async_k);
        const unsigned int v_end   = (((i+1)*dof)/async_k);

        #ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::ZIP].start();
        #endif

        for(unsigned int j=v_begin; j < v_end; j++)
        {
            m_uiMesh->zip(in_ptr + j*sz_per_dof_uzip,out_ptr +
j*sz_per_dof_zip);
        }

        #ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::ZIP].stop();
        #endif


        if(i>0)
        {
            const unsigned int vb_prev = (((i-1)*dof)/async_k);
            const unsigned int ve_prev = ((i*dof)/async_k);

            for(unsigned int j=vb_prev; j < ve_prev; j++ )
                m_uiMesh->readFromGhostBegin(out_ptr + j*sz_per_dof_zip,1);
        }

    }

    // the last batch.
    for(unsigned int j= (((async_k-1)*dof)/async_k); j <
((async_k*dof)/async_k); j++ ) m_uiMesh->readFromGhostBegin(out_ptr +
j*sz_per_dof_zip,1);

    for(unsigned int j=0; j < dof; j++)
        m_uiMesh->readFromGhostEnd(out_ptr + j*sz_per_dof_zip,1);


    #ifdef __PROFILE_CTX__
        m_uiCtxpt[CTXPROFILE::ZIP_WCOMM].stop();
    #endif

    return;

}*/

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::zip(ot::DVector<T, I>& in, ot::DVector<T, I>& out) {
    if (!m_uiMesh->isActive()) return;
    dendro::stepprof::Scope spZipTotal(dendro::stepprof::CTX_ZIP_TOTAL);

    // assert( (in.IsUnzip() == true) && (in.get_dof()== out.get_dof()) &&
    // (out.IsUnzip()==false) && (out.IsGhosted()==true));
    const unsigned int dof             = in.get_dof();
    const unsigned int sz_per_dof_uzip = (dof != 0) ? in.get_size() / dof : 0;
    const unsigned int sz_per_dof_zip  = (dof != 0) ? out.get_size() / dof : 0;

    T* in_ptr                          = in.get_vec_ptr();
    T* out_ptr                         = out.get_vec_ptr();

    assert(sz_per_dof_uzip == m_uiMesh->getDegOfFreedomUnZip());
    assert(sz_per_dof_zip == m_uiMesh->getDegOfFreedom());

#ifdef __PROFILE_CTX__
    m_uiCtxpt[CTXPROFILE::ZIP].start();
#endif
    if (in.get_loc() == ot::DVEC_LOC::HOST) {
        for (unsigned int j = 0; j < dof; j++)
            m_uiMesh->zip(in_ptr + j * sz_per_dof_uzip,
                          out_ptr + j * sz_per_dof_zip);
    } else if (in.get_loc() == ot::DVEC_LOC::DEVICE) {
#ifdef __CUDACC__
        device::MeshGPU* dptr_mesh = this->get_meshgpu_device_ptr();
        device::MeshGPU* mesh_gpu  = this->get_meshgpu_host_handle();
        mesh_gpu->zip_cg<DEVICE_REAL, cudaStream_t>(m_uiMesh, dptr_mesh, in_ptr,
                                                    out_ptr, out.get_dof(), 0);
        GPUDevice::device_synchronize();
#endif
    }

#ifdef __PROFILE_CTX__
    m_uiCtxpt[CTXPROFILE::ZIP].stop();
#endif
}

template <typename DerivedCtx, typename T, typename I>
ot::Mesh* Ctx<DerivedCtx, T, I>::remesh(unsigned int grain_sz, double ld_tol,
                                        unsigned int sf_k) {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[CTXPROFILE::REMESH].start();
#endif

#ifdef DEBUG_IS_REMESH
    unsigned int rank   = m_uiMesh->getMPIRankGlobal();
    MPI_Comm globalComm = m_uiMesh->getMPIGlobalCommunicator();
    std::vector<ot::TreeNode> unChanged;
    std::vector<ot::TreeNode> refined;
    std::vector<ot::TreeNode> coarsened;
    std::vector<ot::TreeNode> localBlocks;

    const ot::Block* blkList = &(*(m_uiMesh->getLocalBlockList().begin()));
    for (unsigned int ele = 0; ele < m_uiMesh->getLocalBlockList().size();
         ele++) {
        localBlocks.push_back(blkList[ele].getBlockNode());
    }

    const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin()));
    for (unsigned int ele = m_uiMesh->getElementLocalBegin();
         ele < m_uiMesh->getElementLocalEnd(); ele++) {
        if ((pNodes[ele].getFlag() >> NUM_LEVEL_BITS) == OCT_NO_CHANGE) {
            unChanged.push_back(pNodes[ele]);
        } else if ((pNodes[ele].getFlag() >> NUM_LEVEL_BITS) == OCT_SPLIT) {
            refined.push_back(pNodes[ele]);
        } else {
            assert((pNodes[ele].getFlag() >> NUM_LEVEL_BITS) == OCT_COARSE);
            coarsened.push_back(pNodes[ele]);
        }
    }

    char fN1[256];
    char fN2[256];
    char fN3[256];
    char fN4[256];

    sprintf(fN1, "unchanged_%d", m_uiCurrentStep);
    sprintf(fN2, "refined_%d", m_uiCurrentStep);
    sprintf(fN3, "coarsend_%d", m_uiCurrentStep);
    sprintf(fN4, "blocks_%d", m_uiCurrentStep);

    DendroIntL localSz = unChanged.size();
    DendroIntL globalSz;
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, globalComm);
    if (!rank) std::cout << " total unchanged: " << globalSz << std::endl;

    localSz = refined.size();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, globalComm);
    if (!rank) std::cout << " total refined: " << globalSz << std::endl;

    localSz = coarsened.size();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, globalComm);
    if (!rank) std::cout << " total coarsend: " << globalSz << std::endl;

    io::vtk::oct2vtu(&(*(unChanged.begin())), unChanged.size(), fN1,
                     globalComm);
    io::vtk::oct2vtu(&(*(refined.begin())), refined.size(), fN2, globalComm);
    io::vtk::oct2vtu(&(*(coarsened.begin())), coarsened.size(), fN3,
                     globalComm);
    io::vtk::oct2vtu(&(*(localBlocks.begin())), localBlocks.size(), fN4,
                     globalComm);

#endif

    ot::Mesh* newMesh = m_uiMesh->ReMesh(grain_sz, ld_tol, sf_k, m_getWeight);

#ifdef __PROFILE_CTX__
    m_uiCtxpt[CTXPROFILE::REMESH].stop();
#endif

    return newMesh;
}

template <typename DerivedCtx, typename T, typename I>
int Ctx<DerivedCtx, T, I>::remesh_and_gridtransfer(unsigned int grain_sz,
                                                   double ld_tol,
                                                   unsigned int sf_k) {
    dendro::logger::debug(dendro::logger::Scope{"CTX"},
                          "Now doing full remesh and grid transfer!");

    // probe (off by default): dump EV at base-class boundaries to align
    // with graph-mode sandwich bisection. tags: a = pre-remesh (matches
    // graph-mode a-pre-g2s); d = post-IGT (matches graph-mode d-post-s2g).
    // gate: EM4_REMESH_DUMP_DIR. one call_id per remesh.
    static int s_remesh_call_id = 0;
    const char* remesh_dump_dir = DENDRO_PROBE_GETENV("EM4_REMESH_DUMP_DIR");
    auto dumpEV = [&](const ot::Mesh* pMesh, const char tag) {
        if (!remesh_dump_dir) return;
        if (!pMesh || !pMesh->isActive()) return;
        T* dvec_ptr = static_cast<DerivedCtx*>(this)->
                        get_evolution_vars().get_vec_ptr();
        const unsigned int DOFV =
            static_cast<DerivedCtx*>(this)->get_evolution_vars().get_dof();
        if (!dvec_ptr) return;
        const unsigned int szPDof = pMesh->getDegOfFreedom();
        const auto& pNodes = pMesh->getAllElements();
        const unsigned int NLB = pMesh->getNodeLocalBegin();
        const unsigned int NLE = pMesh->getNodeLocalEnd();
        const unsigned int eOrd = pMesh->getElementOrder();
        const unsigned int nPe = pMesh->getNumNodesPerElement();
        const unsigned int maxD = m_uiMaxDepth;
        const auto& cg2dg = pMesh->getCG2DGMap();
        const int mpiRank = pMesh->getMPIRank();
        for (unsigned int v = 0; v < DOFV; v++) {
            T* vec_v = dvec_ptr + v * szPDof;
            char fn[1024];
            std::snprintf(fn, sizeof(fn),
                          "%s/remesh%d_%c_v%u_r%d.txt",
                          remesh_dump_dir, s_remesh_call_id, tag,
                          v, mpiRank);
            FILE* fp = std::fopen(fn, "w");
            if (!fp) continue;
            const unsigned int NACT = pMesh->getDegOfFreedom();
            std::fprintf(fp,
                "# remesh=%d tag=%c v=%u rank=%d NLB=%u NLE=%u NACT=%u\n"
                "# loc cg phys_x phys_y phys_z hex\n",
                s_remesh_call_id, tag, v, mpiRank, NLB, NLE, NACT);
            for (unsigned int cg = 0; cg < NACT; cg++) {
                if (cg >= cg2dg.size()) continue;
                const unsigned int dg = cg2dg[cg];
                const unsigned int oe = dg / nPe;
                const unsigned int on = dg % nPe;
                if (oe >= pNodes.size()) continue;
                const unsigned int oni = on % (eOrd+1);
                const unsigned int onj = (on/(eOrd+1)) % (eOrd+1);
                const unsigned int onk = on / ((eOrd+1)*(eOrd+1));
                const ot::TreeNode& oTN = pNodes[oe];
                const unsigned int olen =
                    (unsigned int)1u << (maxD - oTN.getLevel());
                const unsigned long long px =
                    (unsigned long long)oTN.getX() * eOrd
                    + (unsigned long long)oni * olen;
                const unsigned long long py =
                    (unsigned long long)oTN.getY() * eOrd
                    + (unsigned long long)onj * olen;
                const unsigned long long pz =
                    (unsigned long long)oTN.getZ() * eOrd
                    + (unsigned long long)onk * olen;
                uint64_t hb = 0;
                T val = vec_v[cg];
                std::memcpy(&hb, &val, sizeof(hb));
                const char loc = (cg >= NLB && cg < NLE) ? 'L' : 'G';
                std::fprintf(fp, "%c %u %llu %llu %llu %lx\n",
                             loc, cg, px, py, pz, (unsigned long)hb);
            }
            std::fclose(fp);
        }
    };

    dumpEV(m_uiMesh, 'a');
    ot::Mesh* newMesh      = remesh(grain_sz, ld_tol, sf_k);

    DendroIntL oldElements = m_uiMesh->getNumLocalMeshElements();
    DendroIntL newElements = newMesh->getNumLocalMeshElements();

    DendroIntL oldElements_g, newElements_g;

    par::Mpi_Reduce(&oldElements, &oldElements_g, 1, MPI_SUM, 0,
                    m_uiMesh->getMPIGlobalCommunicator());
    par::Mpi_Reduce(&newElements, &newElements_g, 1, MPI_SUM, 0,
                    newMesh->getMPIGlobalCommunicator());

    dendro::logger::info(
        dendro::logger::Scope{"CTX"},
        "step : {} | time : {} | old mesh size : {} | new mesh size : {}",
        m_uiTinfo._m_uiStep, m_uiTinfo._m_uiT, oldElements_g, newElements_g);

    this->grid_transfer(newMesh);
    dumpEV(newMesh, 'd');
    if (remesh_dump_dir) s_remesh_call_id++;

    std::swap(newMesh, m_uiMesh);
    delete newMesh;

#ifdef __CUDACC__
    device::MeshGPU*& dptr_mesh = this->get_meshgpu_device_ptr();
    device::MeshGPU* mesh_gpu   = this->get_meshgpu_host_handle();

    mesh_gpu->dealloc_mesh_on_device(dptr_mesh);
    dptr_mesh = mesh_gpu->alloc_mesh_on_device(m_uiMesh);
#endif

    m_uiIsETSSynced = false;
    dendro::logger::info(dendro::logger::Scope{"CTX"},
                         "Finished performing full remesh and grid transfer!");
    return 0;
}

}  // end of namespace ts
