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
#include <mpi.h>

#include <functional>
#include <iostream>
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

    std::vector<unsigned int> m_uiTotalBytesSend;
    std::vector<unsigned int> m_uiTotalBytesRecv;
    std::vector<unsigned int> m_uiTotalBytesSendCompress;
    std::vector<unsigned int> m_uiTotalBytesRecvCompress;
    unsigned int m_uiTotalBytesCounter = 0;

   public:
    /**@brief: default constructor*/
    Ctx(){};

    /**@brief: default destructor*/
    ~Ctx(){};

    /**@brief: derived class static cast*/
    inline DerivedCtx& asLeaf() { return static_cast<DerivedCtx&>(*this); }

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
     * @brief compute the block for the rhs (used in LTS).
     * @param in :  blk vectot in
     * @param out : blk vector out
     * @param local_blk_id : blkid
     * @param blk_time : blk time
     * @return int
     */
    int rhs_blk(const T* in, T* out, unsigned int dof,
                unsigned int local_blk_id, T blk_time) {
        return asLeaf().rhs_blk(in, out, dof, local_blk_id, blk_time);
    }

    /**
     * @brief apply pre_stage compute for u vector F(u) at the block level.
     * @param in :  blk vectot in
     * @param out : blk vector out
     * @param local_blk_id : blkid
     * @param blk_time : blk time
     * @return int
     */
    int pre_stage_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                      T blk_time) {
        return asLeaf().pre_stage_blk(in, dof, local_blk_id, blk_time);
    }

    /**
     * @brief apply post_stage compute for v vector v=F(u) at the block level.
     * @param in :  blk vectot in
     * @param out : blk vector out
     * @param local_blk_id : blkid
     * @param blk_time : blk time
     * @return int
     */
    int post_stage_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                       T blk_time) {
        return asLeaf().post_stage_blk(in, dof, local_blk_id, blk_time);
    }

    /**
     * @brief apply pre_timestep operation before entering the rk solver  apply
     * for y_{k}.
     * @param in :  blk vectot in
     * @param out : blk vector out
     * @param local_blk_id : blkid
     * @param blk_time : blk time
     * @return int
     */
    int pre_timestep_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                         T blk_time) {
        return asLeaf().pre_timestep_blk(in, dof, local_blk_id, blk_time);
    }

    /**
     * @brief apply post_timestep operation for each y_{k+1} computed from y_{k}
     * @param in :  blk vectot in
     * @param out : blk vector out
     * @param local_blk_id : blkid
     * @param blk_time : blk time
     * @return int
     */
    int post_timestep_blk(T* in, unsigned int dof, unsigned int local_blk_id,
                          T blk_time) {
        return asLeaf().post_timestep_blk(in, dof, local_blk_id, blk_time);
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

    void process_finished_unzip(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
                                unsigned int async_k, bool use_compression,
                                std::vector<int>& completed_batches,
                                std::vector<MPI_Request>& send_requests,
                                std::vector<MPI_Request>& recv_requests,
                                std::vector<unsigned int>& send_requests_ctx,
                                std::vector<unsigned int>& recv_requests_ctx,
                                std::vector<MPI_Status>& statuses);

    void unzip_rewrite(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
                       unsigned int async_k, bool use_compression);

    void unzip_device(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
                      unsigned int async_k, bool use_compression);

    void unzip_host_compression(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
                                unsigned int async_k);

    void unzip_host_default(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
                            unsigned int async_k);

    /**
     * @brief performs unzip operation,
     *
     * @param in : input zip vector.
     * @param out : output unzip vector.
     * @param async_k : async communicator.
     */
    void unzip(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
               unsigned int async_k = 1, bool use_compression = false);

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
    void compute_lts_ts_offset() { return asLeaf().compute_lts_ts_offset(); }

#ifdef __CUDACC__
    device::MeshGPU*& get_meshgpu_device_ptr() {
        return asLeaf().get_meshgpu_device_ptr();
    }

    device::MeshGPU* get_meshgpu_host_handle() {
        return asLeaf().get_meshgpu_host_handle();
    }
#endif

    /**@brief: retunrs time step size factor for the  specified block*/
    static unsigned int getBlkTimestepFac(unsigned int blev, unsigned int lmin,
                                          unsigned int lmax) {
        return DerivedCtx::getBlkTimestepFac(blev, lmin, lmax);
    }

    inline void prepareBytesVectors() {
        m_uiTotalBytesSend.assign(m_uiMesh->getMPICommSizeGlobal(), 0);
        m_uiTotalBytesRecv.assign(m_uiMesh->getMPICommSizeGlobal(), 0);
        m_uiTotalBytesSendCompress.assign(m_uiMesh->getMPICommSizeGlobal(), 0);
        m_uiTotalBytesRecvCompress.assign(m_uiMesh->getMPICommSizeGlobal(), 0);

        m_uiTotalBytesCounter = 0;
    }

    inline void averageBytesVectors() {
        // assumes we haven't done this yet!
        for (auto& ele : m_uiTotalBytesSend) {
            ele /= m_uiTotalBytesCounter;
        }
        for (auto& ele : m_uiTotalBytesRecv) {
            ele /= m_uiTotalBytesCounter;
        }
        for (auto& ele : m_uiTotalBytesSendCompress) {
            ele /= m_uiTotalBytesCounter;
        }
        for (auto& ele : m_uiTotalBytesRecvCompress) {
            ele /= m_uiTotalBytesCounter;
        }
    }

    inline std::vector<unsigned int>& getTotalBytesSend() {
        return m_uiTotalBytesSend;
    }
    inline std::vector<unsigned int>& getTotalBytesRecv() {
        return m_uiTotalBytesRecv;
    }
    inline std::vector<unsigned int>& getTotalBytesSendCompress() {
        return m_uiTotalBytesSendCompress;
    }
    inline std::vector<unsigned int>& getTotalBytesRecvCompress() {
        return m_uiTotalBytesRecvCompress;
    }

    unsigned int getTotalBytesSendSum() const {
        return std::accumulate(m_uiTotalBytesSend.begin(),
                               m_uiTotalBytesSend.end(), 0);
    }
    unsigned int getTotalBytesRecvSum() const {
        return std::accumulate(m_uiTotalBytesRecv.begin(),
                               m_uiTotalBytesRecv.end(), 0);
    }
    unsigned int getTotalBytesSendCompressSum() const {
        return std::accumulate(m_uiTotalBytesSendCompress.begin(),
                               m_uiTotalBytesSendCompress.end(), 0);
    }
    unsigned int getTotalBytesRecvCompressSum() const {
        return std::accumulate(m_uiTotalBytesRecvCompress.begin(),
                               m_uiTotalBytesRecvCompress.end(), 0);
    }
};

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::process_finished_unzip(
    ot::DVector<T, I>& in, ot::DVector<T, I>& out, unsigned int async_k,
    bool use_compression, std::vector<int>& completed_indices,
    std::vector<MPI_Request>& send_requests,
    std::vector<MPI_Request>& recv_requests,
    std::vector<unsigned int>& send_requests_ctx,
    std::vector<unsigned int>& recv_requests_ctx,
    std::vector<MPI_Status>& statuses) {
    if (!m_uiMesh->isActive()) return;
    // NOTE: this function should never be called by the device code

    const unsigned int dof             = in.get_dof();
    T* in_ptr                          = in.get_vec_ptr();
    T* out_ptr                         = out.get_vec_ptr();

    const unsigned int sz_per_dof_zip  = in.get_size() / dof;
    const unsigned int sz_per_dof_uzip = out.get_size() / dof;
    T* temp_ptr;

    // preallocate the completed indices and statuses to use in both Testsome
    completed_indices.resize(
        std::max(send_requests.size(), recv_requests.size()));
    statuses.resize(completed_indices.size());

    int outcount = 0;
    if (!send_requests.empty()) {
        // HANDLE completed sends
        dendro::timer::t_compression_wait_comms.start();
        MPI_Testsome(send_requests.size(), send_requests.data(), &outcount,
                     completed_indices.data(), statuses.data());

        // NOTE: insert iteration over the completed values if things need to be
        // done with the send requests

        // if the sends are completed, we can clear them from the request list
        if (outcount > 0) {
            dendro::timer::t_compression_wait_comms.stop();
            std::vector<int> indices_remove(
                completed_indices.begin(),
                completed_indices.begin() + outcount);

            std::sort(indices_remove.begin(), indices_remove.end(),
                      std::greater<int>());

            for (int index : indices_remove) {
                if (index < send_requests.size()) {
                    // remove the recv requests by swapping to end, and popping
                    // back
                    send_requests[index] = std::move(send_requests.back());
                    send_requests.pop_back();

                    // then do the same for the ctx vector
                    send_requests_ctx[index] =
                        std::move(send_requests_ctx.back());
                    send_requests_ctx.pop_back();
                }
            }
        } else {
            dendro::timer::t_compression_wait_comms.stop();
        }
    }

    outcount = 0;
    if (!recv_requests.empty()) {
        // HANDLE COMPLETED RECEIVES
        dendro::timer::t_compression_wait_comms.start();
        MPI_Testsome(recv_requests.size(), recv_requests.data(), &outcount,
                     completed_indices.data(), statuses.data());

        if (outcount > 0) {
            dendro::timer::t_compression_wait_comms.stop();
            for (int i = 0; i < outcount; ++i) {
                // this handles any completed send requests
                const unsigned int ctx_idx =
                    recv_requests_ctx[completed_indices[i]];

                const unsigned int v_begin  = (ctx_idx * dof) / async_k;
                const unsigned int v_end    = ((ctx_idx + 1) * dof) / async_k;
                const unsigned int batch_sz = v_end - v_begin;

                temp_ptr                    = in_ptr + v_begin * sz_per_dof_zip;

                if (use_compression) {
                    // need to decompress to the recv buffer
                    m_uiMesh->decompressSingleProcess<T>(
                        m_mpi_ctx[ctx_idx], batch_sz, statuses[i].MPI_SOURCE);
                }

                m_uiMesh->unextractSingleProcess(m_mpi_ctx[ctx_idx], temp_ptr,
                                                 batch_sz,
                                                 statuses[i].MPI_SOURCE);
            }
            // make sure to remove the values from recv_requests_ctx

            std::vector<int> indices_remove(
                completed_indices.begin(),
                completed_indices.begin() + outcount);

            std::sort(indices_remove.begin(), indices_remove.end(),
                      std::greater<int>());

            for (int index : indices_remove) {
                if (index < recv_requests.size()) {
                    // remove the recv requests by swapping to end, and popping
                    // back
                    recv_requests[index] = std::move(recv_requests.back());
                    recv_requests.pop_back();

                    // then do the same for the ctx vector
                    recv_requests_ctx[index] =
                        std::move(recv_requests_ctx.back());
                    recv_requests_ctx.pop_back();
                }
            }
        } else {
            dendro::timer::t_compression_wait_comms.stop();
        }
    }
}

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::unzip_host_compression(ot::DVector<T, I>& in,
                                                   ot::DVector<T, I>& out,
                                                   unsigned int async_k) {
    const unsigned int dof             = in.get_dof();
    T* in_ptr                          = in.get_vec_ptr();
    T* out_ptr                         = out.get_vec_ptr();

    const unsigned int sz_per_dof_zip  = in.get_size() / dof;
    const unsigned int sz_per_dof_uzip = out.get_size() / dof;

    assert(sz_per_dof_uzip == m_uiMesh->getDegOfFreedomUnZip());
    assert(sz_per_dof_zip == m_uiMesh->getDegOfFreedom());

    std::vector<int> completed_indices;
    std::vector<MPI_Request> send_requests, recv_requests;
    std::vector<unsigned int> send_requests_ctx, recv_requests_ctx;
    std::vector<MPI_Status> statuses;

    int mpi_comm_tag_compression   = 5098;

    // a vector of send_requests based on the size we need
    const unsigned int n_send_proc = m_uiMesh->getSendProcListSize();
    const unsigned int n_recv_proc = m_uiMesh->getRecvProcListSize();
    std::vector<MPI_Request> size_requests(n_send_proc + n_recv_proc);

    T* temp_ptr_next;
    const unsigned int THRESHOLD = m_uiMesh->getMPICommSize() * 1;

    for (unsigned int i = 0; i < async_k; i++) {
        // we need to know where we're at with our variables
        const unsigned int v_begin  = (i * dof) / async_k;
        const unsigned int v_end    = ((i + 1) * dof) / async_k;
        const unsigned int batch_sz = v_end - v_begin;
        unsigned int compressOffset = 0;

        auto& send_compress_counts  = m_mpi_ctx[i].getSendCompressCounts();
        auto& recv_compress_counts  = m_mpi_ctx[i].getReceiveCompressCounts();
        auto& send_compress_offsets = m_mpi_ctx[i].getSendCompressOffsets();
        auto& recv_compress_offsets = m_mpi_ctx[i].getReceiveCompressOffsets();

        // allocate the recv_requests for size
        recv_requests.reserve(recv_requests.size() +
                              m_uiMesh->getRecvProcList().size());
        recv_requests_ctx.reserve(recv_requests_ctx.size() +
                                  m_uiMesh->getRecvProcList().size());
        send_requests.reserve(send_requests.size() +
                              m_uiMesh->getSendProcList().size());
        send_requests_ctx.reserve(send_requests_ctx.size() +
                                  m_uiMesh->getSendProcList().size());

        // IMPORTANT: this is the pointer to the current batch of data!
        T* temp_ptr = in_ptr + v_begin * sz_per_dof_zip;

        // make sure send compress counts is filled with zeros!
        std::fill(send_compress_counts.begin(), send_compress_counts.end(), 0);
        std::fill(recv_compress_counts.begin(), recv_compress_counts.end(), 0);
        send_compress_offsets[0] = recv_compress_offsets[0] = 0;

        for (unsigned int proc_id = 0; proc_id < n_recv_proc; ++proc_id) {
            unsigned int recv_p_id = m_uiMesh->getRecvProcList()[proc_id];
            par::Mpi_Irecv(&recv_compress_counts[recv_p_id], 1, recv_p_id,
                           mpi_comm_tag_compression,
                           m_uiMesh->getMPICommunicator(),
                           &size_requests[proc_id]);
        }

        // for each process that needs data, we need to extract the data out
        for (unsigned int proc_id = 0; proc_id < n_send_proc; ++proc_id) {
            unsigned int send_p_id = m_uiMesh->getSendProcList()[proc_id];
            // extract and then compress the data
            m_uiMesh->extractFullSingleProcess(m_mpi_ctx[i], temp_ptr, batch_sz,
                                               send_p_id);
            m_uiMesh->compressSingleProcess(m_mpi_ctx[i], temp_ptr, batch_sz,
                                            send_p_id, compressOffset);

            // now we set up the send part of our non-blocking "all-to-all"
            // NOTE: size_requests is offset by **recv** procid
            par::Mpi_Isend(&send_compress_counts[send_p_id], 1, send_p_id,
                           mpi_comm_tag_compression,
                           m_uiMesh->getMPICommunicator(),
                           &size_requests[n_recv_proc + proc_id]);
        }

        // TODO: potentially start extracting the next one

        // compute the offsets for the send values
        omp_par::scan(&(*(send_compress_counts.begin())),
                      &(*(send_compress_offsets.begin())),
                      send_compress_counts.size());

        // now we want to process our send sizes, but we need them all to
        // finish because we need the proper receive offsets
        MPI_Waitall(size_requests.size(), size_requests.data(),
                    MPI_STATUSES_IGNORE);

        // compute the offsets for the recv values
        omp_par::scan(&(*(recv_compress_counts.begin())),
                      &(*(recv_compress_offsets.begin())),
                      recv_compress_counts.size());

        // then we can set up the sends and receives, they can just get started
        m_uiMesh->setUpSendRecvCompressionRequests<T>(
            m_mpi_ctx[i], send_requests, recv_requests, send_requests_ctx,
            recv_requests_ctx, i);

        // then if we have enough, we can start processing some communications,
        // while others finish
        if (send_requests.size() + recv_requests.size() > THRESHOLD) {
            this->process_finished_unzip(
                in, out, async_k, true, completed_indices, send_requests,
                recv_requests, send_requests_ctx, recv_requests_ctx, statuses);
        }

        ++mpi_comm_tag_compression;
    }

    // as long as we have active requests, we need to try and clear them out
    while (!send_requests.empty() || !recv_requests.empty()) {
        this->process_finished_unzip(
            in, out, async_k, true, completed_indices, send_requests,
            recv_requests, send_requests_ctx, recv_requests_ctx, statuses);
    }

    // TEMP: this is only to gather information about the
    // compression/decompression
    for (unsigned i = 0; i < async_k; i++) {
        const unsigned int v_begin  = ((i * dof) / async_k);
        const unsigned int v_end    = (((i + 1) * dof) / async_k);
        const unsigned int batch_sz = (v_end - v_begin);
        for (unsigned int j = 0; j < m_uiMesh->getMPICommSize(); j++) {
            m_uiTotalBytesSend[j] +=
                m_uiMesh->getNodalSendCounts()[j] * batch_sz * sizeof(T);
            m_uiTotalBytesRecv[j] +=
                m_uiMesh->getNodalRecvCounts()[j] * batch_sz * sizeof(T);
            // then the compress amounts, which is the *total* amount not
            // including batch syze
            m_uiTotalBytesSendCompress[j] +=
                m_mpi_ctx[i].getSendCompressCounts()[j];
            m_uiTotalBytesRecvCompress[j] +=
                m_mpi_ctx[i].getReceiveCompressCounts()[j];
        }
    }

    // now that they're all done, we can have the mesh do its own unzip
    // TODO: this could probably be thrown in process_finished_unzip, but
    // would need to track completed ctx lists since unzip isn't currently
    // async
    dendro::timer::t_compression_uzip_post.start();
    for (unsigned int i = 0; i < async_k; i++) {
        const unsigned int v_begin  = ((i * dof) / async_k);
        const unsigned int v_end    = (((i + 1) * dof) / async_k);
        const unsigned int batch_sz = (v_end - v_begin);

        m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip,
                        out_ptr + v_begin * sz_per_dof_uzip, batch_sz);
    }
    dendro::timer::t_compression_uzip_post.stop();
}

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::unzip_host_default(ot::DVector<T, I>& in,
                                               ot::DVector<T, I>& out,
                                               unsigned int async_k) {
    const unsigned int dof             = in.get_dof();
    T* in_ptr                          = in.get_vec_ptr();
    T* out_ptr                         = out.get_vec_ptr();

    const unsigned int sz_per_dof_zip  = in.get_size() / dof;
    const unsigned int sz_per_dof_uzip = out.get_size() / dof;

    assert(sz_per_dof_uzip == m_uiMesh->getDegOfFreedomUnZip());
    assert(sz_per_dof_zip == m_uiMesh->getDegOfFreedom());

    std::vector<int> completed_indices;
    std::vector<MPI_Request> send_requests, recv_requests;
    std::vector<unsigned int> send_requests_ctx, recv_requests_ctx;
    std::vector<MPI_Status> statuses;

    T *temp_ptr, *temp_ptr_next;
    const unsigned int THRESHOLD = m_uiMesh->getMPICommSize() * 1;

    for (unsigned int i = 0; i < async_k; i++) {
        // we need to know where we're at with our variables
        const unsigned int v_begin  = (i * dof) / async_k;
        const unsigned int v_end    = ((i + 1) * dof) / async_k;
        const unsigned int batch_sz = v_end - v_begin;

        temp_ptr                    = in_ptr + v_begin * sz_per_dof_zip;

        // start by extracting the data for this batch

        // only the first one will be extracted on the first iter of the loop,
        // since we start extracting while waiting for IAlltoAll
        if (i == 0) m_uiMesh->extractFullData(m_mpi_ctx[i], temp_ptr, batch_sz);

        m_uiMesh->setUpSendRecvRequests<T>(
            m_mpi_ctx[i], batch_sz, send_requests, recv_requests,
            send_requests_ctx, recv_requests_ctx, i);

        // do some work while we wait for processes to get here..., might as
        // well start extracting the next chunk, which doesn't take long
        if (i < async_k - 1) {
            const unsigned int v_begin_next  = ((i + 1) * dof) / async_k;
            const unsigned int v_end_next    = ((i + 2) * dof) / async_k;
            const unsigned int batch_sz_next = v_end_next - v_begin_next;

            temp_ptr_next = in_ptr + v_begin_next * sz_per_dof_zip;
            m_uiMesh->extractFullData(m_mpi_ctx[i + 1], temp_ptr_next,
                                      batch_sz_next);
        }

        // as long as we have "more than our threshold" we can actually handle
        // stuff
        if (send_requests.size() + recv_requests.size() > THRESHOLD) {
            this->process_finished_unzip(
                in, out, async_k, false, completed_indices, send_requests,
                recv_requests, send_requests_ctx, recv_requests_ctx, statuses);
        }
    }

    // as long as we have active requests, we need to try and clear them out
    while (!send_requests.empty() || !recv_requests.empty()) {
        this->process_finished_unzip(
            in, out, async_k, false, completed_indices, send_requests,
            recv_requests, send_requests_ctx, recv_requests_ctx, statuses);
    }

    // TEMP: this is only to gather information about the
    // compression/decompression
    for (unsigned i = 0; i < async_k; i++) {
        const unsigned int v_begin  = ((i * dof) / async_k);
        const unsigned int v_end    = (((i + 1) * dof) / async_k);
        const unsigned int batch_sz = (v_end - v_begin);
        for (unsigned int j = 0; j < m_uiMesh->getMPICommSize(); j++) {
            m_uiTotalBytesSend[j] +=
                m_uiMesh->getNodalSendCounts()[j] * batch_sz * sizeof(T);
            m_uiTotalBytesRecv[j] +=
                m_uiMesh->getNodalRecvCounts()[j] * batch_sz * sizeof(T);
            // then the compress amounts, which is the *total* amount not
            // including batch syze
            m_uiTotalBytesSendCompress[j] +=
                m_uiMesh->getNodalSendCounts()[j] * batch_sz * sizeof(T);
            m_uiTotalBytesRecvCompress[j] +=
                m_uiMesh->getNodalRecvCounts()[j] * batch_sz * sizeof(T);
        }
    }
}

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::unzip(ot::DVector<T, I>& in, ot::DVector<T, I>& out,
                                  unsigned int async_k, bool use_compression) {
    // ONLY ACTIVE PROCS IN THE MESH INTERACT
    if (!m_uiMesh->isActive()) return;

    // if we're on the device, we should skip this function and call
    // unzip_device
    if (in.get_loc() == ot::DVEC_LOC::DEVICE) {
        this->unzip_device(in, out, async_k, use_compression);
        return;
    }

    if (use_compression) {
        this->unzip_host_compression(in, out, async_k);
    } else {
        this->unzip_host_default(in, out, async_k);
    }

    const unsigned int dof             = in.get_dof();
    T* in_ptr                          = in.get_vec_ptr();
    T* out_ptr                         = out.get_vec_ptr();

    const unsigned int sz_per_dof_zip  = in.get_size() / dof;
    const unsigned int sz_per_dof_uzip = out.get_size() / dof;

    // now that they're all done, we can have the mesh do its own unzip
    // TODO: this could probably be thrown in process_finished_unzip, but
    // would need to track completed ctx lists since unzip isn't currently
    // async
    dendro::timer::t_compression_uzip_post.start();
    for (unsigned int i = 0; i < async_k; i++) {
        const unsigned int v_begin  = ((i * dof) / async_k);
        const unsigned int v_end    = (((i + 1) * dof) / async_k);
        const unsigned int batch_sz = (v_end - v_begin);

        m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip,
                        out_ptr + v_begin * sz_per_dof_uzip, batch_sz);
    }
    dendro::timer::t_compression_uzip_post.stop();
}

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::unzip_device(ot::DVector<T, I>& in,
                                         ot::DVector<T, I>& out,
                                         unsigned int async_k,
                                         bool use_compression) {
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

template <typename DerivedCtx, typename T, typename I>
void Ctx<DerivedCtx, T, I>::unzip_rewrite(ot::DVector<T, I>& in,
                                          ot::DVector<T, I>& out,
                                          unsigned int async_k,
                                          bool use_compression) {
    if (!m_uiMesh->isActive()) return;

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
        // unzip on the host.
        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);

            m_uiMesh->readFromGhostBeginWrapper(
                m_mpi_ctx[i], in_ptr + v_begin * sz_per_dof_zip, batch_sz,
                use_compression);
        }

        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);

            m_uiMesh->readFromGhostEndWrapper(m_mpi_ctx[i],
                                              in_ptr + v_begin * sz_per_dof_zip,
                                              batch_sz, use_compression);

#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].start();
#endif

            m_uiMesh->unzip(in_ptr + v_begin * sz_per_dof_zip,
                            out_ptr + v_begin * sz_per_dof_uzip, batch_sz);

#ifdef __PROFILE_CTX__
            m_uiCtxpt[CTXPROFILE::UNZIP].stop();
#endif
        }

        // now that it's all done, we can gather up from the m_uiMesh unzip:
        for (unsigned int i = 0; i < async_k; i++) {
            const unsigned int v_begin  = ((i * dof) / async_k);
            const unsigned int v_end    = (((i + 1) * dof) / async_k);
            const unsigned int batch_sz = (v_end - v_begin);

            for (unsigned int j = 0; j < m_uiMesh->getMPICommSize(); j++) {
                m_uiTotalBytesSend[j] +=
                    m_uiMesh->getNodalSendCounts()[j] * batch_sz * sizeof(T);
                m_uiTotalBytesRecv[j] +=
                    m_uiMesh->getNodalRecvCounts()[j] * batch_sz * sizeof(T);
                // then the compress amounts, which is the *total* amount not
                // including batch syze
#ifdef DENDRO_ENABLE_GHOST_COMPRESSION
                m_uiTotalBytesSendCompress[j] +=
                    m_mpi_ctx[i].getSendCompressCounts()[j];
                m_uiTotalBytesRecvCompress[j] +=
                    m_mpi_ctx[i].getReceiveCompressCounts()[j];
#endif
            }
        }

        m_uiTotalBytesCounter += 1;

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
    ot::Mesh* newMesh      = remesh(grain_sz, ld_tol, sf_k);

    DendroIntL oldElements = m_uiMesh->getNumLocalMeshElements();
    DendroIntL newElements = newMesh->getNumLocalMeshElements();

    DendroIntL oldElements_g, newElements_g;

    par::Mpi_Reduce(&oldElements, &oldElements_g, 1, MPI_SUM, 0,
                    m_uiMesh->getMPIGlobalCommunicator());
    par::Mpi_Reduce(&newElements, &newElements_g, 1, MPI_SUM, 0,
                    newMesh->getMPIGlobalCommunicator());

    if (!(m_uiMesh->getMPIRankGlobal()))
        std::cout << "[Ctx]: step : " << m_uiTinfo._m_uiStep
                  << "\ttime : " << m_uiTinfo._m_uiT
                  << "\told mesh: " << oldElements_g
                  << "\tnew mesh:" << newElements_g << std::endl;

    this->grid_transfer(newMesh);

    std::swap(newMesh, m_uiMesh);
    delete newMesh;

#ifdef __CUDACC__
    device::MeshGPU*& dptr_mesh = this->get_meshgpu_device_ptr();
    device::MeshGPU* mesh_gpu   = this->get_meshgpu_host_handle();

    mesh_gpu->dealloc_mesh_on_device(dptr_mesh);
    dptr_mesh = mesh_gpu->alloc_mesh_on_device(m_uiMesh);
#endif

    m_uiIsETSSynced = false;

    return 0;
}

}  // end of namespace ts
