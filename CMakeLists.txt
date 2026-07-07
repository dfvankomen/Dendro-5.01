# ##############################################################################
# CMAKE CONFIGURATION FILE FOR DENDRO DIRECTORY
#
# This file in particular contains all of the options that apply specificially
# to the Dendro directory as it is compiled. This includes the example solvers
# as well as the library itself.
#
# NOTE: there is a CMakeLists-NonLibraryDefs.txt file which contains more options.
# The options in there must be included in the root directory of your project
# that you are using this library with. This will configure everything for building,
# but the options must be propagated to your own solvers and projects.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.15)
project(Dendro-5.0-Library C CXX Fortran)


# ==========================
# CPU ARCHITECTURE
# ==========================
# when built standalone, set a default arch target so the compiler can
# actually vectorize. when included as a subdirectory, the parent project
# is expected to set arch flags itself (e.g. via add_compile_options).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(DENDRO_CPU_ARCH "native" CACHE STRING "Target CPU architecture (native, x86-64-v3, znver3, etc.)")
  set(DENDRO_CPU_ARCH_STRINGS
      "native"
      "nosimd"
      "znver1" "znver2" "znver3" "znver4"
      "haswell" "broadwell" "skylake-avx512" "cascadelake" "icelake-server"
      "generic_avx2" "x86-64-v3"
  )
  set_property(CACHE DENDRO_CPU_ARCH PROPERTY STRINGS ${DENDRO_CPU_ARCH_STRINGS})

  if(DENDRO_CPU_ARCH STREQUAL "nosimd")
    # Pure-scalar profiling baseline: x86-64 v1 (SSE2 is mandated by the ABI, but
    # disable AVX/AVX2/AVX-512 and auto-vectorization). This controls the COMPILED
    # production explicit derivative kernels. NOTE: the LIBXSMM matrix-form path JITs
    # at runtime from CPUID, not these flags — force it with LIBXSMM_TARGET=sse.
    add_compile_options(-march=x86-64 -mno-avx -mno-avx2 -mno-avx512f -fno-tree-vectorize)
  elseif(DENDRO_CPU_ARCH STREQUAL "generic_avx2")
    add_compile_options(-mavx2 -mfma)
  elseif(DENDRO_CPU_ARCH IN_LIST DENDRO_CPU_ARCH_STRINGS)
    add_compile_options(-march=${DENDRO_CPU_ARCH})
  else()
    message(FATAL_ERROR "Unknown DENDRO_CPU_ARCH: ${DENDRO_CPU_ARCH}")
  endif()

  message(STATUS "Dendro standalone build — target arch: ${DENDRO_CPU_ARCH}")
endif()


# ==========================
# DENDRO LIBRARY OPTIONS
# ==========================
option(USE_64BIT_INDICES "Use 64-Bit indices (Reverts to 32-bit if OFF)" ON)
option(HILBERT_ORDERING "Use Hilbert ordering instead of Morton" ON)
option(RUN_WEAK_SCALING "Run Entire Weak Scaling" ON)
option(ALLTOALLV_FIX "Use K-way all to all v" OFF)
option(SPLITTER_SELECTION_FIX "Turn on Splitter Selection fix" OFF)
option(DIM_2 "Use the two dimensional sorting" OFF)
option(PROFILE_TREE_SORT "Profile the tree sort code" OFF)
option(WITH_BLAS_LAPACK "Build using BLAS and LAPACK" ON)
option(DENDRO_VTK_BINARY "Write VTK/VTU files in binary mode" ON)
option(DENDRO_VTK_ZLIB_COMPRES "Compress VTK/VTU (requires Binary)" OFF)
option(ALLTOALL_SPARSE "Use isend/irecv for ghost exchange" OFF)
option(ENABLE_DENDRO_PROFILE_COUNTERS "Enable dendro internal profile counters" OFF)
option(RK_SOLVER_OVERLAP_COMM_AND_COMP "Enable RK solver comm/comp overlap" ON)
option(WITH_CUDA "Build dendro with CUDA" OFF)
option(BUILD_WITH_PETSC "Build dendro with PETSc" OFF)
option(USE_FD_INTERP_FOR_UNZIP "Use FD style interpolation for unzip" OFF)
option(MPIX_CUDA_AWARE_SUPPORT "Allow MPI to be aware of CUDA support" OFF)
# Default ON: matrix-form/compact derivs read the full block including the
# never-filled boundary-ghost padding, so un-zeroed alloc -> NaN. Hybrid-OMP
# zeros via first-touch; pure-MPI builds rely on this. Turn OFF only for perf A/B.
option(DVEC_ZERO_ALLOC "Initialize dvec memory to zero" ON)
# ---------------------------------------------------------------------------
# Unzip/zip speedup toggles
#
# DENDRO_UNZIP_SPEEDUP is the single master switch for the MPI-safe unzip/zip
# speedups: the integer-index same-level scatter fast path and the SIMD
# wavelet tensor kernels. Flip it to swap between the baseline and the
# optimized path in one place (e.g. -DDENDRO_UNZIP_SPEEDUP=ON for benchmark
# builds).
#
# The two sub-flags below stay individually overridable so each speedup can be
# exercised on its own during experiments (e.g. scatter fast path ON, SIMD
# OFF). DENDRO_UNZIP_OMP (the reserved OpenMP / hybrid hook) forces BOTH
# speedups on, since the hybrid block-parallel path relies on them to run fast.
# ---------------------------------------------------------------------------
option(DENDRO_UNZIP_SPEEDUP
       "Master toggle for the MPI-safe unzip/zip speedups (integer-index scatter fast path + SIMD wavelet tensor kernels)"
       OFF)
option(DENDRO_UNZIP_SCATTER_FAST
       "Use the integer-index same-level scatter fast path in Mesh::unzip_scatter (Phase 1 optimization)"
       OFF)
option(DENDRO_UNZIP_OMP
       "Enable OpenMP block-parallel unzip path (reserved hook; implementation lives on a separate hybrid branch). Implies the unzip/zip speedups."
       OFF)
option(DENDRO_TENSOR_SIMD
       "Enable AVX2/AVX-512 SIMD specializations for the wavelet tensor kernels in FEM/src/tensor.cpp (M=7 for eO=6)"
       OFF)
# Opt-in MPI comm/compute overlap in the unzip: post the ghost exchange, unzip
# the interior (ghost-independent) blocks while messages are in flight, then
# wait and unzip the boundary blocks. Default OFF = unchanged behavior. Implies
# the OpenMP unzip path. Payoff is multi-node (intra-node MPI is shared-memory).
option(DENDRO_UNZIP_OVERLAP
       "Overlap ghost-exchange comm with interior-block unzip in Ctx::unzip (implies DENDRO_UNZIP_OMP; default OFF = unchanged)"
       OFF)
# Project-wide switch for the hybrid OpenMP/MPI path brought in from the
# derivatives workstream. Default OFF keeps the MPI-first behavior. When ON it
# enables the OpenMP threading the hybrid path relies on (e.g. TwoPunctures
# initial-data solve via TP_OMP_THREADS). Propagated through dendro_config so
# downstream solvers that compile the hybrid sources pick it up too.
option(DENDRO_HYBRID_OMP
       "Enable the hybrid OpenMP/MPI path (OpenMP threading on top of MPI; default OFF = MPI-first)"
       OFF)

# Configuration Variables
set(KWAY 128 CACHE STRING "K parameter for alltoallv_kway")
set(OCT2BLK_COARSEST_LEV 0 CACHE STRING "Coarsest Level")
set(NUM_NPES_THRESHOLD 2 CACHE STRING "NPES Threshold")



# ==========================
# DEPENDENCIES
# ==========================
include(FetchContent)

# OpenMP
find_package(OpenMP REQUIRED)

# MPI
find_package(MPI REQUIRED)

# GSL
find_package(GSL REQUIRED)

# spdlog allows for logging!
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
)
FetchContent_MakeAvailable(spdlog)

# libxsmm for optimized small matrix multiplications
option(USE_LOCAL_XSMM "Use a local copy of libxsmm's source instead of fetching at build" OFF)

if (USE_LOCAL_XSMM)
  set(LOCAL_XSMM_PATH "" CACHE PATH "Path to local libxsmm repository")
  if(NOT EXISTS "${LOCAL_XSMM_PATH}/CMakeLists.txt")
    message(FATAL_ERROR "LOCAL_XSMM_PATH is either not set or is pointing to an invalid libxsmm source")
  endif()
endif()

if (USE_LOCAL_XSMM)
  FetchContent_Declare(
    xsmm
    SOURCE_DIR ${LOCAL_XSMM_PATH}
  )
else()
  FetchContent_Declare(
    xsmm
    GIT_REPOSITORY https://github.com/libxsmm/libxsmm.git
    GIT_TAG main
    GIT_SHALLOW TRUE
  )
endif()

FetchContent_GetProperties(xsmm)
if(NOT xsmm_POPULATED)
  FetchContent_Populate(xsmm)
endif()
set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
list(REMOVE_ITEM _GLOB_XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)
add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${XSMM_INCLUDE_DIRS})
target_compile_definitions(xsmm PUBLIC LIBXSMM_DEFAULT_CONFIG)
target_compile_definitions(xsmm PRIVATE __BLAS=0)
target_compile_definitions(xsmm PRIVATE _GNU_SOURCE)
target_compile_definitions(xsmm PRIVATE STATIC=0)
target_compile_options(xsmm PRIVATE "-fPIC")

# make sure we have BhaHAHA included
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib/BHaHAHA")
  add_subdirectory(lib/BHaHAHA)
endif()


# GR utilities (BH struct, TwoPunctures, etc.)
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/GR")
  add_subdirectory(GR)
endif()

# ==========================
# SOURCE FILES
# ==========================
message(STATUS "Configuring Dendro Library...")

set(DENDRO_SRC
    src/dendro.cpp src/point.cpp src/binUtils.cpp src/hcurvedata.cpp
    src/genPts_par.cpp src/TreeNode.cpp src/treenode2vtk.cpp src/parUtils.cpp
    src/mesh.cpp src/octUtils.cpp src/dendroIO.cpp src/block.cpp
    src/sc_flops.cpp src/profiler.cpp src/dendroProfileParams.cpp src/oda.cpp
    src/odaUtils.cpp src/sub_oda.cpp src/meshUtils.cpp src/waveletRefEl.cpp
    test/src/meshTestUtils.cpp src/aeh_bhahaha.cpp src/bh_history.cpp
    src/logger.cpp
    src/derivatives.cpp
    src/derivatives/derivs_explicit.cpp
    src/derivatives/adv_derivs_explicit.cpp
    src/derivatives/derivs_compact.cpp
    src/derivatives/derivs_matrixonly.cpp
    src/derivatives/derivs_banded.cpp
    src/derivatives/derivs_utils.cpp
    src/derivatives/impl_explicitmatrix.cpp
    src/derivatives/impl_byuderivs.cpp
    src/derivatives/impl_kimderivs.cpp
    src/derivatives/impl_jonathantyler.cpp
    src/derivatives/impl_bradylivescu.cpp
    src/derivatives/impl_boris.cpp
    src/derivatives/impl_hybrid_approaches.cpp
    src/derivatives/filt_kodiss_explicit.cpp
    src/derivatives/nova_derivs.cpp
)

# NOTE: generated derivative scheme files (src/derivatives/generated/*.cpp) are
# NOT standalone compilation units -- they're #include'd into impl_*.cpp files.
# Do NOT glob them as source files.

set(FEM_SRC
    FEM/examples/src/laplace.cpp FEM/src/basis.cpp FEM/src/refel.cpp
    FEM/src/tensor.cpp FEM/src/workspace.cpp)

set(ODE_SRC
    ODE/src/rk.cpp ODE/src/subSM.cpp ODE/src/rkMaxwell.cpp
    ODE/src/rkTransport.cpp ODE/src/enutsOp.cpp)

set(IO_SRC IO/vtk/src/oct2vtk.cpp IO/vtk/src/checkPoint.cpp)

set(ZLIB_SRC
    IO/zlib/src/adler32.c IO/zlib/src/compress.c IO/zlib/src/crc32.c
    IO/zlib/src/deflate.c IO/zlib/src/gzclose.c IO/zlib/src/gzlib.c
    IO/zlib/src/gzread.c IO/zlib/src/gzwrite.c IO/zlib/src/infback.c
    IO/zlib/src/inffast.c IO/zlib/src/inflate.c IO/zlib/src/inftrees.c
    IO/zlib/src/trees.c IO/zlib/src/uncompr.c IO/zlib/src/zutil.c)

if(WITH_CUDA)
  list(APPEND DENDRO_SRC src/cudaUtils.cpp src/oda.cpp FEM/examples/src/heatMat.cpp)
endif()

# ==========================
# DENDRO LIBRARY DEFINITION
# ==========================
add_library(dendro5 ${DENDRO_SRC} ${FEM_SRC} ${ODE_SRC} ${IO_SRC} ${ZLIB_SRC})

# create an interface library for both dendro5 and dendro5_cuda
add_library(dendro_config INTERFACE)

# add definitions to COMPILE for dendro5
macro(dendro_add_def OPTION_NAME DEF_NAME)
  if(${OPTION_NAME})
    target_compile_definitions(dendro_config INTERFACE ${DEF_NAME})
  endif()
endmacro()

target_compile_definitions(dendro_config INTERFACE -DMATVEC_PROFILE)
target_compile_definitions(dendro_config INTERFACE -DOCT2BLK_COARSEST_LEV=${OCT2BLK_COARSEST_LEV})

dendro_add_def(USE_64BIT_INDICES USE_64BIT_INDICES)
dendro_add_def(HILBERT_ORDERING HILBERT_ORDERING)
dendro_add_def(RUN_WEAK_SCALING RUN_WEAK_SCALING)
dendro_add_def(DIM_2 DIM_2)
dendro_add_def(PROFILE_TREE_SORT PROFILE_TREE_SORT)
dendro_add_def(ALLTOALL_SPARSE ALLTOALL_SPARSE)
dendro_add_def(RK_SOLVER_OVERLAP_COMM_AND_COMP RK_SOLVER_OVERLAP_COMM_AND_COMP)
dendro_add_def(USE_FD_INTERP_FOR_UNZIP USE_FD_INTERP_FOR_UNZIP)
dendro_add_def(DVEC_ZERO_ALLOC DVEC_ZERO_ALLOC)

# Resolve the effective unzip/zip + threading flags. DENDRO_HYBRID_OMP is the
# umbrella for all intra-rank OpenMP threading: it implies the block-parallel
# unzip path (DENDRO_UNZIP_OMP). The master switch and the (effective) unzip-OMP
# hook both force the scatter fast path + SIMD tensor kernels on; otherwise each
# sub-flag is honored as set.
set(DENDRO_UNZIP_SCATTER_FAST_EFFECTIVE ${DENDRO_UNZIP_SCATTER_FAST})
set(DENDRO_TENSOR_SIMD_EFFECTIVE ${DENDRO_TENSOR_SIMD})
set(DENDRO_UNZIP_SPEEDUP_EFFECTIVE ${DENDRO_UNZIP_SPEEDUP})
set(DENDRO_UNZIP_OMP_EFFECTIVE ${DENDRO_UNZIP_OMP})
if(DENDRO_HYBRID_OMP)
  set(DENDRO_UNZIP_OMP_EFFECTIVE ON)
endif()
# The comm/compute overlap needs the block-parallel (OMP) unzip path.
if(DENDRO_UNZIP_OVERLAP)
  set(DENDRO_UNZIP_OMP_EFFECTIVE ON)
endif()
if(DENDRO_UNZIP_SPEEDUP OR DENDRO_UNZIP_OMP_EFFECTIVE)
  set(DENDRO_UNZIP_SCATTER_FAST_EFFECTIVE ON)
  set(DENDRO_TENSOR_SIMD_EFFECTIVE ON)
  set(DENDRO_UNZIP_SPEEDUP_EFFECTIVE ON)
endif()
message(STATUS
        "Dendro unzip/zip speedups: master=${DENDRO_UNZIP_SPEEDUP_EFFECTIVE} "
        "scatter_fast=${DENDRO_UNZIP_SCATTER_FAST_EFFECTIVE} "
        "tensor_simd=${DENDRO_TENSOR_SIMD_EFFECTIVE} omp=${DENDRO_UNZIP_OMP_EFFECTIVE}")
message(STATUS "Dendro hybrid OpenMP/MPI path: ${DENDRO_HYBRID_OMP} (implies unzip OMP)")

# DENDRO_UNZIP_SPEEDUP is also emitted as a single umbrella compile def so
# downstream code can `#ifdef DENDRO_UNZIP_SPEEDUP` instead of testing both
# sub-macros. It is defined whenever the bundled speedups are active (master
# switch on, or forced on by the OpenMP hooks).
dendro_add_def(DENDRO_UNZIP_SPEEDUP_EFFECTIVE DENDRO_UNZIP_SPEEDUP)
dendro_add_def(DENDRO_UNZIP_SCATTER_FAST_EFFECTIVE DENDRO_UNZIP_SCATTER_FAST)
dendro_add_def(DENDRO_UNZIP_OMP_EFFECTIVE DENDRO_UNZIP_OMP)
dendro_add_def(DENDRO_TENSOR_SIMD_EFFECTIVE DENDRO_TENSOR_SIMD)
dendro_add_def(DENDRO_HYBRID_OMP DENDRO_HYBRID_OMP)
dendro_add_def(DENDRO_UNZIP_OVERLAP DENDRO_UNZIP_OVERLAP)

if(ALLTOALLV_FIX)
  target_compile_definitions(dendro_config INTERFACE ALLTOALLV_FIX KWAY=${KWAY})
endif()

if(SPLITTER_SELECTION_FIX)
  target_compile_definitions(dendro_config INTERFACE SPLITTER_SELECTION_FIX NUM_NPES_THRESHOLD=${NUM_NPES_THRESHOLD})
endif()

if(ENABLE_DENDRO_PROFILE_COUNTERS)
  target_compile_definitions(dendro_config INTERFACE ENABLE_DENDRO_PROFILE_COUNTERS __PROFILE_CTX__ __PROFILE_ETS__)
endif()

# VTK Logic
if(DENDRO_VTK_BINARY)
  target_compile_definitions(dendro_config INTERFACE DENDRO_VTU_BINARY)
  if(DENDRO_VTK_ZLIB_COMPRES)
    target_compile_definitions(dendro_config INTERFACE DENDRO_VTU_ZLIB)
  endif()
else()
  target_compile_definitions(dendro_config INTERFACE DENDRO_VTU_ASCII)
endif()

target_include_directories(dendro_config INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/include
    ${CMAKE_CURRENT_SOURCE_DIR}/FEM/include
    ${CMAKE_CURRENT_SOURCE_DIR}/FEM/examples/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ODE/include
    ${CMAKE_CURRENT_SOURCE_DIR}/LinAlg/include
    ${CMAKE_CURRENT_SOURCE_DIR}/IO/include
    ${CMAKE_CURRENT_SOURCE_DIR}/IO/vtk/include
    ${CMAKE_CURRENT_SOURCE_DIR}/IO/zlib/inc
)

target_link_libraries(dendro5 PUBLIC dendro_config MPI::MPI_CXX OpenMP::OpenMP_CXX GSL::gsl spdlog::spdlog BHaHAHA xsmm m)

if(WITH_BLAS_LAPACK)
  target_compile_definitions(dendro5 PUBLIC WITH_BLAS_LAPACK)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_link_libraries(dendro5 PUBLIC BLAS::BLAS LAPACK::LAPACK)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_compile_options(dendro5 PUBLIC "-mkl")
    target_link_options(dendro5 PUBLIC "-mkl")
  endif()
endif()

if(BUILD_WITH_PETSC)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
  find_package(PETSc REQUIRED)
  target_compile_definitions(dendro5 PUBLIC BUILD_WITH_PETSC)
  target_include_directories(dendro5 PUBLIC ${PETSC_INCLUDES})
  target_link_libraries(dendro5 PUBLIC ${PETSC_LIBRARIES})
endif()

if(WITH_CUDA)
  enable_language(CUDA)

  # supress the declared but never referenced warnings on nvcc
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=177")

  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    # if we'r in full debug mode, then we want all debug symbols (includes lineinfo)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
  elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    # deb-info means we can keep line info for compute-sanatizer
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  endif()

  # define the target GPU architectures (e.g., 70=Volta, 75=Turing, 80=Ampere, 89=Ada, 90=Hopper)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 CACHE STRING "CUDA Architectures" FORCE)

  # turn on relocatable device code!
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  # cuda toolkit makes sure regular C++ files get access to GPU-based code
  find_package(CUDAToolkit REQUIRED)

  target_include_directories(dendro_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/GPU/include)
  set(OCT2BLK_COARSEST_LEV 31 CACHE STRING "Coarsest Level for CUDA" FORCE)
  target_compile_definitions(dendro_config INTERFACE WITH_CUDA OCT2BLK_COARSEST_LEV=31)

  if(MPIX_CUDA_AWARE_SUPPORT)
    target_compile_definitions(dendro_config PUBLIC MPIX_CUDA_AWARE_SUPPORT)
  endif()

  target_sources(dendro5 PRIVATE
        GPU/include/derivs_cu.cuh
        GPU/src/mesh_gpu.cu
        GPU/src/refel_const.cu
  )

  set_target_properties(dendro5 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(dendro5 PUBLIC CUDA::cudart)
endif()

# ==========================
# Examples and Tests!
# ==========================

option(BUILD_DENDRO_EXAMPLES "Build Dendro Examples" OFF)

if(BUILD_DENDRO_EXAMPLES)
  function(dendro_add_example TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    # NOTE: linking dendro5 includes **all** flags and includes!
    target_link_libraries(${TARGET_NAME} PRIVATE dendro5 MPI::MPI_CXX m)
  endfunction()

  dendro_add_example(genHtables Hilbert/src/genHTables.cpp include/radix.h)
  dendro_add_example(genSFCMatVecTables Hilbert/src/sfcMatvecTable.cpp)

  dendro_add_example(ptsSort examples/src/ptsSort.cpp)
  dendro_add_example(testOctUtils examples/src/testOCtUtils.cpp)
  dendro_add_example(tstTreeSearch examples/src/tstTreeSearch.cpp)
  dendro_add_example(octUtilBenchmark examples/src/octUtilsBenchmark.cpp)
  dendro_add_example(bucketBench examples/src/bucketBench.cpp)
  dendro_add_example(mfreeBench examples/src/mfreeTest.cpp)
  dendro_add_example(meshBench examples/src/meshBenchmark.cpp)
  dendro_add_example(meshE2NCheck examples/src/meshE2NCheck.cpp)
  dendro_add_example(run_all_tests test/src/run_all_tests.cpp)
  dendro_add_example(benchUtils test/src/benchUtils.cpp)
  dendro_add_example(benchUnzipSameLevel test/src/benchUnzipSameLevel.cpp)
  dendro_add_example(benchUnzipGather test/src/benchUnzipGather.cpp)
  dendro_add_example(testStabilityEvolution test/src/testStabilityEvolution.cpp)
  dendro_add_example(testIsReMeshUnzip test/src/testIsReMeshUnzip.cpp)

  dendro_add_example(testLTSBlkVec ODE/examples/testLTSBlkVec.cpp)
  dendro_add_example(dgVecTest FEM/examples/src/dgVecTest.cpp)
  dendro_add_example(testInterp FEM/examples/src/testInterp.cpp)
  dendro_add_example(meshLoop examples/src/meshLooping.cpp)
  dendro_add_example(refEl FEM/examples/src/refelExample.cpp)
  dendro_add_example(refElOctree FEM/examples/src/refelOctreeExample.cpp)
  dendro_add_example(transportEq ODE/examples/transportEq.cpp)
  dendro_add_example(laplaceEq FEM/examples/src/laplace.cpp)
  dendro_add_example(heatEq FEM/examples/src/heatMat.cpp FEM/examples/src/heatVec.cpp FEM/examples/src/heatEq.cpp)
  dendro_add_example(subDAExample FEM/examples/src/subdaExample.cpp)
  dendro_add_example(sfcLaplaceEq FEM/examples/src/sfcLaplace.cpp)

  # --- Derivative tests (require toml11 for config) ---
  # fetch toml11 if the parent project hasn't already provided it
  if(NOT TARGET toml11::toml11)
    FetchContent_Declare(
        toml11
        GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
        GIT_TAG        v4.4.0
    )
    FetchContent_MakeAvailable(toml11)
  endif()
  function(dendro_add_deriv_test TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE dendro5 MPI::MPI_CXX toml11::toml11 m)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/derivatives)
  endfunction()

  dendro_add_deriv_test(testCompactDerivs test/src/testCompactDerivs.cpp)
  dendro_add_deriv_test(testMatrixMul test/src/testMatMul.cpp)
  dendro_add_deriv_test(testNOVA test/src/testNova.cpp)
  dendro_add_deriv_test(testKODiss test/src/testKODiss.cpp)
  dendro_add_deriv_test(testDerivOnGrid test/src/derivativeConvergence.cpp test/src/derivativeCtx.cpp)
  dendro_add_deriv_test(buildGlobalMat test/src/globalMatrixBuilding.cpp test/src/derivativeCtx.cpp)
  dendro_add_deriv_test(testLibXSMMvBLAS test/src/testLibXSMMvBLAS.cpp)
  dendro_add_deriv_test(testAllDerivs test/src/testAllDerivs.cpp)
  dendro_add_deriv_test(testDerivConvergence test/src/testDerivConvergence.cpp)
  dendro_add_deriv_test(testBandedVsXSMM test/src/testBandedVsXSMM.cpp)
  dendro_add_deriv_test(testFusedMixed test/src/testFusedMixed.cpp)
  dendro_add_deriv_test(testRhsWorkflow test/src/testRhsWorkflow.cpp)
  dendro_add_deriv_test(testTimeIntegratorOrder test/src/testTimeIntegratorOrder.cpp)

  if(WITH_CUDA)
    # cuda GPU tests
    add_executable(run_meshgpu_tests ${CMAKE_CURRENT_SOURCE_DIR}/GPU/src/run_meshgpu_tests.cu)
    target_link_libraries(run_meshgpu_tests PRIVATE dendro5) # dendro5_cuda)
    target_include_directories(run_meshgpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/GPU/include)
    # NOTE: CUDA_SEPARABLE_COMPILATION is a necessity for the refel stuff (keeps us from having to redefine it)
    set_target_properties(run_meshgpu_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()
endif()
