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

option(DENDROLIB_ISOLATED_DIR "Only modify this if Dendro is not a library" OFF)

# we need a lot of other definitions if we are calling cmake directly for this folder.
if(DENDROLIB_ISOLATED_DIR)
  include("CMakeLists-NonLibraryDefs.txt")
endif()

# GR's important AEH Solver
add_subdirectory(lib/BHaHAHA)

message(STATUS "Configuring Dendro Library...")

set(DENDRO_SRC
    src/dendro.cpp src/point.cpp src/binUtils.cpp src/hcurvedata.cpp
    src/genPts_par.cpp src/TreeNode.cpp src/treenode2vtk.cpp src/parUtils.cpp
    src/mesh.cpp src/octUtils.cpp src/dendroIO.cpp src/block.cpp
    src/sc_flops.cpp src/profiler.cpp src/dendroProfileParams.cpp src/oda.cpp
    src/odaUtils.cpp src/sub_oda.cpp src/meshUtils.cpp src/waveletRefEl.cpp
    test/src/meshTestUtils.cpp src/aeh_bhahaha.cpp src/logger.cpp
)

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

include(FetchContent)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.14.1
    )
FetchContent_MakeAvailable(spdlog)

# ==========================
# DENDRO LIBRARY DEFINITION
# ==========================
add_library(dendro5 ${DENDRO_SRC} ${FEM_SRC} ${ODE_SRC} ${IO_SRC} ${ZLIB_SRC})

target_include_directories(dendro5 PUBLIC
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

target_link_libraries(dendro5 PUBLIC MPI::MPI_CXX GSL::gsl m BHaHAHA spdlog::spdlog)

if(WITH_BLAS_LAPACK)
  target_link_libraries(dendro5 PUBLIC ${LAPACK_LIBRARIES})
  # Note: Add LAPACK_INCLUDE_DIR if strictly needed, but LAPACK usually just needs link
endif()

if(BUILD_WITH_PETSC)
  target_include_directories(dendro5 PUBLIC ${PETSC_INCLUDES})
  target_link_libraries(dendro5 PUBLIC ${PETSC_LIBRARIES})
endif()

if(WITH_CUDA)
  target_include_directories(dendro5 PUBLIC ${CUDA_INCLUDE_DIRS})

  # Define the CUDA version of the lib
  set(CUDA_SRCS
      ${DENDRO_SRC} ${FEM_SRC} ${ODE_SRC} ${IO_SRC} ${ZLIB_SRC}
      GPU/include/derivs_cu.cuh GPU/src/refel_const.cu GPU/src/mesh_gpu.cu)

  cuda_add_library(dendro5_cuda ${CUDA_SRCS})

  # Inherit includes/links from main lib logic manually since it's a separate target
  target_include_directories(dendro5_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/GPU/include)
  # (Re-applying the main includes here is necessary for the separate cuda target)
  target_include_directories(dendro5_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/FEM/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ODE/include
    ${CMAKE_CURRENT_SOURCE_DIR}/IO/vtk/include
    ${CMAKE_CURRENT_SOURCE_DIR}/IO/zlib/inc
    ${CUDA_INCLUDE_DIRS}
  )
  target_link_libraries(dendro5_cuda ${CUDA_LIBRARIES} MPI::MPI_CXX)
endif()

# ==========================
# Examples and Tests!
# ==========================

function(dendro_add_example TARGET_NAME)
  add_executable(${TARGET_NAME} ${ARGN})
  # NOTE: linking dendro5 includes **all** flags and includes!
  target_link_libraries(${TARGET_NAME} PRIVATE dendro5)

  # We explicitly link MPI/m to be safe for the executable itself
  target_link_libraries(${TARGET_NAME} PRIVATE MPI::MPI_CXX m)
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

if(WITH_CUDA)
  # cuda GPU tests
  cuda_add_executable(run_meshgpu_tests ${CMAKE_CURRENT_SOURCE_DIR}/GPU/src/run_meshgpu_tests.cu)
  target_link_libraries(run_meshgpu_tests dendro5 dendro5_cuda)
  target_include_directories(run_meshgpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/GPU/include)
endif()
