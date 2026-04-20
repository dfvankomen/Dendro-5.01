# GR utilities library for Dendro-based general relativity solvers.
#
# Provides common infrastructure shared across GR formulations:
#   - BH struct and coordinate extraction
#   - Profile/timer declarations
#   - TwoPunctures initial data solver

# header-only pieces (bh.h) don't need a source file,
# but grDataUtils and profile params do
set(DENDRO_GR_SRC
    # src/grDataUtils.cpp
    # src/grProfileParams.cpp
)

set(DENDRO_GR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# for now this is header-only + TwoPunctures
# uncomment the add_library when we have .cpp sources
# add_library(dendro_gr STATIC ${DENDRO_GR_SRC})
# target_include_directories(dendro_gr PUBLIC ${DENDRO_GR_INCLUDE_DIR})
# target_link_libraries(dendro_gr PUBLIC dendro5)

# TwoPunctures source files -- compiled as part of the solver, not as
# a separate library, because they call back into solver-specific code
# (grUtils.h for initDataFuncToPhysCoords). The headers and parameter
# declarations live here so solvers can include them.
#
# To use TwoPunctures in a solver, add the .cpp files from
# GR/TwoPunctures/ to your solver's source list.
set(DENDRO_GR_TWOPUNCTURES_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/TwoPunctures/TPCoordTransf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TwoPunctures/TPEquations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TwoPunctures/TPFuncAndJacobian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TwoPunctures/TPNewton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TwoPunctures/TPUtilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TwoPunctures/TwoPunctures.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TwoPunctures/TPID.cpp
    PARENT_SCOPE
)

# export include directory so solvers can find GR headers
set(DENDRO_GR_INCLUDE_DIR ${DENDRO_GR_INCLUDE_DIR} PARENT_SCOPE)
