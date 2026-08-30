# GR utilities library for Dendro-based general relativity solvers.
#
# Provides common infrastructure shared across GR formulations:
#   - BH struct and coordinate extraction
#   - Profile/timer declarations
#   - TwoPunctures initial data solver

set(DENDRO_GR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# INTERFACE rather than STATIC: everything here is header-only. TwoPunctures
# has .cpp files but they are deliberately compiled into the *solver* (see
# below), so this target owns no objects of its own. a STATIC library with an
# empty source list is a hard CMake error, which is why this used to sit
# commented out -- consumers reached into GR/include by path instead.
add_library(dendro_gr INTERFACE)
target_include_directories(dendro_gr INTERFACE ${DENDRO_GR_INCLUDE_DIR})
target_link_libraries(dendro_gr INTERFACE dendro5)

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
