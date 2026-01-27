cmake_minimum_required(VERSION 3.12)
project(BHaHAHAlib LANGUAGES C)

# set the default compile flags for this library, based on makefile
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
# extensions are required, as that's what the gnu99 line does
set(CMAKE_C_EXTENSIONS ON)

# base compiler flags

set(COMMON_COMPILE_FLAGS "-Wall -fPIC")
# NOTE: we originally had -g, which adds debug symbols to release, we'll ignore it

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_COMPILE_FLAGS}")

# check for openmp, like the makefile wants
find_package(OpenMP)
if(OpenMP_C_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# sources for the BHaHAHAlib, thanks to OBJ_FILES
set(BHAHAHA_SOURCES
  bah_apply_bcs_inner_only.c
  bah_apply_bcs_r_maxmin_partial_r_hDD_upwinding.c
  bah_bcstruct_set_up.c
  bah_cfl_limited_timestep_based_on_h_equals_r.c
  bah_commondata_struct_set_to_default.c
  bah_diagnostics.c
  bah_diagnostics_area_centroid_and_Theta_norms.c
  bah_diagnostics_file_output.c
  bah_diagnostics_integration_weights.c
  bah_diagnostics_min_max_mean_radii_wrt_centroid.c
  bah_diagnostics_proper_circumferences.c
  bah_error_message.c
  bah_find_horizon.c
  bah_hDD_dD_and_W_dD_in_interp_src_grid_interior.c
  bah_initial_data.c
  bah_interpolation_1d_radial_spokes_on_3d_src_grid.c
  bah_interpolation_2d_external_input_to_interp_src_grid.c
  bah_interpolation_2d_general__uniform_src_grid.c
  bah_KO_apply.c
  bah_MoL_free_memory_non_y_n_gfs.c
  bah_MoL_free_memory_y_n_gfs.c
  bah_MoL_malloc_non_y_n_gfs.c
  bah_MoL_malloc_y_n_gfs.c
  bah_MoL_step_forward_in_time.c
  bah_numgrid__evol_set_up.c
  bah_numgrid__external_input_set_up.c
  bah_numgrid__interp_src_set_up.c
  bah_over_relaxation.c
  bah_params_struct_set_to_default.c
  bah_poisoning_check_inputs.c
  bah_poisoning_set_inputs.c
  bah_quadratic_extrapolation.c
  bah_radial_grid_cell_centered_set_up.c
  bah_rfm_precompute_defines.c
  bah_rfm_precompute_free.c
  bah_rfm_precompute_malloc.c
  bah_rhs_eval.c
  bah_xx_to_Cart.c
  bah_xyz_center_r_minmax.c
)

# create the library
add_library(BHaHAHA STATIC ${BHAHAHA_SOURCES})

# link required library (math)
target_link_libraries(BHaHAHA PRIVATE m)

# make sure the current directory is set as an include directory!
target_include_directories(BHaHAHA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# TODO: installation?

