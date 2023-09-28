message("-- Now configuring libxsmm")

include(FetchContent)
FetchContent_Declare(
  xsmm
  URL https://github.com/libxsmm/libxsmm/archive/refs/tags/1.17.tar.gz
  URL_HASH SHA256=8b642127880e92e8a75400125307724635ecdf4020ca4481e5efe7640451bb92
)
FetchContent_GetProperties(xsmm)
if(NOT xsmm_POPULATED)
  FetchContent_Populate(xsmm)
endif()

# this is required because the release above doesn't have the fix for pthread_yield being depreciated...
add_definitions(-D_GNU_SOURCE)
add_definitions(-DSTATIC=0)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -llapacke -llapack -lblas -lgfortran -lquadmath")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -llapacke -llapack -lblas -lgfortran -lquadmath")

set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
list(REMOVE_ITEM _GLOB_XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)

add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${XSMM_INCLUDE_DIRS})
target_compile_definitions(xsmm PUBLIC
  LIBXSMM_DEFAULT_CONFIG
)
# target_compile_definitions(xsmm PRIVATE
#   BLAS=0
# )

message(${LIBXSMMROOT})

# target_link_libraries(xsmm PRIVATE Threads::Threads)
target_link_libraries(xsmm PRIVATE blas)
# target_link_libraries()