#------------------------------------------------------------------------------#
# Copyright (c) 2014 Los Alamos National Security, LLC
# All rights reserved.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.27.2)
cmake_policy(SET CMP0074 NEW)

#~----------------------------------------------------------------------------~#
# VPIC project configuration
#~----------------------------------------------------------------------------~#

project(vpic CXX)

#------------------------------------------------------------------------------#
# If a C++17 compiler is available, then set the appropriate flags
# Note: As of February 2021, VPIC requires only C++11, but requiring C++14
# makes building Kokkos easier.  If you are adventurous and using your own
# Kokkos build, feel free to try a lower standard.
# Fixing the this capture requires C++17, but if you don't mind the warnings,
# feel free to experiment with lower standards.
#------------------------------------------------------------------------------#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#------------------------------------------------------------------------------#
# MPI detection
#------------------------------------------------------------------------------#

# TODO: it seems like Cmake version 3.12 (and likely others) have a bug/change
# where it has a hard time detecting MPI. Cmake version. 3.9.0 works great
# though..
find_package(MPI REQUIRED)
list(APPEND MPI_CXX_COMPILE_FLAGS -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX )
include_directories(${MPI_C_INCLUDE_PATH})
include_directories(${MPI_CXX_INCLUDE_PATH})

#------------------------------------------------------------------------------#
# Add build options
#------------------------------------------------------------------------------#

option(ENABLE_INTEGRATED_TESTS "enable integrated tests" OFF)

option(ENABLE_UNIT_TESTS "enable unit tests" OFF)

option(USE_V4_ALTIVEC "Enable V4 Altivec" OFF)

option(USE_V4_PORTABLE "Enable V4 Portable" OFF)

option(USE_V4_SSE "Enable V4 SSE" OFF)

option(ENABLE_OPENSSL "Enable OpenSSL support for checksums" OFF)

option(ENABLE_KOKKOS_OPENMP "Enable Kokkos OpenMP" ON)

option(ENABLE_KOKKOS_CUDA "Enable Kokkos CUDA" OFF)

option(BUILD_INTERNAL_KOKKOS "Have VPIC build it's own Kokkos rather than look in $CMAKE_PREFIX_PATH" OFF)

option(VPIC_DUMP_ENERGIES "If the code should dump energies everytimestep for debugging" OFF)

option(VPIC_ENABLE_AUTO_TUNING "Enable automatic tuning based on architecture" OFF)

option(VPIC_ENABLE_HIERARCHICAL "Enable hierarchical parallelism" OFF)

option(VPIC_ENABLE_TEAM_REDUCTION "Enable team reduction for current accumulation" OFF)

option(VPIC_ENABLE_VECTORIZATION "Enable improved vectorization for CPUs" OFF)

option(VPIC_ENABLE_ACCUMULATOR "Enable explicit accumulators for better performnace on CPUs" OFF)

option(VPIC_ENABLE_HDF5 "Enable HDF5 for use during IO. VPIC does not help you install HDF5" OFF)

option(VPIC_ENABLE_HDF5_ASYNC "Enable asynchronous support for HDF5 (EXPERIMENTAL!!!)" OFF)

option(VPIC_ENABLE_PARTICLE_ANNOTATIONS "Enable particle annotations for additional per particle values" OFF)

option(VPIC_ENABLE_TRACER_PARTICLES "Enable tracer particles" OFF)

option(VPIC_ENABLE_VARIABLE_CHARGE "Enable variable charge for particles" OFF)

option(VPIC_ENABLE_EXTERNAL_FORCE "Enable external force for particles" OFF)

option(VPIC_ENABLE_GPU_AWARE_MPI "Enable GPU-aware MPI. Allows GPUs to communicate with each other directly rather than going through the CPU. If VPIC_ENABLE_GPU_AWARE_MPI is enabled but not supported in MPI, the code may crash." OFF)

option(VPIC_ENABLE_HALO_EXCHANGE "Enable halo exchange code path for remote communication. Temporary option until halo exchange code is accurately tested." ON)

option(VPIC_ENABLE_LEGACY_DATA_STRUCTURES "Enable use of legacy data structures for compatibility" ON)

option(HYB_USE_SEPARATE_PE "Use separate electron energy equation instead of EoS." OFF)

option(VPIC_SHAPE_QS "Use quadratic-sum particle shape" OFF)

option(HYB_USE_STATIC_E "Use electrostatic hybrid field solver." OFF)

option(HYB_USE_RADIATION "Include user radiation transport." OFF)

add_definitions(-DUSE_KOKKOS)
set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DUSE_KOKKOS") # Set it here for ./deck/ files

#------------------------------------------------------------------------------#
# Create include and link aggregates
#
# NOTE: These must be set before creating the compile scripts below.
#------------------------------------------------------------------------------#

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set(VPIC_CPPFLAGS)
if(MPI_CPPFLAGS)
  string(REPLACE ";" " " string_cppflags "${MPI_CPPFLAGS}")
  set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} ${string_cppflags}")
endif(MPI_CPPFLAGS)

# Append extra implicit includes
string(REPLACE ";" " -I" string_cppflags "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")
set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -I${string_cppflags}")
#message(${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
#message(${VPIC_CPPFLAGS})


string(REPLACE ";" " -I" string_includes "${MPI_CXX_INCLUDE_PATH}")
if(NOT ${string_includes} STREQUAL "")
  set(VPIC_CXX_FLAGS "-I${string_includes} ${MPI_CXX_LINK_FLAGS}")
endif(NOT ${string_includes} STREQUAL "")
set(VPIC_CXX_FLAGS "${CMAKE_CXX17_STANDARD_COMPILE_OPTION} ${VPIC_CXX_FLAGS}")

# Add Debug flags to VPIC_CXX_FLAGS
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

# Add RelWithDebInfo flags to VPIC_CXX_FLAGS
if("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")

# Add Release flags to VPIC_CXX_FLAGS
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")

string(REPLACE ";" " " string_libraries "${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES}")
set(VPIC_CXX_LIBRARIES "${string_libraries}")

#------------------------------------------------------------------------------#
# OpenMP
#------------------------------------------------------------------------------#

find_package(OpenMP)

# FIXME: Sometimes OpenMP is not found fully, when it really should be. Is this
# a CMAKE issue?
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(VPIC_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${VPIC_CXX_FLAGS}")
  #set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
else()
  message("OpenMP not found...")
endif()

#----------------------------------------------------------------------------#
# HDF5 Configuration
#----------------------------------------------------------------------------#

if(VPIC_ENABLE_HDF5)
  enable_language(C)
  set(HDF5_PREFER_PARALLEL ON)

  # Find HDF5 package
  find_package(HDF5 COMPONENTS C HL REQUIRED)

  add_definitions(-DVPIC_ENABLE_HDF5)
  include_directories(${HDF5_INCLUDE_DIRS})

  # -----------------------------------------------------------------------------
  # RESOLVE HDF5 LIBRARIES (h5pcc Interrogation + CMake Fallback)
  # -----------------------------------------------------------------------------
  set(HDF5_VPIC_LIBS "")  # This will hold actual library objects for target_link_libraries

  find_program(H5PCC_EXECUTABLE h5pcc)
  if(H5PCC_EXECUTABLE)
      execute_process(
          COMMAND ${H5PCC_EXECUTABLE} -show
          OUTPUT_VARIABLE H5PCC_SHOW_OUTPUT
          OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      message(STATUS "VPIC: h5pcc output: ${H5PCC_SHOW_OUTPUT}")

      # Extract -L directory using regex (captures the path after -L)
      string(REGEX MATCH "-L([^ ]+)" HDF5_L_MATCH "${H5PCC_SHOW_OUTPUT}")

      if(CMAKE_MATCH_1)
          set(HDF5_LIB_PATH "${CMAKE_MATCH_1}")
          message(STATUS "VPIC: HDF5 library path extracted: ${HDF5_LIB_PATH}")

          # Try to find the actual library files
          find_library(HDF5_HL_LIBRARY 
                       NAMES hdf5_hl libhdf5_hl 
                       PATHS ${HDF5_LIB_PATH} 
                       NO_DEFAULT_PATH)
          find_library(HDF5_LIBRARY 
                       NAMES hdf5 libhdf5 
                       PATHS ${HDF5_LIB_PATH} 
                       NO_DEFAULT_PATH)

          if(HDF5_HL_LIBRARY AND HDF5_LIBRARY)
              # Found actual paths - use them (best option, enables RPATH)
              list(APPEND HDF5_VPIC_LIBS ${HDF5_HL_LIBRARY} ${HDF5_LIBRARY})
              message(STATUS "VPIC: HDF5 libraries found:")
              message(STATUS "  HL: ${HDF5_HL_LIBRARY}")
              message(STATUS "  Core: ${HDF5_LIBRARY}")
          else()
              # find_library failed - let linker search using -L flag
              message(WARNING "VPIC: find_library failed. Using linker flags.")
              link_directories(${HDF5_LIB_PATH})
              list(APPEND HDF5_VPIC_LIBS "hdf5_hl" "hdf5")
          endif()
      else()
          message(WARNING "VPIC: Could not extract -L path from h5pcc output")
          list(APPEND HDF5_VPIC_LIBS "hdf5_hl" "hdf5")
      endif()
  else()
      # No h5pcc found - try CMake's discovered libraries
      if(HDF5_LIBRARIES)
          set(HDF5_VPIC_LIBS ${HDF5_LIBRARIES})
          message(STATUS "VPIC: Using CMake-discovered HDF5 libraries: ${HDF5_LIBRARIES}")
      else()
          # Last resort: bare library names
          list(APPEND HDF5_VPIC_LIBS "hdf5_hl" "hdf5")
          message(WARNING "VPIC: h5pcc not found, using default library names")
      endif()
  endif()

  # -----------------------------------------------------------------------------
  # ADD TO SHELL WRAPPER (Keep existing behavior for bin/vpic script)
  # -----------------------------------------------------------------------------
  # Convert to space-separated string for the wrapper script
  # Convert to space-separated string for the wrapper script
  string(REPLACE ";" " " HDF5_LIB_STRING "${HDF5_VPIC_LIBS}")

  # Determine the directory containing the HDF5 libs so we can bake an rpath.
  # Prefer the path extracted from h5pcc; otherwise derive it from the
  # resolved library file.
  set(HDF5_RPATH_DIR "")
  if(DEFINED HDF5_LIB_PATH)
    set(HDF5_RPATH_DIR "${HDF5_LIB_PATH}")
  elseif(HDF5_LIBRARY)
    get_filename_component(HDF5_RPATH_DIR "${HDF5_LIBRARY}" DIRECTORY)
  endif()

  if(HDF5_RPATH_DIR)
    set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} -L${HDF5_RPATH_DIR} -Wl,-rpath,${HDF5_RPATH_DIR} ${HDF5_LIB_STRING}")
  else()
    set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} ${HDF5_LIB_STRING}")
  endif()

  # Status Reporting
  if(HDF5_IS_PARALLEL)
    message(STATUS "VPIC: HDF5 Parallel Enabled")
    add_definitions(-DVPIC_HDF5_PARALLEL)
  else()
    message(STATUS "VPIC: HDF5 Serial Enabled")
  endif()
endif()

# Allow files to be references from the root, (eg #include "src/vpic/vpic.h")
include_directories(${PROJECT_SOURCE_DIR})

#------------------------------------------------------------------------------#
# Kokkos
#------------------------------------------------------------------------------#
set(Kokkos_ENABLE_DEBUG OFF)

option(VPIC_KOKKOS_DEBUG "Enable Debug In Kokkos" OFF)

if (VPIC_KOKKOS_DEBUG)
    set(Kokkos_ENABLE_DEBUG ON)
endif(VPIC_KOKKOS_DEBUG)

set(Kokkos_ENABLE_EXPLICIT_INSTANTIATION OFF)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Default allow serial execution")

if(ENABLE_KOKKOS_OPENMP)
  if (OPENMP_FOUND)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Use OpenMP if found")
  endif()
else()
    set(Kokkos_ENABLE_OPENMP OFF)
endif()

if(ENABLE_KOKKOS_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    add_definitions(-DENABLE_KOKKOS_CUDA)
    # --- Ensure the CUDA driver library (libcuda) is linked into deck builds ---
    find_library(CUDA_DRIVER_LIB cuda
      HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs
        /usr/lib64
    )
    if(CUDA_DRIVER_LIB)
      get_filename_component(_cuda_dir "${CUDA_DRIVER_LIB}" DIRECTORY)
      string(APPEND VPIC_CXX_LIBRARIES " -L${_cuda_dir} -lcuda ")
      message(STATUS "Added CUDA driver lib: ${CUDA_DRIVER_LIB}")
    else()
      message(WARNING "libcuda not found; deck link may fail")
    endif()
    # If not on a Cray system, this can cause CMake to stall
    if(DEFINED ENV{CRAYPE_VERSION})
      set(ENV{NVCC_WRAPPER_DEFAULT_COMPILER} ${CMAKE_CXX_COMPILER})
    endif()
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Use Cuda with Kokkos")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable cuda lambda support")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")

    # turn off -std=gnu++ to avoid error: identifier "__float128" is undefined issue
    # see https://github.com/UoB-HPC/BabelStream/issues/16
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Don't use intel intrinsics if we are using CUDA
    option(USE_V4_ALTIVEC "Enable V4 Altivec" OFF)
    option(USE_V4_PORTABLE "Enable V4 Portable" OFF)
    option(USE_V4_SSE "Enable V4 SSE" OFF)

  else()
    message("CUDA not found and options not changed")
  endif()
endif()

# Disable platform specific options
if (APPLE)
  set(Kokkos_ENABLE_LIBRT OFF CACHE BOOL "librt is not available on macOS")
endif()



# TODO: This is only need for non-linux platforms
set(Kokkos_ENABLE_LIBDL OFF)
set(Kokkos_ENABLE_PROFILING OFF) # NO libDL => no profiling


if (BUILD_INTERNAL_KOKKOS)
  message("Building our own kokkos..")
  add_subdirectory(kokkos)
  set(Kokkos_ENABLE_LIBDL OFF)
  set(Kokkos_ENABLE_PROFILING OFF) # NO libDL => no profiling
  #include_directories(${Kokkos_INCLUDE_DIRS_RET})
  #set(KOKKOS_INCLUDE_DIRS ${Kokkos_INCLUDE_DIRS_RET})
else()
  # CONSIDER ADDING --expt-extended-lambda
#  find_package(KOKKOS REQUIRED)
  find_package(Kokkos REQUIRED)

  #get_target_property(KOKKOS_INCLUDE_DIRS Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  #get_target_property(KOKKOS_LIB_DIR Kokkos::kokkos INTERFACE_LINK_LIBRARIES)
  #message(${KOKKOS_INCLUDE_DIRS})
  #link_directories(${KOKKOS_INCLUDE_DIRS}/../lib) # TODO: bad idea..
endif(BUILD_INTERNAL_KOKKOS)


get_target_property(KOKKOS_COMPILE_OPTIONS Kokkos::kokkoscore INTERFACE_COMPILE_OPTIONS)
get_target_property(KOKKOS_INCLUDE_DIRS Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(KOKKOS_INCLUDE_CONTAINERS Kokkos::kokkoscontainers INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(KOKKOS_INCLUDE_ALGORITHMS Kokkos::kokkosalgorithms INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(KOKKOS_INCLUDE_SIMD Kokkos::kokkossimd INTERFACE_INCLUDE_DIRECTORIES)

set(KOKKOS_CORE_INCLUDES_FILE ${PROJECT_BINARY_DIR}/kokkos_core_includes)
set(KOKKOS_CONTAINER_INCLUDES_FILE ${PROJECT_BINARY_DIR}/kokkos_container_includes)
set(KOKKOS_ALGORITHM_INCLUDES_FILE ${PROJECT_BINARY_DIR}/kokkos_algorithm_includes)
set(KOKKOS_COMPILE_OPTIONS_FILE ${PROJECT_BINARY_DIR}/kokkos_compile_options)
set(KOKKOS_INCLUDES_FILE ${PROJECT_BINARY_DIR}/kokkos_includes)

# For an "external" build these will be the same, but for an internal we need to gather both
file(GENERATE OUTPUT ${KOKKOS_CORE_INCLUDES_FILE} CONTENT "${KOKKOS_INCLUDE_DIRS}")
file(GENERATE OUTPUT ${KOKKOS_CONTAINER_INCLUDES_FILE} CONTENT "${KOKKOS_INCLUDE_CONTAINERS}")
file(GENERATE OUTPUT ${KOKKOS_ALGORITHM_INCLUDES_FILE} CONTENT "${KOKKOS_INCLUDE_ALGORITHMS}")
# restrict compile options file to C++ to avoid multi-language error (BUILD_INTERNAL_KOKKOS=OFF)
file(GENERATE
     OUTPUT ${KOKKOS_COMPILE_OPTIONS_FILE}
     CONTENT "$<LIST:JOIN,$<LIST:TRANSFORM,$<TARGET_PROPERTY:INTERFACE_COMPILE_OPTIONS>,REPLACE,SHELL[:],>, >"
     CONDITION $<COMPILE_LANGUAGE:CXX>
     TARGET Kokkos::kokkoscore
)
if(NOT "${KOKKOS_INCLUDE_DIRS}" STREQUAL "${KOKKOS_INCLUDE_CONTAINERS}")
  set(KOKKOS_INCLUDE_DIRS "${KOKKOS_INCLUDE_DIRS};${KOKKOS_INCLUDE_CONTAINERS}")
endif()
if(NOT "${KOKKOS_INCLUDE_DIRS}" STREQUAL "${KOKKOS_INCLUDE_ALGORITHMS}")
  set(KOKKOS_INCLUDE_DIRS "${KOKKOS_INCLUDE_DIRS};${KOKKOS_INCLUDE_ALGORITHMS}")
endif()
if(NOT "${KOKKOS_INCLUDE_DIRS}" STREQUAL "${KOKKOS_INCLUDE_SIMD}")
  set(KOKKOS_INCLUDE_DIRS "${KOKKOS_INCLUDE_DIRS};${KOKKOS_INCLUDE_SIMD}")
endif()
file(GENERATE OUTPUT ${KOKKOS_INCLUDES_FILE} CONTENT "${KOKKOS_INCLUDE_DIRS}")

# Libs
string(REPLACE ";" " " string_libraries "${KOKKOS_LIBS} ${KOKKOS_LDFLAGS}")
set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} ${string_libraries}")

#------------------------------------------------------------------------------#
# OpenSSL
#------------------------------------------------------------------------------#

if(ENABLE_OPENSSL)
  find_package(OpenSSL REQUIRED)

  include_directories(${OPENSSL_INCLUDE_DIR})
  string(REPLACE ";" " " string_libraries "${OPENSSL_LIBRARIES}")
  set(VPIC_CXX_LIBRARIES "${VPIC_CXX_LIBRARIES} ${string_libraries}")
endif(ENABLE_OPENSSL)

find_package(Threads REQUIRED)

#------------------------------------------------------------------------------#
# Act on build options set in project.cmake
#------------------------------------------------------------------------------#

message("-- VPIC Optimization Settings")
if (VPIC_DUMP_ENERGIES)
    add_definitions(-DDUMP_ENERGIES)
endif(VPIC_DUMP_ENERGIES)

if(VPIC_ENABLE_AUTO_TUNING)
  add_definitions(-DVPIC_ENABLE_AUTO_TUNING)
  message("--     VPIC: Enabled automatic tuning.  Optimizations will be selected during compilation.")
endif(VPIC_ENABLE_AUTO_TUNING)

if(VPIC_ENABLE_TEAM_REDUCTION)
  add_definitions(-DVPIC_DISABLE_AUTOSETTINGS -DVPIC_ENABLE_HIERARCHICAL -DVPIC_ENABLE_TEAM_REDUCTION)
  set(VPIC_ENABLE_HIERARCHICAL ON CACHE INTERNAL "Enable Hierarchical parallelism")
  message("--     VPIC: Enabled reduction optimization")
endif(VPIC_ENABLE_TEAM_REDUCTION)

if (VPIC_ENABLE_VECTORIZATION)
  add_definitions(-DVPIC_DISABLE_AUTOSETTINGS -DVPIC_ENABLE_HIERARCHICAL -DVPIC_ENABLE_VECTORIZATION)
  set(VPIC_ENABLE_HIERARCHICAL ON CACHE INTERNAL "Enable Hierarchical parallelism")
  message("--     VPIC: Enabled improved CPU vectorization")
endif(VPIC_ENABLE_VECTORIZATION)

if (VPIC_ENABLE_HIERARCHICAL)
  add_definitions(-DVPIC_DISABLE_AUTOSETTINGS -DVPIC_ENABLE_HIERARCHICAL)
  message("--     VPIC: Enabled hierarchical parallelism")
endif(VPIC_ENABLE_HIERARCHICAL)

if (VPIC_ENABLE_ACCUMULATORS)
  add_definitions(-DVPIC_DISABLE_AUTOSETTINGS -DVPIC_ENABLE_ACCUMULATORS)
  message("--     VPIC: Enabled accumulators")
endif(VPIC_ENABLE_ACCUMULATORS)

if (VPIC_ENABLE_VARIABLE_CHARGE)
   add_definitions(-DVARIABLE_CHARGE)
   set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DVARIABLE_CHARGE")
   message("--     VPIC: Enabled variable charge for particles")
endif(VPIC_ENABLE_VARIABLE_CHARGE)

if (VPIC_ENABLE_EXTERNAL_FORCE)
   add_definitions(-DEXTERNAL_FORCE)
   set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DEXTERNAL_FORCE")
   message("--     VPIC: Enabled external force for particles")
endif(VPIC_ENABLE_EXTERNAL_FORCE)

if (VPIC_ENABLE_GPU_AWARE_MPI)
  add_definitions(-DVPIC_ENABLE_GPU_AWARE_MPI)
  message("--     VPIC: Enabled GPU-Aware MPI")
endif(VPIC_ENABLE_GPU_AWARE_MPI)

if (VPIC_ENABLE_HALO_EXCHANGE)
  add_definitions(-DVPIC_ENABLE_HALO_EXCHANGE)
  message("--     VPIC: Enabled halo exchange path")
endif(VPIC_ENABLE_HALO_EXCHANGE)

if (VPIC_ENABLE_PARTICLE_ANNOTATIONS)
  add_definitions(-DVPIC_ENABLE_PARTICLE_ANNOTATIONS)
  set(ENABLE_ANNOTATIONS ON CACHE INTERNAL "")
endif(VPIC_ENABLE_PARTICLE_ANNOTATIONS)

if (VPIC_ENABLE_TRACER_PARTICLES)
  add_definitions(-DVPIC_ENABLE_TRACER_PARTICLES -DVPIC_ENABLE_PARTICLE_ANNOTATIONS -DVPIC_ENABLE_HDF5)
  set(ENABLE_ANNOTATIONS ON CACHE INTERNAL "")
  set(ENABLE_TRACERS ON CACHE INTERNAL "")
  set(ENABLE_HDF5 ON CACHE INTERNAL "")
endif(VPIC_ENABLE_TRACER_PARTICLES)

if (VPIC_ENABLE_LEGACY_DATA_STRUCTURES)
  add_definitions(-DVPIC_ENABLE_LEGACY_DATA_STRUCTURES)
  set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DVPIC_ENABLE_LEGACY_DATA_STRUCTURES")
  message("--     VPIC: Enabled legacy data structures")
endif(VPIC_ENABLE_LEGACY_DATA_STRUCTURES)

if (VPIC_SHAPE_QS)
  add_definitions(-DSHAPE_QS)
  message("--     VPIC: Using quadratic-sum particle shape")
  set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} -DSHAPE_QS")
else()
  add_definitions(-DSHAPE_NGP)
  message("--     VPIC: Using nearest-grid-point particle shape")
  set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} -DSHAPE_NGP")
endif(VPIC_SHAPE_QS)

if (HYB_USE_SEPARATE_PE)
   add_definitions(-DHYB_USE_SEPARATE_PE)
   set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DHYB_USE_SEPARATE_PE")
   message("--     HYB: Using separate electron energy equation.")
endif(HYB_USE_SEPARATE_PE)

if (HYB_USE_STATIC_E)
   add_definitions(-DHYB_USE_STATIC_E)
   set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DHYB_USE_STATIC_E")
   message("--     HYB: Using electrostatic field solve.")
endif(HYB_USE_STATIC_E)

if (HYB_USE_RADIATION)
   add_definitions(-DHYB_USE_RADIATION)
   set(VPIC_CPPFLAGS "${VPIC_CPPFLAGS} -DHYB_USE_RADIATION")
   message("--     HYB: User radiation transport enabled.")
endif(HYB_USE_RADIATION)

set(USE_V4)
if(USE_V4_ALTIVEC)
  add_definitions(-DUSE_V4_ALTIVEC)
  set(USE_V4 True)
endif(USE_V4_ALTIVEC)

if(USE_V4_PORTABLE)
  add_definitions(-DUSE_V4_PORTABLE)
  set(USE_V4 True)
endif(USE_V4_PORTABLE)

if(USE_V4_SSE)
  add_definitions(-DUSE_V4_SSE)
  set(USE_V4 True)
endif(USE_V4_SSE)

if(ENABLE_OPENSSL)
  add_definitions(-DENABLE_OPENSSL)
endif(ENABLE_OPENSSL)

#------------------------------------------------------------------------------#
# Add compile definitions
#------------------------------------------------------------------------------#
#set(EXTERNAL_DEFINITIONS_FILE ${PROJECT_BINARY_DIR}/external_definitions)
get_directory_property( DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
string(REPLACE ";" " -D" DirDefs "-D${DirDefs}")

#------------------------------------------------------------------------------#
# Output log variables
#------------------------------------------------------------------------------#
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   VPIC_GIT_COMMIT_HASH
        )
ELSE()
    SET(VPIC_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
string(TIMESTAMP _build_time_stamp)

MESSAGE(STATUS "VPIC Revision = '${VPIC_GIT_COMMIT_HASH}'")
#add_definitions("-DVPIC_GIT_HASH=${VPIC_GIT_COMMIT_HASH}")

configure_file (
    "${PROJECT_SOURCE_DIR}/cmake/git_version.h.in"
    "${PROJECT_BINARY_DIR}/include/git_version.h"
)

include_directories(${PROJECT_BINARY_DIR}/include) # for generated git_version.h

#------------------------------------------------------------------------------#
# Handle vpic compile script last
#------------------------------------------------------------------------------#

# Allow files to be references from the root, (eg #include "src/vpic/vpic.h")
#include_directories(${PROJECT_SOURCE_DIR})

if(BUILD_SHARED_LIBS)
    set(VPIC_CXX_FLAGS "-rdynamic ${VPIC_CXX_FLAGS}")
endif(BUILD_SHARED_LIBS)

if(ENABLE_COVERAGE_BUILD)
    set(VPIC_CXX_FLAGS "${VPIC_CXX_FLAGS} --coverage")
endif(ENABLE_COVERAGE_BUILD)

# process Makefile.run.in to get a simple Makefile.run for a run. Points to
# local built exe wrapper, and has example deck/platform.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/sample/Makefile.run.in
    ${CMAKE_BINARY_DIR}/bin/Makefile.run
)

# install script
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/vpic.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic-install
    @ONLY
)

install(FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic-install
  DESTINATION bin
  RENAME vpic
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    )

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/deck/main.cc
  DESTINATION share/vpic)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/deck/wrapper.cc
  DESTINATION share/vpic)

if(ENABLE_KOKKOS_CUDA AND DEFINED ENV{CRAYPE_VERSION})
# local script
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/bin/vpic-localCrayA100.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic
    @ONLY
)
else()
# local script
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/vpic-local.in
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic
    @ONLY
)
endif()

file(COPY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/vpic
  DESTINATION ${CMAKE_BINARY_DIR}/bin
  FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

#------------------------------------------------------------------------------#
# Add library target
#------------------------------------------------------------------------------#

file(GLOB_RECURSE VPIC_SRC src/*.c src/*.cc)
file(GLOB_RECURSE VPIC_NOT_SRC src/util/v4/test/v4.cc src/util/rng/test/rng.cc)
list(REMOVE_ITEM VPIC_SRC ${VPIC_NOT_SRC})
option(NO_LIBVPIC "Don't build a libvpic, but all in one" OFF)
if(NO_LIBVPIC)
  set(VPIC_EXPOSE "INTERFACE")
  add_library(vpic INTERFACE)
else()
  set(VPIC_EXPOSE "PRIVATE")
  # TODO: this does a bad job of detecting when changes are need in ./bin/vpic
  # for example if you delete ./bin/vpic it won't get rebuilt
  add_library(vpic ${VPIC_SRC})
  set(VPIC_SRC)
  install(TARGETS vpic LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
endif()
target_include_directories(vpic INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vpic ${VPIC_EXPOSE} ${MPI_CXX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OPENSSL_LIBRARIES} ${CMAKE_DL_LIBS} Kokkos::kokkos)

# Link HDF5 if enabled
if(VPIC_ENABLE_HDF5 AND HDF5_VPIC_LIBS)
  target_link_libraries(vpic ${VPIC_EXPOSE} ${HDF5_VPIC_LIBS})
endif()

target_compile_options(vpic ${VPIC_EXPOSE} ${MPI_C_COMPILE_FLAGS} ${KOKKOS_COMPILE_OPTIONS})
#-- make some options public
if (VPIC_SHAPE_QS)
  target_compile_definitions(vpic PUBLIC SHAPE_QS)
else()
  target_compile_definitions(vpic PUBLIC SHAPE_NGP)
endif(VPIC_SHAPE_QS)
if (HYB_USE_RADIATION)
  target_compile_definitions(vpic PUBLIC HYB_USE_RADIATION)
  target_compile_definitions(vpic PUBLIC VPIC_ENABLE_HDF5)
  target_compile_definitions(vpic PUBLIC VPIC_ENABLE_LEGACY_DATA_STRUCTURES)
endif(HYB_USE_RADIATION)

macro(build_a_vpic name deck)
  if(NOT EXISTS ${deck})
    message(FATAL_ERROR "Could not find deck '${deck}'")
  endif()
  add_executable(${name} ${CMAKE_SOURCE_DIR}/deck/main.cc ${CMAKE_SOURCE_DIR}/deck/wrapper.cc ${VPIC_SRC})
  target_link_directories(${name} PUBLIC "/vast/home/nphtan/vol-async/install/lib/")
  target_compile_options(${name} PRIVATE ${KOKKOS_COMPILE_OPTIONS})
  if(VPIC_ENABLE_HDF5)
    target_link_libraries(${name} vpic Kokkos::kokkos HDF5::HDF5)
  else()
    target_link_libraries(${name} vpic Kokkos::kokkos)
  endif()
  target_compile_definitions(${name} PRIVATE INPUT_DECK="${deck}")
endmacro(build_a_vpic)

if(USER_DECKS)
  foreach(DECK ${USER_DECK})
    get_filename_component(NAME "${DECK}" NAME_WE)
    get_filename_component(DECKDIR "${DECK}" DIRECTORY)
    if(NOT DECKDIR)
      set(DECK "${CMAKE_CURRENT_BINARY_DIR}/${DECK}")
    endif()
    build_a_vpic(${NAME} ${DECK})
  endforeach()
endif()

#string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
#message("Build mode flags ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
#message("Debug build flags ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
#message("Release build flags ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
#message("RelWithDebInfo build flags ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

#------------------------------------------------------------------------------#
# Add VPIC integrated test mechanism
#------------------------------------------------------------------------------#

if(ENABLE_INTEGRATED_TESTS)
  enable_testing()
  add_subdirectory(test/integrated)

endif(ENABLE_INTEGRATED_TESTS)

if(ENABLE_UNIT_TESTS)

  enable_testing()

  # NOTE: This can also be done by a subdirectory (useful when we have multiple tests)
  # Add includes for Catch2
  set(TEST_DIR "./test/include")
  include_directories(${TEST_DIR})

  # Add tests individually
  if (USE_V4)
    add_executable(v4 src/util/v4/test/v4.cc)
    add_test(NAME v4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./v4)
  endif(USE_V4)

  # RNG tests
  add_executable(rng src/util/rng/test/rng.cc)
  target_link_libraries(rng vpic Kokkos::kokkos)
  get_target_property(OUT rng LINK_LIBRARIES)

  add_test(NAME rng COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ./rng)

  add_subdirectory(test/unit)

endif(ENABLE_UNIT_TESTS)
#~---------------------------------------------------------------------------~-#
# vim: set tabstop=2 shiftwidth=2 expandtab :
#~---------------------------------------------------------------------------~-#
