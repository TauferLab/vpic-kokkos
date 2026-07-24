# ==============================================================================
# VPIC I/O Correctness Tests
# ==============================================================================
# Tests the new binary and HDF5 writers against legacy reference data
# 
# Test Modes:
#   - Binary: Always runs (no external dependencies)
#   - HDF5:   Only runs if VPIC_ENABLE_HDF5=ON
#
# Each mode has:
#   - Single-rank test (basic correctness)
#   - 4-rank test (domain decomposition + MPI oversubscription support)
#
# Cleanup Behavior:
#   - Success: Output files removed automatically
#   - Failure: Output preserved for debugging
# ==============================================================================

cmake_minimum_required(VERSION 3.18)
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_EXECUTABLE)
  find_program(Python3_EXECUTABLE python3 REQUIRED)
endif()

# ==============================================================================
# MPI OVERSUBSCRIPTION DETECTION
# ==============================================================================
# Different MPI implementations handle oversubscription differently:
#   - Open MPI:  Requires explicit --oversubscribe flag
#   - MPICH:     Enabled by default (no flag needed)
#   - MVAPICH:   Enabled by default (no flag needed)
#   - Intel MPI: Generally enabled by default
#   - Cray MPI:  Uses aprun/srun, handles implicitly
# ==============================================================================

if(MPI_C_COMPILER)
  execute_process(
    COMMAND ${MPI_C_COMPILER} --version
    OUTPUT_VARIABLE MPI_VERSION_STRING
    ERROR_VARIABLE MPI_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  
  # Check for specific MPI implementations
  if(MPI_VERSION_STRING MATCHES "Open MPI")
    set(MPI_OVERSUBSCRIBE_FLAG "--oversubscribe")
    message(STATUS "Detected Open MPI - using ' ${MPI_OVERSUBSCRIBE_FLAG}' flag")
    
  elseif(MPI_VERSION_STRING MATCHES "MVAPICH")
    # MVAPICH/MVAPICH2 allows oversubscription by default
    set(MPI_OVERSUBSCRIBE_FLAG "")
    message(STATUS "Detected MVAPICH - oversubscription enabled by default")
    
  elseif(MPI_VERSION_STRING MATCHES "MPICH")
    # MPICH allows oversubscription by default
    set(MPI_OVERSUBSCRIBE_FLAG "")
    message(STATUS "Detected MPICH - oversubscription enabled by default")
    
  elseif(MPI_VERSION_STRING MATCHES "Intel.*MPI")
    # Intel MPI typically allows oversubscription by default
    set(MPI_OVERSUBSCRIBE_FLAG "")
    message(STATUS "Detected Intel MPI - oversubscription enabled by default")
    
  else()
    # Unknown MPI implementation - assume no special flag needed
    set(MPI_OVERSUBSCRIBE_FLAG "")
    message(STATUS "Unknown MPI implementation - no oversubscription flag (assuming enabled)")
    message(STATUS "  MPI version string: ${MPI_VERSION_STRING}")
  endif()
  
else()
  # No MPI compiler found
  set(MPI_OVERSUBSCRIBE_FLAG "")
  message(WARNING "MPI_C_COMPILER not defined - cannot detect MPI implementation")
endif()

# ==============================================================================
# TEST SUBDIRECTORIES
# ==============================================================================

# Binary tests always run (no external dependencies)
add_subdirectory(binary)

# HDF5 tests only run if HDF5 support is enabled
if(VPIC_ENABLE_HDF5)
  add_subdirectory(hdf5)
  message(STATUS "HDF5 I/O tests enabled")
else()
  message(STATUS "HDF5 I/O tests disabled (VPIC_ENABLE_HDF5=OFF)")
endif()

# ==============================================================================
# SUMMARY
# ==============================================================================

message(STATUS "I/O Test Configuration:")
message(STATUS "  Binary tests:        Enabled")
message(STATUS "  HDF5 tests: ${VPIC_ENABLE_HDF5}")
message(STATUS "  MPI oversubscribe:   ' ${MPI_OVERSUBSCRIBE_FLAG}'")