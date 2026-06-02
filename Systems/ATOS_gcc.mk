# systems/ATOS_gcc.mk
# System file for ATOS-like environment using GNU toolchain + OpenMPI
# Built NetCDF-C (parallel via HDF5) and HDF5 installed under $HOME/opt
#
# Module stack you should load before building:
#   module --ignore_cache load prgenv/gnu
#   module --ignore_cache load gcc/13.2.0
#   module --ignore_cache load openmpi/4.1.1.1
#
# Paths used in this file (adjust if you installed somewhere else)
NETCDF_PAR_DIR := $(HOME)/opt/netcdf-parallel-4.6.2
HDF5_PAR_DIR   := $(HOME)/opt/hdf5-par

#---------------------------------------------------------------------
# Compilers
CXX     := g++
MPICXX  := mpicxx

#---------------------------------------------------------------------
# Include and library dirs (explicit)
NETCDF_INC  := $(NETCDF_PAR_DIR)/include
NETCDF_LIB  := $(NETCDF_PAR_DIR)/lib
HDF5_INC    := $(HDF5_PAR_DIR)/include
HDF5_LIB    := $(HDF5_PAR_DIR)/lib

INC_DIRS := -I$(NETCDF_INC) -I$(HDF5_INC)
LIB_DIRS := -L$(NETCDF_LIB) -L$(HDF5_LIB)

#---------------------------------------------------------------------
# Compiler and linker flags
CFLAGS          := -Wall -std=c++14
DEBUG_FLAGS     := -g3 -ggdb
DEBUG_LDFLAGS   := -g3 -ggdb
OPT_FLAGS       := -O3 -march=native -funroll-loops
EXTRA_OPT_FLAGS :=
ALGLIB_OPT_FLAGS:= -O3 -DAE_CPU=AE_GNU

# append include dirs to CXXFLAGS and MPICXXFLAGS if used directly
CXXFLAGS := $(CFLAGS) $(OPT_FLAGS) $(EXTRA_OPT_FLAGS) $(INC_DIRS)
MPICXXFLAGS := $(CXXFLAGS)

#---------------------------------------------------------------------
# Libraries and link order
# NetCDF (uses HDF5), HDF5 high-level, HDF5 core, zlib, math, OpenMP
LINKS := -lnetcdf -lhdf5_hl -lhdf5 -lz -lm -fopenmp

# Provide complete link line used by the Makefile
LDLIBS := $(LIB_DIRS) $(LINKS)

RPATH_FLAGS := -Wl,-rpath,$(NETCDF_LIB) -Wl,-rpath,$(HDF5_LIB)
LDFLAGS += $(RPATH_FLAGS)

#---------------------------------------------------------------------
# Depth/lat/lon etc defaults used by the build system
# (keep these so Makefile's rules find something sensible)
LIB_DIRS  := $(LIB_DIRS)
INC_DIRS  := $(INC_DIRS)

#---------------------------------------------------------------------
# Optional: user can override CC/MPICC if needed
# Example: make CC=/path/to/mpicc CXX=/path/to/mpicxx
CC      ?= $(shell which gcc 2>/dev/null || echo gcc)
MPICC   ?= $(shell which mpicc 2>/dev/null || echo mpicc)

#---------------------------------------------------------------------
# Diagnostic target (helpful)
.PHONY: info
info:
	@echo "Using MPICXX: $(MPICXX)"
	@echo "NETCDF include: $(NETCDF_INC)"
	@echo "NETCDF lib:     $(NETCDF_LIB)"
	@echo "HDF5 include:   $(HDF5_INC)"
	@echo "HDF5 lib:       $(HDF5_LIB)"
	@echo "CXXFLAGS:       $(CXXFLAGS)"
	@echo "LDLIBS:         $(LDLIBS)"

# End of file

