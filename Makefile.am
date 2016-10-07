CUDA_PATH = @CUDA_PATH@

AM_CPPFLAGS  = -I$(srcdir)
AM_CPPFLAGS += -I$(srcdir)/include
AM_CPPFLAGS += -I$(srcdir)/src
AM_CPPFLAGS += -I$(CUDA_PATH)/include
AM_CPPFLAGS += -D__STDC_FORMAT_MACROS

AM_LDFLAGS   = -L$(CUDA_PATH)/lib64

lib_LTLIBRARIES = src/libgdsync.la

ACLOCAL_AMFLAGS = -Iconfig
AM_CFLAGS = -g -Wall

EXTRA_DIST = autogen.sh
EXTRA_DIST += scripts/expose_libmlx5_headers/libmlx_expose_headers scripts/expose_libmlx5_headers/defines.txt scripts/expose_libmlx5_headers/structures.txt scripts/expose_libmlx5_headers/enumerations.txt

src_libgdsync_la_CFLAGS = $(AM_CFLAGS)
src_libgdsync_la_SOURCES = src/gdsync.cpp src/memmgr.cpp src/mem.cpp src/objs.cpp src/apis.cpp src/mlx5.cpp
src_libgdsync_la_LDFLAGS = -version-info 1:0:0

noinst_HEADERS = src/mem.hpp src/memmgr.hpp src/objs.hpp src/rangeset.hpp src/utils.hpp

# if enabled at configure time

if TEST_ENABLE

bin_PROGRAMS = tests/gds_kernel_latency tests/rstest
noinst_PROGRAMS = 

tests_gds_kernel_latency_SOURCES = tests/gds_kernel_latency.c tests/gpu_kernels.cu tests/pingpong.c tests/gpu.cpp
tests_gds_kernel_latency_LDADD = $(top_builddir)/src/libgdsync.la -lgdrapi -lmpi -lcuda -lcudart

tests_rstest_SOURCES = tests/rstest.cpp
tests_rstest_LDADD = 


SUFFIXES= .cu

.cu.o:
	$(NVCC) $(NVCCFLAGS)  -c -o $@ $<


.cu.lo:
	$(LIBTOOL) --tag=CC --mode=compile $(NVCC) -o $@ -c $< $(NVCCFLAGS)

endif