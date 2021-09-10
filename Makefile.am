CUDA_PATH = @CUDA_PATH@

AM_CPPFLAGS  = -I$(srcdir)
AM_CPPFLAGS += -I$(srcdir)/include
AM_CPPFLAGS += -I$(srcdir)/src
#AM_CPPFLAGS += -I$(CUDA_PATH)/include
AM_CPPFLAGS += -D__STDC_FORMAT_MACROS

#AM_LDFLAGS   = -L$(CUDA_PATH)/lib64
LIBGDSTOOLS = @LIBGDSTOOLS@
LIBNVTX = @LIBNVTX@

lib_LTLIBRARIES = src/libgdsync.la

ACLOCAL_AMFLAGS = -Iconfig
AM_CFLAGS = -g -Wall
AM_CFLAGS += $(PTHREAD_CFLAGS)
AM_CXXFLAGS = $(PTHREAD_CFLAGS)
EXTRA_DIST = autogen.sh

include_HEADERS = include/gdsync.h
libgdsyncincludedir = $(includedir)/gdsync
libgdsyncinclude_HEADERS = include/gdsync/core.h include/gdsync/device.cuh include/gdsync/mlx5.h include/gdsync/tools.h

src_libgdsync_la_CFLAGS = $(AM_CFLAGS)
src_libgdsync_la_SOURCES = src/gdsync.cpp src/memmgr.cpp src/mem.cpp src/objs.cpp src/apis.cpp src/mlx5.cpp src/mlx5-exp.cpp include/gdsync.h 
src_libgdsync_la_LDFLAGS = -version-info @VERSION_INFO@

noinst_HEADERS = src/mem.hpp src/memmgr.hpp src/objs.hpp src/rangeset.hpp src/utils.hpp src/archutils.h src/mlnxutils.h src/mlx5-exp.hpp

# if enabled at configure time

if TEST_ENABLE

bin_PROGRAMS = tests/gds_kernel_latency tests/gds_poll_lat tests/gds_kernel_loopback_latency tests/gds_sanity
noinst_PROGRAMS = tests/rstest tests/wqtest

tests_gds_kernel_latency_SOURCES = tests/gds_kernel_latency.c tests/gpu_kernels.cu tests/pingpong.c tests/gpu.cpp
tests_gds_kernel_latency_LDADD = $(top_builddir)/src/libgdsync.la $(MPILDFLAGS) $(LIBGDSTOOLS) -lgdrapi $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)

tests_rstest_SOURCES = tests/rstest.cpp
tests_rstest_LDADD = 

tests_wqtest_SOURCES = tests/task_queue_test.cpp
tests_wqtest_LDADD = $(PTHREAD_LIBS)

tests_gds_poll_lat_SOURCES = tests/gds_poll_lat.c tests/gpu.cpp tests/gpu_kernels.cu
tests_gds_poll_lat_LDADD = $(top_builddir)/src/libgdsync.la $(LIBGDSTOOLS) -lgdrapi $(MPILDFLAGS) $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)

tests_gds_sanity_SOURCES = tests/gds_sanity.cpp tests/gpu.cpp tests/gpu_kernels.cu
tests_gds_sanity_LDADD = $(top_builddir)/src/libgdsync.la $(LIBGDSTOOLS) -lgdrapi $(MPILDFLAGS) $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)

tests_gds_kernel_loopback_latency_SOURCES = tests/gds_kernel_loopback_latency.c tests/pingpong.c tests/gpu.cpp tests/gpu_kernels.cu
tests_gds_kernel_loopback_latency_LDADD = $(top_builddir)/src/libgdsync.la $(LIBGDSTOOLS) -lgdrapi $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)


SUFFIXES= .cu

.cu.o:
	$(NVCC) $(CPPFLAGS) $(AM_CPPFLAGS) $(NVCCFLAGS)  -c -o $@ $<


.cu.lo:
	$(LIBTOOL) --tag=CC --mode=compile $(NVCC) -o $@ -c $< $(CPPFLAGS) $(AM_CPPFLAGS) $(NVCCFLAGS)

endif
