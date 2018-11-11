CUDA_PATH = @CUDA_PATH@

AM_CPPFLAGS  = -I$(srcdir)
AM_CPPFLAGS += -I$(srcdir)/include
AM_CPPFLAGS += -I$(srcdir)/src
#AM_CPPFLAGS += -I$(CUDA_PATH)/include
AM_CPPFLAGS += -D__STDC_FORMAT_MACROS

AM_LDFLAGS = -lmlx5
LIBGDSTOOLS = @LIBGDSTOOLS@
LIBNVTX = @LIBNVTX@

SMS = @GPU_ARCH@

GENMODULE = $(top_srcdir)/genmodule
BIN2C = $(CUDA_PATH)/bin/bin2c

lib_LTLIBRARIES = src/libgdsync.la

ACLOCAL_AMFLAGS = -Iconfig
AM_CFLAGS = -g -Wall
AM_CFLAGS += $(PTHREAD_CFLAGS)
AM_CXXFLAGS = $(PTHREAD_CFLAGS)
EXTRA_DIST = autogen.sh

include_HEADERS = include/gdsync.h
libgdsyncincludedir = $(includedir)/gdsync
libgdsyncinclude_HEADERS = include/gdsync/core.h include/gdsync/device.cuh  include/gdsync/mlx5.h include/gdsync/tools.h

src_libgdsync_la_CFLAGS = $(AM_CFLAGS)
src_libgdsync_la_SOURCES = src/gdsync.cpp src/memmgr.cpp src/mem.cpp src/objs.cpp src/apis.cpp src/mlx5.cpp src/gdsync_kernels.cu src/kernel_desc.cpp include/gdsync.h
src_libgdsync_la_LDFLAGS = -version-info 2:0:1
nodist_src_libgdsync_la_SOURCES = src/module_1snd2wait.cpp src/1snd2wait_fatbin.h src/1snd2wait.fatbin src/module_setsndpar.cpp src/setsndpar_fatbin.h src/setsndpar.fatbin

# these are generated files that must be cleaned
BUILT_SOURCES = src/module_1snd2wait.cpp src/1snd2wait_fatbin.h src/1snd2wait.fatbin
BUILT_SOURCES += src/module_setsndpar.cpp src/setsndpar_fatbin.h src/setsndpar.fatbin
#CLEANFILES = src/module_1snd2wait.cpp src/1snd2wait_fatbin.h src/1snd2wait.fatbin

src/module_1snd2wait.cpp: src/1snd2wait_fatbin.h src/kernel_desc.hpp
	$(GENMODULE) -k krn1snd2wait -o $@ -f $<

src/1snd2wait_fatbin.h: src/1snd2wait.fatbin
	$(BIN2C) --type char --padd \00,\00,\00,\00 --const --name krn1snd2wait_fatbin $< >$@ 

src/module_setsndpar.cpp: src/setsndpar_fatbin.h src/kernel_desc.hpp
	$(GENMODULE) -k krnsetsndpar -o $@ -f $<
src/setsndpar_fatbin.h: src/setsndpar.fatbin
	$(BIN2C) --type char --padd \00,\00,\00,\00 --const --name krnsetsndpar_fatbin $< >$@ 


GENCODE_FLAGS=
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM = $(lastword $(sort $(SMS)))
NUM_SMS = $(words $(sort $(SMS)))
$(eval GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM))

src/1snd2wait.fatbin: src/kernels/1snd2wait.cu
	nvcc $(GENCODE_FLAGS) $(CPPFLAGS) $(AM_CPPFLAGS)  -fatbin -o $@ $<

src/setsndpar.fatbin: src/kernels/setsndpar.cu
	nvcc $(GENCODE_FLAGS) $(CPPFLAGS) $(AM_CPPFLAGS)  -fatbin -o $@ $<

noinst_HEADERS = src/mem.hpp src/memmgr.hpp src/objs.hpp src/rangeset.hpp src/utils.hpp src/archutils.h src/mlnxutils.h

# if enabled at configure time

if TEST_ENABLE

bin_PROGRAMS = tests/gds_kernel_latency tests/gds_poll_lat tests/gds_kernel_loopback_latency tests/gds_sanity
noinst_PROGRAMS = tests/rstest tests/wqtest

tests_gds_kernel_latency_SOURCES = tests/gds_kernel_latency.c tests/gpu_kernels.cu tests/pingpong.c tests/gpu.cpp
tests_gds_kernel_latency_LDADD = $(top_builddir)/src/libgdsync.la -lmpi $(LIBGDSTOOLS) -lgdrapi -lmlx5 $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)

tests_rstest_SOURCES = tests/rstest.cpp
tests_rstest_LDADD = 

tests_wqtest_SOURCES = tests/task_queue_test.cpp
tests_wqtest_LDADD = $(PTHREAD_LIBS)

tests_gds_poll_lat_SOURCES = tests/gds_poll_lat.c tests/gpu.cpp tests/gpu_kernels.cu
tests_gds_poll_lat_LDADD = $(top_builddir)/src/libgdsync.la $(LIBGDSTOOLS) -lgdrapi -lmlx5 -lmpi $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)

tests_gds_sanity_SOURCES = tests/gds_sanity.cpp tests/gpu.cpp tests/gpu_kernels.cu
tests_gds_sanity_LDADD = $(top_builddir)/src/libgdsync.la $(LIBGDSTOOLS) -lgdrapi -lmlx5 -lmpi $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)

tests_gds_kernel_loopback_latency_SOURCES = tests/gds_kernel_loopback_latency.c tests/pingpong.c tests/gpu.cpp tests/gpu_kernels.cu
tests_gds_kernel_loopback_latency_LDADD = $(top_builddir)/src/libgdsync.la $(LIBGDSTOOLS) -lgdrapi -lmlx5 $(LIBNVTX) -lcuda -lcudart $(PTHREAD_LIBS)

endif

SUFFIXES= .cu

.cu.o:
	$(NVCC) $(CPPFLAGS) $(AM_LDFLAGS)  $(AM_CPPFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -c -o $@ $<

.cu.lo:
	$(LIBTOOL) --tag=CXX --mode=compile $(top_srcdir)/cudalt $(NVCC) --resource-usage -o $@ -c $< $(CPPFLAGS) $(AM_CPPFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS)
