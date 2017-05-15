dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.57)
AC_INIT(libgdsync, 1.1, gpudirect@github.com)
AC_CONFIG_SRCDIR([src/mem.hpp])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE([1.10 foreign tar-ustar silent-rules subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_PROG_CC
AC_PROG_CXX
AC_GNU_SOURCE
AC_PROG_LN_S
AC_PROG_LIBTOOL

LT_INIT

AC_ARG_ENABLE(
    [test],
    [AC_HELP_STRING([--enable-test],
                   [Build test programs (default=no)])],
    [enable_test=$enableval],
    [enable_test=no])
AM_CONDITIONAL(TEST_ENABLE, test x$enable_test = xyes)

AC_ARG_ENABLE(
    [extended-memops],
    [AC_HELP_STRING([--enable-extended-memops],
                   [Enable support for CUDA 9.0 MemOps (default=no)])],
    [enable_ext_memops=$enableval],
    [enable_ext_memops=no])
AM_CONDITIONAL(EXT_MEMOPS, test x$enable_ext_memops = xyes)

AC_ARG_WITH([libibverbs],
    AC_HELP_STRING([--with-libibverbs], [ Set path to libibverbs installation ]))
if test x$with_libibverbs = x || test x$with_libibverbs = xno; then
    want_libibverbs=no
else
    want_libibverbs=yes
    if test -d $with_libibverbs; then
        CPPFLAGS="$CPPFLAGS -I$with_libibverbs/include"
        LDFLAGS="$LDFLAGS -L$with_libibverbs/lib -L$with_libibverbs/lib64"
    fi
fi

AC_ARG_WITH([gdrcopy],
    AC_HELP_STRING([--with-gdrcopy], [ Set path to gdrcopy installation ]))
if test x$with_gdrcopy = x || test x$with_gdrcopy = xno; then
    want_gdrcopy=no
else
    want_gdrcopy=yes
    if test -d $with_gdrcopy; then
        CPPFLAGS="$CPPFLAGS -I$with_gdrcopy/include"
        LDFLAGS="$LDFLAGS -L$with_gdrcopy/lib -L$with_gdrcopy/lib64"
    fi
fi

AC_ARG_WITH([gdstools],
    AC_HELP_STRING([--with-gdstools], [ Set path to gdstools installation ]))
if test x$with_gdstools = x || test x$with_gdstools = xno; then
    want_gdstools=no
    LIBGDSTOOLS=
else
    want_gdstools=yes
    if test -d $with_gdstools; then
        CPPFLAGS="$CPPFLAGS -I$with_gdstools/include -DUSE_PROF -DUSE_PERF"
        LDFLAGS="$LDFLAGS -L$with_gdstools/lib -L$with_gdstools/lib64"
    else
        echo "inexistent $with_gdstools directory"
    fi
    LIBGDSTOOLS=-lgdstools
    AC_SUBST(LIBGDSTOOLS)
fi

AC_ARG_WITH([mpi],
    AC_HELP_STRING([--with-mpi], [ Set path to mpi installation ]))
if test x$with_mpi = x || test x$with_mpi = xno; then
    # assuming system location
    mpi_home=/usr
    MPICC=$with_home/bin/mpicc
    MPICXX=$with_home/bin/mpic++
else
    if test -d $with_mpi; then
        mpi_home=$with_mpi
        MPICC=${mpi_home}/bin/mpicc
        MPICXX=${mpi_home}/bin/mpic++
        CPPFLAGS="$CPPFLAGS -I${mpi_home}/include"
        LDFLAGS="$LDFLAGS -L${mpi_home}/lib -L${mpi_home}/lib64"
    else
        echo "MPI dir does not exist"
    fi
fi

dnl Specify CUDA Location
AC_ARG_WITH(cuda-toolkit,
  AC_HELP_STRING([--with-cuda-toolkit=CUDATKDIR], [ Specify CUDA toolkit installation directory (default: /usr/local/cuda)]),
  [ cuda_home=${withval} ],
  [ cuda_home=/usr/local/cuda ]
)

AC_ARG_WITH(cuda-driver,
  AC_HELP_STRING([--with-cuda-driver=CUDADRVDIR], [ Specify CUDA driver installation directory (default: /usr/local/cuda)]),
  [ cuda_drv_home=${withval} ],
  [ cuda_drv_home=/usr/local/cuda ]
)

dnl Specify GPU Arch
AC_ARG_ENABLE(gpu-arch,
 AC_HELP_STRING([--enable-gpu-arch=arch], [ Set GPU arch: sm_20, sm_21, sm_30, sm_35, sm_50, sm_52 (default: sm_35)]),
 [ gpu_arch=${enableval} ],
 [ gpu_arch="sm_35" ]
)


dnl Checks for programs
AC_PROG_CC

dnl Checks for header files.
AC_HEADER_STDC

dnl Checks for Verbs support
AC_CHECK_LIB(ibverbs, ibv_get_device_list, [],
    AC_MSG_ERROR([ibv_get_device_list() not found.  libgdsync requires libibverbs.]))

AC_CHECK_LIB(ibverbs, ibv_exp_create_qp,
    AC_MSG_ERROR([ibv_exp_create_qp not found. libgdsync requires verbs extension support.]))

AC_CHECK_HEADER(infiniband/peer_ops.h, [],
    AC_MSG_ERROR([<infiniband/peer_ops.h> not found.  libgdsync requires verbs peer-direct support.]))
AC_HEADER_STDC

dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_CHECK_SIZEOF(long)

dnl Output Substitutions
AC_MSG_NOTICE([Setting CUDA_PATH = ${cuda_home} ])
AC_SUBST( CUDA_PATH, [${cuda_home}])

AC_MSG_NOTICE([Setting CUDA_DRV_PATH = ${cuda_drv_home} ])
AC_SUBST( CUDA_DRV_PATH, [${cuda_drv_home}])

AC_MSG_NOTICE([Setting GPU_ARCH = ${gpu_arch} ])
AC_SUBST( GPU_ARCH, [${gpu_arch}] )

AC_MSG_NOTICE([Setting MPI_PATH = ${mpi_home} ])
AC_SUBST( MPI_PATH, [${mpi_home} ])
AC_SUBST( MPICC, [${MPICC} ])
AC_SUBST( MPICXX, [${MPICXX} ])

CPPFLAGS="$CPPFLAGS -I$CUDA_DRV_PATH/include -I$CUDA_PATH/include"
LDFLAGS="$LDFLAGS -L$CUDA_DRV_PATH/lib64 -L$CUDA_DRV_PATH/lib -L$CUDA_PATH/lib64 -L$CUDA_PATH/lib"
NVCCFLAGS="$NVCCFLAGS"
CUDA_CFLAGS="$CUDA_CFLAGS -arch=${gpu_arch}"
CUDA_LDFLAGS="-L$CUDA_DRV_PATH/lib64 -L$CUDA_DRV_PATH/lib -L$CUDA_PATH/lib64 -L$CUDA_PATH/lib"
CUDA_LIBS="-lcuda -lcudart -lcufft"
NVCCFLAGS="$NVCCFLAGS $CUDA_CFLAGS $CUDA_LDFLAGS $CUDA_LIBS"
AC_SUBST(NVCC, [nvcc])
AC_SUBST(NVCCFLAGS)


dnl AC_CHECK_MEMBER([union CUstreamBatchMemOpParams_union.flushRemoteWrites],
dnl   [AC_SUBST( HAS_CUDA_MEMOP_FLUSH_REMOTE_WRITES, 1 )],
dnl   [AC_MSG_NOTICE([flushRemoteWrites is not defined])],
dnl   [[#include <cuda.h>]])

dnl Checks for CUDA >= 8.0
AC_CHECK_LIB(cuda, cuStreamBatchMemOp, [],
    AC_MSG_ERROR([cuStreamBatchMemOp() not found.  libgdsync requires CUDA 8.0 or later.]))

if test x$enable_ext_memops = xyes; then
   AC_CHECK_DECLS([CU_STREAM_MEM_OP_WRITE_MEMORY],    [], [], [[#include <cuda.h>]])
   AC_CHECK_DECLS([CU_STREAM_MEM_OP_MEMORY_BARRIER], [], [], [[#include <cuda.h>]])
   AC_CHECK_DECLS([CU_STREAM_MEM_OP_WRITE_VALUE_64], [], [], [[#include <cuda.h>]])
   AC_CHECK_DECLS([CU_STREAM_BATCH_MEM_OP_CONSISTENCY_WEAK], [], [], [[#include <cuda.h>]])
fi

AC_CONFIG_FILES([Makefile libgdsync.spec])
AC_OUTPUT
