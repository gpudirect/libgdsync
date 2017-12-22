#!/bin/bash

[ ! -d config ] && mkdir -p config

[ ! -e configure ] && ./autogen.sh

[ ! -d build ] && mkdir build

cd build
echo "CUDADRV=$CUDADRV"
if [ ! -e Makefile ]; then
    echo "configuring..."
    WITHCUDADRV=
    if [ "x$CUDADRV" != "x" ]; then
        WITHCUDADRV="--with-cuda-driver=${CUDADRV}"
    fi

    ../configure \
        --prefix=$PREFIX \
        --with-libibverbs=$PREFIX \
        $WITHCUDADRV \
        --with-cuda-toolkit=$CUDA \
        --with-gdrcopy=$PREFIX \
        --with-mpi=$MPI_HOME \
        --enable-test

fi

make V=1 clean all install
