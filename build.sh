#!/bin/bash

[ ! -d config ] && mkdir -p config

[ ! -e configure ] && autoreconf -fv -i

[ ! -d build ] && mkdir build

cd build
echo "PREFIX=$PREFIX"
echo "CUDADRV=$CUDADRV"
echo "CUDA=$CUDA"
echo "MPI_HOME=$MPI_HOME"

if [ ! -e Makefile ]; then
    echo "configuring..."
    EXTRA=
    if [ "x$CUDADRV" != "x" ]; then
        EXTRA="$EXTRA --with-cuda-driver=${CUDADRV}"
    fi
    EXTRA="$EXTRA --enable-test"
    EXTRA="$EXTRA --enable-extended-memops"
    #EXTRA="$EXTRA --with-gdstools=$PREFIX"

    ../configure \
        --prefix=$PREFIX \
        --with-libibverbs=$PREFIX \
        --with-cuda-toolkit=$CUDA \
        --with-gdrcopy=$PREFIX \
        --with-mpi=$MPI_HOME \
        $EXTRA

fi

make V=1 clean all install
