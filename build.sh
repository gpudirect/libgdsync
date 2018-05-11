#!/bin/bash

[ ! -d config ] && mkdir -p config

[ ! -e configure ] && autoreconf -fv -i

[ ! -d build ] && mkdir build

cd build
echo "PREFIX=$PREFIX"
echo "CUDADRV=$CUDADRV"
echo "CUDATK=$CUDATK"
echo "CUDA=$CUDA"
echo "MPI_HOME=$MPI_HOME"

if [ ! -e Makefile ]; then
    echo "configuring..."
    EXTRA=
    if [ "x$CUDADRV" != "x" ]; then
        EXTRA+=" --with-cuda-driver=${CUDADRV}"
    fi
    if [ "x$CUDATK" != "x" ]; then
        EXTRA+=" --with-cuda-toolkit=$CUDATK"
    elif [ "x$CUDA" != "x" ]; then
        EXTRA+=" --with-cuda-toolkit=$CUDA"
    else
        echo "ERROR: CUDA toolkit path not passed"
        exit
    fi
    if [ "x$OFED" != "x" ]; then
        echo "picking OFED libibverbs from $OFED"
        EXTRA+=" --with-libibverbs=$OFED"
    else
        echo "WARNING: assuming IB Verbs is installed in /usr"
        EXTRA+=" --with-libibverbs=/usr"
    fi

    if [ "x$GDRCOPY" != "x" ]; then
        EXTRA+=" --with-gdrcopy=$GDRCOPY"
    else
        echo "WARNING: assuming GDRcopy is installed in /usr"
        EXTRA+=" --with-gdrcopy=/usr"
    fi

    EXTRA+=" --enable-test"
    EXTRA+=" --enable-extended-memops"
    #EXTRA="$EXTRA --with-gdstools=$PREFIX"

    ../configure \
        --prefix=$PREFIX \
        --with-mpi=$MPI_HOME \
        $EXTRA

fi

make V=1 clean all install
