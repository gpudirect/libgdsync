#!/bin/bash

[ ! -d config ] && mkdir -p config

[ ! -e configure ] && ./autogen.sh

[ ! -d build ] && mkdir build

cd build

if [ ! -e Makefile ]; then
    echo "configuring..."
    ../configure \
        --prefix=$PREFIX \
        --with-libibverbs=$PREFIX \
        --with-cuda=$CUDA \
        --with-gdrcopy=$PREFIX \
        --with-mpi=$MPI_HOME \
        --enable-test

fi

make V=1 clean all install
