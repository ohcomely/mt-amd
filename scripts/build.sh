#!/bin/bash

if [[ $# -eq 0 ]] ; then
  echo 'The path to a clean scratch directory needs to be provided!'
  exit 1
fi

mkdir -p $1
pushd $1
SCRATCH_DIR=$(pwd)
git clone --depth 1 -b v7.10.1 https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$SCRATCH_DIR -DSUITESPARSE_ENABLE_PROJECTS="amd" ..
cmake --build . -- -j8
cmake --install .
popd

pushd $(dirname "$0")/..
mkdir -p build && cd build
cmake -DSUITESPARSE_ROOT=$SCRATCH_DIR ..
cmake --build . -- -j8
popd
