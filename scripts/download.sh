#!/bin/bash

if [[ $# -eq 0 ]] ; then
  echo 'The path to a clean scratch directory needs to be provided!'
  exit 1
fi

mkdir -p $1/dataset
pushd $1/dataset

wget https://suitesparse-collection-website.herokuapp.com/MM/ATandT/pre2.tar.gz
tar xvzf pre2.tar.gz && rm pre2.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/ND/nd24k.tar.gz
tar xvzf nd24k.tar.gz && rm nd24k.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz
tar xvzf ldoor.tar.gz && rm ldoor.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV3real.tar.gz
tar xvzf dielFilterV3real.tar.gz && rm dielFilterV3real.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Flan_1565.tar.gz
tar xvzf Flan_1565.tar.gz && rm Flan_1565.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/HV15R.tar.gz
tar xvzf HV15R.tar.gz && rm HV15R.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Queen_4147.tar.gz
tar xvzf Queen_4147.tar.gz && rm Queen_4147.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/VLSI/stokes.tar.gz
tar xvzf stokes.tar.gz && rm stokes.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt240.tar.gz
tar xvzf nlpkkt240.tar.gz && rm nlpkkt240.tar.gz

popd 