#!/bin/bash

if [[ $# -eq 0 ]] ; then
  echo 'The path to a clean scratch directory needs to be provided!'
  exit 1
fi

mkdir -p $1
date
$(dirname "$0")/download.sh $1
date
$(dirname "$0")/build.sh $1
date
$(dirname "$0")/scaling.sh $1
date
$(dirname "$0")/breakdown.sh $1
date
$(dirname "$0")/distribution.sh $1
date
$(dirname "$0")/tuning.sh $1
date