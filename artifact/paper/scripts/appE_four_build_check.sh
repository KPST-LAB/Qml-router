#!/bin/bash
#------------------------------------------------------------
#appE_four_build_check.sh
#Purpose : Appendix A-E, four-build check
#------------------------------------------------------------
set -u
R=$REPO
A=$R/ablation
OUT=$A/results2
mkdir -p $OUT
: > $A/progress2
one () {
  if [ "$2" = "qml" ]; then
    env DENS_BIN=$A/$1/build/bin/llama-server DENS_LEVELS=8 DENS_STAGGER=4 \
      DENS_RESULTS=$OUT/${2}_$1_r$3.json \
      timeout 3600 python3 $A/bench_density_ablation.py build > $OUT/build_$1_r$3.log 2>&1
  fi
  env DENS_BIN=$A/$1/build/bin/llama-server DENS_LEVELS=8 DENS_STAGGER=4 \
    DENS_RESULTS=$OUT/${2}_$1_r$3.json \
    timeout 3600 python3 $A/bench_density_ablation.py $2 > $OUT/${2}_$1_r$3.log 2>&1
  echo "done $2 $1 r$3 $(date +%H:%M:%S)" >> $A/progress2
}
for r in 1 2 3; do one no-patch   qml   $r; done
for r in 1 2 3; do one with-patch qml   $r; done
for r in 1 2 3; do one with-patch stock $r; done
for r in 1 2 3; do one no-patch   stock $r; done
touch $A/DONE2
