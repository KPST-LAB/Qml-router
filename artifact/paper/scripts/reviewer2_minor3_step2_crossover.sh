#!/bin/bash
# TEST-LEDGER 110 (2026-10-19): Table 6 crossover procedure, test store and port 8597; then the
# Table 7 configuration-A run of ledger 109 back to back under the same GPU state.
W=$REPO/r2-3-check
gpu() { echo "== GPU $1 $(date +%T)"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
{
gpu before
rm -rf $W/store_xover; mkdir -p $W/store_xover/qml_lmdb
cd $REPO && DENS_CTX=131072 python3 $W/bench_crossover_r23.py
gpu after-crossover
} > $W/xover_r23.log 2>&1
( gpu before-tab7 ) > $W/gpu_tab7.log 2>&1
cp $W/results_r23.log $W/results_r23_first.log
bash $W/run_r23.sh
( gpu after-tab7 ) >> $W/gpu_tab7.log 2>&1
echo XOVER_R23_DONE >> $W/xover_r23.log
