#!/bin/bash
#------------------------------------------------------------
#fig01_density.sh
#Purpose : Fig. 1(a)-(b) with the paper's conditions (artifact scripts/reviewer1_edit_14_step1_density_raw.sh):
#          N = 2, 4, 8, 16 users reconnecting 4 s apart, 2,650 filler items (about 17,650 tokens of history
#          per user), 131,072-token context, a fresh store; the median and every reconnect latency are kept
#------------------------------------------------------------
set -u
REPO=${REPO:-$HOME/qml-router}
cd $REPO
SRV=third_party/llama.cpp/build/bin/llama-server
MODEL=models/Llama-3.2-3B-Instruct-F16.gguf
PORT=8397
FLAGS="-ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $PORT"
STORE=$REPO/store-density

# The program starts, restarts and stops its own llama-server (the flags above, on its store
# QML_BENCH_STORE, here $STORE). The server this script used to start here was
# replaced by the program's first restart, so the script no longer starts or stops any server.
rm -rf $STORE; mkdir -p $STORE
QML_BENCH_SERVER=$REPO/$SRV QML_BENCH_MODEL=$REPO/$MODEL QML_BENCH_PORT=$PORT QML_BENCH_STORE=$STORE \
  DENS_LEVELS=2,4,8,16 DENS_H_REPS=2650 DENS_STAGGER=4 DENS_CTX=131072 \
  python3 tests/reconnect-restore/bench_density_curve.py all
cp /tmp/density_results.json fig01_density.json
