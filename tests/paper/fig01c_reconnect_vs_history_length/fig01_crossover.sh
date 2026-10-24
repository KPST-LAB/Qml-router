#!/bin/bash
#------------------------------------------------------------
#fig01_crossover.sh
#Purpose : reconstructed 2026-09-07 from the usage text of bin/bench_crossover.py; the original one-line invocation was not kept as a file
#------------------------------------------------------------
set -u
REPO=${REPO:-$HOME/qml-router}
cd $REPO
SRV=third_party/llama.cpp/build/bin/llama-server
MODEL=models/Llama-3.2-3B-Instruct-F16.gguf
PORT=8397
FLAGS="-ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $PORT"
STORE=$REPO/store-xover
rm -rf $STORE; mkdir -p $STORE
# The program starts, restarts and stops its own llama-server (the flags above, on its store
# QML_BENCH_STORE, default $REPO/store-tplseed). The server this script used to start here was
# replaced by the program's first restart, so the script no longer starts or stops any server.
QML_BENCH_SERVER=$REPO/$SRV QML_BENCH_MODEL=$REPO/$MODEL QML_BENCH_PORT=$PORT \
  DENS_CTX=131072 python3 tests/reconnect-restore/bench_crossover.py
cp /tmp/crossover_results.json fig01_crossover.json
