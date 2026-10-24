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
nohup $SRV -m $MODEL $FLAGS --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb > srv_xover.log 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
DENS_CTX=131072 python3 tests/bench_crossover.py
cp /tmp/crossover_results.json fig01_crossover.json
for p in $(pgrep -x llama-server); do kill $p; done
