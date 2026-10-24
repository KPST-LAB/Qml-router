#!/bin/bash
#------------------------------------------------------------
#tab06_repeat_mixed.sh
#Purpose : Table 6, rows 3-4
#------------------------------------------------------------
exec 9>/tmp/mixed_fix.lock
flock -n 9 || { echo "[fix] another instance holds the lock -- exiting"; exit 0; }
cd $REPO
OUT=$REPO/statcamp
export LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64
start_qml()
{
  cd $REPO && nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 -np 4 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  --qml-session-store $REPO/store-mixstat4/qml.db \
  --qml-lmdb-store $REPO/store-mixstat4/qml_lmdb \
  </dev/null > $REPO/store-mixstat4/server.log 2>&1 &
}
start_plain()
{
  cd $REPO && nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 -np 4 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  </dev/null > $REPO/store-mixstat4/server_plain.log 2>&1 &
}
build_store()
{
  rm -rf $REPO/store-mixstat4; mkdir -p $REPO/store-mixstat4
  start_qml; wait_up || return 1
  echo "[fix] build(np4) start $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/bench_mixed_load.py 8397 build $REPO/store-mixstat4 > $OUT/mix4_build.log 2>&1
  echo "[fix] build(np4) done $(date +%H:%M)" | tee -a $OUT/progress.log
}
wait_up()
{
    for k in $(seq 1 40); do env -u LD_LIBRARY_PATH curl -s -m 2 http://127.0.0.1:8397/health | grep -q "\"status\":\"ok\"" && return 0; sleep 3; done; return 1; }
stop_engines()
{
    for P in $(pgrep -u jun -x llama-server); do kill $P 2>/dev/null; done; sleep 6; }

for i in $(seq 1 5); do
  stop_engines
  build_store || { echo "[fix] engine fail (build)" | tee -a $OUT/progress.log; exit 1; }
  stop_engines; start_qml; wait_up || { echo "[fix] engine fail" | tee -a $OUT/progress.log; exit 1; }
  echo "[fix] mixed session(np4) run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/bench_mixed_load.py 8397 session $REPO/store-mixstat4 6 32000 150 $REPO/store-mixstat4/server.log > $OUT/mix4_session$i.log 2>&1
done
for i in $(seq 1 5); do
  stop_engines
  build_store || { echo "[fix] engine fail (build)" | tee -a $OUT/progress.log; exit 1; }
  stop_engines; start_plain; wait_up || { echo "[fix] engine fail" | tee -a $OUT/progress.log; exit 1; }
  echo "[fix] mixed stateless(np4) run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/bench_mixed_load.py 8397 stateless $REPO/store-mixstat4 6 32000 150 $REPO/store-mixstat4/server_plain.log > $OUT/mix4_stateless$i.log 2>&1
done
stop_engines
cd $REPO
nohup third_party/llama.cpp/build/bin/llama-server -m models/bge-m3-f16.gguf --embedding -ngl 99 -c 8192 -ub 8192 -np 1 --port 58006 </dev/null > store-tplseed/server_embed.log 2>&1 &
sleep 3
env QML_ARCHIVE_RECALL=1 QML_EMBED_URL=http://127.0.0.1:58006 \
  nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  --qml-session-store $REPO/store-tplseed/qml.db \
  --qml-lmdb-store $REPO/store-tplseed/qml_lmdb \
  </dev/null >> store-tplseed/server_8397.log 2>&1 &
wait_up && echo "[fix] services restored $(date +%H:%M)" | tee -a $OUT/progress.log
echo "[fix] MIXED-FIX-DONE $(date +%H:%M)" | tee -a $OUT/progress.log
