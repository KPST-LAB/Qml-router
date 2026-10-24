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
SRV_PID=""   # process group of the server this script started (setsid makes the server the group leader)
# stop_group PID [KILL] : stop only the process group this script started with setsid, never by name;
# SIGTERM, then SIGKILL after 30 s; KILL sends SIGKILL at once
stop_group()
{
  local pid=${1:-} k
  [ -n "$pid" ] || return 0
  if [ "${2:-}" = KILL ]; then kill -9 -- -$pid 2>/dev/null; else kill -- -$pid 2>/dev/null; fi
  for k in $(seq 1 30); do kill -0 -- -$pid 2>/dev/null || return 0; sleep 1; done
  kill -9 -- -$pid 2>/dev/null
}
start_qml()
{
  cd $REPO
  setsid nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 -np 4 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  --qml-session-store $REPO/store-mixstat4/qml.db \
  --qml-lmdb-store $REPO/store-mixstat4/qml_lmdb \
  </dev/null > $REPO/store-mixstat4/server.log 2>&1 &
  SRV_PID=$!
}
start_plain()
{
  cd $REPO
  setsid nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 -np 4 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  </dev/null > $REPO/store-mixstat4/server_plain.log 2>&1 &
  SRV_PID=$!
}
build_store()
{
  rm -rf $REPO/store-mixstat4; mkdir -p $REPO/store-mixstat4
  start_qml; wait_up || return 1
  echo "[fix] build(np4) start $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/mixed-load/bench_mixed_load.py 8397 build $REPO/store-mixstat4 > $OUT/mix4_build.log 2>&1
  echo "[fix] build(np4) done $(date +%H:%M)" | tee -a $OUT/progress.log
}
wait_up()
{
    for k in $(seq 1 40); do env -u LD_LIBRARY_PATH curl -s -m 2 http://127.0.0.1:8397/health | grep -q "\"status\":\"ok\"" && return 0; sleep 3; done; return 1; }
stop_engines()
{
    stop_group $SRV_PID; SRV_PID=""; sleep 6; }

for i in $(seq 1 5); do
  stop_engines
  build_store || { echo "[fix] engine fail (build)" | tee -a $OUT/progress.log; exit 1; }
  stop_engines; start_qml; wait_up || { echo "[fix] engine fail" | tee -a $OUT/progress.log; exit 1; }
  echo "[fix] mixed session(np4) run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/mixed-load/bench_mixed_load.py 8397 session $REPO/store-mixstat4 6 32000 150 $REPO/store-mixstat4/server.log > $OUT/mix4_session$i.log 2>&1
done
for i in $(seq 1 5); do
  stop_engines
  build_store || { echo "[fix] engine fail (build)" | tee -a $OUT/progress.log; exit 1; }
  stop_engines; start_plain; wait_up || { echo "[fix] engine fail" | tee -a $OUT/progress.log; exit 1; }
  echo "[fix] mixed stateless(np4) run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/mixed-load/bench_mixed_load.py 8397 stateless $REPO/store-mixstat4 6 32000 150 $REPO/store-mixstat4/server_plain.log > $OUT/mix4_stateless$i.log 2>&1
done
stop_engines
# The script no longer restarts any service at the end. It used to start the embedding server
# (port 58006) and a recall-enabled llama-server on the protected store store-tplseed (port 8397).
echo "[fix] MIXED-FIX-DONE $(date +%H:%M)" | tee -a $OUT/progress.log
