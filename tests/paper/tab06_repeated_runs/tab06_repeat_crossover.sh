#!/bin/bash
#------------------------------------------------------------
#tab06_repeat_crossover.sh
#Purpose : Table 6, rows 1-2 (Section V-A3)
#------------------------------------------------------------
cd $REPO
OUT=$REPO/statcamp
mkdir -p $OUT
export LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64
# The script no longer stops services it did not start (ports 8397 and 58006 used to be stopped by
# name here). Port 8397 must be free: bench_crossover.py starts and stops its own server there.
echo "[camp] checking port 8397 $(date +%H:%M)" | tee -a $OUT/progress.log
if env -u LD_LIBRARY_PATH curl -s -m 2 -o /dev/null http://127.0.0.1:8397/health; then
  echo "[camp] port 8397 is in use by a server this script did not start -- exiting" | tee -a $OUT/progress.log
  exit 1
fi
sleep 8

for i in $(seq 1 10); do
  echo "[camp] crossover run $i start $(date +%H:%M)" | tee -a $OUT/progress.log
  DENS_CTX=131072 python3 tests/reconnect-restore/bench_crossover.py > $OUT/xover_run$i.log 2>&1
  cp /tmp/crossover_results.json $OUT/xover_run$i.json 2>/dev/null
  echo "[camp] crossover run $i done $(date +%H:%M)" | tee -a $OUT/progress.log
done

sleep 6   # bench_crossover.py stopped its own server when it exited
rm -rf $REPO/store-mixstat; mkdir -p $REPO/store-mixstat
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
  setsid env LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64 \
  nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  --qml-session-store $REPO/store-mixstat/qml.db \
  --qml-lmdb-store $REPO/store-mixstat/qml_lmdb \
  </dev/null > $REPO/store-mixstat/server.log 2>&1 &
  SRV_PID=$!
}
start_plain()
{
  cd $REPO
  setsid env LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64 \
  nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  </dev/null > $REPO/store-mixstat/server_plain.log 2>&1 &
  SRV_PID=$!
}
wait_up()
{
    for k in $(seq 1 40); do env -u LD_LIBRARY_PATH curl -s -m 2 http://127.0.0.1:8397/health | grep -q '"status":"ok"' && return 0; sleep 3; done; return 1; }
start_qml; wait_up || { echo "[camp] build engine FAILED" | tee -a $OUT/progress.log; exit 1; }
echo "[camp] mixed build start $(date +%H:%M)" | tee -a $OUT/progress.log
python3 tests/mixed-load/bench_mixed_load.py 8397 build $REPO/store-mixstat > $OUT/mix_build.log 2>&1
echo "[camp] mixed build done $(date +%H:%M)" | tee -a $OUT/progress.log

for i in $(seq 1 5); do
  stop_group $SRV_PID; sleep 6; start_qml; wait_up
  echo "[camp] mixed session run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/mixed-load/bench_mixed_load.py 8397 session $REPO/store-mixstat 6 32000 150 $REPO/store-mixstat/server.log > $OUT/mix_session$i.log 2>&1
done
for i in $(seq 1 5); do
  stop_group $SRV_PID; sleep 6; start_plain; wait_up
  echo "[camp] mixed stateless run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/mixed-load/bench_mixed_load.py 8397 stateless $REPO/store-mixstat 6 32000 150 $REPO/store-mixstat/server_plain.log > $OUT/mix_stateless$i.log 2>&1
done

stop_group $SRV_PID; sleep 8
# The script no longer restarts any service at the end. It used to start the embedding server
# (port 58006) and a recall-enabled llama-server on the protected store store-tplseed (port 8397).
echo "[camp] CAMPAIGN-DONE $(date +%H:%M)" | tee -a $OUT/progress.log
