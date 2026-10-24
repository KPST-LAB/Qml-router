#!/bin/bash
#------------------------------------------------------------
#tab06_repeat_crossover.sh
#Purpose : Table 6, rows 1-2 (Section V-A3)
#------------------------------------------------------------
cd $REPO
OUT=$REPO/statcamp
mkdir -p $OUT
export LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64
echo "[camp] stopping services $(date +%H:%M)" | tee -a $OUT/progress.log
for PAT in "--port 8397" "--port 58006"; do
  P=$(pgrep -f "llama-server.*$PAT" | head -1); [ -n "$P" ] && kill $P
done
sleep 8

for i in $(seq 1 10); do
  echo "[camp] crossover run $i start $(date +%H:%M)" | tee -a $OUT/progress.log
  DENS_CTX=131072 python3 tests/bench_crossover.py > $OUT/xover_run$i.log 2>&1
  cp /tmp/crossover_results.json $OUT/xover_run$i.json 2>/dev/null
  echo "[camp] crossover run $i done $(date +%H:%M)" | tee -a $OUT/progress.log
done

pkill -x llama-server; sleep 6
rm -rf $REPO/store-mixstat; mkdir -p $REPO/store-mixstat
start_qml()
{
  cd $REPO && env LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64 \
  nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  --qml-session-store $REPO/store-mixstat/qml.db \
  --qml-lmdb-store $REPO/store-mixstat/qml_lmdb \
  </dev/null > $REPO/store-mixstat/server.log 2>&1 &
}
start_plain()
{
  cd $REPO && env LD_LIBRARY_PATH=$REPO/env/lib:/usr/local/cuda/lib64 \
  nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  </dev/null > $REPO/store-mixstat/server_plain.log 2>&1 &
}
wait_up()
{
    for k in $(seq 1 40); do env -u LD_LIBRARY_PATH curl -s -m 2 http://127.0.0.1:8397/health | grep -q '"status":"ok"' && return 0; sleep 3; done; return 1; }
start_qml; wait_up || { echo "[camp] build engine FAILED" | tee -a $OUT/progress.log; exit 1; }
echo "[camp] mixed build start $(date +%H:%M)" | tee -a $OUT/progress.log
python3 tests/bench_mixed_load.py 8397 build $REPO/store-mixstat > $OUT/mix_build.log 2>&1
echo "[camp] mixed build done $(date +%H:%M)" | tee -a $OUT/progress.log

for i in $(seq 1 5); do
  pkill -x llama-server; sleep 6; start_qml; wait_up
  echo "[camp] mixed session run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/bench_mixed_load.py 8397 session $REPO/store-mixstat 6 32000 150 $REPO/store-mixstat/server.log > $OUT/mix_session$i.log 2>&1
done
for i in $(seq 1 5); do
  pkill -x llama-server; sleep 6; start_plain; wait_up
  echo "[camp] mixed stateless run $i $(date +%H:%M)" | tee -a $OUT/progress.log
  python3 tests/bench_mixed_load.py 8397 stateless $REPO/store-mixstat 6 32000 150 $REPO/store-mixstat/server_plain.log > $OUT/mix_stateless$i.log 2>&1
done

pkill -x llama-server; sleep 8
cd $REPO
nohup third_party/llama.cpp/build/bin/llama-server -m models/bge-m3-f16.gguf --embedding -ngl 99 -c 8192 -ub 8192 -np 1 --port 58006 </dev/null > store-tplseed/server_embed.log 2>&1 &
sleep 3
env QML_ARCHIVE_RECALL=1 QML_EMBED_URL=http://127.0.0.1:58006 \
  nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 131072 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397 \
  --qml-session-store $REPO/store-tplseed/qml.db \
  --qml-lmdb-store $REPO/store-tplseed/qml_lmdb \
  </dev/null >> store-tplseed/server_8397.log 2>&1 &
wait_up && echo "[camp] services restored $(date +%H:%M)" | tee -a $OUT/progress.log
echo "[camp] CAMPAIGN-DONE $(date +%H:%M)" | tee -a $OUT/progress.log
