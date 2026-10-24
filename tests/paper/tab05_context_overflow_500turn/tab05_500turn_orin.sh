#!/bin/bash
#------------------------------------------------------------
#tab05_500turn_orin.sh
#Purpose : Table 9 (tab:crossover) / TEST-LEDGER row 90 -- one engine+mode run
#------------------------------------------------------------
R=$REPO
DN=/media$BENCH/_x32k
ENGINE=$1; MODE=$2; TURNS=${3:-500}; NS=${4:-2}
TAG=$ENGINE-$MODE-t$TURNS
OUT=$DN/result_$TAG.log
ST=$DN/store-$TAG
mkdir -p $DN; : > $OUT; rm -rf $ST; mkdir -p $ST
echo "=== x32k $TAG (sessions $NS) $(date) ===" >> $OUT
df --output=source $DN | tail -1 >> $OUT

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
# stopall: SIGKILL to the process group of the server this script started (the engine and its
# child processes such as VLLM::EngineCore, spawn_main, resource_tracker), as the earlier
# name-based stop did with -9; then wait for the GPU memory to drop as before
stopall()
{
  stop_group $SRV_PID KILL; SRV_PID=""
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -z "$u" ] && break; [ "$u" -lt 3000 ] && break; sleep 3
  done
}
stopall

CLIENT_ENV=""
case $ENGINE in
llama)
  B=$R/third_party/llama.cpp/build/bin/llama-server
  QF=""
  [ "$MODE" = "qml" ] && QF="--qml-session-store $ST/qml.db --qml-lmdb-store $ST/qml_lmdb"
  RECALL=""
  [ "$MODE" = "qml" ] && RECALL="QML_ARCHIVE_RECALL=1"
  cd $R
  setsid env $RECALL LD_LIBRARY_PATH=$R/deps/lib:$LD_LIBRARY_PATH nohup $B -m models/Llama-3.2-3B-Instruct-F16.gguf -ngl 99 -fa on \
    -c 32768 --kv-unified -np $NS -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8408 $QF \
    > $DN/srv_$TAG.log 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:8408/health 2>/dev/null | grep -q ok && break; sleep 3; done
  API=llama; PORT=8408
  CLIENT_ENV="QML_ENGINE_PID=$SRV_PID"
  ;;
vllm)
  QF=""
  [ "$MODE" = "qml" ] && QF="--kv-transfer-config {\"kv_connector\":\"QmlVllmConnector\",\"kv_connector_module_path\":\"qml_vllm_connector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cpu\",\"kv_connector_extra_config\":{\"qml_sqlite_path\":\"$ST/qml.db\",\"qml_lmdb_path\":\"$ST/qml_lmdb\",\"qml_self_url\":\"http://127.0.0.1:8332\",\"qml_read_threads\":\"8\"}}"
  VLLM_PLUGINS="" PYTHONPATH=$R/hooks/vllm LD_LIBRARY_PATH=$R/deps/lib:$LD_LIBRARY_PATH \
    setsid nohup python3 -m vllm.entrypoints.openai.api_server \
    --model unsloth/Llama-3.2-3B-Instruct --port 8332 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 $QF > $DN/srv_$TAG.log 2>&1 &
  SRV_PID=$!
  timeout 900 bash -c 'until curl -s http://127.0.0.1:8332/v1/models 2>/dev/null | grep -q "\"id\""; do sleep 5; done'
  API=openai; PORT=8332
  ;;
sglang)
  CU13=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib
  QF=""
  if [ "$MODE" = "qml" ]; then
    QF="--enable-hierarchical-cache --hicache-write-policy write_through --hicache-storage-backend dynamic --hicache-storage-backend-extra-config {\"module_path\":\"qml_sglang_storage\",\"class_name\":\"QmlSglangStorage\",\"backend_name\":\"qml\",\"qml_sqlite_path\":\"$ST/qml.db\",\"qml_lmdb_path\":\"$ST/qml_lmdb\"}"
  fi
  LD_LIBRARY_PATH=$CU13:$R/deps/lib:$LD_LIBRARY_PATH PYTHONPATH=$R/hooks/sglang \
    setsid nohup python3 -m sglang.launch_server \
    --model-path unsloth/Llama-3.2-3B-Instruct --port 8333 --context-length 32768 $QF \
    > $DN/srv_$TAG.log 2>&1 &
  SRV_PID=$!
  timeout 1200 bash -c 'until curl -s http://127.0.0.1:8333/v1/models 2>/dev/null | grep -q "\"id\""; do sleep 5; done'
  API=openai; PORT=8333
  CLIENT_ENV="QML_SGL_EXTRA=1"
  ;;
esac

if ! curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && \
   ! curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "[$TAG] SERVER FAILED" >> $OUT; tail -5 $DN/srv_$TAG.log >> $OUT; stopall; exit 1
fi
env $CLIENT_ENV python3 $R/tests/multi-turn/continuous-session/bench_tab5_crossover_32k.py $PORT $API $MODE $TAG $NS $TURNS 306 $ST >> $OUT 2>&1
echo "[$TAG] store: $(du -sh $ST 2>/dev/null | cut -f1)" >> $OUT
stopall
echo "X32K DONE $TAG $(date)" >> $OUT
