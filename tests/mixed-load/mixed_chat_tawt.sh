#!/bin/bash
#------------------------------------------------------------
#mixed_chat_tawt.sh
#Purpose : the mixed workload of Table 6 rows 3-4 (tab06_repeat_mixed.sh: 4 slots, 131,072-token
#          context, Q4_0 KV cache, six 32K sessions, 150 s, 5 runs per engine, a fresh store per run)
#          with the sessions in the Llama-3 chat format and the engine's window trim on
#          (QML_TRIM_WINDOW, QML_TRIM_EOT). In the paper's raw-text protocol a session fills its
#          32,768-token slot after 34 reconnects and the server then refuses it; here the qml server
#          trims the session at a user-turn boundary, and the stateless client keeps the same window.
#Usage   : REPO=<repository root> bash tests/mixed-load/mixed_chat_tawt.sh
#          the script stops only the server it started (its process group)
#------------------------------------------------------------
set -u
cd $REPO
OUT=$REPO/mixchat
ST=$REPO/store-mixchat
PORT=8397
WINDOW=32512          # slot 32,768 minus room for the question turn and 12 generated tokens
EOT_ID=128009         # Llama-3 <|eot_id|>
mkdir -p $OUT
SRV_PID=""
stop_group()
{
  local pid=${1:-} k
  [ -n "$pid" ] || return 0
  kill -- -$pid 2>/dev/null
  for k in $(seq 1 30); do kill -0 -- -$pid 2>/dev/null || return 0; sleep 1; done
  kill -9 -- -$pid 2>/dev/null
}
start_srv()   # start_srv LOG qml|plain
{
  local extra="" envs=""
  if [ "$2" = qml ]; then
    extra="--qml-session-store $ST/qml.db --qml-lmdb-store $ST/qml_lmdb"
    envs="QML_TRIM_WINDOW=$WINDOW QML_TRIM_EOT=$EOT_ID"
  fi
  setsid env $envs nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
    -ngl 99 -fa on -c 131072 -np 4 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $PORT $extra \
    </dev/null > $1 2>&1 &
  SRV_PID=$!
  for k in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q '"status":"ok"' && return 0; sleep 3; done
  return 1
}
stop_srv() { stop_group $SRV_PID; SRV_PID=""; sleep 6; }
build_store()
{
  rm -rf $ST; mkdir -p $ST
  start_srv $ST/server_build.log qml || return 1
  python3 tests/mixed-load/bench_mixed_load.py $PORT build-chat $ST > $OUT/build$1.log 2>&1
  stop_srv
}
if curl -s -m 2 -o /dev/null http://127.0.0.1:$PORT/health; then
  echo "port $PORT is in use by a server this script did not start -- exiting"; exit 1
fi
for i in $(seq 1 5); do
  build_store $i || { echo "build failed (run $i)"; exit 1; }
  start_srv $ST/server_session.log qml || { echo "server failed"; exit 1; }
  echo "[mixchat] session-chat run $i $(date +%H:%M)"
  python3 tests/mixed-load/bench_mixed_load.py $PORT session-chat $ST 6 32000 150 $ST/server_session.log > $OUT/session$i.log 2>&1
  stop_srv
  cp $ST/server_session.log $OUT/server_session$i.log
done
for i in $(seq 1 5); do
  build_store $i || { echo "build failed (run $i)"; exit 1; }
  start_srv $ST/server_plain.log plain || { echo "server failed"; exit 1; }
  echo "[mixchat] stateless-chat run $i $(date +%H:%M)"
  QML_TRIM_WINDOW=$WINDOW python3 tests/mixed-load/bench_mixed_load.py $PORT stateless-chat $ST 6 32000 150 $ST/server_plain.log > $OUT/stateless$i.log 2>&1
  stop_srv
done
echo "[mixchat] done $(date +%H:%M)"
