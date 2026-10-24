#!/bin/bash
#------------------------------------------------------------
#secVB1_20000turn_session.sh
#Purpose : Section V-B1, the 20,000-turn session (H200)
#------------------------------------------------------------
set -x
export WX_QUESTION_MODE=clone   # the paper's procedure: each question on its own copy (tests/multi-turn/README.md)
cd $REPO
# wait until no server answers on this script's port (other projects' servers on other ports do not matter)
while curl -s -m 3 -o /dev/null http://127.0.0.1:58001/health 2>/dev/null; do sleep 30; done
MODEL=$DATA/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
SRV=$DATA/llama.cpp/build/bin/llama-server
STORE=$REPO/store-wx20k
DS=weather-20k
P=wx_scale_progress.log
export WX_FACT_TURNS="10,500,2000,5000,8000,11000,14000,17000,19500"
export WX_N_TURNS=20000
rm -rf $STORE && mkdir -p $STORE
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

launch()
{
  if [ "$1" = recall ]; then
    QML_ARCHIVE_RECALL=1 QML_ARCHIVE_RECALL_TOPK=10 QML_ARCHIVE_RECALL_BUDGET=1024 \
      QML_ARCHIVE_RECALL_PAIR_TURNS=1 \
      setsid nohup $SRV -m $MODEL -ngl 99 -fa on -c 16384 -np 2 --port 58001 \
      --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb </dev/null > srv_wx20k_$1.log 2>&1 &
  else
    setsid nohup $SRV -m $MODEL -ngl 99 -fa on -c 16384 -np 2 --port 58001 \
      --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb </dev/null > srv_wx20k_$1.log 2>&1 &
  fi
  SRV_PID=$!
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:58001/health 2>/dev/null | grep -q ok && return 0; sleep 5; done
  return 1
}
stop_srv()
{
    stop_group $SRV_PID; sleep 6; }

echo "== phase 1: live feed 20,000 turns (recall off) ==" > $P
launch plain || { echo LAUNCH_FAIL >> $P; exit 1; }
export QML_ENGINE_PID=$SRV_PID
python3 tests/multi-turn/common/bench_weather_longevity.py feed 58001 $STORE $DS 20000 >> $P 2>&1
python3 tests/multi-turn/common/bench_weather_longevity.py clone 58001 $STORE $DS on >> $P 2>&1
python3 tests/multi-turn/common/bench_weather_longevity.py clone 58001 $STORE $DS off >> $P 2>&1
echo "== phase 2: ask condition OFF (control) ==" >> $P
python3 tests/multi-turn/common/bench_weather_longevity.py ask 58001 $STORE $DS off >> $P 2>&1
stop_srv
echo "== phase 3: ask condition ON (recall) ==" >> $P
launch recall || { echo LAUNCH_FAIL >> $P; exit 1; }
export QML_ENGINE_PID=$SRV_PID
python3 tests/multi-turn/common/bench_weather_longevity.py ask 58001 $STORE $DS on >> $P 2>&1
stop_srv
echo "== report ==" >> $P
python3 tests/multi-turn/common/bench_weather_longevity.py report 0 $STORE $DS >> $P 2>&1
python3 - <<PYEOF >> $P 2>&1
import sqlite3
c = sqlite3.connect("file:$STORE/qml.db?mode=ro", uri=True)
print("archive tokens:", c.execute("SELECT COUNT(*) FROM qml_token_log_archive WHERE session_id=?", ("weather-20k",)).fetchone()[0])
print("index documents(one clone):", c.execute("SELECT COUNT(*) FROM qml_archive_fts WHERE session_id=?", ("weather-20k-q0-direct-on",)).fetchone()[0])
PYEOF
echo WX_SCALE_DONE >> $P
