#!/bin/bash
#------------------------------------------------------------
#secVB1_20000turn_session.sh
#Purpose : Section V-B1, the 20,000-turn session (H200)
#------------------------------------------------------------
set -x
cd $REPO
while pgrep -x llama-server >/dev/null; do sleep 30; done
MODEL=$DATA/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
SRV=$DATA/llama.cpp/build/bin/llama-server
STORE=$REPO/store-wx20k
DS=weather-20k
P=wx_scale_progress.log
export WX_FACT_TURNS="10,500,2000,5000,8000,11000,14000,17000,19500"
export WX_N_TURNS=20000
rm -rf $STORE && mkdir -p $STORE

launch()
{
  if [ "$1" = recall ]; then
    QML_ARCHIVE_RECALL=1 QML_ARCHIVE_RECALL_TOPK=10 QML_ARCHIVE_RECALL_BUDGET=1024 \
      QML_ARCHIVE_RECALL_PAIR_TURNS=1 \
      nohup $SRV -m $MODEL -ngl 99 -fa on -c 16384 -np 2 --port 58001 \
      --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb </dev/null > srv_wx20k_$1.log 2>&1 &
  else
    nohup $SRV -m $MODEL -ngl 99 -fa on -c 16384 -np 2 --port 58001 \
      --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb </dev/null > srv_wx20k_$1.log 2>&1 &
  fi
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:58001/health 2>/dev/null | grep -q ok && return 0; sleep 5; done
  return 1
}
stop_srv()
{
    for p in $(pgrep -x llama-server); do kill $p; done; sleep 6; }

echo "== phase 1: live feed 20,000 turns (recall off) ==" > $P
launch plain || { echo LAUNCH_FAIL >> $P; exit 1; }
export QML_ENGINE_PID=$(pgrep -x llama-server | head -1)
python3 tests/bench_weather_longevity.py feed 58001 $STORE $DS 20000 >> $P 2>&1
python3 tests/bench_weather_longevity.py clone 58001 $STORE $DS on >> $P 2>&1
python3 tests/bench_weather_longevity.py clone 58001 $STORE $DS off >> $P 2>&1
echo "== phase 2: ask condition OFF (control) ==" >> $P
python3 tests/bench_weather_longevity.py ask 58001 $STORE $DS off >> $P 2>&1
stop_srv
echo "== phase 3: ask condition ON (recall) ==" >> $P
launch recall || { echo LAUNCH_FAIL >> $P; exit 1; }
export QML_ENGINE_PID=$(pgrep -x llama-server | head -1)
python3 tests/bench_weather_longevity.py ask 58001 $STORE $DS on >> $P 2>&1
stop_srv
echo "== report ==" >> $P
python3 tests/bench_weather_longevity.py report 0 $STORE $DS >> $P 2>&1
python3 - <<PYEOF >> $P 2>&1
import sqlite3
c = sqlite3.connect("file:$STORE/qml.db?mode=ro", uri=True)
print("archive tokens:", c.execute("SELECT COUNT(*) FROM qml_token_log_archive WHERE session_id=?", ("weather-20k",)).fetchone()[0])
print("index documents(one clone):", c.execute("SELECT COUNT(*) FROM qml_archive_fts WHERE session_id=?", ("weather-20k-q0-direct-on",)).fetchone()[0])
PYEOF
echo WX_SCALE_DONE >> $P
