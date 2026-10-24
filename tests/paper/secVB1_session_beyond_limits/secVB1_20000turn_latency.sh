#!/bin/bash
#------------------------------------------------------------
#secVB1_20000turn_latency.sh
#Purpose : Section V-B1, per-question latency split
#------------------------------------------------------------
set -x
export WX_QUESTION_MODE=clone   # the paper's procedure: each question on its own copy (tests/multi-turn/README.md)
cd $REPO
MODEL=$DATA/models/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf
SRV=$DATA/llama.cpp/build/bin/llama-server
STORE=$REPO/store-wx20k
DS=weather-20k
REF=weather-20k-q8-direct-on
P=timer20k_progress.log
export WX_FACT_TURNS="10,500,2000,5000,8000,11000,14000,17000,19500"
export WX_N_TURNS=20000
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

prep_arm()
{
  python3 tests/multi-turn/common/bench_weather_longevity.py clone 58001 $STORE $DS $1 >> $P 2>&1
  python3 - <<PYEOF >> $P 2>&1
import sqlite3
def tag(sid):
    h = 1469598103934665603
    for b in sid.encode(): h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(h, "016x")
c = sqlite3.connect("$STORE/qml.db", timeout=120)
sids = [r[0] for r in c.execute("SELECT session_id FROM qml_session WHERE session_id LIKE ?", ("$DS-q%-$1",))]
c.execute("DELETE FROM qml_archive_fts WHERE session_id LIKE ?", ("$DS-q%-$1",))
for sid in sids:
    c.execute("INSERT INTO qml_archive_fts (content, tag, session_id, turn_pos, turn_end)"
              " SELECT content, ?, ?, turn_pos, turn_end FROM qml_archive_fts WHERE session_id=?",
              (tag(sid), sid, "$REF"))
c.commit()
n = c.execute("SELECT COUNT(*) FROM qml_archive_fts WHERE session_id=?", (sids[0],)).fetchone()[0]
print(f"[index] condition $1: {len(sids)} sessions, {n} documents each (copied)")
PYEOF
}

start_srv()
{
    # $1 = condition, $2 = recall env prefix ("" = none)
  setsid env $2 nohup $SRV -m $MODEL -ngl 99 -fa on -c 16384 -np 2 --port 58001 \
    --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb \
    </dev/null > srv_timer_$1.log 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 120); do curl -s -m 3 http://127.0.0.1:58001/health 2>/dev/null | grep -q ok && break; sleep 5; done
  python3 tests/multi-turn/common/bench_weather_longevity.py ask 58001 $STORE $DS $1 >> $P 2>&1
  stop_group $SRV_PID
  sleep 6
}

echo "== setup: clone + index copy for 2 conditions ==" > $P
for a in t0 t2; do prep_arm $a; done

echo "== t0: recall off (baseline) ==" >> $P
start_srv t0 "QML_ARCHIVE_RECALL=0"

echo "== t2: conf=0.2 (adopted default) ==" >> $P
start_srv t2 "QML_ARCHIVE_RECALL=1 QML_ARCHIVE_RECALL_TOPK=10 QML_ARCHIVE_RECALL_BUDGET=1024 QML_ARCHIVE_RECALL_PAIR_TURNS=1 QML_ARCHIVE_RECALL_CONF=0.2 QML_ARCHIVE_RECALL_NMIN=4"

echo "== hook timer summary (t2) ==" >> $P
grep -h "\[qml\]\[timer\]" srv_timer_t2.log >> $P
echo TIMER20K_DONE >> $P
