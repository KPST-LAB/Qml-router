#!/bin/bash
#------------------------------------------------------------
#fig05_sglang_to_llama.sh
#Purpose : reconstructed 2026-09-07 from fig05_sglang_to_llama.log (run_sg37.sh lived in a session scratch directory and was not kept)
#------------------------------------------------------------
set -u
REPO=${REPO:-$HOME/qml-router}
cd $REPO
STORE=$REPO/store-sg37; rm -rf $STORE; mkdir -p $STORE
SPORT=8333; LPORT=8398; SID="alice::sg-1"
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

# SGLang with the qml-router storage backend, the command recorded in the paper's log
PYTHONPATH=$REPO/hooks/sglang setsid nohup python3 -m sglang.launch_server \
  --model-path unsloth/Llama-3.2-3B-Instruct --port $SPORT --context-length 8192 \
  --enable-hierarchical-cache --hicache-write-policy write_through --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{"module_path":"qml_sglang_storage","class_name":"QmlSglangStorage","backend_name":"qml","qml_sqlite_path":"'$STORE'/qml.db","qml_lmdb_path":"'$STORE'/qml_lmdb"}' \
  > srv_sg37.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$SPORT/health >/dev/null && break; sleep 5; done
python3 - <<EOF
import json,urllib.request
hist="".join(f"Note {i}: the weather log entry {i} is routine.\n" for i in range(220))
prompt=hist+"The vault code is 0x12345678. Remember it.\nUser: what is the vault code?\nAssistant:"
# the session identity goes in extra_key, which SGLang folds into its page hashes and the
# qml-router backend records as the session id
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$SPORT/v1/completions",
   data=json.dumps({"model":"unsloth/Llama-3.2-3B-Instruct","prompt":prompt,"max_tokens":16,"temperature":0,
                    "extra_key":"$SID"}).encode(),
   headers={"Content-Type":"application/json"}))
text=json.load(r)["choices"][0]["text"]
print("[1.build@SGLang] correct=%s" % ("OK" if "0x12345678" in text else "WRONG(" + text.strip()[:40] + ")"))
EOF
stop_group $SRV_PID; sleep 5
python3 - <<EOF
import sqlite3
c=sqlite3.connect("$STORE/qml.db")
print("qml_session:", c.execute("SELECT session_id,user_id,engine_type,n_tokens FROM qml_session").fetchall())
print("token_log rows:", c.execute("SELECT COUNT(*),MIN(pos),MAX(pos) FROM qml_token_log WHERE session_id='$SID'").fetchone())
EOF

setsid nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on --port $LPORT -c 8192 -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 \
  --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb > srv_sg37_llama.log 2>&1 &
SRV_PID=$!
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$LPORT/health >/dev/null && break; sleep 2; done
python3 - <<EOF
import json,time,urllib.request
t=time.time()
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$LPORT/completion",
   data=json.dumps({"prompt":"User: what is the vault code?\nAssistant:","session_id":"$SID","n_predict":16,"temperature":0}).encode(),
   headers={"Content-Type":"application/json"}))
d=json.load(r); ok="0x12345678" in d["content"]
print("[2.continue@llama.cpp] latency=%d ms correct=%s prefill=%s" % ((time.time()-t)*1000,
      "OK" if ok else "WRONG(" + d["content"].strip()[:40] + ")", format(d["timings"]["prompt_n"], ",")))
EOF
stop_group $SRV_PID
