#!/bin/bash
#------------------------------------------------------------
#secVB2_restart_recovery.sh
#Purpose : reconstructed 2026-09-07 from secVB2_restart_recovery.log (the run script run_t4.sh lived in a session scratch directory and was not kept)
#------------------------------------------------------------
set -u
REPO=${REPO:-$HOME/qml-router}
cd $REPO
LL="third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf -ngl 99 -fa on -c 65536 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port 8397"
S=$REPO/store-t4; rm -rf $S; mkdir -p $S
SID="alice::t4"
FACT="The vault code is 0x5A5A1234."
start()
{
    nohup $LL --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb > srv_t4.log 2>&1 & for i in $(seq 1 60); do curl -sf http://127.0.0.1:8397/health >/dev/null && break; sleep 2; done; }
ask()
{
  python3 - "$1" "$SID" <<'EOF'
import json,sys,time,urllib.request
p,sid=sys.argv[1],sys.argv[2]
t=time.time()
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8397/completion",
    data=json.dumps({"prompt":p,"session_id":sid,"n_predict":16,"temperature":0,"cache_prompt":True}).encode(),
    headers={"Content-Type":"application/json"}))
d=json.load(r); print(int((time.time()-t)*1000),"ms |",d["content"].strip()[:60],"| prefill=",d["timings"]["prompt_n"])
EOF
}
start
echo "[turn 1]";  ask "Remember this: $FACT Reply OK."
sleep 3; echo "[saved bytes] $(stat -c %s $S/qml_lmdb/data.mdb)"
echo "[kill] SIGKILL during generation"
( ask "Write a long story about a lighthouse." >/dev/null 2>&1 & ); sleep 0.3; pkill -9 -x llama-server; sleep 2
echo "--- restart, then reconnect with only the new question ---"
start
echo "[recover]"; ask "What was the vault code?"
for p in $(pgrep -x llama-server); do kill $p; done
