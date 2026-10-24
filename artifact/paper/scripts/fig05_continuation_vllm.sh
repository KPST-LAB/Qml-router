#!/bin/bash
#------------------------------------------------------------
#fig05_continuation_vllm.sh
#Purpose : reconstructed 2026-09-07 from fig05_continuation_vllm.log and the usage text of bin/bench_migration.py (run_mig_vllm.sh lived in a session scratch directory and was not kept); the vLLM command is the one recorded in the log
#------------------------------------------------------------
set -u
REPO=${REPO:-$HOME/qml-router}
cd $REPO
STORE=$REPO/store-mig; rm -rf $STORE; mkdir -p $STORE
LPORT=8397; VPORT=8331

nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 65536 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $LPORT \
  --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb > srv_mig_llama1.log 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$LPORT/health >/dev/null && break; sleep 2; done
python3 tests/bench_migration.py build_llama $LPORT $STORE
pkill -x llama-server; sleep 3

VLLM_PLUGINS="" PYTHONPATH=$REPO/hooks/vllm nohup python3 -m vllm.entrypoints.openai.api_server \
  --model unsloth/Llama-3.2-3B-Instruct --port $VPORT --max-model-len 10240 --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"QmlVllmConnector","kv_connector_module_path":"qml_vllm_connector","kv_role":"kv_both","kv_buffer_device":"cpu","kv_connector_extra_config":{"qml_sqlite_path":"'$STORE'/qml.db","qml_lmdb_path":"'$STORE'/qml_lmdb","qml_self_url":"http://127.0.0.1:'$VPORT'","qml_read_threads":"8"}}' \
  > srv_mig_vllm.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$VPORT/health >/dev/null && break; sleep 5; done
python3 tests/bench_migration.py continue_vllm $VPORT $STORE
pkill -f "vllm.entrypoints"; sleep 5

nohup third_party/llama.cpp/build/bin/llama-server -m models/Llama-3.2-3B-Instruct-F16.gguf \
  -ngl 99 -fa on -c 65536 --kv-unified -np 8 -ctk q4_0 -ctv q4_0 --cache-ram 0 --port $LPORT \
  --qml-session-store $STORE/qml.db --qml-lmdb-store $STORE/qml_lmdb > srv_mig_llama2.log 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:$LPORT/health >/dev/null && break; sleep 2; done
python3 tests/bench_migration.py continue_llama $LPORT $STORE
pkill -x llama-server
