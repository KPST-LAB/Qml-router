#!/bin/bash
#------------------------------------------------------------
#tab07_hw_nvme.sh
#Purpose : NVMe host measurement
#------------------------------------------------------------
R=$REPO
B=$R/third_party/llama.cpp/build/bin
M=$R/models/Llama-3.2-3B-Instruct-F16.gguf
S=$R/store
LOG=$R/results_nvme.log
PORT=8397
QML_LIBS=$R/env/lib:/usr/local/cuda/lib64
exec > $LOG 2>&1

echo "=== NVME HOST MEASUREMENT $(date) ==="
echo "--- machine ---"
hostname
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader
lscpu | grep -E "^Model name|^CPU\(s\):"
free -g | sed -n 2p
grep PRETTY_NAME /etc/os-release
echo "--- storage ---"
df -h $R | tail -1
cat /sys/block/nvme0n1/device/model 2>/dev/null
echo "PCIe link: $(cat /sys/class/nvme/nvme0/device/current_link_speed 2>/dev/null) x$(cat /sys/class/nvme/nvme0/device/current_link_width 2>/dev/null)"
python3 --version

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

wait_up()
{
    local i=0
    while [ $i -lt 200 ]; do
        curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 'ok\|status' && return 0
        kill -0 $SRV_PID 2>/dev/null || return 2
        sleep 2; i=$((i + 1))
    done
    return 1
}

launch()
{
    local tag=$1; shift
    setsid env LD_LIBRARY_PATH=$QML_LIBS $B/llama-server -m $M -ngl 99 -fa on --port $PORT "$@" > $R/srv_$tag.log 2>&1 &
    SRV_PID=$!
    wait_up
}

stop()
{
    stop_group $SRV_PID KILL; SRV_PID=""; sleep 3; }

settle()
{
    local prev=0 i=0
    while [ $i -lt 120 ]; do
        cur=$(du -sb $S/qml_lmdb 2>/dev/null | cut -f1)
        [ "$cur" = "$prev" ] && [ "$cur" != "0" ] && [ -n "$cur" ] && return 0
        prev=$cur; sleep 3; i=$((i + 1))
    done
}

QF="-ctk q4_0 -ctv q4_0 --cache-ram 0"

for TOK in 8000 32000; do
    echo ""
    echo "########## Context ${TOK} tokens ##########"
    CTX=$((TOK + 2048))
    rm -rf $S; mkdir -p $S

    stop
    launch b$TOK -c $CTX -np 1 $QF --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb \
        || { echo "SERVER FAILED build $TOK"; continue; }
    grep -oE "kv_unified = .[a-z]+." $R/srv_b$TOK.log | head -1
    python3 $R/tests/reconnect-restore/bench_long_context.py $PORT build $S $TOK $R/srv_b$TOK.log
    settle
    du -sh $S/qml_lmdb
    stop

    launch c$TOK -c $CTX -np 1 $QF --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb \
        || { echo "SERVER FAILED cold $TOK"; continue; }
    echo "--- qml restore, page cache dropped (disk cold) ---"
    python3 $R/tests/reconnect-restore/bench_long_context.py $PORT restore $S $TOK $R/srv_c$TOK.log
    stop

    launch w$TOK -c $CTX -np 1 $QF --qml-session-store $S/qml.db --qml-lmdb-store $S/qml_lmdb \
        || { echo "SERVER FAILED warm $TOK"; continue; }
    echo "--- qml restore, page cache resident (CPU-side floor) ---"
    python3 $R/tests/reconnect-restore/bench_long_context.py $PORT restore_warm $S $TOK $R/srv_w$TOK.log
    stop

    launch p$TOK -c $CTX -np 1 $QF || { echo "SERVER FAILED stock $TOK"; continue; }
    echo "--- stock prefill (qml not applied) ---"
    python3 $R/tests/reconnect-restore/bench_long_context.py $PORT recompute $S $TOK
    stop

    echo "--- device sequential read on the blob file ---"
    python3 $R/tests/reconnect-restore/bench_long_context.py $PORT diskbw $S $TOK
done

echo ""
echo "=== NVME MEASUREMENT DONE ==="
