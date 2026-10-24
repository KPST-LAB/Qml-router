#!/bin/bash
#------------------------------------------------------------
#tab07_hw_devserver.sh
#Purpose : AGX Orin measurement: restore(qml) vs prefill(stock) crossover
#------------------------------------------------------------
R=$REPO
B=$R/third_party/llama.cpp/build/bin
M=$R/models/Llama-3.2-3B-Instruct-F16.gguf
LOG=$R/tests/_dev3b/results_devsrv_3b.log
:
exec > $LOG 2>&1

echo "=== AGX ORIN MEASUREMENT $(date) ==="
echo "--- Device State ---"
echo "Dev Server (x86_64 + discrete GPU)"
sudo -n nvpmodel -q 2>/dev/null | head -2 || echo "power mode: no permission (measured in current mode)"
free -g | sed -n 2p
df -h $NVME_MOUNT | tail -1
nproc

wait_up()
{
    local i=0
    while [ $i -lt 200 ]; do
        if curl -s http://127.0.0.1:8395/health 2>/dev/null | grep -q 'ok\|status'; then
            return 0
        fi
        if ! pgrep -f '[l]lama-server' > /dev/null; then
            return 2
        fi
        sleep 3
        i=$((i + 1))
    done
    return 1
}

launch()
{
    local tag=$1
    shift
    $B/llama-server -m $M -ngl 99 -fa on --port 8395 "$@" > $R/tests/_dev3b/srv_$tag.log 2>&1 &
    wait_up
}

stop()
{
    pkill -9 -f '[l]lama-server'
    sleep 4
}

settle()
{
    local prev=0
    local i=0
    while [ $i -lt 120 ]; do
        cur=$(du -sb $R/tests/_dev3b/store/qml_lmdb 2>/dev/null | cut -f1)
        if [ "$cur" = "$prev" ] && [ "$cur" != "0" ] && [ -n "$cur" ]; then
            return 0
        fi
        prev=$cur
        sleep 5
        i=$((i + 1))
    done
}

for TOK in 8000 32000; do
    echo ""
    echo "########## Context ${TOK} tokens ##########"
    CTX=$((TOK + 2048))
    rm -rf $R/tests/_dev3b/store
    mkdir -p $R/tests/_dev3b/store

    stop
    launch q$TOK -c $CTX -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 \
        --qml-session-store $R/tests/_dev3b/store/qml.db --qml-lmdb-store $R/tests/_dev3b/store/qml_lmdb
    if [ $? -ne 0 ]; then
        echo "SERVER FAILED (qml build, $TOK)"
        continue
    fi
    grep -oE "kv_unified = .[a-z]+." $R/tests/_dev3b/srv_q$TOK.log | head -1
    timeout 3600 python3 $R/tests/bench_long_context.py 8395 build $R/tests/_dev3b/store $TOK $R/tests/_dev3b/srv_q$TOK.log
    settle
    du -sh $R/tests/_dev3b/store/qml_lmdb
    stop

    launch r$TOK -c $CTX -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 \
        --qml-session-store $R/tests/_dev3b/store/qml.db --qml-lmdb-store $R/tests/_dev3b/store/qml_lmdb
    if [ $? -ne 0 ]; then
        echo "SERVER FAILED (qml restore, $TOK)"
        continue
    fi
    echo "--- qml restore (after restart, disk cold) ---"
    timeout 3600 python3 $R/tests/bench_long_context.py 8395 restore $R/tests/_dev3b/store $TOK $R/tests/_dev3b/srv_r$TOK.log
    stop

    launch p$TOK -c $CTX -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0
    if [ $? -ne 0 ]; then
        echo "SERVER FAILED (stock, $TOK)"
        continue
    fi
    echo "--- stock prefill (qml not applied) ---"
    timeout 3600 python3 $R/tests/bench_long_context.py 8395 recompute $R/tests/_dev3b/store $TOK
    stop
done

echo ""
echo "=== ORIN MEASUREMENT DONE ==="
