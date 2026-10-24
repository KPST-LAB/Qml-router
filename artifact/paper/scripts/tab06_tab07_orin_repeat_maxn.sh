#!/bin/bash
#------------------------------------------------------------
#tab06_tab07_orin_repeat_maxn.sh
#Purpose : Table 6 row 5, Table 7 row 3 (Orin, MAXN)
#------------------------------------------------------------
set -u
R=$REPO
B=$R/third_party/llama.cpp/build/bin
M=$R/models/Llama-3.2-3B-Instruct-F16.gguf
OUT=$R/m4_results
mkdir -p $OUT
LOG=$OUT/m4_run.log
: > $LOG
: > $OUT/progress

envline()
{
  echo "[env] pmode=$(cat /var/lib/nvpmodel/status 2>/dev/null) cores=$(nproc) nvme_fault=$(sudo -n dmesg 2>/dev/null | grep -c 'completion polled')" >> $LOG
}
polled()
{
    sudo -n dmesg 2>/dev/null | grep -c 'completion polled'; }
wait_up()
{
  local i=0
  while [ $i -lt 300 ]; do
    curl -s http://127.0.0.1:8390/health 2>/dev/null | grep -q '"ok"' && return 0
    if ! pgrep -x llama-server >/dev/null; then
      echo "[wait_up] server process gone at iter $i" >> $LOG
      return 2
    fi
    sleep 3; i=$((i+1))
  done
  echo "[wait_up] health timeout (900s)" >> $LOG
  return 1
}
launch()
{
  local tag=$1; shift
  local try
  for try in 1 2; do
    $B/llama-server -m $M -ngl 99 -fa on --port 8390 "$@" > $OUT/srv_$tag.log 2>&1 &
    wait_up && return 0
    echo "[launch] $tag try$try failed; server log tail:" >> $LOG
    tail -4 $OUT/srv_$tag.log >> $LOG
    pkill -x llama-server 2>/dev/null; sleep 5
  done
  return 1
}
stop()
{
    pkill -x llama-server 2>/dev/null; sleep 4; }
settle()
{
  local prev=0 i=0
  while [ $i -lt 120 ]; do
    cur=$(du -sb $R/m4_store/qml_lmdb 2>/dev/null | cut -f1)
    [ "$cur" = "$prev" ] && [ -n "$cur" ] && [ "$cur" != "0" ] && return 0
    prev=$cur; sleep 5; i=$((i+1))
  done
}

run_mode()
{
  sudo -n nvpmodel -m $1 >> $LOG 2>&1; sleep 5
  echo "===== POWER MODE $2 =====" >> $LOG
  envline
  for TOK in 8000 32000; do
    CTX=$((TOK+2048))
    echo "--- [$2] ${TOK}tok build ---" >> $LOG
    rm -rf $R/m4_store; mkdir -p $R/m4_store
    stop
    launch b_${2}_$TOK -c $CTX -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 \
      --qml-session-store $R/m4_store/qml.db --qml-lmdb-store $R/m4_store/qml_lmdb || { echo BUILDFAIL >> $LOG; continue; }
    timeout 3600 python3 $R/tests/bench_long_context.py 8390 build $R/m4_store $TOK $OUT/srv_b_${2}_$TOK.log >> $LOG 2>&1
    settle; stop
    for rep in 1 2 3; do
      for try in 1 2 3; do
        envline
        P0=$(polled)
        echo "--- [$2] ${TOK}tok restore rep$rep try$try ---" >> $LOG
        launch r_${2}_${TOK}_$rep -c $CTX -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 \
          --qml-session-store $R/m4_store/qml.db --qml-lmdb-store $R/m4_store/qml_lmdb || { echo RFAIL >> $LOG; continue; }
        timeout 3600 python3 $R/tests/bench_long_context.py 8390 restore $R/m4_store $TOK $OUT/srv_r_${2}_${TOK}_$rep.log >> $LOG 2>&1
        stop
        P1=$(polled)
        if [ "$P1" = "$P0" ]; then
          echo "done $2 $TOK restore r$rep clean $(date +%H:%M:%S)" >> $OUT/progress
          break
        fi
        echo "DIRTY $2 $TOK restore r$rep try$try (polled $P0->$P1) -- retry" >> $OUT/progress
      done
    done
    for rep in 1 2 3; do
      for try in 1 2 3; do
        envline
        P0=$(polled)
        echo "--- [$2] ${TOK}tok recompute rep$rep try$try ---" >> $LOG
        launch p_${2}_${TOK}_$rep -c $CTX -np 1 -ctk q4_0 -ctv q4_0 --cache-ram 0 || { echo PFAIL >> $LOG; continue; }
        timeout 3600 python3 $R/tests/bench_long_context.py 8390 recompute $R/m4_store $TOK >> $LOG 2>&1
        stop
        P1=$(polled)
        if [ "$P1" = "$P0" ]; then
          echo "done $2 $TOK recompute r$rep clean $(date +%H:%M:%S)" >> $OUT/progress
          break
        fi
        echo "DIRTY $2 $TOK recompute r$rep try$try (polled $P0->$P1) -- retry" >> $OUT/progress
      done
    done
  done
}

run_mode 0 maxn
run_mode 2 mode30w
sudo -n nvpmodel -m 0 >> $LOG 2>&1
touch $OUT/DONE
