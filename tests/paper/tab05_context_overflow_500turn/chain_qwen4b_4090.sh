#!/bin/bash
# Size-matched non-Llama check on the RTX 4090: Qwen3-4B F16, llama.cpp only,
# 500-turn overflow run (Table 5 protocol), stock then qml. Run inside the
# container used for Table 5 (store on NVMe). The adoption-cost panel
# (tests/paper/tab04_adoption_cost/panel_adoption_qwen4b_4090.sh) runs separately in the Table 4
# container environment before this chain, never concurrently (one GPU).
set -u
R=$REPO
DN=$BENCH/_x32k_qwen4b
LOG=$DN/chain.log
mkdir -p $DN; : > $LOG; rm -f $DN/QWEN4B_DONE
run(){ echo "=== $* $(date) ===" >> $LOG; bash "$@" >> $LOG 2>&1; }
run $R/tests/paper/tab05_context_overflow_500turn/panel_tab5_qwen4b_4090.sh stock 500 2
run $R/tests/paper/tab05_context_overflow_500turn/panel_tab5_qwen4b_4090.sh qml 500 2
echo "QWEN4B CHAIN ALL DONE $(date)" >> $LOG
touch $DN/QWEN4B_DONE
