#!/bin/bash
#------------------------------------------------------------
#chain_qwen_h200.sh
#Purpose : Reviewer item 6: one non-Llama model family
#------------------------------------------------------------
set -u
DATA=${DATA:-$HOME/.DATA}; R=${REPO:-$DATA/qml-router}
LOG=$DATA/qml-bench/qwen_chain.log
mkdir -p $DATA/qml-bench; : > $LOG; rm -f $DATA/qml-bench/QWEN_DONE
run()
{
    echo "=== $* $(date) ===" >> $LOG; bash "$@" >> $LOG 2>&1; }
run $R/tests/h200/panel_adoption_qwen_h200.sh
run $R/tests/h200/panel_tab5_qwen_h200.sh stock 500 2
run $R/tests/h200/panel_tab5_qwen_h200.sh qml 500 2
echo "QWEN CHAIN ALL DONE $(date)" >> $LOG
touch $DATA/qml-bench/QWEN_DONE
