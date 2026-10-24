#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_10_18_chain_4090.sh
#Purpose : runs TEST-LEDGER 103 then 104 on the RTX 4090 host, sequentially (one GPU)
#Record  : TEST-LEDGER 103
#------------------------------------------------------------
R=$NVME_MOUNT/qml-router/ieee-rev1
bash $R/scripts/reviewer1_edit_10_step1_shared_prefix_storage.sh
bash $R/scripts/reviewer1_edit_18_step2_burst_kill.sh
touch $R/logs/reviewer1_edit_10_18_chain_DONE
