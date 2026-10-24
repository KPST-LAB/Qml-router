#!/bin/bash
#------------------------------------------------------------
#reviewer1_edit_14_step1_density_raw.sh
#Purpose : reviewer1_edit_14 step1 on the RTX 4090 host: Fig
#Record  : TEST-LEDGER 106
#------------------------------------------------------------
set -u
B=$NVME_MOUNT/qml-router; R=$B/ieee-rev1; L=$R/logs; OUT=$R/data; TAG=reviewer1_edit_14_step1_density_raw
S=$R/store-density-raw; mkdir -p $L $OUT $R/no-mps
exec > >(tee -a $L/$TAG.log) 2>&1
echo "=== $TAG start $(date -Is) ==="
md5sum $B/ablation/with-patch/build/bin/llama-server $B/models/Llama-3.2-3B-Instruct-F16.gguf $R/scripts/$TAG.py
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] || { echo "ABORT: another GPU process is running"; exit 2; }
FREE=$(df -BG --output=avail /nvme | tail -1 | tr -dc 0-9); echo "free on /nvme: ${FREE} GB"
[ "$FREE" -ge 18 ] || { echo "ABORT: need 18 GB free on /nvme (30 x 17.6K-token blobs)"; exit 3; }
rm -rf $S; mkdir -p $S
export DENS_LEVELS=2,4,8,16 DENS_H_REPS=2650 DENS_STAGGER=4 DENS_CTX=131072
export DENS_PORT=58178 DENS_STORE=$S DENS_SERVER=$B/ablation/with-patch/build/bin/llama-server
export DENS_MODEL=$B/models/Llama-3.2-3B-Instruct-F16.gguf DENS_SRVLOG=$L/${TAG}_srv.log
export DENS_ENV="CUDA_MPS_PIPE_DIRECTORY=$R/no-mps CUDA_MPS_LOG_DIRECTORY=$R/no-mps LD_LIBRARY_PATH=$B/env/lib:/usr/local/cuda/lib64"
export DENS_RESULTS=$OUT/$TAG.json
rm -f $DENS_RESULTS
python3 -u $R/scripts/$TAG.py all
echo "python rc=$? $(date -Is)"; df -h /nvme | tail -1; du -sh $S
echo "=== $TAG done $(date -Is) ==="
