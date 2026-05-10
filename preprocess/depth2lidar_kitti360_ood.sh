#!/usr/bin/env bash

set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar_kitti360.py \
    --depth_dir /root/autodl-tmp/sscbench-kitti-360/sscbench-kitti/sql_depth/sequences/$num_seq \
    --save_dir /root/autodl-tmp/sscbench-kitti-360/sscbench-kitti/sql_pseudo_lidar/$num_seq
}

seqs=("2013_05_28_drive_0007_sync" "2013_05_28_drive_0009_sync" "2013_05_28_drive_0010_sync")

for i in ${seqs[@]}
do
    exeFunc $i
done
