#!/usr/bin/env bash

set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar.py --calib_dir  /root/autodl-tmp/kitti/dataset/sequences/$num_seq \
    --depth_dir /root/autodl-tmp/kitti/dataset/sequences_sql_depth/sequences/$num_seq \
    --save_dir /root/autodl-tmp/kitti/dataset/sequences_sql_lidar/sequences/$num_seq
}
exeFunc 08
for i in {00..06}
do
    exeFunc $i
done
