#!/usr/bin/env bash

set -e
exeFunc(){
    num_seq=$1
    CUDA_VISIBLE_DEVICES=0 python prediction.py --src_dir /root/autodl-tmp/stu_dataset/real/image \
    --dst_dir /root/autodl-tmp/stu_dataset/sequences_sql_depth/real \
    --dataset kitti
}

for i in {00}
do
    exeFunc $i
done
