#!/usr/bin/env bash

set -e
exeFunc(){
    cd mobilestereonet
    baseline=$1
    num_seq=$2
    CUDA_VISIBLE_DEVICES=0 python prediction.py --datapath /root/autodl-tmp/sscbench-kitti-360/sscbench-kitti/data_2d_raw/$num_seq \
    --testlist ./kitti360_filenames/$num_seq.txt --num_seq $num_seq --loadckpt ./MSNet3D_SF_DS_KITTI2015.ckpt --dataset kitti360 \
    --model MSNet3D --savepath "/root/autodl-tmp/sscbench-kitti-360/sscbench-kitti/msnet3d_depth" --baseline $baseline
    cd ..
}
seqs=("2013_05_28_drive_0000_sync" "2013_05_28_drive_0002_sync" "2013_05_28_drive_0003_sync" "2013_05_28_drive_0004_sync" "2013_05_28_drive_0005_sync" "2013_05_28_drive_0006_sync" "2013_05_28_drive_0007_sync" "2013_05_28_drive_0009_sync" "2013_05_28_drive_0010_sync")
for i in ${seqs[@]}
do
    exeFunc 331.5325566 $i     
done

