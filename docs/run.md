# Run and Eval
  
Train OccOoD with 4 GPUs 
```
./tools/dist_train.sh ./projects/configs/occood/occood-t.py 4
```

Eval OccOoD with 4 GPUs
```
./tools/dist_test.sh ./projects/configs/occood/occood-t.py ./path/to/ckpts.pth 4
```