# Step-by-step installation instructions
OccOoD is developed based on the official SGN codebase and the installation follows similar steps.

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n occood python=3.8 -y
conda activate occood
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9

```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
pip install -v -e .
```

**f. Install other dependencies.**
```shell
pip install timm
pip install spconv-cu111==2.1.25
pip install torch-scatter==2.0.8
pip install tochmetrics>=0.9.0
```




