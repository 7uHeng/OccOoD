# Out-of-Distribution Semantic Occupancy Prediction
## OccOoD
![visualization](https://github.com/7uHeng/OccOoD/blob/main/asserts/Visualization.png)
## Synthetic Anomaly Integration Pipeline
Collecting images and point cloud data of OoD objects is
challenging due to their rarity in real-world scenes and the
high cost of data collection and annotation. To address this,
we propose a Synthetic Anomaly Integration Pipeline that
generates synthetic anomalies under physical and environmental constraints, ensuring plausibility and challenge for
robust OoD detection model evaluation.
![Synthetic Anomaly Integration Pipeline](https://github.com/7uHeng/OccOoD/blob/main/asserts/Pipeline.png)
## Datasets
We applied the synthesis pipeline to the SemanticKITTI and SSCBench-KITTI-360 datasets,
resulting in the creation of two new synthetic datasets:
VAA-KITTI and VAA-KITTI-360. These synthesized
datasets extend the original label sets by introducing
anomaly labels, encompassing 26 distinct categories of
anomalies such as animals, furniture, and garbage bags.
.<div align=center><img src="https://github.com/7uHeng/OccOoD/blob/main/asserts/Distribution.png" width="523.5" height="488.25" /></div>
## Getting Started
* Refer to [install.md](https://github.com/7uHeng/OccOoD/blob/main/docs/install.md) to install the environment.
* Refer to [dataset.md](https://github.com/7uHeng/OccOoD/blob/main/docs/dataset.md) to prepare SemanticKITTI and KITTI360 dataset.
* Refer to [run.md](https://github.com/7uHeng/OccOoD/blob/main/docs/run.md) for training and evaluation.
## Updates
* [2025/08] Code is now publicly available! The OOD (Out-of-Distribution) dataset will be released upon acceptance of the paper.  Please stay tuned for updates.
* [2025/06] Init repository. The code and datasets will be made publicly available upon acceptance of the paper. Thank you for your interest in our work!
## Acknowledgement
This project is based on the following open-source projects. We thank their authors for making the source code publically available.
* [SGN](https://github.com/Jieqianyu/SGN)
* [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
* [OccRWKV](https://github.com/jmwang0117/OccRWKV)

