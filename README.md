# [ArXiv 25] Out-of-Distribution Semantic Occupancy Prediction
3D Semantic Occupancy Prediction is crucial for autonomous driving, providing a dense, semantically rich environmental representation. However, existing methods focus on in-distribution scenes, making them susceptible to Out-of-Distribution (OoD) objects and long-tail distributions, which increases the risk of undetected anomalies and misinterpretations, posing safety hazards. To address these challenges, we introduce Out-of-Distribution Semantic Occupancy Prediction, targeting OoD detection in 3D voxel space. To fill the gaps in the dataset, we propose a Synthetic Anomaly Integration Pipeline that injects synthetic anomalies while preserving realistic spatial and occlusion patterns, enabling the creation of two datasets:
VAA-KITTI and VAA-KITTI-360. We introduce OccOoD, a novel framework integrating OoD detection into 3D semantic occupancy prediction, with Voxel-BEV Progressive Fusion (VBPF) leveraging an RWKV-based branch to enhance
OoD detection via geometry-semantic fusion.

*Code is coming soon!*
## OccOoD
![visualization](https://github.com/7uHeng/OccOoD/blob/main/asserts/Visualization.png)
## Synthetic Anomaly Integration Pipeline
![Synthetic Anomaly Integration Pipeline](https://github.com/7uHeng/OccOoD/blob/main/asserts/Pipeline.png)
## Datasets
.<div align=center><img src="https://github.com/7uHeng/OccOoD/blob/main/asserts/Distribution.png" width="698" height="651" /></div>
