# [ArXiv 25] Out-of-Distribution Semantic Occupancy Prediction
*Code is coming soon!*
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
## Updates
* [2025/06] Init repository. The code and datasets will be made publicly available upon acceptance of the paper. Thank you for your interest in our work!
## Acknowledgement
This project is based on the following open-source projects. We thank their authors for making the source code publically available.
* [SGN](https://github.com/Jieqianyu/SGN)
* [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
* [OccRWKV](https://github.com/jmwang0117/OccRWKV)
## Citation
If you find this work useful in your research, please cite the following BibTeX entry:

```bibtex
@article{zhang2025out,
  title={Out-of-Distribution Semantic Occupancy Prediction},
  author={Zhang, Yuheng and Duan, Mengfei and Peng, Kunyu and Wang, Yuhang and Liu, Ruiping and Teng, Fei and Luo, Kai and Li, Zhiyong and Yang, Kailun},
  journal={arXiv preprint arXiv:2506.21185},
  year={2025}
}
```
