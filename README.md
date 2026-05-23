<div align="center">

# 🌍 OccOoD

### Out-of-Distribution 3D Semantic Occupancy Prediction

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<br/>

<img src="https://github.com/7uHeng/OccOoD/blob/main/asserts/Visualization.png" width="100%"/>

---

> **💡 What makes OOD evaluation hard?** Real-world anomalies are too rare to collect at scale.  
> **💡 Our answer:** A **Synthetic Anomaly Integration Pipeline** that inserts physically-plausible OOD objects into existing 3D occupancy data — producing **3 benchmark datasets** spanning **26 anomaly categories**.

---

## 🧪 Pipeline

| | |
|---|---|
| **Problem** | OOD object data is scarce — real anomalies are rare and annotation is expensive |
| **Solution** | Synthetic anomaly generation under physical & environmental constraints |
| **Result** | Plausible, challenging OOD evaluation data at scale |

<p align="center">
  <img src="https://github.com/7uHeng/OccOoD/blob/main/asserts/Pipeline.png" width="85%"/>
</p>

---

## 📦 Datasets & Downloads

<details open>
<summary><b>📥 3 benchmarks built on SemanticKITTI & KITTI-360</b> (click to collapse)</summary>
<br/>

| # | Dataset | Base | Anomaly Objects | Download |
|:--:|:--|:--|:--|:--|
| 1 | **VAA-KITTI** | SemanticKITTI | 26 categories | [⬇️ Google Drive](https://drive.google.com/file/d/1rMymO3W2BfuupIdan93DcpCv2JzI7r6l/view?usp=drive_link) |
| 2 | **VAA-KITTI-360** | KITTI-360 | 26 categories | [⬇️ Google Drive](https://drive.google.com/file/d/123yYb2P69JhJ5DwrwRlgZMMH7xobzfyN/view?usp=drive_link) |
| 3 | **VAA-STU** | STU | - | [⬇️ Google Drive](https://drive.google.com/file/d/19r2o652-58XDhvl1tM8bIO_39G6U709q/view?usp=sharing) |

> 📊 **Anomaly types:** animals · furniture · garbage bags · construction debris · vegetation overgrowth · vehicles · and more

</details>

<p align="center">
  <img src="https://github.com/7uHeng/OccOoD/blob/main/asserts/Distribution.png" width="55%"/>
</p>

---

## 🤖 Model Zoo

<p align="center">Pretrained checkpoints on 🤗 <a href="https://huggingface.co/7uheng/occood">Hugging Face</a> &nbsp;·&nbsp;</p>

| Dataset | Model | mIoU | Log | Weight |
|:--|:--|:--:|:--|:--|
| SemanticKITTI | OccOoD-T | 16.80 | [log](https://huggingface.co/7uheng/occood/blob/main/10sq_OccOoD-T16.8.log) | [pth](https://huggingface.co/7uheng/occood/blob/main/10sq_OccOoD-T16.8.pth) |
| SemanticKITTI | OccOoD-S | 13.79 | [log](https://huggingface.co/7uheng/occood/blob/main/7sq_OccOoD-S-spldepth_13.79.log) | [pth](https://huggingface.co/7uheng/occood/blob/main/7sq_OccOoD-S-spldepth_13.79.pth) |
| KITTI-360 | OccOoD-T | 18.38 | [log](https://huggingface.co/7uheng/occood/blob/main/kitti360OccOoD-T_18.38.log) | [pth](https://huggingface.co/7uheng/occood/blob/main/kitti360OccOoD-T_18.38.pth) |
| KITTI-360 | OccOoD-S | 12.47 | [log](https://huggingface.co/7uheng/occood/blob/main/occood-s-spl-kitti360-23epoch-12.47.log) | [pth](https://huggingface.co/7uheng/occood/blob/main/occood-s-spl-kitti360-23epoch-12.47.pth) |

---

## 🚀 Quick Start

```bash
# 1. Install environment
#    → docs/install.md

# 2. Download datasets
#    → docs/dataset.md

# 3. Train & evaluate
#    → docs/run.md
```

> 📖 Full guides: **[install.md](https://github.com/7uHeng/OccOoD/blob/main/docs/install.md)** · **[dataset.md](https://github.com/7uHeng/OccOoD/blob/main/docs/dataset.md)** · **[run.md](https://github.com/7uHeng/OccOoD/blob/main/docs/run.md)**

---

<div align="center">

### 📅 Timeline

| 2025.06 | 2025.08 |
|:--:|:--:|
| 🏗️ Repo initialized | 🚀 Code released |

</div>

---

<div align="center">
<sub><b>Built on</b>&nbsp; <a href="https://github.com/Jieqianyu/SGN">SGN</a> · <a href="https://github.com/open-mmlab/mmdetection3d">mmdet3d</a> · <a href="https://github.com/jmwang0117/OccRWKV">OccRWKV</a></sub>
</div>
