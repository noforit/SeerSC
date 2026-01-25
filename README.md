# Seer Self-Consistency (SeerSC)

[![arXiv](https://img.shields.io/badge/arXiv-2511.09345-b31b1b.svg)](https://arxiv.org/abs/2511.09345)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for the paper **[Seer Self-Consistency: Advance Budget Estimation for Adaptive Test-Time Scaling](https://arxiv.org/abs/2511.09345)**.

## 📖 Introduction

**SeerSC** is a framework designed to optimize both **token consumption** and **latency** for adaptive test-time scaling in Large Language Models (LLMs).

Existing dynamic methods often suffer from high latency due to sequential requests. SeerSC overcomes this by mimicking the human cognitive process of **System 1 (Intuition)** and **System 2 (Reasoning)**:

1.  **System 1:** Rapidly estimates the answer entropy of a query.
2.  **Budget Estimation:** Determines the necessary computational budget in advance based on the entropy.
3.  **System 2:** Performs parallel sampling efficiently.


## 📂 Structure

* `configs/`: Configuration files for experiments.
* `dataset/`: Math datasets for evaluation.
* `evaluation/`: Scripts for evaluating model performance.
* `inference/`: Core inference logic.
* `method/`: Implementation of the SeerSC algorithm.

## 🚀 Quick Start

### 1. Installation

```bash
git clone [https://github.com/noforit/SeerSC.git](https://github.com/noforit/SeerSC.git)
cd SeerSC
# Install dependencies
pip install -r requirements.txt
```

### 2. Usage

```bash
bash run.sh
```

## 🎖️ Acknowledgements

We appreciate the open-source contribution of the [Qwen2.5-MATH](https://github.com/QwenLM/Qwen2.5-MATH) project. Our evaluation code is primarily derived from their repository.


## 🔗 Citation
If you find this work useful, please cite our paper:
```bibtex
@article{ji2025seer,
  title={Seer Self-Consistency: Advance Budget Estimation for Adaptive Test-Time Scaling},
  author={Ji, Shiyu and Wang, Yixuan and Liu, Yijun and Zhu, Qingfu and Che, Wanxiang},
  journal={arXiv preprint arXiv:2511.09345},
  year={2025}
}
```




