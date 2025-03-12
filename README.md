# Robustness Evaluation and Bias Mitigation in AVQA

## Overview

This repository provides the dataset and implementation for our paper:

**"Robustness Evaluation and Bias Mitigation in AVQA: Insights from Datasets and Models"**

![Static Badge](https://img.shields.io/badge/paper-pdf-red)


## Abstract

Audio-Visual Question Answering (AVQA) is a challenging multimodal reasoning task requiring models to accurately answer natural language queries based on paired audio-video inputs. However, existing AVQA approaches often suffer from overfitting to dataset biases, leading to poor robustness. Moreover, current datasets may not effectively diagnose these methods.

To address these challenges, we introduce:

- **FortisAVQA**, a novel dataset designed for robustness evaluation, constructed in two stages:
  1. Rephrasing test split questions from the MUSIC-AVQA dataset to increase linguistic diversity.
  2. Introducing distribution shifts across question splits to enable a refined robustness evaluation across rare, frequent, and overall question distributions.
- **MAVEN (Multimodal Audio-Visual Epistemic Network)**, a robust AVQA model leveraging a multifaceted cycle collaborative debiasing strategy to mitigate bias learning.

Our experiments show that MAVEN achieves state-of-the-art performance on FortisAVQA, with a **9.32%** improvement over previous approaches. Extensive ablation studies validate the effectiveness of our debiasing strategy and demonstrate the limited robustness of existing multimodal QA methods.

## Repository Structure

📂 MUSIC-AVQA-R/ ├── 📂 dataset/ # FortisAVQA dataset and processing scripts ├── 📂 models/ # MAVEN model implementation ├── 📂 scripts/ # Training and evaluation scripts ├── 📂 configs/ # Configuration files for experiments ├── 📜 requirements.txt # Python dependencies ├── 📜 README.md # Documentation └── 📜 LICENSE # License details


## Dataset

### FortisAVQA

FortisAVQA is built upon MUSIC-AVQA with enhanced robustness properties. The dataset consists of:

- **Diverse rephrased test questions** to prevent models from exploiting lexical biases.
- **Distribution shifts across question splits** (rare, frequent, and overall) for robustness evaluation.

Dataset and preprocessing scripts can be found in the [`dataset/`](./dataset) directory.

## Model: MAVEN

MAVEN is a robust AVQA model that incorporates:

- **Multifaceted cycle collaborative debiasing**, which counteracts dataset biases.
- **Plug-and-play compatibility** with baseline models across AVQA datasets.
- **State-of-the-art performance** on FortisAVQA, improving previous results by **9.32%**.

Implementation details are available in the [`models/`](./models) directory.

## Installation

### Requirements

Ensure you have Python 3.8+ installed, then install dependencies via:

```bash
pip install -r requirements.txt

Setup
Clone this repository and navigate to the project directory:
git clone https://github.com/reml-group/MUSIC-AVQA-R.git
cd MUSIC-AVQA-R

Training and Evaluation
Training MAVEN
Run the following command to train MAVEN:
python scripts/evaluate.py --dataset FortisAVQA


Evaluating MAVEN on FortisAVQA
python scripts/evaluate.py --dataset FortisAVQA

Results

Citation
If you find our dataset or code useful, please cite our work:
@article{ma2024look,
  title={Look, Listen, and Answer: Overcoming Biases for Audio-Visual Question Answering},
  author={Ma, Jie and Hu, Min and Wang, Pinghui and Sun, Wangchun and Song, Lingyun and Pei, Hongbin and Liu, Jun and Du, Youtian},
  journal={arXiv preprint arXiv:2404.12020},
  year={2024}
}

License
This project is licensed under the MIT License. See the LICENSE file for details.
