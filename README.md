# Robustness Evaluation and Bias Mitigation in AVQA: Insights from Datasets and Models
This repository provides the dataset and implementation for our paper: **"Robustness Evaluation and Bias Mitigation in AVQA: Insights from Datasets and Models"**. This work is an improved and extended version of our [previously published paper](https://github.com/reml-group/MUSIC-AVQA-R) in NeurIPS 2024. *Compared to MUSIC-AVQA-R, we think the head/tail splitting in FortisAVQA is more rational.*

![Static Badge](https://img.shields.io/badge/python-3.10-blue)
[![Static Badge](https://img.shields.io/badge/FortisAVQA-pdf-red)](https://openreview.net/pdf?id=twpPD9UMUN)
[![Static Badge](https://img.shields.io/badge/MUSIC_AVQA_R-pdf-red)](https://openreview.net/pdf?id=twpPD9UMUN)


## Overview

Audio-Visual Question Answering (AVQA) is a challenging multimodal reasoning task requiring models to accurately answer natural language queries based on paired audio-video inputs. However, existing AVQA approaches often suffer from overfitting to dataset biases, leading to poor robustness. Moreover, current datasets may not effectively diagnose these methods.

To address these challenges, we introduce:

- **FortisAVQA**, a novel dataset designed for robustness evaluation, constructed in two stages:
  1. Rephrasing test split questions from the MUSIC-AVQA dataset to increase linguistic diversity.
  2. Introducing distribution shifts across question splits to enable a refined robustness evaluation across rare, frequent, and overall question distributions.
- **MAVEN (Multimodal Audio-Visual Epistemic Network)**, a robust generative AVQA model leveraging a multifaceted cycle collaborative debiasing strategy to mitigate bias learning.

## FortisAVQA
We introduce FortisAVQA, the first dataset designed to assess the robustness of AVQA models. Its construction involves two key processes: rephrasing and splitting. Rephrasing modifies questions from the test set of MUSIC-AVQA to enhance linguistic diversity, thereby mitigating the reliance of models on spurious correlations between key question terms and answers. Splitting entails the automatic and reasonable categorization of questions into frequent (head) and rare (tail) subsets, enabling a more comprehensive evaluation of model performance in both in-distribution and out-of-distribution scenarios.

1. You can download the original dataset including videos and questions in [here]().
2. The question format is shown as follows:
```json
[
    {
        "video_id": "00000823",
        "question_id": 2945,
        "_comment_question_id": the id of questions,

        "type": "[\"Audio\", \"Comparative\"]",
        "_comment_type":  "[audio task, question type]",

        "question_content": "Is the clarinet louder than the acoustic_guitar",
        "anser": "yes",
        "_comment_anser": the answer to questions,

        "split": "tail",  # head or tail split
        "_comment_split": head or tail split

        "x": 0.7958979489744872
        "_comment_x": k in the Equation (9) of our paper
    },
    {
        "video_id": "00000823",
        "question_id": 2945,  # the id of questions
        "type": "[\"Audio\", \"Comparative\"]",
        "question_content": "When compared to the acoustic_guitar does the clarinet sound louder?",
        "anser": "yes",
        "split": "tail",
        "x": 0.7958979489744872
    }
]
```
3. 
3 处理好的音频、视频文件；
4 利用均匀分布采样出来的FortisAVQA数据集，视频、音频、问题；

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
