# FortisAVQA and MAVEN: a Benchmark Dataset and Debiasing Framework for Robust Multimodal Reasoning
This repository provides the dataset and implementation for our paper: **"FortisAVQA and MAVEN: a Benchmark Dataset and Debiasing Framework for Robust Multimodal Reasoning"**. This work is an improved and extended version of our [previously published paper](https://github.com/reml-group/MUSIC-AVQA-R) in NeurIPS 2024. *Compared to MUSIC-AVQA-R, we think the head/tail splitting in FortisAVQA is more rational.*

![Static Badge](https://img.shields.io/badge/python-3.10-blue)[![Static Badge](https://img.shields.io/badge/FortisAVQA-pdf-red)](https://arxiv.org/abs/2504.00487)[![Static Badge](https://img.shields.io/badge/MAVEN-Model-yellow)](https://huggingface.co/reml-group/MAVEN)[![Static Badge](https://img.shields.io/badge/FortisAVQA-Dataset-yellow)](https://huggingface.co/datasets/reml-group/FortisAVQA)[![Static Badge](https://img.shields.io/badge/MUSIC_AVQA_R-pdf-red)](https://openreview.net/pdf?id=twpPD9UMUN)


## Overview

Audio-Visual Question Answering (AVQA) is a challenging multimodal reasoning task requiring models to accurately answer natural language queries based on paired audio-video inputs. However, existing AVQA approaches often suffer from overfitting to dataset biases, leading to poor robustness. Moreover, current datasets may not effectively diagnose these methods.

To address these challenges, we introduce:

- **FortisAVQA**, a novel dataset designed for robustness evaluation, constructed in two stages:
  1. Rephrasing test split questions from the MUSIC-AVQA dataset to increase linguistic diversity.
  2. Introducing distribution shifts across question splits to enable a refined robustness evaluation across rare, frequent, and overall question distributions.
- **MAVEN (Multimodal Audio-Visual Epistemic Network)**, a robust generative AVQA model leveraging a multifaceted cycle collaborative debiasing strategy to mitigate bias learning.

## FortisAVQA
We introduce FortisAVQA, the first dataset designed to assess the robustness of AVQA models. Its construction involves two key processes: rephrasing and splitting. Rephrasing modifies questions from the test set of MUSIC-AVQA to enhance linguistic diversity, thereby mitigating the reliance of models on spurious correlations between key question terms and answers. Splitting entails the automatic and reasonable categorization of questions into frequent (head) and rare (tail) subsets, enabling a more comprehensive evaluation of model performance in both in-distribution and out-of-distribution scenarios.

1. You can download the original dataset, including videos and questions in [Videos](https://drive.google.com/drive/folders/1WAryZZE0srLIZG8VHl22uZ3tpbGHtsrQ) and [Questions](https://pan.baidu.com/s/1lJ1cPl4tnbgfIPS-UxbVZA)(password: AVQA). These are provided by the repository of MUSIC-AVQA.
2. Our proposed method, MAVEN, is capable of processing audio, video, and text inputs simultaneously. To facilitate this, we have extracted and separated the audio and video components from the original files. The processed data is available for download [here](). You can also process the original data by using the code located in `./MAVEN/data_tools/extract_audio_multiprocessing.py` .
3. The question annotation in our proposed dataset (FortisAVQA) is shown as follows. You can download the questions [here](https://pan.baidu.com/s/1mrLeoCrLn2AbdVxkCQ3hjg)(password: AVQA).
```json
[
    {
        "video_id": "00000823",
        "question_id": 2945,
        "_comment_question_id": "the id of questions",

        "type": "[\"Audio\", \"Comparative\"]",
        "_comment_type":  "[audio task, question type]",

        "question_content": "Is the clarinet louder than the acoustic_guitar",
        "anser": "yes",
        "_comment_anser": "the answer to questions",

        "split": "tail", 
        "_comment_split": "head or tail split",

        "x": 0.7958979489744872,
        "_comment_x": "k in the Equation (9) of our paper"
    },
    {
        "video_id": "00000823",
        "question_id": 2945,  
        "type": "[\"Audio\", \"Comparative\"]",
        "question_content": "When compared to the acoustic_guitar does the clarinet sound louder?",
        "anser": "yes",
        "split": "tail",
        "x": 0.7958979489744872
    }
]
```
4. The test split of FortisAVQA consists of 211,572 samples. To manage testing costs efficiently, we applied uniform sampling on the MUSIC-AVQA and FortisAVQA to create a representative subset specifically for evaluating large models. You can download the test splits of the mentioned two datasets [here](https://pan.baidu.com/s/1hCGZmf9z3cWy0EA7mb1Hvg) (password: AVQA). The details of this split are shown as follows. ![The statistics of FortisAVQA and MUSIC-AVQA](./imgs/data-stas.png)
## MAVEN
MAVEN first extracts unimodal embeddings using modality-specific encoders. It then fine-tunes a parameter-sharing generative model to learn multimodal fusion while capturing three unimodal biases, guided by four distinct prompts. Finally, a collaborative debiasing strategy is applied to amplify the disparity between the fusion logit and the bias logit. Simultaneously, a cycle-guidance mechanism is introduced to maintain consistency among bias logit. Our model is developed based on the code of [VITA](https://vita-home.github.io), which is a multimodal large language model adept at processing video, image, text, and audio modalities. However, this model can only process text (audio) and images simultaneously. In other words, it does not support the input of audio, text, and image modalities at the same time.

## Installation

### Requirements

Ensure you have Python 3.10 installed and 4+ Tesla A800/100, then install dependencies via:

```bash
pip install -r requirements.txt
```
### Setup

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/reml-group/fortisavqa.git
cd MAVEN
```

### Traning
1. Process the data according to the following format. If you do not perform instruction tuning, you may not need to execute this step. The code is located in `MAVEN/data_tools/preprocess_trainset.py`. You can download the [processed question](https://pan.baidu.com/s/1DRxxsQHc85YF20PuQSxr8Q) here(password: AVQA).

   ```bash
   [
       {
           "set": "music_avqa",
           "id": "00006427_1607",
           "conversations": [
               {
                   "from": "human",
                   "value": "<video>\n<audio>\nHow many instruments are sounding in the video?"
               },
               {
                   "from": "gpt",
                   "value": "There are two instruments sounding in the video."
               }
           ],
           "video": "video/00006427.mp4",
           "audio": [
               "00006427.wav"
           ]
       }
   ]
   ```
2. You should add its key-value pair to the `FolderDict` in `MAVEN/vita/config/dataset_config.py` to retrieve the video folder for data loading.

   ```bash
   AudioFolder = "/path/to/your/datahub/FortisAVQA"
   FolderDict = {
       "music_avqa": "/path/to/your/datahub/FortisAVQA"
   }
   ShareGPT4V = {"chat_path": "/path/to/your/FortisAVQA/fine_tune_vitar.json"}
   ```

3. Download the required weights: (1) [VITA checkpoint](https://huggingface.co/VITA-MLLM/VITA), (2) [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px), and (3) [VITA's pretrained audio encoder](https://huggingface.co/VITA-MLLM/VITA)

4. Replace the model paths in `MAVEN/script/train/finetuneTask_single_node.sh`.

   ```sh
       ...
       --model_name_or_path /path/to/your/VITA_ckpt \
       ...
       --vision_tower /path/to/your/InternViT-300M-448px \
       ...
       --audio_encoder /path/to/your/audio-encoder-2wh_zh_en_audioset_Mixtral-8x7B_New-base-tunning \
       ...
   ```

5. Execute the following commands to start the training process:

   ```bash
   export PYTHONPATH=./
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   OUTPUT_DIR=/path/to/your/OUTPUT_DIR
   bash script/train/finetuneTask_single_node.sh ${OUTPUT_DIR}
   ```

6. You can modify the `MCCD` training configuration in `MAVEN/vita/constants.py`: 

   ```python
   MCCD = {"flag": True,
           "lambda_multifaceted": 0.001,
           "lambda_cycle": 0.005,
           "multifaceted":{"lang": True, "audio": True, "video": True},
           "cycle": False
           }
   ```

### Evaluation

1. We have released the LoRA fine-tuned model weights and related configurations. You can download them from [here]() to reproduce our experimental results.

2. Batch Testing with JSON Input: 

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python avqa_test.py \
          --model_base	/path/to/your/VITA_ckpt/ \
          --test_json	/path/to/your/FortisAVQA-test.json  \
          --output_path	/path/to/your/output.json \
          --model_path	/path/to/your/lora_checkpoint \
          --video_dir	/path/to/your/FortisAVQA/video \
          --audio_dir	/path/to/your/FortisAVQA/audio
   ```

   **Parameters**

   - `--model_base`: Path to the pre-trained model base checkpoint.
   - `--test_json`: Path to the JSON file containing the test data.
   - `--output_path`: Path where the results will be saved.
   - `--model_path`: Path to the fine-tuned model checkpoint.
   - `--video_dir`: Directory containing the video files.
   - `--audio_dir`: Directory containing the audio files.

3. Single Question Inference: 

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python video_audio_demo.py \
      --model_base	/path/to/your/VITA_ckpt/ \
      --model_path /path/to/your/lora_checkpoint \
      --video_path /path/to/your/FortisAVQA/video/00000046.mp4 \
      --audio_path /path/to/your/FortisAVQA/audio/00000046.wav \
      --question "How many instruments are sounding in the video?"
   ```

   **Parameters**

   - `--model_base`: Path to the pre-trained model base checkpoint.
   - `--model_path`: Path to the fine-tuned model checkpoint.
   - `--video_path`: Path to the video file to be tested.
   - `--audio_path`: Path to the audio file corresponding to the video.
   - `--question`: The question that the model should answer based on the video and audio.

### Results

- Experimental results(%) on the MUSIC-AVQA test split:
![Experimental results(%) on the MUSIC-AVQA test split.](./imgs/MUSIC-AVQA-results.png)
- Experimental results(%) on the FortisAVQA test split:
![Experimental results(%) on thr FortisAVQA test split.](./imgs/Fortisavqa-results.png)

### Citation

If you find our dataset or code useful, please cite our work:

```
@article{ma2025fortisavqa,
  title={FortisAVQA and MAVEN: a Benchmark Dataset and Debiasing Framework for Robust Multimodal Reasoning},
  author={Ma, Jie and Gao, Zhitao and Chai, Qi and Liu, Jun and Wang, Pinghui and Tao, Jing and Su, Zhou},
  journal={arXiv preprint arXiv:2504.00487},
  year={2025}
}

@inproceedings{malook,
  title={Look, Listen, and Answer: Overcoming Biases for Audio-Visual Question Answering},
  author={Ma, Jie and Hu, Min and Wang, Pinghui and Sun, Wangchun and Song, Lingyun and Pei, Hongbin and Liu, Jun and Du, Youtian},
  booktitle={NeurIPS},
  year={2024}
}
```

## License

- **Code**: This project's code is licensed under the [MIT License](MIT_LICENSE).
- **Dataset**: The dataset provided in this repository is licensed under the [GNU General Public License v3.0](GNU_LICENSE).
