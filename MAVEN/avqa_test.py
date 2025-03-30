import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
import json

from decord import VideoReader, cpu
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init
import pdb

def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_IMAGE_LENGTH,
    min_frames=4,
    image_resolution=384,
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    # speed up video decode via decord.

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [
                expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        patch_images = torch.stack(patch_images)
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process model and video paths.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--test_json", type=str, required=True, help="Path to the test JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the results")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None, help="Directory containing videos")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing audios")
    parser.add_argument("--model_type", type=str, default="mixtral-8x7b")
    parser.add_argument("--conv_mode", type=str, default="mixtral_two")
    args = parser.parse_args()

    conv_mode = args.conv_mode

    # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
    # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
    max_frames = MAX_IMAGE_LENGTH  # 100

    # The number of frames retained per second in the video.
    video_framerate = 1

    # Sampling Parameter
    temperature = 0.01
    top_p = None
    num_beams = 1

    disable_torch_init()

    # 加载模型
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.model_type
    )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor
    
    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor

    model.eval()

    # 加载测试集
    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    results = []

    correct_count = 0
    total_count = 0
    # 批量处理测试数据
    for idx, entry in enumerate(test_data):
        video_id = entry.get("video_id")
        question_content = entry.get("question_content")
        expected_answer = entry.get("anser")

        # 路径推断
        video_path = (
            os.path.join(args.video_dir, f"{video_id}.mp4") if args.video_dir else None
        )
        audio_path = (
            os.path.join(args.audio_dir, f"{video_id}.wav") if args.audio_dir else None
        )
        # if args.model_base == None:
        #     question_content += "Provide a brief answer as much as possible,If unable to answer, please answer: 'I don't know'."
        qs = question_content
        modality = "lang"
        
        if audio_path is not None:
            audio, audio_for_llm_lens = audio_processor.process(os.path.join(audio_path))
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audios = dict()
            audios["audios"] = audio.half().cuda()
            audios["lengths"] = audio_length.half().cuda()
        else:
            audio = torch.zeros(400, 80)
            audio_length = audio.shape[0]
            audio = torch.unsqueeze(audio, dim=0)
            audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
            audios = dict()
            audios["audios"] = audio.half().cuda()
            audios["lengths"] = audio_length.half().cuda()
            # audios = None

        if video_path and os.path.exists(video_path):
            video_frames, slice_len = _get_rawvideo_dec(
                video_path,
                image_processor,
                max_frames=MAX_IMAGE_LENGTH,
                video_framerate=1,
                image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
            )
            image_tensor = video_frames.half().cuda()
            qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs
            modality = "video"
        elif image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image, p_num = dynamic_preprocess(
                image, min_num=1, max_num=12, image_size=448, use_thumbnail=True
            )
            image_tensor = model.process_images(image, model.config).to(
                dtype=model.dtype, device="cuda"
            )
            qs = DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + qs
            modality = "image"
        else:
            image_tensor = torch.zeros((1, 3, 448, 448)).to(dtype=model.dtype, device="cuda")

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt(modality)
        # pdb.set_trace()
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        start_time = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        infer_time = time.time() - start_time

        output_ids = output_ids.sequences
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[
            0
        ].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()

        # 保存结果
        is_correct = expected_answer.strip().lower() in outputs.strip().lower()
        correct_count += is_correct
        total_count += 1

        results.append(
            {
                "video_id": video_id,
                "question_id": entry.get("question_id"),
                "question": question_content,
                "expected_answer": expected_answer,
                "predicted_answer": outputs,
                "is_correct": is_correct,
            }
        )

        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Processed {idx + 1}/{len(test_data)}: {outputs}")

    accuracy = correct_count / total_count * 100
    print(f"准确率: {accuracy:.2f}%")

    with open(args.output_path, "w") as f:
        json.dump({"results": results, "accuracy": accuracy}, f, indent=4)