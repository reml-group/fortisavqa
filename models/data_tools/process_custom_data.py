import json

with open("/home/majie/datahub/MUSIC_AVQA_R/avqa-train.json", "r") as f:
    original_data = json.load(f)

# 转换为VITA格式
vita_data = []
for i, item in enumerate(original_data):
    video_path = f"{item['video_id']}.mp4"  # 待替换为实际路径
    audio_path = f"{item['video_id']}.wav"  # 待替换为实际路径
    
    vita_entry = {
        "set": "music_avqa",
        "id": f"{item['video_id']}_{item['question_id']}",
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n<audio>\n{item['question_content']}"
            },
            {
                "from": "gpt",
                "value": item["anser"] # 改写答案
            }
        ],
        "video": "video/" + video_path,
        "audio": [audio_path]
    }
    
    vita_data.append(vita_entry)

# 保存为VITA格式的JSON文件
with open("/home/majie/datahub/MUSIC_AVQA_R/fine_tune_vitar_1.json", "w") as f:
    json.dump(vita_data, f, indent=2)

