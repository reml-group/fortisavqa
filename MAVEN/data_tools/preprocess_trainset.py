import json

with open("avqa-train.json", "r") as f: 
    original_data = json.load(f)

vita_data = []
for item in original_data:
    video_path = f"datahub/MAVEN/video/{item['video_id']}.mp4" 
    audio_path = f"datahub/MAVEN/audio/{item['video_id']}.wav" 
    
    vita_entry = {
        "set": "train_dataset",
        "id": f"{item['video_id']}_{item['question_id']}",
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n<audio>\n{item['question_content']}"
            },
            {
                "from": "gpt",
                "value": item["anser"]
            }
        ],
        "video": video_path,
        "audio": [audio_path]
    }
    
    vita_data.append(vita_entry)

with open("maven-train.json", "w") as f:
    json.dump(vita_data, f, indent=2)

