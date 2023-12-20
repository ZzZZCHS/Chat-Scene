import json
import glob

split = "train"

annos = []
for file_name in glob.glob(f"annotations/scene_dataset/gpt_generation/{split}/*.json"):
    scene_id = file_name.split("/")[-1][:-5]
    obj_id = 0
    prompt = "Provide a comprehensive description of the entire scene."
    caption = json.load(open(file_name, "r"))
    annos.append({
        "scene_id": scene_id,
        "obj_id": obj_id,
        "prompt": prompt,
        "caption": caption
    })

with open(f"annotations/scene_dataset_{split}_stage2.json", "w") as f:
    json.dump(annos, f, indent=4)
