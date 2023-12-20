import json
import os

dataset = "sr3d_merge"
val_split_path = "/root/scene-LLaMA/datasets/referit3d/annotations/splits/scannetv2_val.txt"
train_split_path = "/root/scene-LLaMA/datasets/referit3d/annotations/splits/scannetv2_train.txt"

caption_path = f"anno/{dataset}_captions_noun.json"

train_scene_ids = []
val_scene_ids = []

with open(train_split_path, "r") as f:
    for line in f.readlines():
        train_scene_ids.append(line.strip())

with open(val_split_path, "r") as f:
    for line in f.readlines():
        val_scene_ids.append(line.strip())

captions = json.load(open(caption_path, "r"))

train_captions = {}
val_captions = {}

for k, v in captions.items():
    scene_id = "_".join(k.split("_")[:-1])
    if scene_id in train_scene_ids:
        train_captions[k] = v
    if scene_id in val_scene_ids:
        val_captions[k] = v

output_train_path = f"anno/{dataset}_train_captions_noun.json"
output_val_path = f"anno/{dataset}_val_captions_noun.json"

with open(output_train_path, "w") as f:
    json.dump(train_captions, f)

with open(output_val_path, "w") as f:
    json.dump(val_captions, f)
