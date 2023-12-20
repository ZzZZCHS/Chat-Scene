import json
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict

dataset_name = "nr3d"

dataset_dir = f"/root/autodl-tmp/scene-LLaMA/datasets/{dataset_name}"
if dataset_name == "scanrefer":
    dataset_train = json.load(open(os.path.join(dataset_dir, "ScanRefer_filtered_train.json"), "r"))
    dataset_val = json.load(open(os.path.join(dataset_dir, "ScanRefer_filtered_val.json"), "r"))
else:
    dataset_train = json.load(open(os.path.join(dataset_dir, "train.json"), "r"))
    dataset_val = json.load(open(os.path.join(dataset_dir, "val.json"), "r"))
captions = defaultdict(list)


def process(dataset_data):
    new_list = []
    for data in dataset_data:
        scene_id = data["scene_id"] if dataset_name == "scanrefer" else data["scan_id"]
        obj_id = int(data["object_id"]) if dataset_name == "scanrefer" else int(data["tgt_idx"])
        feat_path = f"{scene_id}/{obj_id:03}.pt"
        caption = data["description"] if dataset_name == "scanrefer" else data["query"]
        new_data = {
            "pc_feat_path": feat_path,
            "caption": caption,
            "scene_id": scene_id,
            "obj_id": obj_id
        }
        captions[f"{scene_id}_{obj_id}"].append(caption)
        new_list.append(new_data)
    return new_list


output_train = process(dataset_train)
output_val = process(dataset_val)


with open(f"anno/{dataset_name}_train_stage2.json", "w") as f:
    json.dump(output_train, f)

with open(f"anno/{dataset_name}_val_stage2.json", "w") as f:
    json.dump(output_val, f)

with open(f"anno/{dataset_name}_captions.json", "w") as f:
    json.dump(captions, f)
