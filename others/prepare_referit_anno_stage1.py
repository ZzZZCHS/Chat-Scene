import json
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict

dataset_name = "sr3d_merge"
referit_anno_root = "/root/scene-LLaMA/datasets/referit3d/annotations"

dataset_dir = os.path.join(referit_anno_root, "bert_tokenized")
train_split_file = os.path.join(referit_anno_root, "splits/scannetv2_train.txt")
val_split_file = os.path.join(referit_anno_root, "splits/scannetv2_val.txt")
# dataset_train = json.load(open(os.path.join(dataset_dir, "train.json"), "r"))
# dataset_val = json.load(open(os.path.join(dataset_dir, "val.json"), "r"))

sr3d_data = []
with open(os.path.join(dataset_dir, f"sr3d.jsonl"), "r") as f:
    for line in f:
        data = json.loads(line)
        for k, v in data.items():
            if type(v) == list:
                data[k] = frozenset(v)
        sr3d_data.append(data)

sr3d_plus_data = []
with open(os.path.join(dataset_dir, f"sr3d+.jsonl"), "r") as f:
    for line in f:
        data = json.loads(line)
        for k, v in data.items():
            if type(v) == list:
                data[k] = frozenset(v)
        sr3d_plus_data.append(data)

sr3d_data = {frozenset(d.items()) for d in sr3d_data}
sr3d_plus_data = {frozenset(d.items()) for d in sr3d_plus_data}
sr3d_merged_data = sr3d_data | sr3d_plus_data
print(len(sr3d_data), len(sr3d_plus_data), len(sr3d_merged_data))
#
# exit()

train_scenes = []
with open(train_split_file, "r") as f:
    for line in f.readlines():
        train_scenes.append(line.strip())

val_scenes = []
with open(val_split_file, "r") as f:
    for line in f.readlines():
        val_scenes.append(line.strip())

print(len(train_scenes), len(val_scenes))


train_data = []
val_data = []

correct_false = 0
other_data = 0

# with open(os.path.join(dataset_dir, f"{dataset_name}.jsonl"), "r") as f:
#     for line in f:
#         tmp_data = json.loads(line)
#         if not tmp_data["correct_guess"]:
#             correct_false += 1
#             continue
#         if tmp_data["scan_id"] in train_scenes:
#             train_data.append(tmp_data)
#         elif tmp_data["scan_id"] in val_scenes:
#             val_data.append(tmp_data)
#         else:
#             # print(tmp_data["scan_id"])
#             other_data += 1

for tmp_data in sr3d_merged_data:
    scan_id, target_id, utterance = None, None, None
    for k, v in tmp_data:
        if k == "scan_id":
            scan_id = v
        if k == "target_id":
            target_id = v
        if k == "utterance":
            utterance = v
    tmp_data = {
        "scan_id": scan_id,
        "target_id": target_id,
        "utterance": utterance
    }
    if tmp_data["scan_id"] in train_scenes:
        train_data.append(tmp_data)
    elif tmp_data["scan_id"] in val_scenes:
        val_data.append(tmp_data)
    else:
        # print(tmp_data["scan_id"])
        other_data += 1

print(len(train_data), len(val_data), correct_false, other_data)

captions = defaultdict(list)


def process(dataset_data):
    new_list = []
    for data in dataset_data:
        scene_id = data["scan_id"]
        obj_id = int(data["target_id"])
        feat_path = f"{scene_id}/{obj_id:03}.pt"
        caption = data["utterance"]
        new_data = {
            "pc_feat_path": feat_path,
            "caption": caption,
            "scene_id": scene_id,
            "obj_id": obj_id
        }
        captions[f"{scene_id}_{obj_id}"].append(caption)
        new_list.append(new_data)
    return new_list


output_train = process(train_data)
output_val = process(val_data)


with open(f"anno/{dataset_name}_train_stage1.json", "w") as f:
    json.dump(output_train, f)

with open(f"anno/{dataset_name}_val_stage1.json", "w") as f:
    json.dump(output_val, f)

with open(f"anno/{dataset_name}_captions.json", "w") as f:
    json.dump(captions, f)
