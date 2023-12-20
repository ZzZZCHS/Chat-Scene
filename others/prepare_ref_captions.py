import json
import os
import random

val_anno = json.load(open("anno/scanrefer_val_stage2.json", "r"))
ref_captions = json.load(open("anno/scanrefer_captions.json", "r"))
train_convs = json.load(open("anno/scanrefer_train_conversation.json", "r"))
output_anno = []

for k, v in train_convs.items():
    scene_id = "_".join(k.split("_")[:-1])
    obj_id = int(k.split("_")[-1])
    if scene_id[:7] != "scene00" or len(v) == 0:
        continue
    qa = random.choice(v)
    output_anno.append({
        "scene_id": scene_id,
        "obj_id": obj_id,
        "prompt": qa["Question"],
        "ref_captions": []
    })


# cap_list = []
# for k, v in ref_captions.items():
#     if k[:9] == "scene0000":
#         cap_list.append({
#             "scene_obj": k,
#             "captions": v
#         })
# cap_list = sorted(cap_list, key=lambda x: x["scene_obj"])
# with open("anno/tmp.json", "w") as f:
#     json.dump(cap_list, f)
#
# exit()

# prompt_list = []
# with open("prompts/conv_description.txt", "r") as f:
#     for line in f.readlines():
#         prompt = line.strip().split("</Scene> ")[-1]
#         prompt_list.append(prompt)
#
# id_set = set()
#
# output_anno = []
#
# for anno in val_anno:
#     scene_id = anno["scene_id"]
#     obj_id = anno["obj_id"]
#     item_id = f"{scene_id}_{obj_id}"
#     if scene_id[:7] == "scene00" and item_id not in id_set:
#         id_set.add(item_id)
#         prompt = random.choice(prompt_list)
#         output_anno.append({
#             "scene_id": scene_id,
#             "obj_id": obj_id,
#             "prompt": prompt,
#             "ref_captions": []
#         })

print(len(output_anno))
output_anno = sorted(output_anno, key=lambda x: f"{x['scene_id']}_{x['obj_id']:03}")
output_anno = output_anno[:200]

with open("anno/scanrefer_val_convs.json", "w") as f:
    json.dump(output_anno, f, indent=4)
