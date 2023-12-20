"""
loss/og3d: 2.9594, loss/obj3d_clf: 3.3753, loss/obj3d_clf_pre: 2.0714, loss/txt_clf: 0.6708, loss/total: 10.2789, loss/cross_attn_0: 0.0032, loss/cross_attn_1: 0.0011, loss/cross_attn_2: 0.0011, loss/cross_attn_3: 0.0012, loss/self_attn_0: 0.1595, loss/self_attn_1: 0.0425, loss/self_attn_2: 0.0541, loss/self_attn_3: 0.1030, loss/hidden_state_0: 0.3919, loss/hidden_state_1: 0.0765, loss/hidden_state_2: 0.1033, loss/hidden_state_3: 0.1308, loss/hidden_state_4: 0.1337, acc/og3d: 0.6373, acc/og3d_class: 0.8903, acc/obj3d_clf: 0.6828, acc/obj3d_clf_pre: 0.6131, acc/txt_clf: 0.9281
"""

import json
import jsonlines
import math
import torch
from random import shuffle
import random
split = "val"
val_file = f"/root/scene-LLaMA/datasets/exprs_neurips22/gtlabelpcd_mix/nr3d/preds/{split}_outs.json"
nr3d_anno_file = "/root/scene-LLaMA/datasets/referit3d/annotations/bert_tokenized/nr3d.jsonl"

val_results = json.load(open(val_file))

nr3d_anno = {}
with jsonlines.open(nr3d_anno_file, "r") as reader:
    for l in reader:
        nr3d_anno[l["item_id"]] = l

val_split_path = "/root/scene-LLaMA/datasets/referit3d/annotations/splits/scannetv2_val.txt"
scene_ids = []
with open(val_split_path, "r") as f:
    for line in f.readlines():
        scene_ids.append(line.strip())

shuffle(scene_ids)
scene_num = len(scene_ids)
train_scene_num = int(scene_num * 0.8)
train_scene_ids, val_scene_ids = scene_ids[:train_scene_num], scene_ids[train_scene_num:]



multi_choice_template = "Given the description of \"<desc>,\" I have received a list of possible objects from a robust 3D localization model: [<list>]. These objects are considered potential matches for the given description. Kindly provide the object ID that you believe is the closest match to the description. If you believe none of the listed objects are a correct match, please specify an alternative object ID."

item_list = []
train_output_annos = []
val_output_annos = []

acc = 0
random_acc = 0
origin_acc = 0
tot_len = 0
max_len = 0
thres = 1e-2
tot_num = 0
for k, v in val_results.items():
    obj_ids = v["obj_ids"]
    obj_logits = v["obj_logits"]
    obj_logits = (torch.tensor(obj_logits)).softmax(dim=-1).tolist()
    scan_id = nr3d_anno[k]["scan_id"]
    utter = nr3d_anno[k]["utterance"]
    target_id = nr3d_anno[k]["target_id"]
    logit_ids = zip(obj_logits, obj_ids)
    logit_ids = sorted(logit_ids, reverse=True)
    logits, ids = zip(*logit_ids)
    # print(logits)
    # print(ids)
    # print(target_id)
    # breakpoint()
    can_ids = []
    if split == "val":
        for i in range(min(len(logits), 5)):
            if logits[i] > thres:
                can_ids.append(ids[i])
    else:
        can_num = random.randint(1, 5)
        can_ids = ids[:can_num]
    if len(can_ids) == 1:
        continue
    # can_ids = sorted(can_ids)
    id_list = ""
    for i in range(len(can_ids)):
        if i > 0:
            id_list += ", "
        id_list += f"obj{can_ids[i]:02}"
    if utter[-1] == ".":
        utter = utter[:-1]
    prompt = multi_choice_template.replace("<desc>", utter).replace("<list>", id_list)
    answer = f"obj{target_id:02}.".capitalize()
    # logits = (torch.tensor(logits[:5]) / 5.).softmax(dim=-1).tolist()
    # print(logits)
    # if ids[0] == target_id:
    #     acc += 1

    # item_list.append({
    #     "can_ids": ids[:5],
    #     "can_preds": logits[:5],
    #     "utter": utter,
    #     "target_id": target_id,
    #     "scan_id": scan_id
    # })
    if scan_id in train_scene_ids:
        train_output_annos.append({
            "scene_id": scan_id,
            "obj_id": target_id,
            "prompt": prompt,
            "caption": answer,
        })
    else:
        val_output_annos.append({
            "scene_id": scan_id,
            "obj_id": target_id,
            "prompt": prompt,
            "ref_captions": [answer],
            "qid": k
        })
        if target_id in can_ids:
            acc += 1
        if random.choice(can_ids) == target_id:
            random_acc += 1
        if ids[0] == target_id:
            origin_acc += 1
        tot_len += len(can_ids)
        max_len = len(can_ids) if len(can_ids) > max_len else max_len
        tot_num += 1

# if split == "val":
#     with open(f"annotations/nr3d_{split}_stage2_multichoice{str(thres)}.json", "w") as f:
#         json.dump(train_output_annos, f, indent=4)
# else:
#     with open(f"annotations/nr3d_{split}_stage2_multichoice.json", "w") as f:
#         json.dump(val_output_annos, f, indent=4)

with open(f"annotations/nr3d_train_stage2_multichoice{str(thres)}.json", "w") as f:
    json.dump(train_output_annos, f, indent=4)
with open(f"annotations/nr3d_val_stage2_multichoice{str(thres)}.json", "w") as f:
    json.dump(val_output_annos, f, indent=4)

print(tot_num)
print("Origin Acc:", float(origin_acc) / tot_num)
print("Upper Acc:", float(acc) / tot_num)
print("Random Acc:", float(random_acc) / tot_num)
print("mean len:", float(tot_len) / tot_num)
print("max len:", max_len)
# print(len(item_list))
# print(item_list[:5])
# exit()
