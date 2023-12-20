"""
loss/og3d: 2.9594, loss/obj3d_clf: 3.3753, loss/obj3d_clf_pre: 2.0714, loss/txt_clf: 0.6708, loss/total: 10.2789, loss/cross_attn_0: 0.0032, loss/cross_attn_1: 0.0011, loss/cross_attn_2: 0.0011, loss/cross_attn_3: 0.0012, loss/self_attn_0: 0.1595, loss/self_attn_1: 0.0425, loss/self_attn_2: 0.0541, loss/self_attn_3: 0.1030, loss/hidden_state_0: 0.3919, loss/hidden_state_1: 0.0765, loss/hidden_state_2: 0.1033, loss/hidden_state_3: 0.1308, loss/hidden_state_4: 0.1337, acc/og3d: 0.6373, acc/og3d_class: 0.8903, acc/obj3d_clf: 0.6828, acc/obj3d_clf_pre: 0.6131, acc/txt_clf: 0.9281
"""

import json
import jsonlines
import math
import torch
val_file = "/root/scene-LLaMA/datasets/exprs_neurips22/gtlabelpcd_mix/nr3d/preds/val_outs.json"
nr3d_anno_file = "/root/scene-LLaMA/datasets/referit3d/annotations/bert_tokenized/nr3d.jsonl"

anno_root = "annotations"  # annotation dir
attribute_file = f"{anno_root}/scannet_attributes_old.json"
attributes = json.load(open(attribute_file, 'r'))
val_results = json.load(open(val_file))
nr3d_anno = {}
with jsonlines.open(nr3d_anno_file, "r") as reader:
    for l in reader:
        nr3d_anno[l["item_id"]] = l

item_list = []
acc = 0
for k, v in val_results.items():
    obj_ids = v["obj_ids"]
    obj_logits = v["obj_logits"]
    obj_logits = (torch.tensor(obj_logits)).softmax(dim=-1).tolist()
    scan_id = nr3d_anno[k]["scan_id"]
    utter = nr3d_anno[k]["utterance"]
    target_id = nr3d_anno[k]["target_id"]
    obj_num = len(attributes[scan_id]["locs"])
    assert target_id < obj_num, f"{obj_num}, {target_id}, {scan_id}"
    logit_ids = zip(obj_logits, obj_ids)
    logit_ids = sorted(logit_ids, reverse=True)
    logits, ids = zip(*logit_ids)
    # logits = (torch.tensor(logits[:5]) / 5.).softmax(dim=-1).tolist()
    print(logits)
    if ids[0] == target_id:
        acc += 1
    item_list.append({
        "can_ids": ids[:5],
        "can_preds": logits[:5],
        "utter": utter,
        "target_id": target_id,
        "scan_id": scan_id
    })
    # print(target_id)
    # print(ids[:5])
    # print(logits[:5])
    # exit()

print("Acc:", float(acc) / len(item_list))

# print(item_list[:5])
# exit()

import sys
sys.path.append(".")
from models.chat3d import Chat3D
from utils.config_utils import setup_main
import torch
from tasks.shared_utils import setup_model
from utils.basic_utils import setup_seed
from utils.distributed import get_rank
from dataset.base_dataset import process_batch_data

config = setup_main()
setup_seed(config.seed + get_rank())
device = torch.device(config.device)
num_steps_per_epoch = 7000
config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
model_cls = eval(config.model.get('model_cls', 'Chat3D'))
(
    model,
    _,
    optimizer,
    scheduler,
    scaler,
    start_epoch,
    global_step,
) = setup_model(
    config,
    model_cls=model_cls,
    find_unused_parameters=True,
)


pc_encoder = "ulip2"
feat_file = f"{anno_root}/scannet_{pc_encoder}_feats.pt"

with open("prompts/system.txt", "r") as f:
    system = "\n".join([x.strip() for x in f.readlines()])
with open("prompts/score_template.txt", "r") as f:
    prompt_template = "\n".join([x.strip() for x in f.readlines()])
feats = torch.load(feat_file)


def get_feat(scene_id):
    scene_attr = attributes[scene_id]
    obj_num = len(scene_attr["locs"])
    scene_locs = torch.tensor(scene_attr["locs"])
    scene_colors = torch.tensor(scene_attr["colors"])
    scene_attr = torch.cat([scene_locs, scene_colors], dim=1)
    scene_feat = []
    for _i in range(obj_num):
        item_id = "_".join([scene_id, f"{_i:02}"])
        scene_feat.append(feats[item_id])
    scene_feat = torch.stack(scene_feat, dim=0)
    return scene_feat, scene_attr

item_list = item_list[:1000]

from tqdm import tqdm
acc_num = 0
ori_acc_num = 0
true_acc_num = 0
outputs = []
for i, item in tqdm(enumerate(item_list)):
    print("="*20)
    scene_id = item["scan_id"]
    utter = item["utter"]
    tid = item["target_id"]
    # prompt = "\n\n".join([system, prompt_template.format(utter)])
    # print(prompt)
    print("target:", tid)
    max_rating = -1
    max_id = -1
    ori_scene_feat, ori_scene_attr = get_feat(scene_id)
    for _i in range(len(item["can_ids"])):
        print("-"*10)
        obj_id = item["can_ids"][_i]
        obj_pred = item["can_preds"][_i]
        prompt = "\n\n".join([system, prompt_template.format(utter, round(obj_pred*100, 2))])
        # print(prompt)
        assert obj_id < ori_scene_feat.shape[0]
        scene_feats = [ori_scene_feat]
        scene_attrs = [ori_scene_attr]
        scene_feat, scene_attr, scene_mask = process_batch_data(scene_feats, scene_attrs)
        obj_ids = torch.tensor([obj_id])
        scene_feat = scene_feat.to(device)
        scene_attr = scene_attr.to(device)
        scene_mask = scene_mask.to(device)
        target_id = obj_ids.to(device)
        try:
            pred = model(scene_feat=scene_feat, scene_attr=scene_attr, scene_mask=scene_mask, target_id=target_id, is_eval=True, custom_prompt=[prompt])
        except:
            exit()
        print("now_id:", obj_id)
        print(pred[0])
        try:
            # tmp = int(pred[0].split("Rating:")[1].split("/")[0])
            tmp = 1 if "True" in pred[0] else 0
            print("Extract rating:", tmp)
        except:
            print("Fail to extract rating!")
            tmp = 0
        if tmp > max_rating:
            max_id = obj_id
            max_rating = tmp
    print("max_id:", max_id, "target_id:", tid)
    if max_id == tid:
        acc_num += 1
        if max_rating == 1:
            true_acc_num += 1
    if item["can_ids"][0] == tid:
        ori_acc_num += 1
    print("True Acc:", float(true_acc_num) / (i+1))
    print("Current Acc:", float(acc_num) / (i+1))
    print("Ori Acc:", float(ori_acc_num) / (i+1))

print("Acc:", float(acc_num) / len(item_list))
