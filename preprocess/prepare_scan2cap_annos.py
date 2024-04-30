import numpy as np
import json
import sys
sys.path.append('.')
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()


for split in ["train", "val"]:
    segmentor = args.segmentor
    version = args.version
    thr = 0.
    annos = json.load(open(f"annotations/scanrefer_{split}_stage2_objxx.json", "r"))
    new_annos = []

    print(len(annos))
    corpus = defaultdict(list)
    for anno in annos:
        gt_key = f"{anno['scene_id']}|{anno['obj_id']}"
        if split == "train":
            corpus[gt_key].append(anno['caption'])
        else:
            corpus[gt_key] = anno['ref_captions']

    count = [0] * 100
    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file)
    scannet_attrs = torch.load(scannet_attribute_file)

    prompt_templates = []
    with open('prompts/scanrefer_caption_templates.txt') as f:
        prompt_templates = [p.strip() for p in f.readlines()]


    covered25_num, covered50_num = 0, 0
    count_all = 0
    for scene_id in tqdm(instance_attrs.keys()):
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        gt_match_id = [-1] * len(scannet_locs)
        gt_match_iou = [-1] * len(scannet_locs)
        for pred_id in range(len(instance_locs)):
            pred_locs = instance_locs[pred_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            max_id = max_iou = -1
            for gt_id in range(len(scannet_locs)):
                if f"{scene_id}|{gt_id}" not in corpus:
                    continue
                gt_locs = scannet_locs[gt_id].tolist()
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = gt_id
            if f"{scene_id}|{max_id}" not in corpus:
                continue
            if max_iou > gt_match_iou[max_id]:
                gt_match_iou[max_id] = max_iou
                gt_match_id[max_id] = pred_id
        for gt_id, pred_id in enumerate(gt_match_id):
            if f"{scene_id}|{gt_id}" in corpus:
                count_all += len(corpus[f"{scene_id}|{gt_id}"])
            if pred_id == -1:
                continue
            if split == 'train' and gt_match_iou[gt_id] < args.train_iou_thres:
                continue
            if gt_match_iou[gt_id] >= 0.25:
                covered25_num += len(corpus[f"{scene_id}|{gt_id}"])
            if gt_match_iou[gt_id] >= 0.5:
                covered50_num += len(corpus[f"{scene_id}|{gt_id}"])
            count[pred_id] += 1
            if split == 'train':
                for caption in corpus[f"{scene_id}|{gt_id}"]:
                    new_annos.append({
                        'scene_id': scene_id,
                        'obj_id': gt_id,
                        'pred_id': pred_id,
                        'prompt': random.choice(prompt_templates).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                        "caption": caption,
                        "iou": gt_match_iou[gt_id]
                    })
            else:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': gt_id,
                    'pred_id': pred_id,
                    'prompt': random.choice(prompt_templates).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                    "ref_captions": corpus[f"{scene_id}|{gt_id}"],
                    "iou": gt_match_iou[gt_id]
                })

    print(len(new_annos))
    print(covered25_num, covered50_num)
    print(count_all)
    # print(count)

    with open(f"annotations/scanrefer_{segmentor}_{split}_caption{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)