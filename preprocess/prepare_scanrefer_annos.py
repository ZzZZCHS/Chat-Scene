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
from prompts.prompts import grounding_prompt
import string


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--max_obj_num', type=int, default=150)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

for split in ["train", "val"]:
    count = [0] * args.max_obj_num
    annos = json.load(open(f"annotations/scanrefer/ScanRefer_filtered_{split}.json", "r"))
    annos = sorted(annos, key=lambda p: f"{p['scene_id']}_{int(p['object_id']):03}")
    new_annos = []

    if segmentor == 'deva':
        seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
    else:
        instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
        scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
        scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    iou25_count = 0
    iou50_count = 0
    # maxiou_count = 0
    # valid_count = 0
    for i, anno in tqdm(enumerate(annos)):
        scene_id = anno['scene_id']
        obj_id = int(anno['object_id'])
        desc = anno['description']
        if desc[-1] in string.punctuation:
            desc = desc[:-1]
        prompt = random.choice(grounding_prompt).replace('<description>', desc)
        
        if segmentor == 'deva':
            if scene_id not in seg_gt_ious:
                continue
            seg_gt_iou = seg_gt_ious[scene_id]
            if obj_id >= seg_gt_iou.shape[1]:
                continue
            max_iou, max_id = seg_gt_iou[:, obj_id].max(0)
            max_iou = float(max_iou)
            max_id = int(max_id)
        else:
            if scene_id not in instance_attrs:
                continue
            instance_locs = instance_attrs[scene_id]["locs"]
            scannet_locs = scannet_attrs[scene_id]["locs"]
            instance_num = instance_locs.shape[0]
            max_iou, max_id = -1, -1
            for pred_id in range(instance_num):
                pred_locs = instance_locs[pred_id].tolist()
                gt_locs = scannet_locs[obj_id].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = pred_id
        # maxiou_count += max_iou
        # valid_count += 1
        if max_iou >= 0.25:
            iou25_count += 1
        if max_iou >= 0.5:
            iou50_count += 1
        count[max_id] += 1
        if split == "train":
            if max_iou >= args.train_iou_thres:
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": max_id,
                    "caption": f"<OBJ{max_id:03}>.",
                    "prompt": prompt
                })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "ref_captions": [f"<OBJ{max_id:03}>."],
                "prompt": prompt
            })

    print(len(new_annos))
    print(count)
    # print(maxiou_count / valid_count)
    # print(f"max iou@0.25: {iou25_count / len(new_annos)}")
    # print(f"max iou@0.5: {iou50_count / len(new_annos)}")

    with open(f"annotations/scanrefer_{segmentor}_{split}{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)