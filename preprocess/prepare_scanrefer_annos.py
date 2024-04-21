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
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

for split in ["train", "val"]:
    count = [0] * 100
    annos = json.load(open(f"annotations/scanrefer_{split}_grounding.json", "r"))
    new_annos = []

    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file)
    scannet_attrs = torch.load(scannet_attribute_file)


    for i, anno in tqdm(enumerate(annos)):
        scene_id = anno["scene_id"]
        obj_id = anno["obj_id"]
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
        count[max_id] += 1
        if split == "train":
            if max_iou > 0.75:
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": max_id,
                    "caption": f"<OBJ{max_id:03}>.",
                    "prompt": anno["prompt"]
                })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "ref_captions": [f"<OBJ{max_id:03}>."],
                "prompt": anno["prompt"]
            })

    print(len(new_annos))
    print(count)

    with open(f"annotations/scanrefer_{segmentor}_{split}_grounding{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)