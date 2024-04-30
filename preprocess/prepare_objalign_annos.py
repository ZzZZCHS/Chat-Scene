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

unwanted_words = ["wall", "ceiling", "floor", "object", "item"]

segmentor = args.segmentor
version = args.version

for split in ["train", "val"]:
    annos = json.load(open(f"annotations/obj_align_train.json"))
    new_annos = []

    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file)
    scannet_attrs = torch.load(scannet_attribute_file)

    for scene_id in tqdm(scannet_attrs.keys()):
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]['locs']
        scannet_locs = scannet_attrs[scene_id]['locs']
        scannet_class_labels = scannet_attrs[scene_id]['objects']
        for obj_id in range(scannet_locs.shape[0]):
            class_label = scannet_class_labels[obj_id]
            if any(x in class_label for x in unwanted_words):
                continue
            max_iou, max_id = -1, -1
            for pred_id in range(instance_locs.shape[0]):
                pred_locs = instance_locs[pred_id].tolist()
                gt_locs = scannet_locs[obj_id].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = pred_id
            prompt = f"What is the <OBJ{max_id:03}>?"
            caption = f"<OBJ{max_id:03}> is a {class_label}."
            if split == 'train':
                if max_iou > 0.5:
                    new_annos.append({
                        'scene_id': scene_id,
                        'obj_id': obj_id,
                        'prompt': prompt,
                        'caption': caption
                    })
            else:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': obj_id,
                    'prompt': prompt,
                    'ref_captions': [caption]
                })
    
    print(len(new_annos))
    with open(f"annotations/obj_align_{segmentor}_{split}{version}.json", 'w') as f:
        json.dump(new_annos, f, indent=4)