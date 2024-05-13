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

unwanted_words = ["wall", "ceiling", "floor", "object", "item"]

segmentor = args.segmentor
version = args.version

for split in ["train", "val"]:
    new_annos = []

    if segmentor == 'deva':
        seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
    else:
        instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for scene_id in tqdm(scannet_attrs.keys()):
        if segmentor == 'deva':
            if scene_id not in seg_gt_ious:
                continue
            seg_gt_iou = seg_gt_ious[scene_id]
            segmented_num = seg_gt_iou.shape[0]
            gt_num = seg_gt_iou.shape[1]
        else:
            if scene_id not in instance_attrs:
                continue
            instance_locs = instance_attrs[scene_id]['locs']
            scannet_locs = scannet_attrs[scene_id]['locs']
            segmented_num = len(instance_locs)
            gt_num = len(scannet_locs)
        scannet_class_labels = scannet_attrs[scene_id]['objects']
        for obj_id in range(gt_num):
            class_label = scannet_class_labels[obj_id]
            if any(x in class_label for x in unwanted_words):
                continue
            if segmentor == 'deva':
                max_iou, max_id = seg_gt_iou[:, obj_id].max(0)
                max_iou = float(max_iou)
                max_id = int(max_id)
            else:
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
                if max_iou >= args.train_iou_thres:
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