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
from prompts.prompts import obj_caption_wid_prompt

parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

for split in ['train']:
    annos = json.load(open(f"annotations/scannet_{split}_caption.json"))
    new_annos = []
    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file)
    scannet_attrs = torch.load(scannet_attribute_file)

    for i, anno in tqdm(enumerate(annos)):
        scene_id = anno['scene_id']
        obj_id = anno['obj_id']
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
        if split == 'train':
            if max_iou > args.train_iou_thres:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': obj_id,
                    'prompt': random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{max_id:03}>"),
                    'caption': anno['caption']
                })
        else:
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{max_id:03}>"),
                'ref_captions': anno['ref_captions']
            })
    
    print(f"Split: {split}")
    print(f"{len(annos)} -> {len(new_annos)}")

    with open(f"annotations/scannet_{segmentor}_{split}_caption{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)
