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
from prompts.prompts import nr3d_caption_prompt
import csv


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

train_scenes = [x.strip() for x in open('annotations/scannet/scannetv2_train.txt').readlines()]
val_scenes = [x.strip() for x in open('annotations/scannet/scannetv2_val.txt').readlines()]
scene_lists = {
    'train': train_scenes,
    'val': val_scenes
}

raw_annos = []
with open('annotations/referit3d/nr3d.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        raw_annos.append({
            'scene_id': row['scan_id'],
            'obj_id': int(row['target_id']),
            'caption': row['utterance']
        })

for split in ["train"]:
    annos = [anno for anno in raw_annos if anno['scene_id'] in scene_lists[split]]
    new_annos = []

    if segmentor == 'deva':
        seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
    else:
        instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
        scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
        scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for anno in tqdm(annos):
        scene_id = anno['scene_id']
        obj_id = anno['obj_id']
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
            instance_locs = instance_attrs[scene_id]['locs']
            scannet_locs = scannet_attrs[scene_id]['locs']
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
        prompt = random.choice(nr3d_caption_prompt).replace('<id>', f"<OBJ{max_id:03}>")
        if split == 'train':
            if max_iou >= args.train_iou_thres:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': obj_id,
                    'prompt': prompt,
                    'caption': anno['caption']
                })
        else:
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': prompt,
                'ref_captions': [anno['caption']]
            })
    print(len(new_annos))

    with open(f"annotations/nr3d_caption_{segmentor}_{split}{version}.json", 'w') as f:
        json.dump(new_annos, f, indent=4)
