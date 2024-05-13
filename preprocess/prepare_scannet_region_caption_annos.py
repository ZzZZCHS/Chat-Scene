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
from prompts.prompts import region_caption_prompt
import re


def get_caption(clean_text, id_positions):
    p = defaultdict(list)
    sorted_id_positions_items = [(k, id_positions[k]) for k in sorted(id_positions.keys(), key=int)]
    for k, v in sorted_id_positions_items:
        for interval in v:
            # p[interval[0]].append('[')
            p[interval[1]].append(f"<OBJ{int(k):03}>")
    caption = ''
    for idx in range(len(clean_text)):
        if idx in p:
            caption += ' (' + ', '.join(p[idx]) + ')'
        caption += clean_text[idx]
    return caption


def update_caption(caption, new_ids):
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        old_id = int(caption[idx+4:idx+7])
        new_id = int(new_ids[old_id])
        caption = caption[:idx+4] + f"{new_id:03}" + caption[idx+7:]
    return caption


def recover_caption(caption, new_ids):
    old_ids = {new_id: old_id for old_id, new_id in new_ids.items()}
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        new_id = int(caption[idx+4:idx+7])
        try:
            old_id = int(old_ids[new_id])
        except:
            old_id = random.randint(0, len(new_ids)-1)
        caption = caption[:idx+4] + f"{old_id:03}" + caption[idx+7:]
    return caption


def extract_ids(caption):
    id_format = "<OBJ\\d{3}>"
    id_list = set()
    for match in re.finditer(id_format, caption):
        idx = match.start()
        cur_id = int(caption[idx+4:idx+7])
        id_list.add(cur_id)
    id_list = sorted(list(id_list))
    return id_list


def find_match_in_pred(gt_id, instance_locs=None, scannet_locs=None, seg_gt_iou=None):
    if seg_gt_iou is not None:
        if gt_id >= seg_gt_iou.shape[1]:
            return -1, -1
        max_iou, max_id = seg_gt_iou[:, gt_id].max(0)
        return int(max_id), float(max_iou)
    max_iou, max_id = -1, -1
    instance_num = instance_locs.shape[0]
    for pred_id in range(instance_num):
        pred_locs = instance_locs[pred_id].tolist()
        gt_locs = scannet_locs[gt_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        if iou > max_iou:
            max_iou = iou
            max_id = pred_id
    return max_id, max_iou
        

parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

region_captions = json.load(open('annotations/step2_captions_by_scene_v2_anchor.json'))

for split in ['train']:
    scan_list = [x.strip() for x in open(f"annotations/scannet/scannetv2_{split}.txt").readlines()]
    new_annos = []

    if segmentor == 'deva':
        seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
    else:
        instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
        scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
        scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for scene_id in tqdm(scan_list):
        if scene_id not in region_captions:
            continue
        if segmentor == 'deva':
            if scene_id not in seg_gt_ious:
                continue
            seg_gt_iou = seg_gt_ious[scene_id]
            gt_num = seg_gt_iou.shape[1]
        else:
            if scene_id not in instance_attrs:
                continue
            instance_locs = instance_attrs[scene_id]["locs"]
            scannet_locs = scannet_attrs[scene_id]["locs"]
            gt_num = len(scannet_locs)
        for region in region_captions[scene_id]:
            anchor_id = region['anchor_id']
            if anchor_id >= gt_num:
                continue
            if segmentor == 'deva':
                pred_anchor_id, pred_anchor_iou = find_match_in_pred(anchor_id, seg_gt_iou=seg_gt_iou)
            else:
                pred_anchor_id, pred_anchor_iou = find_match_in_pred(anchor_id, instance_locs, scannet_locs)
            if pred_anchor_iou < args.train_iou_thres:
                continue
            positive = region['positive']
            new_positive = {}
            flag = 1
            for k, v in positive.items():
                if int(k) >= gt_num:
                    flag = 0
                    break
                if segmentor == 'deva':
                    new_k, new_iou = find_match_in_pred(int(k), seg_gt_iou=seg_gt_iou)
                else:
                    new_k, new_iou = find_match_in_pred(int(k), instance_locs, scannet_locs)
                if new_iou < args.train_iou_thres:
                    flag = 0
                    break
                new_positive[new_k] = v
            if flag == 0:
                continue
            caption = get_caption(region['sentence'], new_positive)
            if split == 'train':
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': anchor_id,
                    'prompt': random.choice(region_caption_prompt).format(f"<OBJ{pred_anchor_id:03}>"),
                    'caption': caption
                })
            else:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': anchor_id,
                    'prompt': random.choice(region_caption_prompt).format(f"<OBJ{pred_anchor_id:03}>"),
                    'ref_captions': [caption]
                })
    
    print(f"Split: {split}")
    print(len(new_annos))

    with open(f"annotations/scannet_region_caption_{segmentor}_{split}{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)
