import numpy as np
import json
import sys
sys.path.append('.')
import torch
import random
from tqdm import tqdm
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
args = parser.parse_args()


segmentor = args.segmentor
version = args.version
# annos = json.load(open(f"annotations/scanrefer_{split}_stage2_grounding_OBJ.json", "r"))
feats = torch.load(f'annotations/scannet_img_dinov2_features.pt')
new_feats = {}
item2iou = {}
iou_thres = 0.5

for split in ['train', 'val']:
    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file)
    scannet_attrs = torch.load(scannet_attribute_file)
    for k, v in tqdm(feats.items()):
        scene_id = '_'.join(k.split('_')[:2])
        if scene_id not in instance_attrs:
            continue
        obj_id = int(k.split('_')[-1])
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        max_iou, max_id = -1, -1
        for pred_id in range(instance_num):
            pred_locs = instance_locs[pred_id].tolist()
            try:
                gt_locs = scannet_locs[obj_id].tolist()
            except:
                gt_locs = scannet_locs[obj_id-31].tolist()
                # print(k)
                # breakpoint()
                # break
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            iou = box3d_iou(pred_corners, gt_corners)
            if iou > max_iou:
                max_iou = iou
                max_id = pred_id
        item_id = f"{scene_id}_{max_id:02}"
        if max_iou > iou_thres and (item_id not in new_feats or item2iou[item_id] < max_iou):
            new_feats[item_id] = v
            item2iou[item_id] = max_iou

torch.save(new_feats, f'annotations/scannet_img_mask3d_dinov2_features{version}.pt')