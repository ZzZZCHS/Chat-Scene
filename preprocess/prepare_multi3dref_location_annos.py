import json
import torch
import os
import sys
sys.path.append('.')
from tqdm import tqdm
import argparse
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
from prompts.prompts import multi3dref_location_prompt, ID_format
import random
from collections import defaultdict
import string

parser = argparse.ArgumentParser()
parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

def num_to_location_token(ori_num):
    ori_num = int(ori_num * 100) + 500
    if ori_num < 0:
        ori_num = 0
    if ori_num > 999:
        ori_num = 999
    return f"<LOC{ori_num:03}>"

for split in ['train', 'val']:
    annos = json.load(open(f"annotations/multi3drefer/multi3drefer_{split}.json"))
    new_annos = []

    count_all = defaultdict(int)
    count_used = defaultdict(int)
    instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for i, anno in tqdm(enumerate(annos)):
        scene_id = anno['scene_id']
        count_all[anno['eval_type']] += 1
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        gt_ids = anno['object_ids']
        caption = anno['description']
        if caption[-1] in string.punctuation:
            caption = caption[:-1]
        prompt = random.choice(multi3dref_location_prompt).replace("<description>", caption)
        locs_caption_list = []
        flag = 1
        for gt_id in gt_ids:
            max_iou, max_id = -1, -1
            # for pred_id in range(instance_num):
            #     pred_locs = instance_locs[pred_id].tolist()
            #     gt_locs = scannet_locs[gt_id].tolist()
            #     pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            #     gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            #     iou = box3d_iou(pred_corners, gt_corners)
            #     if iou > max_iou:
            #         max_iou = iou
            #         max_id = pred_id
            gt_locs = scannet_locs[gt_id].tolist()
            gt_loc_tokens = [num_to_location_token(x) for x in gt_locs]
            tmp_loc_caption = "<LOCATION> " + " ".join(gt_loc_tokens) + " </LOCATION>"
            locs_caption_list.append(tmp_loc_caption)
        if flag == 0:
            continue
        count_used[anno['eval_type']] += 1
        locs_caption_list = sorted(locs_caption_list)
        # pred_ids = sorted(pred_ids)
        # pred_id_strs = [ID_format.format(pred_id) for pred_id in pred_ids]
        if len(locs_caption_list) == 0:
            answer = "No."
        elif len(locs_caption_list) == 1:
            answer = f"Yes. {locs_caption_list[0]}."
        elif len(locs_caption_list) == 2:
            answer = f"Yes. {locs_caption_list[0]} and {locs_caption_list[1]}."
        else:
            answer = f"Yes. {', '.join(locs_caption_list[:-1])}, and {locs_caption_list[-1]}."
        if split == 'train':
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': 0,
                'prompt': prompt,
                'caption': answer,
                'eval_type': anno['eval_type']
            })
        else:
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': 0,
                'prompt': prompt,
                'ref_captions': gt_ids,
                'eval_type': anno['eval_type']
            })

    print(f"Split: {split}")
    print(f"Count all: {len(annos)}", count_all)
    print(f"Count used: {len(new_annos)}", count_used)
    
    with open(f"annotations/multi3dref_{segmentor}_{split}_location{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)
