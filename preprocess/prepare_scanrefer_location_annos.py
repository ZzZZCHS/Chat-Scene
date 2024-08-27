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
from prompts.prompts import grounding_location_prompt
import string


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--max_obj_num', type=int, default=150)
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

for split in ["train", "val"]:
    count = [0] * args.max_obj_num
    annos = json.load(open(f"annotations/scanrefer/ScanRefer_filtered_{split}.json", "r"))
    annos = sorted(annos, key=lambda p: f"{p['scene_id']}_{int(p['object_id']):03}")
    new_annos = []

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
        prompt = random.choice(grounding_location_prompt).replace('<description>', desc)
        
        if scene_id not in instance_attrs:
            continue
        scannet_locs = scannet_attrs[scene_id]["locs"]
        gt_locs = scannet_locs[obj_id].tolist()
        
        gt_loc_tokens = [num_to_location_token(x) for x in gt_locs]
        caption = "<LOCATION> " + " ".join(gt_loc_tokens) + " </LOCATION>"
        
        if split == "train":
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "caption": caption,
                "prompt": prompt
            })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "ref_captions": [caption],
                "prompt": prompt
            })

    print(len(new_annos))
    print(count)
    # print(maxiou_count / valid_count)
    # print(f"max iou@0.25: {iou25_count / len(new_annos)}")
    # print(f"max iou@0.5: {iou50_count / len(new_annos)}")

    with open(f"annotations/scanrefer_{segmentor}_{split}_location{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)