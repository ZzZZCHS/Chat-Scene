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
from prompts.prompts import scan2cap_prompt
import nltk


def capitalize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    result = ' '.join(capitalized_sentences)
    return result


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--max_obj_num', type=int, default=150)
args = parser.parse_args()


for split in ["train", "val"]:
    segmentor = args.segmentor
    version = args.version
    annos = json.load(open(f"annotations/scanrefer/ScanRefer_filtered_{split}.json", "r"))
    new_annos = []

    print(len(annos))
    scene_ids = set()
    corpus = defaultdict(list)
    for anno in annos:
        gt_key = f"{anno['scene_id']}|{anno['object_id']}"
        description = capitalize_sentences(anno['description'])
        corpus[gt_key].append(description)
        scene_ids.add(anno['scene_id'])
    scene_ids = list(scene_ids)

    count = [0] * args.max_obj_num
    if segmentor == 'deva':
        if split == 'train':
            seg_gt_ious = torch.load(f"annotations/scannet_{segmentor}_seg_gt_ious.pt", map_location='cpu')
        else:
            instance_attribute_file = f"annotations/scannet_{segmentor}_attributes{version}.pt"
            scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
            instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
            scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    else:
        instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
        scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"
        instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
        scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')


    covered25_num, covered50_num = 0, 0
    count_all = 0
    for scene_id in tqdm(scene_ids):
        if segmentor == 'deva' and split =='train':
            if scene_id not in seg_gt_ious:
                continue
            seg_gt_iou = seg_gt_ious[scene_id]
            segmented_num = seg_gt_iou.shape[0]
            gt_num = seg_gt_iou.shape[1]
        else:
            if scene_id not in instance_attrs:
                continue
            instance_locs = instance_attrs[scene_id]["locs"]
            scannet_locs = scannet_attrs[scene_id]["locs"]
            segmented_num = len(instance_locs)
            gt_num = len(scannet_locs)
        gt_match_id = [-1] * gt_num
        gt_match_iou = [-1] * gt_num
        for pred_id in range(segmented_num):
            if segmentor == 'deva' and split == 'train':
                max_iou, max_id = seg_gt_iou[pred_id, :].max(0)
                max_iou = float(max_iou)
                max_id = int(max_id)
            else:
                pred_locs = instance_locs[pred_id].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                max_id = max_iou = -1
                for gt_id in range(len(scannet_locs)):
                    if f"{scene_id}|{gt_id}" not in corpus:
                        continue
                    gt_locs = scannet_locs[gt_id].tolist()
                    gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                    iou = box3d_iou(pred_corners, gt_corners)
                    if iou > max_iou:
                        max_iou = iou
                        max_id = gt_id
            if f"{scene_id}|{max_id}" not in corpus:
                continue
            if max_iou > gt_match_iou[max_id]:
                gt_match_iou[max_id] = max_iou
                gt_match_id[max_id] = pred_id
        for gt_id, pred_id in enumerate(gt_match_id):
            if f"{scene_id}|{gt_id}" in corpus:
                count_all += len(corpus[f"{scene_id}|{gt_id}"])
            if pred_id == -1:
                continue
            if split == 'train' and gt_match_iou[gt_id] < args.train_iou_thres:
                continue
            if gt_match_iou[gt_id] >= 0.25:
                covered25_num += len(corpus[f"{scene_id}|{gt_id}"])
            if gt_match_iou[gt_id] >= 0.5:
                covered50_num += len(corpus[f"{scene_id}|{gt_id}"])
            count[pred_id] += 1
            if split == 'train':
                for caption in corpus[f"{scene_id}|{gt_id}"]:
                    new_annos.append({
                        'scene_id': scene_id,
                        'obj_id': gt_id,
                        'pred_id': pred_id,
                        'prompt': random.choice(scan2cap_prompt).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                        "caption": caption,
                        "iou": gt_match_iou[gt_id]
                    })
            else:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': gt_id,
                    'pred_id': pred_id,
                    'prompt': random.choice(scan2cap_prompt).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                    "ref_captions": corpus[f"{scene_id}|{gt_id}"],
                    "iou": gt_match_iou[gt_id]
                })

    print(len(new_annos))
    print(covered25_num, covered50_num)
    print(count_all)
    # print(count)

    with open(f"annotations/scan2cap_{segmentor}_{split}{version}.json", "w") as f:
        json.dump(new_annos, f, indent=4)