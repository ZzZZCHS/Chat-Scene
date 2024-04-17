import numpy as np
import json
import sys
import torch
import random
from tqdm import tqdm
from collections import defaultdict

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


# count = [0] * 100
# split = "val"
# segmentor = "mask3d"
# annos = json.load(open(f"annotations/scanrefer_{split}_stage2_grounding_OBJ.json", "r"))
# new_annos = []
# version = "100_v2"

# instance_attribute_file = f"annotations/scannet_{segmentor}_{split}_attributes{version}.pt"
# scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"

# instance_attrs = torch.load(instance_attribute_file)
# scannet_attrs = torch.load(scannet_attribute_file)


# for i, anno in tqdm(enumerate(annos)):
#     scene_id = anno["scene_id"]
#     obj_id = anno["obj_id"]
#     if scene_id not in instance_attrs:
#         continue
#     instance_locs = instance_attrs[scene_id]["locs"]
#     scannet_locs = scannet_attrs[scene_id]["locs"]
#     instance_num = instance_locs.shape[0]
#     max_iou, max_id = -1, -1
#     for pred_id in range(instance_num):
#         pred_locs = instance_locs[pred_id].tolist()
#         gt_locs = scannet_locs[obj_id].tolist()
#         pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
#         gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
#         iou = box3d_iou(pred_corners, gt_corners)
#         if iou > max_iou:
#             max_iou = iou
#             max_id = pred_id
#     count[max_id] += 1
#     if split == "train":
#         if max_iou > 0.75:
#             new_annos.append({
#                 "scene_id": scene_id,
#                 "obj_id": max_id,
#                 "caption": f"<OBJ{max_id:03}>.",
#                 "prompt": anno["prompt"]
#             })
#     else:
#         new_annos.append({
#             "scene_id": scene_id,
#             "obj_id": obj_id,
#             "ref_captions": [f"<OBJ{max_id:03}>."],
#             "prompt": anno["prompt"]
#         })

# print(len(new_annos))
# print(count)

# with open(f"annotations/scanrefer_{segmentor}_{split}_stage2_grounding_OBJ{version}.json", "w") as f:
#     json.dump(new_annos, f, indent=4)
# exit()

split = "val"
encoder = "mask3d"
thr = 0.
annos = json.load(open(f"annotations/scanrefer_{split}_stage2_objxx.json", "r"))
new_annos = []

print(len(annos))
corpus = defaultdict(list)
for anno in annos:
    gt_key = f"{anno['scene_id']}|{anno['obj_id']}"
    # corpus[gt_key] = [f"sos {caption} eos".replace('\n', ' ') for caption in anno['ref_captions']]
    if split == "train":
        corpus[gt_key].append(anno['caption'])
    else:
        corpus[gt_key] = anno['ref_captions']

# myKeys = list(corpus.keys())
# myKeys.sort()
# corpus = {i: corpus[i] for i in myKeys}
# with open('annotations/scan2cap_val_corpus.json', 'w') as f:
#     json.dump(corpus, f, indent=4)
# exit()
count = [0] * 100
version = "100_v2"
instance_attribute_file = f"annotations/scannet_{encoder}_{split}_attributes{version}.pt"
scannet_attribute_file = f"annotations/scannet_{split}_attributes.pt"

instance_attrs = torch.load(instance_attribute_file)
scannet_attrs = torch.load(scannet_attribute_file)

prompt_templates = []
with open('prompts/scanrefer_caption_templates.txt') as f:
    prompt_templates = [p.strip() for p in f.readlines()]


covered25_num, covered50_num = 0, 0
count_all, count_0 = 0, 0
for scene_id in tqdm(instance_attrs.keys()):
    instance_locs = instance_attrs[scene_id]["locs"]
    scannet_locs = scannet_attrs[scene_id]["locs"]
    gt_match_id = [-1] * len(scannet_locs)
    gt_match_iou = [-1] * len(scannet_locs)
    for pred_id in range(len(instance_locs)):
        pred_locs = instance_locs[pred_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        max_id = max_iou = -1
        for gt_id in range(len(scannet_locs)):
            # if f"{scene_id}|{gt_id}" not in corpus:
            #     continue
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
    # for gt_id in range(len(scannet_locs)):
    #     if f"{scene_id}|{gt_id}" not in corpus:
    #         continue
    #     gt_locs = scannet_locs[gt_id].tolist()
    #     gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
    #     max_id = max_iou = -1
    #     for pred_id in range(len(instance_locs)):
    #         pred_locs = instance_locs[pred_id].tolist()
    #         pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
    #         iou = box3d_iou(pred_corners, gt_corners)
    #         if iou > max_iou:
    #             max_iou = iou
    #             max_id = pred_id
    #     gt_match_iou[gt_id] = max_iou
    #     gt_match_id[gt_id] = max_id
    for gt_id, pred_id in enumerate(gt_match_id):
        if f"{scene_id}|{gt_id}" in corpus:
            count_all += len(corpus[f"{scene_id}|{gt_id}"])
        if pred_id == -1:
            continue
        if split == 'train' and gt_match_iou[gt_id] < 0.75:
            continue
        if gt_match_iou[gt_id] >= 0.25:
            covered25_num += len(corpus[f"{scene_id}|{gt_id}"])
        if gt_match_iou[gt_id] >= 0.5:
            covered50_num += len(corpus[f"{scene_id}|{gt_id}"])
        if gt_match_iou[gt_id] == 0:
            count_0 += len(corpus[f"{scene_id}|{gt_id}"])
        count[pred_id] += 1
        if split == 'train':
            for caption in corpus[f"{scene_id}|{gt_id}"]:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': gt_id,
                    'pred_id': pred_id,
                    'prompt': random.choice(prompt_templates).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                    "caption": caption,
                    "iou": gt_match_iou[gt_id]
                })
        else:
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': gt_id,
                'pred_id': pred_id,
                'prompt': random.choice(prompt_templates).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                "ref_captions": corpus[f"{scene_id}|{gt_id}"],
                "iou": gt_match_iou[gt_id]
            })

# filtered_annos = {}

# for i, anno in tqdm(enumerate(annos)):
#     scene_id = anno["scene_id"]
#     obj_id = anno["obj_id"]
#     prompt = anno["prompt"]
#     qid = f"{scene_id}_{obj_id}"
#     if scene_id not in instance_attrs:
#         continue
#     instance_locs = instance_attrs[scene_id]["locs"]
#     scannet_locs = scannet_attrs[scene_id]["locs"]
#     instance_num = instance_locs.shape[0]
#     max_iou, max_id = -1, -1
#     for pred_id in range(instance_num):
#         pred_locs = instance_locs[pred_id].tolist()
#         gt_locs = scannet_locs[obj_id].tolist()
#         pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
#         gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
#         iou = box3d_iou(pred_corners, gt_corners)
#         if iou > max_iou:
#             max_iou = iou
#             max_id = pred_id
#     if max_iou >= thr:
#         if split == "train":
#             new_annos.append({
#                 "scene_id": scene_id,
#                 "obj_id": max_id,
#                 "prompt": prompt.replace(f"obj{obj_id:02}", f"<OBJ{max_id:03}>"),
#                 "caption": anno["caption"],
#                 "iou": max_iou
#             })
#         else:
#             if qid not in filtered_annos or max_iou >= filtered_annos[qid]["iou"]:
#                 filtered_annos[qid] = {
#                     "scene_id": scene_id,
#                     "obj_id": obj_id,
#                     "pred_id": max_id,
#                     "prompt": prompt.replace(f"obj{obj_id:02}", f"<OBJ{max_id:03}>"),
#                     "ref_captions": anno["ref_captions"],
#                     "iou": max_iou
#                 }

# if len(new_annos) == 0:
#     new_annos = list(filtered_annos.values())
print(len(new_annos))
print(covered25_num, covered50_num)
print(count_all, count_0)
# print(count)

# with open(f"annotations/scanrefer_{encoder}_{split}_stage2_caption_OBJ{version}.json", "w") as f:
#     json.dump(new_annos, f, indent=4)
