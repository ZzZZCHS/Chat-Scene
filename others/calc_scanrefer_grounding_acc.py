import numpy as np
import json
import sys
import torch
import random
from tqdm import tqdm
import re
import os
import csv
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


output_file = "/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240419_171033_dp0.1_lr5e-6_sta2_ep3_scanrefer_seg#scan2cap_seg#nr3d_caption_seg#obj_align_seg#scanqa_seg#sqa3d_seg__scanqa#scanrefer#sqa3d#scan2cap__debug/preds_epoch-1_step0_scanrefer.json"
outputs = json.load(open(output_file, "r"))


unique_multiple_lookup_file = 'annotations/scanrefer_unique_multiple_lookup.json'
if not os.path.exists(unique_multiple_lookup_file):
    type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}
    scannet_labels = type2class.keys()
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}
    label_classes_set = set(scannet_labels)
    raw2label = {}
    with open('annotations/scannet/scannetv2-labels.combined.tsv', 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        csvreader.__next__()
        for line in csvreader:
            raw_name = line[1]
            nyu40_name = line[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]
    all_sem_labels = defaultdict(list)
    cache = defaultdict(dict)
    scanrefer_data = json.load(open('annotations/scanrefer/ScanRefer_filtered.json'))
    for data in scanrefer_data:
        scene_id = data['scene_id']
        object_id = data['object_id']
        object_name = ' '.join(data['object_name'].split('_'))
        if object_id not in cache[scene_id]:
            cache[scene_id][object_id] = {}
            try:
                all_sem_labels[scene_id].append(raw2label[object_name])
            except:
                all_sem_labels[scene_id].append(17)
    all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}
    unique_multiple_lookup = defaultdict(dict)
    for data in scanrefer_data:
        scene_id = data['scene_id']
        object_id = data['object_id']
        object_name = ' '.join(data['object_name'].split('_'))
        try:
            sem_label = raw2label[object_name]
        except:
            sem_label = 17
        unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
        unique_multiple_lookup[scene_id][object_id] = unique_multiple
    with open(unique_multiple_lookup_file, 'w') as f:
        json.dump(unique_multiple_lookup, f, indent=4)
else:
    unique_multiple_lookup = json.load(open(unique_multiple_lookup_file))


instance_attribute_file = "annotations/scannet_mask3d_val_attributes_v2.pt"
scannet_attribute_file = "annotations/scannet_val_attributes.pt"

instance_attrs = torch.load(instance_attribute_file)
scannet_attrs = torch.load(scannet_attribute_file)

iou25_acc = 0
iou50_acc = 0
unique_iou25_acc = 0
unique_iou50_acc = 0
unique_all = 0
multiple_iou25_acc = 0
multiple_iou50_acc = 0
multiple_all = 0

count_list = [0] * 150
iou25_acc_list = [0] * 150
iou50_acc_list = [0] * 150
id_format = "<OBJ\\d{3}>"

for i, output in tqdm(enumerate(outputs)):
    scene_id = output["scene_id"]
    obj_id = output["gt_id"]
    instance_locs = instance_attrs[scene_id]["locs"]
    scannet_locs = scannet_attrs[scene_id]["locs"]
    unique_multiple = unique_multiple_lookup[scene_id][str(obj_id)]
    if unique_multiple == 0:
        unique_all += 1
    else:
        multiple_all += 1
    # instance_name = instance_attrs[scene_id]['objects']
    # scannet_name = scannet_attrs[scene_id]['objects']

    pred = output["pred"]
    instance_num = instance_locs.shape[0]
    # pred_id = random.randint(0, instance_num-1)
    pred_id = 0
    for match in re.finditer(id_format, pred):
            idx = match.start()
            cur_id = int(pred[idx+4:idx+7])
            if cur_id < instance_num:
                pred_id = cur_id
                break
    pred_locs = instance_locs[pred_id].tolist()
    gt_locs = scannet_locs[obj_id].tolist()
    pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
    gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
    iou = box3d_iou(pred_corners, gt_corners)
    if iou >= 0.25:
        iou25_acc += 1
        if unique_multiple == 0:
            unique_iou25_acc += 1
        else:
            multiple_iou25_acc += 1
        iou25_acc_list[scannet_locs.shape[0]] += 1
    if iou >= 0.5:
        iou50_acc += 1
        if unique_multiple == 0:
            unique_iou50_acc += 1
        else:
            multiple_iou50_acc += 1
        iou50_acc_list[scannet_locs.shape[0]] += 1
    count_list[scannet_locs.shape[0]] += 1
    # max_iou = 0.
    # for pred_id in range(instance_num):
    #     pred_locs = instance_locs[pred_id].tolist()
    #     gt_locs = scannet_locs[obj_id].tolist()
    #     pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
    #     gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
    #     iou = box3d_iou(pred_corners, gt_corners)
    #     if iou > max_iou:
    #         max_iou = iou
    # if max_iou >= 0.25:
    #     iou25_acc += 1
    # if max_iou >= 0.5:
    #     iou50_acc += 1
    # print(iou)
print(unique_all, multiple_all)
print(len(outputs))
print(f"Acc 0.25 {float(iou25_acc) / len(outputs)}")
print(f"Acc 0.50 {float(iou50_acc) / len(outputs)}")
print(f"Unique Acc 0.25 {float(unique_iou25_acc) / unique_all}")
print(f"Unique Acc 0.50 {float(unique_iou50_acc) / unique_all}")
print(f"Multiple Acc 0.25 {float(multiple_iou25_acc) / multiple_all}")
print(f"Multiple Acc 0.50 {float(multiple_iou50_acc) / multiple_all}")



split_nums = [0, 25, 35, 50, 70, 150]

for i in range(len(split_nums)-1):
    tot, iou25, iou50 = 0, 0, 0
    for j in range(split_nums[i], split_nums[i+1]):
        tot += count_list[j]
        iou25 += iou25_acc_list[j]
        iou50 += iou50_acc_list[j]
    print(f"{split_nums[i]} <= x < {split_nums[i+1]}: {tot} {iou25 / tot} {iou50 / tot}")

# for i in range(len(iou25_acc_list)):
#     iou25_acc_list[i] = iou25_acc_list[i] / count_list[i] if count_list[i] > 0 else 0
# for i in range(len(iou50_acc_list)):
#     iou50_acc_list[i] = iou50_acc_list[i] / count_list[i] if count_list[i] > 0 else 0



# print(iou25_acc_list)
# print(iou50_acc_list)