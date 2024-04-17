import numpy as np
import json
import sys
import torch
import random
from tqdm import tqdm

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


# outputs = json.load(open("outputs/2023-11-15-231032_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign_mean/preds_epoch-1_step0.json", "r"))


segmentor = "mask3d"
# annos = json.load(open(f"annotations/scanrefer_{split}_stage2_grounding_OBJ.json", "r"))
feats = torch.load(f'annotations/scannet_img_dinov2_features.pt')
new_feats = {}
item2iou = {}
version = "100_v3"

for split in ['val']:
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
        if max_iou > 0 and (item_id not in new_feats or item2iou[item_id] < max_iou):
            new_feats[item_id] = v
            item2iou[item_id] = max_iou
        

torch.save(new_feats, f'annotations/scannet_img_mask3d_dinov2_features{version}.pt')