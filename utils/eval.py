import torch
import random
from utils.box_utils import box3d_iou, construct_bbox_corners


def calc_scanrefer_acc(preds, config):
    instance_attribute_file = f"annotations/scannet_{config.segmentor}_val_attributes.pt"
    scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file)
    scannet_attrs = torch.load(scannet_attribute_file)

    iou25_acc = 0
    iou50_acc = 0

    count_list = [0] * 150
    iou25_acc_list = [0] * 150
    iou50_acc_list = [0] * 150

    for i, output in enumerate(preds):
        scene_id = output["scene_id"]
        obj_id = output["obj_id"]
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        pred = output["pred"]
        instance_num = instance_locs.shape[0]
        pred_id = random.randint(0, instance_num-1)
        flag = 0
        if "OBJ" in pred:
            tmp = pred.split("OBJ")[1]
            j = 0
            while tmp[:j+1].isdigit() and j < len(tmp):
                j = j + 1
            if j > 0:
                flag = 1
                if int(tmp[:j]) < instance_num:
                    pred_id = int(tmp[:j])
        pred_locs = instance_locs[pred_id].tolist()
        gt_locs = scannet_locs[obj_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        if iou >= 0.25:
            iou25_acc += 1
            iou25_acc_list[scannet_locs.shape[0]] += 1
        if iou >= 0.5:
            iou50_acc += 1
            iou50_acc_list[scannet_locs.shape[0]] += 1
        count_list[scannet_locs.shape[0]] += 1

    # print(f"Acc 0.25 {float(iou25_acc) / len(preds)}")
    # print(f"Acc 0.50 {float(iou50_acc) / len(preds)}")
    val_scores = {
        '[scanrefer] Acc@0.25': float(iou25_acc) / len(preds),
        '[scanrefer] Acc@0.50': float(iou50_acc) / len(preds)
    }

    return val_scores