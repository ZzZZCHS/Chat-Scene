import torch
import random
import sys
sys.path.append('.')
from utils.box_utils import box3d_iou, construct_bbox_corners
import re
from collections import defaultdict, OrderedDict
import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.helper import clean_answer, answer_match, scanrefer_get_unique_multiple_lookup

# default_instance_attr_file = "annotations/scannet_mask3d_val_attributes.pt"
default_instance_attr_file = 'annotations/scannet_deva_attributes_new.pt'

def calc_scanrefer_score(preds, config=None):
    instance_attribute_file = config.val_file_dict['scanrefer'][2] if config is not None else default_instance_attr_file
    scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    unique_multiple_lookup = scanrefer_get_unique_multiple_lookup()

    iou25_acc = 0
    iou50_acc = 0
    unique_iou25_acc = 0
    unique_iou50_acc = 0
    unique_all = 0
    multiple_iou25_acc = 0
    multiple_iou50_acc = 0
    multiple_all = 0

    # count_list = [0] * 150
    # iou25_acc_list = [0] * 150
    # iou50_acc_list = [0] * 150
    id_format = "<OBJ\\d{3}>"

    for i, output in enumerate(preds):
        scene_id = output["scene_id"]
        obj_id = output["gt_id"]
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        unique_multiple = unique_multiple_lookup[scene_id][str(obj_id)]
        if unique_multiple == 0:
            unique_all += 1
        else:
            multiple_all += 1
        pred = output["pred"]
        instance_num = instance_locs.shape[0]
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
            # iou25_acc_list[scannet_locs.shape[0]] += 1
        if iou >= 0.5:
            iou50_acc += 1
            if unique_multiple == 0:
                unique_iou50_acc += 1
            else:
                multiple_iou50_acc += 1
            # iou50_acc_list[scannet_locs.shape[0]] += 1
        # count_list[scannet_locs.shape[0]] += 1

    val_scores = {
        '[scanrefer] Acc@0.25': float(iou25_acc) / len(preds),
        '[scanrefer] Acc@0.50': float(iou50_acc) / len(preds),
        '[scanrefer] Unique Acc@0.25': float(unique_iou25_acc) / unique_all,
        '[scanrefer] Unique Acc@0.50': float(unique_iou50_acc) / unique_all,
        '[scanrefer] Multiple Acc@0.25': float(multiple_iou25_acc) / multiple_all,
        '[scanrefer] Multiple Acc@0.50': float(multiple_iou50_acc) / multiple_all
    }

    return val_scores


def calc_nr3d_score(preds, config=None):
    instance_attribute_file = config.val_file_dict['scanrefer'][2] if config is not None else default_instance_attr_file
    scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    unique_multiple_lookup = scanrefer_get_unique_multiple_lookup()

    iou25_acc = 0
    iou50_acc = 0
    unique_iou25_acc = 0
    unique_iou50_acc = 0
    unique_all = 0
    multiple_iou25_acc = 0
    multiple_iou50_acc = 0
    multiple_all = 0

    # count_list = [0] * 150
    # iou25_acc_list = [0] * 150
    # iou50_acc_list = [0] * 150
    id_format = "<OBJ\\d{3}>"

    for i, output in enumerate(preds):
        scene_id = output["scene_id"]
        obj_id = output["gt_id"]
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        unique_multiple = unique_multiple_lookup[scene_id][str(obj_id)]
        if unique_multiple == 0:
            unique_all += 1
        else:
            multiple_all += 1
        pred = output["pred"]
        instance_num = instance_locs.shape[0]
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
            # iou25_acc_list[scannet_locs.shape[0]] += 1
        if iou >= 0.5:
            iou50_acc += 1
            if unique_multiple == 0:
                unique_iou50_acc += 1
            else:
                multiple_iou50_acc += 1
            # iou50_acc_list[scannet_locs.shape[0]] += 1
        # count_list[scannet_locs.shape[0]] += 1

    val_scores = {
        '[nr3d] Acc@0.25': float(iou25_acc) / len(preds),
        '[nr3d] Acc@0.50': float(iou50_acc) / len(preds),
        '[nr3d] Unique Acc@0.25': float(unique_iou25_acc) / unique_all,
        '[nr3d] Unique Acc@0.50': float(unique_iou50_acc) / unique_all,
        '[nr3d] Multiple Acc@0.25': float(multiple_iou25_acc) / multiple_all,
        '[nr3d] Multiple Acc@0.50': float(multiple_iou50_acc) / multiple_all
    }

    return val_scores


def calc_multi3dref_score(preds, config=None):
    instance_attribute_file = config.val_file_dict['multi3dref'][2] if config is not None else default_instance_attr_file
    scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    id_format = "<OBJ\\d{3}>"

    evaluation_types = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}
    eval_type_mask = np.empty(len(preds), dtype=np.uint8)
    # iou_25_f1_scores = np.empty(len(preds), dtype=np.float32)
    # iou_50_f1_scores = np.empty(len(preds), dtype=np.float32)
    iou_25_f1_scores = defaultdict(list)
    iou_50_f1_scores = defaultdict(list)

    for i, pred in enumerate(preds):
        scene_id = pred['scene_id']
        obj_id = pred['gt_id']
        gt_ids = pred['ref_captions']
        pred_sentence = pred['pred']
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        pred_ids = []
        for match in re.finditer(id_format, pred_sentence):
            idx = match.start()
            cur_id = int(pred_sentence[idx+4:idx+7])
            if cur_id < instance_num:
                pred_ids.append(cur_id)
        eval_type = pred['type_info']
        eval_type_mask[i] = evaluation_types[eval_type]
        iou_25_f1, iou_50_f1 = 0, 0
        if eval_type in ['zt_wo_d', 'zt_w_d']:
            if len(pred_ids) == 0:
                iou_25_f1 = iou_50_f1 = 1
            else:
                iou_25_f1 = iou_50_f1 = 0
        else:
            pred_corners_list = []
            gt_corners_list = []
            for pred_id in pred_ids:
                pred_locs = instance_locs[pred_id].tolist()
                pred_corners_list.append(construct_bbox_corners(pred_locs[:3], pred_locs[3:]))
            for gt_id in gt_ids:
                gt_locs = scannet_locs[gt_id].tolist()
                gt_corners_list.append(construct_bbox_corners(gt_locs[:3], gt_locs[3:]))
            square_matrix_len = max(len(pred_ids), len(gt_ids))
            iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
            for pred_idx, pred_corners in enumerate(pred_corners_list):
                for gt_idx, gt_corners in enumerate(gt_corners_list):
                    iou_matrix[pred_idx, gt_idx] = box3d_iou(pred_corners, gt_corners)
            iou_25_tp = 0
            iou_50_tp = 0
            row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)
            for ii in range(len(pred_ids)):
                iou = iou_matrix[row_idx[ii], col_idx[ii]]
                if iou >= 0.25:
                    iou_25_tp += 1
                if iou >= 0.5:
                    iou_50_tp += 1
            iou_25_f1 = 2 * iou_25_tp / (len(pred_ids) + len(gt_ids))
            iou_50_f1 = 2 * iou_50_tp / (len(pred_ids) + len(gt_ids))
        iou_25_f1_scores['all'].append(iou_25_f1)
        iou_50_f1_scores['all'].append(iou_50_f1)
        iou_25_f1_scores[eval_type].append(iou_25_f1)
        iou_50_f1_scores[eval_type].append(iou_50_f1)

    val_scores = {}
    for k in iou_25_f1_scores.keys():
        val_scores[f"[multi3dref] {k} F1@0.25"] = np.mean(iou_25_f1_scores[k])
        val_scores[f"[multi3dref] {k} F1@0.50"] = np.mean(iou_50_f1_scores[k])
    return val_scores


def calc_scan2cap_score(preds, tokenizer, scorers, config=None):
    instance_attribute_file = config.val_file_dict['scan2cap'][2] if config is not None else default_instance_attr_file
    scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    gt_dict = json.load(open('annotations/scan2cap_val_corpus.json'))
    tmp_preds_iou25 = {}
    tmp_preds_iou50 = {}
    tmp_targets = {}
    for pred in preds:
        scene_id = pred['scene_id']
        pred_id = pred['pred_id']
        gt_id = pred['gt_id']
        pred_locs = instance_attrs[scene_id]['locs'][pred_id].tolist()
        gt_locs = scannet_attrs[scene_id]['locs'][gt_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        key = f"{scene_id}|{gt_id}"
        if iou >= 0.25:
            tmp_preds_iou25[key] = [{'caption': f"sos {pred['pred']} eos".replace('\n', ' ')}]
        else:
            tmp_preds_iou25[key] = [{'caption': f"sos eos"}]
        if iou >= 0.5:
            tmp_preds_iou50[key] = [{'caption': f"sos {pred['pred']} eos".replace('\n', ' ')}]
        else:
            tmp_preds_iou50[key] = [{'caption': f"sos eos"}]
        tmp_targets[key] = [{'caption': caption} for caption in gt_dict[key]]
    
    missing_keys = gt_dict.keys() - tmp_targets.keys()

    for missing_key in missing_keys:
        tmp_preds_iou25[missing_key] = [{'caption': "sos eos"}]
        tmp_preds_iou50[missing_key] = [{'caption': "sos eos"}]
        tmp_targets[missing_key] = [{'caption': caption} for caption in gt_dict[missing_key]]
    
    tmp_preds_iou25 = tokenizer.tokenize(tmp_preds_iou25)
    tmp_preds_iou50 = tokenizer.tokenize(tmp_preds_iou50)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    val_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds_iou25)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scan2cap] {m}@0.25"] = sc
        else:
            val_scores[f"[scan2cap] {method}@0.25"] = score
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds_iou50)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scan2cap] {m}@0.50"] = sc
        else:
            val_scores[f"[scan2cap] {method}@0.50"] = score
    return val_scores


def calc_scanqa_score(preds, tokenizer, scorers, config=None):
    with open(f"annotations/scanqa/ScanQA_v1.0_val.json", "r") as f:
        annos = json.load(f)
    outputs = []
    qset = set()
    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    acc, refined_acc = 0, 0
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        item_id = f"{output['scene_id']}_{output['gt_id']}_{output['qid']}_{i}"
        pred = output["pred"]
        if len(pred) > 1:
            if pred[-1] == '.':
                pred = pred[:-1]
            pred = pred[0].lower() + pred[1:]
        pred = clean_answer(pred)
        ref_captions = [caption for caption in output['ref_captions']]
        tmp_acc, tmp_refined_acc = answer_match(pred, ref_captions)
        acc += tmp_acc
        refined_acc += tmp_refined_acc
        tmp_preds[item_id] = [{'caption': pred}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]

    #     ori_question = output['prompt'].split(' Answer the question using a single word or phrase.')[0]
    #     flag = 0
    #     for anno in annos:
    #         if anno['scene_id'] == output['scene_id'] and anno['question'] == ori_question:
    #             flag = 1
    #             outputs.append({
    #                 'scene_id': anno['scene_id'],
    #                 'question_id': anno['question_id'],
    #                 'answer_top10': [pred] * 10,
    #                 'bbox': []
    #             })
    #             qset.add(anno['question_id'])
    #             break
    #     if flag == 0:
    #         breakpoint()
    # print(len(annos))
    # print(len(qset))
    # print(len(outputs))
    json.dump(outputs, open('scanqa_val.json', 'w'), indent=4)
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    acc = acc / len(preds)
    refined_acc = refined_acc / len(preds)
    val_scores["[scanqa] EM1"] = acc
    val_scores["[scanqa] EM1_refined"] = refined_acc
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[scanqa] {m}"] = sc
        else:
            val_scores[f"[scanqa] {method}"] = score
    return val_scores


def calc_sqa3d_score(preds, tokenizer, scorers, config=None):
    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    metrics = {
        'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
        'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10,
    }
    em_overall = 0
    em_refined_overall = 0
    em_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    em_refined_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        item_id = f"{output['scene_id']}_{output['gt_id']}_{output['qid']}_{i}"
        pred = output["pred"]
        if len(pred) > 1:
            if pred[-1] == '.':
                pred = pred[:-1]
            pred = pred[0].lower() + pred[1:]
        pred = clean_answer(pred)
        ref_captions = [clean_answer(caption) for caption in output['ref_captions']]
        em_flag, em_refined_flag = answer_match(pred, ref_captions)
        em_overall += em_flag
        em_refined_overall += em_refined_flag
        sqa_type = int(output['type_info'])
        em_type[sqa_type] += em_flag
        em_refined_type[sqa_type] += em_refined_flag
        metrics[f'type{sqa_type}_count'] += 1
        tmp_preds[item_id] = [{'caption': pred}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    em_overall = em_overall / len(preds)
    em_refined_overall = em_refined_overall / len(preds)
    val_scores["[sqa3d] EM1"] = em_overall
    val_scores["[sqa3d] EM1_refined"] = em_refined_overall
    for key in em_type.keys():
        val_scores[f'[sqa3d] EM_type{key}'] = em_type[key] / metrics[f'type{key}_count']
        val_scores[f'[sqa3d] EM_refined_type{key}'] = em_refined_type[key] / metrics[f'type{key}_count']
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                val_scores[f"[sqa3d] {m}"] = sc
        else:
            val_scores[f"[sqa3d] {method}"] = score
    return val_scores


if __name__ == '__main__':
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    # from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    saved_preds = json.load(open('/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240504_213518_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#scannet_caption#scannet_region_caption#nr3d_caption__scanrefer#scan2cap#scanqa#multi3dref#sqa3d__mask3d_video/preds_epoch1_step3812_scanqa.json'))
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]
    tokenizer = PTBTokenizer()
    val_scores = calc_scanqa_score(saved_preds, tokenizer=tokenizer, scorers=scorers)
    print(json.dumps(val_scores, indent=4))
