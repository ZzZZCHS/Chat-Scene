import torch
import random
from utils.box_utils import box3d_iou, construct_bbox_corners
import re
from collections import defaultdict, OrderedDict
import json


def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def answer_match(pred, gts):
    # return EM and refined EM
    for gt in gts:
        if pred == gt:
            return 1, 1
        elif ''.join(pred.split()) in ''.join(gt.split()):
            return 0, 1
        elif ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


def calc_scanrefer_score(preds, config):
    instance_attribute_file = f"annotations/scannet_{config.segmentor}_val_attributes.pt"
    scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    iou25_acc = 0
    iou50_acc = 0

    count_list = [0] * 150
    iou25_acc_list = [0] * 150
    iou50_acc_list = [0] * 150

    for i, output in enumerate(preds):
        scene_id = output["scene_id"]
        obj_id = output["gt_id"]
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


def calc_scan2cap_score(preds, tokenizer, scorers, config):
    instance_attribute_file = f"annotations/scannet_{config.segmentor}_val_attributes.pt"
    scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    # pred_dict = {}
    gt_dict = json.load(open('annotations/scan2cap_val_corpus.json'))
    # for pred in preds:
    #     pred_key = f"{pred['scene_id']}|{pred['pred_id']}"
    #     pred_dict[pred_key] = f"sos {pred['pred']} eos".replace('\n', ' ')
    
    # candidates = {'caption': defaultdict(str), 'iou': defaultdict(float)}
    # for scene_id in instance_attrs.keys():
    #     instance_locs = instance_attrs[scene_id]['locs']
    #     scannet_locs = scannet_attrs[scene_id]['locs']
    #     for pred_id in range(len(instance_locs)):
    #         pred_locs = instance_locs[pred_id].tolist()
    #         pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
    #         match_id = -1
    #         match_iou = -1
    #         for gt_id in range(len(scannet_locs)):
    #             gt_locs = scannet_locs[gt_id].tolist()
    #             gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
    #             iou = box3d_iou(pred_corners, gt_corners)
    #             if iou > match_iou:
    #                 match_iou = iou
    #                 match_id = gt_id
    #         if match_id != -1:
    #             key = f"{scene_id}|{match_id}"
    #             if match_iou > candidates['iou'][key]:
    #                 candidates['iou'][key] = match_iou
    #                 candidates['caption'][key] = pred_dict[f"{scene_id}|{pred_id}"]
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
    
    # for missing_key in (gt_dict.keys() - candidates['caption'].keys()):
    #     candidates['caption'][missing_key] = "sos eos"
    # print("# missing keys:", len((gt_dict.keys() - candidates['caption'].keys())))
    
    # for item_id in gt_dict.keys():
    #     tmp_preds[item_id] = [{'caption': candidates['caption'][item_id]}]
    #     tmp_targets[item_id] = [{'caption': caption} for caption in gt_dict[item_id]]

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


def calc_scanqa_score(preds, tokenizer, scorers, config):
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
        ref_captions = [clean_answer(caption) for caption in output['ref_captions']]
        tmp_acc, tmp_refined_acc = answer_match(pred, ref_captions)
        acc += tmp_acc
        refined_acc += tmp_refined_acc
        tmp_preds[item_id] = [{'caption': pred}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
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


def calc_sqa3d_score(preds, tokenizer, scorers, config):
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
        sqa_type = output['sqa_type']
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