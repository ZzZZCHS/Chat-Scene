import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, update_caption
import glob
import random
from prompts.prompts import obj_caption_wid_prompt
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class ValPTDataset(PTBaseDataset):

    def __init__(self, ann_list, dataset_name, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
        self.feats = torch.load(feat_file, map_location='cpu')
        self.img_feats = torch.load(img_feat_file, map_location='cpu') if img_feat_file is not None else None
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = json.load(open(anno_file, 'r'))
        if self.attributes is None:
            self.scene_feats = self.feats
            self.scene_img_feats = self.scene_masks = None
        else:
            self.scene_feats, self.scene_img_feats, self.scene_masks = self.prepare_scene_features()

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids = self.get_anno(index)
        obj_id = int(self.anno[index].get('obj_id', 0))
        pred_id = int(self.anno[index].get('pred_id', 0))
        sqa_type = int(self.anno[index].get('sqa_type', 0))
        if 'prompt' not in self.anno[index]:
            prompt = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            prompt = self.anno[index]["prompt"]
        ref_captions = self.anno[index]["ref_captions"].copy() if "ref_captions" in self.anno[index] else []
        qid = self.anno[index]["qid"] if "qid" in self.anno[index] else 0
        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, prompt, ref_captions, scene_id, qid, pred_id, sqa_type


def valuate_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_ids, assigned_ids, prompts, ref_captions, scene_ids, qids, pred_ids, sqa_types = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    obj_ids = torch.tensor(obj_ids)
    pred_ids = torch.tensor(pred_ids)
    sqa_types = torch.tensor(sqa_types)
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
        "obj_ids": obj_ids,
        "custom_prompt": prompts,
        "ref_captions": ref_captions,
        "scene_id": scene_ids,
        "qid": qids,
        "pred_ids": pred_ids,
        "sqa_types": sqa_types
        # "ids": index
    }

