import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import BaseDataset, update_caption
import glob
import random
from prompts.prompts import obj_caption_wid_prompt
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)



class TrainDataset(BaseDataset):

    cached_feats = {}

    def __init__(self, ann_list, config, **kwargs):
        super().__init__()
        self.feat_dim = config.model.input_dim
        self.img_feat_dim = config.model.img_input_dim
        self.max_obj_num = config.model.max_obj_num

        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = json.load(open(anno_file, 'r'))

        if len(ann_list) > 4:
            sample_ratio = ann_list[-1]
            if sample_ratio < 1:
                self.anno = random.sample(self.anno, int(sample_ratio * len(self.anno)))
        
        if feat_file in TrainDataset.cached_feats and img_feat_file in TrainDataset.cached_feats:
            self.scene_feats, self.scene_masks = TrainDataset.cached_feats[feat_file]
            self.scene_img_feats = TrainDataset.cached_feats[img_feat_file]
        else:
            if feat_file is not None and os.path.exists(feat_file):
                self.feats = torch.load(feat_file, map_location='cpu')
            else:
                self.feats = None
            if img_feat_file is not None and os.path.exists(img_feat_file):
                self.img_feats = torch.load(img_feat_file, map_location='cpu')
            else:
                self.img_feats = None
            if self.attributes is None:
                self.scene_feats = self.feats
                self.scene_img_feats = self.scene_masks = None
            else:
                self.scene_feats, self.scene_img_feats, self.scene_masks = self.prepare_scene_features()
            TrainDataset.cached_feats[feat_file] = (self.scene_feats, self.scene_masks)
            TrainDataset.cached_feats[img_feat_file] = self.scene_img_feats


    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        if self.attributes is not None and self.anno[index]['scene_id'] not in self.attributes:
            # print(f"{self.anno[index]['scene_id']} not in attribute file!")
            return self.__getitem__(random.randint(0, len(self.anno)-1))
        if "obj_id" in self.anno[index]:
            obj_id = int(self.anno[index]["obj_id"])
        else:
            obj_id = random.randint(0, self.max_obj_num - 1)
        if 'prompt' not in self.anno[index]:
            question = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            question = self.anno[index]["prompt"]
        caption = self.anno[index]["caption"]
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids = self.get_anno(index)
        caption = update_caption(caption, assigned_ids)
        question = update_caption(question, assigned_ids)
        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, assigned_ids, caption, question


def train_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_ids, assigned_ids, captions, questions = zip(*batch)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True).to(torch.bool)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    batch_assigned_ids = pad_sequence(assigned_ids, batch_first=True)
    # batch_detach_mask = torch.ones_like(batch_scene_mask, dtype=torch.bool)
    # for i in range(batch_detach_mask.shape[0]):
    #     batch_detach_mask[i][:detach_masks[i].shape[0]] = detach_masks[i]
    obj_ids = torch.tensor(obj_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        "assigned_ids": batch_assigned_ids,
        # "detach_mask": batch_detach_mask,
        "obj_ids": obj_ids,
        "answers": captions,
        "questions": questions
        # "ref_captions": ref_captions,
        # "ids": index
    }
