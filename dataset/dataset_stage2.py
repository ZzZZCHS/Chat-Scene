import logging
import os
import json

import numpy as np
import torch

from dataset.base_dataset import PTBaseDataset, process_batch_data, replace_old_id, extract_all_ids
import glob
import random
from prompts.prompts import obj_caption_wid_prompt

logger = logging.getLogger(__name__)



class S2PTDataset(PTBaseDataset):

    def __init__(self, ann_list, **kwargs):
        super().__init__()
        feat_file, img_feat_file, attribute_file, anno_file = ann_list[:4]
        self.feats = torch.load(feat_file, map_location='cpu')
        self.img_feats = torch.load(img_feat_file, map_location='cpu') if img_feat_file is not None else None
        self.attributes = torch.load(attribute_file, map_location='cpu') if attribute_file is not None else None
        self.anno = json.load(open(anno_file, 'r'))

        if len(ann_list) > 4:
            sample_ratio = ann_list[-1]
            if sample_ratio < 1:
                self.anno = random.sample(self.anno, int(sample_ratio * len(self.anno)))
        if self.attributes is None:
            self.scene_feats = self.feats
            self.scene_img_feats = self.scene_masks = None
        else:
            self.scene_feats, self.scene_img_feats, self.scene_masks = self.prepare_scene_features()

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        if self.attributes is not None and self.anno[index]['scene_id'] not in self.attributes:
            print(f"{self.anno[index]['scene_id']} not in attribute file!!!")
            return self.__getitem__(random.randint(0, len(self.anno)-1))
        if "obj_id" in self.anno[index]:
            obj_id = int(self.anno[index]["obj_id"])
        else:
            obj_id = random.randint(0, 199)
        if 'prompt' not in self.anno[index]:
            question = random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{obj_id:03}>")
        else:
            question = self.anno[index]["prompt"]
        caption = self.anno[index]["caption"]
        scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs = self.get_anno(index)
        return scene_feat, scene_img_feat, scene_mask, scene_locs, obj_id, caption, question


def s2_collate_fn(batch):
    scene_feats, scene_img_feats, scene_masks, scene_locs, obj_ids, captions, questions = zip(*batch)
    batch_scene_feat, batch_scene_img_feat, batch_scene_locs, batch_scene_mask = process_batch_data(
        scene_feats,
        scene_img_feats,
        scene_masks,
        scene_locs
    )
    # batch_detach_mask = torch.ones_like(batch_scene_mask, dtype=torch.bool)
    # for i in range(batch_detach_mask.shape[0]):
    #     batch_detach_mask[i][:detach_masks[i].shape[0]] = detach_masks[i]
    obj_ids = torch.tensor(obj_ids)
    return {
        "scene_feat": batch_scene_feat,
        "scene_img_feat": batch_scene_img_feat,
        "scene_locs": batch_scene_locs,
        "scene_mask": batch_scene_mask,
        # "detach_mask": batch_detach_mask,
        "obj_ids": obj_ids,
        "answers": captions,
        "questions": questions
        # "ref_captions": ref_captions,
        # "ids": index
    }
