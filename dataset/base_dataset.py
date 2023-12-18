import logging
import os
import random
from torch.utils.data import Dataset
import torch
import glob

logger = logging.getLogger(__name__)


class PTBaseDataset(Dataset):

    def __init__(self):
        self.media_type = "point_cloud"
        self.anno = None
        self.attributes = None
        self.feats = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index, stage=2):
        scene_id = self.anno[index]["scene_id"]
        if "obj_id" in self.anno[index]:
            obj_id = int(self.anno[index]["obj_id"])
        else:
            obj_id = 0
        scene_attr = self.attributes[scene_id]
        obj_num = scene_attr["locs"].shape[0]
        scene_locs = scene_attr["locs"]
        scene_colors = scene_attr["colors"]
        obj_ids = scene_attr["obj_ids"] if "obj_ids" in scene_attr else [_i for _i in range(obj_num)]
        # scene_attr = torch.cat([scene_locs, scene_colors], dim=1)
        scene_feat = []
        feat_shape = 0
        for _i, _id in enumerate(obj_ids):
            item_id = "_".join([scene_id, f"{_id:02}"])
            if item_id not in self.feats:  # or torch.sum(torch.isnan(self.feats[item_id]) == 1) > 0:
                scene_feat.append(None)
            else:
                scene_feat.append(self.feats[item_id])
                feat_shape = scene_feat[-1].shape[0]
        for _i in range(len(scene_feat)):
            if scene_feat[_i] is None:
                scene_feat[_i] = torch.zeros(feat_shape)
        scene_feat = torch.stack(scene_feat, dim=0)
        return scene_id, obj_id, scene_feat, scene_locs, scene_colors


def process_batch_data(scene_feats, scene_locs, scene_colors):
    max_obj_num = max([e.shape[0] for e in scene_feats])
    # max_obj_num = 110
    batch_size = len(scene_feats)
    batch_scene_feat = torch.zeros(batch_size, max_obj_num, scene_feats[0].shape[-1])
    batch_scene_locs = torch.zeros(batch_size, max_obj_num, scene_locs[0].shape[-1])
    batch_scene_colors = torch.zeros(batch_size, max_obj_num, scene_colors[0].shape[-2], scene_colors[0].shape[-1])
    batch_scene_mask = torch.zeros(batch_size, max_obj_num, dtype=torch.long)
    for i in range(batch_size):
        batch_scene_feat[i][:scene_feats[i].shape[0]] = scene_feats[i]
        batch_scene_locs[i][:scene_locs[i].shape[0]] = scene_locs[i]
        batch_scene_colors[i][:scene_colors[i].shape[0]] = scene_colors[i]
        batch_scene_mask[i][:scene_feats[i].shape[0]] = 1
    return batch_scene_feat, batch_scene_locs, batch_scene_colors, batch_scene_mask


def replace_old_id(s, prefix_sum):

    def is_digit(char):
        return ord("0") <= ord(char) <= ord("9")

    ret = ""
    i = 0
    while i < len(s):
        if i + 4 < len(s) and (s[i:i+3] == "obj" or s[i:i+3] == "Obj"):
            ret += s[i:i+3]
            if is_digit(s[i+3]) and is_digit(s[i+4]):
                if i < len(s) - 5 and is_digit(s[i+5]):
                    old_id = int(s[i+3:i+6])
                    i += 6
                else:
                    old_id = int(s[i+3:i+5])
                    i += 5
                new_id = int(prefix_sum[old_id]) - 1
                ret += f"{new_id:02}"
            else:
                i += 3
        else:
            ret += s[i]
            i += 1
    return ret