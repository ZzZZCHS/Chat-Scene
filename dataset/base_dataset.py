import logging
import os
import random
from torch.utils.data import Dataset
import torch
import glob
from torch.nn.utils.rnn import pad_sequence
import re

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):

    def __init__(self):
        self.media_type = "point_cloud"
        self.anno = None
        self.attributes = None
        self.feats = None
        self.img_feats = None
        self.scene_feats = None
        self.scene_img_feats = None
        self.scene_masks = None
        self.feat_dim = 1024
        self.img_feat_dim = 1024

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def prepare_scene_features(self):
        scan_ids = set('_'.join(x.split('_')[:2]) for x in self.feats.keys())
        scene_feats = {}
        scene_img_feats = {}
        scene_masks = {}
        unwanted_words = ["wall", "ceiling", "floor", "object", "item"]
        for scan_id in scan_ids:
            if scan_id not in self.attributes:
                continue
            scene_attr = self.attributes[scan_id]
            obj_num = scene_attr['locs'].shape[0]
            obj_ids = scene_attr['obj_ids'] if 'obj_ids' in scene_attr else [_ for _ in range(obj_num)]
            obj_labels = scene_attr['objects']
            scene_feat = []
            scene_img_feat = []
            scene_mask = []
            for _i, _id in enumerate(obj_ids):
                item_id = '_'.join([scan_id, f'{_id:02}'])
                if item_id not in self.feats:
                    # scene_feat.append(torch.randn((self.feat_dim)))
                    scene_feat.append(torch.zeros(self.feat_dim))
                else:
                    scene_feat.append(self.feats[item_id])
                if self.img_feats is None or item_id not in self.img_feats:
                    # scene_img_feat.append(torch.randn((self.img_feat_dim)))
                    scene_img_feat.append(torch.zeros(self.img_feat_dim))
                else:
                    scene_img_feat.append(self.img_feats[item_id].float())
                if scene_feat[-1] is None or any(x in obj_labels[_id] for x in unwanted_words):
                    scene_mask.append(0)
                else:
                    scene_mask.append(1)
            scene_feat = torch.stack(scene_feat, dim=0)
            scene_img_feat = torch.stack(scene_img_feat, dim=0)
            scene_mask = torch.tensor(scene_mask, dtype=torch.int)
            scene_feats[scan_id] = scene_feat
            scene_img_feats[scan_id] = scene_img_feat
            scene_masks[scan_id] = scene_mask
        return scene_feats, scene_img_feats, scene_masks

    def get_anno(self, index):
        scene_id = self.anno[index]["scene_id"]
        if self.attributes is not None:
            scene_attr = self.attributes[scene_id]
            # obj_num = scene_attr["locs"].shape[0]
            scene_locs = scene_attr["locs"]
        else:
            scene_locs = torch.randn((1, 6))
        scene_feat = self.scene_feats[scene_id]
        if scene_feat.ndim == 1:
            scene_feat = scene_feat.unsqueeze(0)
        scene_img_feat = self.scene_img_feats[scene_id] if self.scene_img_feats is not None else torch.zeros((scene_feat.shape[0], self.img_feat_dim))
        scene_mask = self.scene_masks[scene_id] if self.scene_masks is not None else torch.ones(scene_feat.shape[0], dtype=torch.int)
        assigned_ids = torch.randperm(200)[:len(scene_locs)]
        # assigned_ids = torch.randperm(len(scene_locs))
        return scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids
    

def update_caption(caption, new_ids):
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        old_id = int(caption[idx+4:idx+7])
        new_id = int(new_ids[old_id])
        caption = caption[:idx+4] + f"{new_id:03}" + caption[idx+7:]
    return caption


def recover_caption(caption, new_ids):
    old_ids = {new_id: i for i, new_id in enumerate(new_ids)}
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        new_id = int(caption[idx+4:idx+7])
        try:
            old_id = int(old_ids[new_id])
        except:
            old_id = random.randint(0, len(new_ids)-1)
        caption = caption[:idx+4] + f"{old_id:03}" + caption[idx+7:]
    return caption


if __name__ == "__main__":
    caption = "<OBJ001> <OBJ002>"
    assigned_ids = [1, 2, 3]
    caption = update_caption(caption, assigned_ids)
    print(caption)
    caption = recover_caption(caption, assigned_ids)
    print(caption)