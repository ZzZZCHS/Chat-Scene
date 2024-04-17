import torch
import json
import os
from collections import defaultdict

# annos = json.load(open("annotations/obj_align_train_OBJ.json"))

# new_annos = []
# obj_ids = set()

# for anno in annos:
#     if anno['scene_id'] == "scene0000_00":
#         tmp_id = int(anno["caption"].split("OBJ")[1][:3])
#         if tmp_id in obj_ids:
#             continue
#         obj_ids.add(tmp_id)
#         if anno["caption"].startswith("<OBJ"):
#             anno["caption"] = "The " + anno["caption"]
        
#         anno["ref_captions"] = [anno["caption"]]
#         del anno["caption"]
#         new_annos.append(anno)

# print(len(new_annos))
# with open("annotations/obj_align_val_one_scene.json", "w") as f:
#     json.dump(new_annos, f, indent=4)

# annos = json.load(open("/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/annotations/scanrefer_mask3d_val_stage2_grounding_OBJ.json"))

# import random
# annos = random.sample(annos, 200)

# with open("annotations/scanrefer_mask3d_val_stage2_grounding_OBJ100.json", "w") as f:
#     json.dump(annos, f, indent=4)


# def recover_caption(clean_text, id_positions):
#     p = defaultdict(list)
#     sorted_id_positions_items = [(k, id_positions[k]) for k in sorted(id_positions.keys())]
#     for k, v in sorted_id_positions_items:
#         for interval in v:
#             p[interval[0]].append('[')
#             p[interval[1]].append(f"<OBJ{k:03}>")
#     caption = ''
#     for idx in range(len(clean_text)):
#         if idx in p:
#             if p[idx][0] == '[':
#                 caption += '['
#             else:
#                 caption += ' ' + ', '.join(p[idx]) + ']'
#         caption += clean_text[idx]
#     return caption


# scannet_root = '/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet'
# x = json.load(open('data/step2_captions_by_scene_v2.json'))

# train_annos = []
# val_annos = []

# train_scans = [line.rstrip() for line in open(os.path.join(scannet_root, f'train.txt'))]
# val_scans = [line.rstrip() for line in open(os.path.join(scannet_root, f'val.txt'))]


# import pandas as pd
# from tqdm import tqdm

# obj_csv = pd.read_csv('annotations/Cap3D_automated_Objaverse_no3Dword.csv', header=None)
# obj_ids = []
# obj_cap_dict = {}
# feats = torch.load('annotations/objaverse_uni3d_feature.pt')

# for obj_id, cap in tqdm(zip(obj_csv[0].values, obj_csv[1].values)):
#     # remove redundant quotation marks, here we do not directly strip because the mark may appear only at one side
#     if obj_id not in feats:
#         continue
#     if cap.startswith('"') and cap.endswith('"'):
#         cap = cap.strip('"')
#     elif cap.startswith("'") and cap.endswith("'"):
#         cap = cap.strip("'")
#     cap = cap.capitalize()
#     obj_ids.append(obj_id)
#     obj_cap_dict[obj_id] = cap

# train_annos = []
# val_annos = []
# train_obj_ids = obj_ids[:-1000]
# val_obj_ids = obj_ids[-1000:]


# for obj_id in train_obj_ids:
#     train_annos.append({
#         'scene_id': obj_id,
#         'caption': obj_cap_dict[obj_id]
#     })

# for obj_id in val_obj_ids:
#     val_annos.append({
#         'scene_id': obj_id,
#         'ref_captions': [obj_cap_dict[obj_id]]
#     })

# print(len(train_annos))
# print(len(val_annos))


# with open('annotations/objaverse_caption_train.json', 'w') as f:
#     json.dump(train_annos, f, indent=4)

# with open('annotations/objaverse_caption_val.json', 'w') as f:
#     json.dump(val_annos, f, indent=4)


# train_feats = {}
# val_feats = {}

# for obj_id in train_obj_ids:
#     train_feats[obj_id] = feats[obj_id]
# for obj_id in val_obj_ids:
#     val_feats[obj_id] = feats[obj_id]
    
# torch.save(train_feats, 'annotations/objaverse_uni3d_feature_train.pt')
# torch.save(val_feats, 'annotations/objaverse_uni3d_feature_val.pt')


# import os
# import gzip
# import numpy as np
# from tqdm import tqdm

# folder_path = '/mnt/petrelfs/huanghaifeng/share/data/cap3d/8192_npy'

# for filename in tqdm(os.listdir(folder_path)):
#     if filename.endswith('.npy'):
#         obj_id = filename.split('_8192')[0]
#         data = np.load(os.path.join(folder_path, filename))
#         with gzip.open(os.path.join(folder_path, obj_id + '.gz'), 'wb') as f:
#             np.save(f, data)
#         os.remove(os.path.join(folder_path, filename))

import csv
id2class = {}
labels = set()
class_label_file = "annotations/scannet/scannetv2-labels.combined.tsv"
with open(class_label_file, "r") as f:
    csvreader = csv.reader(f, delimiter='\t')
    csvreader.__next__()
    for line in csvreader:
        id2class[line[0]] = line[1]
        labels.add(line[2])
print(len(labels))

