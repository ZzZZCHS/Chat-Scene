import torch
import json
import os
import glob
import numpy as np

encoder = "mask3d"

split = "val"
version = ""
scan_dir = f"/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/{encoder}_ins_data"
output_dir = "annotations"
split_path = f"annotations/scannet/scannetv2_{split}.txt"

scan_ids = []
with open(split_path, "r") as f:
    for l in f.readlines():
        scan_ids.append(l[:-1])

scan_ids = sorted(scan_ids)
# print(scan_ids)
from tqdm import tqdm

scans = {}
for scan_id in tqdm(scan_ids):
    if not os.path.exists(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)):
        continue
    inst_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))
    inst_locs = np.load(os.path.join(scan_dir, f'instance_id_to_loc{version}', f'{scan_id}.npy'))
    inst_colors = json.load(open(os.path.join(scan_dir, f'instance_id_to_gmm_color{version}', '%s.json' % scan_id)))
    inst_colors = [np.concatenate(
        [np.array(x['weights'])[:, None] if "weights" in x else np.array(x['weight'])[:, None], np.array(x['means'])],
        axis=1
    ).astype(np.float32) for x in inst_colors]
    # inst_colors = np.array(inst_colors).tolist()
    # inst_colors = list(map(lambda x: x[0][1:], inst_colors))
    # inst_locs = inst_locs.tolist()
    assert(len(inst_colors) == len(inst_locs))
    # print(inst_labels, inst_colors, inst_locs)
    # exit()
    inst_locs = torch.tensor(inst_locs, dtype=torch.float32)
    inst_colors = torch.tensor(inst_colors, dtype=torch.float32)
    breakpoint()
    scans[scan_id] = {
        'objects': inst_labels,  # (n_obj, )
        'locs': inst_locs,  # (n_obj, 6) center xyz, whl
        'colors': inst_colors,  # (n_obj, 3x4) cluster * (weight, mean rgb)
    }

# with open(os.path.join(output_dir, f"scannet_pointgroup_{split}_attributes.json"), "w") as f:
#     json.dump(scans, f)
torch.save(scans, os.path.join(output_dir, f"scannet_{encoder}_{split}_attributes{version}.pt"))