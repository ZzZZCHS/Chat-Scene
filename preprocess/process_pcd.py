import torch
import os
import glob
import numpy as np
from tqdm import tqdm

scan_dir = "datasets/referit3d/scan_data"
output_dir = "datasets/referit3d/pcd_by_instance"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
pcd_dir = os.path.join(scan_dir, "pcd_with_global_alignment")

num_points = 1024

files = glob.glob(os.path.join(pcd_dir, "*.pth"))
files = sorted(files)
tot_objs = 0

for file in tqdm(files):
    print(file)
    scene_id = file[-16:-4]
    output_scene_dir = os.path.join(output_dir, scene_id)
    if not os.path.exists(output_scene_dir):
        os.mkdir(output_scene_dir)
    pcd_data = torch.load(file)
    pcds, instance_labels = pcd_data[0], pcd_data[-1]
    obj_pcds = []
    if instance_labels is None:
        continue
    for i in range(instance_labels.max() + 1):
        mask = instance_labels == i
        obj_pcds.append(pcds[mask])

    tot_objs += len(obj_pcds)

    # obj_fts = []
    for i, obj_pcd in enumerate(obj_pcds):
        pcd_idxs = np.random.choice(len(obj_pcd), size=num_points, replace=(len(obj_pcd) < num_points))
        obj_pcd = obj_pcd[pcd_idxs]
        print(len(obj_pcd))
        obj_pcd = obj_pcd - obj_pcd.mean(0)
        max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
        if max_dist < 1e-6:
            max_dist = 1
        obj_pcd = obj_pcd / max_dist
        # obj_fts.append(obj_pcd)
        torch.save(torch.tensor(obj_pcd), os.path.join(output_scene_dir, f"{i:03}.pt"))
    # exit()

print("tot:", tot_objs)



