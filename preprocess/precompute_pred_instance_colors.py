import os
import json
import glob
import torch
import numpy as np

from sklearn.mixture import GaussianMixture


scan_dir = '/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/mask3d_ins_data'
output_dir = os.path.join(scan_dir, 'instance_id_to_gmm_color')
os.makedirs(output_dir, exist_ok=True)

for scan_file in glob.glob(os.path.join(scan_dir, 'pcd_with_global_alignment', '*')):
    scan_id = os.path.basename(scan_file).split('.')[0]

    if os.path.exists(os.path.join(output_dir, '%s.json'%scan_id)):
        continue
    print(scan_file)

    data = torch.load(scan_file) # xyz, rgb, semantic_labels, instance_labels
    colors = data[1]
    instance_labels = data[3]

    if instance_labels is None:
        continue

    # normalize
    colors = colors / 127.5 - 1

    clustered_colors = []
    for i in range(instance_labels.max() + 1):
        mask = instance_labels == i     # time consuming
        obj_colors = colors[mask]
        if obj_colors.shape[0] <= 2:
            clustered_colors.append({
                'weights': [1., 0., 0.],
                'means': [
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]
                ]
            })
            continue

        gm = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(obj_colors)
        clustered_colors.append({
            'weights': gm.weights_.tolist(),
            'means': gm.means_.tolist(),
        })

    json.dump(
        clustered_colors,
        open(os.path.join(output_dir, '%s.json'%scan_id), 'w')
    )