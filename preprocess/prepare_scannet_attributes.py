from plyfile import PlyData
import numpy as np
import os
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--scannet_dir', required=True, type=str,
                    help='the path of the directory to scannet scans')
args = parser.parse_args()

raw_data_dir = os.path.join(args.scannet_dir, 'scans')


for split in ["train", "val"]:
    split_file = f"annotations/scannet/scannetv2_{split}.txt"
    scan_names = [line.rstrip() for line in open(split_file)]
    print(f'{split} split scans: {len(scan_names)}')
    outputs = {}
    for scan_id in tqdm(scan_names):
        aggregation_path = os.path.join(raw_data_dir, scan_id, scan_id + '.aggregation.json')
        segs_path = os.path.join(raw_data_dir, scan_id, scan_id + '_vh_clean_2.0.010000.segs.json')
        scan_ply_path = os.path.join(raw_data_dir, scan_id, scan_id + '_vh_clean_2.labels.ply')

        data = PlyData.read(scan_ply_path)
        x = np.asarray(data.elements[0].data['x']).astype(np.float32)
        y = np.asarray(data.elements[0].data['y']).astype(np.float32)
        z = np.asarray(data.elements[0].data['z']).astype(np.float32)
        pc = np.stack([x, y, z], axis=1)

        align_matrix = np.eye(4)
        with open(os.path.join(raw_data_dir, scan_id, '%s.txt'%(scan_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
                    break

        pts = np.ones((pc.shape[0], 4), dtype=pc.dtype)
        pts[:, 0:3] = pc
        pc = np.dot(pts, align_matrix.transpose())[:, :3]

        scan_aggregation = json.load(open(aggregation_path))
        segments_info = json.load(open(segs_path))
        segment_indices = segments_info["segIndices"]
        segment_indices_dict = defaultdict(list)
        for i, s in enumerate(segment_indices):
            segment_indices_dict[s].append(i)
        
        pc_segment_label = [''] * pc.shape[0]

        instance_labels = []
        inst_locs = []
        for idx, object_info in enumerate(scan_aggregation['segGroups']):
            object_instance_label = object_info['label']
            object_id = object_info['objectId']
            segments = object_info["segments"]
            pc_ids = []
            for s in segments:
                pc_ids.extend(segment_indices_dict[s])
            object_pc = pc[pc_ids]
            object_center = (np.max(object_pc, axis=0) + np.min(object_pc, axis=0)) / 2.0
            object_size = np.max(object_pc, axis=0) - np.min(object_pc, axis=0)
            object_bbox = torch.from_numpy(np.concatenate([object_center, object_size], axis=0))
            inst_locs.append(object_bbox)
            instance_labels.append(object_instance_label)
        inst_locs = torch.stack(inst_locs, dim=0)
        outputs[scan_id] = {
            'objects': instance_labels,
            'locs': inst_locs
        }
    
    torch.save(outputs, f"annotations/scannet_{split}_attributes.pt")

