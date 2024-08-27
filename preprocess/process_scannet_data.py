from plyfile import PlyData
import numpy as np
import os
import json
from pytorch3d.io import load_obj
import torch
from collections import defaultdict
from tqdm import tqdm
import mmengine

data_root = '/mnt/petrelfs/share_data/huanghaifeng/maoxiaohan/ScanNet_v2'
raw_data_dir = os.path.join(data_root, 'scans')
meta_data_dir = os.path.join(data_root, 'meta_data')
output_dir = '/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet'
idx2class = json.load(open('/mnt/petrelfs/share_data/huanghaifeng/referit3d/referit3d/data/mappings/scannet_idx_to_semantic_class.json'))
idx2class = {int(k): v for k, v in idx2class.items()}
class2idx = {v: k for k, v in idx2class.items()}
scan2axis_align = json.load(open('/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/scans_axis_alignment_matrices.json'))


def process_one_scan(scan_id):
    save_dir = os.path.join(output_dir, 'scans', scan_id)
    if os.path.exists(os.path.join(save_dir, 'object_infos.json')):
        return
    # label_path = os.path.join(raw_data_dir, scan_id, "labels.instances.annotated.v2.ply")
    aggregation_path = os.path.join(raw_data_dir, scan_id, scan_id + '.aggregation.json')
    segs_path = os.path.join(raw_data_dir, scan_id, scan_id + '_vh_clean_2.0.010000.segs.json')
    scan_ply_path = os.path.join(raw_data_dir, scan_id, scan_id + '_vh_clean_2.labels.ply')

    data = PlyData.read(scan_ply_path)
    x = np.asarray(data.elements[0].data['x']).astype(np.float32)
    y = np.asarray(data.elements[0].data['y']).astype(np.float32)
    z = np.asarray(data.elements[0].data['z']).astype(np.float32)
    pc = np.stack([x, y, z], axis=1)

    axis_align_matrix = np.array(scan2axis_align[scan_id], dtype=np.float32).reshape(4, 4)
    pts = np.ones((pc.shape[0], 4), dtype=pc.dtype)
    pts[:, :3] = pc
    pc = np.dot(pts, axis_align_matrix.transpose())[:, :3]

    scan_aggregation = json.load(open(aggregation_path))
    segments_info = json.load(open(segs_path))
    segment_indices = segments_info["segIndices"]
    segment_indices_dict = defaultdict(list)
    for i, s in enumerate(segment_indices):
        segment_indices_dict[s].append(i)
     
    pc_instance_id = np.zeros(pc.shape[0]).astype(np.int32) * -1
    # pc_semantic_label_id = np.zeros(pc.shape[0]).astype(np.int32) * -1
    pc_segment_label = [''] * pc.shape[0]

    valid_ids = []
    all_objects = []
    for idx, object_info in enumerate(scan_aggregation['segGroups']):
        object_instance_label = object_info['label']
        object_id = object_info['objectId']
        segments = object_info["segments"]
        valid_ids.append(idx)
        pc_ids = []
        for s in segments:
            pc_ids.extend(segment_indices_dict[s])
        pc_instance_id[pc_ids] = object_id
        object_pc = pc[pc_ids]
        object_center = (np.max(object_pc, axis=0) + np.min(object_pc, axis=0)) / 2.0
        object_size = np.max(object_pc, axis=0) - np.min(object_pc, axis=0)
        object_bbox = np.concatenate([object_center, object_size], axis=0)
        all_objects.append({
            'bbox': object_bbox.tolist(),
            'label': object_instance_label
        })
    object_infos = {
        'valid_ids': valid_ids,
        'object_list': all_objects
    }
    
    save_dir = os.path.join(output_dir, 'scans', scan_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'object_infos.json'), 'w') as f:
        json.dump(object_infos, f, indent=4)
    # np.save(os.path.join(save_dir, 'object_infos.npy'), object_infos)
    # np.save(os.path.join(save_dir, 'axis_align_matrix.npy'), axis_align_matrix)


def process_split(split):
    assert split in ['train', 'val', 'test']
    split_file = os.path.join(meta_data_dir, f'scannetv2_{split}.txt')
    scan_names = [line.rstrip() for line in open(split_file)]
    print(f'{split} split scans: {len(scan_names)}')
    # new_split_file = os.path.join(output_dir, f'{split}.txt')
    # valid_names = []
    # for scan_name in tqdm(scan_names):
    #     if scan_name in mapping:
    #         new_scan_name = mapping[scan_name]
    #         process_one_scan(scan_name, new_scan_name)
    #         valid_names.append(scan_name)
    # params = []
    # for scan_name in scan_names:
    #     if scan_name in mapping:
    #         new_scan_name = mapping[scan_name]
    #         params.append((scan_name, new_scan_name))
    
    parallel = True

    if parallel:
        mmengine.utils.track_parallel_progress(process_one_scan, scan_names, 8)
    else:
        for scan_id in tqdm(scan_names):
            process_one_scan(scan_id)
    
    # if not os.path.exists(new_split_file):
    #     with open(new_split_file, 'w') as f:
    #         for t in valid_names:
    #             f.write(f'{t}\n')
    

for s in ['train', 'val']:
    process_split(s)

