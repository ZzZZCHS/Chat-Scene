import os
import argparse
import json
import numpy as np
import pprint
import time
import multiprocessing as mp
from functools import partial

from plyfile import PlyData

import torch

import csv
import glob
from collections import defaultdict
from tqdm import tqdm

import mmengine

ids = set()
def process_one_scene(params, tmp_dir, id2class):
    cur_dir, file_path = params
    scene_id = file_path.split("/")[-1].split(".txt")[0]
    cur_scene_out = []
    save_file = os.path.join(tmp_dir, f"{scene_id}.pt")
    if os.path.exists(save_file):
        return
    with open(file_path, "r") as f:
        for line in f.readlines():
            predict_path, class_id, score = line.split(" ")
            ids.add(class_id)
            tmp = predict_path.split('_')
            predict_path = os.path.join('_'.join(tmp[:-1]), tmp[-1])
            predict_path = os.path.join(cur_dir, predict_path)
            segments = []
            with open(predict_path, "r") as f2:
                for i, l in enumerate(f2.readlines()):
                    if l[0] == "1":
                        segments.append(i)
            cur_scene_out.append({
                "label": id2class[class_id],
                "segments": segments
            })
    torch.save(cur_scene_out, save_file)

def process_per_scan(scan_id, scan_dir, out_dir, tmp_dir, apply_global_alignment=True, is_test=False):
    pcd_out_dir = os.path.join(out_dir, 'pcd_all')
    # if os.path.exists(os.path.join(pcd_out_dir, '%s.pth'%(scan_id))):
    #     print(f"skipping {scan_id}...")
    #     return
    print(f"processing {scan_id}...")
    os.makedirs(pcd_out_dir, exist_ok=True)
    # obj_out_dir = os.path.join(out_dir, 'instance_id_to_name_all')
    # os.makedirs(obj_out_dir, exist_ok=True)

    # Load point clouds with colors
    with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.ply'%(scan_id)), 'rb') as f:
        plydata = PlyData.read(f) # elements: vertex, face
    points = np.array([list(x) for x in plydata.elements[0]]) # [[x, y, z, r, g, b, alpha]]
    coords = np.ascontiguousarray(points[:, :3])
    colors = np.ascontiguousarray(points[:, 3:6])

    # # TODO: normalize the coords and colors
    # coords = coords - coords.mean(0)
    # colors = colors / 127.5 - 1

    if apply_global_alignment:
        align_matrix = np.eye(4)
        with open(os.path.join(scan_dir, scan_id, '%s.txt'%(scan_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
                    break
        # Transform the points
        pts = np.ones((coords.shape[0], 4), dtype=coords.dtype)
        pts[:, 0:3] = coords
        coords = np.dot(pts, align_matrix.transpose())[:, :3]  # Nx4
        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(coords)) == 0)

    # Load point labels if any
    # colored by nyu40 labels (ply property 'label' denotes the nyu40 label id)
    # with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply'%(scan_id)), 'rb') as f:
    #     plydata = PlyData.read(f)
    # sem_labels = np.array(plydata.elements[0]['label']).astype(np.long)
    # assert len(coords) == len(colors) == len(sem_labels)
    # sem_labels = None

    # Map each point to segment id
    # if not os.path.exists(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json'%(scan_id))):
    #     return
    # with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json'%(scan_id)), 'r') as f:
    #     d = json.load(f)
    # seg = d['segIndices']
    # segid_to_pointid = {}
    # for i, segid in enumerate(seg):
    #     segid_to_pointid.setdefault(segid, [])
    #     segid_to_pointid[segid].append(i)

    # Map object to segments
    instance_class_labels = []
    instance_segids = []

    # cur_instances = pointgroup_instances[scan_id].copy()
    if not os.path.exists(os.path.join(tmp_dir, f"{scan_id}.pt")):
        return
    cur_instances = torch.load(os.path.join(tmp_dir, f"{scan_id}.pt"))
    for instance in cur_instances:
        instance_class_labels.append(instance["label"])
        instance_segids.append(instance["segments"])

    torch.save(
        (coords, colors, instance_class_labels, instance_segids), 
        os.path.join(pcd_out_dir, '%s.pth'%(scan_id))
    )
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scannet_dir', required=True, type=str,
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='the path of the directory to be saved preprocessed scans')
    parser.add_argument('--class_label_file', required=True, type=str)

    # Optional arguments.
    parser.add_argument('--inst_seg_dir', default=None, type=str)
    parser.add_argument('--segment_dir', default=None, type=str,
                        help='the path to the predicted masks of pretrained segmentor')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='the number of processes, -1 means use the available max')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='use mmengine to process in a parallel manner')
    parser.add_argument('--apply_global_alignment', default=False, action='store_true',
                        help='rotate/translate entire scan globally to aligned it with other scans')
    args = parser.parse_args()

    # Print the args
    args_string = pprint.pformat(vars(args))
    print(args_string)

    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    num_workers = args.num_workers

    id2class = {}
    with open(args.class_label_file, "r") as f:
        csvreader = csv.reader(f, delimiter='\t')
        csvreader.__next__()
        for line in csvreader:
            id2class[line[0]] = line[2]
    
    if args.segment_dir:
        tmp_dir = os.path.join(args.segment_dir, 'mask3d_inst_seg')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        params = []
        for split in ["train", "val"]:
        # for split in ["test"]:
            cur_dir = os.path.join(args.segment_dir, split)
            for file_path in glob.glob(os.path.join(cur_dir, "*.txt")):
                params.append((cur_dir, file_path))
        fn = partial(
            process_one_scene,
            tmp_dir=tmp_dir,
            id2class=id2class
        )
        if args.parallel:
            mmengine.utils.track_parallel_progress(fn, params, num_workers)
        else:
            for param in tqdm(params):
                fn(param)
                print(len(ids))
    else:
        tmp_dir = args.inst_seg_dir

    # for split in ['scans', 'scans_test']:
    for split in ['scans']:
        scannet_dir = os.path.join(args.scannet_dir, split)

        fn = partial(
            process_per_scan,
            scan_dir=scannet_dir,
            out_dir=args.output_dir,
            tmp_dir=tmp_dir,
            apply_global_alignment=args.apply_global_alignment,
            is_test='test' in split
        )

        scan_ids = os.listdir(scannet_dir)
        scan_ids.sort()
        print(split, '%d scans' % (len(scan_ids)))

        if args.parallel:
            mmengine.utils.track_parallel_progress(fn, scan_ids, num_workers)
        else:
            for scan_id in scan_ids:
                fn(scan_id)

    # all_feats = {}
    # for split in ['train', 'val']:
    #     cur_feat_dir = os.path.join(args.segment_dir, split, 'features')
    #     for filename in tqdm(os.listdir(cur_feat_dir)):
    #         if not filename.endswith('.pt'):
    #             continue
    #         scene_id = filename.split('.pt')[0]
    #         tmp_feat = torch.load(os.path.join(cur_feat_dir, filename), map_location='cpu')
    #         for i in range(tmp_feat.shape[1]):
    #             all_feats[f"{scene_id}_{i:02}"] = tmp_feat[0, i]
    # torch.save(all_feats, "annotations/scannet_mask3d_mask3d_feats.pt")


if __name__ == '__main__':
    main()

