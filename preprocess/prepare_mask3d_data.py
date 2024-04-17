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

parallel = True
num_threads = 16

nyu40_label_file = "/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/meta_data/nyu40_labels.csv"
nyu40_id2class = {}
with open(nyu40_label_file, "r") as f:
    csvreader = csv.reader(f, delimiter=',')
    csvreader.__next__()
    for line in csvreader:
        nyu40_id2class[line[0]] = line[1]

pointgroup_dir = "/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/Mask3DInst"
tmp_dir = os.path.join(pointgroup_dir, 'tmp')
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)


# def process_one_scene(params):
#     cur_dir, file_path = params
#     scene_id = file_path.split("/")[-1].split(".txt")[0]
#     cur_scene_out = []
#     with open(file_path, "r") as f:
#         for line in f.readlines():
#             predict_path, nyuid, score = line.split(" ")
#             # if float(score) < 0.5:
#             #     continue
#             tmp = predict_path.split('_')
#             predict_path = os.path.join('_'.join(tmp[:-1]), tmp[-1])
#             predict_path = os.path.join(cur_dir, predict_path)
#             segments = []
#             with open(predict_path, "r") as f2:
#                 for i, l in enumerate(f2.readlines()):
#                     if l[0] == "1":
#                         segments.append(i)
#             cur_scene_out.append({
#                 "label": nyu40_id2class[nyuid],
#                 "segments": segments
#             })
#         print(len(cur_scene_out))
#     torch.save(cur_scene_out, os.path.join(tmp_dir, f"{scene_id}.pt"))

# params = []
# for split in ["train", "val"]:
#     cur_dir = os.path.join(pointgroup_dir, split)
#     for file_path in tqdm(glob.glob(os.path.join(cur_dir, "*.txt"))):
#         params.append((cur_dir, file_path))

# if parallel:
#     mmengine.utils.track_parallel_progress(process_one_scene, params, num_threads)
# else:
#     for param in params:
#         process_one_scene(param)


def process_per_scan(scan_id, scan_dir, out_dir, apply_global_alignment=True, is_test=False):
    pcd_out_dir = os.path.join(out_dir, 'pcd_with_global_alignment' if apply_global_alignment else 'pcd_no_global_alignment')
    if os.path.exists(os.path.join(pcd_out_dir, '%s.pth'%(scan_id))):
        print(f"skipping {scan_id}...")
        return
    print(f"processing {scan_id}...")
    pcd_out_dir = os.path.join(out_dir, 'pcd_with_global_alignment' if apply_global_alignment else 'pcd_no_global_alignment')
    os.makedirs(pcd_out_dir, exist_ok=True)
    obj_out_dir = os.path.join(out_dir, 'instance_id_to_name')
    os.makedirs(obj_out_dir, exist_ok=True)

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
    if is_test:
        sem_labels = None
        instance_labels = None
    else:
        # colored by nyu40 labels (ply property 'label' denotes the nyu40 label id)
        # with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.labels.ply'%(scan_id)), 'rb') as f:
        #     plydata = PlyData.read(f)
        # sem_labels = np.array(plydata.elements[0]['label']).astype(np.long)
        # assert len(coords) == len(colors) == len(sem_labels)
        sem_labels = None

        # Map each point to segment id
        if not os.path.exists(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json'%(scan_id))):
            return
        with open(os.path.join(scan_dir, scan_id, '%s_vh_clean_2.0.010000.segs.json'%(scan_id)), 'r') as f:
            d = json.load(f)
        seg = d['segIndices']
        segid_to_pointid = {}
        for i, segid in enumerate(seg):
            segid_to_pointid.setdefault(segid, [])
            segid_to_pointid[segid].append(i)

        # Map object to segments
        instance_class_labels = []
        instance_segids = []
        # with open(os.path.join(scan_dir, scan_id, '%s.aggregation.json'%(scan_id)), 'r') as f:
        #     d = json.load(f)
        # for i, x in enumerate(d['segGroups']):
        #     assert x['id'] == x['objectId'] == i
        #     instance_class_labels.append(x['label'])
        #     instance_segids.append(x['segments'])

        # cur_instances = pointgroup_instances[scan_id].copy()
        if not os.path.exists(os.path.join(tmp_dir, f"{scan_id}.pt")):
            return
        cur_instances = torch.load(os.path.join(tmp_dir, f"{scan_id}.pt"))
        for instance in cur_instances:
            instance_class_labels.append(instance["label"])
            instance_segids.append(instance["segments"])

        final_instance_class_labels = []
        now_i = 0
        instance_labels = np.ones(coords.shape[0], dtype=np.int32) * -100
        for i, segids in enumerate(instance_segids):
            # pointids = []
            # for segid in segids:
            #     pointids += segid_to_pointid[segid]
            pointids = segids
            if np.sum(instance_labels[pointids] != -100) > 0:
                # scene0217_00 contains some overlapped instances
                if len(pointids) - np.sum(instance_labels[pointids] != -100) > 10:
                    print(scan_id, i, np.sum(instance_labels[pointids] != -100), len(pointids))
                    pointids = np.array(pointids)[np.where(instance_labels[pointids] == -100)[0]].tolist()
                    instance_labels[pointids] = now_i
                    final_instance_class_labels.append(instance_class_labels[i])
                    now_i += 1
                else:
                    continue
            else:
                instance_labels[pointids] = now_i
                final_instance_class_labels.append(instance_class_labels[i])
                now_i += 1
            # assert len(np.unique(sem_labels[pointids])) == 1, 'points of each instance should have the same label'
        print(now_i)
        json.dump(
            final_instance_class_labels, 
            open(os.path.join(obj_out_dir, '%s.json'%scan_id), 'w'), 
            indent=2
        )

    torch.save(
        (coords, colors, sem_labels, instance_labels), 
        os.path.join(pcd_out_dir, '%s.pth'%(scan_id))
    )
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scannet_dir', required=True, type=str,
                        help='the path to the downloaded ScanNet scans')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='the path of the directory to be saved preprocessed scans')

    # Optional arguments.
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='the number of processes, -1 means use the available max')
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

    # for split in ['scans', 'scans_test']:
    for split in ['scans']:
        scannet_dir = os.path.join(args.scannet_dir, split)

        fn = partial(
            process_per_scan, 
            scan_dir=scannet_dir, 
            out_dir=args.output_dir, 
            apply_global_alignment=args.apply_global_alignment,
            is_test='test' in split
        )

        scan_ids = os.listdir(scannet_dir)
        scan_ids.sort()
        print(split, '%d scans' % (len(scan_ids)))

        # start_time = time.time()
        if args.num_workers == -1:
            num_workers = min(mp.cpu_count(), len(scan_ids))
        else:
            num_workers = args.num_workers
        print('num workers:', num_workers)

        # pool = mp.Pool(num_workers)
        # pool.map(fn, scan_ids)
        # pool.close()
        # pool.join()
        mmengine.utils.track_parallel_progress(fn, scan_ids, num_workers)
        # for scan_id in scan_ids:
        #     fn(scan_id)
            

        # print("Process data took {:.4} minutes.".format((time.time() - start_time) / 60.0))


if __name__ == '__main__':
    main()
