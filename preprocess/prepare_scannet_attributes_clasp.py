import torch
import json
import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--scan_dir', required=True, type=str,
                    help='the path of the directory to be saved preprocessed scans')
parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--max_inst_num', required=True, type=int)
parser.add_argument('--version', type=str, default='')
args = parser.parse_args()



for split in ["train", "val"]:
    scan_dir = args.scan_dir
    output_dir = "annotations"
    split_path = f"annotations/scannet/scannetv2_{split}.txt"

    scan_ids = [line.strip() for line in open(split_path).readlines()]

    scan_ids = sorted(scan_ids)
    # print(scan_ids)

    scans = {}
    for scan_id in scan_ids:
        pcd_path = os.path.join(scan_dir, f"{scan_id}.pth")
        if not os.path.exists(pcd_path):
            # print('skip', scan_id)
            continue
        pred_results = torch.load(pcd_path, map_location='cpu')
        inst_locs = []
        num_insts = pred_results['pred_boxes'].shape[0]
        for i in range(min(num_insts, args.max_inst_num)):
            center = pred_results['pred_boxes'].mean(dim=0)
            size = pred_results['pred_boxes'][1] - pred_results['pred_boxes'][0]
            inst_locs.append(torch.cat([center, size], 0))
        inst_locs = torch.stack(inst_locs, dim=0).to(torch.float32)
        scans[scan_id] = {
            # 'objects': instance_class_labels,  # (n_obj, )
            'locs': inst_locs,  # (n_obj, 6) center xyz, whl
        }
    print(f"{split}: {len(scans)}")

    torch.save(scans, os.path.join(output_dir, f"scannet_{args.segmentor}_{split}_attributes{args.version}.pt"))