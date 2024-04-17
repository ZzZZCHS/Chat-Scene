import os
import argparse
import numpy as np
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_data_dir')
    parser.add_argument('--bbox_out_dir')
    args = parser.parse_args()

    os.makedirs(args.bbox_out_dir, exist_ok=True)

    scan_ids = [x.split('.')[0] for x in os.listdir(args.pcd_data_dir)]
    scan_ids.sort()

    for scan_id in tqdm(scan_ids):
        if os.path.exists(os.path.join(args.bbox_out_dir, '%s.npy'%scan_id)):
            continue
        points, colors, _, inst_labels = torch.load(
            os.path.join(args.pcd_data_dir, '%s.pth'%scan_id)
        )
        if inst_labels is None:
            continue
        num_insts = inst_labels.max()
        outs = []
        for i in range(num_insts+1):
            inst_mask = inst_labels == i
            inst_points = points[inst_mask]
            if len(inst_points) < 10:
                print(scan_id, i, 'empty bbox')
                outs.append(np.zeros(6, ).astype(np.float32))
            else:
                bbox_center = inst_points.mean(0)
                bbox_size = inst_points.max(0) - inst_points.min(0)
                outs.append(np.concatenate([bbox_center, bbox_size], 0))
        outs = np.stack(outs, 0).astype(np.float32)

        np.save(os.path.join(args.bbox_out_dir, '%s.npy'%scan_id), outs)

if __name__ == '__main__':
    main()