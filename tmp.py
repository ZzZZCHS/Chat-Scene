import torch
import os


for split in ["train", "val"]:
    attrs = torch.load(f"annotations/scannet_{split}_attributes.pt", map_location='cpu')
    for scan_id in attrs.keys():
        locs = attrs[scan_id]['locs']
        locs = [locs[i:i+6] for i in range(0, locs.shape[0], 6)]
        locs = torch.stack(locs, dim=0)
        attrs[scan_id]['locs'] = locs
    torch.save(attrs, f"annotations/scannet_{split}_attributes.pt")