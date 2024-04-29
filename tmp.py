import json

x = json.load(open('/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/annotations/scanrefer_mask3d_val_grounding.json'))

scene_ids = [p['scene_id'] for p in x]
print(len(set(scene_ids)))