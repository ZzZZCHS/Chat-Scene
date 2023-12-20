import json
import torch
from collections import defaultdict

split = "val"

annos = json.load(open(f"annotations/scanrefer_{split}_stage2_objxx.json", "r"))
attrs = torch.load(f"annotations/scannet_{split}_attributes.pt")

new_annos = defaultdict(dict)

for anno in annos:
    scene_id = anno["scene_id"]
    obj_id = anno["obj_id"]
    caption = anno["caption"] if "caption" in anno else anno["ref_captions"][0]
    new_annos[scene_id][f"{obj_id:03}"] = {
        "loc": attrs[scene_id]["locs"][obj_id][:3].tolist(),
        "caption": caption,
        "class_name": attrs[scene_id]["objects"][obj_id]
    }


print(len(new_annos))

for scene_id in new_annos.keys():
    message = ""
    for i in range(200):
        obj_id = f"{i:03}"
        if obj_id not in new_annos[scene_id]:
            continue
        obj_anno = new_annos[scene_id][obj_id]
        class_name = obj_anno["class_name"]
        loc = obj_anno["loc"]
        for j in range(len(loc)):
            loc[j] = round(loc[j], 2)
        caption = obj_anno["caption"]
        message += f"obj{i:02}: {class_name}; {loc}; {caption}\n"
    with open(f"annotations/scene_dataset/obj_info_list/{split}/{scene_id}.json", "w") as f:
        json.dump(message, f)

