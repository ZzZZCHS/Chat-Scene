import json
import random
import torch
import random

split = "train"

annos = []

if split == "val":
    with open(f"annotations/scanrefer_{split}_stage1.json", "r") as f:
        annos.extend(json.load(f))

if split == "train":
    with open(f"annotations/scannet_{split}_stage1.json", "r") as f:
        annos.extend(json.load(f))

scene_attrs = json.load(open("annotations/scannet_attributes.json", "r"))

print(len(annos))

new_annos = []

for anno in annos:
    scene_id = anno["scene_id"]
    obj_id = int(anno["obj_id"])
    # caption = anno["captions"][0]

    locs = torch.tensor(scene_attrs[scene_id]["locs"])
    obj_num = locs.shape[0]
    if obj_num <= 6:
        continue
    centers = locs[:, :3]
    dis = ((centers - centers[obj_id])**2).sum(dim=-1)
    center_diff = (centers - centers[obj_id]).abs()

    mode = random.randint(0, 3)
    if mode == 0:
        if random.randint(0, 1) == 0:
            prompt = f"Which object is closest to obj{obj_id:02}?"
            answer_id = int(dis.topk(k=2, largest=False)[1][1])
            answer = f"The closest object to obj{obj_id:02} is obj{answer_id:02}."
        else:
            prompt = f"Which object is farthest from obj{obj_id:02}?"
            answer_id = int(dis.topk(k=2, largest=True)[1][0])
            answer = f"The farthest object from obj{obj_id:02} is obj{answer_id:02}."
    if mode == 1:
        a = random.randint(0, obj_num-1)
        b = random.randint(0, obj_num-1)
        while a == obj_id or b == obj_id or a == b:
            a = random.randint(0, obj_num - 1)
            b = random.randint(0, obj_num - 1)
        if random.randint(0, 1):
            prompt = f"Which object is closer to obj{obj_id:02}, obj{a:02} or obj{b:02}?"
            answer_id = a if dis[a] < dis[b] else b
            answer = f"The closer object to obj{obj_id:02} is obj{answer_id:02}."
        else:
            prompt = f"Which object is farther from obj{obj_id:02}, obj{a:02} or obj{b:02}?"
            answer_id = a if dis[a] > dis[b] else b
            answer = f"The farther object from obj{obj_id:02} is obj{answer_id:02}."
    if mode == 2:
        a = random.randint(0, obj_num - 1)
        b = random.randint(0, obj_num - 1)
        while a == obj_id or b == obj_id or a == b:
            a = random.randint(0, obj_num - 1)
            b = random.randint(0, obj_num - 1)
        z_a = locs[a][2] - locs[a][5]
        z_b = locs[b][2] - locs[b][5]
        if random.randint(0, 1):
            prompt = f"Which object is located at the higher position, obj{a:02} or obj{b:02}?"
            if z_a < z_b:
                a, b = b, a
            answer = f"Obj{a:02} is located at a higher position compared to obj{b:02}."
        else:
            prompt = f"Which object is located at the lower position, obj{a:02} or obj{b:02}?"
            if z_a > z_b:
                a, b = b, a
            answer = f"Obj{a:02} is located at a lower position compared to obj{b:02}."
    if mode == 3:
        prompt = f"List the five closest objects to obj{obj_id:02} in ascending order of their object IDs."
        answer_ids = dis.topk(k=6, largest=False)[1][1:6].tolist()
        answer_ids.sort()
        answer = f"The five closest objects to obj{obj_id:02} in ascending order are: obj{answer_ids[0]:02}, obj{answer_ids[1]:02}, obj{answer_ids[2]:02}, obj{answer_ids[3]:02}, and obj{answer_ids[4]:02}."
    # nearest
    if split == "train":
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "prompt": prompt,
            "caption": answer
        })
    else:
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "prompt": prompt,
            "ref_captions": [answer]
        })

with open(f"annotations/scene_align_{split}.json", "w") as f:
    json.dump(new_annos, f, indent=4)
