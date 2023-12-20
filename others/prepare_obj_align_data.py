import json
import random

split = "val"
with open(f"annotations/scanrefer_{split}_stage1.json", "r") as f:
    annos = json.load(f)

if split == "train":
    with open(f"annotations/scannet_{split}_stage1.json", "r") as f:
        annos.extend(json.load(f))

print(len(annos))

new_annos = []

with open("prompts/obj_align_template.txt", 'r') as f:
    answer_templates = f.read().splitlines()

for anno in annos:
    scene_id = anno["scene_id"]
    obj_id = anno["obj_id"]
    caption = anno["captions"][0]
    prompt = f"What is the obj{int(obj_id):02}?"
    if split == "train":
        answer_template = random.choice(answer_templates)
        if answer_template.count("{}") == 2:
            answer = answer_template.format(str(obj_id).zfill(2), caption)
        else:
            answer = answer_template.format(caption)
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "prompt": prompt,
            "caption": answer
        })
    else:
        answers = []
        for answer_template in answer_templates:
            if answer_template.count("{}") == 2:
                answer = answer_template.format(str(obj_id).zfill(2), caption)
            else:
                answer = answer_template.format(caption)
            answers.append(answer)
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "prompt": prompt,
            "ref_captions": answers
        })

with open(f"annotations/obj_align_{split}.json", "w") as f:
    json.dump(new_annos, f, indent=4)
