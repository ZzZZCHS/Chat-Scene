import json
import os
import random

anno_root = "anno"
anno_file = os.path.join(anno_root, "scanrefer_val_conversation.json")

annos = json.load(open(anno_file, "r"))

# questions = []
# with open("prompts/detailed_description.txt", "r") as f:
#     for q in f.readlines():
#         questions.append(q.strip())

# for k, a in annos.items():
#     q = random.choice(questions)
#     annos[k] = [{
#         "Question": q,
#         "Answer": a
#     }]
output_anno = []
for k, v in annos.items():
    if len(v) == 0:
        continue
    scene_id = "_".join(k.split("_")[:-1])
    obj_id = int(k.split("_")[-1])
    for i in range(len(v)):
        q = v[i]["Question"]
        a = v[i]["Answer"]
        output_anno.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "qid": i,
            "prompt": q,
            "ref": a
        })
    # if len(output_anno) >= 500:
    #     break

output_anno = sorted(output_anno, key=lambda x: f"{x['scene_id']}_{x['obj_id']:03}_{x['qid']:2}")
print(len(output_anno))

with open("anno/scanrefer_val_conv.json", "w") as f:
    json.dump(output_anno, f, indent=4)
