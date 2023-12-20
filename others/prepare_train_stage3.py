import jsonlines
import json

train_split_file = "/root/scene-LLaMA/datasets/referit3d/annotations/splits/scannetv2_train.txt"
train_list = []
with open(train_split_file, "r") as f:
    for line in f.readlines():
        train_list.append(line.strip())

nr3d_anno_file = "/root/scene-LLaMA/datasets/referit3d/annotations/bert_tokenized/nr3d.jsonl"
nr3d_anno = []
with jsonlines.open(nr3d_anno_file, "r") as reader:
    for l in reader:
        nr3d_anno.append(l)

anno_root = "annotations"  # annotation dir
attribute_file = f"{anno_root}/scannet_attributes_old.json"
attributes = json.load(open(attribute_file, 'r'))

q_template = "Evaluate the request below and determine whether it accurately identifies the target object enclosed within the tags '<Target>' and '</Target>': <Request> {} </Request> Please respond with the answer in the following format: 'The answer is: True' if the request correctly localizes the target object, or 'The answer is: False' if it does not."
a_template = "The answer is: {}."

from tqdm import tqdm
import random
from collections import defaultdict
utters = defaultdict(list)
for item in nr3d_anno:
    scene_id = item["scan_id"]
    if scene_id not in train_list:
        continue
    utter = item["utterance"]
    utters[scene_id].append(utter)
    # target_id = item["target_id"]

new_anno = []

for item in tqdm(nr3d_anno):
    scene_id = item["scan_id"]
    if scene_id not in train_list:
        continue
    utter = item["utterance"]
    target_id = item["target_id"]
    new_anno.append({
        "scene_id": scene_id,
        "obj_id": target_id,
        "QA": [{
            "Question": q_template.format(utter),
            "Answer": a_template.format("True")
        }]
    })

    utter = random.choice(utters[scene_id])
    while utter == item["utterance"]:
        utter = random.choice(utters[scene_id])
    new_anno.append({
        "scene_id": scene_id,
        "obj_id": target_id,
        "QA": [{
            "Question": q_template.format(utter),
            "Answer": a_template.format("False")
        }]
    })

with open("annotations/nr3d_train_tf.json", "w") as f:
    json.dump(new_anno, f)
