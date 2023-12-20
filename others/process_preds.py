import json
import os

model = "2023-09-17-144020_dp0.1_lr5e-5_sta3_ep3"
conv_pred_file = f"eval/conv100_{model}.json"
detail_pred_file = f"eval/detail100_{model}.json"

qa_name = "qa60"
conv_preds_ = json.load(open(conv_pred_file, "r"))
detail_preds_ = json.load(open(detail_pred_file, "r"))
ques = json.load(open(f"eval/{qa_name}_questions.json", "r"))

conv_preds = {}
detail_preds = {}

for pred in conv_preds_:
    item_id = f"{pred['scene_id']}_{pred['obj_id']}"
    conv_preds[item_id] = pred["pred"]
for pred in detail_preds_:
    item_id = f"{pred['scene_id']}_{pred['obj_id']}"
    detail_preds[item_id] = pred["pred"]

answers = []

for que in ques:
    item_id = que["item_id"]
    answer = conv_preds[item_id] if que["category"] == "conv" else detail_preds[item_id]
    if ": " in answer:
        answer = ": ".join(answer.split(": ")[1:])
    answers.append({
        "answer_id": que["question_id"],
        "text": answer
    })

with open(f"eval/{qa_name}_{model}_answer.json", "w") as f:
    json.dump(answers, f, indent=4)
