import json
import os
import random

anno_root = "anno"
detail_refs = json.load(open(os.path.join(anno_root, "scanrefer_val_describe.json"), "r"))
conv_refs = json.load(open(os.path.join(anno_root, "scanrefer_val_conversation.json"), "r"))
target_infos = json.load(open(os.path.join(anno_root, "scanrefer_val_content.json"), "r"))

# detail_items = set(detail_refs.keys())
# conv_items = set(conv_refs.keys())
# item_list = list(detail_items & conv_items)
# print(len(item_list))
#
# sampled_items = sorted(random.sample(item_list, 100))
#
# with open("scripts/eval/item_list.json", "w") as f:
#     json.dump(sampled_items, f)

item_list = json.load(open("eval/item_list.json", "r"))

# detail_samples = []
# conv_samples = []
# qid = 0
# answers = []
# questions = []
#
# for item in item_list:
#     scene_id = "_".join(item.split("_")[:-1])
#     obj_id = int(item.split("_")[-1])
#     detail_samples.append({
#         "scene_id": scene_id,
#         "obj_id": obj_id,
#         "prompt": detail_refs[item][0]["Question"]
#     })
#     answers.append({
#         "question_id": qid,
#         "text": detail_refs[item][0]["Answer"],
#         "category": "detail"
#     })
#     questions.append({
#         "question_id": qid,
#         "text": detail_refs[item][0]["Question"],
#         "category": "detail",
#         "item_id": item
#     })
#     qid += 1
#     conv = random.choice(conv_refs[item])
#     while len(conv["Answer"]) == 0:
#         print(f"empty answer in {item}...")
#         conv = random.choice(conv_refs[item])
#     conv_samples.append({
#         "scene_id": scene_id,
#         "obj_id": obj_id,
#         "prompt": conv["Question"]
#     })
#     answers.append({
#         "question_id": qid,
#         "text": conv["Answer"],
#         "category": "conv"
#     })
#     questions.append({
#         "question_id": qid,
#         "text": conv["Question"],
#         "category": "conv",
#         "item_id": item
#     })
#     qid += 1
#
# with open(os.path.join(anno_root, "scanrefer_val_describe100.json"), "w") as f:
#     json.dump(detail_samples, f)
# with open(os.path.join(anno_root, "scanrefer_val_conversation100.json"), "w") as f:
#     json.dump(conv_samples, f)
#
# with open("eval/qa200_gpt4_answer.json", "w") as f:
#     json.dump(answers, f, indent=4)
# with open("eval/qa200_questions.json", "w") as f:
#     json.dump(questions, f, indent=4)

old_answers = json.load(open("eval/qa200_gpt4_answer.json", "r"))
old_questions = json.load(open("eval/qa200_questions.json", "r"))

new_list = [item_list[0]]
for i in range(1, len(item_list)):
    if item_list[i][:11] != item_list[i-1][:11]:
        new_list.append(item_list[i])
new_list = random.sample(new_list, 30)

new_answers = []
new_questions = []
qid = 0

for i in range(len(old_questions)):
    ques = old_questions[i]
    ans = old_answers[i]
    if ques["item_id"] in new_list:
        ques["question_id"] = ans["answer_id"] = qid
        qid += 1
        new_questions.append(ques)
        new_answers.append(ans)

with open("eval/qa60_gpt4_answer.json", "w") as f:
    json.dump(new_answers, f, indent=4)
with open("eval/qa60_questions.json", "w") as f:
    json.dump(new_questions, f, indent=4)
