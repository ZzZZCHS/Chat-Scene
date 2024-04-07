import json

replace_list = [["10", "ten"], ["12", "twelve"], ["1", "one"], ["2", "two"], ["3", "three"], ["4", "four"], ["5", "five"],
                ["6", "six"], ["7", "seven"], ["8", "eight"], ["9", "nine"]]

split = "train"
with open(f"annotations/scanqa/ScanQA_v1.0_{split}.json", "r") as f:
    annos = json.load(f)
print(len(annos))
new_annos = []
for anno in annos:
    scene_id = anno["scene_id"]
    obj_ids = anno["object_ids"] if "object_ids" in anno else []
    question = anno["question"]
    # for (a, b) in replace_list:
    #     question = question.replace(a, b)

    # prompt = f"Pay attention to obj{obj_ids[0]:02}"
    # if len(obj_ids) == 2:
    #     prompt += f" and obj{obj_ids[1]:02}"
    # elif len(obj_ids) > 2:
    #     for i in range(1, len(obj_ids)-1):
    #         prompt += f", obj{obj_ids[i]:02}"
    #     prompt += f", and obj{obj_ids[-1]:02}"
    # if len(obj_ids) > 0:
    #     related_prompt = f"The relevant object{'s are' if len(obj_ids) > 1 else ' is'} obj{obj_ids[0]:02}"
    #     if len(obj_ids) == 2:
    #         related_prompt += f" and obj{obj_ids[1]:02}"
    #     elif len(obj_ids) > 2:
    #         for i in range(1, len(obj_ids)-1):
    #             related_prompt += f", obj{obj_ids[i]:02}"
    #         related_prompt += f", and obj{obj_ids[-1]:02}"
    #     related_prompt += "."
    # prompt += ". " + question + " Answer the question using a single word or phrase."
    prompt = question + " Answer the question using a single word or phrase."
    # prompt = question + " The answer should be a phrase or a single word."

    answers = anno["answers"]
    if split == "val":
        # for i in range(len(answers)):
        #     for (a, b) in replace_list:
        #         answers[i] = answers[i].replace(a, b)
        new_annos.append({
            "scene_id": scene_id,
            "obj_id": obj_ids[0],
            "prompt": prompt,
            "ref_captions": answers
        })
    elif split == "train":
        for i in range(len(answers)):
            if i > 0 and answers[i] == answers[i-1]:
                continue
            answer = answers[i]
            # for (a, b) in replace_list:
            #     answer = answer.replace(a, b)
            answer = answer.capitalize()
            if answer[-1] != ".":
                answer += "."
            # answer = "The answer is " + answer + "."
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_ids[0],
                "prompt": prompt,
                "caption": answer,
                # "related_ids": obj_ids
            })
print(len(new_annos))

with open(f"annotations/scanqa_{split}.json", "w") as f:
    json.dump(new_annos, f, indent=4)
