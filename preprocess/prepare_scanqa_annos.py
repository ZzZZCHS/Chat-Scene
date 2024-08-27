import json

for split in ['train', 'val']:
    if split == 'test':
        with open(f"annotations/scanqa/ScanQA_v1.0_test_w_obj.json", "r") as f:
            annos = json.load(f)
        with open(f"annotations/scanqa/ScanQA_v1.0_test_wo_obj.json", "r") as f:
            annos.extend(json.load(f))
    else:
        with open(f"annotations/scanqa/ScanQA_v1.0_{split}.json", "r") as f:
            annos = json.load(f)
    print(len(annos))
    new_annos = []
    for anno in annos:
        scene_id = anno["scene_id"]
        obj_ids = anno["object_ids"] if "object_ids" in anno else [0]
        question = anno["question"]

        prompt = question + " Answer the question using a single word or phrase."

        answers = anno["answers"] if "answers" in anno else []
        if split == "train":
            for i in range(len(answers)):
                if i > 0 and answers[i] == answers[i-1]:
                    continue
                answer = answers[i]
                answer = answer.capitalize()
                if answer[-1] != ".":
                    answer += "."
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": obj_ids[0],
                    "prompt": prompt,
                    "caption": answer,
                })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_ids[0],
                "prompt": prompt,
                "ref_captions": answers
            })
    print(len(new_annos))

    with open(f"annotations/scanqa_{split}.json", "w") as f:
        json.dump(new_annos, f, indent=4)
