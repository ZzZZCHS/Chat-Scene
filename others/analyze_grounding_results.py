import json
import torch
import jsonlines

def is_explicitly_view_dependent(s):
    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                    'looking', 'across'}
    for word in target_words:
        if word in s:
            return True
    return False


attrs = torch.load("annotations/scannet_val_attributes.pt")


# val_file = "/root/scene-LLaMA/datasets/exprs_neurips22/gtlabelpcd_mix/nr3d/preds/val_outs.json"
val_file = "outputs/2023-11-17-230123_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign/preds_epoch-1_step0.json"
nr3d_anno_file = "/root/scene-LLaMA/datasets/referit3d/annotations/bert_tokenized/sr3d.jsonl"

anno_root = "annotations"  # annotation dir
val_results = json.load(open(val_file))
# val_results_dict = {}
# for val_item in val_results:
#     val_results_dict[val_item["qid"]] = val_item
nr3d_anno = {}
with jsonlines.open(nr3d_anno_file, "r") as reader:
    for l in reader:
        nr3d_anno[l["item_id"]] = l

val_num = len(val_results)

easy_acc, hard_acc = 0, 0
view_indep_acc, view_dep_acc = 0, 0
acc = 0
easy_num, hard_num = 0, 0
view_indep_num, view_dep_num = 0, 0


# from nltk.tokenize import sent_tokenize

# for v in val_results:
#
#     pred = v["pred"]
#     target = v["ref_captions"][0]
#     scene_id = v["scene_id"]
#     obj_id = v["obj_id"]
#     object_labels = attrs[scene_id]["objects"]
#     hardness = object_labels.count(object_labels[obj_id])
#     # print(object_labels)
#     # breakpoint()
#     caption = v["prompt"].split("the given description, \"")[1].split(",\" please provide the")[0]
#     tokens = sent_tokenize(caption)[0].split()
#     print(tokens)
#     # print(caption)
#     flag = pred == target
#     acc += flag
#     if is_explicitly_view_dependent(tokens):
#         view_dep_acc += flag
#         view_dep_num += 1
#     else:
#         view_indep_acc += flag
#         view_indep_num += 1
#     if hardness > 2:
#         hard_acc += flag
#         hard_num += 1
#     else:
#         easy_acc += flag
#         easy_num += 1

dataset = "sr3d"
val_file = f"/root/scene-LLaMA/datasets/exprs_neurips22/gtlabelpcd_mix/{dataset}/preds/val_outs.json"
nr3d_anno_file = f"/root/scene-LLaMA/datasets/referit3d/annotations/bert_tokenized/{dataset}.jsonl"

val_results = json.load(open(val_file))

val_num = len(val_results)

nr3d_anno = {}
with jsonlines.open(nr3d_anno_file, "r") as reader:
    for l in reader:
        nr3d_anno[l["item_id"]] = l

for k, v in val_results.items():
    obj_ids = v["obj_ids"]
    obj_logits = v["obj_logits"]
    obj_logits = (torch.tensor(obj_logits)).softmax(dim=-1).tolist()
    scene_id = nr3d_anno[k]["scan_id"]
    caption = nr3d_anno[k]["utterance"]
    target_id = nr3d_anno[k]["target_id"]
    instance_type = nr3d_anno[k]["instance_type"]
    tokens = nr3d_anno[k]["tokens"]
    logit_ids = zip(obj_logits, obj_ids)
    logit_ids = sorted(logit_ids, reverse=True)
    logits, ids = zip(*logit_ids)
    object_labels = attrs[scene_id]["objects"]
    hardness = object_labels.count(instance_type)
    # if k in val_results_dict:
    #     pred = val_results_dict[k]["pred"]
    #     ref_captions = val_results_dict[k]["ref_captions"]
    #     flag = pred == ref_captions[0]
    #     flag = 0.5
    if logits[1] > 0.01 and logits[2] <= 0.01:
        flag = 0.6
    else:
        flag = ids[0] == target_id
    acc += flag
    if is_explicitly_view_dependent(tokens):
        view_dep_acc += flag
        view_dep_num += 1
    else:
        view_indep_acc += flag
        view_indep_num += 1
    if hardness > 2:
        hard_acc += flag
        hard_num += 1
    else:
        easy_acc += flag
        easy_num += 1

print(f"Acc: {float(acc) / val_num} {acc} {val_num}")
print(f"Easy-Acc: {float(easy_acc) / easy_num} {easy_acc} {easy_num}")
print(f"Hard-Acc: {float(hard_acc) / hard_num} {hard_acc} {hard_num}")

print(f"View-Dep-Acc: {float(view_dep_acc) / view_dep_num}")
print(f"View-Indep-Acc: {float(view_indep_acc) / view_indep_num}")
