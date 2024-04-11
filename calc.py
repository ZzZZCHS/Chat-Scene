import json
import sys
import re
replace_list = [["10", "ten"], ["12", "twelve"], ["1", "one"], ["2", "two"], ["3", "three"], ["4", "four"], ["5", "five"],
                ["6", "six"], ["7", "seven"], ["8", "eight"], ["9", "nine"]]
# outputs = json.load(open("/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/preds_epoch2_step17070.json", "r"))
outputs = json.load(open("/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240408_125254_dp0.1_lr5e-6_sta2_ep1_objaverse#scannet_caption#scanrefer_caption#scannet_region_caption#nr3d_caption#scanrefer#obj_align#scanqa__scanqa#scanrefer/preds_epoch-1_step0_scanqa.json", "r"))
# outputs2 = json.load(open("/root/scene-LLaMA/video_chat/video_chat/outputs/2023-09-17-215602_dp0.1_lr5e-5_sta3_ep3/preds_epoch-1_step0.json", "r"))
# outputs.extend(outputs2)
preds = {}
targets = {}

print(len(outputs))
ref_lens = 0
acc = 0
# max_len = 40

def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data

for i, output in enumerate(outputs):
    item_id = f"{output['scene_id']}_{output['obj_id']}_{output['qid']}_{i}"
    pred = output["pred"]
    ref_captions = output["ref_captions"]
    pred = pred.strip()
    if len(pred) > 0 and pred[-1] == ".":
        pred = pred[:-1]
    if len(pred) > 0:
        pred = pred[0].lower() + pred[1:]
    
    pred = clean_answer(pred)
    ref_captions = [clean_answer(caption) for caption in ref_captions]
    output["pred"] = pred
    output["ref_captions"] = ref_captions
    if pred in ref_captions:
        acc += 1
    preds[item_id] = [{"caption": pred}]
    targets[item_id] = [{"caption": caption} for caption in ref_captions]
    # print(targets[item_id][0])
    ref_lens += len(targets[item_id][0])
    # preds.append(output["pred"])
    # targets.append(output["ref_captions"])

# with open("scanqa_sota_result.json", "w") as f:
#     json.dump(outputs, f, indent=4)

# preds["1"] = [" this is the lamp on the round table. it is by the corner."]
# targets["1"] = ["this  is the lamp on the round table.  it is by the corner."]

# print(ref_lens)
print(len(preds))

# print(preds[:1])
# print(targets[:1])

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# from nltk.translate.bleu_score import sentence_bleu
# score = sentence_bleu(targets[:5], preds[:5], weights=[1.])
# print(score)

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# import utils.capeval.bleu.bleu as capblue
# import utils.capeval.cider.cider as capcider
# import utils.capeval.rouge.rouge as caprouge
# import utils.capeval.meteor.meteor as capmeteor

scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    (Spice(), "SPICE")
]
# scorers = [
#     (capblue.Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#     (capmeteor.Meteor(), "METEOR"),
#     (caprouge.Rouge(), "ROUGE_L"),
#     (capcider.Cider(), "CIDEr")
# ]

tokenizer = PTBTokenizer()
targets = tokenizer.tokenize(targets)
preds = tokenizer.tokenize(preds)
# print(targets)
# print(preds)


val_results = {}

for scorer, method in scorers:
    eprint('computing %s score...'%(scorer.method()))
    score, scores = scorer.compute_score(targets, preds)
    if type(method) == list:
        for sc, scs, m in zip(score, scores, method):
            print("%s: %0.3f"%(m, sc*100))
            val_results[m] = sc
            # rlt[m]=sc*100
    else:
        print("%s: %0.3f"%(method, score*100))
        val_results[method] = score
        # rlt[method]=score*100


print("EM: ", float(acc) / len(preds))
for k, v in val_results.items():
    print(f"{k}: {v}")
