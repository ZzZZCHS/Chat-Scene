import json
import os
import sys
sys.path.append('.')

from utils.eval import calc_scanrefer_score, clean_answer, calc_scan2cap_score, calc_scanqa_score, calc_sqa3d_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

output_dir = '/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240414_044335_dp0.1_lr5e-6_sta2_ep3_scanrefer#scanqa#sqa3d#scan2cap#nr3d_caption#obj_align#scannet_caption#scannet_region_caption#scanrefer_seg#scan2cap_seg__scanqa#scanrefer#sqa3d#scan2cap__seg100train'

tokenizer = PTBTokenizer()
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    (Spice(), "SPICE")
]


prefix = 'preds_epoch0_step1361'

all_val_scores = {}

for task in ['scanqa', 'scanrefer', 'scan2cap', 'sqa3d']:
    save_preds = []
    for filename in os.listdir(output_dir):
        if filename.startswith(prefix) and task in filename:
            preds = json.load(open(os.path.join(output_dir, filename)))
            save_preds += preds
    val_scores = {}
    if task == 'scanqa':
        val_scores = calc_scanqa_score(save_preds, tokenizer, scorers)
    if task == 'scanrefer':
        val_scores = calc_scanrefer_score(save_preds)
    if task == 'scan2cap':
        val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers)
    if task == 'sqa3d':
        val_scores = calc_sqa3d_score(save_preds, tokenizer, scorers)
    all_val_scores = {**all_val_scores, **val_scores}

print(json.dumps(all_val_scores, indent=4))