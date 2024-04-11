import json
import numpy as np
import os
import nltk
import random
from tqdm import tqdm

anno_dir = 'annotations/sqa3d'

# for filename in os.listdir(anno_dir):
#     x = json.load(open(os.path.join(anno_dir, filename)))
#     with open(os.path.join(anno_dir, filename), 'w') as f:
#         json.dump(x, f, indent=4)


def convert_person_view(sentence):
    # first-person view to second-person view
    forms = {'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'am': 'are'}
    def translate(word):
        if word.lower() in forms:
            return forms[word.lower()]
        return word
    result = ' '.join([translate(word) for word in nltk.wordpunct_tokenize(sentence)])
    return result.capitalize()


def get_sqa_question_type(question):
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return 0
    elif question[:2].lower() == 'is':
        return 1
    elif question[:3].lower() == 'how':
        return 2
    elif question[:3].lower() == 'can':
        return 3
    elif question[:5].lower() == 'which':
        return 4
    else:
        return 5   # others


for split in ['train', 'val']:
    scan_ids = []
    sqa_annos = []
    question_file = os.path.join(anno_dir, f'v1_balanced_questions_{split}_scannetv2.json')
    with open(question_file, 'r', encoding='utf-8') as f:
        question_data = json.load(f)['questions']
    question_map = {}
    for item in question_data:
        question_map[item['question_id']] = {
            's': [item['situation']] + item['alternative_situation'],   # list of str
            'q': item['question'],   # str
        }

    anno_file = os.path.join(anno_dir, f'v1_balanced_sqa_annotations_{split}_scannetv2.json')
    with open(anno_file, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)['annotations']
    for item in tqdm(anno_data):
        scan_ids.append(item['scene_id'])
        # sqa_annos.append({
        #     's': question_map[item['question_id']]['s'],   # list of str
        #     'q': question_map[item['question_id']]['q'],   # str
        #     'a': [meta['answer'] for meta in item['answers']],   # list of str
        #     'pos': np.array(list(item['position'].values())),   # array (3,)
        #     'rot': np.array(list(item['rotation'].values())),   # array (4,)
        # })
        scene_id = item['scene_id']
        obj_id = 0
        situation = random.choice(question_map[item['question_id']]['s'])
        question = question_map[item['question_id']]['q']
        question_type = get_sqa_question_type(question)
        prompt = situation + ' ' + question + " Answer the question using a single word or phrase."
        answers = [meta['answer'] for meta in item['answers']]
        if split == 'train':
            answer = random.choice(answers)
            answer = answer.capitalize()
            if answer[-1] != ".":
                answer += "."
            sqa_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': prompt,
                'caption': answer,
                'sqa_type': question_type
            })
        else:
            sqa_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': prompt,
                'ref_captions': answers,
                'sqa_type': question_type
            })
        # print(sqa_annos[-1])
        # breakpoint()
    with open(f"annotations/sqa3d_{split}.json", "w") as f:
        json.dump(sqa_annos, f, indent=4)

    

