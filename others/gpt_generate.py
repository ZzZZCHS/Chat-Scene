import argparse
import json
import os

import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(content: str, max_tokens: int):
    sleep_time = NUM_SECONDS_TO_SLEEP
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4',
                # model="gpt-3.5-turbo-0613",
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except Exception as e:
            print(e)
        print(f"!!sleep for {sleep_time}")
        time.sleep(sleep_time)
        sleep_time *= 2

    return response['choices'][0]['message']['content']


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based dataset generation.')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()


    idx = 0
    prompt_head = "You are a 3D scene understanding expert specializing in 3D visual assistance. I will provide you with a catalog of objects within a 3D scene, with each object\'s information presented in the following format: object\'s ID, object\'s class name, object\'s 3D coordinates, and a concise description of the object. The object list is as follows:\n"
    prompt_end = "Your task is to generate a comprehensive description of the entire scene. This description should encompass an analysis of the scene's functionality, an examination of the key objects within the scene, their spatial relationships with the surrounding objects, an assessment of the arrangement of the objects within the scene, and other relevant insights. An important guideline to follow is that when referring to an object, you must explicitly include its object ID. The description should be more than 200 words and less than 300 words."
    import glob
    for split in ["train", "val"]:
        for file_path in glob.glob(f"annotations/scene_dataset/obj_info_list/{split}/*.json"):
            scene_id = file_path.split("/")[-1][:-5]
            print("-" * 20)
            print(scene_id)
            output_path = f"annotations/scene_dataset/gpt_generation/{split}/{scene_id}.json"
            if os.path.exists(output_path):
                print("skip")
                continue
            obj_infos = json.load(open(file_path, "r"))
            prompt = prompt_head + obj_infos + prompt_end
            print(prompt)
            answer = get_eval(prompt, args.max_tokens)
            print(answer)
            with open(output_path, "w") as f:
                json.dump(answer, f)



