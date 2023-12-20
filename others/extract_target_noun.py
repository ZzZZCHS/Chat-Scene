import json
import torch
import sys

from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from collections import defaultdict

from tqdm import tqdm


llama_model_path = "model/vicuna-7b-v0"

print("Loading LLaMA")
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
model = LlamaForCausalLM.from_pretrained(
    llama_model_path,
    torch_dtype=torch.float16
).cuda()
for p in model.parameters():
    p.requires_grad = False
print("Loading LLaMA Done")

model.eval()


dataset_name = "sr3d_merge"
anno_file = f"anno/{dataset_name}_captions.json"
output_anno_file = f"anno/{dataset_name}_captions_noun.json"
annos = json.load(open(anno_file, "r"))
output_annos = defaultdict(list)

# nr3d
# prompt_head = "###System: Given a sentence that asks for an object in a scene. Extract the primary subject from each sentence and include any accompanying adjectives, if present. " \
#               "###Human: When facing the bookcases, choose the plant directly on the right, next to the right most bookcase. " \
#               "###Assistant: plant. " \
#               "###Human: The big black box between the door and the couch. " \
#               "###Assistant: big black box. " \
#               "###Human: pick the white pillow that has a pillow above and under it. " \
#               "###Assistant: white pillow. "

# sr3d
prompt_head = "###System: Given a sentence that asks for an object in a scene. Extract the primary subject from each sentence and include any accompanying adjectives, if present. " \
              "###Human: find the office chair that is near the copier. " \
              "###Assistant: office chair. " \
              "###Human: select the trash can that is near the printer. " \
              "###Assistant: trash can. " \
              "###Human: the monitor that is near the door. " \
              "###Assistant: monitor. "

for i, k in tqdm(enumerate(annos.keys())):
    print(f"{i} / {len(annos)}")
    captions = annos[k]
    for caption in captions:
        end_1 = (caption.find(".") + len(caption)) % len(caption)
        # end_2 = (caption.find(",") + len(caption)) % len(caption)
        bk = end_1
        sen_caption = caption[:bk] + "."
        prompt = prompt_head + "###Human: " + sen_caption + " ###"
        input_token = llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda:0")
        input_embed = model.model.embed_tokens(input_token.input_ids)
        outputs = model.generate(
            inputs_embeds=input_embed,
            max_new_tokens=16,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0
        )
        output = outputs[0]
        if output[0] == 0:
            output = output[1:]
        if output[0] == 1:
            output = output[1:]
        output_text = llama_tokenizer.decode(output, add_special_tokens=False)
        print("INPUT:", sen_caption)
        print("OUTPUT:", output_text)
        try:
            output_annos[k].append(output_text.split("Assistant:")[1].split(".")[0].strip())
            print("EX OUTPUT:", output_text.split("Assistant:")[1].split(".")[0].strip())
        except Exception:
            print("Fail:")
            output_annos[k].append(caption[:bk])

with open(output_anno_file, "w") as f:
    json.dump(output_annos, f, indent=4)
