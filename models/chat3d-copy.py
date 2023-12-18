import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from models.transformer_vanilla import TransformerEncoder
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine

from transformers import StoppingCriteria, StoppingCriteriaList

import contextlib

logger = logging.getLogger(__name__)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        llama_model_path = config.get("llama_model_path")
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # prompt
        prompt_path = config.get("prompt_path", "")
        self.prompt_template = config.get("prompt_template", "")
        self.max_txt_len = config.get("max_txt_len", 32)
        self.end_sym = config.get("end_sym", '\n')
        self.system_path = config.get("system_path", "")
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.pc_start_token, self.pc_end_token = "<Target>", "</Target>"
        self.scene_start_token, self.scene_end_token = "<Scene>", "</Scene>"

        mlp_dropout = config.get("mlp_dropout", 0.5)
        self.stage = config.get("stage", 1)

        self.low_resource = low_resource

        self.input_dim = config.get("input_dim", 512)
        self.input_attr_dim = config.get("input_attr_dim", 9)

        logger.info('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )
        logger.info("freeze LLAMA")
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        # for layer_ind in range(29, 32):
        #     for param in self.llama_model.model.layers[layer_ind].parameters():
        #         param.requires_grad = True
        #         param.data = param.data.float()
        logger.info('Loading LLAMA Done')

        self.scene_proj = GenericMLP(
            input_dim=self.input_dim * 2,
            hidden_dims=[self.llama_model.config.hidden_size],
            output_dim=self.llama_model.config.hidden_size,
            norm_fn_name="ln",
            output_use_activation=False,
            output_use_norm=True,
            output_use_bias=False,
            dropout=mlp_dropout,
            weight_init_std=self.llama_model.config.initializer_range
        )
        self.size_color_proj = nn.Linear(
            6, self.input_dim
        )
        self.pos_proj = nn.Linear(
            3, self.llama_model.config.hidden_size
        )
        # for m in [self.size_color_proj, self.pos_proj]:
        #     m.weight.data.normal_(mean=0.0, std=self.llama_model.config.initializer_range)
        #     m.bias.data.zero_()
        # self.pos_embedding = PositionEmbeddingCoordsSine(
        #     d_pos=self.llama_model.config.hidden_size, pos_type="fourier"   # ??????????
        # )
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.llama_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.encoder_num_layers = config.get("encoder_num_layers", 1)
        self.relation_module = TransformerEncoder(dim=self.llama_model.config.hidden_size, num_layers=self.encoder_num_layers, dropout=mlp_dropout)

        if self.stage == 1:
            for p in self.relation_module.parameters():
                p.requires_grad = False
            for p in self.pos_proj.parameters():
                p.requires_grad = False
        # else:
        #     for p in self.size_color_proj.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_proj.parameters():
        #         p.requires_grad = False
        # else:
        #     for p in self.size_color_proj.parameters():
        #         p.requires_grad = False
        #     for p in self.scene_proj.parameters():
        #         p.requires_grad = False

        with open(self.system_path, "r") as f:
            self.system = "\n".join([x.strip() for x in f.readlines()])
        # if prompt_path:
        #     self.prompt_list = self.process_prompt(prompt_path, prompt_template)
        # else:
        #     self.prompt_list = []

        self.object_norm = torch.norm(self.get_text_emb("object"), p=2)
        self.relation_norm = torch.norm(self.get_text_emb("relation"), p=2)

    # def process_prompt(self, prompt_path, prompt_template):
        # with open(prompt_path, 'r') as f:
        #     prompt_candidates = f.read().splitlines()
        # with open(self.system_path, "r") as f:
        #     system = "\n".join([x.strip() for x in f.readlines()])
        # prompt_list = [system + " " + prompt_template.format(p) for p in prompt_candidates]
        # logger.info(f'Load {len(prompt_list)} training prompts')
        # logger.info(f'Prompt: {prompt_list}')
        # return prompt_list

    # def prompt_wrap(self, scene_embed, scene_mask, prompts, is_eval=False):
    #     batch_size = scene_embed.shape[0]
    #     for i, prompt in enumerate(prompts):
    #         p_0, p_1 = prompt.split('<REPLACE>')
    #         p_0_tokens = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(scene_embed.device)
    #         p_1_tokens = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(scene_embed.device)
    #         # p_2_tokens = self.llama_tokenizer(p_2, return_tensors="pt", add_special_tokens=False).to(pc_embed.device)
    #         p_0_embeds = self.llama_model.model.embed_tokens(p_0_tokens.input_ids).expand(batch_size, -1, -1)
    #         p_1_embeds = self.llama_model.model.embed_tokens(p_1_tokens.input_ids).expand(batch_size, -1, -1)
    #         p_2_embeds = self.llama_model.model.embed_tokens(p_2_tokens.input_ids).expand(batch_size, -1, -1)
    #         wrapped_embeds = torch.cat([p_0_embeds, pc_embed, p_1_embeds, scene_embed, p_2_embeds], dim=1)
    #         wrapped_atts = scene_attn[:, :1].expand(-1, wrapped_embeds.shape[1])
    #     return wrapped_embeds, wrapped_atts

    def get_object_list_embed(self, scene_embed, scene_mask):
        # scene_embed: (obj_num, dim)
        embed_list = []
        for i in range(scene_embed.shape[0]):
            if scene_mask[i] == 0:
                break
            text = ""
            if i > 0:
                text += ", "
            text += f"obj{i:02}: "
            text_embeds = self.get_text_emb(text, scene_embed.device).detach()
            embed_list.extend([text_embeds.squeeze(0), scene_embed[i:i+1]])
        return torch.cat(embed_list, dim=0)

    def get_text_emb(self, text, device="cpu"):
        text_tokens = self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        return self.llama_model.model.embed_tokens(text_tokens.input_ids)

    def encode_and_project(self, feat, attr):
        # pos_emb = self.pos_embedding(attr[:, :, :3], self.input_dim).permute(0, 2, 1)
        # feat = torch.nn.functional.normalize(feat, dim=-1)
        # pos_emb = self.pos_proj(attr[:, :, :3])
        size_color_emb = self.size_color_proj(attr[:, :, 3:])
        # feat = self.input_norm(feat + size_color_emb + pos_emb)
        feat = torch.cat([feat, size_color_emb], dim=-1)
        feat = self.scene_proj(feat)
        # feat = torch.nn.functional.normalize(feat, dim=-1) * self.object_norm
        return feat

    def forward_stage1(self, scene_feat, scene_attr, obj_ids, target_captions, is_eval=False, **kwargs):
        pc_feat = torch.gather(scene_feat, 1, obj_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_feat.shape[-1]))
        pc_attr = torch.gather(scene_attr, 1, obj_ids.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_attr.shape[-1]))
        pc_embed = self.encode_and_project(pc_feat, pc_attr).squeeze(1)
        pc_embed = torch.nn.functional.normalize(pc_embed, dim=-1) * self.object_norm

        target_embeds = []
        for target_caption in target_captions:
            target_tokens = self.llama_tokenizer(
                target_caption,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(pc_embed.device)
            token_mask = target_tokens["attention_mask"].unsqueeze(-1)
            target_embed = self.llama_model.model.embed_tokens(target_tokens.input_ids)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            target_embed = (target_embed * token_mask).sum(1) / token_mask.sum(1)
            target_embed = target_embed.mean(dim=0)
            target_embeds.append(target_embed)
        target_embeds = torch.stack(target_embeds, dim=0).to(pc_embed.device)
        cosine_loss = F.cosine_embedding_loss(pc_embed, target_embeds.detach(), torch.tensor([1]).to(pc_embed.device))
        l2_loss = F.mse_loss(pc_embed, target_embeds.detach())
        # print(torch.norm(pc_embed[:1], p=2), torch.norm(target_embeds[:1], p=2))
        cosine_score = 1. - cosine_loss.detach().cpu()
        return dict(
            loss=cosine_loss,
            cosine_score=cosine_score,
            l2_dis=l2_loss.detach().cpu()
        )

    def forward_stage2(self, scene_feat, scene_attr, scene_mask, obj_ids, questions, answers, is_eval=False, **kwargs):
        scene_embed = self.encode_and_project(scene_feat, scene_attr)
        # pos_embed = self.pos_proj(scene_attr[:, :, :3])
        # scene_relation = self.relation_module(self.llama_norm(scene_embed+pos_embed), mask=scene_mask.bool())
        # scene_embed = torch.nn.functional.normalize(scene_embed, dim=-1) * self.object_norm
        # scene_relation = torch.nn.functional.normalize(scene_relation, dim=-1) * self.relation_norm
            # breakpoint()
        #     scene_feat = torch.nn.functional.normalize(scene_feat, dim=-1) * self.object_norm
        # pc_embed = torch.nn.functional.normalize(pc_embed, dim=-1) * self.object_norm

        # scene_feat = self.scene_proj(scene_feat)
        # scene_embed = scene_feat * scene_mask.unsqueeze(-1)
        # scene_attn = torch.ones(scene_embed.size()[:-1], dtype=torch.long).to(scene_embed.device)

        # prompt = random.choice(self.prompt_list)
        # scene_embed, scene_attn = self.prompt_wrap(scene_embed, scene_mask, questions, is_eval)

        # self.llama_tokenizer.padding_side = "right"
        # answers = [t + self.end_sym for t in answers]
        # to_regress_tokens = self.llama_tokenizer(
        #     answers,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     add_special_tokens=False
        # ).to(scene_embed.device)
        # answer_targets = to_regress_tokens.input_ids.masked_fill(
        #     to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        # )
        # to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        input_embed_list, attn_list, target_list = [], [], []
        max_seq_len = 0
        for i, question in enumerate(questions):
            prompt = self.system + "\n" + self.prompt_template.format(question)
            p_0, p_1 = prompt.split("<REPLACE>")
            p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(scene_embed.device)
            p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(scene_embed.device)
            # p_2_tokens = self.llama_tokenizer(p_2, return_tensors="pt", add_special_tokens=False).to(pc_embed.device)
            p_0_embed = self.llama_model.model.embed_tokens(p_0_token.input_ids).detach()
            p_1_embed = self.llama_model.model.embed_tokens(p_1_token.input_ids).detach()
            # p_2_embeds = self.llama_model.model.embed_tokens(p_2_tokens.input_ids).expand(batch_size, -1, -1)
            object_list_embed = self.get_object_list_embed(scene_embed[i], scene_mask[i])
            wrapped_embed = torch.cat([p_0_embed.squeeze(0), object_list_embed, p_1_embed.squeeze(0)], dim=0)
            wrapped_attn = torch.ones(wrapped_embed.size()[:-1], dtype=torch.long).to(wrapped_embed.device)
            empty_target = (
                torch.ones([wrapped_attn.shape[0]+1], dtype=torch.long).to(scene_embed.device).fill_(-100)
            )

            answer = answers[i] + self.end_sym
            to_regress_token = self.llama_tokenizer(answer, return_tensors="pt").to(scene_embed.device)
            # breakpoint()
            answer_target = to_regress_token.input_ids.masked_fill(
                to_regress_token.input_ids == self.llama_tokenizer.pad_token_id, -100
            ).squeeze(0)
            to_regress_embed = self.llama_model.model.embed_tokens(to_regress_token.input_ids).squeeze(0)

            target = torch.cat([empty_target, answer_target], dim=0)
            bos = torch.ones([1], dtype=to_regress_token.input_ids.dtype, device=to_regress_token.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embed = self.llama_model.model.embed_tokens(bos)
            bos_attn = wrapped_attn[:1]
            input_embed = torch.cat([bos_embed, wrapped_embed, to_regress_embed], dim=0)
            attn = torch.cat([bos_attn, wrapped_attn, to_regress_token.attention_mask[0]], dim=0)
            input_embed_list.append(input_embed)
            attn_list.append(attn)
            target_list.append(target)
            max_seq_len = max(max_seq_len, target.shape[0])

        batch_size = scene_embed.shape[0]
        dim = scene_embed.shape[2]

        input_embeds = torch.zeros([batch_size, max_seq_len, dim], dtype=input_embed_list[0].dtype).to(scene_embed.device)
        attention_mask = torch.zeros([batch_size, max_seq_len], dtype=attn_list[0].dtype).to(scene_embed.device)
        targets = torch.zeros([batch_size, max_seq_len], dtype=target_list[0].dtype).to(scene_embed.device).fill_(-100)
        for i in range(len(input_embed_list)):
            input_embed = input_embed_list[i]
            attn = attn_list[i]
            target = target_list[i]
            input_embeds[i, :input_embed.shape[0], :] = input_embed
            attention_mask[i, :attn.shape[0]] = attn
            targets[i, :target.shape[0]] = target

        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        return dict(
            loss=outputs.loss,
        )

    def forward_stage3(self, scene_feat, scene_attr, scene_mask, target_id, conversations, is_eval=False, **kwargs):
        batch_size, obj_num, _ = scene_feat.shape
        scene_feat = self.encode_and_project(scene_feat, scene_attr)
        pc_embed = torch.gather(scene_feat, 1, target_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_feat.shape[-1]))
        if self.encoder_num_layers > 0:
            scene_feat = self.relation_module(scene_feat, mask=(~scene_mask.bool()).unsqueeze(1).expand(-1, obj_num, -1).unsqueeze(1))

        scene_embed = scene_feat * scene_mask.unsqueeze(-1)
        # scene_attn = torch.ones(scene_embed.size()[:-1], dtype=torch.long).to(scene_embed.device)
        max_len = 0
        input_embed_list = []
        p_0_len_list, p_1_len_list = [], []
        target_list = []
        for idx, prompt in enumerate(conversations):
            tmp_scene_embed = scene_embed[idx:idx+1]
            tmp_pc_embed = pc_embed[idx:idx+1]
            p_0, p_ = prompt.split("<TargetHere>")
            p_1, p_2 = p_.split("<SceneHere>")
            p_1 = self.pc_end_token + p_1
            p_0_tokens = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(tmp_pc_embed.device)
            p_1_tokens = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(tmp_pc_embed.device)
            p_2_tokens = self.llama_tokenizer(p_2, return_tensors="pt", add_special_tokens=False).to(tmp_pc_embed.device)
            p_0_embeds = self.llama_model.model.embed_tokens(p_0_tokens.input_ids)
            p_1_embeds = self.llama_model.model.embed_tokens(p_1_tokens.input_ids)
            p_2_embeds = self.llama_model.model.embed_tokens(p_2_tokens.input_ids)
            input_embeds = torch.cat([p_0_embeds, tmp_pc_embed, p_1_embeds, tmp_scene_embed, p_2_embeds], dim=1)

            sep1 = self.begin_signal + self.role[0] + ":"
            sep2 = self.begin_signal + self.role[1] + ":"
            raw_text = p_2.split(sep2)
            for _idx in range(1, len(raw_text)):
                raw_text[_idx] = sep2 + raw_text[_idx]
            answer_targets = p_2_tokens.input_ids.clone()
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system.rstrip())
            sep_len = self._get_text_len(sep1.rstrip())
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]:
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            if cur_len != answer_targets.shape[1]:
                print(f"The final length is not equal to the original prompt: {prompt}")
            assert cur_len == answer_targets.shape[1], (cur_len, answer_targets.shape[1])

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_0_len_list.append(p_0_tokens.input_ids.shape[1])
            p_1_len_list.append(p_1_tokens.input_ids.shape[1])
            target_list.append(answer_targets)

        txt_len = min(max_len + 1, self.max_txt_len + obj_num + 1)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(pc_embed.device) * self.llama_tokenizer.pad_token_id
        inputs_embeds = self.llama_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(pc_embed.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(pc_embed.device).fill_(-100)
        inputs_embeds[:, :1] = self.llama_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            attention_mask[idx, :(input_len+1)] = 1
            p_0_len = p_0_len_list[idx]
            p_1_len = p_1_len_list[idx]
            targets[idx, (p_0_len+p_1_len+obj_num+2):(input_len+1)] = target_list[idx][0, :(input_len-p_0_len-p_1_len-obj_num-1)]

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets
        )

        return dict(
            loss=outputs.loss
        )

    def evaluate(self, scene_feat, scene_attr, scene_mask, custom_prompt, is_eval=True, **kwargs):
        batch_size, obj_num = scene_feat.shape[:2]
        scene_embed = self.encode_and_project(scene_feat, scene_attr)
        # pos_embed = self.pos_proj(scene_attr[:, :, :3])
        # scene_relation = self.relation_module(self.llama_norm(scene_embed+pos_embed), mask=scene_mask.bool())
        scene_embed = torch.nn.functional.normalize(scene_embed, dim=-1) * self.object_norm
        # scene_relation = torch.nn.functional.normalize(scene_relation, dim=-1) * self.relation_norm
        # if self.encoder_num_layers > 0:
        #     scene_feat = self.relation_module(scene_feat, mask=scene_mask.bool())
        #     scene_feat = torch.nn.functional.normalize(scene_feat, dim=-1) * self.object_norm

        # scene_feat = self.scene_proj(scene_feat)

        # scene_embed = scene_feat * scene_mask.unsqueeze(-1)
        # scene_attn = torch.ones(scene_embed.size()[:-1], dtype=torch.long).to(scene_embed.device)

        output_texts = []
        for idx in range(batch_size):
            # tmp_scene_embed, _ = self.prompt_wrap(pc_embed[idx:idx+1], scene_embed[idx:idx+1], scene_attn[idx:idx+1], custom_prompt[idx], is_eval)
            p_0, p_1 = custom_prompt[idx].split("<REPLACE>")
            p_0_token = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(scene_embed.device)
            p_1_token = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(scene_embed.device)
            p_0_embed = self.llama_model.model.embed_tokens(p_0_token.input_ids)
            p_1_embed = self.llama_model.model.embed_tokens(p_1_token.input_ids)
            object_list_embed = self.get_object_list_embed(scene_embed[idx], scene_mask[idx]).unsqueeze(0)
            wrapped_embed = torch.cat([p_0_embed, object_list_embed, p_1_embed], dim=1)
            stop_words_ids = [torch.tensor([2]).to(wrapped_embed.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            outputs = self.llama_model.generate(
                inputs_embeds=wrapped_embed,
                max_new_tokens=min(self.max_txt_len * 2, 512),
                stopping_criteria=stopping_criteria,
                num_beams=1,
                do_sample=True,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=1,
                temperature=1.0,
            )
            output_token = outputs[0]
            if output_token[0] == 0:  # the model might output an unknown token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split(self.end_sym)[0]
            output_texts.append(output_text)

        return output_texts

    def forward(self, **kwargs):
        if "target_captions" in kwargs:
            return self.forward_stage1(**kwargs)
        if "answers" in kwargs:
            return self.forward_stage2(**kwargs)
        if "conversations" in kwargs:
            return self.forward_stage3(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device


# class CustomBackwardFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, scaling_factor):
#         # Save the scaling factor for later use in backward pass
#         ctx.scaling_factor = scaling_factor
#         return input
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # Scale the gradients by the previously saved scaling factor
#         return grad_output * ctx.scaling_factor, None
#
#
# def scale_gradients(grad):
#     return grad * 10.
