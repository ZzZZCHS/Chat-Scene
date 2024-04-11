# ========================= data ==========================
anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d"
segmentor = "mask3d"
feat_file = f"{anno_root}/scannet_{pc_encoder}_feats.pt"
img_feat_file = f"{anno_root}/scannet_img_dinov2_features.pt"
seg_img_feat_file = f"{anno_root}/scannet_img_mask3d_dinov2_features.pt"
# img_feat_file = f"{anno_root}/scannet_llava_img_features.pt"
# attribute_file = f"{anno_root}/scannet_attributes.json"
train_file_s2 = [
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/obj_align_train_one_scene.json"
    # ],
    [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scannet_train_stage2_caption_OBJ.json",
    ],
    [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanrefer_train_stage2_caption_OBJ.json",
    ],
    [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scannet_train_region_caption.json",
    ],
    [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/nr3d_train_stage2_caption_OBJ.json"
    ],
    [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanrefer_train_stage2_grounding_OBJ.json"
    ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_img_mask3d_dinov2_features.pt",
    #     f"{anno_root}/scannet_{segmentor}_train_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_train_stage2_grounding_OBJ.json"
    # ],
    [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/obj_align_train_OBJ.json"
    ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_img_mask3d_dinov2_features.pt",
    #     f"{anno_root}/scannet_{segmentor}_train_attributes.pt",
    #     f"{anno_root}/scanqa_train_stage2.json"
    # ],
    [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanqa_train_stage2.json"
    ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_train_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_train_stage2_caption_iou50.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_train_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_train_stage2_caption_iou25.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/sr3d_train_stage2_objxx.json"
    # # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scene_align_train.json",
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/obj_align_train.json",
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scanqa_train_stage2_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/nr3d_train_stage2_grounding_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/sr3d_train_stage2_grounding_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/nr3d_train_stage2_multichoice0.01.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/scene_dataset_train_stage2.json"
    # ]
]
val_file_s2 = [
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_train_attributes.pt",
    #     f"{anno_root}/obj_align_val_one_scene.json"
    # ],
    # [
    #     feat_file,
    #     img_feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanrefer_val_stage2_caption_OBJ.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/nr3d_val_stage2_caption100_OBJ.json"
    # ],
    # [
    #     feat_file,
    #     img_feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/obj_align_val_OBJ.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_img_mask3d_dinov2_features.pt",
    #     f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_val_stage2_grounding_OBJ.json"
    # ],
    [
        f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
        f"{anno_root}/scannet_img_mask3d_dinov2_features.pt",
        f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
        f"{anno_root}/scanqa_val_stage2.json"
    ],
    # [
    #     feat_file,
    #     img_feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanqa_val_stage2.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanqa_val_stage2_objxx100.json"
    # ],
    # [
    #     f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
    #     f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
    #     f"{anno_root}/scanrefer_{segmentor}_val_stage2_caption_iou25.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/stage2_val400.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/nr3d_val_stage2_objxx.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scene_align_val.json",
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/obj_align_val.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scanqa_val_stage2_objxx.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/sr3d_val_stage2_grounding_new.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/nr3d_val_stage2_multichoice0.01.json"
    # ],
    # [
    #     feat_file,
    #     f"{anno_root}/scannet_val_attributes.pt",
    #     f"{anno_root}/scene_dataset_val_stage2.json"
    # ]
]

train_tag = 'scannet_caption'
val_tag = 'scanqa'

train_file_dict = {
    'scannet_caption': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scannet_train_stage2_caption_OBJ.json"
    ],
    'scanrefer_caption': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanrefer_train_stage2_caption_OBJ.json",
    ],
    'scannet_region_caption': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scannet_train_region_caption.json",
    ],
    'nr3d_caption': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/nr3d_train_stage2_caption_OBJ.json"
    ],
    'scanrefer': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanrefer_train_stage2_grounding_OBJ.json"
    ],
    'obj_align': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/obj_align_train_OBJ.json"
    ],
    'scanqa': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/scanqa_train.json"
    ],
    'objaverse': [
        f"{anno_root}/objaverse_uni3d_feature_train.pt",
        None,
        None,
        f"{anno_root}/objaverse_caption_train.json",
        1
    ],
    'sqa3d': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_train_attributes.pt",
        f"{anno_root}/sqa3d_train.json"
    ]
}

val_file_dict = {
    'scanqa': [
        f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
        seg_img_feat_file,
        f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
        f"{anno_root}/scanqa_val.json"
    ],
    'scanrefer': [
        f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
        seg_img_feat_file,
        f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
        f"{anno_root}/scanrefer_{segmentor}_val_stage2_grounding_OBJ.json"
    ],
    'scan2cap': [
        f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
        seg_img_feat_file,
        f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
        f"{anno_root}/scanrefer_{segmentor}_val_stage2_caption_OBJ.json"
    ],
    'scanrefer_caption': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_val_attributes.pt",
        f"{anno_root}/scanrefer_val_stage2_caption_OBJ.json"
    ],
    'objaverse': [
        f"{anno_root}/objaverse_uni3d_feature_val.pt",
        None,
        None,
        f"{anno_root}/objaverse_caption_val.json"
    ],
    'obj_align': [
        feat_file,
        img_feat_file,
        f"{anno_root}/scannet_val_attributes.pt",
        f"{anno_root}/obj_align_val.json"
    ],
    'sqa3d': [
        f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt",
        seg_img_feat_file,
        f"{anno_root}/scannet_{segmentor}_val_attributes.pt",
        f"{anno_root}/sqa3d_val.json"
    ]
}



test_types = []
num_workers = 32

# ========================= input ==========================
batch_size = 32

pre_text = False


# ========================= model ==========================
model = dict(
    llama_model_path="model/vicuna-7b-v1.5",
    input_dim=1024 if pc_encoder == "uni3d" else 512,
    img_input_dim=1024,
    attr_dim=512,
    scene_dim=256,
    encoder_num_layers=3,
    mlp_dropout=0.1,
    low_resource=False,
    system_path="prompts/system.txt",
    instruction_path="prompts/instruction.txt",
    max_txt_len=512,
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    stage=3,
    add_scene_token=True,
    add_img_token=True,
    obj_norm_scale=200,
    scene_norm_scale=50,
    grad_scale=1,
    use_lora=False,
    train_emb=True,
    train_img_proj=False,
    no_obj=False
)

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05
)

optimizer = dict(
    opt="adamW",
    lr=5e-3,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["module.object_img_proj"],
        lr=[5e-7],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False
deep_fusion = False

fp16 = False
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="huanghaifeng",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="Scene-LLM",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 20
# eval_freq = 500
seed = 42

save_latest = False
do_save = True
auto_resume = True
pretrained_path = ""
img_projector_path = None

debug=False
gpu_num=1