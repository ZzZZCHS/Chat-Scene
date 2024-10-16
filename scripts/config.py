# ========================= data ==========================
anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d"
segmentor = "mask3d"
version = ""

gt_feat_file = f"{anno_root}/scannet_gt_{pc_encoder}_feats.pt"
seg_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt"
seg_all_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats_all.pt"
gt_img_feat_file = f"{anno_root}/scannet_gt_videofeats.pt"
seg_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats.pt"
seg_all_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats_all.pt"
gt_train_attr_file = f"{anno_root}/scannet_train_attributes.pt"
gt_val_attr_file = f"{anno_root}/scannet_val_attributes.pt"
seg_train_attr_file = f"{anno_root}/scannet_{segmentor}_train_attributes.pt"
seg_val_attr_file = f"{anno_root}/scannet_{segmentor}_val_attributes.pt"
seg_all_attr_file = f"{anno_root}/scannet_{segmentor}_all_attributes.pt"

train_tag = 'scanqa'
val_tag = 'scanqa'

train_file_dict = {
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_train{version}.json"
    ],
    'scanrefer_location': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_train_location{version}.json"
    ],
    'nr3d_seg': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_{segmentor}_train{version}.json"
    ],
    'sr3d_seg': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sr3d_{segmentor}_train{version}.json"
    ],
    'nr3d': [
        gt_feat_file,
        gt_img_feat_file,
        gt_train_attr_file,
        f"{anno_root}/nr3d_train{version}.json"
    ],
    'sr3d': [
        gt_feat_file,
        gt_img_feat_file,
        gt_train_attr_file,
        f"{anno_root}/sr3d_train{version}.json"
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_train{version}.json"
    ],
    'scan2cap_location': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_train_location{version}.json"
    ],
    'nr3d_caption': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_caption_{segmentor}_train{version}.json"
    ],
    'obj_align': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/obj_align_{segmentor}_train{version}.json"
    ],
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanqa_train.json"
    ],
    'sqa3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sqa3d_train.json"
    ],
    'multi3dref': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_train{version}.json"
    ],
    'multi3dref_location': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_train_location{version}.json"
    ],
    'scannet_caption': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scannet_caption_{segmentor}_train{version}.json"
    ],
    'scannet_region_caption': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scannet_region_caption_{segmentor}_train{version}.json",
    ]
}

val_file_dict = {
    'scanqa': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanqa_val.json"
    ],
    'scanqa_test': [
        seg_all_feat_file,
        seg_all_img_feat_file,
        seg_all_attr_file,
        f"{anno_root}/scanqa_test.json"
    ],
    'scanrefer': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_val{version}.json"
    ],
    'scanrefer_test': [
        seg_all_feat_file,
        seg_all_img_feat_file,
        seg_all_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_test{version}.json"
    ],
    'scanrefer_location': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_val_location{version}.json"
    ],
    'nr3d': [
        gt_feat_file,
        gt_img_feat_file,
        gt_val_attr_file,
        f"{anno_root}/nr3d_val{version}.json"
    ],
    'nr3d_seg': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/nr3d_{segmentor}_val{version}.json"
    ],
    'sr3d': [
        gt_feat_file,
        gt_img_feat_file,
        gt_val_attr_file,
        f"{anno_root}/sr3d_val{version}.json"
    ],
    'sr3d_seg': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sr3d_{segmentor}_val{version}.json"
    ],
    'scan2cap': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_val{version}.json"
    ],
    'scan2cap_test': [
        seg_all_feat_file,
        seg_all_img_feat_file,
        seg_all_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_test{version}.json"
    ],
    'scan2cap_location': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_val_location{version}.json"
    ],
    'sqa3d': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sqa3d_test.json"
    ],
    'sqa3d_val': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sqa3d_val.json"
    ],
    'multi3dref': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_val{version}.json"
    ],
    'multi3dref_location': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_val_location{version}.json"
    ],
}


num_workers = 32
batch_size = 32


# ========================= model ==========================
model = dict(
    llama_model_path="llm/vicuna-7b-v1.5",
    input_dim=1024,
    img_input_dim=1024,
    attr_dim=512,
    scene_dim=256,
    pos_dim=128,
    encoder_num_layers=3,
    low_resource=False,
    system_path="prompts/system.txt",
    instruction_path="prompts/instruction.txt",
    max_txt_len=64,
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    add_scene_token=True,
    add_img_token=True,
    use_lora=True,
    train_emb=True,
    train_img_proj=False,
    no_obj=False,
    max_obj_num=200,
    bidirection=False,
    add_pos_emb=False,
    feat_fusion=False,
    fuse_with_id=False,
    use_objid=True,
    use_location_token=False
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
    scaler_enable=False,
    max_grad_norm=5,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["model.embed_tokens"],
        lr=[5e-4],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False

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
img_projector_path = ""

debug=False
gpu_num=1