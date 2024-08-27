which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

epoch=3
batch_size=32
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=True
add_scene_token=False
no_obj=False
input_dim=1024 # 1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
config=""
max_grad_norm=0.01
seed=42
use_location_token=False

llama_model_path="llm/vicuna-7b-v1.5"

train_tag="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref"
val_tag="sqa3d#scanrefer#scan2cap#scanqa#multi3dref"

evaluate=True
debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=False
    gpu_num=4
    do_save=True
    other_info="new_max1"
fi

tag="${train_tag}__${val_tag}__${other_info}"

# pretrained_path="/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240517_142546_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#nr3d_caption__scanrefer#scan2cap#scanqa#multi3dref#sqa3d__r16alpha32_videofeats_maxgrad1e-1_fusefeat/ckpt_01_3214.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240517_221942_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#nr3d_caption__scanrefer#scan2cap#scanqa#multi3dref#sqa3d__r16alpha32_maxgrad1e-1/ckpt_01_3214.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240512_015550_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#nr3d_caption__scanrefer#scan2cap#scanqa#multi3dref#sqa3d__v2.1_videofeat_r16alpha8/ckpt_01_3214.pth"  # SOTA
# pretrained_path="/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240802_150615_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#nr3d_caption__scanrefer#scanqa__3d2d/ckpt_01_3214.pth" # re-train 3d2d
pretrained_path="/mnt/petrelfs/huanghaifeng/share_hw/Chat-3D-v2/outputs/20240819_130946_lr5e-6_ep3_scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref__scanrefer#scan2cap#scanqa#multi3dref__only_scanrefer/ckpt_01_3214.pth"

OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
mkdir -p ${OUTPUT_DIR}

srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=${gpu_num} --kill-on-bad-exit --quotatype=reserved \
python tasks/train.py \
    "$(dirname $0)/${config}config.py" \
    output_dir "$OUTPUT_DIR" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.add_scene_token "$add_scene_token" \
    model.add_img_token "$add_img_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    model.train_emb "$train_emb" \
    model.train_img_proj "$train_img_proj" \
    train_tag "$train_tag" \
    val_tag "$val_tag" \
    model.no_obj "$no_obj" \
    segmentor "$segmentor" \
    pc_encoder "$pc_encoder" \
    model.input_dim "$input_dim" \
    model.bidirection "$bidirection" \
    optimizer.different_lr.enable "$different_lr" \
    model.max_obj_num "$max_obj_num" \
    lora.lora_r "$lora_r" \
    lora.lora_alpha "$lora_alpha" \
    model.add_pos_emb "$add_pos_emb" \
    model.feat_fusion "$feat_fusion" \
    optimizer.max_grad_norm "$max_grad_norm" \
    seed "$seed" \
    model.fuse_with_id "$fuse_with_id" \
    model.llama_model_path "$llama_model_path" \
    model.use_location_token "$use_location_token"

