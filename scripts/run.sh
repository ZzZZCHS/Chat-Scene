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
bidirection=False  # !!!
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=8
add_pos_emb=False
feat_fusion=False
config=""

# train_tag="scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#scannet_caption#scannet_region_caption#nr3d_caption"
train_tag="scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#nr3d_caption"
val_tag="scanrefer#scan2cap#scanqa#multi3dref#sqa3d"
# val_tag="scan2cap"

evaluate=False
debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=True
    gpu_num=4
    do_save=True
    other_info="r16alpha8_videofeats_maxgrad5"
fi

tag="${train_tag}__${val_tag}__${other_info}"

pretrained_path=""
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240508_023155_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#nr3d_caption__scanrefer#scan2cap#scanqa#multi3dref#sqa3d__v2.1_bidirection/ckpt_00_1607.pth"

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
    model.add_pos_emb "$add_pos_emb" \
    model.feat_fusion "$feat_fusion"
