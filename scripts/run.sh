which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((53000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

epoch=3
batch_size=1
lr=5e-6
train_emb=True
train_img_proj=False
add_img_token=False
add_scene_token=False

# train_tag="scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#scannet_caption#scannet_region_caption"
train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
val_tag="multi3dref#scanqa#scanrefer#sqa3d#scan2cap"
# val_tag="sqa3d"

evaluate=True
debug=true
if [ $debug = "true" ]; then
    enable_wandb=False
    gpu_num=3
    do_save=False
    other_info="debug"
else
    enable_wandb=True
    gpu_num=4
    do_save=True
    other_info="v2.1"
fi

tag="${train_tag}__${val_tag}__${other_info}"

# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240420_102938_dp0.1_lr5e-6_sta2_ep3_scanrefer_seg#scan2cap_seg#nr3d_caption_seg#obj_align_seg#scanqa_seg#sqa3d_seg#multi3dref_seg__multi3dref#scanqa#scanrefer#sqa3d#scan2cap__v2.1/ckpt_01_3028.pth"

OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_lr"$lr"_ep"$epoch"_"$tag"
mkdir -p ${OUTPUT_DIR}

srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=${gpu_num} --kill-on-bad-exit \
python tasks/train.py \
    $(dirname $0)/config.py \
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
    val_tag "$val_tag"
