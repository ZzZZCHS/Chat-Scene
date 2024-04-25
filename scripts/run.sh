which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((53000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

epoch=3
batch_size=32
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=True
add_scene_token=False
no_obj=False

train_tag="scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#scannet_caption#scannet_region_caption#nr3d_caption"
# train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"

evaluate=False
debug=false
if [ $debug = "true" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=True
    gpu_num=4
    do_save=True
    other_info="pos+video"
fi

tag="${train_tag}__${val_tag}__${other_info}"

pretrained_path=""

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
    val_tag "$val_tag" \
    model.no_obj "$no_obj"
