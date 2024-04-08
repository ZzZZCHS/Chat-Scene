which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=53173 # 53173 53179 53181 53183 53187
export MASTER_ADDR=localhost
# echo "MASTER_ADDR="$MASTER_ADDR
# export OMP_NUM_THREADS=1

stage=2
epoch=1
batch_size=32
max_txt_len=32
lr=5e-6
dp=0.1
scene_dim=256
encoder_num_layers=3
train_emb=True
train_img_proj=False
no_obj=False
add_img_token=True
add_scene_token=True
img_projector_path="annotations/img_projector_llava15.pt"
train_tag="objaverse#scannet_caption#scanrefer_caption#scannet_region_caption#nr3d_caption#scanrefer#obj_align#scanqa"
# train_tag="scannet_caption"
val_tag="scanqa#scanrefer#scanrefer_caption"
tag="${train_tag}__${val_tag}"
evaluate=False

debug=false
if [ $debug = "true" ]; then
    enable_wandb=False
    gpu_num=2
    do_save=False
else
    enable_wandb=True # !!!
    gpu_num=4
    do_save=True
fi

pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240408_024901_dp0.1_lr5e-6_sta2_ep1_objaverse#scannet_caption#scanrefer_caption#scannet_region_caption#nr3d_caption#scanrefer#obj_align#scanqa__scanqa#scanrefer#scanrefer_caption#objaverse/ckpt_00.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240407_120136_dp0.1_lr5e-6_sta2_ep1_objaverse-scannet_caption-scanrefer_caption-scannet_region_caption-nr3d_caption-scanrefer-obj_align-scanqa__scanrefer_caption/ckpt_00_4000.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240404_021034_dp0.1_lr5e-6_sta2_ep5_objalign+objcaption+grounding+caption+regioncaption+qa_scanqa/ckpt_04.pth"

# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240403_121800_dp0.1_lr5e-6_sta2_ep5_wd0.05_objalign+objcaption_linear_grounding+caption+regioncaption+qa/ckpt_00.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240401_180315_dp0.1_lr5e-6_sta2_ep5_wd0.05_objalign+objcaption_linear/ckpt_02.pth"


# pretrained_path="20240331_183348_dp0.1_lr5e-6_sta2_ep9_objalign+objcaption_grounding+caption+regioncaption+qa_addimg_layernorm1_scenedim512/ckpt_02.pth"
# pretrained_path="outputs/20240331_183517_dp0.1_lr5e-6_sta2_ep9_objalign+objcaption_grounding+caption+regioncaption+qa_addimg_layernorm1_scenedim256/ckpt_05.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240331_230405_dp0.1_lr5e-6_sta2_ep9_objalign+objcaption_grounding+caption+regioncaption+qa_addimg_layernorm1_scenedim256_allsave/ckpt_07.pth"
# pretrained_path="outputs/20240330_175418_dp0.1_lr5e-6_sta2_ep9_objalign+objcaption_addimage/ckpt_02.pth"

# pretrained_path="outputs/20240329_195835_dp0.1_lr5e-6_sta2_ep10_objalign_nonormalize_addimage/ckpt_02.pth" # obj align add img
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240325_033055_dp0.1_lr2e-6_sta2_ep10_scenealign_special_trainemb/ckpt_02.pth"    # joint training
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240325_171146_dp0.1_lr5e-6_sta2_ep10_objalign_special_trainemb/ckpt_02.pth"  # old obj align
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240325_033443_dp0.1_lr5e-6_sta2_ep10_scenealign_special_trainemb/ckpt_02.pth"
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240326_232242_dp0.1_lr5e-6_sta2_ep10_objalign_special_trianemb/ckpt_05.pth" # obj align with obj caption (with scene token)
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240327_010356_dp0.1_lr5e-6_sta2_ep10_objalign_special_trianemb/ckpt_02.pth" # obj align (with scene token)
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240327_030916_dp0.1_lr5e-6_sta2_ep10_objalign/ckpt_02.pth"  # obj align
# pretrained_path="/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240327_031656_dp0.1_lr5e-6_sta2_ep10_objalign_grounding+caption/ckpt_02.pth"


OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_dp"$dp"_lr"$lr"_sta"$stage"_ep"$epoch"_"$tag"
mkdir -p ${OUTPUT_DIR}

srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=${gpu_num} --kill-on-bad-exit \
python tasks/train.py \
    $(dirname $0)/config.py \
    output_dir "$OUTPUT_DIR" \
    model.stage "$stage" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.max_txt_len "$max_txt_len" \
    model.add_scene_token "$add_scene_token" \
    model.add_img_token "$add_img_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    model.mlp_dropout  "$dp" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    model.train_emb "$train_emb" \
    model.scene_dim "$scene_dim" \
    img_projector_path "$img_projector_path" \
    model.train_img_proj "$train_img_proj" \
    model.encoder_num_layers "$encoder_num_layers" \
    model.no_obj "$no_obj" \
    train_tag "$train_tag" \
    val_tag "$val_tag"
