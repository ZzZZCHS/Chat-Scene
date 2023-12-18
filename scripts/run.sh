export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=2
MASTER_NODE='localhost'

stage=2  # change for different stage (1, 2, 3)
epoch=3

other_info="bs1_cosine_objalign_scenealign_scanqa" # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if [ $stage -eq 3 ]; then
  max_txt_len=512
else
  max_txt_len=32   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
fi

#relation_lrs=("2e-4" "1e-4" "5e-5")
lrs=("2e-4")
num_layers=("1")
dps=("0.1")
#wds=("0.02")
objscales=("200")
scenescales=("50")
#gradscales=("0.1" "0.5" "0.01" "0.05")


#pretrained_path="outputs/2023-10-18-205911_dp0.1_lr5e-3_sta1_ep3/ckpt_02.pth"  # old stage1
#pretrained_path="outputs/2023-10-30-230154_dp0.1_lr5e-3_sta1_ep3_uni3d_scanrefer/ckpt_08.pth"  # new stage1
#pretrained_path="outputs/2023-11-03-181947_dp0.1_lr5e-3_sta1_ep10_wd0.02_bs1_stage1/ckpt_06.pth"
#pretrained_path="outputs/2023-11-04-163550_dp0.1_lr5e-3_sta1_ep6_objscale100_bs1_objscale/ckpt_05.pth"
#pretrained_path="outputs/2023-11-06-003247_dp0.1_lr1e-3_sta1_ep6_objscale200_scenescale50_bs1_obj_align_scanrefer/ckpt_05.pth" # cosine
#pretrained_path="outputs/2023-11-08-101707_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign/ckpt_00.pth" # cosine+obj_align
#pretrained_path="outputs/2023-11-22-140941_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign/ckpt_00.pth"
#pretrained_path="outputs/2023-11-19-164049_dp0.1_lr1e-4_sta2_ep3_objscale50_scenescale50_bs1_cosine_objalign_scenealign/ckpt_00.pth"  # new cosine+obj_align
#pretrained_path="outputs/2023-11-15-150429_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign_mean/ckpt_00.pth" # mean: cosine+objalign
#pretrained_path="outputs/2023-11-07-134152_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scanrefer/ckpt_00.pth"  # new stage2
#pretrained_path="outputs/2023-11-08-154308_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scanrefer/ckpt_01.pth"
#pretrained_path="outputs/2023-11-11-002134_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scanrefer+nr3d/ckpt_00.pth"  # scene-align scanrefer+nr3d
#pretrained_path="outputs/2023-11-09-013434_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scenealign/ckpt_01.pth"  # scene-align scanrefer+nr3d+scene-aware-qa
#pretrained_path="outputs/2023-11-16-214919_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign/ckpt_01.pth"  # new scene-align scanrefer+nr3d+scene-aware-qa
#pretrained_path="outputs/2023-11-20-000306_dp0.1_lr1e-4_sta2_ep3_objscale200_scenescale10_bs1_cosine_objalign_scenealign/ckpt_01.pth"  # new scene-align scanrefer+nr3d+sr3d+scene-aware-qa
#pretrained_path="outputs/2023-11-15-165912_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign_mean/ckpt_00.pth"  # mean: scene-align
#pretrained_path="outputs/2023-11-09-144038_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scenealign/ckpt_00.pth"  # nr3d-grounding
#pretrained_path="outputs/2023-11-10-100620_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scenealign_nr3dgrounding/ckpt_02.pth"
#pretrained_path="outputs/2023-11-06-161413_dp0.1_lr1e-3_sta2_ep3_objscale200_scenescale50_bs1_objalign_scanrefer/ckpt_00.pth"  # scanqa
#pretrained_path=
#pretrained_path="outputs/2023-11-13-210816_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_objalign_scenealign_nr3dgrounding/ckpt_02.pth"
#pretrained_path="outputs/2023-11-17-133632_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign/ckpt_02.pth"
#pretrained_path="outputs/2023-11-24-192539_dp0.1_lr2e-4_sta2_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign_scanqa/ckpt_01.pth"

#pretrained_path="outputs/2023-10-16-095101_dp0.1_lr5e-4_sta2_ep3/ckpt_00.pth"
pretrained_path="outputs/2023-12-06-204120_dp0.1_lr2e-4_sta1_ep3_objscale200_scenescale50_bs1_cosine_objalign_scenealign_scanqa/ckpt_02.pth"


for dp in "${dps[@]}"
do
#  for relation_lr in "${relation_lrs[@]}"; do
  for lr in "${lrs[@]}"
  do
  for objscale in "${objscales[@]}"
  do
  for scenescale in "${scenescales[@]}"
  do
#  for gradscale in "${gradscales[@]}"
#  do
    OUTPUT_DIR=outputs/"$(date +"%Y-%m-%d-%T" | tr -d ':')"_dp"$dp"_lr"$lr"_sta"$stage"_ep"$epoch"_objscale"$objscale"_scenescale"$scenescale"_"$other_info"
    torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
        --rdzv_endpoint=${MASTER_NODE}:${MASTER_PORT} \
        --rdzv_backend=c10d \
        tasks/train.py \
        $(dirname $0)/config.py \
        output_dir ${OUTPUT_DIR} \
        model.stage "$stage" \
        scheduler.epochs "$epoch" \
        optimizer.lr "$lr" \
        model.mlp_dropout "$dp" \
        model.max_txt_len 32 \
        model.add_scene_token True \
        wandb.enable False \
        model.debug False \
        pretrained_path "$pretrained_path" \
        model.obj_norm_scale "$objscale" \
        model.scene_norm_scale "$scenescale" \
        do_save False \
#        evaluate True \
#        model.grad_scale "$gradscale" \
#        num_workers 0 \
#        s2_batch_size 2 \
#        optimizer.relation_lr "$relation_lr" \
#  done
  done
  done
  done
#  done
done
