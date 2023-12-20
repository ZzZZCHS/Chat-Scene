export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1
MASTER_NODE='localhost'

stage=1
epoch=3
max_txt_len=32
lr=2e-4
add_scene_token=False
evaluate=False
pretrained_path=""


OUTPUT_DIR=outputs/"$(date +"%Y-%m-%d-%T" | tr -d ':')"_dp"$dp"_lr"$lr"_sta"$stage"_ep"$epoch"
torchrun  --nnodes=${NNODE} --nproc_per_node=${NUM_GPUS} \
    --rdzv_endpoint=${MASTER_NODE}:${MASTER_PORT} \
    --rdzv_backend=c10d \
    tasks/train.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR} \
    model.stage "$stage" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.max_txt_len "$max_txt_len" \
    model.add_scene_token "$add_scene_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate"
