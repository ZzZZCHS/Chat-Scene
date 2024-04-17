#! /bin/bash

scannet_dir="/mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2"
segment_result_dir="/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/Mask3DInst_v2"
processed_data_dir="/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/mask3d_ins_data_v2"
class_label_file="annotations/scannet/scannetv2-labels.combined.tsv"
segmentor="mask3d"

# python preprocess/prepare_mask3d_data_all.py \
#     --scannet_dir "$scannet_dir" \
#     --output_dir "$processed_data_dir" \
#     --segment_dir "$segment_result_dir" \
#     --class_label_file "$class_label_file" \
#     --apply_global_alignment \
#     --num_workers 16 \
#     --parallel

python preprocess/prepare_scannet_attributes_all.py \
    --scan_dir "$processed_data_dir" \
    --segmentor "mask3d" \
    --max_inst_num 100