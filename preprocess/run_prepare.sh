#! /bin/bash

scannet_dir="/mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2"
version=""
segment_result_dir="/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/Mask3DInst${version}"
inst_seg_dir=""
processed_data_dir="/mnt/petrelfs/share_data/huanghaifeng/data/processed/scannet/mask3d_ins_data${version}"
class_label_file="annotations/scannet/scannetv2-labels.combined.tsv"
segmentor="mask3d"

python preprocess/prepare_mask3d_data.py \
    --scannet_dir "$scannet_dir" \
    --output_dir "$processed_data_dir" \
    --segment_dir "$segment_result_dir" \
    --inst_seg_dir "$inst_seg_dir" \
    --class_label_file "$class_label_file" \
    --apply_global_alignment \
    --num_workers 16 \
    --parallel

python preprocess/prepare_scannet_attributes.py \
    --scan_dir "$processed_data_dir" \
    --segmentor "$segmentor" \
    --max_inst_num 100

python preprocess/prepare_scanrefer_annos.py \
    --segmentor "$segmentor" \
    --version "$version"

python preprocess/prepare_scan2cap_annos.py \
    --segmentor "$segmentor" \
    --version "$version"

python preprocess/prepare_objalign_annos.py \
    --segmentor "$segmentor" \
    --version "$version"

python preprocess/prepare_nr3dcaption_annos.py \
    --segmentor "$segmentor" \
    --version "$version"

python preprocess/prepare_multi3dref_annos.py \
    --segmentor "$segmentor" \
    --version "$version"

# python preprocess/prepare_scannet_caption_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version"

# python preprocess/prepare_scannet_region_caption_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version"