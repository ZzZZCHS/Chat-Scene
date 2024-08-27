#! /bin/bash

# scannet_dir="/mnt/petrelfs/share_data/maoxiaohan/ScanNet_v2"
scannet_dir="/mnt/hwfile/OpenRobotLab/huanghaifeng/data/scannet"
version=""
segment_result_dir="/mnt/hwfile/OpenRobotLab/huanghaifeng/data/processed/scannet/Mask3DInst"
# segment_result_dir="/mnt/petrelfs/share_data/chenyilun/haifeng/Mask3DInst"
inst_seg_dir=""
class_label_file="annotations/scannet/scannetv2-labels.combined.tsv"
max_obj_num=100

train_iou_thres=0.5

processed_data_dir="/mnt/hwfile/OpenRobotLab/huanghaifeng/data/processed/scannet/mask3d_ins_data"
# processed_data_dir="/mnt/petrelfs/share_data/chenyilun/haifeng/mask3d_ins_data"
segmentor="mask3d"
# processed_data_dir="/mnt/petrelfs/share_data/chenyilun/share/mask3d/proposals"
# segmentor="clasp"
# segmentor="deva"

python preprocess/prepare_mask3d_data.py \
    --scannet_dir "$scannet_dir" \
    --output_dir "$processed_data_dir" \
    --segment_dir "$segment_result_dir" \
    --inst_seg_dir "$inst_seg_dir" \
    --class_label_file "$class_label_file" \
    --apply_global_alignment \
    --num_workers 16 \
    --parallel

python preprocess/prepare_scannet_mask3d_attributes.py \
    --scan_dir "$processed_data_dir" \
    --segmentor "$segmentor" \
    --max_inst_num "$max_obj_num" \
    --version "$version"

# python preprocess/prepare_scannet_attributes.py \
#     --scannet_dir "$scannet_dir"

# python preprocess/prepare_scannet_attributes_clasp.py \
#     --scan_dir "$processed_data_dir" \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --max_inst_num "$max_obj_num"

python preprocess/prepare_scanrefer_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres" \
    --max_obj_num "$max_obj_num"

python preprocess/prepare_scan2cap_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres" \
    --max_obj_num "$max_obj_num"

python preprocess/prepare_objalign_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres"

python preprocess/prepare_nr3dcaption_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres"

python preprocess/prepare_multi3dref_annos.py \
    --segmentor "$segmentor" \
    --version "$version" \
    --train_iou_thres "$train_iou_thres"

python preprocess/prepare_scanqa_annos.py

python preprocess/prepare_sqa3d_annos.py

# python preprocess/prepare_nr3d_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --max_obj_num "$max_obj_num"

# python preprocess/prepare_sr3d_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --max_obj_num "$max_obj_num"



# python preprocess/prepare_scannet_caption_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres"

# python preprocess/prepare_scannet_region_caption_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres"

# python preprocess/prepare_scanrefer_location_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --max_obj_num "$max_obj_num"

# python preprocess/prepare_multi3dref_location_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres"

# python preprocess/prepare_scan2cap_location_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --max_obj_num "$max_obj_num"