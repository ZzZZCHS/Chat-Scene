
pretrained_path=outputs/2023-09-07-205742_dp0.1_lr5e-5_sta3_ep3/ckpt_02.pth

CUDA_VISIBLE_DEVICES=2 python others/process_vil3dref_results.py \
        scripts/config_old.py \
        pretrained_path "$pretrained_path" \
        model.max_txt_len 20
