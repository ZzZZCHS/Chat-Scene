## Skip the data preparation

- We've provided all the prepared data needed for running the code in [Google Drive](https://drive.google.com/drive/folders/1feFAsDmeDmx_PVLcuQm8aZFkNuUGz4Gu?usp=sharing). Download and put them into `annotations/` directory. Then you can run and test the code.

## Prepare data

- Follow the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to download the ScanNet dataset.

- Use pretrained 3D instance segmentor:
    - Follow [Mask3D](https://github.com/JonasSchult/Mask3D) for instance segmentation. We use the [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_val.ckpt) pretrained on ScanNet200.
    - The whole prediceted results (especially the masks) for train/val set are too large to share (~40G). We share the post-processed [results](https://drive.google.com/file/d/1eIdmuEBeM4OxJ9dmucZvHK8_4mOhcbEV/view?usp=sharing):
        - Unzip the `mask3d_inst_seg.tar.gz`.
        - Each file under `mask3d_inst_seg` consists of the predicted results of one scene. It records a list of segmented instances with their labels and segmented indices.

- Process data:
    - If you use pretrained Mask3D to do instance segmentation, please set the `segment_result_dir` in [run_prepare.sh](run_prepare.sh) to the output dir of Mask3D.
    - Otherwise, if you directly use the downloaded `mask3d_inst_seg`, please set the `segment_result_dir` to None and set the `inst_seg_dir` to the path to `mask3d_inst_seg`.
    - Run: `bash preprocess/run_prepare.sh`

- Use pretrained 3D encoder:
    - Follow [Uni3D](https://github.com/baaivision/Uni3D?tab=readme-ov-file) to extract 3D features of each instance. We use the pretrained model [uni3d-g](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-g/model.pt).
    - We also provide the hacked code for extracting features in the forked [repo](https://github.com/ZzZZCHS/Uni3D). Set `data_dir` [here](https://github.com/ZzZZCHS/Uni3D/blob/main/main.py#L620) to the path to `${processed_data_dir}/pcd_all` (`processed_data_dir` is an intermediate dir set in `run_prepare.sh`). Prepare the environment, and run `bash scripts/inference.sh`.
