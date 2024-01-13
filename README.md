# Chat-3D v2

This is an official repo for paper "Chat-3D v2: Bridging 3D Scene and Large Language Models with Object Identifiers". 
[[paper](https://arxiv.org/abs/2312.08168)]


## News

[2024.01.13] 🔥 Update training guide for grounding on ScanRefer.

[2023.12.19] Code release. The main training architecture is based on our former work [Chat-3D](https://github.com/Chat-3D/Chat-3D).

## 🔨 Preparation

- Prepare the environment:

  ```shell
  conda create -n chat-3d-v2 python=3.9.17
  conda activate chat-3d-v2
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```
  
- Download LLaMA model:
  - Currently, we choose 
Vicuna-7B as the LLM in our model, which is finetuned from LLaMA-7B.
  - Download LLaMA-7B from [hugging face](https://huggingface.co/docs/transformers/main/model_doc/llama).
  - Download [vicuna-7b-delta-v0](https://huggingface.co/lmsys/vicuna-7b-delta-v0) and process it: (`apply_delta.py` is from [huggingface](https://huggingface.co/CarperAI/stable-vicuna-13b-delta/raw/main/apply_delta.py))
  
  ```shell
  python3 model/apply_delta.py \
          --base /path/to/model_weights/llama-7b \
          --target vicuna-7b-v0 \
          --delta lmsys/vicuna-7b-delta-v0
  ```

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the location of `vicuna-7b-v0`.
  

- Annotations and extracted features:
  
  Please follow the instructions in [preprocess](preprocess/).


## 🤖 Training and Inference

  For each training/inference stage, we list the necessary modifications to configs in related files, such as [config.py](scripts/config.py) and [run.sh](scripts/run.sh). Other unmentioned configs are set to their default values.


- Training
  
  <details>
  <summary>Training for grounding task on ScanRefer</summary>
  
  - Step 1: Object-level Alignment (Attribute-aware Embedding Similarity)
    - modify [config.py](scripts/config.py):
      ```python
      train_file_s1=[
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/scanrefer_train_stage1.json"
        ],
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/scannet_train_stage1.json"
        ]
      ]
      val_file_s1=[
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_val_attributes.pt",
          "annotations/scannet_val_stage1.json"
        ]
      ]
      ```
    - modify [run.sh](scripts/run.sh):
      ```python
      stage=1
      epoch=6
      add_scene_token=False
      evaluate=False
      pretrained_path=""
      ```
    - run: `./scripts/run.sh`
    - The trained model's checkpoints are saved under `outputs/<step1_exp_name>/`.
  - Step 2: Object-level Alignment (Object-level Question-Answering)
    - modify [config.py](scripts/config.py):
      ```python
      train_file_s2=[
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/obj_align_train.json"
        ]
      ]
      val_file_s2=[
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_val_attributes.pt",
          "annotations/obj_align_val.json"
        ]
      ]
      ```
    - modify [run.sh](scripts/run.sh):
      ```python
      stage=2
      epoch=3
      add_scene_token=False
      evaluate=False
      pretrained_path="outputs/<step1_exp_name>/ckpt_05.pth"
      ```
    - run: `./scripts/run.sh`
    - The trained model's checkpoints are saved under `outputs/<step2_exp_name>/`.
  - Step 3: Scene-level Alignment
    - modify [config.py](scripts/config.py):
      ```python
      train_file_s2=[
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/scanrefer_train_stage2_objxx.json"
        ],
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/nr3d_train_stage2_objxx.json"
        ],
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/scene_align_train.json"
        ]
      ]
      val_file_s2=[
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_val_attributes.pt",
          "annotations/stage2_val400.json"
        ]
      ]
      ```
    - modify [run.sh](scripts/run.sh):
      ```python
      stage=2
      epoch=3
      add_scene_token=True
      evaluate=False
      pretrained_path="outputs/<step2_exp_name>/ckpt_00.pth"
      ```
    - run: `./scripts/run.sh`
    - The trained model's checkpoints are saved under `outputs/<step3_exp_name>/`.
  - Step 4: Fine-tuning on Grounding Task
    - modify [config.py](scripts/config.py):
      ```python
      train_file_s2=[
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/nr3d_train_stage2_grounding.json"
        ],
        [
          "annotations/scannet_uni3d_feats.pt",
          "annotations/scannet_train_attributes.pt",
          "annotations/scanrefer_train_stage2_grounding.json"
        ],
        [
          "annotations/scannet_pointgroup_uni3d_feats.pt",
          "annotations/scannet_pointgroup_train_attributes.pt",
          "annotations/scanrefer_pointgroup_train_stage2_grounding.json"
        ]
      ]
      val_file_s2=[
        [
          "annotations/scannet_pointgroup_uni3d_feats.pt",
          "annotations/scannet_pointgroup_val_attributes.pt",
          "annotations/scanrefer_pointgroup_val_stage2_grounding.json"
        ]
      ]
      ```
    - modify [run.sh](scripts/run.sh):
      ```python
      stage=2
      epoch=3
      add_scene_token=True
      evaluate=False
      pretrained_path="outputs/<step3_exp_name>/ckpt_01.pth"
      ```
    - run: `./scripts/run.sh`
    - The trained model's checkpoints are saved under `outputs/<step4_exp_name>/`. You can evaluate them following the inference guide.
    - Simultaneously, the predicted results are already saved as `outputs/<step4_exp_name>/preds_epochX_stepXXXX.json`. You can directly calculate the IoU metrics using `others/calc_scanrefer_grounding_acc.py` (see inference guide)
  </details>

- Inference
  
  <details>
  <summary>Evaluate grounding performance on ScanRefer</summary>

  - modify [config.py](scripts/config.py):
  
    ```python
    val_file_s2=[
      [
        "annotations/scannet_pointgroup_uni3d_feats.pt",
        "annotations/scannet_pointgroup_val_attributes.pt",
        "annotations/scanrefer_pointgroup_val_stage2_grounding.json"
      ]
    ]
    ```
  
  - modify [run.sh](scripts/run.sh): (We provide the pretrained checkpoint in [Google Drive](https://drive.google.com/drive/folders/19wOjXYjca6w3JRVzbbFMgwiQj6kd6MXQ?usp=drive_link))
  
    ```python
    stage=2
    add_scene_token=True
    evaluate=True
    pretrained_path="/path/to/pretrained_model.pth"
    ```
  
  - run evaluate:
  
    ```shell
    ./scripts/run.sh
    ```
    
    The predicted results (raw answers) are saved in `outputs/<exp_name>/preds_epoch-1_step0.json.json`
    
  - modify [calc_scanrefer_grounding_acc.py](others/calc_scanrefer_grounding_acc.py):
    
    ```python
    output_file="outputs/<exp_name>/preds_epoch-1_step0.json.json"
    ```
  
  - calculate IoU metrics:
  
    ```shell
    python others/calc_scanrefer_grounding_acc.py
    ```
  
  </details>
  

## 📄 Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{huang2023chat,
  title={Chat-3D v2: Bridging 3D Scene and Large Language Models with Object Identifiers},
  author={Huang, Haifeng and Wang, Zehan and Huang, Rongjie and Liu, Luping and Cheng, Xize and Zhao, Yang and Jin, Tao and Zhao, Zhou},
  journal={arXiv preprint arXiv:2312.08168},
  year={2023}
}
@article{wang2023chat,
  title={Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes},
  author={Wang, Zehan and Huang, Haifeng and Zhao, Yang and Zhang, Ziang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2308.08769},
  year={2023}
}
```

Stay tuned for our project. 🔥

If you have any questions or suggestions, feel free to drop us an email (`huanghaifeng@zju.edu.cn`, `wangzehan01@zju.edu.cn`) or open an issue.

## 😊 Acknowledgement

Thanks to the open source of the following projects:

[VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat), [LLaMA](https://github.com/facebookresearch/llama), [ULIP](https://github.com/salesforce/ULIP), [ScanRefer](https://github.com/daveredrum/ScanRefer), [ReferIt3D](https://github.com/referit3d/referit3d), [vil3dref](https://github.com/cshizhe/vil3dref), [ScanNet](https://github.com/ScanNet/ScanNet), [Uni3D](https://github.com/baaivision/Uni3D), [PointGroup](https://github.com/dvlab-research/PointGroup)