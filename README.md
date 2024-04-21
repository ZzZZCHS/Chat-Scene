# Chat-3D v2

This is an official repo for paper "Chat-3D v2: Bridging 3D Scene and Large Language Models with Object Identifiers". 
[[paper](https://arxiv.org/abs/2312.08168)]


## News

[2024.04] 🔥 A refined implementation of Chat-3D v2 is released. The old version v2.0 has been archived in branch v2_0. This main branch is now for the new version (v2.1).

[2024.01] Update training guide for grounding on ScanRefer.

[2023.12] Code release. The main training architecture is based on our former work [Chat-3D](https://github.com/Chat-3D/Chat-3D).

## v2.1 vs v2.0

- <details>
  <summary>Performance comparison</summary>

  <small>

  |      	| [ScanRefer]((https://github.com/daveredrum/ScanRefer)) 	|         	| [ScanQA](https://github.com/ATR-DBI/ScanQA) 	|        	|  [Scan2Cap](https://github.com/daveredrum/Scan2Cap) 	|            	| [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) 	|        	| [SQA3D](https://github.com/SilongYong/SQA3D) 	|
  |:----:	|:---------:	|:-------:	|:------:	|:------:	|:---------:	|:----------:	|:------------:	|:------:	|:-----:	|
  |      	|  Acc@0.25 	| Acc@0.5 	|  CIDEr 	| B-4 	| CIDEr@0.5 	| B-4@0.5 	|    F1@0.25   	| F1@0.5 	|   EM  	|
  | v2.0 	|    35.9   	|   30.4  	|  77.1  	|   7.3  	|    28.1   	|    15.5    	|       -      	|    -   	|   -   	|
  | **v2.1** 	|   **43.2**    	|  **39.3**   	|  **88.0**  	|  **13.8**  	|   **62.0**    	|    **31.6**    	|     **44.9**     	|  **41.4**  	| **54.4**  	|

  <sub> All results of v2.1 are evaluated on the same model without finetuning on specific tasks.</sub>

  </small>

- <details>
  <summary>Main changes</summary>

  - LLM backbone: Vicuna v0 -> [Vicuna v1.5](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) + LoRA finetuning

  - Training scheme: three-stage training -> one-stage joint training

  - Segmentor: [PointGroup](https://github.com/dvlab-research/PointGroup) -> [Mask3D](https://github.com/JonasSchult/Mask3D)
  
  - Code Optimization:
    - batch size: 1 -> 32
    - Simpler training and evaluating process

## 🔨 Preparation

- Prepare the environment:
  
  (Different from v2.0)
  ```shell
  conda create -n chat-3d-v2 python=3.9.17
  conda activate chat-3d-v2
  conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```
  
- Download LLM backbone:
  -  We use Vicuna-7B v1.5 in our experiments, which can be downloaded from [Hugging Face](https://huggingface.co/lmsys/vicuna-7b-v1.5).

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the location of `vicuna-7b-v1.5`.
  

- Annotations and extracted features:
  
  Please follow the instructions in [preprocess](preprocess/).


## 🤖 Training and Inference

- Training
  - Modify [run.sh](scripts/run.sh):
    ```python
    train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
    val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
    evaluate=False
    ```

    <details>
    <summary> Explanation of "train_tag" and "val_tag" </summary>

    - Use `#` to seperate different datasets

    - Datasets:
      - `scanrefer`: [ScanRefer](https://github.com/daveredrum/ScanRefer) Dataset
      - `scan2cap`: [Scan2Cap](https://github.com/daveredrum/Scan2Cap) Dataset
      - `scanqa`: [ScanQA](https://github.com/ATR-DBI/ScanQA) Dataset
      - `sqa3d`: [SQA3D](https://github.com/SilongYong/SQA3D) Dataset
      - `multi3dref`: [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) Dataset_
      - `nr3d_caption`: A captioning dataset originated from [Nr3D](https://github.com/referit3d/referit3d).
      - `obj_align`: A dataset originated from ScanRefer to align the object identifiers with object tokens.
    
    - You can try different combination of training datasets or add costumized datasets.

    </details>
  - Run: `bash scripts/run.sh`

  - Brief training info:

    | Batch Size | GPU | VRAM Usage per GPU | Training Time | ckpt |
    | :---: | :---: | :---: | :---: | :---: |
    | 32 | 4 * A100 | ~ 70 GB | ~ 8 hours | [Google Drive](https://drive.google.com/file/d/1hv-N-p9tm6nhoe6tlbZANgxYIjuVvX1n/view?usp=sharing) |
    | 1 | 1 * A100 | ~ 28 GB | ~ 3 days | - |


- Inference
  
  - Modify [run.sh](scripts/run.sh): (We provide the pretrained checkpoint in [Google Drive](https://drive.google.com/drive/folders/19wOjXYjca6w3JRVzbbFMgwiQj6kd6MXQ?usp=drive_link))
  
    ```python
    val_tag="multi3dref#scanqa#scanrefer#sqa3d#scan2cap"
    evaluate=False
    pretrained_path="/path/to/pretrained_model.pth"
    ```
  
  - Run: `bash scripts/run.sh`
  

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

LLMs:
[LLaMA](https://github.com/facebookresearch/llama), 
[Vicuna](https://github.com/lm-sys/FastChat)

3D Datasets:
[ScanNet](https://github.com/ScanNet/ScanNet), 
[ScanRefer](https://github.com/daveredrum/ScanRefer), 
[ReferIt3D](https://github.com/referit3d/referit3d), 
[Scan2Cap](https://github.com/daveredrum/Scan2Cap), 
[ScanQA](https://github.com/ATR-DBI/ScanQA), 
[SQA3D](https://github.com/SilongYong/SQA3D), 
[Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP)

3D Segmentors:
[PointGroup](https://github.com/dvlab-research/PointGroup), 
[Mask3D](https://github.com/JonasSchult/Mask3D)

3D Encoders:
[ULIP](https://github.com/salesforce/ULIP), 
[Uni3D](https://github.com/baaivision/Uni3D)

Multi-modal LLMs:
[VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat), 
[LEO](https://github.com/embodied-generalist/embodied-generalist)

3D Expert Models:
[vil3dref](https://github.com/cshizhe/vil3dref)