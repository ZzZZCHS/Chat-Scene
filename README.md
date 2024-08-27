# Chat-Scene

We build a multi-modal large language model for 3D scene understanding, excelling in tasks such as 3D grounding, captioning, and question answering.


## News

**[2024.08]** 🔥 We released Chat-Scene, capable of processing both 3D point clouds and 2D multi-view images for improved 3D scene understanding, leading to significant advancements in grounding and captioning performance. (Paper to be released soon.)

**[2024.04]** We released a refined implementation (v2.1), which achieved better performance on grounding, captioning, and QA tasks. The code is available in branch [v2.1](https://github.com/Chat-3D/Chat-3D-v2/tree/v2.1).

**[2023.12]** We released Chat-3D v2 [[paper](https://arxiv.org/abs/2312.08168)], introducing object identifiers for enhanced object referencing and grounding in 3D scenes. The original code is available in branch [v2.0](https://github.com/Chat-3D/Chat-3D-v2/tree/v2.0).

## 🔥 Chat-Scene vs Chat-3D v2

- Performance Comparison

  |      	| [ScanRefer](https://github.com/daveredrum/ScanRefer) 	|         	| [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) 	|        	|  [Scan2Cap](https://github.com/daveredrum/Scan2Cap) 	|            	| [ScanQA](https://github.com/ATR-DBI/ScanQA) 	|        	| [SQA3D](https://github.com/SilongYong/SQA3D) 	|
  | :----:	|:---------:	|:-------:	|:------:	|:------:	|:---------:	|:----------:	|:------------:	|:------:	|:-----:	|
  |      	|  Acc@0.25 	| Acc@0.5 	|    F1@0.25   	| F1@0.5 	| CIDEr@0.5 	|   B-4@0.5 	|  CIDEr 	| B-4 	|   EM  	|
  | v2.0 	|    35.9   	|   30.4  	|       -      	|    -   	|    28.1   	|    15.5    	|  77.1  	|   7.3  	|   -   	|
  | v2.1 	|   42.5    	|  38.4   	|     45.1     	|  41.6  	|   63.9    	|    31.8    	|  87.6  	|  14.0  	| **54.7**  	|
  | **Chat-Scene** | **55.5** | **49.6** | **57.1** | **52.4** | **77.1** | **36.3** | **87.7** | **14.3** | 54.6 |

  <small>\*The v2.1 and Chat-Scene results are based on single models **without task-specific finetuning**.

  \*All results are from the validation set. (Test set results will be released soon.)</small>

- Main Changes
  <details>
  <summary> New features in Chat-Scene </summary>

  - Introduce a 2D token for each object, with 2D representations extracted from multi-view images using [DINOv2](https://github.com/facebookresearch/dinov2).

  - Enable processing of 2D ego-centric video using a tracking-based detector when 3D input is unavailable.

  </details>

  <details>
  <summary> New features in v2.1 (Chat-Scene is built upon v2.1) </summary>

  - LLM backbone: Vicuna v0 -> [Vicuna v1.5](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) + LoRA.

  - Training scheme: three-stage training -> one-stage joint training.

  - Detector: [PointGroup](https://github.com/dvlab-research/PointGroup) -> [Mask3D](https://github.com/JonasSchult/Mask3D).
  
  - Code Optimization:
    - batch size: 1 -> 32.
    - Simplified training and evaluation processes.
  </details>

## 🔨 Preparation

- Prepare the environment:
  
  ```shell
  conda create -n chat-scene python=3.9.17
  conda activate chat-scene
  conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```
  
- Download LLM backbone:
  -  We use Vicuna-7B v1.5 in our experiments, which can be downloaded from [Hugging Face](https://huggingface.co/lmsys/vicuna-7b-v1.5).

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the path of `vicuna-7b-v1.5`.
  

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
      - `multi3dref`: [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) Dataset
      - `nr3d_caption`: A captioning dataset originated from [Nr3D](https://github.com/referit3d/referit3d).
      - `obj_align`: A dataset originated from ScanRefer to align the object identifiers with object tokens.
    
    - You can try different combination of training datasets or add costumized datasets.

    </details>
  - Run: `bash scripts/run.sh`


- Inference
  
  - Modify [run.sh](scripts/run.sh): (We provide the pretrained checkpoint in [Google Drive](https://drive.google.com/file/d/1Ziz7Be9l6MEbn3Qmlyr9gv42C0iJQgAn/view?usp=sharing))
  
    ```python
    val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
    evaluate=True
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

(Multi-modal) LLMs:
[LLaMA](https://github.com/facebookresearch/llama), 
[Vicuna](https://github.com/lm-sys/FastChat),
[VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat), 
[LEO](https://github.com/embodied-generalist/embodied-generalist)

3D Datasets:
[ScanNet](https://github.com/ScanNet/ScanNet), 
[ScanRefer](https://github.com/daveredrum/ScanRefer), 
[ReferIt3D](https://github.com/referit3d/referit3d), 
[Scan2Cap](https://github.com/daveredrum/Scan2Cap), 
[ScanQA](https://github.com/ATR-DBI/ScanQA), 
[SQA3D](https://github.com/SilongYong/SQA3D), 
[Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP)

Detectors:
[PointGroup](https://github.com/dvlab-research/PointGroup), 
[Mask3D](https://github.com/JonasSchult/Mask3D),
[DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

Representations:
[ULIP](https://github.com/salesforce/ULIP), 
[Uni3D](https://github.com/baaivision/Uni3D),
[DINOv2](https://github.com/facebookresearch/dinov2)

3D Models:
[vil3dref](https://github.com/cshizhe/vil3dref),
[OpenScene](https://github.com/pengsongyou/openscene)

