# Chat-Scene

We build a multi-modal large language model for 3D scene understanding, which achieves state-of-the-art performance across 3D grounding, captioning, and QA tasks.


## News

[2024.08] We released Chat-Scene, which can handle both 3D and 2D ego-centric video input for 3D scene understanding. The grounding performance on ScanRefer and Multi3DRefer has been largely improved. (Paper will be released soon.)

[2024.04] We released a refined implementation (v2.1), which achieved better performance on grounding, captioning, and QA tasks. The code is in branch [v2.1](https://github.com/Chat-3D/Chat-3D-v2/tree/v2.1).

[2023.12] We released Chat-3D v2 [[paper](https://arxiv.org/abs/2312.08168)], which proposes using object identifiers for object referencing and grounding in 3D scenes. The original code is in branch [v2.0](https://github.com/Chat-3D/Chat-3D-v2/tree/v2.0).

## 🔥 Chat-Scene vs Chat-3D v2

- Performance Comparison

  |      	| [ScanRefer](https://github.com/daveredrum/ScanRefer) 	|         	| [ScanQA](https://github.com/ATR-DBI/ScanQA) 	|        	|  [Scan2Cap](https://github.com/daveredrum/Scan2Cap) 	|            	| [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) 	|        	| [SQA3D](https://github.com/SilongYong/SQA3D) 	|
  | :----:	|:---------:	|:-------:	|:------:	|:------:	|:---------:	|:----------:	|:------------:	|:------:	|:-----:	|
  |      	|  Acc@0.25 	| Acc@0.5 	|  CIDEr 	| B-4 	| CIDEr@0.5 	| B-4@0.5 	|    F1@0.25   	| F1@0.5 	|   EM  	|
  | v2.0 	|    35.9   	|   30.4  	|  77.1  	|   7.3  	|    28.1   	|    15.5    	|       -      	|    -   	|   -   	|
  | v2.1 	|   42.5    	|  38.4   	|  87.6  	|  14.0  	|   63.9    	|    31.8    	|     45.1     	|  41.6  	| 54.7  	|
  | **Chat-Scene** | 54.6 | 49.4 |  |  |  |  |  |  |  |

  \*The results of v2.1 and Chat-Scene are evaluated on single model **without finetuning on specific tasks**.

  \*The results in the table are all on the validation set. (Test set results will be released soon.)

- Main Changes
  <details>
  <summary> Changes of v2.1 </summary>
  - LLM backbone: Vicuna v0 -> [Vicuna v1.5](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md) + LoRA finetuning

  - Training scheme: three-stage training -> one-stage joint training

  - Segmentor: [PointGroup](https://github.com/dvlab-research/PointGroup) -> [Mask3D](https://github.com/JonasSchult/Mask3D)
  
  - Code Optimization:
    - batch size: 1 -> 32
    - Simpler training and evaluating process
  </details>

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
      - `multi3dref`: [Multi3dRefer](https://github.com/3dlg-hcvc/M3DRef-CLIP) Dataset
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
  
  - Modify [run.sh](scripts/run.sh): (We provide the pretrained checkpoint in [Google Drive](https://drive.google.com/file/d/1hv-N-p9tm6nhoe6tlbZANgxYIjuVvX1n/view?usp=sharing))
  
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