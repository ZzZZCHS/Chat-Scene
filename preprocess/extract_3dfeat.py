import argparse
from collections import OrderedDict

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from data.dataset_3d import *

import models.ULIP_models as models
from utils import utils
import glob
from tqdm import tqdm
import os



def get_args_parser():
    parser = argparse.ArgumentParser(description='ULIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=1024, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='ULIP_PointBERT', type=str)
    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate_3d', default=True, type=bool, help='eval 3d only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    parser.add_argument('--test_ckpt_addr', default='pointbert_ULIP-2.pt', help='the ckpt to test 3d zero shot')
    parser.add_argument('--pcds_dir', default='datasets/referit3d/pcd_by_instance', help='point clouds dir')
    parser.add_argument('--output_feat_dir', default='datasets/referit3d/pcd_feats_ulip2', help='output dir for extracted features')
    return parser


def load_and_transform_point_cloud_data(point_paths, device):
    point_outputs = []

    for point_path in point_paths:
        if type(point_path) != str:
            point_path = point_path.name
        point = torch.load(point_path).to(device)
        point_outputs.append(point)

    return torch.stack(point_outputs, dim=0)


def main(args):
    utils.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    try:
        model = getattr(models, old_args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    except:
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))

    pcds_dir = args.pcds_dir
    feat_dir = args.output_feat_dir
    if not os.path.exists(feat_dir):
        os.mkdir(feat_dir)

    for scene_dir in tqdm(sorted(glob.glob(os.path.join(pcds_dir, "scene*")))):
        scene_id = scene_dir.split("/")[-1]
        for obj_file in sorted(glob.glob(os.path.join(scene_dir, "*.pt"))):
            file_name = obj_file.split("/")[-1]
            point = load_and_transform_point_cloud_data([obj_file], device="cuda")
            pcd_feat = model.encode_pc(point)
            # pcd_feat = pcd_feat / pcd_feat.norm(dim=-1, keepdim=True)

            feat_scene_dir = os.path.join(feat_dir, scene_id)
            if not os.path.exists(feat_scene_dir):
                os.mkdir(feat_scene_dir)
            torch.save(pcd_feat.squeeze().detach().cpu(), os.path.join(feat_scene_dir, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
