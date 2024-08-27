import json
import numpy as np
import os
# import pyviz3d.visualizer as vis
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import pandas as pd
import colorsys
import functools
from typing import List, Tuple
import random
import torch


def write_ply(verts, colors, output_file, indices=None):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0]),
                                                            int(color[1]), int(color[2])))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

# def write_ply(points, colors, save_path, text=True):
#     points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
#     vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
#     colors = [(colors[i,0], colors[i,1], colors[i,2]) for i in range(colors.shape[0])]
#     colors = np.array(colors, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
#     point = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
#     color = PlyElement.describe(colors, 'color', comments=['colors'])
#     PlyData([point, color], text=text).write(save_path)
    


@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )

def read_axis_align_matrix(file_path):
    axis_align_matrix = None
    with open(file_path, "r") as f:
        for line in f:
            line_content = line.strip()
            if 'axisAlignment' in line_content:
                axis_align_matrix = [float(x) for x in line_content.strip('axisAlignment = ').split(' ')]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                break
    return axis_align_matrix

def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    print(data.dtype.names)
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array(
            [data["red"], data["green"], data["blue"]], dtype=np.uint8
        ).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.uint32)
    return coords, feats, labels

def collect_Nr3d_gt(now_scene_id):
    nr3d_gt_path = "/mnt/petrelfs/share_data/chenyilun/share/mask3d/M3DRef-CLIP/dataset/nr3d/metadata/nr3d_test.csv"
    # "/mnt/petrelfs/share_data/chenyilun/share/mask3d/mask3d_data/langdata/referit3d/nr3d.csv"
    nr3d_gt = pd.read_csv(nr3d_gt_path).to_dict(orient='records')
    collect_nr3d = []
    for record in nr3d_gt:
        if record["scan_id"] == f"scene{now_scene_id}":
            collect_nr3d.append(record)
    return collect_nr3d


if __name__ =="__main__":
    scene_id = 'scene0046_00'
    used_ids = [0, 13, 12, 1, 17, 5, 21, 25]
    output_dir = os.path.join('ply_files', scene_id)

    # gt_coords, all_colors, _ = load_ply('ply_files/0515_00/origin.ply')
    # breakpoint()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # dataset to show: scanrefer/nr3d/
    # test_dataset = "scanrefer"
    
    scan_data = np.load(f'/mnt/hwfile/OpenRobotLab/huanghaifeng/openscene/data/scans/{scene_id}/pc_infos.npy')
    mask3d_data = torch.load(f'/mnt/hwfile/OpenRobotLab/huanghaifeng/openscene/data/mask3d_ins_data/pcd_all/{scene_id}.pth')
    
    
    gt_coords, gt_vertices, all_colors = scan_data[:, :3], scan_data[:, 3:6], scan_data[:, 6:9]
    inst_masks = mask3d_data[-1]

    instance_num = len(inst_masks)
    
    all_colors = all_colors * 255
    
    write_ply(gt_vertices, all_colors, os.path.join(output_dir, 'origin.ply'))
    instances_colors = np.vstack(get_evenly_distributed_colors(instance_num))

    # all_colors_bk = all_colors.copy()
    all_colors.fill(128)

    color_idx = -1
    for inst_id in range(instance_num):
        color_idx += 1
        mask = inst_masks[inst_id]
        all_colors[mask] = np.tile(instances_colors[color_idx], (len(mask), 1))
    
    write_ply(gt_vertices, all_colors, os.path.join(output_dir, f'{scene_id}.ply'))

    # all_colors = all_colors_bk

    # color_idx = -1
    # for i, agg_item in tqdm(enumerate(agg)):
    #     if i not in used_ids:
    #         continue
    #     obj_id = agg_item["objectId"]
    #     for item in agg:
    #         if item["objectId"] == obj_id:
    #             gt_segments = item["segments"]
    #             break
    #     idx = [(i in gt_segments) for i in gt_seg]
    #     # seg parts
    #     # parts = gt_vertices[idx,:]
    #     # gt_segs[obj_id] = parts
    #     color_idx += 1
    #     all_colors[idx, :] = np.tile(instances_colors[color_idx], (sum(idx), 1))
    
    # write_ply(gt_vertices, all_colors, os.path.join(output_dir, 'seg_with_origin.ply'))
    
    exit()
    
    if vis_pc:
        for agg_item in tqdm(agg):
            object_id = agg_item["objectId"]
            object_label = agg_item["label"]
            gt_coords = gt_segs[object_id]
            # pd_coords = coordinates[labels[:,1] == object_id]
            # print(gt_coords.shape,pd_coords.shape)
            # intersection = np.array([x for x in set(tuple(x) for x in gt_coords) & set(tuple(x) for x in pd_coords)])
            # missing_points = np.array([x for x in set(tuple(x) for x in gt_coords) - set(tuple(x) for x in pd_coords)])
            # error_points = np.array([x for x in set(tuple(x) for x in pd_coords) - set(tuple(x) for x in gt_coords)])

            color_intersection = np.array([0, 255, 0])
            # color_missing = np.array([255, 255, 0])
            # color_error = np.array([255, 0, 0])

            all_points = np.empty((0, 3))
            all_colors = np.empty((0, 3))

            # if intersection.size > 0:
            #     intersection_color = np.tile(color_intersection, (intersection.shape[0], 1))
            #     all_points = np.vstack((all_points, intersection))
            #     all_colors = np.vstack((all_colors, intersection_color))

            # if missing_points.size > 0:
            #     missing_points_color = np.tile(color_missing, (missing_points.shape[0], 1))
            #     all_points = np.vstack((all_points, missing_points))
            #     all_colors = np.vstack((all_colors, missing_points_color))

            # if error_points.size > 0:
            #     error_points_color = np.tile(color_error, (error_points.shape[0], 1))
            #     all_points = np.vstack((all_points, error_points))
            #     all_colors = np.vstack((all_colors, error_points_color))


            v.add_points(
            f"segments: {object_id} {object_label}",
            all_points,
            colors=all_colors,
            visible=False,
            point_size=25
            )


    vis_label = True
    if vis_label:
        pass

    if vis_bbox:
        if test_dataset == "scanrefer":
            scanrefer_label = json.load(open('/mnt/petrelfs/share_data/chenyilun/share/mask3d/mask3d_data/langdata/scanrefer/ScanRefer_filtered.json'))
            scanrefer_label = [i for i in scanrefer_label if i['scene_id'] == f'scene{now_scene_id}']
            assert len(box_result) == len(scanrefer_label)
            scanrefer_dict = {}
            for sl in scanrefer_label:
                scanrefer_dict[(f'scene{now_scene_id}', sl['object_id'], sl['ann_id'])] = sl
            for i, boxes in enumerate(box_result):
                ann_id = boxes['ann_id']
                object_id = boxes['object_id']
                # assert str(boxes['ann_id']) == sl['ann_id']
                assert len(boxes['aabb']) == 1
                print(i, scanrefer_dict[(f'scene{now_scene_id}', str(boxes['object_id']), str(boxes['ann_id']))]['description'])
                for box in boxes['aabb']:
                    box = np.asarray(box)
                    v.add_bounding_box(f'bbox: predict box {i}',
                        position = box.mean(axis=0),
                        size = box.max(axis=0) - box.min(axis=0),
                        color = np.array([0, 0, 255]),
                        visible = False,
                    )
                    gt_points = gt_segs[boxes['object_id']]
                    min_vals = gt_points.min(axis=0)
                    max_vals = gt_points.max(axis=0)
                    center = (max_vals + min_vals) / 2
                    dimensions = max_vals - min_vals
                    gt_bbox = np.concatenate((center, dimensions))
                    v.add_bounding_box(f'bbox: gt box {i}',
                        position = gt_bbox[:3],
                        size = gt_bbox[3:],
                        color = np.array([0, 255, 255]),
                        visible = False,
                    )
        elif test_dataset == "nr3d":
            collect_nr3d = collect_Nr3d_gt(now_scene_id)
            assert len(box_result) == len(collect_nr3d),print(f"box_result len: {len(box_result)},nr3d gt len : {len(collect_nr3d)}")
            nr3d_gt_dict = {}
            scene_ids = {}
            tmp_ann_id_count = {}
            raw_data = []

            for query in tqdm(collect_nr3d, desc="Initializing ground truths"):
                scene_id = query["scan_id"]
                scene_ids[scene_id] = True
                object_id = int(query["target_id"])

                scene_obj_key = (scene_id, object_id)
                if scene_obj_key not in tmp_ann_id_count:
                    tmp_ann_id_count[scene_obj_key] = 0
                else:
                    tmp_ann_id_count[scene_obj_key] += 1

                object_ids = [object_id]
                gt_points = gt_segs[object_id]
                min_vals = gt_points.min(axis=0)
                max_vals = gt_points.max(axis=0)
                center = (max_vals + min_vals) / 2
                dimensions = max_vals - min_vals
                gt_bbox = np.concatenate((center, dimensions))

                is_easy = query["is_easy"] == True
                is_view_dep = query["is_view_dep"] == True
                if is_easy and is_view_dep:
                    eval_type = "easy_dep"
                elif is_easy:
                    eval_type = "easy_indep"
                elif is_view_dep:
                    eval_type = "hard_dep"
                else:
                    eval_type = "hard_indep"

                nr3d_gt_dict[(scene_id, str(object_id), str(tmp_ann_id_count[scene_obj_key]))] = {
                    "aabb_bound": gt_bbox,
                    "description": query["utterance"],
                    "eval_type": eval_type
                }

            
            for i, boxes in enumerate(box_result):
                ann_id = boxes['ann_id']
                object_id = boxes['object_id']
                # assert str(boxes['ann_id']) == sl['ann_id']
                assert len(boxes['aabb']) == 1
                print(i, nr3d_gt_dict[(f'scene{now_scene_id}', str(boxes['object_id']), str(boxes['ann_id']))]['description'],
                      nr3d_gt_dict[(f'scene{now_scene_id}', str(boxes['object_id']), str(boxes['ann_id']))]["eval_type"])
                for box in boxes['aabb']:
                    box = np.asarray(box)
                    v.add_bounding_box(f'bbox: predict box {i}',
                        position = box.mean(axis=0),
                        size = box.max(axis=0) - box.min(axis=0),
                        color = np.array([0, 0, 255]),
                        visible = False,
                    )
                    gt_bbox = nr3d_gt_dict[(f'scene{now_scene_id}', str(boxes['object_id']), str(boxes['ann_id']))]["aabb_bound"]
                    v.add_bounding_box(f'bbox: gt box {i}',
                        position = gt_bbox[:3],
                        size = gt_bbox[3:],
                        color = np.array([0, 255, 255]),
                        visible = False,
                    )


            # v.add_labels(f'bbox id: {i}',
            #             [f"{(str(boxes['object_id']), str(boxes['ann_id']))}"],
            #             [box.mean(axis=0)],
            #             [np.array([255,0,0])],
            #             visible=False)

    v.save(
        f"visualizations/{pc_path.split('/')[-1].split('.')[0]}"
    )
