import argparse
import copy
import math
import multiprocessing
import os
import pickle
import random
import zlib
from collections import defaultdict
from multiprocessing import Process
from random import choice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import utils
from utils import get_name, get_file_name_int, get_angle, logging, rotate, round_value, get_pad_vector, get_dis, get_subdivide_polygons
from utils import get_points_remove_repeated, get_one_subdivide_polygon, get_dis_point_2_polygons, larger, equal, assert_
from utils import get_neighbour_points, get_subdivide_points, get_unit_vector, get_dis_point_2_points

TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3

def nuscene_get_instance(args: utils.Args, instance_dir, uncertainty=False, centerline=False, boundaries=True):
    with open(instance_dir, 'rb') as file:
        data = pickle.load(file)
    dt = data['dt']
    agent_type = data['agent_type']
    agent_hist = data['agent_hist']
    ego_pos = data['ego_pos'][0]
    ego_heading = data['ego_heading'][0]
    ego_hist = data['ego_hist']
    ego_fut = data['ego_fut']
    vec_map = data['predicted_map']
    x_i, y_i, vx_i, vy_i, heading_i = (0, 1, 2, 3, 4)

    mapping = {}
    vectors = []
    polyline_spans = []
    agents = []
    polygons = []
    labels = []
    gt_trajectory_global_coordinates = []

    if True:

        cent_x = ego_pos[0]
        cent_y = ego_pos[1]
        angle = -ego_heading + math.radians(90)
        normalizer = utils.Normalizer(cent_x, cent_y, angle)

        assert len(ego_hist) == 20
        assert len(ego_fut) == 30
        for timestep, pos in enumerate(ego_fut):
            labels.append(normalizer((pos[0], pos[1])))
            gt_trajectory_global_coordinates.append((pos[0], pos[1]))

        assert cent_x is not None
        mapping.update(dict(
            cent_x=cent_x,
            cent_y=cent_y,
            angle=angle,
            normalizer=normalizer,
        ))
    
    origin = torch.tensor(data['ego_hist'][-1], dtype=torch.float)
    av_heading_vector = origin - torch.tensor(data['ego_hist'][-2], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]])

    agent_indices_original = np.where(data['agent_type'] == 1)[0]
    keep_indices = []
    for index in agent_indices_original:
        n_point = torch.matmul((torch.tensor(data['agent_hist'][index, -1, :2]) - origin).unsqueeze(0), rotate_mat)
        # n_point = torch.matmul(torch.tensor([data['agent_hist'][index, -1, :2]]) - origin, rotate_mat)
        if -30 < n_point[0][0].item() < 30 and -15 < n_point[0][1].item() < 15:
            keep_indices.append(index)

    if len(keep_indices) == 0:
        history = expand_coordinates_with_motion_data(ego_hist)
        history = np.expand_dims(history, axis=0)
        agent_type = np.append(data['agent_type'], 1)
    else:
        agent_indices = np.array(keep_indices)
        history = agent_hist[agent_indices]

    for agent_id, hist in enumerate(history):
        start = len(vectors)
        agent = []
        timestep_to_state = {}
        for timestep, state in enumerate(hist):
            timestep_to_state[timestep] = (state, agent_id)
            agent.append(normalizer([state[x_i], state[y_i]]))

        i = 0
        while i < 20:
            if i in timestep_to_state:
                state, agent_id = timestep_to_state[i]

                vector = np.zeros(args.hidden_size)

                vector[0], vector[1] = normalizer((state[x_i], state[y_i]))
                vector[2], vector[3] = rotate(state[vx_i], state[vy_i], angle)  # velocity
                vector[4] = state[heading_i] + angle  # heading
                vector[5] = i  # timestep
                
                try:
                    vector[10 + agent_type[agent_id]] = 1
                except:
                    breakpoint()

                offset = 20
                for j in range(8):
                    if (i + j) in timestep_to_state:
                        t = timestep_to_state[i + j][0]
                        vector[offset + j * 3], vector[offset + j * 3 + 1] = normalizer((t[x_i], t[y_i]))
                        vector[offset + j * 3 + 2] = 1

                i += 4
                vectors.append(vector[::-1])
            else:
                i += 1

        end = len(vectors)
        if end > start:
            agents.append(np.array(agent))
            polyline_spans.append([start, end])

    map_start_polyline_idx = len(polyline_spans)
    score_thres = 0.0
    if args.use_map:
        if centerline:
            for idx, centerline in enumerate(vec_map['centerlines']):
                if vec_map['centerline_scores'][idx] < score_thres:
                    continue
                start = len(vectors)
                polyline = []
                for point in centerline:
                    polyline.append(normalizer([point[0], point[1]]))
                polyline = np.array(polyline)
                polygons.append(polyline)
                if uncertainty:
                    betas = vec_map['centerline_betas'][idx]
                for i in range(len(polyline)):
                    vector = np.zeros(args.hidden_size)
                    # if uncertainty:
                    offset = 10
                    if uncertainty:
                        for j in range(5):
                            if i + j < len(polyline):
                                vector[offset + j * 4] = polyline[i + j, 0]
                                vector[offset + j * 4 + 1] = polyline[i + j, 1]
                                var_x = 2 * betas[i + j][0] ** 2
                                var_y = 2 * betas[i + j][1] ** 2
                                new_var_x = var_x * np.cos(angle) ** 2 + var_y * np.sin(angle) ** 2
                                new_var_y = var_x * np.sin(angle) ** 2 + var_y * np.cos(angle) ** 2
                                # print(np.sqrt(new_var_x), np.sqrt(new_var_y))
                                vector[offset + j * 4 + 2] = np.sqrt(new_var_x)
                                vector[offset + j * 4 + 3] = np.sqrt(new_var_y)
                    else:
                        for j in range(5):
                            if i + j < len(polyline):
                                vector[offset + j * 2] = polyline[i + j, 0]
                                vector[offset + j * 2 + 1] = polyline[i + j, 1]
                    vectors.append(vector)
                end = len(vectors)
                if end > start:
                    polyline_spans.append([start, end])

        if boundaries:
            for idx, bdy in enumerate(vec_map['boundary']):
                if idx < len(vec_map['boundary_scores']):

                    if vec_map['boundary_scores'][idx] < score_thres:
                        continue
                if len(bdy) <= 1:
                    continue
                start = len(vectors)
                polyline = []
                for point in bdy:
                    polyline.append(normalizer([point[0], point[1]]))
                polyline = np.array(polyline)
                polygons.append(polyline)
                if uncertainty:
                    betas = vec_map['boundary_betas'][idx]
                for i in range(len(polyline)):
                    vector = np.zeros(args.hidden_size)
                    offset = 10
                    if uncertainty:
                        for j in range(5):
                            if i + j < len(polyline):
                                vector[offset + j * 4] = polyline[i + j, 0]
                                vector[offset + j * 4 + 1] = polyline[i + j, 1]
                                var_x = 2 * betas[i + j][0] ** 2
                                var_y = 2 * betas[i + j][1] ** 2
                                new_var_x = var_x * np.cos(angle) ** 2 + var_y * np.sin(angle) ** 2
                                new_var_y = var_x * np.sin(angle) ** 2 + var_y * np.cos(angle) ** 2
                                # print(np.sqrt(new_var_x), np.sqrt(new_var_y))
                                vector[offset + j * 4 + 2] = np.sqrt(new_var_x)
                                vector[offset + j * 4 + 3] = np.sqrt(new_var_y)
                    else:
                        for j in range(5):
                            if i + j < len(polyline):
                                vector[offset + j * 2] = polyline[i + j, 0]
                                vector[offset + j * 2 + 1] = polyline[i + j, 1]
                    vectors.append(vector)
                end = len(vectors)
                if end > start:
                    polyline_spans.append([start, end])
        if len(polygons) == 0:
            for idx, divider in enumerate(vec_map['divider']):
                if len(divider) <= 1:
                    continue
                start = len(vectors)
                polyline = []
                for point in divider:
                    polyline.append(normalizer([point[0], point[1]]))
                polyline = np.array(polyline)
                polygons.append(polyline)
                if uncertainty:
                    betas = vec_map['divider_betas'][idx]
                for i in range(len(polyline)):
                    vector = np.zeros(args.hidden_size)
                    offset = 10
                    if uncertainty:
                        for j in range(5):
                            if i + j < len(polyline):
                                vector[offset + j * 4] = polyline[i + j, 0]
                                vector[offset + j * 4 + 1] = polyline[i + j, 1]
                                var_x = 2 * betas[i + j][0] ** 2
                                var_y = 2 * betas[i + j][1] ** 2
                                new_var_x = var_x * np.cos(angle) ** 2 + var_y * np.sin(angle) ** 2
                                new_var_y = var_x * np.sin(angle) ** 2 + var_y * np.cos(angle) ** 2
                                # print(np.sqrt(new_var_x), np.sqrt(new_var_y))
                                vector[offset + j * 4 + 2] = np.sqrt(new_var_x)
                                vector[offset + j * 4 + 3] = np.sqrt(new_var_y)
                    else:
                        for j in range(5):
                            if i + j < len(polyline):
                                vector[offset + j * 2] = polyline[i + j, 0]
                                vector[offset + j * 2 + 1] = polyline[i + j, 1]
                    vectors.append(vector)
                end = len(vectors)
                if end > start:
                    polyline_spans.append([start, end])

        assert polygons
        if 'goals_2D' in args.other_params:
            points = []
            visit = {}

            def get_hash(point):
                return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

            for index_polygon, polygon in enumerate(polygons):
                for i, point in enumerate(polygon):
                    hash = get_hash(point)
                    if hash not in visit:
                        visit[hash] = True
                        points.append(point)

                # Subdivide lanes to get more fine-grained 2][ goals.
                # if 'subdivide' in args.other_params:
                #     subdivide_points = get_subdivide_points(polygon)
                #     points.extend(subdivide_points)
            # print(len(points))
            mapping['goals_2D'] = np.array(points)

        pass
    if 'goals_2D' in args.other_params:
        point_label = np.array(labels[-1])
        mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))

        if 'lane_scoring' in args.other_params:
            stage_one_label = 0
            min_dis = 10000.0
            for i, polygon in enumerate(polygons):
                temp = np.min(get_dis(polygon, point_label))
                if temp < min_dis:
                    min_dis = temp
                    # A lane consists of two left polyline and right polyline
                    stage_one_label = i // 2

            mapping['stage_one_label'] = stage_one_label
            
    bev_embed = torch.tensor(vec_map['bev_features'])

    mapping.update(dict(
        matrix=np.array(vectors),
        labels=np.array(labels).reshape([args.future_frame_num, 2]),
        gt_trajectory_global_coordinates=np.array(gt_trajectory_global_coordinates),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=30,
        agents=agents,
        map_start_polyline_idx=map_start_polyline_idx,
        polygons=polygons,
        file_name=os.path.split(instance_dir)[-1],
        sample_token=data['sample_token'],
        trajs=agents,
        vis_lanes=polygons,
        bev_embed=bev_embed,
    ))

    return mapping


def expand_coordinates_with_motion_data(coords):
    """
    Expand a (N, 2) array of 2D coordinates into a (N, 5) array,
    where the extra three columns represent x velocity (x_vel),
    y velocity (y_vel), and heading.
    
    Args:
    - coords (numpy.ndarray): A (N, 2) array of 2D coordinates.
    
    Returns:
    - numpy.ndarray: A (N, 5) array with the original coordinates,
                     x velocity, y velocity, and heading.
    """
    # Calculate velocities
    x_vel = np.diff(coords[:, 0], prepend=coords[0, 0])  # Prepend first x to make the shape consistent
    y_vel = np.diff(coords[:, 1], prepend=coords[0, 1])  # Prepend first y to make the shape consistent

    # Calculate headings in radians
    headings = np.arctan2(y_vel, x_vel)

    # Combine all the data into a single array
    expanded_array = np.hstack((coords, x_vel[:, None], y_vel[:, None], headings[:, None]))

    return expanded_array

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, to_screen=True):
        self.data_dir = args.data_dir
        self.args = args
        self.files = []

        # Collect all file paths
        for each_dir in self.data_dir:
            for root, dirs, cur_files in os.walk(each_dir):
                self.files.extend([os.path.join(root, file) for file in cur_files])

        if to_screen:
            print("Total files:", len(self.files))
        
        self.batch_size = batch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        instance = self.load_and_preprocess_data(self.args, file_path)
        return instance

    def load_and_preprocess_data(self, args, file_path):
        data = nuscene_get_instance(args, file_path)
        return data


def post_eval(args, file2pred, file2labels, DEs):
    score_file = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15:
            each = 'long'
        score_file += '.' + str(each)
        # if 'minFDE' in args.other_params:
        #     score_file += '.minFDE'
    if args.method_span[0] >= utils.NMS_START:
        score_file += '.NMS'
    else:
        score_file += '.score'

    for method in utils.method2FDEs:
        FDEs = utils.method2FDEs[method]
        miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)
        if method >= utils.NMS_START:
            method = 'NMS=' + str(utils.NMS_LIST[method - utils.NMS_START])
        utils.logging(
            'method {}, FDE {}, MR {}, other_errors {}'.format(method, np.mean(FDEs), miss_rate, utils.other_errors_to_string()),
            type=score_file, to_screen=True, append_time=True)
    utils.logging('other_errors {}'.format(utils.other_errors_to_string()),
                  type=score_file, to_screen=True, append_time=True)
    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        utils.logging('ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3', score,
                      type=score_file, to_screen=True, append_time=True)

    utils.logging(vars(args), is_json=True,
                  type=score_file, to_screen=True, append_time=True)