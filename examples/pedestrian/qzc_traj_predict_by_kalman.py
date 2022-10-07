#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File          :qzc_traj_predict.py
@Description:  :
    ref:
    alg : https://sourcegraph.com/github.com/vita-epfl/trajnetplusplusbaselines/-/blob/trajnetbaselines/classical/kalman.py?L6
    data:
        [1] 仿真数据: https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v3.1
            five_parallel_synth.zip 数据截取了552393帧前的数据
        [2] 数据格式: https://github.com/vita-epfl/trajnetplusplustools
        Datasets are split into train, val and test set. Every line is a self contained JSON string (ndJSON).
        Scene: {"scene": {"id": 266, "p": 254, "s": 10238, "e": 10358, "fps": 2.5, "tag": 2}}
        Track: {"track": {"f": 10238, "p": 248, "x": 13.2, "y": 5.85}}
        with:
            id: scene id
            p: pedestrian id
            s, e: start and end frame id
            fps: frame rate
            tag: trajectory type
            f: frame id
            x, y: x- and y-coordinate in meters
            pred_number: (optional) prediction number for multiple output predictions
            scene_id: (optional) corresponding scene_id for multiple output predictions

        [3] 生成目标数据格式： https://thedebugger811.github.io/posts/2020/10/data_conversion/
        1. In our external dataset, each trajectory point is delimited by ‘\t’
        2. TrackRow takes the arguments ‘frame’, ‘ped_id’, ‘x’, ‘y’ in order.
        对应的数据生成和读取方法：
            https://github.com/vita-epfl/trajnetplusplusdataset/blob/master/trajnetdataset/convert.py
            https://github.com/vita-epfl/trajnetplusplusdataset/blob/eth/trajnetdataset/readers.py

@Date          :2022/10/07 10:56:49
@Author        :aqiu
@version       :1.0
'''
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import json

# print("__package__", __package__)
if __package__ == "":
    from data import SceneRow, TrackRow
    from reader import Reader
    from qzc_traj_visualization import *
else:
    from .data import SceneRow, TrackRow
    from .reader import Reader
    from .qzc_traj_visualization import *


def init_argument():
    parser = argparse.ArgumentParser()
    traj_path = 'data/five_parallel_synth/test_pred/'
    traj_path = os.path.join(os.path.dirname(__file__), traj_path)

    parser.add_argument('--path',
                        default=traj_path,
                        help='directory of data to test')
    parser.add_argument('--output',
                        nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length',
                        default=9,
                        type=int,
                        help='observation length')
    parser.add_argument('--pred_length',
                        default=12,
                        type=int,
                        help='prediction length')
    parser.add_argument('--write_only',
                        action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision',
                        action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels',
                        required=False,
                        nargs='+',
                        help='labels of models')
    parser.add_argument('--normalize_scene',
                        action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes',
                        default=1,
                        type=int,
                        help='number of modes to predict')
    parser.add_argument('--kf',
                        default=True,
                        action='store_true',
                        help='consider kalman in evaluation')
    parser.add_argument('--viz_folder',
                        default='./visualizations',
                        help='base folder to store visualizations')
    args = parser.parse_args()

    args.output = []
    ## assert length of output models is not None
    if (not args.kf):
        assert 'No handcrafted baseline mentioned'

    return args


def preprocess_test(scene, obs_len):
    """Remove pedestrian trajectories that appear post observation period.
    Can occur when the test set has overlapping scenes."""
    obs_frames = [primary_row.frame for primary_row in scene[0]][:obs_len]
    last_obs_frame = obs_frames[-1]
    scene = [[row for row in ped if row.frame <= last_obs_frame]
             for ped in scene if ped[0].frame <= last_obs_frame]
    return scene


def load_test_datasets(dataset, args):
    """Load Test Prediction file with goals (optional)"""
    # dataset_name = dataset.replace(
    #     args.path.replace('_pred', '') + 'test/', '') + '.ndjson'
    # print('Dataset Name: ', dataset_name)
    dataset_name = dataset

    # Read Scenes from 'test' folder
    reader = Reader(args.path + dataset_name + '.ndjson', scene_type='paths')
    ## Necessary modification of train scene to add filename (for goals)
    scenes = [(dataset, s_id, s) for s_id, s in reader.scenes()]

    return dataset_name, scenes


def trajnet_tracks(row):
    x = round(row.x, 2)
    y = round(row.y, 2)
    if row.prediction_number is None:
        return json.dumps(
            {'track': {
                'f': row.frame,
                'p': row.pedestrian,
                'x': x,
                'y': y
            }})
    return json.dumps({
        'track': {
            'f': row.frame,
            'p': row.pedestrian,
            'x': x,
            'y': y,
            'prediction_number': row.prediction_number,
            'scene_id': row.scene_id
        }
    })


def trajnet_scenes(row):
    return json.dumps({
        'scene': {
            'id': row.scene,
            'p': row.pedestrian,
            's': row.start,
            'e': row.end,
            'fps': row.fps,
            'tag': row.tag
        }
    })


def trajnet(row):
    if isinstance(row, TrackRow):
        return trajnet_tracks(row)
    if isinstance(row, SceneRow):
        return trajnet_scenes(row)

    raise Exception('unknown row type')


def write_predictions(pred_list, scenes, model_name, dataset_name, args):
    """Write predictions corresponding to the scenes in the respective file"""
    seq_length = args.obs_length + args.pred_length
    with open(args.path + '{}/{}.ndjson'.format(model_name, dataset_name),
              "a") as myfile:
        ## Write All Predictions
        for (predictions, (_, scene_id, paths)) in zip(pred_list, scenes):
            ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
            observed_path = paths[0]
            frame_diff = observed_path[1].frame - observed_path[0].frame
            first_frame = observed_path[args.obs_length - 1].frame + frame_diff
            ped_id = observed_path[0].pedestrian
            ped_id_ = []
            for j, _ in enumerate(paths[1:]):  ## Only need neighbour ids
                ped_id_.append(paths[j + 1][0].pedestrian)

            ## Write SceneRow
            scenerow = SceneRow(
                scene_id, ped_id, observed_path[0].frame,
                observed_path[0].frame + (seq_length - 1) * frame_diff, 2.5, 0)
            # scenerow = SceneRow(scenerow.scene, scenerow.pedestrian, scenerow.start, scenerow.end, 2.5, 0)
            myfile.write(trajnet(scenerow))
            myfile.write('\n')

            for m in range(len(predictions)):
                prediction, neigh_predictions = predictions[m]
                ## Write Primary
                for i in range(len(prediction)):
                    track = TrackRow(first_frame + i * frame_diff, ped_id,
                                     prediction[i, 0].item(),
                                     prediction[i, 1].item(), m, scene_id)
                    myfile.write(trajnet(track))
                    myfile.write('\n')

                ## Write Neighbours (if non-empty)
                if len(neigh_predictions):
                    for n in range(neigh_predictions.shape[1]):
                        neigh = neigh_predictions[:, n]
                        for j in range(len(neigh)):
                            track = TrackRow(first_frame + j * frame_diff,
                                             ped_id_[n], neigh[j, 0].item(),
                                             neigh[j, 1].item(), m, scene_id)
                            myfile.write(trajnet(track))
                            myfile.write('\n')


def predict_scene(predictor, model_name, paths, args):
    """For each scene, get model predictions"""
    paths = preprocess_test(paths, args.obs_length)
    if 'kf' in model_name:
        predictions = predictor(paths,
                                n_predict=args.pred_length,
                                obs_length=args.obs_length)
    elif 'cv' in model_name:
        print("not implement")
    else:
        raise NotImplementedError
    return predictions


def load_predictor(model_name):
    """Loading the APPROPRIATE model"""
    if 'kf' in model_name:
        print("Kalman")
        if __package__ == "":
            from kalman import predict as predictor
        else:
            from .kalman import predict as predictor
    elif 'cv' in model_name:
        print("not implement")
        # from .constant_velocity import predict as predictor

    return predictor


def get_predictions(args):
    """Get model predictions for each test scene and write the predictions in appropriate folders"""
    ## 1 解析数据集 List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([
        f.split('.')[-2] for f in os.listdir(args.path)
        if not f.startswith('.') and f.endswith('.ndjson')
    ])

    ## Handcrafted Baselines (if included)
    if args.kf:
        args.output.append('/kf.pkl')

    ## Extract Model names from arguments and create its own folder in 'test_pred' for storing predictions
    ## WARNING: If Model predictions already exist from previous run, this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print('Predictions corresponding to {} already exist.'.format(
                model_name))
            # print('Loading the saved predictions')
            # continue

        ## 2 选择预测器，这里使用kf
        print("Model Name: ", model_name)
        predictor = load_predictor(model_name)

        # Iterate over test datasets
        for dataset in datasets:
            ## 3加载数据集 Load dataset
            dataset_name, scenes_origin = load_test_datasets(dataset, args)
            ## 添加真值
            paths_file_gt = args.path.replace(
                '_pred', '_private') + dataset_name + '.ndjson'
            reader_gt = Reader(paths_file_gt, scene_type='paths')
            print("gt: ", paths_file_gt)

            ## 4 并行预测所有数据 Get all predictions in parallel. Faster!
            scenes = tqdm(scenes_origin)
            pred_list = Parallel(n_jobs=12)(
                delayed(predict_scene)(predictor, model_name, paths[2], args)
                for paths in scenes)
            # for paths in scenes:
            #     pred_list = predict_scene(predictor, model_name, paths[2], args)
            #     print("pred_list ", pred_list)

            ## 5 保存预测的结果 Write all predictions
            write_predictions(pred_list, scenes, model_name, dataset_name, args)
            # https://sourcegraph.com/github.com/vita-epfl/trajnetplusplusbaselines/-/blob/evaluator/write_utils.py?L42
            print("hi")


if __name__ == '__main__':
    args = init_argument()
    get_predictions(args)
