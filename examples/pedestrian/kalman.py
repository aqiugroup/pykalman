#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File          :kalman.py
@Description:  :
    ref:
    alg : https://sourcegraph.com/github.com/vita-epfl/trajnetplusplusbaselines/-/blob/trajnetbaselines/classical/kalman.py?L6
    data:
        [1] https://github.com/vita-epfl/trajnetplusplustools
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

        [2] https://thedebugger811.github.io/posts/2020/10/data_conversion/
        1. In our external dataset, each trajectory point is delimited by ‘\t’
        2. TrackRow takes the arguments ‘frame’, ‘ped_id’, ‘x’, ‘y’ in order.
        对应的数据生成和读取方法：
            https://github.com/vita-epfl/trajnetplusplusdataset/blob/master/trajnetdataset/convert.py
            https://github.com/vita-epfl/trajnetplusplusdataset/blob/eth/trajnetdataset/readers.py

@Date          :2022/10/07 10:56:49
@Author        :aqiu
@version       :1.0
'''

import numpy as np
import pykalman


def predict(paths, predict_all=True, n_predict=12, obs_length=9):
    multimodal_outputs = {}
    neighbours_tracks = []

    primary = paths[0]
    start_frame = primary[obs_length - 1].frame
    frame_diff = primary[1].frame - primary[0].frame
    first_frame = start_frame + frame_diff

    ## Primary Prediction
    if not predict_all:
        paths = paths[0:1]

    for i, path in enumerate(paths):
        path = paths[i]
        ped_id = path[0].pedestrian
        past_path = [t for t in path if t.frame <= start_frame]
        past_frames = [t.frame for t in path if t.frame <= start_frame]

        ## To consider agent or not consider.
        if start_frame not in past_frames:
            continue
        if len(past_path) < 2:
            continue

        initial_state_mean = [path[0].x, 0, path[0].y, 0]
        transition_matrix = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1],
                             [0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0], [0, 0, 1, 0]]

        kf = pykalman.KalmanFilter(transition_matrices=transition_matrix,
                                   observation_matrices=observation_matrix,
                                   transition_covariance=1e-5 * np.eye(4),
                                   observation_covariance=0.05**2 * np.eye(2),
                                   initial_state_mean=initial_state_mean)
        # kf.em([(r.x, r.y) for r in path[:9]], em_vars=['transition_matrices',
        #                                                'observation_matrices'])
        kf.em([(r.x, r.y) for r in past_path])
        observed_states, _ = kf.smooth([(r.x, r.y) for r in past_path])

        # sample predictions (first sample corresponds to last state)
        # average 5 sampled predictions
        predictions = None
        for _ in range(5):
            _, pred = kf.sample(n_predict + 1,
                                initial_state=observed_states[-1])
            if predictions is None:
                predictions = pred
            else:
                predictions += pred
        predictions /= 5.0

        #write
        if i == 0:
            primary_track = predictions[1:]
        else:
            neighbours_tracks.append(np.array(predictions[1:]))

    ## Unimodal Ouput
    if len(np.array(neighbours_tracks)):
        neighbours_tracks = np.array(neighbours_tracks).transpose(1, 0, 2)

    multimodal_outputs[0] = primary_track, neighbours_tracks
    return multimodal_outputs
