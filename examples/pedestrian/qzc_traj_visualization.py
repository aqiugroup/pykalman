#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File          :traj_visualization.py
@Description:  :
    ref: https://sourcegraph.com/github.com/vita-epfl/trajnetplusplusbaselines/-/blob/evaluator/visualize_prediction_as_gif.py?L75
@Date          :2022/10/07 17:49:49
@Author        :aqiu
@version       :1.0
'''
if __package__ == "":
    from reader import Reader
else:
    from .reader import Reader

import os
from pathlib import Path
import argparse
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def add_gt_observation_to_prediction(gt_observation, model_prediction):
    obs_length = len(gt_observation[0]) - len(model_prediction[0])
    full_predicted_paths = [
        gt_observation[ped_id][obs_length - 3:obs_length] + pred
        for ped_id, pred in enumerate(model_prediction)
    ]
    return full_predicted_paths


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines


def animate_and_save_scene(scene, dataset_file, scene_id):

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    z_zeros = np.zeros((scene.shape[0], scene.shape[1], 1))
    data = np.concatenate([scene, z_zeros], axis=-1)
    data = data.transpose(0, 2, 1)

    # NOTE: Can't pass empty arrays into 3d version of plot()
    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

    # Setting the axes properties
    ax.set_xlim3d([-5.0, 5.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 0.3])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig,
                                       update_lines,
                                       25,
                                       fargs=(data, lines),
                                       interval=50,
                                       blit=False)
    line_ani.save(f'{dataset_file}/scene{scene_id}_animation.gif',
                  writer='imagemagick',
                  fps=5)
    plt.close()


def main():
    # gt traj
    traj_path_gt = 'data/five_parallel_synth/test_private'
    traj_path_gt = os.path.join(os.path.dirname(__file__), traj_path_gt)
    dataset_name = sorted([
        f for f in os.listdir(traj_path_gt)
        if not f.startswith('.') and f.endswith('.ndjson')
    ])
    traj_path_gt = traj_path_gt + "/" + dataset_name[0]
    # pred traj
    traj_path_pred = 'data/five_parallel_synth/test_pred/kf_modes1'
    traj_path_pred = os.path.join(os.path.dirname(__file__), traj_path_pred)
    dataset_name = sorted([
        f for f in os.listdir(traj_path_pred)
        if not f.startswith('.') and f.endswith('.ndjson')
    ])
    traj_path_pred = traj_path_pred + "/" + dataset_name[0]
    # combine
    traj_path = [traj_path_gt, traj_path_pred]

    # save visualization
    save_visualization_path = os.path.join(os.path.dirname(__file__),
                                           "visualizations")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_files',
        nargs='+',
        default=traj_path,
        help='Provide the ground-truth file followed by model prediction file')
    parser.add_argument('--viz_folder',
                        default=save_visualization_path,
                        help='base folder to store visualizations')
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='sample n trajectories')
    parser.add_argument('--id',
                        type=int,
                        nargs='*',
                        help='plot a particular scene')
    parser.add_argument('--random',
                        default=True,
                        action='store_true',
                        help='randomize scenes')
    args = parser.parse_args()

    # assert len(
    #     args.dataset_files) > 2, "Please provide only one prediction file"

    # Determine and construct appropriate folders to save visualization
    dataset_name = args.dataset_files[1].split('/')[-4]
    model_name = args.dataset_files[1].split('/')[-2]

    folder_name = f"{args.viz_folder}/{dataset_name}/{model_name}"
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    ## Read Scenes
    reader = Reader(args.dataset_files[1], scene_type='paths')
    if args.id:
        scenes = reader.scenes(ids=args.id, randomize=args.random)
    elif args.n:
        scenes = reader.scenes(limit=args.n, randomize=args.random)
    else:
        scenes = reader.scenes(randomize=args.random)

    reader_gt = Reader(args.dataset_files[0], scene_type='paths')
    ## Visualize different scenes as GIF
    for scene_id, paths in scenes:
        print("Scene ID: ", scene_id)
        _, paths_gt = reader_gt.scene(scene_id)
        full_predicted_paths = add_gt_observation_to_prediction(
            paths_gt, paths)
        # scene = Reader.paths_to_xy(full_predicted_paths)
        scene = Reader.paths_to_xy(paths)
        # scene = Reader.paths_to_xy(paths_gt)
        scene = scene.transpose(1, 0, 2)
        animate_and_save_scene(scene, folder_name, scene_id)


if __name__ == '__main__':
    main()
