#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File          :data.py
@Description:  :
    ref: https://sourcegraph.com/github.com/vita-epfl/trajnetplusplustools/-/blob/trajnetplusplustools/data.py
@Date          :2022/10/07 11:49:58
@Author        :aqiu
@version       :1.0
'''

# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from collections import namedtuple


TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number', 'scene_id'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None, None)
SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])
SceneRow.__new__.__defaults__ = (None, None, None, None, None, None)
