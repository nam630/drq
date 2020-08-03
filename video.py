import os
import sys

import imageio
import numpy as np

import utils


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, fps=10):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array',
                               height=self.height,
                               width=self.width)
            # frame shaped in (c, h, w) --> imageio requires h, w, c
            frame = frame.transpose(1, 2, 0)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            print("frame num: {}".format(len(self.frames)))
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
