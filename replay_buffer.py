import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, drq=True, low_dim=False):
        self.capacity = capacity
        self.device = device
        self.drq = drq # If true, apply img augmentation
        self.low_dim = low_dim
        if not low_dim:
            self.aug_trans = nn.Sequential(
                nn.ReplicationPad2d(image_pad),
                kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
            obs_type = np.uint8
        else:
            obs_type = np.float32
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        if not self.low_dim and self.drq:
            # in the original code apply augmentation to both obses and obses_aug?
            obses = self.aug_trans(obses)
            next_obses = self.aug_trans(next_obses)
            obses_aug = self.aug_trans(obses_aug)
            next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
