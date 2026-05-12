import torch
import jaxrl_m.envs
import gym
from pref_learn.models.utils import get_datasets
from pref_learn.utils.data_utils import get_labels
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
from d4rl.pointmaze.gridcraft import grid_env
from d4rl.pointmaze.gridcraft import grid_spec
import tqdm
from functools import partial
from pref_learn.models.utils import get_datasets


def get_latent(obs1, obs2, env, reward_model, mode, labels=None):
    obs_dim = obs1.shape[-1]
    if len(obs1.shape)==3:
        obs1 = np.expand_dims(obs1, 0)
        obs2 = np.expand_dims(obs2, 0)

    if labels is None:
        seg_reward_1 = env.compute_reward(obs1, mode)
        seg_reward_2 = env.compute_reward(obs2, mode)

        shape = obs1.shape
        seg_reward_1 = seg_reward_1.reshape(
            -1, shape[1], shape[2],
        )
        seg_reward_2 = seg_reward_2.reshape(
            -1, shape[1], shape[2],
        )
        labels = get_labels(seg_reward_1, seg_reward_2)
        
    device = next(reward_model.parameters()).device
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    with torch.no_grad():
        mean, logvar = reward_model.encode(obs1, obs2, labels)
    return mean.squeeze().cpu().numpy(), logvar.squeeze().cpu().numpy()


    
def load_reward_model(ckpt):
    with open(os.path.join(ckpt, f"best_model.pt"), "rb") as f:
        reward_model = torch.load(f, weights_only=False)
    print("load reward mode from: ", os.path.join(ckpt, f"best_model.pt"))
    return reward_model
