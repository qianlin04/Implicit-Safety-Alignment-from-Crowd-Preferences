import math
import pickle
import h5py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pref_learn.utils.data_utils import get_labels


def get_datasets(query_path, observation_dim, action_dim, batch_size, set_length, mode_size=-1, access_to_mode=False, extra_info_in_sample=[]):
    try:
        with open(query_path, "rb") as fp:
            batch = pickle.load(fp)
    except Exception as e:
        batch = {}
        with h5py.File(query_path, 'r') as f:
            for k in f.keys():
                if k=='infos' or k=='infos_2':
                    for info_k in f[k]:
                        batch[k+'/'+info_k] = np.array(f[k][info_k])
                else:
                    batch[k] = np.array(f[k])

    batch["observations"] = batch["observations"][..., :observation_dim]
    batch["observations_2"] = batch["observations_2"][..., :observation_dim]
    assert batch["actions"].shape[-1] == action_dim
    if set_length < 0:
        set_length = batch["observations"].shape[1]

    if access_to_mode:
        assert 'human_pref' in batch or 'mode' in batch, "No mode data in datasets"
        # if not 'human_pref' in batch and 'mode' in batch:
        #     batch['human_pref'] = np.zeros((len(batch["observations"]), mode_size))
        #     for i in range(len(batch['mode'])):
        #         batch['human_pref'][i, batch['mode'][i]] = 1
        # human_pref = np.repeat(np.expand_dims(batch['human_pref'], axis=1), batch["observations"].shape[1], axis=1)
        # human_pref = np.repeat(np.expand_dims(human_pref, axis=2), batch["observations"].shape[2], axis=2)                 
        shape = list(batch["observations"].shape)
        shape[-1] = mode_size      
        batch["observations"] = np.concatenate((batch["observations"], np.zeros(shape)), -1)
        batch["observations_2"] = np.concatenate((batch["observations_2"], np.zeros(shape)), -1)
     
    eval_data_size = int(0.1 * len(batch["labels"]))
    train_data_size = len(batch["labels"]) - eval_data_size

    if len(batch['labels'].shape)==2:
        batch['labels'] = batch['labels'][:, :, None]

    train_batch = {}
    eval_batch = {}

    random_idx = np.random.permutation(len(batch["labels"]))
    for k in batch:
        if 'idx' in batch: 
            if k in ['labels', 'idx', 'mode', 'adv1', 'adv2']:
                train_batch[k] = batch[k][random_idx[:train_data_size]] 
                eval_batch[k] = batch[k][random_idx[train_data_size:]]
            else:
                train_batch[k] = batch[k]
                eval_batch[k] = batch[k]
        else:
            train_batch[k] = batch[k][:train_data_size]
            eval_batch[k] = batch[k][train_data_size:]

    train_dataset = PreferenceDataset(train_batch, access_to_mode, mode_size, extra_info_in_sample)
    eval_dataset = PreferenceDataset(eval_batch, access_to_mode, mode_size, extra_info_in_sample)
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    _, _, len_query, obs_dim = batch["observations"].shape
    return (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        set_length,
        len_query,
        obs_dim,
    )


class PreferenceDataset(Dataset):
    def __init__(self, pref_dataset, access_to_mode=False, mode_size=None, extra_info_in_sample=[]):
        self.pref_dataset = pref_dataset
        self.access_to_mode = access_to_mode
        self.mode_size = mode_size
        self.extra_info_in_sample = extra_info_in_sample

    def __len__(self):
        return len(self.pref_dataset["labels"])

    def __getitem__(self, idx):
        labels = self.pref_dataset["labels"][idx]
        data_idx = self.pref_dataset["idx"][idx] if "idx" in self.pref_dataset else idx
        observations = self.pref_dataset["observations"][data_idx]
        observations_2 = self.pref_dataset["observations_2"][data_idx]
        actions=self.pref_dataset["actions"][data_idx]
        actions_2=self.pref_dataset["actions_2"][data_idx]
        if self.access_to_mode:
            observations, observations_2 = observations.copy(), observations_2.copy()
            mode = self.pref_dataset["mode"][idx]
            observations[-self.mode_size+mode], observations_2[-self.mode_size+mode] = 1, 1

        sample = dict(
            observations=observations, observations_2=observations_2, actions=actions, actions_2=actions_2, labels=labels
        )
        for k in self.extra_info_in_sample:
            info_idx = idx if k in ['labels', 'idx', 'mode', 'adv1', 'adv2'] else data_idx
            sample[k] = self.pref_dataset[k][info_idx]
        return sample

    def get_mode_data(self, batch_size):
        batch_size = min(batch_size, len(self))
        #idxs = np.random.choice(range(len(self)), size=batch_size, replace=False)
        idxs = np.random.randint(0, len(self), size=batch_size)
        data_idxs = self.pref_dataset["idx"][idxs] if "idx" in self.pref_dataset else idxs
        
        observations, observations_2 = self.pref_dataset["observations"][data_idxs], self.pref_dataset["observations_2"][data_idxs]
        if self.access_to_mode:
            observations, observations_2 = observations.copy(), observations_2.copy()
            modes = self.pref_dataset["mode"][idxs]
            for i, mode in enumerate(modes):
                observations[i, -self.mode_size+mode], observations_2[i, -self.mode_size+mode] = 1, 1

        sample = dict(
            observations=observations,
            observations_2=observations_2,
            actions=self.pref_dataset["actions"][data_idxs],
            actions_2=self.pref_dataset["actions_2"][data_idxs],
            labels = self.pref_dataset["labels"][idxs],
            mode = self.pref_dataset["mode"][idxs],
        )
        if 'reward' in self.pref_dataset and 'reward_2' in self.pref_dataset:
            sample['reward']=self.pref_dataset["reward"][idxs]
            sample['reward_2']=self.pref_dataset["reward_2"][idxs]
        if 'adv1' in self.pref_dataset and 'adv2' in self.pref_dataset:
            sample['adv1']=self.pref_dataset["adv1"][idxs]
            sample['adv2']=self.pref_dataset["adv2"][idxs]
        for k in self.pref_dataset:
            if k.startswith('infos/') or k.startswith('infos_2/'):
                sample[k]=self.pref_dataset[k][data_idxs]
        return sample, batch_size
    
    def get_obs_mean_std(self,):
        obs_dim = self.pref_dataset["observations"].shape[-1]
        combined_obs = np.concatenate([self.pref_dataset["observations"], self.pref_dataset["observations_2"]], axis=0)
        mean = np.mean(combined_obs.reshape(-1, obs_dim), axis=0)
        std = np.std(combined_obs.reshape(-1, obs_dim), axis=0)
        return mean, std

def get_demo_datasets(demo_path, observation_dim, action_dim, batch_size, set_length, mode_num=-1):
    with open(demo_path, "rb") as fp:
        batch = pickle.load(fp)

    if len(batch["observations"].shape)==3:
        batch["observations"] = np.expand_dims(batch["observations"], 1)
        batch["actions"] = np.expand_dims(batch["actions"], 1)
    batch["observations"] = batch["observations"][..., :observation_dim]
    assert batch["actions"].shape[-1] == action_dim
    if set_length < 0:
        set_length = batch["observations"].shape[1]
    
    if not 'human_pref' in batch:
        if mode_num == -1:
            mode_num = np.max([len(m) for m in batch['mode']])
        batch['human_pref'] = np.zeros((len(batch["observations"]), mode_num))
        for i, mode in enumerate(batch['mode']):
            for m in mode:
                batch['human_pref'][i, m] = 1.0
        del batch['mode']
    
    eval_data_size = int(0.1 * len(batch["observations"]))
    train_data_size = len(batch["observations"]) - eval_data_size

    train_batch = {}
    eval_batch = {}

    for k in batch:
        train_batch[k] = batch[k][:train_data_size]
        eval_batch[k] = batch[k][train_data_size:]

    train_dataset = DemoDataset(train_batch)
    eval_dataset = DemoDataset(eval_batch)
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    _, _, len_query, obs_dim = batch["observations"].shape
    return (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        set_length,
        len_query,
        obs_dim,
    )

class DemoDataset(Dataset):
    def __init__(self, demo_dataset):
        self.demo_dataset = demo_dataset

    def __len__(self):
        return len(self.demo_dataset["observations"])

    def __getitem__(self, idx):
        return {k:self.demo_dataset[k][idx] for k in self.demo_dataset}

    def get_mode_data(self, batch_size):
        batch_size = min(batch_size, len(self))
        #idxs = np.random.choice(range(len(self)), size=batch_size, replace=False)
        idxs = np.random.randint(0, len(self), size=batch_size)
        
        sample = dict(
            observations=self.demo_dataset["observations"][idxs],
            actions=self.demo_dataset["actions"][idxs],
        )
        if 'reward' in self.demo_dataset:
            sample['reward']=self.demo_dataset["reward"][idxs]
        return sample, batch_size
    
    def get_obs_mean_std(self,):
        obs_dim = self.demo_dataset["observations"].shape[-1]
        mean = np.mean(self.demo_dataset["observations"].reshape(-1, obs_dim), axis=0)
        std = np.std(self.demo_dataset["observations"].reshape(-1, obs_dim), axis=0)
        return mean, std

class Annealer:
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_latent(batch, env, reward_model, mode, num_samples):
    # obs_dim = env.reward_observation_space.shape[0]
    obs1 = batch["observations"]
    obs2 = batch["observations_2"]
    obs_dim = obs1.shape[-1]
    seg_reward_1 = env.compute_reward(obs1.reshape(-1, reward_model.size_segment, obs_dim), mode)
    seg_reward_2 = env.compute_reward(obs2.reshape(-1, reward_model.size_segment, obs_dim), mode)

    seg_reward_1 = seg_reward_1.reshape(
        num_samples, reward_model.annotation_size, reward_model.size_segment, 
    )
    seg_reward_2 = seg_reward_2.reshape(
        num_samples, reward_model.annotation_size, reward_model.size_segment,
    )

    labels = get_labels(seg_reward_1, seg_reward_2)

    device = next(reward_model.parameters()).device
    obs1 = torch.from_numpy(obs1).float().to(device)
    obs2 = torch.from_numpy(obs2).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    with torch.no_grad():
        mean, _ = reward_model.encode(obs1, obs2, labels)
    return mean.cpu().numpy()


def get_posterior(env, reward_model, dataset, mode, num_samples):
    batch, num_samples = dataset.get_mode_data(num_samples)
    return get_latent(batch, env, reward_model, mode, num_samples)


def get_all_posterior(env, reward_model, dataset, num_samples):
    means = []
    for mode in range(env.get_num_modes()):
        means.append(get_posterior(env, reward_model, dataset, mode, num_samples))
    return np.stack(means, axis=0)


def get_biased(env, reward_model, dataset=None):
    means = []
    if dataset:
        batch, _ = dataset.get_mode_data(1)
    else:
        obs1, obs2 = env.get_biased_data(reward_model.annotation_size)
        batch = dict(
            observations=obs1[None, :, None],
            observations_2=obs2[None, :, None],
        )
    # import pdb; pdb.set_trace()
    for mode in range(env.get_num_modes()):
        means.append(get_latent(batch, env, reward_model, mode, 1))
    return np.stack(means, axis=0)
