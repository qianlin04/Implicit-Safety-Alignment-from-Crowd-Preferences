import os
import pickle

import absl.app
import absl.flags
import gym
from tqdm import tqdm
import numpy as np
import torch
from pref_learn.sac_collect import *
from pref_learn.utils.eval_utilis import ComputeAdvantage

import jaxrl_m.envs
from jaxrl_m.envs import make_env
from jaxrl_m.learners.d4rl_utils import get_dataset
from pref_learn.utils.utils import (
    define_flags_with_default,
    set_random_seed,
)
from pref_learn.utils.data_utils import (
    sample_from_env,
    get_queries_from_multi,
    get_labels
)
import h5py

if __name__ == "__main__":
    FLAGS_DEF = define_flags_with_default(
        env="maze2d-twogoals-multimodal-v0",
        sample_from_env=False,
        data_dir="./pref_datasets",
        data_seed=0,
        num_query=100,
        query_len=1,
        set_len=32,
        relabel=False,
        dataset_path="",
        label_by_adv=True,
        device="cuda:0",
        dense_annotation=True, # every sample is annotated by all kinds of rewards
        test_only=False,
        minority_ratio=1.0,   # mode 0 is majority with number of num_query, other modes are minority with number of num_query*minority_ratio 
        unsafe_ratio=1.0,  # it controls Num(unsafe traj) / Num(safe traj) during preference generation
        mode_bound_threshold=-1.0,    # filter the pairs with traj deviating from task target 
        trajectory_clip=True,
        flip_ratio=0.0,
    )


# def compute_advantage(agent, obs, env, mode):
#     device = next(agent.actor.parameters()).device

#     def get_value(s):
#         with torch.no_grad():
#             s = s[:, None, :].repeat(64, 1)
#             s = torch.from_numpy(s).float().to(device)
#             a = agent.actor(s, mean=False, with_prob=False)
#             q = torch.min(agent.target_critic1(s, a), agent.target_critic2(s, a)).squeeze(-1)
#         return q.mean(dim=-1).cpu().numpy()
    
#     if 'maze2d' in env.spec.id:
#         target = np.array(env.goals[mode])[None, None, :].repeat(obs.shape[0], 0).repeat(obs.shape[1], 1)
#         obs = np.concatenate([obs, target], -1)
#         reward = np.exp(-np.linalg.norm(obs[..., 0:2] - target, axis=-1))
#         reward = np.sum(reward, axis=-1)

#     adv = get_value(obs[:, -1, :]) - get_value(obs[:, 0, :]) + reward
#     return adv

def main(_):
    FLAGS = absl.flags.FLAGS
    # use fixed seed for collecting segments.
    set_random_seed(FLAGS.data_seed)
    base_path = os.path.join(FLAGS.data_dir, FLAGS.env)
    os.makedirs(base_path, exist_ok=True)

    kwargs = {}
    if 'maze2d' in FLAGS.env and not FLAGS.label_by_adv:
        kwargs['dense_reward']=True

    gym_env = make_env(FLAGS.env, **kwargs)
    gym_env.reset()
    if not FLAGS.dense_annotation:
        FLAGS.num_query *= gym_env.get_num_modes()
    
    get_adv_func = ComputeAdvantage(gym_env, FLAGS.device)
    if FLAGS.relabel:
        assert os.path.exists(FLAGS.dataset_path)
        dataset = pickle.load(open(FLAGS.dataset_path, "rb"))
    else:
        if FLAGS.sample_from_env:
            dataset, _ = sample_from_env(
                gym_env, FLAGS.num_query, FLAGS.set_len, FLAGS.query_len+1, base_path
            )
        else:
            # Collectinf dataset from d4rl
            dataset, _ = get_dataset(gym_env)

            # Creating queries from the dataset
            dataset, _ = get_queries_from_multi(
                gym_env,
                dataset,
                FLAGS.num_query,
                FLAGS.query_len+1,   # for traj with T step, we require a T+1 step to compute the advantage and reward 
                FLAGS.set_len,
                base_path,
                save_queries=False,
                unsafe_ratio=FLAGS.unsafe_ratio,
                trajectory_clip=FLAGS.trajectory_clip,
            )

    query_path = os.path.join(
        base_path, f"queries_num{FLAGS.num_query}_q{FLAGS.query_len}_s{FLAGS.set_len}"
    )
    modes = []
    obs = []
    rewards = []
    del dataset['next_observations']
    new_dataset = dataset
    new_dataset['adv1'], new_dataset['adv2'], new_dataset['mode'], new_dataset['idx'], new_dataset["labels"] = [], [], [], [], []
    for i in tqdm(range(len(dataset["observations"]))):
        if FLAGS.dense_annotation:
            mode_list = np.arange(0, gym_env.get_num_modes()) if i < FLAGS.minority_ratio * len(dataset["observations"]) else [0]
        else:
            prob = np.array([FLAGS.minority_ratio] * gym_env.get_num_modes())
            prob[0] = 1.0
            mode_list = [np.random.choice(gym_env.get_num_modes(), p=prob/np.sum(prob))]

        info_set = {k.replace("infos/", ""):dataset[k][i] for k in dataset.keys() if 'infos/' in k}
        info_set_2 = {k.replace("infos_2/", ""):dataset[k][i] for k in dataset.keys() if 'infos_2/' in k}
        for mode in mode_list:    
            modes.append(mode)
            adv1 = get_adv_func(dataset["observations"][i], mode, info_set, use_adv=FLAGS.label_by_adv)
            adv2 = get_adv_func(dataset["observations_2"][i], mode, info_set_2, use_adv=FLAGS.label_by_adv)

            if i % 100==0:
                print(f"Mode: {mode}, Adv1: {adv1}, Adv2: {adv2}\n\n")

            labels = (adv1 > adv2).reshape(-1, 1)
            flip_mask = np.random.rand(*labels.shape) < FLAGS.flip_ratio
            noisy_label = np.where(flip_mask, 1 - labels, labels)
            new_dataset["labels"].append(noisy_label)
            new_dataset["adv1"].append(adv1)
            new_dataset["adv2"].append(adv2)
            new_dataset["mode"].append(mode)
            new_dataset["idx"].append(i)

    if FLAGS.mode_bound_threshold >= 0: # filter the pairs with traj deviating from task target
        filter_dataset = {k:[] for k in new_dataset}
        assert 'infos/goals' in dataset 
        for mode in range(gym_env.get_num_modes()):
            consistent_pairs = {k:[] for k in new_dataset}
            for i in range( len(new_dataset['mode'])):
                if new_dataset['mode'][i] == mode:
                    for j in range( FLAGS.set_len):
                        idx = new_dataset['idx'][i]
                        goals = new_dataset["infos/goals"][idx][j][0] if new_dataset["labels"][i][j] else new_dataset["infos_2/goals"][idx][j][0]
                        if np.linalg.norm(goals-gym_env.goals[mode], axis=-1) < FLAGS.mode_bound_threshold:
                            for k in consistent_pairs:
                                if k != "mode" and k != "idx":
                                    if k in ["labels", "adv1", "adv2"]:
                                        consistent_pairs[k].append(new_dataset[k][i][j])
                                    else:
                                        consistent_pairs[k].append(new_dataset[k][idx][j])
            n_set = len(consistent_pairs["labels"]) // FLAGS.set_len
            n_pair = n_set * FLAGS.set_len
            for k in filter_dataset:
                if k == 'mode':
                    filter_dataset[k].extend([mode]*n_set)
                elif k == 'idx':
                    filter_dataset[k].extend(np.arange(len(filter_dataset[k]), len(filter_dataset[k])+n_set))
                else:
                    consistent_pairs[k] = np.array(consistent_pairs[k])
                    shape = (-1, FLAGS.set_len) + consistent_pairs[k].shape[1:]
                    filter_dataset[k].extend(consistent_pairs[k][:n_pair].reshape(shape))
            print(f'mode: {mode}, n_set: {n_set}')
        new_dataset = filter_dataset

        


    # for k in new_dataset.keys():
    #     new_dataset[k] = np.stack(new_dataset[k], 0)
    #     print(k, new_dataset[k].shape)

    relabelled_path = str(query_path).replace("queries", "relabelled_queries_by_adv" if FLAGS.label_by_adv else 'relabelled_queries')
    # with open(relabelled_path, "wb") as f:
    #     pickle.dump(new_dataset, f)
    
    with h5py.File(relabelled_path, 'w') as f:
        for k, v in new_dataset.items():
            data = np.stack(new_dataset[k], 0)
            f.create_dataset(k, data=data, compression="gzip", compression_opts=4)
            print(k, data.shape)
            del data

    print("Saved relabelled queries at: ", relabelled_path)
    print("Average mode during relabelling:", sum(modes) / len(modes))

    # Plotting observations to debug
    # rewards = np.array(rewards).reshape(-1, 1)
    # fig = plot_observation_rewards(obs, rewards)
    # fig.savefig("relabelled_queries.png")


if __name__ == "__main__":
    absl.app.run(main)
