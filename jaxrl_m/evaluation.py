from typing import Dict
import jax
import numpy as np
from collections import defaultdict

import gym
from gym.wrappers import RecordEpisodeStatistics
import numpy as np
import os

from jaxrl_m.wandb import WANDBVideo



def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries
    into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def step(env, action):
    # obs, reward, done, info = env.step(action)
    # return (obs, reward, done, info), False
    try:
        obs, reward, done, info = env.step(action)
        return (obs, reward, done, info), False
    except Exception as e:
        print(e)
        return None, True



def evaluate(
    policy_fn,
    env: gym.Env,
    num_episodes: int,
    save_video: bool = False,
    render_frame: bool = True,
    name="eval_video",
    reset_kwargs={},
    obs_fn=lambda x: x,
    reward_fn=None,
    critic_fn=None,
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(
            env,
            name=name,
            max_videos=1,
            render_frame=render_frame,
            obs_fn=obs_fn,
            agent=critic_fn,
        )
    env = RecordEpisodeStatistics(env)

    stats = defaultdict(list)
    stats['obs'] = []
    stats['goal'] = []
    episode_count = 0
    while episode_count < num_episodes:
        observation = env.reset(**reset_kwargs)
        done = False
        learned_reward = []
        fail_flag = False
        infos = []
        obs = []
        rew = 0
        action = None
        while not done:

            observation = obs_fn(observation, reward=rew, action=action)
            if reward_fn is not None:
                learned_reward.append(reward_fn(observation))

            action = policy_fn(observation)

            step_data, fail_flag = step(env, action)
            if fail_flag:
                break
            observation, rew, done, info = step_data
            obs.append(observation[:2])

            done = done
            if done and reward_fn is not None:
                info["episode.learned_reward"] = np.array(learned_reward).mean()
            infos.append(info)
        
        obs = np.array(obs)
        stats['rew_vec'].append(np.sum(np.array([info['rew_vec'] for info in infos]), 0))
        stats['cost'].append(0 if not 'cost' in infos[0] else np.sum([info['cost'] for info in infos]))
        stats['obs_list'].append(obs)
        stats['env_pref'].append(env.get_pref())

        if not fail_flag:
            episode_count += 1
            # for info in infos:
            #     add_to(stats, flatten(info))

    # for k, v in stats.items():
    #     if k=='obs_list' or k=='env_pref': continue
    #     try:
    #         stats[k] = np.mean(v)
    #     except:
    #         pass
    return stats

