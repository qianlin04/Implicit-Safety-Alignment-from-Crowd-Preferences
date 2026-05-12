import numpy as np
import torch
import pickle
from experiments.active_utils import  get_latent
from jaxrl_m.evaluation import evaluate
from functools import partial
import os
from pref_learn.sac_collect import Actor, Critic

class ComputeAdvantage:
    def __init__(self, env, device='cpu', gamma=1.0, agent_dir='./pref_datasets'):
        if hasattr(env, 'full_observation_space'):
            self.obs_dim = env.full_observation_space.shape[0]
        else:
            self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.env_name = env.spec.id
        self.env = env
        self.gamma = 1.0 #if 'maze' in self.env_name else 0.99  #temp
        self.device = device

        model_para = torch.load(f"{agent_dir}/{env.spec.id}/sac_agent.pth", weights_only=False)
        self.actor = Actor(self.obs_dim, self.action_dim)
        self.critic1, self.critic2 = Critic(self.obs_dim, self.action_dim), Critic(self.obs_dim, self.action_dim)
        self.actor.load_state_dict(model_para['actor'])
        self.critic1.load_state_dict(model_para['critic1'])
        self.critic2.load_state_dict(model_para['critic2'])
        self.actor, self.critic1, self.critic2 = self.actor.to(device), self.critic1.to(device), self.critic2.to(device)

    def get_v(self, s):
        with torch.no_grad():
            s = s[:, None, :].repeat(64, 1)
            s = torch.from_numpy(s).float().to(self.device)
            a = self.actor(s, mean=False, with_prob=False)
            q = torch.min(self.critic1(s, a), self.critic2(s, a)).squeeze(-1)
        return q.mean(dim=-1).cpu().numpy()

    def __call__(self, obs, mode, info, debug=False, use_adv=True):
        # for traj with T step, we require a T+1 step to compute the advantage and reward. Thus, obs.shape=[...,T+1]
        obs = self.env.add_goal_into_state(obs, mode)
        obs_shape = obs.shape
        obs_flat = obs.reshape(-1, obs_shape[-1])
        info_flat = {}
        for k in info:
            info_flat[k] = info[k].reshape(-1, info[k].shape[-1])
        reward = self.env.get_r(obs_flat, mode, info_flat, add_cost_to_reward=True) 
        reward = reward.reshape(obs_shape[:-1])

        if 'Reach' in self.env.spec.id:   # get_r() of Reach env can calculate the reward of final state using obs, so need to approximate it  
            reward[..., -1] = reward[..., -2]

        T = reward.shape[-1]
        if not use_adv:
            adv = reward = np.sum(reward[..., :-1], axis=-1)
        else:
            for t in range(T-1):
                reward[..., t] = reward[..., t] * self.gamma**t
            reward = np.sum(reward[..., :-1], axis=-1)
            adv = self.get_v(obs[..., -1, :]) * self.gamma**(T-1) - self.get_v(obs[..., 0, :]) + reward

        if debug:
            v=np.array([self.get_v(obs[..., t, :])[0]  for t in range(obs.shape[1])])
            r = self.env.get_r(obs_flat, mode, info_flat, add_cost_to_reward=True) .reshape(obs_shape[:-1])[0]
            print(self.get_v(obs[..., -1, :])* self.gamma**T , self.get_v(obs[..., 0, :]), reward, np.sum(info_flat['cost']),len(info_flat['cost']))
            print("v: ", v)
            print("r: ", r)
            print("adv per step: ", v[1:]*self.gamma+r[:-1]-v[:-1])
            pos = obs[..., :2] * 10
            dis = np.linalg.norm(pos - self.env.goals[mode], axis=-1)
            print("dis: ", dis.flatten())
            print('\n')
                

        return adv
    
    def get_label(self, batch, mode, idx=None, use_adv=True):
        data = {k: batch[k][idx] for k in batch} if idx is not None else batch
        obs1, obs2 = data['observations'], data['observations_2']
        info_set = {k.replace("infos/", ""):data[k] for k in data.keys() if 'infos/' in k}
        info_set_2 = {k.replace("infos_2/", ""):data[k] for k in data.keys() if 'infos_2/' in k}
        adv1 = self.__call__(obs1, mode, info_set, use_adv=use_adv)
        adv2 = self.__call__(obs2, mode, info_set_2, use_adv=use_adv)
        labels = (adv1>adv2).reshape(-1, 1).astype(np.float32)
        return labels


def latent_sample_evaluate(gym_env, reward_model, eval_dataset, get_adv_func, policy_fn, FLAGS,
         latent_sample_type='prior', sampling_method='random', eval_episodes=10, fix_mode=-1):
    
    if latent_sample_type=='posterior':
        if sampling_method == "random":
            dataset, _ = eval_dataset.get_mode_data(gym_env.get_num_modes()*eval_episodes)  
            obs1 = dataset['observations']
            obs2 = dataset['observations_2']

    
    utility, obs_list, env_pref_list, rew_vec_list, cost = [], [], [], [], []
    n_eval_episode = eval_episodes*gym_env.get_num_modes() if latent_sample_type=='posterior' else eval_episodes
    print('prior: ', reward_model.mean, torch.exp(0.5*reward_model.log_var))
    if hasattr(reward_model, 'biased_latents'):
        for mode, latent in enumerate(reward_model.biased_latents):
            print(f'mode: {mode},    avg posterior latent: {latent}')
    for ep in range(n_eval_episode):
        if latent_sample_type=='posterior':
            if sampling_method != 'posterior':
                mode_n = ep // eval_episodes
                gym_env.set_mode(mode_n)
                sample_id = ep if sampling_method == "random" else 0
                    
                ep_obs1 = obs1[sample_id, None]
                ep_obs2 = obs2[sample_id, None]

                if get_adv_func is not None:
                    labels = get_adv_func.get_label(dataset, mode_n, idx=sample_id, use_adv=FLAGS.label_by_adv)
                else:
                    labels = None      

                # print('true labels:', dataset['labels'][sample_id])
                # for m in range(gym_env.get_num_modes()):
                #     print('mode ', m , 'labels: ', get_adv_func.get_label(dataset, m, idx=sample_id))

                mean, logvar = get_latent(ep_obs1, ep_obs2, gym_env, reward_model, mode_n, labels=labels)       
            else:
                mode_n = ep // eval_episodes if fix_mode==-1 else fix_mode
                gym_env.set_mode(mode_n)
                mean = reward_model.biased_latents[mode_n]
                logvar = np.zeros_like(mean)  
        else:
            mode_n = 0
            mean = np.random.normal(loc = reward_model.mean.detach().cpu().numpy(), scale=np.exp(0.5*reward_model.log_var.detach().cpu().numpy()))
            logvar = np.zeros_like(reward_model.mean.detach().cpu().numpy())

        t = 0
        def obs_fn(observation, mean, logvar, action=None, reward=0):
            nonlocal t
            if FLAGS.access_to_mode:
                shape = list(observation.shape)
                shape[-1] = gym_env.get_num_modes()
                human_pref = np.zeros(shape=shape)
                human_pref[..., mode_n] = 1
                observation = np.concatenate((observation, human_pref), -1)
            return observation, mean

        # for ep in range(FLAGS.eval_episodes):
        eval_info = evaluate(
            policy_fn,
            gym_env,
            num_episodes=1,
            save_video=False,
            name="video",
            obs_fn=partial(obs_fn, mean=mean, logvar=logvar),
        )

        print("pref: ", gym_env.pref_list[mode_n] if latent_sample_type=='posterior' else None, 
              "evaluation latent: ", mean, "rev: ", eval_info["rew_vec"][0], '\n', flush=True)

        primary_dim = np.argmax(reward_model.log_var.detach().cpu().numpy())
        obs_list.append(eval_info['obs_list'][0])
        env_pref_list.append(eval_info['env_pref'][0] if latent_sample_type!='prior' else [mean[primary_dim], 0])
        rew_vec_list.append(eval_info["rew_vec"][0])
        utility.append(np.sum(eval_info["rew_vec"][0]*eval_info['env_pref'][0]))
        cost.append(eval_info['cost'][0])

    os.makedirs("logs/fig", exist_ok=True)
    # if gym_env.get_num_modes()==2:
    #     putils.plot_evaluated_traj(gym_env, obs_list, rew_vec_list, env_pref_list, utility, f'logs/fig/{FLAGS.comment}_{gym_env.spec.id}_{FLAGS.seed}_{latent_sample_type}.png')
    #wandb.log({f"{latent_sample_type}_rewards": np.array(utility).flatten().mean()})
    print(f'{latent_sample_type} rewards: ', utility, '  cost:', cost)
    res = {'utility': utility, 'cost': cost}
    return res
