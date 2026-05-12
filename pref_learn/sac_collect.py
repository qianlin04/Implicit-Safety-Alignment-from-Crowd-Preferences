from jaxrl_m.envs import make_env
import pybullet_envs
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import jaxrl_m.envs
import os 
import d4rl
import h5py
import wandb
from algo.sac_agent import SacAgent, Actor, Critic, ReplayBuffer


def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t

def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.float), device)

def learn(device, env_name, log, seed, args, max_step=5e6, penalty_step=1, penalty_mode='linear'):
    env = make_env(env_name)
    max_cost_penalty = getattr(env, 'cost_penalty', 0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SacAgent(state_dim, action_dim, device=device, max_step=max_step, gamma=args.gamma)

    if args.load_model_path is not None and args.load_model_path!="":
        model_para = torch.load(args.load_model_path)
        agent.actor.load_state_dict(model_para['actor'])
        agent.target_critic1.load_state_dict(model_para['critic1'])
        agent.target_critic2.load_state_dict(model_para['critic2'])
        agent.critic1.load_state_dict(model_para['critic1'])
        agent.critic2.load_state_dict(model_para['critic2'])

    if args.load_offline_dataset is not None:
        if args.reset_by_mode:
            env.set_crowd_goal()
        agent.buffer.load_offline_dataset(env, args.load_offline_dataset)
        # for _ in range(100000):
        #     info = agent.learn() 

    total_frames = 0
    info = {}
    while total_frames < max_step: #temp0914

        if args.reset_by_mode:
            env.set_crowd_goal()
            env.reset_mode()

        state = env.reset()
        done = False
        while 1:
            total_frames += 1
            action = agent.act(state)            
            next_state, reward, done, info = env.step(action)
            agent.remember(state, next_state, action, reward, done and not 'TimeLimit.truncated' in info, info)
            state = next_state
            if agent.buffer.size >= 100000:  
                update_step = 100000 if args.load_offline_dataset and total_frames<=1 and args.load_model_path is not None else 1 #temp
                for us in range(update_step):
                    batch = agent.buffer.sample_batch(batch_size=agent.batch_size)

                    # if total_frames%1000==0 and 'Reach' in env_name:  #temp
                    #     violate = np.any(batch['sta1'][:, 7:31]>0.7, axis=-1)
                    #     print(np.where(violate), np.where(batch['cost'][:, 0]), np.sum(violate!=batch['cost'][:, 0]))
                    #     print(violate.shape, batch['cost'][:, 0].shape)
                    #     print("n_done: ", np.sum(agent.buffer.done_buf))

                    si = np_to_tensor(batch['sta1'], device)
                    sn = np_to_tensor(batch['sta2'], device)
                    ai = np_to_tensor(batch['acts'], device)
                    ri = np_to_tensor(batch['rews'], device)
                    di = np_to_tensor(batch['done'], device)
                    ci = np_to_tensor(batch['cost'], device)
                    
                    if penalty_mode=='linear':
                        cost_penalty = min(total_frames/penalty_step, 1) * max_cost_penalty
                    elif penalty_mode=='switch':
                        cost_penalty = (total_frames>=penalty_step) * max_cost_penalty
                    ri = ri + ci * cost_penalty   #cosider cost during traininig

                    info = agent.learn((si, ai, ri, sn, di), update_step==1) 

                    if update_step > 1:
                        if us%1000==0:
                            info['step'] = total_frames
                            #print(info)
                            wandb.log(info)

                

            if total_frames % 1e5 == 0:
                rs, obs_list, goal_list, cost_list, pref_list = [], [], [], [], []
                eval_env= make_env(env_name)
                for i in range(100):
                    rew, traj, goal, cost = test(eval_env, agent, args)
                    rs.append(rew)
                    cost_list.append(cost)
                    obs_list.append(np.array(traj))
                    goal_list.append(np.array(goal))
                    pref_list.append(eval_env.get_pref())
                rs, goal_list = np.array(rs), np.array(goal_list)
                cost_list, pref_list = np.array(cost_list), np.array(pref_list)
                rew_vec_list = np.zeros((len(rs), eval_env.get_num_modes()))
                os.makedirs("logs/fig", exist_ok=True)
                #plot_evaluated_traj(env, obs_list, rew_vec_list, pref_list, rs, path=f'logs/fig/sac_{env_name}_{seed}.png')
                info['rew'] = np.mean(rs)
                info['cost'] = np.mean(cost_list)
                info['step'] = total_frames
                os.makedirs(log, exist_ok=True)
                torch.save({'actor': agent.actor.state_dict(), 'critic1': agent.target_critic1.state_dict(), 'critic2': agent.target_critic2.state_dict()}, os.path.join(log, f'sac_agent.pth'))
            if total_frames % 5000 == 0:
                n_timeout = np.sum([1 for d in agent.buffer.info_buf if 'TimeLimit.truncated' in d and d['TimeLimit.truncated']])
                n_done = np.sum(agent.buffer.done_buf)
                info['n_timeout'] = n_timeout        
                info['n_done'] = n_done     
                info['t'] = total_frames
                #print("test results: ", info)
                wandb.log(info)
            if args.path_to_save_dataset is not None and total_frames % 10000==0: 
                # Save to HDF5
                with h5py.File(args.path_to_save_dataset, "w") as f:
                    f.create_dataset("observations", data=agent.buffer.sta1_buf)
                    f.create_dataset("next_observations", data=agent.buffer.sta2_buf)
                    f.create_dataset("actions", data=agent.buffer.acts_buf)
                    f.create_dataset("rewards", data=agent.buffer.rews_buf.flatten())
                    f.create_dataset("terminals", data=agent.buffer.done_buf.flatten())
                    info_dict = agent.buffer.trans_info_to_numpy(env.info_list)
                    if hasattr(env, 'info_list'):
                        info_group = f.create_group("infos")
                        for key in env.info_list:
                            if key in info_dict:                        
                                info_group.create_dataset(key, data=info_dict[key])

                    f.create_dataset("timeouts", data=info_dict['TimeLimit.truncated'].flatten())
                print(f"Dataset saved to {args.path_to_save_dataset}, size: {agent.buffer.size}")
                    
            if done:
                break
        


def test(env, agent, args):
    if args.reset_by_mode:
        env.set_crowd_goal()
        env.reset_mode()
    state = env.reset()
    total_reward = 0
    cost = 0
    traj = []
    while 1:
        traj.append(state[:2])
        action = agent.act(state, mean=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        cost += info.get('cost', 0)
        if done:
            break
    print(f"env.target: {env.target}, reward: {total_reward}, cost: {cost}, rew_vev: {info['rew_vec'][::10]}")
    # if 'comment' in info:
    #     print(info['comment'])
    return total_reward, traj, env.target, cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--max_step', type=int, default=5e6)
    parser.add_argument('--penalty_step', type=int, default=1)
    parser.add_argument('--penalty_mode', type=str, default='switch')  #linear or switch
    parser.add_argument('--load_offline_dataset', type=str, default=None)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--path_to_save_dataset', type=str, default=None)
    parser.add_argument('--reset_by_mode', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('-e', '--env', type=str, default='maze2d-twogoals-multimodal-with-goal-v0')
    parser.add_argument('-l', '--log', type=str, default='pref_datasets/maze2d-twogoals-multimodal-v0')
    

    args = parser.parse_args()
    wandb.init(
        project="vpl_rlhf",
        config=args,
        name=f'{args.env}',
        group="sac_oracle",
    )
    learn(device=args.gpu, env_name=args.env, log=args.log, seed=args.seed, 
          args=args, max_step=args.max_step, penalty_step=args.penalty_step, penalty_mode=args.penalty_mode)