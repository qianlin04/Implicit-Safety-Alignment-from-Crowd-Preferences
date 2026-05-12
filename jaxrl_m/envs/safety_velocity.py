import math
import gymnasium as gym 
import numpy as np
from gymnasium.spaces.box import Box
import bullet_safety_gym
from jaxrl_m.envs.multimodal_base import MultiModalBase
import os
import h5py
import d4rl
import safety_gymnasium

class SafetyVelocity(gym.Env, MultiModalBase):
    def __init__(self, 
                 mode=-1, 
                 goal_in_state=False,
                 downstream=False ,
                 agent='Ant',
                 render_mode=None):
        
        gym.Env.__init__(self)
        MultiModalBase.__init__(self, mode, downstream, goal_in_state)
        self.agent_name = agent
        self.env = gym.make(
            f'Safety{agent}VelocityGymnasium-v1',render_mode=render_mode
        )
        
        self.full_observation_space = Box(low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[0]+2,), dtype=np.float64)
        if goal_in_state:
            self.observation_space = self.full_observation_space
        else:
            self.observation_space = self.env.observation_space 
        self.action_space = self.env.action_space

        self.mode = mode
        if downstream:
            assert goal_in_state==True

        if self.downstream or self.goal_in_state:
            self.set_downtream_goal()
        else:
            self.set_crowd_goal()
        
        #self.relabel_offline_reward = True
        self._max_episode_steps = self.env._max_episode_steps
        self.is_multimodal = mode < 0
        self.biased_mode = None
        self.cost_penalty = -10.0
        self.info_list = ['cost', 'agent_specific_reward', 'x_velocity', 'y_velocity']

        self.velocity_threshold_dict = {'Ant':2.6222, "HalfCheetah":3.2096, "Hopper":0.7402, "Swimmer":1.5, "Walker2d":2.3415}
        

    def get_dataset(self, path=None, remove_goal_from_dataset=True):
        if path is None or path == '':
            path = os.path.join('data', f'{self.env.spec.id}.data'.replace("Gymnasium-v1", "-multimodal-v0"))
        assert os.path.exists(path), "Dataset file doesn't exist"
        dataset = {}
        with h5py.File(path, 'r') as f:
            assert "infos" in f
            for key, item in f.items():
                if key != "infos":
                    dataset[key] = item[:]
                    if key == "observations" and remove_goal_from_dataset:
                        goals = dataset[key][..., -2:]
                        goal_len = 1 if np.isscalar(self.goals[0]) else len(self.goals[0])
                        dataset['infos/goals'] = dataset[key][..., -goal_len:]
                        dataset[key] = self.remove_goal_from_state(dataset[key])
                else:
                    info_group = f["infos"]
                    for key, item in info_group.items():
                        if key != 'rew_vec':
                            dataset[f"infos/{key}"] = item[:]
                            if key == "cost":
                                dataset["costs"] = item[:]

        #temp
        # if self.agent_name=='HalfCheetah':  
        #     idx = (goals[..., 0] == 1)
        #     print(np.sum(idx), np.sum(1-idx))
        #     for x in dataset:
        #         dataset[x] = dataset[x][idx]

        #temp 
        # if self.agent_name == 'Ant' or self.agent_name =='HalfCheetah':
        #     min_total_reward=2000
        #     max_step = 500
        #     n_traj = 0
        #     filtered = {k: [] for k in dataset.keys()}
        #     episode_data = {k: [] for k in dataset.keys()}
        #     total_reward = 0.0
        #     vel = []

        #     for i in range(len(dataset['actions'])):
        #         for k in dataset.keys():
        #             episode_data[k].append(dataset[k][i])
        #         total_reward += dataset['rewards'][i]
        #         v = np.sqrt(dataset['infos/x_velocity'][i]**2+dataset['infos/y_velocity'][i]**2)
        #         vel.append(v)

        #         if dataset['terminals'][i] or dataset['timeouts'][i]:
        #             #if np.abs(np.mean(vel))<3.2: #total_reward >= min_total_reward:
        #             n_traj += 1
        #             for k in dataset.keys():
        #                 filtered[k].extend(episode_data[k][:max_step])
                    
        #             episode_data = {k: [] for k in dataset.keys()}
        #             total_reward = 0.0
        #             vel = []
        #     for k in filtered.keys():
        #         dataset[k] = np.array(filtered[k])
        #     print(f'Exclude the traj with reward < {min_total_reward}, traj num: {n_traj}, len: {len(dataset["rewards"])}')

        return dataset

    def reset(self ,**kwargs):
        self.vel_history, self.theta_history = [], []  #temp
        if not self.use_crowd_goal:
            self.mode = self.sample_mode() 
            
        obs, info = self.env.reset()
        if self.goal_in_state:
            obs = self.add_goal_into_state(obs, self.mode)
        return obs, info

    def step(self, action):
        # Compute shaped reward

        #action += np.random.normal(0, 0.1, size=action.shape) #add randomness

        xy_position_before = self.env.data.qpos[0:2].copy() if self.agent_name!='Ant' else self.env.get_body_com('torso')[:2].copy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        xy_position_after = self.env.data.qpos[0:2].copy() if self.agent_name!='Ant' else self.env.get_body_com('torso')[:2].copy()

        info['x_velocity'], info['y_velocity'] = (xy_position_after - xy_position_before) / self.env.dt

        healthy_reward = self.env.healthy_reward if self.agent_name!='HalfCheetah' and self.agent_name!='Swimmer' else 0
        ctrl_cost = self.env.control_cost(action)
        info['agent_specific_reward'] = healthy_reward - ctrl_cost
        info["actual_reward"] = reward
        info["pos"] = xy_position_after
        velocity = np.array([info['x_velocity'], info['y_velocity']])
        forward_reward = np.dot(velocity, self.goals[self.mode])  
        reward = forward_reward + info['agent_specific_reward'] 

        info["cost"] = float(np.linalg.norm(velocity, axis=-1) > self.velocity_threshold_dict[self.agent_name])

        if self.goal_in_state:
            obs = self.add_goal_into_state(obs, self.mode)

        rew_vec = []
        obs_cat = np.vstack([obs])
        info_cat = {
            'x_velocity': np.array([info['x_velocity']]), 
            'y_velocity': np.array([info['y_velocity']]), 
            'agent_specific_reward': np.array([info['agent_specific_reward']]), 
            'cost': np.array([info['cost']])}
        for mode in range(0, len(self.goals)):
            rew_vec.append(self.get_r(obs_cat, mode, info_cat, add_cost_to_reward=False)[0])
        info["rew_vec"] = np.array(rew_vec)

        self.theta_history.append(np.dot(velocity, self.goals[self.mode]) / (np.linalg.norm(self.goals[self.mode]) * np.linalg.norm(velocity)))
        self.vel_history.append(np.linalg.norm(velocity))
        if (terminated or truncated):  #temp
            info['comment'] = f"target: {self.target}, final cos_sim: {self.theta_history[-1]}, avg cos_sim: {np.mean(self.theta_history)}, {self.theta_history[::20]}\n final vel: {self.vel_history[-1]}, avg vel: {np.mean(self.vel_history)}, {self.vel_history[::20]}"
            if not self.collecting_data:
                print(info['comment'])
        
        return obs, reward, terminated, truncated, info
    
    def get_r(self, obs, mode, info={}, add_cost_to_reward=True):
        for k in info:
            if info[k].ndim > 1 and info[k].shape[-1] == 1:
                info[k] = np.squeeze(info[k], axis=-1)
        
        # velocity = np.stack([info['x_velocity'], info['y_velocity']], axis=-1)
        # forward_reward = np.sum(velocity*self.goals[mode], axis=-1)
        forward_reward = self.goals[mode][0]*info['x_velocity'] + self.goals[mode][1]*info['y_velocity']
        reward = forward_reward + info['agent_specific_reward']
        if add_cost_to_reward and 'cost' in info: 
            reward += info['cost'] * self.cost_penalty  #add cost into reward)
        return reward
    

    def render(self, mode="rgb_array"):
        return self.env.render()
    
    @property
    def target(self):
        return self.goals[self.mode] 
    
    def set_mode(self, mode):
        return super().set_mode(mode)
    

    def set_crowd_goal(self):
        if self.agent_name=='HalfCheetah':
            self.goals = np.array([[1,0], [-1,0]])
        else:
            self.goals = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        self.pref_list = np.eye(len(self.goals))
        super().set_crowd_goal()
    
    def set_downtream_goal(self):
        if self.agent_name=='HalfCheetah':
            self.goals = np.array([[1,0], [-1,0]]) 
        else:
            self.goals = np.array([[np.cos(theta), np.sin(theta)] for theta in np.arange(0, 2*np.pi, 2*np.pi/40)])
        self.pref_list = np.eye(len(self.goals))
        super().set_downtream_goal()