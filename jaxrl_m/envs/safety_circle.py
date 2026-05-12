import math
import gymnasium as gym 
import numpy as np
from gymnasium.spaces.box import Box
import bullet_safety_gym
from jaxrl_m.envs.multimodal_base import MultiModalBase
import os
import h5py
import d4rl

class SafetyCircle(gym.Env, MultiModalBase):
    def __init__(self, 
                 mode=-1, 
                 goal_in_state=False,
                 downstream=False , ):
        
        gym.Env.__init__(self)
        MultiModalBase.__init__(self, mode, downstream, goal_in_state)
        self.agent_name = 'Ball'
        self.env = gym.make(
            'SafetyBallCircle-v0',
            agent=self.agent_name,
        )

        self.full_observation_space = Box(low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float64)
        if goal_in_state:
            self.observation_space = self.full_observation_space
        else:
            self.observation_space = self.env.observation_space 
        self.action_space = self.env.action_space
        self.goal_in_state = goal_in_state

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
        self.fixed_goal = (mode != -1)
        self.cost_penalty = -10.0
        self.info_list = ['cost', 'agent_specific_reward', 'rew_vec', 'vel', 'pos', '']
        

    def get_dataset(self, path=None, remove_goal_from_dataset=True):
        if path is None or path == '':
            path = os.path.join('data', f'{self.env.spec.id}.data'.replace("-v0", "-multimodal-v0"))
        assert os.path.exists(path), "Dataset file doesn't exist"
        dataset = {}
        with h5py.File(path, 'r') as f:
            assert "infos" in f
            for key, item in f.items():
                if key != "infos":
                    dataset[key] = item[:]
                    if key == "observations" and remove_goal_from_dataset:
                        goal_len = 1 if np.isscalar(self.goals[0]) else len(self.goals[0])
                        dataset['infos/goals'] = dataset[key][..., -goal_len:]
                        dataset[key] = self.remove_goal_from_state(dataset[key])
                else:
                    info_group = f["infos"]
                    for key, item in info_group.items():
                        dataset[f"infos/{key}"] = item[:]
                        if key == "cost":
                            dataset["costs"] = item[:]
        return dataset
    
    def get_dataset_for_downstream(self, qlearning_dataset=True):  
        dataset = self.get_dataset(remove_goal_from_dataset=False)
        if qlearning_dataset:
            dataset = d4rl.qlearning_dataset(self, dataset=dataset)
        return dataset

    def reset(self, **kwargs):
        self.vel_history, self.dist_history = [], []  #temp
        if self.use_crowd_goal:
            pass
            #assert self.mode!=-1, 'set the crowd mode before env.reset()'
        else:
            self.mode = self.sample_mode() 
            
        obs, info = self.env.reset()
        if self.goal_in_state:
            obs = self.add_goal_into_state(obs, self.mode)
        self.old_obs = obs
        self.t_step = 0
        return obs, info

    def step(self, action):
        # Compute shaped reward

        #action += np.random.normal(0, 0.1, size=action.shape) #add randomness

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info['agent_specific_reward'] = self.env.agent.specific_reward()
        vel = self.env.agent.get_linear_velocity()[:2]
        pos = self.agent.get_position()[:2]
        info['pos'] = pos
        info['vel'] = vel
        info["actual_reward"] = reward
        info["cost"] = info.get("cost_outside_bounds", 0)

        dist = np.linalg.norm(pos)
        tangent_dir = np.array([pos[1], -pos[0]]) 
        tangent_dir /= (np.linalg.norm(tangent_dir) + 1e-8)
        v_tangent = np.dot(vel, tangent_dir)
        reward = np.exp(-0.5 * np.abs(v_tangent - self.goals[self.mode]))  / (1 + np.abs(dist - self.env.task.circle.radius))
        reward += 0.01 * info['agent_specific_reward']

        if truncated:
            info['TimeLimit.truncated'] = True
        if self.goal_in_state:
            obs = self.add_goal_into_state(obs, self.mode)

        rew_vec = []
        obs_cat = np.vstack([self.old_obs])
        info_cat = {
            'vel': np.array([info['vel']]), 
            'pos': np.array([info['pos']]), 
            'agent_specific_reward': np.array([info['agent_specific_reward']]), 
            'cost': np.array([info['cost']])}
        for mode in range(0, len(self.goals)):
            rew_vec.append(self.get_r(obs_cat, mode, info_cat, add_cost_to_reward=False)[0])
        info["rew_vec"] = np.array(rew_vec)
        info["v_tangent"] = v_tangent
        self.vel_history.append(v_tangent)
        self.dist_history.append(dist)

        if (terminated or truncated):
            info['comment'] = f"final vel: {self.vel_history[-1]}, avg vel: {np.mean(self.vel_history)}, history: {self.vel_history}\n"
            info['comment'] += f"final dist: {self.dist_history[-1]}, avg dist: {np.mean(self.dist_history)}, history: {self.dist_history}\n"
            if not self.collecting_data:
                print(info['comment'])
        
        self.old_obs = obs
        self.t_step += 1
        if self.collecting_data and not self.use_crowd_goal:
            # during data collection, we randomly switch the mode at the middle of trajectory with prob p, which will increase the overlap between different tasks
            if self.t_step == self.env._max_episode_steps // 2 and np.random.uniform(0,1)<0.5:
                self.mode = self.sample_mode() 

        return obs, reward, terminated, truncated, info
    
    def get_r(self, obs, mode, info={}, add_cost_to_reward=True):
        for k in info:
            if info[k].ndim > 1 and info[k].shape[-1] == 1:
                info[k] = np.squeeze(info[k], axis=-1)

        dist = np.linalg.norm(info['pos'], axis=-1)
        tangent_dir = np.stack([info['pos'][...,1], -info['pos'][...,0]], axis=-1) 
        tangent_dir /= (np.linalg.norm(tangent_dir, axis=-1, keepdims=1) + 1e-8)
        v_tangent = np.sum(info['vel'] * tangent_dir, axis=-1)
        reward = np.exp(-0.5 * np.abs(v_tangent - self.goals[mode]))  / (1 + np.abs(dist - self.env.task.circle.radius))
        reward += 0.01 * info['agent_specific_reward']
        if add_cost_to_reward and 'cost' in info: 
            reward += info['cost'] * self.cost_penalty  #add cost into reward)
        return reward
    

    def render(self, mode="rgb_array"):
        return self.env.render()
    
    @property
    def target(self):
        return self.goals[self.mode] 
    
    def set_mode(self, mode):
        self.fixed_goal = (mode != -1)
        return super().set_mode(mode)
    
    @property
    def agent(self):
        return self.env.agent

    def set_crowd_goal(self):
        if self.agent_name == 'Ball':
            self.goals = np.array([0.1, 6.0])
        else:
            self.goals = np.array([0.1, 3.0])
        self.pref_list = np.eye(len(self.goals))
        super().set_crowd_goal()
    
    def set_downtream_goal(self):
        if self.agent_name == 'Ball':
            self.goals = np.arange(0.0, 6, 0.1)
        else:
            self.goals = np.arange(0.0, 3.0, 0.1)
        self.pref_list = np.eye(len(self.goals))
        super().set_downtream_goal()
