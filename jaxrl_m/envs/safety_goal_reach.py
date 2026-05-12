import math
import gymnasium as gym 
import numpy as np
from gymnasium.spaces.box import Box
import bullet_safety_gym
from jaxrl_m.envs.multimodal_base import MultiModalBase
import os
import h5py
import d4rl

GOAL_RADIUS = 0.3

class SafetyBallReach(gym.Env, MultiModalBase):
    def __init__(self, 
                 mode=-1, 
                 goal_in_state=False, 
                 downstream=False,
                 fix_obstacle=False):  #temp
        
        gym.Env.__init__(self)
        MultiModalBase.__init__(self, mode, downstream, goal_in_state)
        self.env = gym.make(
            'SafetyBallReach-v0',
            max_episode_steps=80, 
            agent='Ball',
                task='ReachGoalTask',
                obstacles={
                    'Puddle': {
                        'number': 8 if not fix_obstacle else 5,
                        'fixed_base': True,
                        'movement': 'static'
                    },
                },
                world={
                    'name': 'SmallRoom',
                    'factor': 1
                },
        )
        self.env.task.continue_after_goal_achievement = False
        if not downstream and goal_in_state:
            self.env.task.goal.radius = GOAL_RADIUS
        if fix_obstacle:
            self.env.task.obstacles_reset_case = 1
        
        self.full_observation_space = self.env.observation_space 
        if goal_in_state:
            self.observation_space = self.env.observation_space 
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[0]-2,), dtype=np.float64)
        self.action_space = self.env.action_space
        
        self._max_episode_steps = self.env._max_episode_steps
        self.mode = mode

        if downstream or goal_in_state:
            self.set_downtream_goal()
        else:
            self.set_crowd_goal()
        
        #self.relabel_offline_reward = True
        self.is_multimodal = mode < 0
        self.biased_mode = None
        self.cost_penalty = -10.0   #temp
        self.fix_obstacle = fix_obstacle
        self.info_list = ['cost', 'agent_specific_reward', 'rew_vec']
        
        
    # @property
    # def target(self):
    #     return self.env.unwrapped._target

    def get_dataset(self, path=None, remove_goal_from_dataset=True ):
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
        
        if remove_goal_from_dataset:  #temp
            dataset['timeouts'] = np.logical_or(dataset['timeouts'], dataset['terminals'])
            dataset['terminals'] = np.zeros_like(dataset['terminals'])
        return dataset
    
    def get_dataset_for_downstream(self, qlearning_dataset=True):
        dataset = self.get_dataset(remove_goal_from_dataset=False)
        if qlearning_dataset:
            dataset = d4rl.qlearning_dataset(self, dataset=dataset)
        return dataset

    def reset(self, **kwargs):
        self.env.task.continue_after_goal_achievement = False
        if self.collecting_data:
            self.env.task.goal.radius = GOAL_RADIUS
        
        if self.fix_obstacle:
            self.env.task.obstacles_reset_case = 1
        
        if self.use_crowd_goal:
            goal = self.goals[self.mode]
            self.env.task.goal_fixed_position = np.array([goal[0], goal[1], 0.0])

        obs, info = self.env.reset()

        self.start_dis = np.linalg.norm(obs[-2:]*20) #temp


        if not self.goal_in_state:
            obs = obs[..., :-2]
        self.old_obs = obs
        return obs, info
    
    def set_training_goals(self):
        self.goals = self.train_goals
        self.qmatrixes = self.train_qmatrixes
    
    def set_eval_goals(self):
        self.goals = self.test_goals
        self.qmatrixes = self.test_qmatrixes

    def step(self, action):
        # Compute shaped reward

        old_obs = self.env.get_observation()

        obs, reward, terminated, truncated, info = self.env.step(action)

        info['cost'] = float(np.any(old_obs[7:31]>0.7)) #temp
        if not self.downstream:
            reward -= 0.5  #temp

        info['agent_specific_reward'] = self.env.agent.specific_reward()
        if truncated:
            info['TimeLimit.truncated'] = True

        if (terminated or truncated) and not self.collecting_data:
            self.end_dis = np.linalg.norm(obs[-2:]*20) #temp
            print('done' if terminated else 'timeout', f"   ,  start dis: {self.start_dis}, end dis: {self.end_dis}, relative: {obs[-2:]*20}")  


        if not self.goal_in_state:
            obs = obs[..., :-2]

        rew_vec = []
        obs_cat = np.vstack([self.old_obs, obs])
        info_cat = {
            'agent_specific_reward': np.array([info['agent_specific_reward']]*2), 
            'cost': np.array([info['cost']]*2)
        }
        for mode in range(0, len(self.goals)):
            rew_vec.append(self.get_r(obs_cat, mode, info_cat, add_cost_to_reward=False)[0])
        info["rew_vec"] = np.array(rew_vec)

        # Override environment termination
        info["actual_reward"] = reward
        self.old_obs = obs

        import absl.flags, time  #temp
        FLAGS = absl.flags.FLAGS
        if hasattr(FLAGS, 'test_only') and FLAGS.test_only:
            time.sleep(0.1)

        return obs, reward, terminated, truncated, info
    
    def get_r(self, obs, mode, info=[], add_cost_to_reward=True):
        for k in info:
            if info[k].ndim > 1 and info[k].shape[-1] == 1:
                info[k] = np.squeeze(info[k], axis=-1)

        agent_position = obs[..., :2] * 10
        goal_position = self.goals[mode]
        cur_dist = np.linalg.norm(agent_position-goal_position, axis=-1)
        rewards = 3 * (
                cur_dist[:-1] - cur_dist[1:]
            ) + 0.01 * info['agent_specific_reward'][:-1]
    
        if not self.downstream:
            rewards -= 0.5  #temp

        if add_cost_to_reward and 'cost' in info: 
            rewards += info['cost'][:-1] * self.cost_penalty  #add cost into reward
        rewards = np.concatenate([rewards, rewards[..., -1:]], axis=-1)
        return rewards
    
    def get_done(self, obs, mode, info=[], terminal=None):
        agent_position = obs[..., :2] * 10
        goal_position = self.goals[mode]
        cur_dist = np.linalg.norm(agent_position-goal_position, axis=-1)
        done = np.zeros((len(obs)))
        for t in range(0, len(obs)-1):
            if cur_dist[t+1]<GOAL_RADIUS:
                done[t] = 1
        return done
    

    def render(self, mode="rgb_array"):
        return self.env.render()
    
    @property
    def target(self):
        return self.env.task.goal.get_position()

    def add_goal_into_state(self, obs, mode):
        goal = self.goals[mode]
        goal_shape = (1,) * (obs.ndim - 1) + goal.shape
        goal = np.reshape(goal, goal_shape)
        goal = np.broadcast_to(goal, obs.shape[:-1] + goal.shape[-1:])
        agent_position = obs[..., :2] * 10 
        delta = (goal - agent_position) / (2*self.env.world.env_dim)
        obs = np.concatenate([obs, delta], axis=-1)
        return obs

    def set_crowd_goal(self):
        self.goals = np.array(np.meshgrid([-8.0, 8.0], [-8.0, 8.0])).T.reshape(-1, 2) #temp
        #self.goals = np.array([[-6, -6], [6,6,]])  #temp
        self.pref_list = np.eye(len(self.goals))
        super().set_crowd_goal()

    def set_downtream_goal(self):
        self.goals = np.array(np.meshgrid([-8.0, 8.0], [-8.0, 8.0])).T.reshape(-1, 2) #temp
        self.pref_list = np.eye(len(self.goals))
        super().set_downtream_goal()
    