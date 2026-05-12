import numpy as np
import d4rl
from jaxrl_m.learners.d4rl_utils import expand_dataset, new_get_trj_idx
from tqdm import tqdm


class MultiModalBase():
    """
    A base class for environments with multiple modes/preferences.
    """

    def __init__(self, mode=-1, downstream=False, goal_in_state=False):
        super().__init__()
        self.mode = mode
        self.is_multimodal = mode < 0
        self.biased_mode = None

        self.goals = None
        self.pref_list = None  
        self.info_list = [] # Neccessary info for calculating reward
        self.cost_penalty = 0.0
        self.downstream = downstream
        self.use_crowd_goal = False
        self.info_list = []
        self.goal_in_state = goal_in_state
        self.collecting_data = not self.downstream and self.goal_in_state

    def get_num_modes(self):
        if self.is_multimodal:
            return len(self.goals)
        return 1

    def sample_mode(self):
        if self.is_multimodal:
            return np.random.randint(len(self.goals))
        return self.mode

    def reset_mode(self):
        self.set_mode(self.sample_mode())

    def set_mode(self, mode):
        if self.is_multimodal:
            self.mode = mode

    def get_pref(self):
        return self.pref_list[self.mode]

    def set_biased_mode(self, mode):
        self.biased_mode = mode

    def get_r(self, obs, mode, info={}, add_cost_to_reward=True):
        raise NotImplementedError
    
    def get_done(self, obs, mode, info={}, terminal=None):
        return np.zeros((len(obs),)) if terminal is None else terminal
    
    def set_crowd_goal(self, ):
        self.use_crowd_goal = True
    
    def set_downtream_goal(self, ):
        self.use_crowd_goal = False

    def add_goal_into_state(self, obs, mode):
        goal = self.goals[mode]
        if goal.ndim == 0:
            goal = goal[None]
        goal_shape = (1,) * (obs.ndim - 1) + goal.shape
        goal = np.reshape(goal, goal_shape)
        obs = np.concatenate([obs, np.broadcast_to(goal, obs.shape[:-1] + goal.shape[-1:])], axis=-1)
        return obs
    
    def remove_goal_from_state(self, obs):
        goal = self.goals[0]
        if goal.ndim == 0:
            goal = goal[None]
        return obs[..., :-len(goal)]
    
    def get_dataset(self, dir='', remove_goal_from_dataset=True):
        raise NotImplementedError
    
    def get_dataset_for_downstream(self, qlearning_dataset=True): # relabel the reward for downstream tasks
        dataset = self.get_dataset(remove_goal_from_dataset=False)
        # traj_done = np.logical_or(dataset['terminals'], dataset['timeouts']) if 'timeouts' in dataset else dataset['terminals']
        # obs_list, r_list = [], []

        # start = 0
        # for end in np.where(traj_done)[0]:
        #     obs = dataset["observations"][start : end + 1]
        #     mode = self.sample_mode()
        #     info_set = {k.replace("infos/", ""):dataset[k][start : end + 1] for k in dataset if 'infos/' in k}
        #     obs = self.add_goal_into_state(obs, mode)
        #     rew = self.get_r(obs, mode, info_set, add_cost_to_reward=False)

        #     obs_list.append(obs)
        #     r_list.append(rew)
        #     start = end + 1

        # dataset['observations'] = np.concatenate(obs_list, 0)
        # dataset['rewards'] = np.concatenate(r_list, 0)

        if qlearning_dataset:
            dataset = d4rl.qlearning_dataset(self, dataset=dataset)

        return dataset
    