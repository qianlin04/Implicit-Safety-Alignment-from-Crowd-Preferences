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
import copy
import torch.nn.functional as F



def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t

def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.float), device)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        n_neurons = 256
        self.fc = nn.Sequential(
            nn.Linear(state_dim, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU())
        self.mu = nn.Linear(n_neurons, action_dim)
        self.log_std = nn.Linear(n_neurons, action_dim)
        self.max_action = max_action

    def forward(self, s, mean=False, with_prob=True):
        x = self.fc(s)
        mu = self.mu(x)
        if mean:
            return self.max_action * torch.tanh(mu) if self.max_action!=-1 else mu
        else:
            std = torch.clamp(self.log_std(x), -20, 2).exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.rsample()
            real_action = self.max_action * torch.tanh(action) if self.max_action!=-1 else action
            if with_prob:
                log_prob = dist.log_prob(action).sum(1, keepdim=True)
                real_log_prob = log_prob - 2 * (np.log(2) - action - nn.Softplus()(-2*action)).sum(1, keepdim=True)
                return real_action, real_log_prob
            else:
                return real_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_neurons = 256
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, 1))

    def forward(self, s, a):
        q = self.fc(torch.cat((s, a), dim=-1))
        return q


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size):
        self.state_dim, self.action_dim, self.size = state_dim, action_dim, size
        self.init_array(size)

    def init_array(self, size):
        print(f'dataset size: {size}')
        self.sta1_buf = np.zeros([size, self.state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, self.state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, self.action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.float32)
        self.info_buf = []
        self.ptr, self.size, self.max_size = 0, 0, size    

    def store(self, sta, next_sta, act, rew, done, info):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if len(self.info_buf) < self.max_size:
            self.info_buf.append(info)
        else:
            self.info_buf[self.ptr] = info
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_batch(self, sta, next_sta, act, rew, done, info_set={}):
        batch_size = len(sta)
        self.sta1_buf[self.ptr:self.ptr+batch_size] = sta
        self.sta2_buf[self.ptr:self.ptr+batch_size] = next_sta
        self.acts_buf[self.ptr:self.ptr+batch_size] = act
        self.rews_buf[self.ptr:self.ptr+batch_size] = rew
        self.done_buf[self.ptr:self.ptr+batch_size] = done
        self.ptr = (self.ptr+batch_size) % self.max_size
        for i in range(batch_size):
            self.info_buf.append({k:info_set[k][i] for k in info_set})
        self.size = min(self.size + batch_size, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        cost = np.array([self.info_buf[idx].get('cost', 0) for idx in idxs]).reshape(-1, 1)
        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    cost=cost)
    
    def load_offline_dataset(self, env, data_path):
        start_idx = 0
        dataset = env.get_dataset(path = data_path)
        dataset = d4rl.qlearning_dataset(env, dataset=dataset)
        timeouts = np.logical_and(dataset['traj_done'], 1-dataset['terminals'])
        self.init_array(len(dataset['observations'])*len(env.goals)+self.max_size)

        for end_idx in np.where(dataset['traj_done'])[0]:
            info_set = {k.replace('infos/', ''): dataset[k][start_idx:end_idx+1] for k in dataset if 'infos/' in k}
            info_set['TimeLimit.truncated'] = timeouts[start_idx:end_idx+1]
            if start_idx == 0:
                obs = env.add_goal_into_state(dataset['observations'][start_idx:end_idx+1], 0)
                reward = env.get_r(obs, 0, info_set, add_cost_to_reward=False)
                # print(reward)
                # print(obs, env.get_done(obs, 0, terminal=dataset['terminals'][start_idx:end_idx+1]))
                # print(info_set)
            
            for mode in range(len(env.goals)):    
                obs = env.add_goal_into_state(dataset['observations'][start_idx:end_idx+1], mode)
                next_obs = env.add_goal_into_state(dataset['next_observations'][start_idx:end_idx+1], mode)
                reward = env.get_r(obs, mode, info_set, add_cost_to_reward=False)
                #print(mode, obs.shape, reward.shape)
                if hasattr(env, 'get_done'):
                    done = env.get_done(obs, mode, terminal=dataset['terminals'][start_idx:end_idx+1])
                    #print('num of done: ', np.sum(done))
                else:
                    done = np.zeros((end_idx-start_idx+1,))
                self.store_batch(obs, next_obs, dataset['actions'][start_idx:end_idx+1], reward[:, None], done[:, None], info_set)
            start_idx = end_idx+1

            

    def trans_info_to_numpy(self, env_info_list):
        """
        Each unique key becomes a dataset under group 'info/'.
        Missing keys are filled with 0.
        """
        
        # Collect all unique keys
        all_keys = set()
        for d in self.info_buf:
            all_keys.update(d.keys())
        all_keys = sorted(all_keys)


        # Build array for each key
        data_dict = {}
        for key in all_keys:
            if key in env_info_list or key=='TimeLimit.truncated':
                values = []
                for d in self.info_buf:
                    if key in d:
                        values.append(d[key])
                    else:
                        values.append(0)
                data_dict[key] = np.array(values)

        if 'TimeLimit.truncated' not in data_dict:
            data_dict['TimeLimit.truncated'] = np.zeros((self.size,), dtype=bool)
        return data_dict



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class SacAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 device=0, 
                 policy_dim=None,
                 policy_fn=None,
                 max_step=1e7, 
                 gamma=0.99, 
                 learning_rate=3e-4,
                 max_policy_action=3.0, 
                 max_action=1.0, 
                 latent_reg_para = 0.0,
                 use_deterministic_policy=False,):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_dim = policy_dim if policy_dim is not None else action_dim
        self.policy_fn = policy_fn
        self.max_action = max_action
        self.max_policy_action = max_policy_action
        self.latent_reg_para = latent_reg_para
        self.device = device
        self.use_deterministic_policy = use_deterministic_policy #if use_deterministic_policy, use TD3 update, else use SAC update
        self.learning_rate = learning_rate
        self.actor = opt_cuda(Actor(self.state_dim, self.policy_dim, max_policy_action), self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        self.critic2 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        self.target_critic1 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        self.target_critic2 = opt_cuda(Critic(self.state_dim, self.action_dim), self.device)
        soft_update(self.target_critic1, self.critic1, 1)
        soft_update(self.target_critic2, self.critic2, 1)
        self.optim_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.optim_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.log_temp = opt_cuda(torch.tensor(0.0), self.device)
        self.log_temp.requires_grad = True
        self.optim_log_temp = torch.optim.Adam([self.log_temp], lr=learning_rate)
        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, int(max_step))
        self.gamma = gamma
        self.tau = 0.005
        self.batch_size = 256
        self.step = 0

    def act(self, state, mean=False):
        with torch.no_grad():
            state = np_to_tensor(state, self.device).reshape(1, -1)
            action = self.actor(state, mean=mean, with_prob=False)
        return action.cpu().squeeze().numpy()
    
    def get_action(self, state, eval=False, latent_action=None, random_latent=False, latent_noise_scale=0):
        with torch.no_grad():
            state = np_to_tensor(state, self.device).reshape(1, -1)
            if latent_action is None:
                if random_latent:  
                    #latent_action = opt_cuda((torch.rand(size=(1, self.policy_dim))-0.5)*self.max_policy_action*2, self.device)
                    latent_action = opt_cuda(torch.randn(size=(1, self.policy_dim)), self.device) * self.max_policy_action
                else:
                    latent_action = self.actor(state, mean=eval or self.use_deterministic_policy, with_prob=False)
                    latent_noise = opt_cuda(torch.randn_like(latent_action), self.device) * latent_noise_scale * self.max_policy_action
                    latent_action = torch.clip(latent_action+latent_noise, -self.max_policy_action, self.max_policy_action)
            a = self.policy_fn(state, latent_action) if self.policy_fn is not None else latent_action
        return a.cpu().data.numpy().flatten(), latent_action


    def remember(self, state, next_state, action, reward, done, info):
        self.buffer.store(state.reshape(1, -1), next_state.reshape(1, -1), action, reward, done, info)

    def learn(self, experiences, update_actor=True):

        si, ai, ri, sn, di = experiences

        with torch.no_grad():
            if self.use_deterministic_policy:
                a_next = self.actor_target(sn, mean=True, with_prob=False)
                if self.policy_fn is not None:
                    #a_next += (torch.randn_like(a_next) * 0.2).clamp(-0.5, 0.5) * self.max_policy_action
                    a_next = self.policy_fn(sn, a_next)
                noise = (torch.randn_like(ai) * 0.2).clamp(-0.5, 0.5)
                a_next = (a_next + noise).clamp(-self.max_action, self.max_action)
                min_q = torch.min(self.target_critic1(sn, a_next), self.target_critic2(sn, a_next))
                yi = ri + self.gamma * (1-di) * min_q
            else:
                a_next, log_p_next = self.actor(sn)
                entropy_next = - self.log_temp.exp() * log_p_next
                if self.policy_fn is not None:
                    a_next = self.policy_fn(sn, a_next)
                min_q = torch.min(self.target_critic1(sn, a_next), self.target_critic2(sn, a_next))
                yi = ri + self.gamma * (1-di) * (min_q + entropy_next)
            
        self.optim_critic1.zero_grad()
        self.optim_critic2.zero_grad()
        q1, q2 = self.critic1(si, ai), self.critic2(si, ai)
        lc = ((q1 - yi) ** 2 + (q2 - yi) ** 2).mean()
        lc.backward()
        self.optim_critic1.step()
        self.optim_critic2.step()

        if update_actor:
            self.optim_actor.zero_grad()
            if self.use_deterministic_policy:
                latent_action = self.actor(si, mean=True, with_prob=False)
                a = self.policy_fn(si, latent_action) if self.policy_fn is not None else latent_action
                q1_pi = self.critic1(si, a)
                #q2_pi = self.critic2(si, a)
                q_pi = q1_pi
                bc_loss = F.mse_loss(a, ai) #temp
                reg_loss = torch.sum(latent_action ** 2, dim=-1).mean()
                lmbda = 200 / q_pi.abs().mean().detach() #temp
                la = - q_pi.mean()*lmbda + self.latent_reg_para * reg_loss #+bc_loss #temp
                log_p = torch.zeros(1)
            else:
                latent_action, log_p = self.actor(si)
                a = self.policy_fn(si, latent_action) if self.policy_fn is not None else latent_action
                entropy = - self.log_temp.exp() * log_p
                reg_loss = torch.sum(latent_action ** 2, dim=-1).mean()
                la = - ((self.critic1(si, a) + self.critic2(si, a)) / 2 + entropy).mean() + self.latent_reg_para * reg_loss
            la.backward()
            self.optim_actor.step()
        else:
            la, log_p = torch.zeros(1), torch.zeros(1)

        if not self.use_deterministic_policy:
            self.optim_log_temp.zero_grad()
            lt = self.log_temp.exp() * (- log_p + self.action_dim).detach().mean() #temp0913
            lt.backward()
            self.optim_log_temp.step()

        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        self.step += 1
        # if self.step%5000==0:
        #     print(ri.squeeze(), di.squeeze(), '\n\n')

        return {
            "latent_action": latent_action.mean(0).detach().cpu().numpy(),
            "reg_loss": reg_loss.item(),
            "q_value": q1.mean().item(), 
            'critic1_loss': lc.item(), 
            'actor_loss': la.item(), 
            'entropy': log_p.mean().item(), 
            'temp': self.log_temp.exp().item(),
            "target_q_value": min_q.mean().item(),
        }
    
    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.optim_critic1.state_dict(), filename + "_critic1_optimizer")
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.optim_critic2.state_dict(), filename + "_critic2_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.optim_actor.state_dict(), filename + "_actor_optimizer")
        torch.save(self.log_temp, filename + "_log_temp")


    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.optim_critic1.load_state_dict(torch.load(filename + "_critic1_optimizer"))
        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.optim_critic2.load_state_dict(torch.load(filename + "_critic2_optimizer"))
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.optim_actor.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.log_temp = torch.load(filename + "_log_temp")
        self.log_temp.requires_grad = True
        self.optim_log_temp = torch.optim.Adam([self.log_temp], lr=self.learning_rate)