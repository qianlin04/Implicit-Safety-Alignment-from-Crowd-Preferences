import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, max_action=1.0):
        super().__init__()
        self.max_action = max_action
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh if max_action!=-1 else None)

    def forward(self, obs):
        x = self.net(obs)
        return  x * self.max_action if self.max_action!=-1 else x

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

# DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


# class ImplicitQLearning(nn.Module):
#     def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
#                  tau, beta, discount=0.99, alpha=0.005):
#         super().__init__()
#         self.qf = qf
#         self.q_target = copy.deepcopy(qf).requires_grad_(False)
#         self.vf = vf
#         self.policy = policy
#         self.v_optimizer = optimizer_factory(self.vf.parameters())
#         self.q_optimizer = optimizer_factory(self.qf.parameters())
#         self.policy_optimizer = optimizer_factory(self.policy.parameters())
#         self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
#         self.tau = tau
#         self.beta = beta
#         self.discount = discount
#         self.alpha = alpha

#     def update(self, observations, actions, next_observations, rewards, terminals):
#         with torch.no_grad():
#             next_v = self.vf(next_observations)

#         # v, next_v = compute_batched(self.vf, [observations, next_observations])
        
#         # Update Q function
#         targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
#         qs = self.qf.both(observations, actions)
#         q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
#         self.q_optimizer.zero_grad()
#         q_loss.backward()
#         self.q_optimizer.step()

#         # Update target Q network
#         update_exponential_moving_average(self.q_target, self.qf, self.alpha)

#         # Update value function
#         with torch.no_grad():
#             target_q = self.q_target(observations, actions)
#         v = self.vf(observations)
#         adv = target_q - v
#         v_loss = asymmetric_l2_loss(adv, self.tau)
#         self.v_optimizer.zero_grad()
#         v_loss.backward()
#         self.v_optimizer.step()

#         # Update policy
#         exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
#         clip_ratio = torch.mean((torch.exp(self.beta * adv.detach())>EXP_ADV_MAX).float())
#         positive_adv_ratio = torch.mean((adv>0).float())
#         policy_out = self.policy(observations)
#         if isinstance(policy_out, torch.distributions.Distribution):
#             bc_losses = -policy_out.log_prob(actions)
#         elif torch.is_tensor(policy_out):
#             assert policy_out.shape == actions.shape
#             bc_losses = torch.sum((policy_out - actions)**2, dim=1)
#         else:
#             raise NotImplementedError
#         policy_loss = torch.mean(exp_adv * bc_losses)
#         self.policy_optimizer.zero_grad()
#         policy_loss.backward()
#         self.policy_optimizer.step()
#         self.policy_lr_schedule.step()

#         return {'v_loss': v_loss.item(), 'q_loss': q_loss.item(), 'policy_loss': policy_loss.item(), 
#             'v': v.mean().item(), 'q': qs[0].mean().item(), 'adv': adv.mean().item(), 'clip_ratio': clip_ratio.item(), 'positive_adv_ratio': positive_adv_ratio.item(), 
#             'adv_std': torch.std(adv).item(), 'num_terminal': torch.mean((terminals>0).float()).item()}
    
#     def get_policy_input_dim(self, ):
#         return self.policy.net[0].in_features
    




class ImplicitQLearning(nn.Module):
    def __init__(
            self, 
            state_size,
            action_size,
            policy_size=None,
            policy_fn=None,
            learning_rate=3e-4,
            max_action=1.0,
            max_policy_action=3.0,
            discount=0.99,
            tau=0.005,
            max_steps=1000000,
            beta=10, 
            expectile=0.75,
            latent_reg_para=0.0):
        super().__init__()
        if policy_size is None:
            policy_size = action_size
        self.qf = TwinQ(state_size, action_size, hidden_dim=1024, n_hidden=4).to(device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False)
        self.vf = ValueFunction(state_size, hidden_dim=1024, n_hidden=4).to(device)
        if policy_fn is None:
            self.policy = GaussianPolicy(state_size, policy_size, hidden_dim=1024, n_hidden=4).to(device)
        else:
            self.policy = DeterministicPolicy(state_size, policy_size, hidden_dim=1024, n_hidden=4, max_action=max_policy_action).to(device)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.expectile = expectile
        self.policy_fn = policy_fn
        self.max_action = max_action
        self.max_policy_action = max_policy_action
        self.latent_reg_para = latent_reg_para

    def learn(self, experiences):

        observations, actions, rewards, next_observations, terminals = experiences

        with torch.no_grad():
            next_v = self.vf(next_observations)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])
        
        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.tau)

        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.expectile)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        clip_ratio = torch.mean((torch.exp(self.beta * adv.detach())>EXP_ADV_MAX).float())
        positive_adv_ratio = torch.mean((adv>0).float())
        policy_out = self.policy(observations)
        

        if isinstance(policy_out, torch.distributions.Distribution):
            reg_loss = 0.0
            if self.policy_fn is None:
                bc_losses = -policy_out.log_prob(actions)
            else:
                raise NotImplementedError
        elif torch.is_tensor(policy_out):
            reg_loss = torch.sum(policy_out ** 2, dim=-1).mean()
            if self.policy_fn is not None:
                policy_out = self.policy_fn(observations, policy_out)
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses) + self.latent_reg_para * reg_loss
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        return {'v_loss': v_loss.item(), 'q_loss': q_loss.item(), 'policy_loss': policy_loss.item(), 
            'v': v.mean().item(), 'q': qs[0].mean().item(), 'adv': adv.mean().item(), 'clip_ratio': clip_ratio.item(), 'positive_adv_ratio': positive_adv_ratio.item(), 
            'adv_std': torch.std(adv).item(), 'num_terminal': torch.mean((terminals>0).float()).item()}
    
    def update(self,):
        pass

    def get_action(self, state, eval=False, latent_action=None, random_latent=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            if latent_action is None:
                if random_latent:
                    latent_action = torch.randn(size=(1, self.policy_size)).to(self.device)
                else:
                    latent_action = self.policy.act(state, deterministic=True)
                if self.max_policy_action!=-1:
                    latent_action = latent_action.clamp(-self.max_policy_action, self.max_policy_action)
        action = latent_action if self.policy_fn is None else self.policy_fn(state, latent_action)
        return action.cpu().data.numpy().flatten(), latent_action


    def get_policy_input_dim(self, ):
        return self.policy.net[0].in_features
    

    def save(self, filename):
        torch.save(self.qf.state_dict(), filename + "_qf")
        torch.save(self.q_optimizer.state_dict(), filename + "_q_optimizer")
        torch.save(self.vf.state_dict(), filename + "_vf")
        torch.save(self.v_optimizer.state_dict(), filename + "_v_optimizer")
        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")


    def load(self, filename):
        self.qf.load_state_dict(torch.load(filename + "_qf"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_q_optimizer"))
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(torch.load(filename + "_vf"))
        self.v_optimizer.load_state_dict(torch.load(filename + "_v_optimizer"))

        self.policy.load_state_dict(torch.load(filename + "_policy"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))