import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a)) if self.max_action!=-1 else self.l3(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    def __init__(
        self,
        state_size,
        action_size,
        policy_size,
        policy_fn,
        learning_rate=3e-4,
        max_policy_action=3.0,
        max_action=1.0,
        discount=0.99,
        tau=0.005, #temp 0.005 -> 0.001
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=1,  #temp
        alpha=10,
        latent_reg_para = 0.0
    ):
        self.policy_fn = policy_fn
        self.policy_size = policy_size
        self.actor = Actor(state_size, policy_size, max_policy_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.max_action = max_action
        self.max_policy_action = max_policy_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.latent_reg_para = latent_reg_para

        self.total_it = 0


    def get_action(self, state, eval=False, latent_action=None, random_latent=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            if latent_action is None:
                if random_latent:
                    latent_action = torch.randn(size=(1, self.policy_size)).to(self.device)
                else:
                    latent_action = self.actor(state)
        return self.policy_fn(state, latent_action).cpu().data.numpy().flatten(), latent_action


    def learn(self, experiences):
        self.total_it += 1
        info = {}

        state, action, reward, next_state, done = experiences

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            # if self.total_it < 200000:
            #     latent_action = action
            #     next_action = action
            # else:
            latent_action = self.actor_target(next_state)
            next_action = self.policy_fn(next_state, latent_action)

            next_action = (
                next_action + noise
            )
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.discount * target_Q

            if self.total_it % 1000 == 0: #temp
                print(reward.flatten().cpu().data.numpy(), target_Q.flatten().cpu().data.numpy(), done.flatten().cpu().data.numpy())

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            latent_action = self.actor(state)
            pi = self.policy_fn(state, latent_action)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            bc_loss = F.mse_loss(pi, action)
            reg_loss = torch.sum(latent_action ** 2, dim=-1).mean()
            actor_loss = -lmbda * Q.mean() + bc_loss + self.latent_reg_para * reg_loss
            #actor_loss = -Q.mean() + lmbda * bc_loss + self.latent_reg_para * reg_loss
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            info.update( {
                "latent_action": latent_action.mean(0).detach().cpu().numpy(),
                "bc_loss": bc_loss.item(),
                "reg_loss": reg_loss.item(),
                "actor_value": Q.mean().item(),
                "actor_loss": actor_loss.item(),
                "latent_action_std": latent_action.std(0).detach().cpu().numpy(),
                "action_mean": pi.mean(0).detach().cpu().numpy(),
                "action_std": pi.std(0).detach().cpu().numpy(),
                #"lmbda": lmbda.item(),
            })

        info.update({
            "critic1_loss": critic_loss.item(),
            "q_value": current_Q1.mean().item(),
            "target_q_value": target_Q.mean().item(),
            "target_q_std": target_Q.std().item(),
        })

        return info


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)