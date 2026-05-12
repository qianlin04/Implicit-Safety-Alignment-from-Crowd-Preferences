import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
from pref_learn.models.flow import Flow
import torch.nn.functional as F

class BasicAttention(nn.Module):

    def __init__(self, emb_dim, heads=1):
        super(BasicAttention, self).__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = emb_dim // heads

        assert self.head_dim * heads == emb_dim

        self.q_linear = nn.Linear(emb_dim, emb_dim)
        self.k_linear = nn.Linear(emb_dim, emb_dim)
        self.v_linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]

        Q = self.q_linear(query).view(N, -1, self.heads, self.head_dim)
        K = self.k_linear(keys).view(N, -1, self.heads, self.head_dim)
        V = self.v_linear(values).view(N, -1, self.heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)  # [N, heads, seq_len, head_dim]
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, V)  # [N, heads, seq_len, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(N, -1, self.emb_dim)
        #out = self.fc_out(out)
        return out

class SeqEncoder(nn.Module):
    log_std_min: float = -20
    log_std_max: float = 4
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SeqEncoder, self).__init__()

        self.embed_layer = nn.Linear(input_dim, hidden_dim)
        self.time_attention = BasicAttention(hidden_dim)
        self.set_attention = BasicAttention(hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        batch_size, set_size = x.shape[0], x.shape[1]
        x = self.embed_layer(x)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = self.time_attention(x, x, x)
        x = x.mean(dim=-2).reshape(batch_size, set_size, -1)
        x = self.set_attention(x, x, x)
        x = x.mean(dim=-2).reshape(batch_size, -1)
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        log_var = torch.clamp(log_var, self.log_std_min, self.log_std_max)
        return mean, log_var


class Encoder(nn.Module):
    log_std_min: float = -20
    log_std_max: float = 4
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),  #temp
            nn.LeakyReLU(0.2),
        )

        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self.model(x)
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        log_var = torch.clamp(log_var, self.log_std_min, self.log_std_max)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            # nn.Linear(hidden_dim, hidden_dim),  #temp
            # nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x_hat = self.model(x)
        return x_hat


class VAEModel(nn.Module):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        action_dim=None,
        use_seq_encode=False,
        add_action_to_decoder=True,
        obs_dim=None,
    ):
        super(VAEModel, self).__init__()
        Encoder_class = Encoder if not use_seq_encode else SeqEncoder
        self.Encoder = Encoder_class(encoder_input, hidden_dim, latent_dim) 
        self.Decoder = Decoder(decoder_input, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.use_seq_encode = use_seq_encode
        self.add_action_to_decoder = add_action_to_decoder
        self.mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)

        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encode(self, s1, s2, y):
        s1 = s1[:, :, :self.size_segment, :]
        s2 = s2[:, :, :self.size_segment, :]

        if not self.use_seq_encode:
            s1_ = s1.view(s1.shape[0], s1.shape[1], -1)
            s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
            y = y.reshape(s1.shape[0], s1.shape[1], -1)

            encoder_input = torch.cat([s1_, s2_, y], dim=-1).view(
                s1.shape[0], -1
            )  # Batch x Ann x (2*T*State + 1)
        else:
            y = y.reshape(s1.shape[0], s1.shape[1], 1, 1).repeat(1, 1, s1.shape[2], 1)
            encoder_input = torch.cat([s1, s2, y], dim=-1) # Batch x Ann x T x (2*State + 1)
        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var
    

    def decode(self, obs, z, a=None):
        if not self.add_action_to_decoder:
            r = torch.cat([obs, z], dim=-1)  # Batch x Ann x T x (State + Z)
        else:
            r = torch.cat([obs, z, a], dim=-1)
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def get_reward(self, r, a=None):
        if self.add_action_to_decoder:
            r = torch.cat([r, a], dim=-1)
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    def transform(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x).mean()
        # return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum") #temp

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = -torch.sum(
                1
                + (log_var - self.log_var)
                - (log_var - self.log_var).exp()
                - (mean - self.mean).pow(2) / (self.log_var.exp()), -1
            ).mean()
        else:
            kl = -torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def forward(self, s1, s2, y, a1=None, a2=None):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        s1 = s1[:, :, :self.size_segment, :]
        s2 = s2[:, :, :self.size_segment, :]
        a1 = a1[:, :, :self.size_segment, :]
        a2 = a2[:, :, :self.size_segment, :]

        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None

        z = z.repeat((1, self.annotation_size * self.size_segment)).view(
            -1, self.annotation_size, self.size_segment, z.shape[1]
        )

        r0 = self.decode(s1, z, a1)
        r1 = self.decode(s2, z, a2)

        r_hat1 = r0.sum(axis=2) / self.scaling
        r_hat2 = r1.sum(axis=2) / self.scaling

        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2).view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() * self.kl_weight if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
            "z_mean": mean.mean().item(),
            'z_log_var': log_var.mean().item(),
            'r_scale': r0.abs().mean().item(),
            'r_delta': (r_hat1 - r_hat2).abs().mean().item(),
            'p_hat': p_hat.abs().mean().item(),

        }

        return loss, metrics

    def sample_prior(self, size):
        z = torch.randn(size, self.latent_dim).to(next(self.parameters()).device)
        if self.learned_prior:
            z = z * torch.exp(0.5 * self.log_var) + self.mean
        elif self.flow_prior:
            z, _ = self.flow(z)
        return z

    def sample_posterior(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents

    def encode_with_preprocess(self, ):
        pass


class VAEClassifier(VAEModel):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        action_dim=None,
    ):
        super(VAEClassifier, self).__init__(
            encoder_input,
            decoder_input,
            latent_dim,
            hidden_dim,
            annotation_size,
            size_segment,
            kl_weight,
            learned_prior,
            flow_prior,
            annealer,
            reward_scaling,
        )

    def forward(self, s1, s2, y):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        mean, log_var = self.encode(s1, s2, y)

        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
            log_det = None
        z = z.repeat((1, self.annotation_size * self.size_segment)).view(
            -1, self.annotation_size, self.size_segment, z.shape[1]
        )

        p_hat = self.Decoder(torch.cat([s1, s2, z], dim=-1)).view(-1, 1)
        p_hat = torch.nn.functional.sigmoid(p_hat).view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
            "z_mean": mean.mean().item(),
            'z_log_var': log_var.mean().item(),
        }

        return loss, metrics

    def decode(self, x, y, z):  # B x S, N x S, B x Z
        x = x[:, None].repeat(1, y.shape[0], 1)  # B x N x S
        z = z[:, None].repeat(1, y.shape[0], 1)  # B x N x Z
        y = y[None].repeat(x.shape[0], 1, 1)  # B x N x S
        x = torch.cat([x, y, z], dim=-1)  # B x N x (2S + Z)
        x = torch.nn.functional.sigmoid(self.Decoder(x))  # B x N x 1
        return x[:, :, 0].mean(dim=-1)  # (B, )


class PolicyDecoder(nn.Module):
    log_std_min: float = -10
    log_std_max: float = 2

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyDecoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.Dropout(0.25),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Dropout(0.25),
        #     nn.ReLU(),
        #     # nn.Linear(hidden_dim, hidden_dim),
        #     # nn.Dropout(0.25),
        #     # nn.LeakyReLU(0.2),
        # )
        self.mu = nn.Linear(hidden_dim, output_dim)
        # self.log_std_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_hat = self.model(x)
        mu = torch.tanh(self.mu(x_hat))
        # log_std = self.log_std_linear(x_hat)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, None #log_std

from torch.distributions import Normal


def biased_bce_with_logits(adv1, adv2, y, bias=1.0):
    # Apply  
    # y = 1 if we prefer x1 to x2
    # We need to implement the numerical stability trick.
    assert adv1.shape==y.shape

    # TRIPLET_THRESHOLD=0.8 #temp0906 triplet loss
    # with torch.no_grad():
    #     prob12 = torch.sigmoid(adv1-adv2)
    #     prob21 = torch.sigmoid(adv2-adv1)
    #     prob = y * prob12 + (1-y) * prob21
    #     adv1_clip_gradient = torch.where(prob>TRIPLET_THRESHOLD, adv1.detach(), adv1)
    #     adv2_clip_gradient = torch.where(prob>TRIPLET_THRESHOLD, adv2.detach(), adv2)
    # logit21 = adv2 - bias * adv1_clip_gradient
    # logit12 = adv1 - bias * adv2_clip_gradient

    logit21 = adv2 - bias * adv1
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    #nlp21 = torch.sigmoid(logit21)
    #nlp12 = torch.sigmoid(logit12)
    loss = y * nlp12 + (1 - y) * nlp21
    #loss = nn.functional.binary_cross_entropy(nlp12, y, reduction="sum")
    loss = loss.mean()

    # Now compute the accuracy
    with torch.no_grad():
        accuracy = ((adv1 > adv2) == torch.round(y)).float().mean()

    return loss, accuracy

class VAEPolicyModel(nn.Module):
    def __init__(
        self,
        encoder_input,
        decoder_input,
        latent_dim,
        hidden_dim,
        annotation_size,
        size_segment,
        kl_weight=1.0,
        learned_prior=False,
        flow_prior=False,
        annealer=None,
        reward_scaling=1.0,
        cpl_alpha=0.2, 
        cpl_bc_coeff=0.0,
        cpl_contrastive_bias=0.5,
        action_dim=None,
        obs_dim=None,
        state_noise_scale=0,
        use_seq_encode=False,
    ):
        super(VAEPolicyModel, self).__init__()
        Encoder_class = Encoder if not use_seq_encode else SeqEncoder
        self.use_seq_encode = use_seq_encode
        self.Encoder = Encoder_class(encoder_input, hidden_dim, latent_dim)
        self.Decoder = PolicyDecoder(decoder_input, hidden_dim, action_dim)
        self.latent_dim = latent_dim
        self.mean = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.log_var = torch.nn.Parameter(
            torch.zeros(latent_dim), requires_grad=learned_prior
        )
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.learned_prior = learned_prior

        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)

        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling
        self.cpl_alpha = cpl_alpha
        self.cpl_bc_coeff = cpl_bc_coeff
        self.cpl_contrastive_bias = cpl_contrastive_bias
        self.state_noise_scale = torch.tensor(state_noise_scale).float()
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def encode(self, s1, s2, y):
        s1 = s1[:, :, :self.size_segment, :]
        s2 = s2[:, :, :self.size_segment, :]

        if not self.use_seq_encode:
            s1_ = s1.view(s1.shape[0], s1.shape[1], -1)
            s2_ = s2.view(s2.shape[0], s2.shape[1], -1)
            y = y.reshape(s1.shape[0], s1.shape[1], -1)

            encoder_input = torch.cat([s1_, s2_, y], dim=-1).view(
                s1.shape[0], -1
            )  # Batch x Ann x (2*T*State + 1)
        else:
            y = y.reshape(s1.shape[0], s1.shape[1], 1, 1).repeat(1, 1, s1.shape[2], 1)
            encoder_input = torch.cat([s1, s2, y], dim=-1) # Batch x Ann x T x (2*State + 1)
        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var

    def decode(self, obs, z):
        r = torch.cat([obs, z], dim=-1)  # Batch x Ann x T x (State + Z)
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r

    # def get_reward(self, r):
    #     r = self.Decoder(r)  # Batch x Ann x T x 1
    #     return r

    def transform(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return self.flow(z)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = -torch.sum(
                1
                + (log_var - self.log_var)
                - (log_var - self.log_var).exp()
                - (mean - self.mean).pow(2) / (self.log_var.exp()), -1
            ).mean()
        else:
            kl = -torch.sum(1.0 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def forward(self, s1, a1, s2, a2, y, bc_loss_only=False, encoder_sampling=True):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        s1 = s1[:, :, :self.size_segment, :]
        s2 = s2[:, :, :self.size_segment, :]
        a1 = a1[:, :, :self.size_segment, :]
        a2 = a2[:, :, :self.size_segment, :]

        mean, log_var = self.encode(s1, s2, y)

        if encoder_sampling:
            if self.flow_prior:
                z, log_det = self.transform(mean, log_var)
            else:
                z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # Batch x Z
                log_det = None
        else:
            z = mean
            log_det = None

        z = z.repeat((1, self.annotation_size * self.size_segment)).view(
            -1, self.annotation_size, self.size_segment, z.shape[1]
        )

        s1 = s1 + torch.randn_like(s1) * self.state_noise_scale.to(s1.device) 
        s2 = s2 + torch.randn_like(s2) * self.state_noise_scale.to(s2.device) 

        # mu, log_std = self.decode(torch.cat([s1, s2], 0), torch.cat([z, z], 0))
        # lp = -torch.square(mu - torch.cat([a1, a2], 0)).sum(dim=-1).sum(dim=-1).view(-1)
        # lp1, lp2 = lp[:len(lp)//2], lp[len(lp)//2:]
        
        #a1, a2 = a1.view(-1, a1.shape[-1]), a2.view(-1, a2.shape[-1])
        #s1, s2 = s1.view(-1, s1.shape[-1]), s2.view(-1, s2.shape[-1])
        mu1, log_std1 = self.decode(s1, z)
        mu2, log_std2 = self.decode(s2, z)
        # dist1 = Normal(mu1, log_std1.exp())
        # dist2 = Normal(mu2, log_std2.exp())
    
        # lp1 = dist1.log_prob(a1).sum(dim=-1).sum(dim=-1).view(-1)
        # lp2 = dist2.log_prob(a2).sum(dim=-1).sum(dim=-1).view(-1)
        lp1 = -torch.square(mu1 - a1).sum(dim=-1).sum(dim=-1).view(-1)
        lp2 = -torch.square(mu2 - a2).sum(dim=-1).sum(dim=-1).view(-1)

        assert lp1.shape==y.view(-1).shape, f'{lp1.shape}  {y.view(-1).shape}'

        bc_loss = -(lp1*y.view(-1)+lp2*(1-y.view(-1))).mean() #temp

        adv1 = self.cpl_alpha * lp1
        adv2 = self.cpl_alpha * lp2

        cpl_loss, accuracy = biased_bce_with_logits(adv1, adv2, y.view(-1), bias=self.cpl_contrastive_bias)
        
        reconstruction_loss = bc_loss if bc_loss_only else cpl_loss + self.cpl_bc_coeff * bc_loss
        latent_loss = self.latent_loss(mean, log_var)

        kl_weight = self.annealer.slope()*self.kl_weight if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
            "cpl_loss": cpl_loss.item(),
            "pos_bc": -(lp1*y.view(-1)+lp2*(1-y.view(-1))).mean().item(),
            "neg_bc": -(lp1*(1-y.view(-1))+lp2*y.view(-1)).mean().item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
            "z_mean": mean.mean(0).detach().cpu().numpy(),
            'z_log_var': log_var.mean(0).detach().cpu().numpy(),
        }

        return loss, metrics

    def sample_prior(self, size):
        z = torch.randn(size, self.latent_dim).to(next(self.parameters()).device)
        if self.learned_prior:
            z = z * torch.exp(0.5 * self.log_var) + self.mean
        elif self.flow_prior:
            z, _ = self.flow(z)
        return z

    def sample_posterior(self, s1, s2, y):
        mean, log_var = self.encode(s1, s2, y)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return mean, log_var, z

    def update_posteriors(self, posteriors, biased_latents):
        self.posteriors = posteriors
        self.biased_latents = biased_latents

    def get_action(self, s, z):
        s = torch.from_numpy(s).float().unsqueeze(0).to(next(self.parameters()).device)
        z = torch.from_numpy(z).float().unsqueeze(0).to(next(self.parameters()).device)
        mu, log_std = self.decode(s, z)
        return mu.detach().cpu().numpy()[0]
    
    def get_policy_input_dim(self, ):
        return self.Decoder.model[0].in_features
    
    def encode_with_preprocess(self, ):
        pass


class UnimodalRewardModel(nn.Module):
    def __init__(
        self,
        model_input,
        hidden_dim,
        annotation_size,
        size_segment,
        reward_scaling=1.0,
        action_dim=None,
        add_action_to_decoder=True,
        obs_dim=None,
    ):
        super(UnimodalRewardModel, self).__init__()
        self.Decoder = Decoder(model_input, hidden_dim, 1)
        self.add_action_to_decoder = add_action_to_decoder
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.model_input = model_input

        self.scaling = reward_scaling
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def predict(self, obs, a=None):
        if not self.add_action_to_decoder:
            r = obs 
        else:
            r = torch.cat([obs, a], dim=-1)
        r = self.Decoder(r)
        return r

    def get_reward(self, r, a=None):
        r = r[..., :self.obs_dim]
        if self.add_action_to_decoder:
            r = torch.cat([r, a], dim=-1)
        r = self.Decoder(r)  # Batch x Ann x T x 1
        return r


    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x).mean()
        # return nn.functional.binary_cross_entropy(x_hat, x, reduction="sum") #temp

    def accuracy(self, x, x_hat):
        predicted_class = (x_hat > 0.5).float()
        return torch.mean((predicted_class == x).float())


    def forward(self, s1, s2, y, a1=None, a2=None, base_reward=None, base_reward_2=None):  # Batch x Ann x T x State, Batch x Ann x 1
        # import pdb; pdb.set_trace()
        s1 = s1[:, :, :self.size_segment, :]
        s2 = s2[:, :, :self.size_segment, :]
        a1 = a1[:, :, :self.size_segment, :]
        a2 = a2[:, :, :self.size_segment, :]

        r0 = self.predict(s1, a1)
        r1 = self.predict(s2, a2)

        r_hat1 = r0.sum(axis=2).squeeze(-1) / self.scaling
        r_hat2 = r1.sum(axis=2).squeeze(-1) / self.scaling
        
        if base_reward is not None:
            assert r_hat1.shape==base_reward.shape, f'r_hat1: {r_hat1.shape}, base_reward: {base_reward.shape} '
            r_hat1 = r_hat1 + base_reward
            r_hat2 = r_hat2 + base_reward_2

        p_hat = torch.nn.functional.sigmoid(r_hat1 - r_hat2).view(-1, 1)
        labels = y.view(-1, 1)

        l2_loss = ((r0 ** 2).mean() + (r1 ** 2).mean()) / 2
        loss = self.reconstruction_loss(labels, p_hat) + 0.1 * l2_loss #temp
        accuracy = self.accuracy(labels, p_hat)

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            'r_scale': r0.abs().mean().item(),
            'r_delta': (r_hat1 - r_hat2).abs().mean().item(),
            'p_hat': p_hat.abs().mean().item(),
            'l2_loss': l2_loss.item(),
        }

        return loss, metrics


    def encode_with_preprocess(self, ):
        pass
