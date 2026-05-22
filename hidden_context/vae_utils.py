import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import Trainer, EvalPrediction
import wandb
from transformers.optimization import get_cosine_schedule_with_warmup


class PairEncoder(nn.Module):
    """
    Model to encode pairs of accepted and rejected responses
    """

    def __init__(self, embed_dim, hidden_dim, output_dim):
        super(PairEncoder, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, e_c, e_r):
        x = torch.cat([e_c, e_r], dim=1)
        return self._model(x)


class SequenceEncoder(nn.Module):
    """
    Model to encode sequence of responses
    """

    def __init__(self, input_dim, latent_dim):
        super(SequenceEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear = nn.Identity()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)
        self.layer_norm = nn.Identity()     # nn.LayerNorm(latent_dim)

    def forward(
        self, sequences, seq_start_end
    ):  # (C_1+C_2+...+C_n, D), [(0, C_1), (C_1, C_1+C_2), ..., (C_1+...+C_n-1, C_1+...+C_n)]
        outputs = []
        for _, (start, end) in enumerate(seq_start_end):
            context = sequences[start:end]  # C_i x D
            q = self.w_q(context)
            k = self.w_k(context)
            attention_scores = torch.matmul(
                q, k.transpose(0, 1)
            )
            attention_scores = attention_scores / (context.shape[-1] ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)  # C_i x C_i
            weighted_values = torch.matmul(attention_weights, self.w_v(context))  # C_i x D
            output = torch.mean(weighted_values, dim=0)  # D
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # n x D

        mean = self.layer_norm(self.mean_layer(outputs))
        log_var = self.layer_norm(self.log_var_layer(outputs))
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xc, xr, z):
        xc = torch.cat([xc, z], dim=1)
        xr = torch.cat([xr, z], dim=1)
        rc = self._model(xc)
        rr = self._model(xr)
        return rc, rr


class VAEModel(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, hidden_dim, latent_dim, llm_encoder, llm_contexts_encoder,
                 fixed_contexts=False, fixed_llm_embeddings=False, use_causal_lm=False, use_attention_layer=False,
                 use_transformer=False, concat_chosen_rejected=False):
        super(VAEModel, self).__init__()
        self.llm_encoder = llm_encoder
        self.llm_contexts_encoder = llm_contexts_encoder
        self.pair_encoder = PairEncoder(encoder_embed_dim, hidden_dim, latent_dim)
        self.sequence_encoder = SequenceEncoder(latent_dim, latent_dim)
        self.decoder = Decoder(decoder_embed_dim + latent_dim, hidden_dim)

        self.latent_dim = latent_dim
        self.fixed_contexts = fixed_contexts
        self.fixed_llm_embeddings = fixed_llm_embeddings
        self.use_causal_lm = use_causal_lm
        self.use_attention_layer = use_attention_layer
        self.use_transformer = use_transformer
        self.concat_chosen_rejected = concat_chosen_rejected

        self.saved_embeddings = torch.Tensor(4, latent_dim)
        self.saved_embeddings.uniform_(-1, 1)

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std).to(mean.device)  # sampling epsilon
        z = mean + std * epsilon                         # reparameterization trick
        z = F.normalize(z, p=2, dim=-1) * math.sqrt(z.shape[-1])
        return z

    def encode_pair(self, e_c, e_r):
        return self.pair_encoder(e_c, e_r)

    def encode_sequence(self, sequences, seq_start_end):
        return self.sequence_encoder(sequences, seq_start_end)

    def decode(self, e_c, e_r, z):
        return self.decoder(e_c, e_r, z)

    def forward(
        self,
        target_chosen,
        target_rejected,
        context_chosen,
        context_rejected,
        seq_start_end,
        user_type,
        ground_truth_user_vector=False,
    ):
        pair_embed = self.encode_pair(context_chosen, context_rejected)
        mean, log_var = self.encode_sequence(pair_embed, seq_start_end)
        mean = torch.clamp(mean, -1, 1)

        _log_var = torch.clamp(log_var, -1, 1)
        if ground_truth_user_vector:
            z = torch.zeros_like(mean)
            self.saved_embeddings = self.saved_embeddings.to(mean.device)
            for idx in range(user_type.shape[0]):
                z[idx] = self.saved_embeddings[int(user_type[idx])]
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * _log_var))

        if not self.training and not ground_truth_user_vector:
            z = mean
        rc, rr = self.decode(target_chosen, target_rejected, z)

        return rc, rr, mean, _log_var, z

    def save_model(self, path):
        torch.save(self, path)


class VAETrainer(Trainer):
    def __init__(
        self, *args, lr_lambda=None, kl_loss_weight=None, use_annealing=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda
        self.kl_loss_weight = kl_loss_weight
        self.use_annealing = use_annealing
        self.annealer = Annealer(
            total_steps=1e4, shape="cosine", baseline=0.1, cyclical=True    # todo: change total_step here
        )

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, wrapped_model, inputs, return_outputs=False, num_items_in_batch=None):
        if isinstance(wrapped_model, VAEModel):
            model = wrapped_model  # .module
        else:
            model = wrapped_model.module
        device = model.llm_encoder.device
        batch_size = inputs["seq_start_end"].shape[0]
        if model.fixed_llm_embeddings:
            embeddings_chosen = torch.tensor(inputs["embeddings_chosen"]).to(device).bfloat16()
            embeddings_rejected = torch.tensor(inputs["embeddings_rejected"]).to(device).bfloat16()
        else:
            embeddings = model.llm_encoder(
                input_ids=torch.concatenate(
                    [
                        inputs["input_ids_chosen"],
                        inputs["input_ids_rejected"],
                    ],
                    dim=0,
                ),
                attention_mask=torch.concatenate(
                    [
                        inputs["attention_mask_chosen"],
                        inputs["attention_mask_rejected"],
                    ],
                    dim=0,
                ),
            )[0]
            embeddings_chosen = embeddings[:batch_size]
            embeddings_rejected = embeddings[batch_size:]

        if model.fixed_contexts:
            contexts_embeddings_chosen = torch.tensor(inputs["contexts_embeddings_chosen"]).to(device).bfloat16()
            contexts_embeddings_rejected = torch.tensor(inputs["contexts_embeddings_rejected"]).to(device).bfloat16()
        else:
            input_ids_chosen = inputs["contexts_input_ids_chosen"]
            attention_mask_chosen = inputs["contexts_attention_mask_chosen"]
            token_length_chosen = torch.eq(input_ids_chosen,
                                           model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1
            input_ids_rejected = inputs["contexts_input_ids_rejected"]
            attention_mask_rejected = inputs["contexts_attention_mask_rejected"]
            token_length_rejected = torch.eq(input_ids_rejected,
                                             model.llm_contexts_encoder.config.pad_token_id).int().argmax(-1) - 1

            with torch.no_grad():
                last_hidden_state_chosen = model.llm_contexts_encoder(
                    input_ids=input_ids_chosen,
                    attention_mask=attention_mask_chosen,
                    output_hidden_states=True
                ).hidden_states[-1]

                weights_for_non_padding_chosen = attention_mask_chosen * torch.arange(
                    start=1, end=last_hidden_state_chosen.shape[1] + 1
                ).unsqueeze(0).to(attention_mask_chosen.device).float()
                sum_embeddings = torch.sum(last_hidden_state_chosen * weights_for_non_padding_chosen.unsqueeze(-1),
                                           dim=1)
                num_of_none_padding_tokens_chosen = torch.sum(weights_for_non_padding_chosen, dim=-1).unsqueeze(-1)
                contexts_embeddings_chosen = sum_embeddings / num_of_none_padding_tokens_chosen
                last_hidden_state_rejected = model.llm_contexts_encoder(
                    input_ids=input_ids_rejected,
                    attention_mask=attention_mask_rejected,
                    output_hidden_states=True
                ).hidden_states[-1]

                weights_for_non_padding_rejected = attention_mask_rejected * torch.arange(
                    start=1, end=last_hidden_state_rejected.shape[1] + 1
                ).unsqueeze(0).to(attention_mask_rejected.device).float()
                sum_embeddings = torch.sum(last_hidden_state_rejected * weights_for_non_padding_rejected.unsqueeze(-1),
                                           dim=1)
                num_of_none_padding_tokens_rejected = torch.sum(weights_for_non_padding_rejected, dim=-1).unsqueeze(-1)
                contexts_embeddings_rejected = sum_embeddings / num_of_none_padding_tokens_rejected
        seq_start_end = inputs["seq_start_end"]
        user_type = torch.tensor(inputs["user_type"]).to(device).bfloat16()
        rewards_chosen, rewards_rejected, mean, log_var, z = model(
            embeddings_chosen,
            embeddings_rejected,
            contexts_embeddings_chosen,
            contexts_embeddings_rejected,
            seq_start_end,
            user_type,
            ground_truth_user_vector=False,  # todo: set to True for debug usage
            # mask_chosen=inputs["attention_mask_chosen"],
            # mask_rejected=inputs["attention_mask_rejected"],
        )

        reproduction_loss = self.loss(rewards_chosen, rewards_rejected)
        if self.kl_loss_weight == 0:
            loss = reproduction_loss
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "train_recon_loss": reproduction_loss.mean().item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "log_var": log_var.mean().item()
                    }
                )
        else:
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).mean()
            if self.use_annealing:
                kld = self.annealer(kld)
                self.annealer.step()
            kld = self.kl_loss_weight * kld
            loss = reproduction_loss + kld
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "train_recon_loss": reproduction_loss.mean().item(),
                        "train_kld": kld.mean().item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "log_var": log_var.mean().item()
                    }
                )
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
                "mean": mean,
                "log_var": log_var,
                "z": z,
                "user_type": user_type,
            }
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        if optimizer is None:
            raise ValueError("Optimizer has not been created yet before scheduler creation.")


        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.lr_scheduler = scheduler
        return scheduler

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected, mean, log_var, z, user_type = (
            eval_prediction.predictions
        )
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)
        mean = torch.from_numpy(mean)
        log_var = torch.from_numpy(log_var)
        z = torch.from_numpy(z)
        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        kld = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        accuracy = torch.mean((loss < np.log(2)).float())

        def plot_latent(latent):
            from sklearn.manifold import TSNE
            z_embedding = TSNE(n_components=2, init='random', perplexity=20, learning_rate="auto").fit_transform(latent.numpy())
            import matplotlib.pyplot as plt
            colors = [f"C{int(i)}" for i in user_type]
            plt.scatter(z_embedding[:, 0], z_embedding[:, 1], c=colors)
            im = wandb.Image(plt)
            plt.close()
            return im
        im1 = plot_latent(mean)
        im2 = plot_latent(z)

        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
            "kld": kld.mean().item(),
            # "mean_embeddings": im1,
            # "z_embeddings": im2,
        }


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return
