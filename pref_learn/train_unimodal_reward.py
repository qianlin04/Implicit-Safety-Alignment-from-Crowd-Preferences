import os
from collections import defaultdict

import absl.app
import absl.flags
from jaxrl_m.envs import make_env
import numpy as np
import torch

import time
from ml_collections import config_flags
import jaxrl_m.envs
from pref_learn.utils.utils import (
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    WandBLogger,
    prefix_metrics,
)
from pref_learn.models.utils import get_datasets, Annealer, EarlyStopper, get_all_posterior
from pref_learn.utils.eval_utilis import ComputeAdvantage, latent_sample_evaluate
from pref_learn.models.vae import UnimodalRewardModel
from pref_learn.models.mlp import (
    MLPModel,
    CategoricalModel,
    MeanVarianceModel,
    MLPClassifier,
)
from algo.iql_agent import ImplicitQLearning, GaussianPolicy, DeterministicPolicy, TwinQ, ValueFunction
import jaxrl_m.learners.d4rl_utils as d4rl_utils
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import pickle
import tqdm
from functools import partial
import shutil

FLAGS_DEF = define_flags_with_default(
    env="maze2d-target-v0",  # can change
    dataset_type='expert_uniform',
    comment="",
    n_sample=5000,
    data_seed=42,
    batch_size=256,
    set_size=-1,
    early_stop=False,
    min_delta=3e-4,
    patience=10,
    lr=1e-4,
    model_type="MLP",  # can change
    # MLP
    hidden_dim=256,
    # Categorical
    num_atoms=10,
    r_min=0,
    r_max=1,
    entropy_coeff=0.1,
    # Mean Var
    variance_penalty=0.0,
    # VAE
    latent_dim=32,
    kl_weight=1.0,
    learned_prior=False,
    flow_prior=False,
    use_annealing=False,
    annealer_baseline=0.0,
    annealer_type="cosine",
    annealer_cycles=4,
    use_seq_encode=True,
    add_action_to_decoder=True,
    # VAE Training
    n_epochs=500,
    eval_freq=10,
    save_freq=200,
    device="cuda",
    # Dataset
    dataset_path="",
    traj_len=8,
    logging=WandBLogger.get_default_config(),
    seed=42,
    # plotting
    debug_plots=False,
    plot_observations=False,
    reward_scaling=1.0,
    # biased
    biased_mode="grid",
    # Policy training 
    load_reward_model_path="",
    prior_eval_episodes=100,
    eval_episodes=20,
    log_interval=1000,
    eval_interval=20000,
    save_interval=50000,
    max_steps=1000000,
    save_video=False,
    fix_mode=-1,
    reward_sample_mode='static',  # prior, posterior, static
    iql_expectile=0.7,
    iql_batch_size=512,
    # Generate dataset for policy
    use_reward_model=True,
    vae_sampling=True,
    comp_size=1000,
    vae_norm="norm",
    sample_freq=10,
    next_step=False,
    exp_scaling_temp=1.0,
    access_to_mode=False, 
    iql_train_on_preference=True,
    append_goal=False,
    # Policy eval
    sampling_method='random',  # 'random, posterior
    label_by_adv=True,
    gamma_for_adv_labeling=1.0,
    test_only=False,
    access_to_real_reward=False,
    access_to_real_task_reward=False,
)




def log_metrics(metrics, epoch, logger):
    for key, val in metrics.items():
        if isinstance(val, list):
            metrics[key] = np.mean(val)
    logger.log(metrics, step=epoch)


def train_reward_model(FLAGS, wb_logger, save_dir):
    gym_env = make_env(FLAGS.env)
    if hasattr(gym_env, 'seed'):
        gym_env.seed(FLAGS.seed)
    gym_env.action_space.seed(FLAGS.seed)
    gym_env.observation_space.seed(FLAGS.seed)
    set_random_seed(FLAGS.seed)
    if hasattr(gym_env, "reward_observation_space") and not FLAGS.label_by_adv:
        observation_dim = gym_env.reward_observation_space.shape[0]
    else:
        observation_dim = gym_env.observation_space.shape[0]
    if "maze" in FLAGS.env:
        gym_env.set_biased_mode(FLAGS.biased_mode)
    action_dim = gym_env.action_space.shape[0]
    len_query = FLAGS.traj_len

    extra_info_in_sample = ['infos/cost', 'infos_2/cost', 'adv1', 'adv2'] if FLAGS.access_to_real_task_reward else []
    (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        len_set,
        _,
        obs_dim,
    ) = get_datasets(
        FLAGS.dataset_path,
        observation_dim,
        action_dim,
        FLAGS.batch_size,
        FLAGS.set_size,
        mode_size=gym_env.get_num_modes(),
        access_to_mode=FLAGS.access_to_mode,
        extra_info_in_sample=extra_info_in_sample
    )

    
    model_input = obs_dim + action_dim * FLAGS.add_action_to_decoder   
    reward_model = UnimodalRewardModel(
        model_input=model_input,
        hidden_dim=FLAGS.hidden_dim,
        annotation_size=len_set,
        size_segment=len_query,
        obs_dim=observation_dim,
        action_dim=action_dim,
        reward_scaling=FLAGS.reward_scaling,
        add_action_to_decoder=FLAGS.add_action_to_decoder,
    )

    device = FLAGS.device
    reward_model = reward_model.to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=FLAGS.lr)
    early_stop = EarlyStopper(FLAGS.patience, FLAGS.min_delta)
    best_criteria = None
    steps = 0
    start_time=time.time()
    for epoch in range(FLAGS.n_epochs):
        metrics = defaultdict(list)
        metrics["epoch"] = epoch

        for batch in train_loader:
            optimizer.zero_grad()
            observations = batch["observations"].to(device).float()
            observations_2 = batch["observations_2"].to(device).float()
            actions = batch["actions"].to(device).float()
            actions_2 = batch["actions_2"].to(device).float()
            labels = batch["labels"].to(device).float()

            if FLAGS.access_to_real_task_reward:
                task_rewards = batch['adv1'] - batch['infos/cost'][:, :, :-1, 0].sum(-1) * gym_env.cost_penalty
                task_rewards_2 = batch['adv2'] - batch['infos_2/cost'][:, :, :-1, 0].sum(-1) * gym_env.cost_penalty
                task_rewards = task_rewards.to(device).float()
                task_rewards_2 = task_rewards_2.to(device).float()
                loss, batch_metrics = reward_model(observations, observations_2, labels, actions, actions_2, task_rewards, task_rewards_2)
            else:
                loss, batch_metrics = reward_model(observations, observations_2, labels, actions, actions_2)
            loss.backward()
            optimizer.step()
            steps += 1

            for key, val in prefix_metrics(batch_metrics, "train").items():
                metrics[key].append(val)

        if epoch % 10==0:
            print("Steps per second: ", steps/(time.time()-start_time))
            for k,v in metrics.items():
                print(f'{k}: {np.mean(v)}')
            print()

        if epoch % FLAGS.eval_freq == 0:
            for batch in test_loader:
                with torch.no_grad():
                    observations = batch["observations"].to(device).float()
                    observations_2 = batch["observations_2"].to(device).float()
                    actions = batch["actions"].to(device).float()
                    actions_2 = batch["actions_2"].to(device).float()
                    labels = batch["labels"].to(device).float()

                    if FLAGS.access_to_real_task_reward:
                        task_rewards = batch['adv1'] - batch['infos/cost'][:, :, :-1, 0].sum(-1) * gym_env.cost_penalty
                        task_rewards_2 = batch['adv2'] - batch['infos_2/cost'][:, :, :-1, 0].sum(-1) * gym_env.cost_penalty
                        task_rewards = task_rewards.to(device).float()
                        task_rewards_2 = task_rewards_2.to(device).float()
                        loss, batch_metrics = reward_model(observations, observations_2, labels, actions, actions_2, task_rewards, task_rewards_2)
                    else:
                        loss, batch_metrics = reward_model(observations, observations_2, labels, actions, actions_2)

                    for key, val in prefix_metrics(batch_metrics, "eval").items():
                        metrics[key].append(val)
            
            criteria = epoch  
            if best_criteria is None or criteria > best_criteria:
                #update biased latent of reward model
                torch.save(reward_model, save_dir + f"/best_model.pt")
                print(f'save to {save_dir + f"/best_model.pt"}')
                best_criteria = criteria

            if FLAGS.early_stop and early_stop.early_stop(criteria):
                log_metrics(metrics, epoch, wb_logger)
                torch.save(reward_model, save_dir + f"/model_{epoch}.pt")
                break

        if epoch % FLAGS.save_freq == 0:
            torch.save(reward_model, save_dir + f"/model_{epoch}.pt")

        for key, val in metrics.items():
            if isinstance(val, list):
                print(key, np.mean(val))
        log_metrics(metrics, epoch, wb_logger)



def main(_):

    FLAGS = absl.flags.FLAGS
    assert os.path.exists(FLAGS.dataset_path), "You must provide a dataset path."
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    save_dir = FLAGS.logging.output_dir + "/" + FLAGS.env
    save_dir += "/" + str(FLAGS.model_type) + "/"

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir += f"{FLAGS.comment}" + "/"
    save_dir += "s" + str(FLAGS.seed)
    FLAGS.logging.output_dir = save_dir

    wb_logger = WandBLogger(FLAGS.logging, variant=variant) if not FLAGS.test_only else None
    print('\n\n', FLAGS.flag_values_dict(), '\n\n')
    train_reward_model(FLAGS, wb_logger, save_dir)

if __name__ == "__main__":
    absl.app.run(main)
