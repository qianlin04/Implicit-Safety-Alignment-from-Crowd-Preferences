import os
from collections import defaultdict

import absl.app
import absl.flags
import numpy as np
import torch
import wandb
import tqdm
import pickle
import time

import jaxrl_m.envs
from jaxrl_m.envs import make_env
from pref_learn.utils.utils import (
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    WandBLogger,
    prefix_metrics,
)
from pref_learn.models.utils import get_datasets, Annealer, EarlyStopper, get_all_posterior
from pref_learn.models.vae import VAEModel, VAEClassifier, VAEPolicyModel
from pref_learn.models.mlp import (
    MLPModel,
    CategoricalModel,
    MeanVarianceModel,
    MLPClassifier,
)
from functools import partial
import matplotlib.pyplot as plt
from experiments.active_utils import get_latent, load_reward_model
from pref_learn.utils.eval_utilis import latent_sample_evaluate, ComputeAdvantage
from types import MethodType
import shutil
from configs.customize_config import CPL_CONFIG

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
    lr=3e-4, 
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
    state_noise_scale=0.0,
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
    eval_interval=50000,
    save_interval=50000,
    max_steps=500000,
    fix_mode=-1,
    # Policy eval
    sampling_method='random', # 'random, posterior
    label_by_adv=True,
    gamma_for_adv_labeling=1.0,
    # Generate dataset for policy
    use_reward_model=True,
    access_to_mode=False, 
    # cpl para
    cpl_contrastive_bias=0.5,
    cpl_bc_coeff=0.0,
    bc_steps=200000,
    test_only=False,
    cpl_lr=0.0001,
    cpl_batch_size=96,
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
    if FLAGS.access_to_mode:
        observation_dim += gym_env.get_num_modes()  #temp
    action_dim = gym_env.action_space.shape[0]

    (
        train_loader,
        test_loader,
        train_dataset,
        eval_dataset,
        len_set,
        len_query,
        obs_dim,
    ) = get_datasets(
        FLAGS.dataset_path,
        observation_dim,
        action_dim,
        FLAGS.batch_size,
        FLAGS.set_size,
        mode_size=gym_env.get_num_modes(),
        access_to_mode=FLAGS.access_to_mode,
    )

    if FLAGS.model_type == "MLP":
        reward_model = MLPModel(obs_dim, FLAGS.hidden_dim)
    elif FLAGS.model_type == "Categorical":
        reward_model = CategoricalModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            n_atoms=FLAGS.num_atoms,
            r_min=FLAGS.r_min,
            r_max=FLAGS.r_max,
            entropy_coeff=FLAGS.entropy_coeff,
        )
    elif FLAGS.model_type == "MeanVar":
        reward_model = MeanVarianceModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            variance_penalty=FLAGS.variance_penalty,
        )
    elif "VAE" in FLAGS.model_type:
        annealer = None
        if FLAGS.use_annealing:
            annealer = Annealer(
                total_steps=max(1, FLAGS.n_epochs // FLAGS.annealer_cycles),
                shape=FLAGS.annealer_type,
                baseline=FLAGS.annealer_baseline,
                cyclical=FLAGS.annealer_cycles > 1,
            )
        if FLAGS.model_type == "VAEClassifier":
            reward_model = VAEClassifier
            decoder_input = 2 * obs_dim + FLAGS.latent_dim
        else:
            reward_model = VAEModel
            decoder_input = obs_dim + FLAGS.latent_dim + action_dim * FLAGS.add_action_to_decoder 
        
        encoder_input = len_set * (2 * observation_dim * len_query + 1) if not FLAGS.use_seq_encode else 2 * observation_dim + 1
        reward_model = reward_model(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            latent_dim=FLAGS.latent_dim,
            hidden_dim=FLAGS.hidden_dim,
            annotation_size=len_set,
            size_segment=len_query,
            kl_weight=FLAGS.kl_weight,
            learned_prior=FLAGS.learned_prior,
            flow_prior=FLAGS.flow_prior,
            annealer=annealer,
            obs_dim=observation_dim,
            action_dim=action_dim,
            reward_scaling=FLAGS.reward_scaling,
            use_seq_encode=FLAGS.use_seq_encode,
            add_action_to_decoder=FLAGS.add_action_to_decoder,
        )
    elif FLAGS.model_type == "MLPClassifier":
        reward_model = MLPClassifier(obs_dim, FLAGS.hidden_dim)
    else:
        raise NotImplementedError

    device = FLAGS.device
    reward_model = reward_model.to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=FLAGS.lr)
    early_stop = EarlyStopper(FLAGS.patience, FLAGS.min_delta)
    best_criteria = None
    steps = 0
    start_time=time.time()
    for epoch in range(0, FLAGS.n_epochs+1):
        metrics = defaultdict(list)
        metrics["epoch"] = epoch

        for batch in train_loader:
            optimizer.zero_grad()
            observations = batch["observations"].to(device).float()
            observations_2 = batch["observations_2"].to(device).float()
            actions = batch["actions"].to(device).float()
            actions_2 = batch["actions_2"].to(device).float()
            labels = batch["labels"].to(device).float()
            loss, batch_metrics = reward_model(observations, observations_2, labels, actions, actions_2)
            loss.backward()
            optimizer.step()
            steps += 1

            for key, val in prefix_metrics(batch_metrics, "train").items():
                metrics[key].append(val)

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
                    loss, batch_metrics = reward_model(
                        observations, observations_2, labels, actions, actions_2
                    )

                    for key, val in prefix_metrics(batch_metrics, "eval").items():
                        metrics[key].append(val)
            print('prior: mean, std:', reward_model.mean, torch.exp(0.5*reward_model.log_var))

            if FLAGS.debug_plots and "maze2d" in FLAGS.env:
                metrics.update(prefix_metrics(fig_dict, "debug_plots"))
            elif "VAE" in FLAGS.model_type:
                pass

            criteria = epoch  

            if best_criteria is None or criteria > best_criteria:
                #update biased latent of reward model
                biased_latents = []
                for mode in range(gym_env.get_num_modes()):
                    batch, num_samples = train_dataset.get_mode_data(1000*gym_env.get_num_modes())
                    obs1 = batch['observations'][batch['mode']==mode]
                    obs2 = batch['observations_2'][batch['mode']==mode]
                    labels = batch['labels'][batch['mode']==mode]
                    means, _ = get_latent(obs1, obs2, gym_env, reward_model, mode, labels=labels) 
                    biased_latents.append(np.mean(means, axis=0))
                reward_model.biased_latents = biased_latents
                for mode, latent in enumerate(reward_model.biased_latents):
                    print(f'mode: {mode},    avg posterior latent: {latent}')

                torch.save(reward_model, save_dir + f"/best_model.pt")
                print(f'save to {save_dir + f"/best_model.pt"}')
                best_criteria = criteria

            if FLAGS.early_stop and early_stop.early_stop(criteria):
                log_metrics(metrics, epoch, wb_logger)
                torch.save(reward_model, save_dir + f"/model_{epoch}.pt")
                break

        if epoch % FLAGS.save_freq == 0:
            torch.save(reward_model, save_dir + f"/model_{epoch}.pt")

        if (
            FLAGS.model_type == "VAE" or "VAE" in FLAGS.model_type
        ) and FLAGS.use_annealing:
            reward_model.annealer.step()

        for key, val in metrics.items():
            if isinstance(val, list):
                print(key, np.mean(val))
        log_metrics(metrics, epoch, wb_logger)




def train_cpl_agent(FLAGS, wb_logger, save_dir):
    device = FLAGS.device
    env = make_env(FLAGS.env)
    if hasattr(env, 'seed'):
        env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    set_random_seed(FLAGS.seed)
    if hasattr(env, "reward_observation_space") and not FLAGS.label_by_adv:
        encoder_obs_dim = env.reward_observation_space.shape[0]
    else:
        encoder_obs_dim = env.observation_space.shape[0]

    observation_dim = env.observation_space.shape[0]
    reward_model = None
    if FLAGS.use_reward_model:
        pretrained_reward_model = load_reward_model(FLAGS.load_reward_model_path).to(device)
    if "maze" in FLAGS.env:
        env.set_biased_mode(FLAGS.biased_mode)
    if FLAGS.access_to_mode:
        observation_dim += env.get_num_modes()  #temp
        encoder_obs_dim += env.get_num_modes()  #temp
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    if not FLAGS.test_only:
        (
            train_loader,
            test_loader,
            train_dataset,
            eval_dataset,
            len_set,
            len_query,
            obs_dim,
        ) = get_datasets(
            FLAGS.dataset_path,
            observation_dim,
            action_dim,
            FLAGS.cpl_batch_size,
            FLAGS.set_size,
            mode_size=env.get_num_modes(),
            access_to_mode=FLAGS.access_to_mode,
        )

    if FLAGS.model_type == "MLP":
        reward_model = MLPModel(obs_dim, FLAGS.hidden_dim)
    elif FLAGS.model_type == "Categorical":
        reward_model = CategoricalModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            n_atoms=FLAGS.num_atoms,
            r_min=FLAGS.r_min,
            r_max=FLAGS.r_max,
            entropy_coeff=FLAGS.entropy_coeff,
        )
    elif FLAGS.model_type == "MeanVar":
        reward_model = MeanVarianceModel(
            input_dim=obs_dim,
            hidden_dim=FLAGS.hidden_dim,
            variance_penalty=FLAGS.variance_penalty,
        )
    elif "VAE" in FLAGS.model_type:
        annealer = None
        if FLAGS.model_type == "VAEClassifier":
            reward_model = VAEClassifier
            decoder_input = 2 * obs_dim + FLAGS.latent_dim
        elif FLAGS.model_type == "VAEPolicy":
            reward_model = VAEPolicyModel
            decoder_input = obs_dim + FLAGS.latent_dim
        else:
            reward_model = VAEModel
            decoder_input = obs_dim + FLAGS.latent_dim
        
        if not FLAGS.test_only:
            _, std = train_dataset.get_obs_mean_std()
            state_noise_scale = std * FLAGS.state_noise_scale
            encoder_input = len_set * (2 * encoder_obs_dim * len_query + 1) if not FLAGS.use_seq_encode else 2 * encoder_obs_dim + 1
            reward_model = reward_model(
                encoder_input=encoder_input,
                decoder_input=decoder_input,
                latent_dim=FLAGS.latent_dim,
                hidden_dim=FLAGS.hidden_dim,
                annotation_size=len_set,
                size_segment=len_query,
                kl_weight=FLAGS.kl_weight,
                learned_prior=FLAGS.learned_prior,
                flow_prior=FLAGS.flow_prior,
                annealer=annealer,
                reward_scaling=FLAGS.reward_scaling,
                action_dim=action_dim,
                obs_dim=encoder_obs_dim,
                cpl_contrastive_bias=FLAGS.cpl_contrastive_bias,
                use_seq_encode=FLAGS.use_seq_encode,
                state_noise_scale=state_noise_scale,
                cpl_bc_coeff=FLAGS.cpl_bc_coeff,
            )
    elif FLAGS.model_type == "MLPClassifier":
        reward_model = MLPClassifier(obs_dim, FLAGS.hidden_dim)
    else:
        raise NotImplementedError

    get_adv_func = ComputeAdvantage(env, device=device)  

    # if FLAGS.label_by_adv:
    #     if 0 and "maze" in FLAGS.env:
    #         env.compute_reward = partial(env.compute_regret, gamma=FLAGS.gamma_for_adv_labeling) 
    #         get_adv_func = None
    #     else:
    #         get_adv_func = ComputeAdvantage(env, device=device)  
    # else: 
    #     get_adv_func = None
    device = FLAGS.device
    policy_fn = lambda x: reward_model.get_action(x[0][..., :obs_dim], x[1])

    if not FLAGS.test_only:
        # replace with pretrained encoder and frozen it
        assert type(reward_model.Encoder) == type(pretrained_reward_model.Encoder)
        reward_model.Encoder = pretrained_reward_model.Encoder
        reward_model.mean, reward_model.log_var = pretrained_reward_model.mean, pretrained_reward_model.log_var
        if hasattr(pretrained_reward_model, 'biased_latents'):
            reward_model.biased_latents = pretrained_reward_model.biased_latents
            for mode, latent in enumerate(reward_model.biased_latents):
                print(f'mode: {mode},    avg posterior latent: {latent}')
        reward_model.mean.requires_grad = reward_model.log_var.requires_grad = False #temp
        for param in reward_model.Encoder.parameters():
            param.requires_grad = False
        reward_model = reward_model.to(device)

        reward_model._original_encode = reward_model.encode
        def encode_with_preprocess(self, s1, s2, y):
            s1, s2 = s1[...,:encoder_obs_dim], s2[...,:encoder_obs_dim]
            return self._original_encode(s1, s2, y)
        reward_model.encode = MethodType(encode_with_preprocess, reward_model)
    else:
        env = make_env(FLAGS.env, render_mode='human')
        reward_model = torch.load(save_dir + f"/CPL_final.pt", weights_only=False).to(device)
        if FLAGS.access_to_mode:
            obs_dim += len(env.goals)
        policy_fn = lambda x: reward_model.get_action(x[0][..., :obs_dim], x[1])
        print(f'load cpl agent from {os.path.join(save_dir, f"CPL_final.pth")}')
        env.env.env._disable_render_order_enforcing=True
        env.reset()
        env.render()
        res = latent_sample_evaluate(env, reward_model, None, get_adv_func, policy_fn, FLAGS, latent_sample_type='posterior',
                                         eval_episodes=2, sampling_method=FLAGS.sampling_method, fix_mode=FLAGS.fix_mode)
        exit(0)


    optimizer = torch.optim.Adam(reward_model.parameters(), lr=FLAGS.cpl_lr)
    early_stop = EarlyStopper(FLAGS.patience, FLAGS.min_delta)
    best_criteria = None
    metrics = defaultdict(list)
    for step in tqdm.tqdm(
        range(0, FLAGS.max_steps+1 ), smoothing=0.1, dynamic_ncols=True
    ):
        reward_model.train()
        metrics["cpl/step"].append(step)

        batch, _ = train_dataset.get_mode_data(FLAGS.cpl_batch_size)
        optimizer.zero_grad()
        observations = torch.tensor(batch["observations"]).to(device).float()
        observations_2 = torch.tensor(batch["observations_2"]).to(device).float()
        actions = torch.tensor(batch["actions"]).to(device).float()
        actions_2 = torch.tensor(batch["actions_2"]).to(device).float()
        labels = torch.tensor(batch["labels"]).to(device).float()
        loss, batch_metrics = reward_model(observations, actions, observations_2, actions_2, labels, bc_loss_only=(step<=FLAGS.bc_steps))
        loss.backward()
        optimizer.step()

        for key, val in prefix_metrics(batch_metrics, "cpl/train").items():
            metrics[key].append(val)

        if step % FLAGS.log_interval == 0:
            for k,v in metrics.items():
                if isinstance(v, list):
                    v=np.mean(np.array(v), 0)
                print(f'{k}: {v}')
            log_metrics(metrics, step, wb_logger)    
            metrics = defaultdict(list)
            print(flush=True)

        if step % FLAGS.eval_interval == 0:
            reward_model.eval()
            eval_metrics = defaultdict(list)
            eval_metrics["cpl/step"].append(step)
            for batch in test_loader:
                with torch.no_grad():
                    observations = batch["observations"].to(device).float()
                    observations_2 = batch["observations_2"].to(device).float()
                    actions = batch["actions"].to(device).float()
                    actions_2 = batch["actions_2"].to(device).float()
                    labels = batch["labels"].to(device).float()
                    loss, batch_metrics = reward_model(
                        observations, actions, observations_2, actions_2, labels, bc_loss_only=(step<=FLAGS.bc_steps), encoder_sampling=False,
                    )

                    for key, val in prefix_metrics(batch_metrics, "cpl/eval").items():
                        eval_metrics[key].append(val)

            res = latent_sample_evaluate(env, reward_model, eval_dataset, get_adv_func, policy_fn, FLAGS, latent_sample_type='prior', eval_episodes=FLAGS.prior_eval_episodes)
            eval_metrics["prior_reward"].append(np.mean(res['utility']))
            eval_metrics["prior_cost"].append(np.mean(res['cost']))
            res = latent_sample_evaluate(env, reward_model, eval_dataset, get_adv_func, policy_fn, FLAGS, latent_sample_type='posterior',
                                         eval_episodes=FLAGS.eval_episodes, sampling_method=FLAGS.sampling_method, fix_mode=FLAGS.fix_mode)
            eval_metrics["posterior_reward"].append(np.mean(res['utility']))
            eval_metrics["episode.r"].append(np.mean(res['utility']))
            eval_metrics["posterior_cost"].append(np.mean(res['cost']))       

            print({f"sampling/{k}":v for k,v in eval_metrics.items()})            

            #criteria = np.mean(metrics["eval/loss"])
            criteria = step #eval_metrics["posterior_reward"] #np.mean(eval_metrics["cpl/eval/accuracy"]) #np.mean(metrics["episode.r"])

            if step>FLAGS.bc_steps:

                if best_criteria is None or criteria >= best_criteria:
                    torch.save(reward_model, save_dir + f"/CPL_final.pt")
                    print(f'save to {save_dir + f"/CPL_final.pt"}')
                    best_criteria = criteria

                if FLAGS.early_stop and early_stop.early_stop(criteria):
                    log_metrics(eval_metrics, step, wb_logger)
                    torch.save(reward_model, save_dir + f"/CPL_{step}.pt")
                    break

            log_metrics(eval_metrics, step, wb_logger)    

        if step % FLAGS.save_interval == 0:
            torch.save(reward_model, save_dir + f"/CPL_{step}.pt")

        

def main(_):

    FLAGS = absl.flags.FLAGS
    if FLAGS.env in CPL_CONFIG:
        print(f"Loading config for {FLAGS.env} ...")
        for k, v in CPL_CONFIG[FLAGS.env].items():
            setattr(FLAGS, k, v)
        print(f"Updated FLAGS: {CPL_CONFIG[FLAGS.env]}")
    else:
        print(f"No config found for {FLAGS.env}, using default FLAGS.")
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

    if FLAGS.load_reward_model_path == "":
        train_reward_model(FLAGS, wb_logger, save_dir)
        FLAGS.load_reward_model_path = save_dir
    else:
        model_path = FLAGS.load_reward_model_path + "/best_model.pt"
        assert os.path.exists(model_path), "Model doesn't exist"
        if model_path!=os.path.join(save_dir, "best_model.pt"):
            os.makedirs(save_dir, exist_ok=True)
            shutil.copy(model_path, os.path.join(save_dir, "best_model.pt"))
            print(f'copy {model_path} to {os.path.join(save_dir, "best_model.pt")}')
    train_cpl_agent(FLAGS, wb_logger, save_dir)

if __name__ == "__main__":
    absl.app.run(main)
