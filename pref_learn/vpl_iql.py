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
from pref_learn.models.vae import VAEModel, VAEClassifier
from pref_learn.models.mlp import (
    MLPModel,
    CategoricalModel,
    MeanVarianceModel,
    MLPClassifier,
)
from algo.iql_agent import ImplicitQLearning, GaussianPolicy, DeterministicPolicy, TwinQ, ValueFunction
import jaxrl_m.learners.d4rl_utils as d4rl_utils
from experiments.active_utils import load_reward_model
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import pickle
import tqdm
from experiments.active_utils import get_latent
from functools import partial
import shutil
from configs.customize_config import IQL_CONFIG

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
)


config_flags.DEFINE_config_file(
    "config", "configs/default_config.py", lock_config=False
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
                total_steps=FLAGS.n_epochs // FLAGS.annealer_cycles,
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
            obs_dim=observation_dim,
            action_dim=action_dim,
            annealer=annealer,
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
                    loss, batch_metrics = reward_model(
                        observations, observations_2, labels, actions, actions_2
                    )

                    for key, val in prefix_metrics(batch_metrics, "eval").items():
                        metrics[key].append(val)
            print('prior: mean, std:', reward_model.mean, torch.exp(0.5*reward_model.log_var))

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


def train_iql_agent(FLAGS, wb_logger, save_dir):
    print(FLAGS.config.to_dict())
    env = make_env(FLAGS.env)
    dataset = None
    if 'MO-' in FLAGS.env:
        env.set_dataset_type=FLAGS.dataset_type
        dataset = env.get_dataset(convert_to_qlearning_dataset=True)
    device = FLAGS.device

    reward_model = None
    if FLAGS.use_reward_model:
        reward_model = load_reward_model(FLAGS.load_reward_model_path).to(device)


    get_adv_func = ComputeAdvantage(env, device=device)  

    # if FLAGS.label_by_adv:
    #     # if 0 and "maze" in FLAGS.env: #temp0912
    #     #     env.compute_reward = partial(env.compute_regret, gamma=FLAGS.gamma_for_adv_labeling) 
    #     #     get_adv_func = None
    #     # else:
    #     get_adv_func = ComputeAdvantage(env, device=device)  
    # else: 
    #     get_adv_func = None

    preference_dataset = None
    if hasattr(env, "reward_observation_space") and not FLAGS.label_by_adv:
        obs_dim = env.reward_observation_space.shape[0]
    else:
        obs_dim = env.observation_space.shape[0]
    if FLAGS.use_reward_model and "VAE" in FLAGS.model_type and FLAGS.sampling_method=='random':
        (
            train_loader,
            _,
            _,
            preference_dataset,
            set_len,
            _,
            _,
        ) = get_datasets(
            FLAGS.dataset_path,
            obs_dim,
            env.action_space.shape[0],
            FLAGS.batch_size,
            reward_model.annotation_size
        )
        assert set_len == reward_model.annotation_size

    # if getattr(reward_model, 'biased_latents', None) is None and not FLAGS.test_only:
    #     biased_latents = []
    #     for mode in range(env.get_num_modes()):
    #         means = []
    #         batch, num_samples = train_preference_dataset.get_mode_data(1000)
    #         for i in range(num_samples):
    #             if get_adv_func is not None:
    #                 labels = get_adv_func.get_label(batch, mode, idx=i)
    #             else:
    #                 labels = None      
    #             mean, _ = get_latent(batch['observations'][i], batch['observations_2'][i], env, reward_model, mode, labels=labels) 
    #             means.append(mean)
    #         biased_latents.append(np.mean(means, axis=0))
    #     reward_model.biased_latents = biased_latents
    #     print(f'fix_mode: {FLAGS.fix_mode}')
    #     for mode, latent in enumerate(reward_model.biased_latents):
    #         print(f'mode: {mode},    posterior latent: {latent}')


    if FLAGS.iql_train_on_preference:
        dataset = d4rl_utils.transform_preference_to_transition(path=FLAGS.dataset_path)

    if not FLAGS.test_only:
        dataset, comp_obs = d4rl_utils.get_dataset(
            env,
            use_reward_model=FLAGS.use_reward_model and FLAGS.reward_sample_mode=='static',
            append_goal=FLAGS.append_goal,
            fix_mode=FLAGS.fix_mode,
            model_type=FLAGS.model_type,
            reward_model=reward_model,
            vae_sampling=FLAGS.vae_sampling,
            comp_size=FLAGS.comp_size,
            vae_norm=FLAGS.vae_norm,
            terminate_on_end="sort" in FLAGS.env,
            sample_freq=FLAGS.sample_freq,
            next_step=FLAGS.next_step,
            exp_scaling_temp=FLAGS.exp_scaling_temp,
            wrap_dataset=False,
            dataset=dataset,
            obs_dim = obs_dim,
            access_to_real_reward = FLAGS.access_to_real_reward
        )
        if FLAGS.reward_sample_mode != 'static':
            augment_obs_dim = len(dataset['observations'][0]) + FLAGS.latent_dim
        else:
            augment_obs_dim = len(dataset['observations'][0])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving config to {save_dir}/config.pkl")
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)

    
    
    if FLAGS.test_only:
        agent = torch.load(os.path.join(save_dir, f'IQL_final.pth'), weights_only=False)
        print(f'load iql agent from {os.path.join(save_dir, f"IQL_final.pth")}')
    else:
        agent = ImplicitQLearning(
            state_size=augment_obs_dim,
            action_size=env.action_space.shape[0],
            learning_rate=FLAGS.config.actor_lr,
            max_steps=FLAGS.max_steps,
            expectile=FLAGS.iql_expectile,
            beta=FLAGS.config.temperature,
            tau=FLAGS.config.tau,
            discount=FLAGS.config.discount,
        ).to(device)

    def policy_fn(x):
        obs = torch.from_numpy(x[0]).float().unsqueeze(0).to(device)
        z = torch.from_numpy(x[1]).float().unsqueeze(0).to(device)
        if not FLAGS.append_goal:
            obs = torch.cat([obs, z], dim=-1)
        else:
            goal = torch.from_numpy(env.goals[env.mode]).float().unsqueeze(0).to(device)
            obs = torch.cat([obs, goal], dim=-1)
        return agent.policy.act(obs, deterministic=True)[0].detach().cpu().numpy()

    if FLAGS.test_only:
        env.env.env._disable_render_order_enforcing=True
        env.render()
        # reward_model.biased_latents = np.array([
        #     [-0.22624934,  0.71198285 , 0.85982555 ,-0.5488922 ],
        #     [-0.24614264, -0.5732463 ,  0.8777762 , -1.5732071 ],
        #     [-0.14857805, -0.19090356 , 0.9002441  , 0.9287147 ],
        #     [-0.19418697, -1.426705 ,   0.890713 ,  -0.00897022],]
        # )
        res = latent_sample_evaluate(env, reward_model, preference_dataset, get_adv_func, policy_fn, FLAGS, latent_sample_type='posterior', 
                                         eval_episodes=16//env.get_num_modes(), sampling_method=FLAGS.sampling_method, fix_mode=FLAGS.fix_mode)
        exit(0)

    def sample_from_dataset(dataset, batch_size):
        idx = np.random.randint(0, len(dataset["observations"]), size=batch_size)
        s = torch.FloatTensor(dataset["observations"][idx]).to(FLAGS.device)
        a = torch.FloatTensor(dataset["actions"][idx]).to(FLAGS.device)
        r = torch.FloatTensor(dataset["rewards"][idx]).to(FLAGS.device)
        next_s = torch.FloatTensor(dataset["next_observations"][idx]).to(FLAGS.device)
        dones_float = torch.FloatTensor(dataset["dones_float"][idx]).to(FLAGS.device)
        return s, a, r, next_s, dones_float
    
    
    # posterior_z = None    
    # if FLAGS.reward_sample_mode == 'posterior':
    #     posterior_z = []
    #     for batch in train_loader:
    #         observations = batch["observations"].to(device).float()
    #         observations_2 = batch["observations_2"].to(device).float()
    #         labels = batch["labels"].to(device).float()
    #         with torch.no_grad():
    #             z, _ = reward_model.encode(observations, observations_2, labels)
    #         posterior_z.append(z)   
    #     posterior_z = torch.cat(posterior_z, dim=0) 
    #     print(f'posterior_z: {posterior_z.shape}')

    rew_mean_std = None
    def dyna_reward_sampling(reward_model, states ,next_states, actions, rew_mean_std=None):
        batch_size = states.shape[0]
        with torch.no_grad():
            if FLAGS.reward_sample_mode == 'posterior':
                idx = np.random.randint(0, len(reward_model.biased_latents), (batch_size,))
                batch_z = torch.from_numpy(np.array(reward_model.biased_latents)[idx]).float().to(device) #temp0911
                # idx = torch.randint(0, len(posterior_z), (batch_size,))
                # batch_z = posterior_z[idx]
            else:
                batch_z = reward_model.sample_prior(batch_size)
            aug_states = torch.cat([states, batch_z], dim=-1)
            aug_next_states = torch.cat([next_states, batch_z], dim=-1)
            rewards = reward_model.get_reward(torch.cat([states[..., :obs_dim], batch_z], dim=-1), actions)

            if FLAGS.vae_norm=='mean':
                if rew_mean_std is None:  #temp
                    rew_mean_std = (rewards.mean(), rewards.std())
                    print('rew_mean_std: ', rew_mean_std)
                rewards = (rewards - rew_mean_std[0]) / rew_mean_std[1]

            return aug_states, aug_next_states, rewards.squeeze(-1), rew_mean_std

    if FLAGS.reward_sample_mode != 'static':
        states, actions, rewards, next_states, dones = sample_from_dataset(dataset, 10000)
        _, _, _, rew_mean_std = dyna_reward_sampling(reward_model, states, next_states, actions)

    best_criteria = None
    v_loss_history = []
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        states, actions, rewards, next_states, dones = sample_from_dataset(dataset, FLAGS.iql_batch_size)
        if FLAGS.reward_sample_mode != 'static':
            states, next_states, rewards, _ = dyna_reward_sampling(reward_model, states, next_states, actions, rew_mean_std)

        update_info = agent.learn((states, actions, rewards, next_states, dones))
        v_loss_history.append(update_info['v_loss'])
        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wb_logger.log(train_metrics, step=i)
            print(train_metrics, flush=True) 

        if i % FLAGS.eval_interval == 0:

            # check the value explosion and early terminate if so
            WINDOW_SIZE = 20000
            threshold = 1.1
            avg_loss1 = np.mean(v_loss_history[-WINDOW_SIZE*3:-WINDOW_SIZE*2])
            avg_loss2 = np.mean(v_loss_history[-WINDOW_SIZE*2:-WINDOW_SIZE])
            avg_loss3 = np.mean(v_loss_history[-WINDOW_SIZE:])
            print(f'Value Explosion Detection: avg_loss1: {avg_loss1}, avg_loss2: {avg_loss2}, avg_loss3: {avg_loss3}')
            if i>1e5 and avg_loss3>threshold*avg_loss2:
                print('Detect Value Explosion and Early terminate the run')
                break

            # Sampling evaluation
            eval_metrics = defaultdict(list)
            res = latent_sample_evaluate(env, reward_model, preference_dataset, get_adv_func, policy_fn, FLAGS, latent_sample_type='prior', eval_episodes=FLAGS.prior_eval_episodes)
            eval_metrics["prior_reward"] = np.mean(res['utility'])
            eval_metrics["prior_cost"] = np.mean(res['cost'])
            res = latent_sample_evaluate(env, reward_model, preference_dataset, get_adv_func, policy_fn, FLAGS, latent_sample_type='posterior', 
                                         eval_episodes=FLAGS.eval_episodes, sampling_method=FLAGS.sampling_method, fix_mode=FLAGS.fix_mode)
            eval_metrics["posterior_reward"] = np.mean(res['utility'])
            eval_metrics["episode.r"] = np.mean(res['utility'])
            eval_metrics["posterior_cost"] = np.mean(res['cost'])   
            print(eval_metrics, flush=True)
            wb_logger.log({f"sampling/{k}":v for k,v in eval_metrics.items()}, step=i)

            criteria = i

            if best_criteria is None or criteria >= best_criteria:
                torch.save(agent, os.path.join(save_dir, f'IQL_final.pth'))
                print('Save to '+ os.path.join(save_dir, f'IQL_final.pth'))
                best_criteria = criteria

        if i % FLAGS.save_interval == 0 and save_dir is not None: 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(agent, os.path.join(save_dir, f'IQL_{i}.pth'))
            print('Save to '+ os.path.join(save_dir, f'IQL_{i}.pth'))

    # torch.save(agent, os.path.join(save_dir, f'IQL_final.pth'))
    # print('Save to '+ os.path.join(save_dir, f'IQL_final.pth'))

    # Final evaluation
    eval_metrics = defaultdict(list)
    res = latent_sample_evaluate(env, reward_model, preference_dataset, get_adv_func, policy_fn, FLAGS, latent_sample_type='prior', eval_episodes=FLAGS.prior_eval_episodes)
    eval_metrics["prior_reward"] = np.mean(res['utility'])
    eval_metrics["prior_cost"] = np.mean(res['cost'])
    res = latent_sample_evaluate(env, reward_model, preference_dataset, get_adv_func, policy_fn, FLAGS, latent_sample_type='posterior', 
                                 eval_episodes=FLAGS.eval_episodes, sampling_method=FLAGS.sampling_method, fix_mode=FLAGS.fix_mode)
    eval_metrics["posterior_reward"] = np.mean(res['utility'])
    eval_metrics["posterior_cost"] = np.mean(res['cost'])  
    print(eval_metrics, flush=True)
    for k, v in eval_metrics.items():
        wb_logger.log({f"final-sampling-{k}": v})


def main(_):

    FLAGS = absl.flags.FLAGS
    if FLAGS.env in IQL_CONFIG:
        print(f"Loading config for {FLAGS.env} ...")
        for k, v in IQL_CONFIG[FLAGS.env].items():
            setattr(FLAGS, k, v)
        print(f"Updated FLAGS: {IQL_CONFIG[FLAGS.env]}")
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
    train_iql_agent(FLAGS, wb_logger, save_dir)

if __name__ == "__main__":
    absl.app.run(main)
