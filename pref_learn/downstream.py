import os
from collections import defaultdict

import absl.app
import absl.flags
import gym
import numpy as np
import torch
import wandb

import jaxrl_m.envs
from jaxrl_m.envs import make_env
from pref_learn.utils.utils import (
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    WandBLogger,
    prefix_metrics,
)
from gym.wrappers import RecordEpisodeStatistics
from functools import partial
import jaxrl_m.learners.d4rl_utils as d4rl_utils
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import d4rl
import tqdm
import pickle
from configs.customize_config import DOWNSTREAM_BASELINE_CONFIG, DOWNSTREAM_CONFIG

FLAGS_DEF = define_flags_with_default(
    env="maze2d-target-v0",  # can change
    dataset_type='expert_uniform',
    comment="",
    algo="TD3_BC", # TD3_BC, Diffusion_QL, IQL, SAC
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
    # Training
    log_interval=1000,
    eval_interval=20000,
    save_interval=50000,
    max_steps=5000000,
    device="cuda",
    # Dataset
    dataset_path="",
    logging=WandBLogger.get_default_config(),
    seed=42,
    # plotting
    debug_plots=False,
    plot_observations=False,
    reward_scaling=1.0,
    # biased
    biased_mode="grid",
    eval_episodes=20,
    label_by_adv=True,
    sampling_method='random',
    ckpt="",
    low_policy_type="VAE",
    vae_sampling=False,
    control_interval=-1,
    use_cql_loss=True,
    regularization_weight=1.0,
    latent_perturb_scale = 1.0,
    cql_weight=1.0,
    target_info='[1,0]',
    latent_sample_num=1,
    max_q_backup=True,
    tau=0.005,
    fix_std=-1.0,
    q_latent_sample_type='prior', # prior or policy'
    latent_action_tau=0.0,
    use_low_level_policy=True,
    bc_alpha=1.0,
    max_policy_action=3.0,
    test_only=False,
    augment_reward_path="",
    augment_reward_weight=0.0,
    augment_reward_normalize=True,
    sac_update_per_step=1,
    access_to_real_cost=False,
)

def log_metrics(metrics, epoch, logger):
    for key, val in metrics.items():
        if isinstance(val, list):
            metrics[key] = np.mean(val)
    logger.log(metrics, step=epoch)

def load_reward_model(ckpt):
    with open(os.path.join(ckpt, "best_model.pt"), "rb") as f:
        reward_model = torch.load(f, weights_only=False)
    return reward_model

def evaluate(env, policy_fn, reward_model, eval_runs=10, control_interval=-1, reference_points=None): 
    """
    Makes an evaluation run with the current policy
    """
    FLAGS = absl.flags.FLAGS
    if not reward_model is None:
        print('Use low-level policy with latent prior: ', reward_model.mean, torch.exp(0.5*reward_model.log_var), flush=True)
        if hasattr(reward_model, 'biased_latents'):
            for mode, latent in enumerate(reward_model.biased_latents):
                print(f'mode: {mode},    avg posterior latent: {latent}')
    else:
        print('No low-level policy')
    

    human_rewards = []
    utility_list, obs_list, env_pref_list, rew_vec_list, cost_list, target_list = [], [], [], [], [], []
    all_hist_for_test = []
    eval_idx = 0
    while eval_idx < eval_runs:
        state = env.reset()
        # state = env.env.env.env.reset_to_location(np.array(init_pos_list[eval_idx]))

        latent = None
        rew_vec = 0
        obs_hist, hist_for_test = [], defaultdict(list)
        human_r, target_r, t, cost = 0, 0, 0, 0
        reach_goal = None
        while True:
            FLAGS = absl.flags.FLAGS
            if FLAGS.test_only and 0:
                if t == 0 :
                    if 'Run' in env.spec.id:
                        env.set_mode(min(49, 50-eval_idx*5))
                    elif 'Ant' in env.spec.id:
                        env.set_mode(5*eval_idx)


            action, latent = policy_fn(state, latent_action=latent)            
            state, reward, done, info = env.step(action)
            human_r = 0
            reach_goal = None

            FLAGS = absl.flags.FLAGS
            if FLAGS.test_only and 0:
                import time
                #time.sleep(0.1)
                if 'Run' in env.spec.id:
                    # if t == 0:
                    #     env.set_mode(2)
                    hist_for_test['vel'].append(info['vel'])
                    hist_for_test['z'].append(latent.squeeze(0).cpu().detach().numpy())
                    hist_for_test['pos'].append(info['pos'])
                    
                    #action, latent = policy_fn(state, latent_action=latent)
                elif 'Ant' in env.spec.id:
                    hist_for_test['vel'].append(np.array([info['x_velocity'], info['y_velocity']]))
                    hist_for_test['z'].append(latent.squeeze(0).cpu().detach().numpy())
                    hist_for_test['pos'].append(info['pos'])
                    bias = np.array(reward_model.biased_latents)
                    lat = reward_model.mean.reshape(1, -1) + (0.5*reward_model.log_var).exp().reshape(1, -1) * latent * FLAGS.latent_perturb_scale
                else:
                    bias = np.array(reward_model.biased_latents)
                    lat = reward_model.mean.reshape(1, -1) + (0.5*reward_model.log_var).exp().reshape(1, -1) * latent * FLAGS.latent_perturb_scale
                    print(lat)
                    print(bias)
                    dis = np.linalg.norm(bias-lat.cpu().detach().numpy(), axis=-1)
                    print(dis, np.argmin(dis), flush=True)
                    # if t==0:
                    #     env.set_mode(eval_idx)
                    #     state = env.reset()
                    # print(state[-2:])

                if done:        
                    if 'Ant' in env.spec.id and target_r<1600:
                        eval_idx-=1
                        break            
                    all_hist_for_test.append({'hist_for_test': hist_for_test, 'target': env.target})
                    if eval_idx == 10 and 'Run' in env.spec.id or eval_idx == 7 and 'Ant' in env.spec.id:
                        torch.save({'all_hist_for_test': all_hist_for_test, 'biased': np.array(reward_model.biased_latents), 'mean': reward_model.mean.cpu().detach().numpy(), 'std': (0.5 * reward_model.log_var).exp().cpu().detach().numpy()}, f'imgs/example_run.data' if 'Run' in env.spec.id else f'imgs/example_ant.data')
                        exit(0)

            if control_interval!=-1 and t%control_interval == 0:
                latent = None    

            t+=1
            target_r += reward
            rew_vec = info['rew_vec'] + rew_vec
            cost += info.get('cost', 0) 
            obs_hist.append(state)
            if done:
                break


        obs_list.append(np.array(obs_hist))
        env_pref_list.append(env.get_pref())
        target_list.append(env.target)
        rew_vec_list.append(rew_vec)
        utility_list.append(target_r)
        human_rewards.append(human_r)
        cost_list.append(cost)
        eval_idx+=1
            
        print(f"evaluation latent: {latent}, target: {env.target}, target_r: {target_r}, human_rewards: {human_r}, rew_vec: {rew_vec}, cost: {cost}, t: {t}")
        if 'maze2d' in env.spec.id:
            print(f'start pos: {np.round(obs_hist[0][:2], 2)}, reach goal: {reach_goal}')

    os.makedirs("logs/fig", exist_ok=True) 
    res = {
        'human_rewards': np.mean(human_rewards),
        'utility': np.mean(utility_list),
        'cost': np.mean(cost_list),
        'rew_vec_list': rew_vec_list,
        'raw_data': {
            'utility': utility_list,
            'cost': cost_list,
            'pref': env_pref_list,
            'target': target_list,
        }
        }
    return res

def main(_):
    FLAGS = absl.flags.FLAGS
    CUS_CONFIG = DOWNSTREAM_CONFIG if FLAGS.use_low_level_policy else DOWNSTREAM_BASELINE_CONFIG
    if FLAGS.env in CUS_CONFIG:
        print(f"Loading config for {FLAGS.env} ...")
        for k, v in CUS_CONFIG[FLAGS.env].items():
            setattr(FLAGS, k, v)
        print(f"Updated FLAGS: {CUS_CONFIG[FLAGS.env]}")
    else:
        print(f"No config found for {FLAGS.env}, using default FLAGS.")

    if FLAGS.algo=='TD3' or FLAGS.algo=='SAC':  #temp
        FLAGS.max_steps = max(500000 if not 'Ant' in FLAGS.env else 2000000, FLAGS.max_steps)

    env = make_env(FLAGS.env)
    eval_env = make_env(FLAGS.env)
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    save_dir = f'./policy_model_repo/{FLAGS.env}/{FLAGS.comment}/s{FLAGS.seed}' 

    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}"
    assert FLAGS.comment, "You must leave your comment for logging experiment."
    assert not FLAGS.use_low_level_policy or os.path.exists(FLAGS.ckpt) or FLAGS.test_only, "low-level policy path doesn't exist"
    FLAGS.logging.group += f"_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    FLAGS.logging.output_dir = save_dir
    device = FLAGS.device
    #env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    set_random_seed(FLAGS.seed)
    # if hasattr(env, "reward_observation_space"):
    #     obs_dim = env.reward_observation_space.shape[0]
    # else:
    #     obs_dim = env.observation_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    # if "maze" in FLAGS.env:
    #     env.set_biased_mode(FLAGS.biased_mode)
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    if not FLAGS.use_low_level_policy:
        FLAGS.max_policy_action = max_action

    if FLAGS.use_low_level_policy:
        print('Use low-level policy')
        reward_model = load_reward_model(FLAGS.ckpt)
        reward_model = reward_model.to(device)
        latent_dim = reward_model.latent_dim
    else:
        print('No low-level policy')
        reward_model = None
        latent_dim = act_dim

    if FLAGS.use_low_level_policy:
        if FLAGS.model_type=='VAEPolicy':
            cpl_agent_path = os.path.join(FLAGS.ckpt, 'CPL_final.pt')
            cpl_agent = torch.load(cpl_agent_path, weights_only=False).to(device)
            obs_dim_for_policy = cpl_agent.get_policy_input_dim() - latent_dim
            print("Obs dim for low-level policy:", obs_dim_for_policy)
            low_level_policy_fn = lambda s, z, eval=False: cpl_agent.decode(s[..., :obs_dim_for_policy], z)[0]
        elif FLAGS.model_type=='VAE':
            iql_agent_path = os.path.join(FLAGS.ckpt, 'IQL_final.pth')
            iql_agent = torch.load(iql_agent_path, weights_only=False).to(device)
            obs_dim_for_policy = iql_agent.get_policy_input_dim() - latent_dim
            print("Obs dim for low-level policy:", obs_dim_for_policy)
            def low_level_policy_fn(s, z, eval=False):
                return iql_agent.policy.act(torch.concatenate([s[..., :obs_dim_for_policy], z], -1), deterministic=True, enable_grad=True)
        def policy_fn(state, latent_action, eval=False):
            batch_size = latent_action.shape[0]
            latent = reward_model.mean.reshape(1, -1).repeat(batch_size, 1) + (0.5*reward_model.log_var).exp().reshape(1, -1).repeat(batch_size, 1) * latent_action * FLAGS.latent_perturb_scale
            return low_level_policy_fn(state, latent, eval).clamp(-max_action, max_action)
    else:
        def policy_fn(state, action, eval=False):
            return action.clamp(-max_action, max_action)

    def sample_from_dataset(dataset, batch_size):
        idx = np.random.randint(0, len(dataset["observations"]), size=batch_size)
        s = torch.FloatTensor(dataset["observations"][idx]).to(FLAGS.device)
        a = torch.FloatTensor(dataset["actions"][idx]).to(FLAGS.device)
        r = torch.FloatTensor(dataset["rewards"][idx]).to(FLAGS.device).reshape(-1, 1)
        next_s = torch.FloatTensor(dataset["next_observations"][idx]).to(FLAGS.device)
        dones_float = torch.FloatTensor(dataset["terminals"][idx]).to(FLAGS.device).reshape(-1, 1)
        return s, a, r, next_s, dones_float


    if FLAGS.algo=='TD3_BC':
        from algo.td3bc_agent import TD3_BC
        agent = TD3_BC(state_size=obs_dim,
                            action_size=act_dim,
                            policy_size=latent_dim,
                            policy_fn=policy_fn,
                            learning_rate=FLAGS.lr,
                            latent_reg_para=FLAGS.regularization_weight if FLAGS.use_low_level_policy else 0.0,
                            alpha=FLAGS.bc_alpha,
                            max_policy_action=FLAGS.max_policy_action,
                            max_action=max_action,
                            )
        eval_policy_fn=partial(agent.get_action, eval=True)
    elif FLAGS.algo=='Diffusion_QL':
        from algo.diffusion_ql import Diffusion_QL
        agent = Diffusion_QL(state_size=obs_dim,
                             action_size=act_dim,
                             policy_size=latent_dim,
                             policy_fn=policy_fn,
                             learning_rate=FLAGS.lr,
                             latent_reg_para=FLAGS.regularization_weight if FLAGS.use_low_level_policy else 0.0,
                             alpha=FLAGS.bc_alpha,
                             max_policy_action=FLAGS.max_policy_action,
                             max_action=max_action,
                             )
        eval_policy_fn=partial(agent.get_action, eval=True)
    elif FLAGS.algo=='IQL':
        from algo.iql_agent import ImplicitQLearning
        agent = ImplicitQLearning(state_size=obs_dim,
                             action_size=act_dim,
                             policy_size=latent_dim,
                             policy_fn=policy_fn,
                             learning_rate=FLAGS.lr,
                             latent_reg_para=FLAGS.regularization_weight if FLAGS.use_low_level_policy else 0.0,
                             max_policy_action=FLAGS.max_policy_action,
                             max_action=max_action,
                             )
        eval_policy_fn=partial(agent.get_action, eval=True)
    elif FLAGS.algo=='SAC' or FLAGS.algo=='TD3':
        from algo.sac_agent import SacAgent
        agent = SacAgent(obs_dim, act_dim,
                         policy_dim=latent_dim,
                         policy_fn=policy_fn,
                         learning_rate=FLAGS.lr,
                         max_step=FLAGS.max_steps,
                         latent_reg_para=FLAGS.regularization_weight if FLAGS.use_low_level_policy else 0.0,
                         max_policy_action=FLAGS.max_policy_action,
                         max_action=max_action,
                         use_deterministic_policy=FLAGS.algo=='TD3',
                         )
        eval_policy_fn=partial(agent.get_action, eval=True)

    if FLAGS.test_only:
        # env.env.env._disable_render_order_enforcing=True
        # env.render() 
        save_dir = f'./policy_model_repo/{FLAGS.env}/{FLAGS.comment}/s{FLAGS.seed}' 
        print(f'load model from {os.path.join(save_dir, FLAGS.algo)}')
        agent.load(os.path.join(save_dir, FLAGS.algo))
        res = evaluate(eval_env, eval_policy_fn, reward_model, eval_runs=FLAGS.eval_episodes, control_interval=FLAGS.control_interval, reference_points=None)
        human_reward, target_reward, cost = res['human_rewards'], res['utility'], res['cost']
        print({"Human Reward": human_reward, "Target Reward": target_reward, "Cost": cost, "step": 0})
        res['step'] = 0
        with open(os.path.join(save_dir, 'test_results.pkl'), "wb") as f:
            pickle.dump([res], f)
        exit(0)

    if FLAGS.algo != 'SAC' and FLAGS.algo != 'TD3' and not FLAGS.test_only:
        dataset = env.get_dataset_for_downstream()    
        if FLAGS.augment_reward_path!="" and FLAGS.augment_reward_path is not None:
            assert os.path.exists(FLAGS.augment_reward_path)
            augment_reward_model = load_reward_model(FLAGS.augment_reward_path)
            dataset = d4rl_utils.reward_augment(augment_reward_model, dataset, weight=FLAGS.augment_reward_weight, device=device, normalize=FLAGS.augment_reward_normalize)
    
    def sample_online_fn(step=0):
        batch = agent.buffer.sample_batch(batch_size=agent.batch_size)
        states = torch.FloatTensor(batch['sta1']).to(FLAGS.device)
        actions = torch.FloatTensor(batch['acts']).to(FLAGS.device)
        next_states = torch.FloatTensor(batch['sta2']).to(FLAGS.device)
        rewards = torch.FloatTensor(batch['rews']).to(FLAGS.device)
        dones = torch.FloatTensor(batch['done']).to(FLAGS.device)
        if FLAGS.augment_reward_path!="" and FLAGS.augment_reward_path is not None:
            augment_reward = augment_reward_model.get_reward(states, actions).detach()
            augment_reward = min(2*step/FLAGS.max_steps, 1.0) * augment_reward  #temp
            rewards = (1-FLAGS.augment_reward_weight) * rewards + FLAGS.augment_reward_weight * augment_reward
        if FLAGS.access_to_real_cost:
            cost = torch.FloatTensor(batch['cost']).to(FLAGS.device)
            rewards = rewards + cost * getattr(env, 'cost_penalty', 0)            
        return states, actions, rewards, next_states, dones            

    human_average10, target_average10, cost_average10 = deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)
    # evaluate the random policy
    def eval_random_policy_fn(state, latent_action):
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        with torch.no_grad():
            latent_action = torch.randn(size=(1, latent_dim)).to(device) if latent_action is None else latent_action
            action = policy_fn(state, latent_action.to(device), eval=True)
        return action.squeeze(0).detach().cpu().numpy(), latent_action
    reference_points = None
    # res = evaluate(eval_env, eval_random_policy_fn, reward_model, eval_runs=100, control_interval=FLAGS.control_interval)
    # human_reward, target_reward, reference_points, cost = res['human_rewards'], res['utility'], res['rew_vec_list'], res['cost']
    
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)
    # wb_logger.log({"Human Reward": human_reward, "arget Reward": target_reward, "cost": cost}, step=0)
    # print(f"Init human_rewards: {human_reward}, Init target_reward: {target_reward}, cost: {cost}", flush=True)
    all_results = []
    loss_history, detect_count, detect_count_2 = [], 0, 0

    env_state = None
    env_done = True
    target_r = 0
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        if FLAGS.algo!='SAC' and FLAGS.algo!='TD3': #offline training 
            states, actions, rewards, next_states, dones = sample_from_dataset(dataset, FLAGS.batch_size)
            info = agent.learn((states, actions, rewards, next_states, dones))
        else:
            if env_done:
                print(env.target, target_r)
                target_r = 0
                env_state = env.reset()
                env_done = False
                 
            if i<10000:
                if FLAGS.use_low_level_policy:
                    env_action, latent = agent.get_action(env_state, random_latent=True)
                else:
                    env_action = env.action_space.sample()
            else:
                env_action, latent = agent.get_action(env_state)     


            if FLAGS.algo == 'TD3':
                env_action = np.clip(env_action + np.random.normal(0, 0.1*max_action, size=env_action.shape), -max_action, max_action)
            
            env_next_state, env_reward, env_done, env_info = env.step(env_action)
            target_r += env_reward
            agent.remember(env_state, env_next_state, env_action, env_reward, env_done and not 'TimeLimit.truncated' in env_info, env_info)
            env_state = env_next_state

            if i<10000: # warmup
                continue

            for _ in range(FLAGS.sac_update_per_step if FLAGS.use_low_level_policy else 1):
                states, actions, rewards, next_states, dones = sample_online_fn(step=i)
                info = agent.learn((states, actions, rewards, next_states, dones))
        
        loss_history.append(info['critic1_loss'])

        if i % FLAGS.log_interval == 0:
            info['step'] = i
            for k, v in info.items():
                print(k, v)
            print('-------------------\n', flush=True)
            wb_logger.log(info)

        if i % FLAGS.eval_interval == 0:
            # check the value explosion and early terminate if so
            WINDOW_SIZE = 20000
            threshold1 = 1.1
            avg_loss1 = np.mean(loss_history[-WINDOW_SIZE*3:-WINDOW_SIZE*2])
            avg_loss2 = np.mean(loss_history[-WINDOW_SIZE*2:-WINDOW_SIZE])
            avg_loss3 = np.mean(loss_history[-WINDOW_SIZE:])
            print(f'Value Explosion Detection: avg_loss1: {avg_loss1}, avg_loss2: {avg_loss2}, avg_loss3: {avg_loss3}, ratio: {avg_loss3/avg_loss2}, detect_count: {detect_count}, detect_count_2: {detect_count_2}')
            if avg_loss3 < threshold1*avg_loss2:
                detect_count += 1
            if FLAGS.algo == 'TD3_BC' and i>1e5 and avg_loss3>threshold1*avg_loss2 and detect_count>=2 :
                detect_count_2 += 1
                if detect_count_2 >= 3:
                    print('Detect Value Explosion and Early terminate the run')
                    break
            else:
                detect_count_2 = 0


            res = evaluate(env, eval_policy_fn, reward_model, eval_runs=FLAGS.eval_episodes, control_interval=FLAGS.control_interval, reference_points=reference_points)
            human_reward, target_reward, cost = res['human_rewards'], res['utility'], res['cost']
            wb_logger.log({"Human Reward": human_reward, "Target Reward": target_reward, "Cost": cost, "step": i}, step=i)

            human_average10.append(human_reward)
            target_average10.append(target_reward)
            cost_average10.append(cost)
            info["Human Average10"] = np.mean(human_average10)
            info["Target Average10"] = np.mean(target_average10)
            info["Cost Average10"] = np.mean(cost_average10)
            print("Episode: {} | Human Average10: {} | Target Average10: {} | Cost Average10: {} | Human Reward: {} | Target Reward: {} | Cost: {}".format(i, np.mean(human_average10), np.mean(target_average10), np.mean(cost_average10), human_reward, target_reward, cost), flush=True)

            res['step'] = i
            all_results.append(res)
            with open(os.path.join(save_dir, 'results.pkl'), "wb") as f:
                pickle.dump(all_results, f)

        # if (i %10 == 0) and config.log_video:
        #     mp4list = glob.glob('video/*.mp4')
        #     if len(mp4list) > 1:
        #         mp4 = mp4list[-2]
        #         wb_logger.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

        if i % FLAGS.save_interval == 0  and detect_count_2==0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            agent.save(os.path.join(save_dir, FLAGS.algo))
            

    
            

if __name__ == "__main__":
    absl.app.run(main)


