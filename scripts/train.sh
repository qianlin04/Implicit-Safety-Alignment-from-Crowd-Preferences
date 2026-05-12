set -e

# Create necessary directories if they don't exist
DIRS=(
  "data"
  "logs"
  "policy_model_repo"
  "reward_model_repo"
  "pref_datasets"
)
for dir in "${DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
    echo "Created directory: $dir"
  else
    echo "Directory already exists: $dir"
  fi
done

n_sample=5000
model_type=${2:-VAE}
task_list=("SafetyBallCircle-multimodal-v0" "SafetyBallRun-multimodal-v0" "SafetyBallReach-multimodal-v0" "SafetyAntVelocity-multimodal-v0" "SafetyHalfCheetahVelocity-multimodal-v0" "SafetySwimmerVelocity-multimodal-v0") 
gpu_list=(0 0 0 0 0 0)
id_list=(
        0          #Circle
        0          #Run
        0          #Reach
        0          #Ant
        0          #HalfCheetah
        0          #Swimmer
)
export WANDB_PROJECT='SafeCrowdPref'
export WANDB_GROUP=$model_type
 
i=${1:-0}  # Specify different script parameters to select the task.

dataset_type="expert_uniform"
env=${task_list[$i]}

if [[ ${env} =~ .*SafetyBallReach.* ]]; then
        traj_len=16
        set_len=32
else
        traj_len=64
        set_len=16
fi

if [[ ${model_type} == VAEPolicy ]]; then
        comment="vae-cpl_q${traj_len}"
        label_by_adv=1
else
        comment="vae-iql_q${traj_len}"
        label_by_adv=0
fi


dataset_path="pref_datasets/${env}/$( [ "$label_by_adv" -eq 1 ] && echo relabelled_queries_by_adv_num${n_sample} || echo relabelled_queries_num${n_sample} )_q${traj_len}_s${set_len}"
config="configs/bulletsafetygym_config.py"

seed=${id_list[$i]}
export CUDA_VISIBLE_DEVICES=${gpu_list[$i]}
echo ./logs/${comment}_${env}_${seed}.txt

#skill discovery
if [[ ${model_type} == VAEPolicy ]]; then
        load_reward_model_path="" 
        python pref_learn/vpl_cpl_v2.py \
                --logging.online=True \
                --comment=$comment --n_sample $n_sample \
                --env=$env \
                --dataset_path=$dataset_path --traj_len=$traj_len \
                --model_type=$model_type \
                --logging.output_dir="reward_model_repo" \
                --seed $seed \
                --learned_prior=True \
                --use_annealing=True --kl_weight 0.001 --lr 0.0003 --annealer_cycles 4 \
                --n_epochs 600 --latent_dim 4 --batch_size 128 --hidden_dim 256 \
                --bc_steps 0 --max_steps 500000 \
                --eval_interval 25000 --prior_eval_episodes 20 --eval_episodes 20 \
                --load_reward_model_path "$load_reward_model_path" \
                --label_by_adv=$label_by_adv --state_noise_scale 0.0 \
                --cpl_bc_coeff 0.0 --cpl_lr 0.0003 \
                --access_to_mode=False --sampling_method='posterior' \
                > ./logs/${comment}_${env}_${seed}.txt 2>&1 
else
        load_reward_model_path="" 
        python pref_learn/vpl_iql.py \
                --logging.online=True \
                --comment=$comment --n_sample $n_sample  \
                --env=$env \
                --dataset_path=$dataset_path --traj_len=$traj_len \
                --model_type=$model_type \
                --logging.output_dir="reward_model_repo" \
                --seed $seed \
                --learned_prior=True \
                --use_annealing=True --kl_weight 0.001 --lr 0.0003 --annealer_cycles 4 \
                --n_epochs 600 --latent_dim 4 --batch_size 256 --hidden_dim 256\
                --config=$config \
                --eval_interval 25000 --prior_eval_episodes 20 --eval_episodes 20\
                --fix_mode=-1 --vae_sampling=True \
                --load_reward_model_path "$load_reward_model_path" \
                --label_by_adv=$label_by_adv --add_action_to_decoder=True --iql_train_on_preference=False  \
                --reward_sample_mode='prior' --sampling_method='posterior' --vae_norm 'mean' --iql_batch_size 512 \
                --test_only=False \
                > ./logs/${comment}_${env}_${seed}.txt 2>&1 
fi



#offline downstream training
ds_env_name=${env/multimodal/downstream}
ckpt_dir="reward_model_repo/$env/$model_type/$comment/s${seed}"
max_policy_action=3.0
algo=TD3_BC
regularization_weight=1.0   
echo logs/${comment}_${ds_env_name}_${seed}_ds.txt


python pref_learn/downstream.py \
        --comment=$comment \
        --env=$ds_env_name \
        --model_type=$model_type --algo $algo \
        --logging.output_dir="wandb" \
        --seed $seed  \
        --eval_episodes 200  \
        --ckpt $ckpt_dir \
        --batch_size 512 --hidden_dim 256  --tau 0.005 \
        --q_latent_sample_type 'uniform'  \
        --logging.online=True \
        --latent_action_tau 0.0  --regularization_weight $regularization_weight --max_policy_action $max_policy_action --lr 0.0003 \
        --cql_weight 1.0 --use_cql_loss=False --use_low_level_policy=True --control_interval 1 \
        --test_only=False \
        > logs/${comment}_${ds_env_name}_${seed}_ds.txt 2>&1  

# online downstream training
seed=$((seed + 1))
algo=TD3
regularization_weight=0.01
sac_update_per_step=1
python pref_learn/downstream.py \
        --comment=$comment \
        --env=$ds_env_name \
        --model_type=$model_type --algo $algo \
        --logging.output_dir="wandb" \
        --seed $seed  \
        --eval_episodes 200  \
        --ckpt $ckpt_dir \
        --batch_size 512 --hidden_dim 256  --tau 0.005 \
        --q_latent_sample_type 'uniform'  \
        --logging.online=True \
        --regularization_weight $regularization_weight --max_policy_action $max_policy_action --lr 0.0003 \
        --cql_weight 1.0 --use_cql_loss=False --use_low_level_policy=True --control_interval 1 \
        --test_only=False  --sac_update_per_step=$sac_update_per_step \
        > logs/${comment}_${ds_env_name}_${seed}_ds.txt 2>&1  
