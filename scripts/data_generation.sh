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

env=SafetyBallCircle-multimodal-with-goal-v0 
seed=0
log_path=${env//-with-goal/}

### collect data with SAC policy
path_to_save_dataset=data/${log_path}.data
python pref_learn/sac_collect.py \
        --seed $seed --env $env --log pref_datasets/$log_path \
        --path_to_save_dataset  $path_to_save_dataset \
        --max_step 2000000 --penalty_step 1000000 --penalty_mode 'switch' \
        > ./logs/${env}_sac_${seed}.txt  2>&1 
seed=$((seed + 1))

### re-train to make value function more precise for argumented data
python pref_learn/sac_collect.py \
        --seed $seed --env $env --log pref_datasets/$log_path \
        --load_offline_dataset $path_to_save_dataset --reset_by_mode 1 \
        --max_step 3000000 \
        > ./logs/${env}_sac_${seed}.txt  2>&1 




num_query=5000
task_list=( 
        "SafetyBallCircle-multimodal-v0" 
        # "SafetyBallRun-multimodal-v0" 
        # "SafetyBallReach-multimodal-v0" 
        # "SafetyAntVelocity-multimodal-v0" 
        # "SafetyHalfCheetahVelocity-multimodal-v0" 
        # "SafetySwimmerVelocity-multimodal-v0" 
        )

for env in ${task_list[@]}
do
        if [[ ${env} =~ .*SafetyBallReach.* ]]; then
                traj_len=16
                set_len=32
        else
                traj_len=64
                set_len=16
        fi
        python -m pref_learn.create_dataset_by_adv --num_query=$num_query --env=$env --query_len=$traj_len --set_len=$set_len --label_by_adv=True > nohup.out 2>&1
        python -m pref_learn.create_dataset_by_adv --num_query=$num_query --env=$env --query_len=$traj_len --set_len=$set_len --label_by_adv=False --trajectory_clip=False > nohup.out 2>&1
done




