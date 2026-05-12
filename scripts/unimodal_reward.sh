n_sample=5000
access_to_real_task_reward=1
augment_reward_normalize=0
# if [ "$access_to_real_task_reward" -eq 1 ]; then
#     augment_reward_normalize=0
# else
#     augment_reward_normalize=1
# fi

task_list=( "SafetyBallCircle-multimodal-v0" "SafetyBallRun-multimodal-v0" "SafetyBallReach-multimodal-v0" "SafetyAntVelocity-multimodal-v0" "SafetyHalfCheetahVelocity-multimodal-v0" "SafetySwimmerVelocity-multimodal-v0" ) #'MO-Swimmer-v2' 'MO-Ant-v2' 'MO-HalfCheetah-v2')
gpu_list=(0 0 0 0 0 0)
id_list=(
        120438          #Circle
        120438    #Run
        120438         #Reach
        120438     #Ant
        120438         #HalfCheetah
        120438          #Swimmer
)
export WANDB_PROJECT='PA_MORL'
export WANDB_GROUP=$model_type
 
i=${1:-0}

env=${task_list[$i]}
if [[ ${env} =~ .*SafetyBallReach.* ]]; then
        traj_len=16
        set_len=32
else
        traj_len=64
        set_len=16
fi

comment="unimodal_q${traj_len}"
label_by_adv=0
dataset_type="expert_uniform"
#seed=${id_list[$i]}
seed=${2}
export CUDA_VISIBLE_DEVICES=${gpu_list[$i]}
dataset_path="pref_datasets/${env}/$( [ "$label_by_adv" -eq 1 ] && echo relabelled_queries_by_adv_num${n_sample} || echo relabelled_queries_num${n_sample} )_q${traj_len}_s${set_len}"

# python pref_learn/train_unimodal_reward.py \
#     --logging.online=True \
#     --comment=$comment --n_sample $n_sample  \
#     --env=$env \
#     --dataset_path=$dataset_path --traj_len=$traj_len \
#     --model_type=$model_type \
#     --logging.output_dir="reward_model_repo" \
#     --seed $seed \
#     --n_epochs 1000 --batch_size 256 --hidden_dim 256\
#     --label_by_adv=$label_by_adv --add_action_to_decoder=True --access_to_real_task_reward=$access_to_real_task_reward \
#     > ./logs/${comment}_${env}_${seed}.txt 2>&1 
echo ./logs/${comment}_${env}_${seed}.txt


#downstream phase
ds_env_name=${env/multimodal/downstream}
augment_reward_path="reward_model_repo/$env/$model_type/$comment/s${seed}"
for augment_reward_weight in 0.5 #0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python pref_learn/downstream.py \
        --comment=$comment \
        --env=$ds_env_name \
        --model_type=$model_type --algo 'TD3_BC' \
        --logging.output_dir="wandb" \
        --seed $seed  \
        --eval_episodes 200  \
        --augment_reward_path $augment_reward_path \
        --batch_size 512 --hidden_dim 256  --tau 0.005 \
        --q_latent_sample_type 'uniform'  \
        --logging.online=True \
        --latent_action_tau 0.0 --lr 0.0003 \
        --use_low_level_policy=False --control_interval 1 \
        --test_only=True --augment_reward_weight $augment_reward_weight \
        --augment_reward_normalize=$augment_reward_normalize \
        > logs/${comment}_${env}_${seed}_w=${augment_reward_weight}_test_only.txt 2>&1  
    echo logs/${comment}_${env}_${seed}_w=${augment_reward_weight}_test_only.txt
    seed=$((seed + 1))
done
