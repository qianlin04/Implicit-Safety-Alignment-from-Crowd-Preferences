export WANDB_MODE=online
export WANDB_PROJECT=vpl
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Set model_name to be 'gpt2' or 'meta-llama/Llama-2-7b-hf' here
model_name='gpt2'

# Reminder: for controversial settings, please use --num_train_epochs=10

model_type=$1

if [ ${model_type} == "vae" ]
then
# Train VPL on full/controversial/up-sampling Pets dataset
python -m hidden_context.train_llm_vae_preference_model \
        --model_name=${model_name} \
        --data_path="data/simple_pets/gpt2" \
        --num_train_epochs=2 \
        --reward_model_type=vae \
        --data_subset=both \
        --log_dir="logs/gpt2_simple_pets" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --latent_dim 512 \
        --hidden_dim 512 \
        --learning_rate 3e-4 \
        --use_annealing True \
        --kl_loss_weight 1e-4 \
        --fixed_contexts True \
        --fixed_llm_embeddings False \
        --use_last_token_embedding True \
        --up_sampling False \
        --controversial_only False \
        --seed 0
else
# Train baseline models on full/controversial/up-sampling Pets dataset
python -m hidden_context.train_llm_preference_model \
        --model_name=${model_name} \
        --data_path="data/simple_pets/gpt2" \
        --num_train_epochs=2 \
        --reward_model_type=${model_type} \
        --data_subset=both \
        --log_dir="logs/gpt2_simple_pets" \
        --bf16 True \
        --fp16 False \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --controversial_only False \
        --up_sampling False \
        --seed 0
fi
