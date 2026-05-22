
# Set model_type to be 'gpt2' or 'llama' here
model_type=$1

# Generate Pets dataset
 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset helpful --data_split test --dataset_size 200

 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset helpful --data_split train --dataset_size 4000

 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset harmless --data_split test --dataset_size 200

 python -m hidden_context.data_utils.generate_simple_data --output_dir data/simple_pets/ \
 --data_path data/relabeled_hh_rlhf --with_embeddings True --synthetic_dataset True \
 --model_type ${model_type} --data_subset harmless --data_split train --dataset_size 1000
