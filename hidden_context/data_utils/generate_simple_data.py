# This file is used to generate synthetic language dataset
from typing import cast

from transformers import (
    HfArgumentParser,
)

import torch
import random

from hidden_context.data_utils.data_processing import (
    ScriptArguments,
    generate_embeddings_with_llm,
    generate_contexts
)

from hidden_context.data_utils.simple_templates import *

from hidden_context.train_llm_preference_model import (
    DataSubset,
    get_hh_rlhf_dataset,
)

import numpy as np


def generate_synthetic_dataset(args):
    data_subset = cast(DataSubset, args.data_subset)
    input_dataset = get_hh_rlhf_dataset(
        data_subset,
        args.data_split,
        args.dataset_size,
        data_path=args.data_path,
        use_subset_as_dir=True
    )
    def generate_simple_data_point(example):
        prompt_length = 1
        prompt = 'Human: Please talk about one kind of pets.'
        if args.data_split == 'train':
            A = bird_sentences[:80]
            B = dog_sentences[:80]
            C = cat_sentences[:80]
        else:
            A = bird_sentences[80:]
            B = dog_sentences[80:]
            C = cat_sentences[80:]
        pair_type = np.random.randint(3) 

        if pair_type == 0:
            chosen = np.random.choice(A) if script_args.data_subset == 'helpful' else np.random.choice(B)
            rejected = np.random.choice(B) if script_args.data_subset == 'helpful' else np.random.choice(A)
        elif pair_type == 1:
            chosen = np.random.choice(A) if script_args.data_subset == 'helpful' else np.random.choice(C)
            rejected = np.random.choice(C) if script_args.data_subset == 'helpful' else np.random.choice(A)
        elif pair_type == 2:
            chosen = np.random.choice(B) if script_args.data_subset == 'helpful' else np.random.choice(C)
            rejected = np.random.choice(C) if script_args.data_subset == 'helpful' else np.random.choice(B) 

        is_harmful = np.random.randint(2)
        if is_harmful:
            harmful = np.random.choice(harmful_sentences)
            if np.random.randint(2) == 0:
                harmful_chosen = chosen + " " +harmful
                chosen, rejected = rejected, harmful_chosen
            else:
                harmful_rejected = rejected + " " + harmful
                chosen, rejected = chosen, harmful_rejected
            
            
        chosen_repeated = ' '.join([chosen] * prompt_length)
        rejected_repeated = ' '.join([rejected] * prompt_length)
        return_dict = {'prompt': prompt, 'chosen': prompt + '\n\n' + 'Assistant: ' + chosen_repeated,
                       'rejected': prompt + '\n\n' + 'Assistant: ' + rejected_repeated}
        if example['label'] == 0:
            return_dict['responses'] = [chosen_repeated, rejected_repeated]
        else:
            return_dict['responses'] = [rejected_repeated, chosen_repeated]
        if not is_harmful:
            return_dict['controversial'] = True
            return_dict['harmful'] = False
        else:
            return_dict['controversial'] = False
            return_dict['harmful'] = True
        return return_dict

    input_dataset = input_dataset.map(generate_simple_data_point)
    print(len(input_dataset.filter(lambda x: x['controversial'] == True)))
    return input_dataset


if __name__ == "__main__":
    # default setting on synthetic language dataset, please iterate over data subsets and data splits
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    dataset = generate_synthetic_dataset(script_args)
    if script_args.with_embeddings:
        dataset = generate_embeddings_with_llm(script_args, dataset)
    generate_contexts(script_args, dataset)
