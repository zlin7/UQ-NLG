
import argparse
import glob
import json
import os

import pandas as pd
import torch
import tqdm
import transformers

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=20)
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--nprocess', type=int, default=None)


args = parser.parse_args()

_UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset

def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config

@torch.no_grad()
def get_generations(model_name:str, args, seed=10, old_sequences=None, max_num_gen_once=4):
    device = args.device

    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            most_likely_generations = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                                    num_beams=1,
                                                    do_sample=False,
                                                    generation_config=generation_config).cpu()[0, input_length:]
        generations = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            _ =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                     num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                                     do_sample=True, top_p=args.top_p, top_k=args.top_k,
                                     temperature=args.temperature, generation_config=generation_config,
                                     )
            generations.append(_[:, input_length:].cpu())
            num_gens -= len(_)

        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        # remember the data
        curr_seq = dict(
            prompt=input_ids.cpu()[0],
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )
        if args.dataset == 'coqa':
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
    return sequences

def get_generations_bb(model_name:str, args, seed=10, old_sequences=None, task_runner:utils.TaskPartitioner=None):
    # ='gpt-3.5-turbo'
    dataset = get_dataset_fn(args.dataset)(_UNUSED_TOKENIZER)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}
    sequences = []
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue
        generated_texts = []
        for _ in range(args.num_generations_per_prompt):
            if task_runner is None:
                generated_texts.append(models.openai_query(batch['prompt'][0], model=model_name, attemptd_id=_, max_tries=50))
            else:
                task_runner.add_task(models.openai_query, batch['prompt'][0], model=model_name, attemptd_id=_, max_tries=50)
        if task_runner is not None:
            continue
        curr_seq = dict(
                prompt=batch['prompt'][0],
                id=batch['id'][0], #NOTE: This changed
                question=batch['question'][0],
                answer=batch['answer'][0], #NOTE: This changed
                additional_answers=[],
        )
        curr_seq.update(
                dict(
                    generations=generated_texts,
                )
            )

        if args.dataset == 'coqa':
            curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]
        sequences.append(curr_seq)
    return task_runner or sequences

def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.seed}')
        os.makedirs(cache_dir, exist_ok=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)


        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    if model_name == 'gpt-3.5-turbo':
        task_runner = None if parallel is None else utils.TaskPartitioner()
        sequences = get_generations_bb(model_name, args, seed=args.seed, old_sequences=old_sequences, task_runner=task_runner)
        if task_runner is not None:
            # caching down the results but will need to run again without parallelisation to save the result in a pkl file
            return task_runner.run_multi_process(parallel)
        print(f'Writing {len(sequences)} generations to {cache_dir}...')
        pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
        return
    else:
        sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
        print(f'Writing {len(sequences)} generations to {cache_dir}...')
        pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
        return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)