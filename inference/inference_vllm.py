import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from datasets import load_from_disk
import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import transformers
import sys
import yaml


import concurrent.futures
from openai import OpenAI

from method.SeerSC import seersc_select_sample_num
from method.ESC import ESC_check
from method.ASC import ASC_check
from method.math.parser import extract_answer



MAX_INT = sys.maxsize
import pdb

def generate_dataset(dataset_path, messages_template):
    query = []
    with open(dataset_path) as f:
        for line in f.readlines():
            query.append(json.loads(line))
    
    messages = []
    for line in query:
        message_line = messages_template.copy()
        message_line[1]['content'] = line['problem']
        messages.append(message_line)
    
    return messages


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def vllm_test(args, cfg, model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model,
    )
    
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    gsm8k_ques_items = []

    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if args.dataset_type == 'gpqa':
                query_prompt = '\nYou should think step-by-step and put your final choice letter within \\boxed{}.'
            else:
                query_prompt = '\nYou should think step-by-step and put your final answer within \\boxed{}.'
            try:
                messages_template = [
                    {"role": "user", "content": item['question']+query_prompt}
                ]

            except:

                messages_template = [
                    {"role": "user", "content": item['problem']+query_prompt}
                ]
            if cfg["mode"] == "thinking":
                temp_instr = tokenizer.apply_chat_template(messages_template, tokenize=False, add_generation_prompt=True)
            else:
                temp_instr = tokenizer.apply_chat_template(messages_template, tokenize=False, add_generation_prompt=True, enable_thinking=False)

            sample_num = cfg["sample_num"]

            gsm8k_ins.extend([temp_instr])
            gsm8k_ques_items.extend([item])
            try:
                temp_ans = item['solution']
            except:
                temp_ans = item['answer']
            gsm8k_answers.extend([temp_ans])

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_ques_items = gsm8k_ques_items[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('dataset ==== ', data_path)
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = gsm8k_ins

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]

    result = []
    res_completions = []

    print(len(batch_gsm8k_ins))


    sampling_params = SamplingParams(n=1, temperature=cfg['temperature'], top_p=0.95, max_tokens=cfg['max_len'], stop=stop_tokens)


    if args.method == 'ESC':
        method_check = ESC_check
    elif args.method == 'ASC':
        method_check = ASC_check
    elif args.method == 'SeerSC':
        method_check = lambda preds: len(preds) >= args.sample_num
        sampling_params = SamplingParams(n=base_num, temperature=cfg['temperature'], top_p=0.95, max_tokens=cfg['max_len'], stop=stop_tokens)
        
    elif args.method == 'SC':
        method_check = lambda preds: len(preds) >= args.sample_num
        # base_num = min(16, args.sample_num)
        base_num = cfg["sample_num"]
        sampling_params = SamplingParams(n=base_num, temperature=cfg['temperature'], top_p=0.95, max_tokens=cfg['max_len'], stop=stop_tokens)

    print(args.method)


    # with open(args.save_path, 'w') as f:

    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.port}/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    client_models = client.models.list()
    client_model = client_models.data[0].id

    sampling_params_dict = {
        "model": client_model,
        "n": sampling_params.n,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "max_tokens": sampling_params.max_tokens,
        "stop": sampling_params.stop,        
    }

    print('sampleing =====', sampling_params_dict)


    sampling_params_dict_seersc = {
        "model": client_model,
        "n": 64,
        "temperature": cfg['system1_temperature'],
        "top_p": sampling_params.top_p,
        "max_tokens": 8192,
        "stop": [' }', '}\n', '}\n\n', '}.', '}.\n', '}\\', '}}', ')}', ')}.', ')}\n', '} ', "\n\n"],
        'logprobs': 1,       
    }


    def request_vllm(client, prompt, sampling_params_dict):

        if args.method == "SeerSC":

            adaptive_n = seersc_select_sample_num(
                client, prompt, sampling_params_dict_seersc, args, cfg
            )
            sampling_params_dict["n"] = adaptive_n

        preds = []
        res_completions = []
        while True:

            completions =  client.completions.create(
                prompt=prompt,
                **sampling_params_dict,
            )

            generated_texts = [completion.text for completion in completions.choices]
            
            for generated_text in generated_texts:
                pred = extract_answer(generated_text, args.dataset_type)
                preds.append(pred)
                    
                res_completions.append(generated_text)        
                

            if method_check(preds) or len(preds) >= args.sample_num:
                break

        progress_bar.update(1)
        
        return res_completions


    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_data = {}
        progress_bar = tqdm(total=len(batch_gsm8k_ins), desc="Processing", dynamic_ncols=True)

        for idx, (prompt, prompt_answer, source_js) in enumerate(
            tqdm(zip(batch_gsm8k_ins, gsm8k_answers, gsm8k_ques_items), total=len(batch_gsm8k_ins))
        ):

            if not isinstance(prompt, list):
                prompt = [prompt]

            future = executor.submit(request_vllm, client, prompt, sampling_params_dict)
            future_to_data[future] = (idx, prompt, prompt_answer, source_js)
            
        results = [None] * len(batch_gsm8k_ins) 
        for future in concurrent.futures.as_completed(future_to_data):
            idx, prompt, prompt_answer, source_js = future_to_data[future]
            res_completions = future.result()  
            results[idx] = (res_completions, source_js)


        progress_bar.close()


        with open(args.save_path, 'w', encoding='utf-8') as f:
            for res_completions, source_js in results:
                for res_completion in res_completions:
                    source_js['model_generation'] = res_completion
                    f.write(json.dumps(source_js, ensure_ascii=False) + '\n')
            
            

    
    # print(f"time: {end_time - start_time:.4f} seconds")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_type", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--save_path",type=str, default='')  # saving path
    parser.add_argument("--port", type=int, default=8000)  # port
    return parser.parse_args()

def merge_dicts(*dicts):
    """
    Merge dictionaries in given order.
    Later dictionaries override earlier ones.
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def load_config(config_path, model_name, dataset_name, method):

    # Load YAML file
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Fetch model/dataset configs
    model_cfg = cfg["models"].get(model_name)
    dataset_cfg = cfg["datasets"].get(dataset_name)

    if model_cfg is None:
        raise ValueError(f"Model '{model_name}' not found in config.yaml")

    if dataset_cfg is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.yaml")

    # Optional model-dataset overrides
    override_cfg = (
        cfg.get("model_dataset_override", {})
          .get(model_name, {})
          .get(dataset_name, {})
    )

    # Merge in correct priority:
    # model < dataset < override
    merged = merge_dicts(model_cfg, dataset_cfg, override_cfg)

    # Final normalized config
    result = {
        "model_path": model_cfg["path"],        # model path never overridden
        "dataset_path": dataset_cfg["path"],    # dataset path never overridden
        "max_len": merged.get("max_len", 8192),
        "sample_num": merged.get("sample_num", 1),
        "temperature": merged.get("temperature", 1.0),
        "tensor_parallel_size": merged.get("tensor_parallel_size", 1),
        "mode": merged.get("mode", "thinking"),
        "system1_temperature": merged.get("system1_temperature", 0.5),
        "method": method,
    }

    return result



if __name__ == "__main__":
    args = parse_args()

    cfg = load_config(args.config, args.model_name, args.dataset_name, args.method)

    print("Loaded config:", cfg)

    vllm_test(
        args=args,
        cfg=cfg,
        model=cfg["model_path"],
        data_path=cfg["dataset_path"],
        start=0,
        end=MAX_INT,
        batch_size=1,
        tensor_parallel_size=cfg["tensor_parallel_size"]
    )