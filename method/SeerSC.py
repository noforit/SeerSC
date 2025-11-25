
import math
from collections import defaultdict
from method.math.parser import extract_answer



def calc_entropy(preds, probs, base=2.0):

    if len(preds) != len(probs):
        raise ValueError("preds and probs length mismatch")

    weights = defaultdict(float)
    for pred, prob in zip(preds, probs):
        p = max(float(prob), 0.0)
        weights[str(pred).strip()] += p

    total = sum(weights.values())
    if total <= 0:
        return 0.0

    log_base = math.log(base)
    H = 0.0
    for w in weights.values():
        p = w / total
        if p > 0:
            H -= p * (math.log(p) / log_base)
    return H



def calculate_avg_prob(logprobs_list, policy="avg2"):
    """
    Compute confidence score from logprobs.
    policy = "min" | "avg1" | "avg2"
    """
    if not logprobs_list:
        return 0.0

    start_idx = 0
    end_idx = len(logprobs_list)

    # Detect "boxed" token to skip
    for i, d in enumerate(logprobs_list):
        token = list(d.items())[0][0]
        if token == "boxed":
            start_idx = i + 1
            break

    probs = []
    for i in range(start_idx, end_idx):
        d = logprobs_list[i]
        p = math.exp(list(d.items())[0][1])
        probs.append(p)

    if not probs:
        return 0.0

    if policy == "min":
        return min(probs)
    elif policy == "avg1":
        return sum(probs) / len(probs)
    elif policy == "avg2":
        s = sum(math.log(max(p, 1e-12)) for p in probs)
        return math.exp(s / len(probs))
    else:
        raise ValueError(f"Unknown policy: {policy}")



def seersc_select_sample_num(client, prompt, sampling_params_seersc, args, cfg):
    """
    Use SeerSC "system-1" sampling to estimate:
        - entropy of predictions
        - confidence score

    Then choose an appropriate sample number for system-2.
    """

    preds = []
    probs = []


    if cfg["mode"] == "thinking":
        completions = client.completions.create(
            prompt=prompt[0] + "\n</think>\nSo, the final answer is \\",
            **sampling_params_seersc
        )
    if cfg["mode"] == "nothink":
        completions = client.completions.create(
            prompt=prompt[0] + "So, the final answer is \\",
            **sampling_params_seersc
        )

    for choice in completions.choices:
        pred = extract_answer(choice.text, args.dataset_type)
        preds.append(pred)

        logprobs_trace = choice.logprobs.top_logprobs
        probs.append(calculate_avg_prob(logprobs_trace))

    entropy = calc_entropy(preds, probs)

    if args.dataset_type == "gpqa":
        if entropy < 0.2:
            return 1
        elif entropy < 0.7:
            return cfg["sample_num"] // 2
        else:
            return cfg["sample_num"]

    if entropy < 0.6:
        return 1
    elif entropy < 2.0:
        return cfg["sample_num"] // 2
    else:
        return cfg["sample_num"]
