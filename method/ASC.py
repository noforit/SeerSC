import time
from collections import Counter
from method.adaptive_consistency import AC, BetaStoppingCriteria


def calculate_ASC_correctness_jsonl(samples, n, beta=0.95, max_gens=40):


    lines = len(samples)
    per_question = lines // n 

    asc_correctness_list = []
    asc_steps_list = []

    ac = AC(stop_criteria=BetaStoppingCriteria(beta), max_gens=per_question)

    for i in range(n):
        selected_samples = samples[i * per_question : (i + 1) * per_question]
        preds = [s['pred'][0] for s in selected_samples]

        for j in range(len(preds)):
            if ac.should_stop(preds[:j+1]):
                break

        majority_vote = Counter(preds[:j+1]).most_common(1)[0][0]

        correct = 0
        for s in selected_samples:
            if s['pred'][0] == majority_vote:
                correct = 1 if s['score'][0] else 0
                break

        asc_correctness_list.append(correct)
        asc_steps_list.append(len(preds[:j+1]))

    avg_samples = sum(asc_steps_list) / n
    acc = sum(asc_correctness_list) / n


    return asc_correctness_list, asc_steps_list, acc, avg_samples



ac = AC(stop_criteria=BetaStoppingCriteria(0.95), max_gens=32)

def ASC_check(preds, beta=0.95):


    return ac.should_stop(preds)