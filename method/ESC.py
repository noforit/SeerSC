import time
from collections import Counter

def calculate_ESC_correctness_jsonl(samples, n, window_size=5):

    lines = len(samples)
    per_question = lines // n  

    es_correctness_list = []
    es_steps_list = []

    for i in range(n):
        selected_samples = samples[i * per_question : (i + 1) * per_question]
        preds = [s['pred'][0] for s in selected_samples]

        steps = window_size - 1
        window_size_adjusted = min(window_size, len(preds))

        stopped = False
        for j in range(len(preds) - window_size_adjusted + 1):
            window = preds[j:j + window_size_adjusted]
            steps += 1
            if all(x == window[0] for x in window):

                vote_result = window[0]
                for s in selected_samples:
                    if s['pred'][0] == vote_result:
                        correct = 1 if s['score'][0] else 0
                        break
                es_correctness_list.append(correct)
                es_steps_list.append(steps)
                stopped = True
                break

        if not stopped:

            vote_result = max(set(preds), key=preds.count)
            for s in selected_samples:
                if s['pred'][0] == vote_result:
                    correct = 1 if s['score'][0] else 0
                    break
            es_correctness_list.append(correct)
            es_steps_list.append(len(preds))

    avg_samples = sum(es_steps_list) / n
    acc = sum(es_correctness_list) / n


    return es_correctness_list, es_steps_list, acc, avg_samples


def ESC_check(preds, window_size=5):

    if len(preds) < window_size:
        return False
    
    steps = window_size - 1
    window_size_adjusted = min(window_size, len(preds))

    for j in range(len(preds) - window_size_adjusted + 1):
        window = preds[j:j + window_size_adjusted]
        steps += 1
        if all(x == window[0] for x in window):  # 提前停止
            return True

    return False
