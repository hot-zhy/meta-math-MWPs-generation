import json
import re

answers = {}


def simplify_answer(answer):
    pattern = r"The answer is(.*)"
    has_match = True
    match = re.search(pattern, answer, re.IGNORECASE)
    if match:
        answer = match.group(1)
        answer = answer.strip().lstrip(':').strip()
    else:
        answer = "no answer"

    match = re.search(pattern, answer, re.IGNORECASE)
    if match:
        answer = match.group(1)
        answer = answer.strip().lstrip(':').strip()

    pattern_frac = r"\\frac{(\d+)}{(\d+)}"
    answer = re.sub(pattern_frac, r"\1/\2", answer)

    return answer


with open('../data/answers.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        que_id = data['queId']
        answer = data['answer']
        answer = simplify_answer(answer)
        answers[que_id] = answer

with open('../data/output.json', 'w', encoding='utf-8') as file:
    json.dump(answers, file, indent=4)
