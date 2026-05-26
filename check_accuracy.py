import re
from datasets import load_dataset
from model import call_slm, call_llm

def extract_number(text):
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers[-1] if numbers else ''

gsm8k = load_dataset('gsm8k', 'main', split='train')

slm_correct = 0
llm_correct = 0
n = 20

for ex in gsm8k.select(range(n)):
    req    = ex['question']
    answer = ex['answer'].split('####')[-1].strip()

    slm_pred = extract_number(call_slm(req))
    llm_pred = extract_number(call_llm(req))

    slm_ok = slm_pred == answer
    llm_ok = llm_pred == answer

    slm_correct += int(slm_ok)
    llm_correct += int(llm_ok)

    slm_str = "OK" if slm_ok else "NO"
    llm_str = "OK" if llm_ok else "NO"
    print(f"SLM={slm_str} LLM={llm_str} | ans={answer} | {req[:40]}")

print(f"\nSLM accuracy: {slm_correct}/{n} = {slm_correct/n*100:.0f}%")
print(f"LLM accuracy: {llm_correct}/{n} = {llm_correct/n*100:.0f}%")
