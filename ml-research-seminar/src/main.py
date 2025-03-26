# main.py
import random
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
import re
import ollama
from worker_module import worker
import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "prompt_hardest.txt")
out_path = os.path.join(current_dir, "train_qwen2-math.txt")

gsm8k = load_dataset('gsm8k', 'main')
prompt_complex = open(file_path, 'r', encoding='utf-8').read()

def extract_ans(ans_model):
    lines = ans_model.split('\n')
    ans = []
    for li, line in enumerate(lines):
        ans.append(line)
        if 'answer is' in line:
            break
    residual = lines[li+1:] if li+1 < len(lines) else []
    return '\n'.join(ans), '\n'.join(residual)

def main():
    total_tests = len(gsm8k["train"]['question'])
    selected_indices = random.sample(range(total_tests), 200)

    with open(out_path, 'w', encoding='utf-8') as fd, tqdm(total=200) as pbar:
        for idx in selected_indices:
            q = gsm8k["train"]['question'][idx]
            a = gsm8k["train"]['answer'][idx]
            prompt_q = prompt_complex + '\nQuestion: ' + q + '\n'

            result_queue = mp.Queue()
            proc = mp.Process(target=worker, args=(result_queue, prompt_q))
            proc.start()

            proc.join(timeout=120)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                print(f"超时：问题 '{q[:30]}...' 跳过")
                fd.write('Q: %s\nResult: TIMEOUT\nA:\n%s\n\n' % (q, a))
            else:
                try:
                    success, result = result_queue.get(timeout=2)
                except Exception:
                    success, result = False, "No result returned"
                if success:
                    response = result
                    ans_model = response.message.content
                    ans, residual = extract_ans(ans_model)
                    fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans, a))
                else:
                    print(f"问题 '{q[:30]}...' 出现错误")
                    fd.write('Q: %s\nResult: ERROR: %s\nA:\n%s\n\n' % (q, result, a))

            pbar.update(1)

if __name__ == "__main__":
    mp.freeze_support()
    main()
