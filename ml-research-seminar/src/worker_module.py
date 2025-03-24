# worker_module.py
import time
import traceback
from tenacity import retry, wait_chain, wait_fixed
import ollama

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def completion_with_backoff(**kwargs):
    return ollama.chat(**kwargs)

def run_completion(prompt_q):
    response = completion_with_backoff(
        model="qwen2-math:7b",
        messages=[
            {"role": "system", "content": "Follow the given examples and answer the question."},
            {"role": "user", "content": prompt_q},
        ]
    )
    return response

def worker(result_queue, prompt_q):
    try:
        response = run_completion(prompt_q)
        result_queue.put((True, response))
    except Exception:
        result_queue.put((False, traceback.format_exc()))