import re

def test_answer(pred_str, ans_str):
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if len(pred) >= 1:
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        gold = gold[-1]
        return pred == gold
    else:
        return False

def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    q = ""
    
    for l in lines:
        if l.startswith('Q: '):
            if am is not None and a is not None:
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if test_answer(am, a):
                    acc += 1
            current_mode = 'q'
            q = l
            am, a = None, None
            num_q += 1
        elif l.startswith('A_model:'):
            current_mode = 'am'
            am = l
        elif l.startswith('A:'):
            current_mode = 'a'
            a = l
        else:
            if current_mode == 'q':
                q += l
            elif current_mode == 'am':
                am += l
            elif current_mode == 'a':
                a += l
            else:
                raise ValueError(f"未知的模式：{current_mode}")
    
    if am is not None and a is not None:
        questions.append(q)
        ans_pred.append(am)
        ans_gold.append(a)
        if test_answer(am, a):
            acc += 1
    
    if num_q > 0:
        accuracy = float(acc) / num_q
        print(f'总问题数: {num_q}, 正确数: {acc}, 准确率: {accuracy:.4f}')
    else:
        print('未找到任何问题。')
    
    return questions, ans_pred, ans_gold

# 使用示例
filename = 'train_qwen2math.txt'
questions, ans_pred, ans_gold = parse_pred_ans(filename)
