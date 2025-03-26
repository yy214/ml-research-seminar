import os
import re
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from tqdm import tqdm

def main():
    # 内部定义 tokenize 函数
    def tokenize_function(examples, tokenizer, max_length=512):
        texts = [f"Question: {q}\nAnswer: {a}<|endoftext|>" for q, a in zip(examples["question"], examples["answer"])]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # 内部定义提取答案的函数
    def extract_answer(text):
        matches = re.findall(r'\d+\.?\d*', text)
        return float(matches[-1]) if matches else None

    # 模型名称与加载 tokenizer 和模型
    model_name = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.train()

    # 加载全部 GSM8K 数据集（train 和 test）
    train_dataset = load_dataset("gsm8k", "main", split="train")
    test_dataset = load_dataset("gsm8k", "main", split="test")

    # 对全部数据进行 tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True, 
        remove_columns=test_dataset.column_names
    )

    # 确保输出目录存在
    current_dir = os.path.dirname(__file__)
    output_dir = os.path.join(current_dir, "smolLM-finetuned")
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(current_dir, "logs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=True,
    )

    # 创建 Trainer 对象并启动训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    print("Start training...")
    trainer.train()
    
    # 训练结束后保存模型和 tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 开始评估
    eval_out_path = os.path.join(current_dir, "smolLM-finetune.txt")
    text_gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    original_test = load_dataset("gsm8k", "main", split="test")
    
    total_runtime = 0.0
    correct = 0
    total_samples = len(original_test)
    
    for sample in tqdm(original_test, desc="Evaluating"):
        prompt = f"Question: {sample['question']}\nAnswer:"
        start_time = time.time()
        response = text_gen_pipe(
            prompt,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )[0]['generated_text']
        runtime = time.time() - start_time
        total_runtime += runtime
        
        generated_answer = response.split("Answer:")[-1].strip()
        pred = extract_answer(generated_answer)
        true = extract_answer(sample['answer'])
        
        if pred is not None and true is not None and abs(pred - true) < 1e-3:
            correct += 1
                
    accuracy = correct / total_samples
    avg_runtime = total_runtime / total_samples
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average runtime per sample: {avg_runtime:.4f} seconds")
    
    with open(eval_out_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Average runtime per sample: {avg_runtime:.4f} seconds\n")

if __name__ == "__main__":
    main()