import torch
import datasets
import lawrouge
import numpy as np
from transformers import TrainerCallback
from typing import List, Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          BartForConditionalGeneration)

# 加载tokenizer,中文bart使用bert的tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
batch_size = 16
epochs = 5
max_input_length = 512  # 最大输入长度
max_target_length = 128  # 最大输出长度
learning_rate = 1e-04
# 读取数据
dataset = load_dataset('json', data_files='nlpcc_data/nlpcc2017_clean.json', field='data')


def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id":"0"
    }


def preprocess_function(examples):
    """
    document作为输入，summary作为标签
    """
    inputs = [doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def collate_fn(features: Dict):
    batch_input_ids = [torch.LongTensor(feature["input_ids"]) for feature in features]
    batch_attention_mask = [torch.LongTensor(feature["attention_mask"]) for feature in features]
    batch_labels = [torch.LongTensor(feature["labels"]) for feature in features]

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

#评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 将id解码为文字
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 替换标签中的-100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # 去掉解码后的空格
    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]
    # 分词计算rouge
    # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
    # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]
    # 计算rouge
    rouge = lawrouge.Rouge()
    result = rouge.get_scores(decoded_preds, decoded_labels,avg=True)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}
    result = {key: value * 100 for key, value in result.items()}
    return result

# class ROUGECallback(TrainerCallback):
#     def __init__(self, eval_frequency=10):
#         self.eval_frequency = eval_frequency
#
#     def on_step_end(self, args, state, control, model, **kwargs):
#         if state.global_step % self.eval_frequency == 0:
#             eval_results = trainer.evaluate()
#             print(f"Step {state.global_step}: ROUGE evaluation results: {eval_results['eval_rouge']}")

if __name__ == '__main__':
    # 将原始数据中的content和title转换为document和summary
    dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
    dataset = dataset.select(range(1000))
    # print(dataset)
    # 划分数据集
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).values()
    train_dataset, test_dataset = train_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).values()
    datasets = datasets.DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})
    # print(datasets["train"][2])
    # tokenized
    tokenized_datasets = datasets
    tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=True,remove_columns=["document", "summary", "id"])
    # print(tokenized_datasets["train"][2].keys())
    # print(tokenized_datasets["train"][2])
    # 构建DataLoader来验证collate_fn
    dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    # print(batch)
    #加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained("bart-base-chinese")
    # output = model(**batch) # 验证前向传播
    # print(output)
    # 设置训练参数
    args = Seq2SeqTrainingArguments(
        output_dir="results",  # 模型保存路径
        num_train_epochs=epochs,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.001,
        predict_with_generate=True,
        logging_dir="logs",
        logging_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        generation_max_length=max_target_length,  # 生成的最大长度
        generation_num_beams=1,  # beam search
        #gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        metric_for_best_model="rouge-1"
    )
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks = [ROUGECallback(eval_frequency=10)]

    )
    train_result = trainer.train()
    #保存模型
    trainer.save_model("results/best")

