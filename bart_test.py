# -*- coding: utf-8 -*-
import time

import torch
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from transformers import BartForConditionalGeneration, BartConfig, BertForMaskedLM
import datasets
import lawrouge
import numpy as np
from lawrouge import Rouge
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
import jieba.analyse
from textrank4zh import TextRank4Sentence, TextRank4Keyword
from sklearn.metrics.pairwise import cosine_similarity
from pgn import find_most_similar_words


def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }
def tf_extract_keywords(text):
    # 分词
    def chinese_word_cut(text):
        return " ".join(jieba.cut(text))
    # 使用jieba分词进行中文分词
    text_cut = chinese_word_cut(text)
    # 创建TF-IDF模型
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_cut])
    # 获取特征词汇表
    feature_names = vectorizer.get_feature_names_out()
    # 获取TF-IDF矩阵
    tfidf_values = tfidf_matrix.toarray()
    # 获取关键词及其TF-IDF值
    keywords = [(feature_names[j], tfidf_values[0][j]) for j in tfidf_values[0].argsort()[::-1] if tfidf_values[0][j] > 0]
    return keywords

def textrank_extract_keywords(text, num_keywords=5):
    # 创建 TextRank4Keyword 对象
    tr4w = TextRank4Keyword()
    # 添加文本
    tr4w.analyze(text=text, lower=True, window=2)
    # 获取关键词
    keywords = tr4w.get_keywords(num=num_keywords)
    # 提取关键词的词语
    key_words = [item.word for item in keywords]
    return key_words
def compute_metrics_batch(reference_texts_list, generated_texts_list):
    rouge = Rouge()
    # print(reference_texts_list)
    # print(generated_texts_list)
    # 存储所有ROUGE分数
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    # 对每一组文本-摘要对计算ROUGE分数
    for reference_texts, generated_texts in zip(reference_texts_list, generated_texts_list):
        rouge_scores = rouge.get_scores(generated_texts, reference_texts, avg=True)

        rouge_1_scores.append(rouge_scores['rouge-1']['f'] * 100)
        rouge_2_scores.append(rouge_scores['rouge-2']['f'] * 100)
        rouge_l_scores.append(rouge_scores['rouge-l']['f'] * 100)

    return {
        "rouge-1": sum(rouge_1_scores) / len(rouge_1_scores),
        "rouge-2": sum(rouge_2_scores) / len(rouge_2_scores),
        "rouge-l": sum(rouge_l_scores) / len(rouge_l_scores),
    }
# 加载训练好的模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

### 加载模型
model = BartForConditionalGeneration.from_pretrained("results/best_mask_textrank_10000")
model.to('cuda:0')
###
model2 = SentenceTransformer('bert-base-chinese')
model3 = BertForMaskedLM.from_pretrained('bert-base-chinese')
dataset = load_dataset('json', data_files='nlpcc_data/nlpcc2017_clean.json', field='data')
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
dataset = dataset.select(range(10001, 12001))
test_examples = [sample["document"] for sample in dataset]
reference_texts = [sample["summary"] for sample in dataset]
document=[]

num=1

for t in test_examples:
    print(num)
    num+=1
    if len(t)>1000:
        key = textrank_extract_keywords(t)
    else:
        key = tf_extract_keywords(t)
        key = [item[0] for item in key][:5]
    t = t + '文章的摘要为[MASK]。' + ''.join(key) + '。'
    # t = t+ '。'+''.join(key) + '。'
    #print("textrank",key)
    inputs = tokenizer(
        t,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    # 生成
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
    # 将token转换为文字
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output_str = [s.replace(" ", "") for s in output_str]
    print('原摘要:', output_str)

    document += output_str

# print(reference_texts)
# 计算评估指标
evaluation_metrics = compute_metrics_batch(reference_texts, document)

print("ROUGE-1 F1 Score: {:.2f}".format(evaluation_metrics["rouge-1"]))
print("ROUGE-2 F1 Score: {:.2f}".format(evaluation_metrics["rouge-2"]))
print("ROUGE-L F1 Score: {:.2f}".format(evaluation_metrics["rouge-l"]))
