# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertForMaskedLM
from lawrouge import Rouge
from datasets import load_dataset
from transformers import (AutoTokenizer,BartForConditionalGeneration)
import jieba.analyse
from textrank4zh import TextRank4Keyword

def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }
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
def Single(text):
    # 加载训练好的模型
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    ### 加载模型
    model = BartForConditionalGeneration.from_pretrained("results/best_mask_textrank_10000")
    model.to('cuda:0')
    ###
    key = textrank_extract_keywords(text)
    # key = [item[0] for item in key][:5]
    text = text + '文章的摘要为[MASK]。' + ''.join(key) + '。'
    # t = t+ '。'+''.join(key) + '。'
    # print("textrank",key)
    inputs = tokenizer(
        text,
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
    print('原摘要:', output_str[0])
    return output_str
