# -*- coding: utf-8 -*-
import re
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import jieba.analyse
from textrank4zh import TextRank4Sentence
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import datasets
import lawrouge
import numpy as np
from transformers import TrainerCallback, BertTokenizer, BertModel
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
from sklearn.metrics.pairwise import cosine_similarity
def get_tfidf_keywords(text):
    # jieba
    #keywords_with_weights = jieba.analyse.textrank(text, withWeight=True)
    keywords_with_weights = jieba.analyse.extract_tags(text, withWeight=True)
    # 获取排名前五的关键词
    top_keywords = sorted(keywords_with_weights, key=lambda x: x[1], reverse=True)[:5]
    # 计算 TF-IDF
    # tfidf_keywords = get_tfidf_keywords(text)
    # #输出排名前 k 的句子及其 TF-IDF 值
    # for sentence in sentences:
    #     print(f"句子：{sentence.sentence}，TextRank权重：{sentence.weight}")
    #     for keyword, tfidf_score in tfidf_keywords:
    #         if keyword in sentence.sentence:
    #             print(f"关键词：{keyword}，TF-IDF值：{tfidf_score}")
    return top_keywords
def get_sentence_embedding(sentence):
    model_name = "bert-base-chinese"  # 中文BERT模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    # 分词并添加特殊标记
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    # 转换为PyTorch张量
    input_ids = torch.tensor([input_ids])
    # 获取BERT模型的输出
    with torch.no_grad():
        outputs = model(input_ids)
    # 提取句子向量（CLS标记对应的向量）
    sentence_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    # 降低维度，使其成为2D数组
    sentence_embedding = np.squeeze(sentence_embedding, axis=0)
    return sentence_embedding

def filter_similar_sentences(sentences, threshold=0.8):
    filtered_sentences = []
    selected_embeddings = []

    for sentence in sentences:
        embedding = get_sentence_embedding(sentence.sentence)  # 假设有获取句子embedding的函数
        if not any(cosine_similarity([embedding], [selected_embedding])[0][0] > threshold for selected_embedding in selected_embeddings):
            filtered_sentences.append(sentence)
            selected_embeddings.append(embedding)

    return filtered_sentences
# def top_k_sentences(text, k=5):
#     # 构建TextRank模型
#     tr4s = TextRank4Sentence()
#     # 添加文本
#     tr4s.analyze(text=text, lower=True, source='no_stop_words')
#     # 为每个句子计算权重
#     sentences = tr4s.get_key_sentences(num=k, sentence_min_len=6)
#     #输出排名前 k 的句子
#     for sentence in sentences:
#         print(f"句子：{sentence.sentence}，权重：{sentence.weight}")

def top_k_sentences(text, k=5, similarity_threshold=0.9):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='no_stop_words')
    sentences = tr4s.get_key_sentences(num=k, sentence_min_len=6)
    filtered_sentences = filter_similar_sentences(sentences, threshold=similarity_threshold)

    for sentence in filtered_sentences:
        print(f"句子：{sentence.sentence}，权重：{sentence.weight}")

# 测试文本
text = "错漏百出的教科书。授课老师挑出《计算机应用基础》百多处错误、不妥之处反被处分引出教科书名利之争一本高职学生使用的计算机教材,仅前三章已被授课老师挑出了68处错误。他向学校反映后,反而遭到了处分。事情发生在广东外语艺术职业学院,事情匪夷所思的背后,其实隐藏着复杂的利益纠葛。而这本出现多处“硬伤”的教科书,至今仍在该校学生手中继续使用。学校方面表示,已经组织教学委员会及校外专家对教材展开鉴定,若发现重大知识性错误,将立即停用。错得太离谱:授课老师边上课边纠错广东外语艺术职业学院副教授叶克江的桌面上,摆着一本《计算机应用基础》教材,里面密密麻麻地布满红色笔圈出的修订。“我前后看了不下3次,错漏触目惊心”。他说,自己一边一字不漏地阅读,一边对比实际操作的软件,并对读其他出版社同类教材。仅前三章,他声称已挑出68个错误和87个不妥之处。"  # 选择排名前 2 的句子
top_k_sentences(text, k=5)

