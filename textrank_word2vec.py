# -*- coding: utf-8 -*-
import jieba
import numpy as np
import networkx as nx
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from lawrouge import Rouge
from datasets import load_dataset
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

word2vec_model = Word2Vec.load('word2vec_model.model')

def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }
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
# 中文分词
def segment_sentences(text):
    return [list(jieba.cut(sent)) for sent in text.split('。') if sent]
# 使用Word2Vec构建句子向量
def build_sentence_vectors_with_word2vec(sentences, word2vec_model):
    sentence_vectors = []
    for sent in sentences:
        if len(sent) != 0:
            v = sum([word2vec_model.wv[word] for word in sent if word in word2vec_model.wv]) / len(sent)
        else:
            v = np.zeros((word2vec_model.vector_size,))
        sentence_vectors.append(v)
    return np.array(sentence_vectors)

# TextRank算法
def textrank(sentence_vectors, max_iter=500, tol=1e-6):
    sim_mat = cosine_similarity(sentence_vectors)
    nx_graph = nx.from_numpy_array(sim_mat)
    try:
        scores = nx.pagerank(nx_graph, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        print("PageRank failed to converge. Returning default scores.")
        scores = {i: 1.0 / len(sentence_vectors) for i in range(len(sentence_vectors))}
    return scores

# 主函数：提取摘要
def summarize(text, word2vec_model, top_n=3):
    sentences = segment_sentences(text)
    sentence_vectors = build_sentence_vectors_with_word2vec(sentences, word2vec_model)
    scores = textrank(sentence_vectors)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return '。'.join([''.join(sent) for _, sent in ranked_sentences[:top_n]])


# 示例文本
dataset = load_dataset('json', data_files='nlpcc_data/nlpcc2017_clean.json', field='data')
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
dataset = dataset.select(range(10001, 12001))
test_examples = [sample["document"] for sample in dataset]
reference_texts = [sample["summary"] for sample in dataset]
# 初始化分词器和模型
document=[]
vectorizer = TfidfVectorizer()
# 生成摘要
for t in test_examples:
    summary = summarize(t, word2vec_model)
    print(summary)
    document += [summary]
# 计算评估指标
evaluation_metrics = compute_metrics_batch(reference_texts, document)

print("ROUGE-1 F1 Score: {:.2f}".format(evaluation_metrics["rouge-1"]))
print("ROUGE-2 F1 Score: {:.2f}".format(evaluation_metrics["rouge-2"]))
print("ROUGE-L F1 Score: {:.2f}".format(evaluation_metrics["rouge-l"]))

