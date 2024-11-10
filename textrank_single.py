# -*- coding: utf-8 -*-
import jieba
import numpy as np
import networkx as nx
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from gensim import corpora, models
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_lda_scores(sentences, num_topics=1):
    dictionary = corpora.Dictionary(sentences)
    corpus = [dictionary.doc2bow(text) for text in sentences]

    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda.get_document_topics(corpus)

    lda_scores = []
    for i, topic_list in enumerate(topics):
        if topic_list:
            lda_scores.append(topic_list[0][1])  # 假设只有一个主题
        else:
            lda_scores.append(0)
    return lda_scores
def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }
# 计算TF-IDF得分
def calculate_tfidf_scores(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in documents])
    return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()

# 中文分词
def segment_sentences(text):
    return [list(jieba.cut(sent)) for sent in text.split('。') if sent]
# 计算位置特征，使开头和结尾的句子权重更高
def calculate_position_weight(i, document_length):
    # 使用正弦函数来增加开头和结尾句子的权重
    if document_length == 1:
        return 0.5
    return 0.5 * (1 + np.sin((i / (document_length - 1) * np.pi) - np.pi / 2))
# 构建句子向量（基于BERT词向量）
def build_sentence_vectors(sentences, model, tokenizer, embedding_size,tfidf_matrix,tfidf_vocab):
    word_to_index = {word: index for index, word in enumerate(tfidf_vocab)}
    sentence_vectors = []
    tfidf_features=[]
    for i, sent in enumerate(sentences):
        # 截断处理
        tokens = tokenizer.encode_plus(''.join(sent), add_special_tokens=True,
                                       max_length=embedding_size, truncation=True,
                                       padding='max_length', return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

        # 获取句子中的词的TF-IDF得分，如果词不在词汇表中则忽略
        tfidf_scores = [tfidf_matrix[i, word_to_index[word]] for word in sent if word in word_to_index]

        # 如果tfidf_scores为空，则用默认值0；否则，计算平均值
        #tfidf_feature = np.mean(tfidf_scores) if tfidf_scores else 0.0
        tfidf_feature = np.nanmean(tfidf_scores) if not np.isnan(np.nanmean(tfidf_scores)) else 0.0

        sentence_vectors.append(sentence_vector)
        tfidf_features.append(tfidf_feature)
    return torch.stack(sentence_vectors).numpy(), np.array(tfidf_features)
# TextRank算法
def textrank(sentence_vectors, lda_scores,tfidf_scores,tfidf_weight,lda_weight):
    num_sentences = len(sentence_vectors)
    sim_mat = np.zeros((num_sentences, num_sentences))
    for l in range(0,len(lda_scores)):
        if l>=3:lda_scores[l]=0.1

    # tfidf_weight = 0.35
    # lda_weight=0.35
    # 计算句子向量之间的相似度
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                # 计算原始的余弦相似度
                cos_sim = cosine_similarity(sentence_vectors[i].reshape(1, -1), sentence_vectors[j].reshape(1, -1))[0, 0]
                #print((lda_scores[i] + lda_scores[j]) / 2)
                # 调整相似度：将LDA得分作为权重因子
                adjusted_sim = cos_sim *(1 - lda_weight - tfidf_weight)+ lda_weight *(lda_scores[i] + lda_scores[j]) / 2 + tfidf_weight*(tfidf_scores[i] + tfidf_scores[j])/2
               # print(adjusted_sim)
                sim_mat[i][j] = adjusted_sim

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_indices = sorted(scores, key=scores.get, reverse=True)

    return ranked_indices

# 主函数：提取摘要

def summarize(text, model, tokenizer,tfidf_weight, lda_weight,top_n=3, embedding_size=512, num_topics=1):
    sentences = segment_sentences(text)
    lda_scores = get_lda_scores(sentences, num_topics=num_topics)
    tfidf_scores, tfidf_vocab = calculate_tfidf_scores(sentences)
    sentence_vectors,tfidf_feature = build_sentence_vectors(sentences, model, tokenizer, embedding_size, tfidf_scores, tfidf_vocab)
    ranked_indices = textrank(sentence_vectors, lda_scores,tfidf_feature,tfidf_weight, lda_weight)
    return '。'.join([''.join(sentences[i]) for i in ranked_indices[:top_n]])

def single(text):

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    model.to(device)
    # 初始化权重组合和结果存储
    summary = summarize(text, model, tokenizer, 0.4, 0.2)

    return summary



