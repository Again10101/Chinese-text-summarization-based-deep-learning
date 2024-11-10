# -*- coding: utf-8 -*-
import jieba
import numpy as np
import networkx as nx
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from gensim import corpora, models
from lawrouge import Rouge
from datasets import load_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
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



# 示例文本
dataset = load_dataset('json', data_files='nlpcc_data/nlpcc2017_clean.json', field='data')
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
dataset = dataset.select(range(10001, 12001))
test_examples = [sample["document"] for sample in dataset]
reference_texts = [sample["summary"] for sample in dataset]
# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
model.to(device)
document=[]

# 初始化权重组合和结果存储
# tfidf_weights = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # 例如，从0到1，总共5个值
# lda_weights = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# results = []

# for tfidf_weight in tfidf_weights:
#     for lda_weight in lda_weights:
#         if tfidf_weight + lda_weight > 1:  # 确保权重总和不超过1
#             continue
document = []
nn=1
for t in test_examples:
    print(nn)
    nn+=1
    summary = summarize(t, model, tokenizer, 0.4, 0.2)
    print(summary)
    document.append(summary)
evaluation_metrics = compute_metrics_batch(reference_texts, document)
# 计算评估指标


print("ROUGE-1 F1 Score: {:.2f}".format(evaluation_metrics["rouge-1"]))
print("ROUGE-2 F1 Score: {:.2f}".format(evaluation_metrics["rouge-2"]))
print("ROUGE-L F1 Score: {:.2f}".format(evaluation_metrics["rouge-l"]))
# print('tfidf_weight:',tfidf_weight,'lda_weight',lda_weight)
# print("ROUGE-1 F1 Score: {:.2f}".format(evaluation_metrics["rouge-1"]))
# print("ROUGE-2 F1 Score: {:.2f}".format(evaluation_metrics["rouge-2"]))
# print("ROUGE-L F1 Score: {:.2f}".format(evaluation_metrics["rouge-l"]))
# results.append(((tfidf_weight, lda_weight), evaluation_metrics))
# 对结果按照 ROUGE-1 分数从高到低进行排序
# sorted_results = sorted(results, key=lambda x: x[1]['rouge-1'], reverse=True)

# # 打印排序后的结果
# for result in sorted_results:
#     weights, scores = result
#     tfidf_weight, lda_weight = weights
#     rouge_1 = scores['rouge-1']
#     rouge_2 = scores['rouge-2']
#     rouge_l = scores['rouge-l']
#     print(f"TF-IDF Weight: {tfidf_weight}, LDA Weight: {lda_weight}, ROUGE-1: {rouge_1:.2f}, ROUGE-2: {rouge_2:.2f}, ROUGE-L: {rouge_l:.2f}")

#
#
# # 绘制三维图表
# for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')  # 创建3D绘图区域
#
#     x = [result[0][0] for result in results]  # tfidf_weight
#     y = [result[0][1] for result in results]  # lda_weight
#     z = [result[1][metric] for result in results]  # ROUGE分数
#
#     # 绘制3D散点图
#     sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', depthshade=True)
#     # 对ROUGE分数进行排序并获取最高的五个点的索引
#     top_indices = sorted(range(len(z)), key=lambda i: z[i], reverse=True)[:5]
#
#     # 仅为最高的五个点添加ROUGE分数标签
#     for i in top_indices:
#         ax.text(x[i], y[i], z[i], '%.2f' % z[i], color='black', ha='center', va='bottom')
#
#     cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)  # 添加颜色条
#     cbar.set_label(metric + ' score')
#
#     ax.set_xlabel('TF-IDF Weight')
#     ax.set_ylabel('LDA Weight')
#     ax.set_zlabel(metric + ' Score')
#     ax.set_title(metric + ' Score by Weights')
#     # 在显示图表之前保存图表
#     plt.savefig(metric + '_score_by_weights.png', dpi=300, bbox_inches='tight')
#
#     plt.show()