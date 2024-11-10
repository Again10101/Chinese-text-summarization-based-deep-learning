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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def build_sentence_vectors(sentences, model, tokenizer, embedding_size, document_length, tfidf_matrix,tfidf_vocab):
    word_to_index = {word: index for index, word in enumerate(tfidf_vocab)}
    sentence_vectors = []
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

        # 位置特征调整为开头和结尾句子权重更高
        position_feature = calculate_position_weight(i, document_length)
        position_feature *=0.05
        # 将位置特征和TF-IDF特征附加到句子向量中
        combined_vector =sentence_vector+torch.tensor([position_feature])

        sentence_vectors.append(combined_vector)

    return torch.stack(sentence_vectors).numpy()

# TextRank算法
def textrank(sentence_vectors):
    sim_mat = cosine_similarity(sentence_vectors)
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    return scores

# 主函数：提取摘要
def summarize(text, model, tokenizer, top_n=3, embedding_size=512):
    sentences = segment_sentences(text)
    document_length = len(sentences)
    tfidf_scores, tfidf_vocab = calculate_tfidf_scores(sentences)
    sentence_vectors = build_sentence_vectors(sentences, model, tokenizer, embedding_size, document_length, tfidf_scores, tfidf_vocab)
    scores = textrank(sentence_vectors)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return '。'.join([''.join(sent) for _, sent in ranked_sentences[:top_n]])


# 示例文本
dataset = load_dataset('json', data_files='nlpcc_data/nlpcc2017_clean.json', field='data')
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
dataset = dataset.select(range(10001, 10011))
test_examples = [sample["document"] for sample in dataset]
reference_texts = [sample["summary"] for sample in dataset]
# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
model.to(device)
document=[]
num=1
# 生成摘要
for t in test_examples:
    print(num)
    num+=1
    summary = summarize(t, model, tokenizer)
    print(summary)
    document+=[summary]
# 计算评估指标
evaluation_metrics = compute_metrics_batch(reference_texts, document)

print("ROUGE-1 F1 Score: {:.2f}".format(evaluation_metrics["rouge-1"]))
print("ROUGE-2 F1 Score: {:.2f}".format(evaluation_metrics["rouge-2"]))
print("ROUGE-L F1 Score: {:.2f}".format(evaluation_metrics["rouge-l"]))

