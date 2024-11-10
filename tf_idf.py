# -*- coding: utf-8 -*-
import jieba
from gensim import corpora, models
from lawrouge import Rouge
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


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
def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }
# 中文分词，将文档分割为句子并分词
def segment_sentences(text):
    sentences = [sent for sent in text.split('。') if sent]
    segmented_sentences = [' '.join(jieba.cut(sent)) for sent in sentences]
    return sentences, segmented_sentences


def tfidf_summarize(text, top_n=3):
    sentences, segmented_sentences = segment_sentences(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(segmented_sentences)
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    ranked_sentences = sorted(((score, index) for index, score in enumerate(sentence_scores)), reverse=True)
    selected_sentences = [sentences[index] for _, index in ranked_sentences[:top_n]]
    return '。'.join(selected_sentences)


# 示例文本
dataset = load_dataset('json', data_files='nlpcc_data/nlpcc2017_clean.json', field='data')
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])
dataset = dataset.select(range(10001, 12001))
test_examples = [sample["document"] for sample in dataset]
reference_texts = [sample["summary"] for sample in dataset]

# 生成摘要
document = []
nn=1
for t in test_examples:
    summary = tfidf_summarize(t)
    print(nn)
    nn+=1
    document.append(summary)
# 计算评估指标
evaluation_metrics = compute_metrics_batch(reference_texts, document)

print("ROUGE-1 F1 Score: {:.2f}".format(evaluation_metrics["rouge-1"]))
print("ROUGE-2 F1 Score: {:.2f}".format(evaluation_metrics["rouge-2"]))
print("ROUGE-L F1 Score: {:.2f}".format(evaluation_metrics["rouge-l"]))