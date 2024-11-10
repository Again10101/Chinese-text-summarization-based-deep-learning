毕业设计开源代码
分别实现了抽取式中文文本摘要和生成式中文文本摘要

使用的数据集为nlpcc2017的clean版本，可自行网上下载，文件太大，不好传输，是开源的
可能涉及到的预训练模型为
bart-base-chinese
bart-large-chinese
bert-base-chinese
代码中尝试了多种方法，很多均为当时为了寻找最佳模型而进行的尝试，最终效果最好的为bart_test.py
分别设计了抽取式摘要生成和预训练模型生成式摘要生成
抽取式摘要更改了相似度计算公式为余弦相似度，并且引入主题词权重和关键词权重(bert_textrank_ida.py)
声称是摘要使用bert用作编码器，bart模型进行摘要任务的训练，并在训练语句中结合了提示词和textrank算法提取的关键词辅助提升模型整体摘要效果(bart_test.py)
使用vue和flask实现了前后端，可以发布在线网页

Graduation project open source code
Realized extractive Chinese text summarization and generative Chinese text summarization

The dataset used is the clean version of NLPCC 2017, which can be downloaded online as it is open-source. The file is too large for easy transfer.

The pre-trained models potentially involved are:

bart-base-chinese
bart-large-chinese
bert-base-chinese
The code attempts various methods, with many being experiments conducted at the time to find the best model. The method that achieved the best results is bart_test.py.

Both extractive and generative summarization methods were designed. The extractive summarization modifies the similarity calculation formula to cosine similarity and introduces topic word weights and keyword weights (bert_textrank_ida.py).

The generative summarization uses BERT as an encoder, and the BART model for the summarization task. During training, it incorporates prompt words and keywords extracted by the TextRank algorithm to enhance the model’s overall summarization effectiveness (bart_test.py).

Vue and Flask were used to implement the frontend and backend, enabling the deployment of an online webpage.
