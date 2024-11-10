from transformers import BertTokenizer, BertForMaskedLM
import torch

# 1. 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# 2. 准备含有掩码的文本
ans = "北京是中国的手读"  # '[MASK]' 是需要模型填充的部分
print(ans.find('手读'))
# 3. 使用分词器预处理文本
inputs = tokenizer(ans, return_tensors="pt")

# 4. 输入模型进行预测
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# 5. 找到掩码位置的预测结果
masked_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(axis=1)
predicted_token = tokenizer.decode(predicted_token_id)

# 打印填充后的文本
print(ans.replace(tokenizer.mask_token, predicted_token))
