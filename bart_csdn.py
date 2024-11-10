import argparse
import json

import torch
import transformers


# 定义BARTFineTuner类
class BARTFineTuner(torch.nn.Module):

    def __init__(self, model_path):
        super().__init__()

        # 加载本地的BART模型
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = transformers.BartForConditionalGeneration.from_pretrained(model_path)

    def forward(self, inputs):
        # 获取encoder输入及attention mask（遮蔽器）
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # 使用模型生成摘要
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids[:, :-1].contiguous(),
            labels=input_ids[:, 1:].contiguous()
        )
        return outputs.loss, outputs.logits


# 定义加载数据集的类
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, max_source_length, max_target_length):
        # 从文件中读取数据
        with open(data_path, "r", encoding="utf-8") as file:
            self.data = json.loads(file.read())[:100]

        # 初始化tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-chinese')

        # 限制输入和输出长度
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __getitem__(self, idx):
        # 划分输入和输出文本，并将其编码以生成输入IDs和输出IDs
        input_text, output_text = self.data[idx]["content"].strip(), self.data[idx]["title"].strip()

        # 使用 tokenizer 的 encode_plus 方法
        encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # 处理输出文本
        output_ids = self.tokenizer.encode(
            output_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 返回字典类型的数据
        return {'input_ids': input_ids.squeeze(0), 'attention_mask': attention_mask.squeeze(0),
                'target_ids': output_ids.squeeze(0)}

    def __len__(self):
        return len(self.data)


# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        # 将数据移到指定计算设备上
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        # 清零梯度
        optimizer.zero_grad()
        # 向前传递批量数据以获得损失
        loss, _ = model(batch_data)
        # 向后传递损失并更新权重
        loss.backward()
        optimizer.step()
        # 每隔一定间隔打印训练状态信息
        if batch_idx % 10 == 0:
            print('Epoch {}\tBatch [{}/{}]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(batch_data['input_ids']), len(train_loader.dataset),
                loss.item()))
            # 在训练时获取一个样本进行推理，并将生成的文本摘要打印出来
            with torch.no_grad():
                _, _ = model(batch_data)  # 注意：这里只是为了获取生成的文本摘要，不使用损失值


if __name__ == '__main__':
    # 设置GPU或CPU作为计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='nlpcc_data/nlpcc_data.json', help='path to the training data')
    parser.add_argument('--model_path', type=str, default='bart-base-chinese',  # 指定本地模型路径
                        help='path to the pre-trained model')
    parser.add_argument('--batch_size', type=int, default=2, help='size of each batch for training')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs for training')
    parser.add_argument('--max_source_length', type=int, default=1024, help='maximum length of input sequences')
    parser.add_argument('--max_target_length', type=int, default=128, help='maximum length of output sequences')
    args = parser.parse_args()

    # 加载模型并移动到指定设备上
    model = BARTFineTuner(args.model_path).to(device)

    # 加载数据集并使用DataLoader迭代数据
    train_dataset = MyDataset(args.data, args.max_source_length, args.max_target_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 定义AdamW优化器并开始训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for epoch in range(1, args.num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    torch.save(model.state_dict(), "BART.pth")