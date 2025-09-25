import torch
import torch.nn as nn
import numpy as np
import re

# 检查是否有可用的GPU或MPS
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# 正弦位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# 优化的Transformer模型
class OptimizedTransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(OptimizedTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 使用正弦位置编码而不是学习位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 增加前馈网络维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512,  # 增加前馈网络维度
            dropout=dropout,
            activation='gelu',  # 使用GELU激活函数
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 更复杂的解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src):
        # src shape: [batch_size, seq_length]
        batch_size, seq_len = src.shape
        
        # 嵌入输入
        src = self.embedding(src.unsqueeze(-1)) * np.sqrt(self.d_model)  # [batch_size, seq_length, d_model]
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # 通过Transformer编码器
        output = self.transformer_encoder(src)
        
        # 使用最后一个时间步的输出进行预测
        output = self.decoder(output[:, -1, :])  # [batch_size, 1]
        
        return output.squeeze(-1)  # [batch_size]

# 数据归一化
def normalize_sequence(sequence):
    """对单个序列进行归一化处理"""
    seq_array = np.array(sequence, dtype=np.float32)
    seq_mean = np.mean(seq_array)
    seq_std = np.std(seq_array)
    if seq_std == 0:
        seq_std = 1.0  # 避免除以0
    
    # 归一化序列
    norm_sequence = (seq_array - seq_mean) / seq_std
    
    return norm_sequence, seq_mean, seq_std

# 反归一化
def denormalize_data(norm_data, mean, std):
    """将归一化的数据还原"""
    return norm_data * std + mean

# 加载模型
def load_model(model_path, input_dim=1, d_model=128, nhead=4, num_layers=3, dropout=0.2):
    """加载预训练模型"""
    model = OptimizedTransformerModel(input_dim, d_model, nhead, num_layers, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 解析用户输入的数列
def parse_sequence(input_str):
    """解析用户输入的数列字符串"""
    # 尝试匹配逗号分隔的数字
    numbers = re.findall(r'[-+]?\d*\.?\d+', input_str)
    if not numbers:
        return None
    
    try:
        sequence = [float(num) for num in numbers]
        return sequence
    except ValueError:
        return None

# 预测数列的下一个数字
def predict_next_number(model, sequence):
    """使用模型预测数列的下一个数字"""
    # 确保序列长度为9
    if len(sequence) > 9:
        sequence = sequence[-9:]  # 取最后9个数字
    elif len(sequence) < 9:
        # 如果序列长度不足9，用第一个数字填充
        sequence = [sequence[0]] * (9 - len(sequence)) + sequence
    
    # 归一化序列
    norm_sequence, seq_mean, seq_std = normalize_sequence(sequence)
    
    # 转换为张量并添加批次维度
    input_tensor = torch.tensor(norm_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 使用模型进行预测
    with torch.no_grad():
        norm_prediction = model(input_tensor).cpu().numpy()[0]
    
    # 反归一化预测结果
    prediction = denormalize_data(norm_prediction, seq_mean, seq_std)
    
    return prediction

# 主交互程序
def main():
    print("=== 数列预测交互程序 ===")
    print("请输入一个数列，程序将预测下一个数字。")
    print("输入格式示例：1, 3, 5, 7, 9, 11, 13, 15, 17")
    print("输入 'quit' 或 'exit' 退出程序。")
    print()
    
    # 加载模型
    model_path = 'optimized_transformer_sequence_predictor.pth'
    try:
        model = load_model(model_path)
        print(f"模型 '{model_path}' 加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 '{model_path}'")
        print("请确保已运行 'simple_sequence_predictor.py' 生成模型文件。")
        return
    
    while True:
        # 获取用户输入
        user_input = input("请输入数列（用逗号分隔）：").strip()
        
        # 检查退出命令
        if user_input.lower() in ['quit', 'exit']:
            print("感谢使用，再见！")
            break
        
        # 解析数列
        sequence = parse_sequence(user_input)
        if sequence is None:
            print("错误：无法解析输入。请确保输入的是用逗号分隔的数字。")
            print()
            continue
        
        if len(sequence) < 2:
            print("错误：数列长度至少为2。")
            print()
            continue
        
        print(f"输入的数列：{sequence}")
        
        # 预测下一个数字
        try:
            prediction = predict_next_number(model, sequence)
            print(f"预测的下一个数字：{prediction:.2f}")
        except Exception as e:
            print(f"预测过程中发生错误：{e}")
        
        print()

if __name__ == "__main__":
    main()