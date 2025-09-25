import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU或MPS
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# 定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# 生成简单等差数列数据
def generate_arithmetic_sequences(num_sequences=1000, seq_length=10):
    """只生成等差数列数据，简化任务"""
    sequences = []
    targets = []
    
    for _ in range(num_sequences):
        # 等差数列: a_n = a_1 + (n-1)*d
        start = np.random.randint(1, 10)
        diff = np.random.randint(1, 5)
        seq = [start + i * diff for i in range(seq_length)]
        
        # 输入是前seq_length-1个数，目标是最后一个数
        sequences.append(seq[:-1])
        targets.append(seq[-1])
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

# 数据归一化
def normalize_data(sequences, targets):
    """对数据进行归一化处理"""
    # 计算每个序列的均值和标准差
    seq_mean = np.mean(sequences, axis=1, keepdims=True)
    seq_std = np.std(sequences, axis=1, keepdims=True)
    seq_std[seq_std == 0] = 1.0  # 避免除以0
    
    # 归一化序列
    norm_sequences = (sequences - seq_mean) / seq_std
    
    # 归一化目标值
    norm_targets = (targets - seq_mean.squeeze()) / seq_std.squeeze()
    
    return norm_sequences, norm_targets, seq_mean, seq_std

# 反归一化
def denormalize_data(norm_data, mean, std):
    """将归一化的数据还原"""
    return norm_data * std + mean

# 定义优化的Transformer模型
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

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            
            # 计算准确率（允许一定的误差范围）
            predicted = outputs.detach().cpu().numpy()
            actual = targets.detach().cpu().numpy()
            tolerance = 0.02 * np.abs(actual) + 0.02  # 更严格的误差范围
            correct += np.sum(np.abs(predicted - actual) < tolerance)
            total += len(actual)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                predicted = outputs.detach().cpu().numpy()
                actual = targets.detach().cpu().numpy()
                tolerance = 0.02 * np.abs(actual) + 0.02
                correct += np.sum(np.abs(predicted - actual) < tolerance)
                total += len(actual)
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# 测试函数
def test_model(model, test_loader, criterion, test_means=None, test_stds=None):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, (sequences, targets) in enumerate(test_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            
            predicted = outputs.detach().cpu().numpy()
            actual = targets.detach().cpu().numpy()
            
            # 如果有归一化参数，则反归一化
            if test_means is not None and test_stds is not None:
                batch_size = sequences.shape[0]
                batch_means = test_means[i*batch_size:(i+1)*batch_size].squeeze()
                batch_stds = test_stds[i*batch_size:(i+1)*batch_size].squeeze()
                predicted = denormalize_data(predicted, batch_means, batch_stds)
                actual = denormalize_data(actual, batch_means, batch_stds)
            
            all_predictions.extend(predicted)
            all_targets.extend(actual)
            
            # 使用原始值计算准确率
            tolerance = 0.02 * np.abs(actual) + 0.02  # 更严格的误差范围
            correct += np.sum(np.abs(predicted - actual) < tolerance)
            total += len(actual)
    
    test_loss = test_loss / len(test_loader)
    test_acc = correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    return test_loss, test_acc, all_predictions, all_targets

# 可视化结果
def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, predictions, targets):
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制预测值与真实值对比
    plt.subplot(1, 3, 3)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.title('Predictions vs Targets')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig('simple_results.png')
    plt.show()

# 设置超参数
seq_length = 9  # 输入序列长度
input_dim = 1   # 输入维度
d_model = 128   # 增加模型维度
nhead = 4       # 增加多头注意力头数
num_layers = 3  # 增加Transformer层数
batch_size = 32
learning_rate = 0.001
num_epochs = 200
dropout = 0.2

# 生成数据
sequences, targets = generate_arithmetic_sequences(num_sequences=5000, seq_length=seq_length+1)

# 数据归一化
norm_sequences, norm_targets, seq_means, seq_stds = normalize_data(sequences, targets)

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp, means_train, means_temp, stds_train, stds_temp = train_test_split(
    norm_sequences, norm_targets, seq_means, seq_stds, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, means_val, means_test, stds_val, stds_test = train_test_split(
    X_temp, y_temp, means_temp, stds_temp, test_size=0.5, random_state=42)

# 创建数据加载器
train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)
test_dataset = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = OptimizedTransformerModel(input_dim, d_model, nhead, num_layers, dropout=dropout).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 训练模型
print("Starting training...")
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=15
)

# 测试模型
print("\nTesting model...")
test_loss, test_acc, predictions, targets = test_model(model, test_loader, criterion, means_test, stds_test)

# 可视化结果
print("\nVisualizing results...")
plot_results(train_losses, val_losses, train_accuracies, val_accuracies, predictions, targets)

# 保存模型
torch.save(model.state_dict(), 'optimized_transformer_sequence_predictor.pth')
print("\nModel saved as 'optimized_transformer_sequence_predictor.pth'")