# 导入PyTorch深度学习框架
import torch
# 导入PyTorch神经网络模块，包含各种层和损失函数
import torch.nn as nn
# 导入PyTorch优化器模块，包含各种优化算法
import torch.optim as optim
# 导入NumPy科学计算库，用于数组操作
import numpy as np
# 导入Matplotlib绘图库，用于数据可视化
import matplotlib.pyplot as plt
# 从PyTorch工具中导入数据集类和数据加载器，用于数据管理
from torch.utils.data import Dataset, DataLoader
# 从Scikit-learn中导入训练集测试集分割函数，用于数据集划分
from sklearn.model_selection import train_test_split

# 设置PyTorch的随机种子为42，确保每次运行代码时随机数生成相同，结果可复现
torch.manual_seed(42)
# 设置NumPy的随机种子为42，确保NumPy的随机操作也是可复现的
np.random.seed(42)

# 检查是否有可用的MPS（Apple Silicon GPU）设备
if torch.backends.mps.is_available():
    # 如果有MPS设备，将设备设置为'mps'
    device = torch.device('mps')
# 如果没有MPS设备，检查是否有可用的CUDA（NVIDIA GPU）设备
elif torch.cuda.is_available():
    # 如果有CUDA设备，将设备设置为'cuda'
    device = torch.device('cuda')
else:
    # 如果既没有MPS也没有CUDA设备，将设备设置为'cpu'
    device = torch.device('cpu')
# 打印当前使用的设备信息
print(f"Using device: {device}")

# 定义一个自定义的数据集类，继承自PyTorch的Dataset类
class SequenceDataset(Dataset):
    # 初始化函数，接收序列数据和对应的目标值
    def __init__(self, sequences, targets):
        # 将输入序列保存为类属性
        self.sequences = sequences
        # 将目标值保存为类属性
        self.targets = targets
    
    # 返回数据集的大小，即序列的数量
    def __len__(self):
        return len(self.sequences)
    
    # 根据索引返回一个数据样本，包含序列和对应的目标值
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# 生成简单等差数列数据的函数
def generate_arithmetic_sequences(num_sequences=1000, seq_length=10):
    """只生成等差数列数据，简化任务"""
    # 创建空列表用于存储生成的序列
    sequences = []
    # 创建空列表用于存储生成的目标值
    targets = []
    
    # 循环生成指定数量的序列
    for _ in range(num_sequences):
        # 等差数列公式: a_n = a_1 + (n-1)*d
        # 随机生成等差数列的起始值，范围在1到10之间
        start = np.random.randint(1, 10)
        # 随机生成等差数列的公差，范围在1到5之间
        diff = np.random.randint(1, 5)
        # 根据起始值和公差生成等差数列
        seq = [start + i * diff for i in range(seq_length)]
        
        # 输入是前seq_length-1个数，目标是最后一个数
        # 将序列除最后一个数外的所有元素添加到sequences列表中
        sequences.append(seq[:-1])
        # 将序列的最后一个元素作为目标值添加到targets列表中
        targets.append(seq[-1])
    
    # 将sequences列表转换为NumPy数组，数据类型为float32
    # 将targets列表转换为NumPy数组，数据类型为float32
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

# 数据归一化函数
def normalize_data(sequences, targets):
    """对数据进行归一化处理"""
    # 计算每个序列的均值，axis=1表示沿行的方向计算，keepdims=True保持维度不变
    seq_mean = np.mean(sequences, axis=1, keepdims=True)
    # 计算每个序列的标准差，axis=1表示沿行的方向计算，keepdims=True保持维度不变
    seq_std = np.std(sequences, axis=1, keepdims=True)
    # 将标准差为0的值设置为1.0，避免在归一化时出现除以0的情况
    seq_std[seq_std == 0] = 1.0  # 避免除以0
    
    # 对序列进行归一化处理：(原始值 - 均值) / 标准差
    norm_sequences = (sequences - seq_mean) / seq_std
    
    # 对目标值进行归一化处理，使用对应序列的均值和标准差
    # squeeze()函数用于去除维度为1的维度，使形状匹配
    norm_targets = (targets - seq_mean.squeeze()) / seq_std.squeeze()
    
    # 返回归一化后的序列、归一化后的目标值、序列均值和序列标准差
    return norm_sequences, norm_targets, seq_mean, seq_std

# 反归一化函数
def denormalize_data(norm_data, mean, std):
    """将归一化的数据还原"""
    # 反归一化公式：归一化值 * 标准差 + 均值
    return norm_data * std + mean

# 定义优化的Transformer模型类，继承自PyTorch的nn.Module
class OptimizedTransformerModel(nn.Module):
    # 初始化函数，定义模型的结构和参数
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        # 调用父类的初始化函数
        super(OptimizedTransformerModel, self).__init__()
        # 保存输入维度
        self.input_dim = input_dim
        # 保存模型维度
        self.d_model = d_model
        # 定义输入嵌入层，将输入维度映射到模型维度
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 使用正弦位置编码而不是学习位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 增加前馈网络维度
        # 定义Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 模型的维度
            nhead=nhead,      # 多头注意力机制的头数
            dim_feedforward=512,  # 前馈神经网络的维度，设置为512以增加模型容量
            dropout=dropout,  # dropout比率，用于防止过拟合
            activation='gelu',  # 使用GELU激活函数，比ReLU更平滑
            batch_first=True   # 输入数据的格式为(batch_size, sequence_length, feature_dim)
        )
        # 创建Transformer编码器，由多个编码器层堆叠而成
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 定义更复杂的解码器，用于将Transformer的输出转换为预测值
        self.decoder = nn.Sequential(
            # 第一层线性变换，将d_model维映射到d_model维
            nn.Linear(d_model, d_model),
            # 层归一化，稳定训练过程
            nn.LayerNorm(d_model),
            # GELU激活函数
            nn.GELU(),
            # Dropout层，防止过拟合
            nn.Dropout(dropout),
            # 第二层线性变换，将d_model维映射到d_model//2维
            nn.Linear(d_model, d_model // 2),
            # 层归一化
            nn.LayerNorm(d_model // 2),
            # GELU激活函数
            nn.GELU(),
            # Dropout层
            nn.Dropout(dropout),
            # 输出层，将d_model//2维映射到1维（预测值）
            nn.Linear(d_model // 2, 1)
        )
        
        # 初始化模型权重
        self._init_weights()
    
    # 权重初始化函数
    def _init_weights(self):
        # 遍历模型的所有参数
        for p in self.parameters():
            # 如果参数的维度大于1（即权重矩阵，不是偏置向量）
            if p.dim() > 1:
                # 使用Xavier均匀初始化方法初始化权重
                nn.init.xavier_uniform_(p)
    
    # 前向传播函数，定义数据如何通过模型
    def forward(self, src):
        # src shape: [batch_size, seq_length]
        # 获取输入数据的批次大小和序列长度
        batch_size, seq_len = src.shape
        
        # 嵌入输入
        # unsqueeze(-1)在最后一个维度增加一个维度，从[batch_size, seq_length]变为[batch_size, seq_length, 1]
        # 通过线性层将输入维度从1映射到d_model
        # 乘以sqrt(d_model)是为了缩放嵌入向量的幅度，有助于训练稳定性
        src = self.embedding(src.unsqueeze(-1)) * np.sqrt(self.d_model)  # [batch_size, seq_length, d_model]
        
        # 添加位置编码
        # 将位置编码加到嵌入向量上，使模型能够区分序列中不同位置的信息
        src = self.pos_encoder(src)
        
        # 通过Transformer编码器
        # 将添加了位置编码的输入送入Transformer编码器进行处理
        output = self.transformer_encoder(src)
        
        # 使用最后一个时间步的输出进行预测
        # output[:, -1, :]选择每个序列的最后一个时间步的输出
        # 通过解码器将最后一个时间步的输出转换为预测值
        output = self.decoder(output[:, -1, :])  # [batch_size, 1]
        
        # squeeze(-1)去除最后一个维度，从[batch_size, 1]变为[batch_size]
        return output.squeeze(-1)  # [batch_size]

# 正弦位置编码类
class PositionalEncoding(nn.Module):
    # 初始化函数
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # 调用父类的初始化函数
        super(PositionalEncoding, self).__init__()
        # 定义dropout层，用于随机置零一些位置编码，增加模型的鲁棒性
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建一个全零的位置编码矩阵，大小为[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 创建位置索引，从0到max_len-1，并增加一个维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码的分母项，使用指数和对数变换来避免数值不稳定
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # 对位置编码的偶数索引应用sin函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对位置编码的奇数索引应用cos函数
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加一个批次维度，并转置以适应后续计算
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将位置编码注册为缓冲区，这样它会被视为模型参数，但不会被训练
        self.register_buffer('pe', pe)
    
    # 前向传播函数
    def forward(self, x):
        # 将位置编码加到输入上，只取与输入序列长度相同的位置编码
        # transpose(0, 1)是为了匹配维度
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        # 应用dropout
        return self.dropout(x)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    # 创建列表用于存储训练过程中的损失值
    train_losses = []
    # 创建列表用于存储验证过程中的损失值
    val_losses = []
    # 创建列表用于存储训练过程中的准确率
    train_accuracies = []
    # 创建列表用于存储验证过程中的准确率
    val_accuracies = []
    
    # 初始化最佳验证损失为正无穷大
    best_val_loss = float('inf')
    # 初始化最佳模型状态为None
    best_model_state = None
    # 初始化早停计数器为0
    patience_counter = 0
    
    # 循环进行指定轮次的训练
    for epoch in range(num_epochs):
        # 将模型设置为训练模式，启用dropout和batch normalization等层
        model.train()
        # 初始化当前轮次的训练损失为0
        running_loss = 0.0
        # 初始化正确预测的样本数为0
        correct = 0
        # 初始化总样本数为0
        total = 0
        
        # 遍历训练数据加载器中的每个批次
        for sequences, targets in train_loader:
            # 将序列和目标值移动到指定的设备（CPU或GPU）
            sequences, targets = sequences.to(device), targets.to(device)
            
            # 前向传播：将输入序列送入模型，得到预测输出
            outputs = model(sequences)
            # 计算预测输出与真实目标值之间的损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            # 清空梯度缓存，防止梯度累积
            optimizer.zero_grad()
            # 计算损失相对于模型参数的梯度
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸，将梯度范数限制在最大值1.0以内
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 根据计算出的梯度更新模型参数
            optimizer.step()
            
            # 统计
            # 将当前批次的损失值累加到总损失中
            running_loss += loss.item()
            
            # 计算准确率（允许一定的误差范围）
            # 将预测输出从GPU移到CPU，并转换为NumPy数组
            predicted = outputs.detach().cpu().numpy()
            # 将真实目标值从GPU移到CPU，并转换为NumPy数组
            actual = targets.detach().cpu().numpy()
            # 定义误差容忍范围：真实值的2%加上0.02的固定值
            tolerance = 0.02 * np.abs(actual) + 0.02  # 更严格的误差范围
            # 计算预测值与真实值的绝对误差小于容忍范围的样本数
            correct += np.sum(np.abs(predicted - actual) < tolerance)
            # 累加当前批次的样本数
            total += len(actual)
        
        # 计算当前轮次的平均训练损失
        train_loss = running_loss / len(train_loader)
        # 计算当前轮次的训练准确率
        train_acc = correct / total
        # 将当前轮次的训练损失添加到训练损失列表中
        train_losses.append(train_loss)
        # 将当前轮次的训练准确率添加到训练准确率列表中
        train_accuracies.append(train_acc)
        
        # 验证
        # 将模型设置为评估模式，禁用dropout和batch normalization等层
        model.eval()
        # 初始化验证损失为0
        val_loss = 0.0
        # 初始化正确预测的样本数为0
        correct = 0
        # 初始化总样本数为0
        total = 0
        
        # 使用torch.no_grad()上下文管理器，禁用梯度计算，节省内存并加速计算
        with torch.no_grad():
            # 遍历验证数据加载器中的每个批次
            for sequences, targets in val_loader:
                # 将序列和目标值移动到指定的设备（CPU或GPU）
                sequences, targets = sequences.to(device), targets.to(device)
                # 前向传播：将输入序列送入模型，得到预测输出
                outputs = model(sequences)
                # 计算预测输出与真实目标值之间的损失
                loss = criterion(outputs, targets)
                
                # 将当前批次的损失值累加到总验证损失中
                val_loss += loss.item()
                
                # 将预测输出从GPU移到CPU，并转换为NumPy数组
                predicted = outputs.detach().cpu().numpy()
                # 将真实目标值从GPU移到CPU，并转换为NumPy数组
                actual = targets.detach().cpu().numpy()
                # 定义误差容忍范围：真实值的2%加上0.02的固定值
                tolerance = 0.02 * np.abs(actual) + 0.02
                # 计算预测值与真实值的绝对误差小于容忍范围的样本数
                correct += np.sum(np.abs(predicted - actual) < tolerance)
                # 累加当前批次的样本数
                total += len(actual)
        
        # 计算当前轮次的平均验证损失
        val_loss = val_loss / len(val_loader)
        # 计算当前轮次的验证准确率
        val_acc = correct / total
        # 将当前轮次的验证损失添加到验证损失列表中
        val_losses.append(val_loss)
        # 将当前轮次的验证准确率添加到验证准确率列表中
        val_accuracies.append(val_acc)
        
        # 更新学习率
        # 根据验证损失调整学习率，如果验证损失不再下降，则降低学习率
        scheduler.step(val_loss)
        
        # 打印当前轮次的训练和验证信息
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停机制
        # 如果当前验证损失小于最佳验证损失
        if val_loss < best_val_loss:
            # 更新最佳验证损失
            best_val_loss = val_loss
            # 保存当前模型状态的副本
            best_model_state = model.state_dict().copy()
            # 重置早停计数器
            patience_counter = 0
        else:
            # 如果验证损失没有改善，增加早停计数器
            patience_counter += 1
            # 如果早停计数器达到设定的耐心值
            if patience_counter >= patience:
                # 打印早停信息
                print(f'Early stopping at epoch {epoch+1}')
                # 跳出训练循环
                break
    
    # 加载最佳模型
    # 如果存在最佳模型状态
    if best_model_state is not None:
        # 将最佳模型状态加载到模型中
        model.load_state_dict(best_model_state)
    
    # 返回训练损失列表、验证损失列表、训练准确率列表和验证准确率列表
    return train_losses, val_losses, train_accuracies, val_accuracies

# 测试函数
def test_model(model, test_loader, criterion, test_means=None, test_stds=None):
    # 将模型设置为评估模式，禁用dropout和batch normalization等层
    model.eval()
    # 初始化测试损失为0
    test_loss = 0.0
    # 初始化正确预测的样本数为0
    correct = 0
    # 初始化总样本数为0
    total = 0
    # 创建列表用于存储所有预测值
    all_predictions = []
    # 创建列表用于存储所有真实目标值
    all_targets = []
    
    # 使用torch.no_grad()上下文管理器，禁用梯度计算，节省内存并加速计算
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次，同时获取批次索引
        for i, (sequences, targets) in enumerate(test_loader):
            # 将序列和目标值移动到指定的设备（CPU或GPU）
            sequences, targets = sequences.to(device), targets.to(device)
            # 前向传播：将输入序列送入模型，得到预测输出
            outputs = model(sequences)
            # 计算预测输出与真实目标值之间的损失
            loss = criterion(outputs, targets)
            
            # 将当前批次的损失值累加到总测试损失中
            test_loss += loss.item()
            
            # 将预测输出从GPU移到CPU，并转换为NumPy数组
            predicted = outputs.detach().cpu().numpy()
            # 将真实目标值从GPU移到CPU，并转换为NumPy数组
            actual = targets.detach().cpu().numpy()
            
            # 如果有归一化参数，则反归一化
            if test_means is not None and test_stds is not None:
                # 获取当前批次的样本数
                batch_size = sequences.shape[0]
                # 获取当前批次对应的均值
                batch_means = test_means[i*batch_size:(i+1)*batch_size].squeeze()
                # 获取当前批次对应的标准差
                batch_stds = test_stds[i*batch_size:(i+1)*batch_size].squeeze()
                # 对预测值进行反归一化
                predicted = denormalize_data(predicted, batch_means, batch_stds)
                # 对真实目标值进行反归一化
                actual = denormalize_data(actual, batch_means, batch_stds)
            
            # 将当前批次的预测值添加到所有预测值列表中
            all_predictions.extend(predicted)
            # 将当前批次的真实目标值添加到所有真实目标值列表中
            all_targets.extend(actual)
            
            # 使用原始值计算准确率
            # 定义误差容忍范围：真实值的2%加上0.02的固定值
            tolerance = 0.02 * np.abs(actual) + 0.02  # 更严格的误差范围
            # 计算预测值与真实值的绝对误差小于容忍范围的样本数
            correct += np.sum(np.abs(predicted - actual) < tolerance)
            # 累加当前批次的样本数
            total += len(actual)
    
    # 计算平均测试损失
    test_loss = test_loss / len(test_loader)
    # 计算测试准确率
    test_acc = correct / total
    
    # 打印测试结果
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # 返回测试损失、测试准确率、所有预测值和所有真实目标值
    return test_loss, test_acc, all_predictions, all_targets

# 可视化结果函数
def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, predictions, targets):
    # 创建一个大小为15x5的图形
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    # 在1行3列的子图中选择第1个位置
    plt.subplot(1, 3, 1)
    # 绘制训练损失曲线
    plt.plot(train_losses, label='Train Loss')
    # 绘制验证损失曲线
    plt.plot(val_losses, label='Validation Loss')
    # 设置子图标题
    plt.title('Loss Curves')
    # 设置x轴标签
    plt.xlabel('Epochs')
    # 设置y轴标签
    plt.ylabel('Loss')
    # 显示图例
    plt.legend()
    
    # 绘制准确率曲线
    # 在1行3列的子图中选择第2个位置
    plt.subplot(1, 3, 2)
    # 绘制训练准确率曲线
    plt.plot(train_accuracies, label='Train Accuracy')
    # 绘制验证准确率曲线
    plt.plot(val_accuracies, label='Validation Accuracy')
    # 设置子图标题
    plt.title('Accuracy Curves')
    # 设置x轴标签
    plt.xlabel('Epochs')
    # 设置y轴标签
    plt.ylabel('Accuracy')
    # 显示图例
    plt.legend()
    
    # 绘制预测值与真实值对比
    # 在1行3列的子图中选择第3个位置
    plt.subplot(1, 3, 3)
    # 绘制预测值与真实值的散点图，alpha=0.5设置点的透明度
    plt.scatter(targets, predictions, alpha=0.5)
    # 绘制一条红色虚线，表示完美预测的情况（y=x）
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    # 设置子图标题
    plt.title('Predictions vs Targets')
    # 设置x轴标签
    plt.xlabel('True Values')
    # 设置y轴标签
    plt.ylabel('Predicted Values')
    
    # 自动调整子图参数，使之填充整个图像区域
    plt.tight_layout()
    # 将图形保存为PNG文件
    plt.savefig('simple_results.png')
    # 显示图形
    plt.show()

# 设置超参数
# 输入序列长度，设置为9
seq_length = 9  # 输入序列长度
# 输入维度，设置为1（每个时间步只有一个数值）
input_dim = 1   # 输入维度
# 模型的维度，设置为128（增加模型容量）
d_model = 128   # 增加模型维度
# 多头注意力机制的头数，设置为4
nhead = 4       # 增加多头注意力头数
# Transformer的层数，设置为3
num_layers = 3  # 增加Transformer层数
# 批次大小，设置为32
batch_size = 32
# 学习率，设置为0.001
learning_rate = 0.001
# 训练轮数，设置为200
num_epochs = 200
# dropout比率，设置为0.2（防止过拟合）
dropout = 0.2

# 生成数据
# 调用函数生成等差数列数据，生成5000个序列，每个序列长度为seq_length+1
sequences, targets = generate_arithmetic_sequences(num_sequences=5000, seq_length=seq_length+1)

# 数据归一化
# 调用函数对生成的序列和目标值进行归一化处理
norm_sequences, norm_targets, seq_means, seq_stds = normalize_data(sequences, targets)

# 划分训练集、验证集和测试集
# 使用train_test_split函数将数据集分为训练集和临时集（70%训练，30%临时）
# 同时分割归一化后的序列、归一化后的目标值、序列均值和序列标准差
X_train, X_temp, y_train, y_temp, means_train, means_temp, stds_train, stds_temp = train_test_split(
    norm_sequences, norm_targets, seq_means, seq_stds, test_size=0.3, random_state=42)
# 再次使用train_test_split函数将临时集分为验证集和测试集（各占临时集的50%，即总数据的15%）
X_val, X_test, y_val, y_test, means_val, means_test, stds_val, stds_test = train_test_split(
    X_temp, y_temp, means_temp, stds_temp, test_size=0.5, random_state=42)

# 创建数据加载器
# 使用训练数据创建数据集对象
train_dataset = SequenceDataset(X_train, y_train)
# 使用验证数据创建数据集对象
val_dataset = SequenceDataset(X_val, y_val)
# 使用测试数据创建数据集对象
test_dataset = SequenceDataset(X_test, y_test)

# 创建训练数据加载器，设置批次大小为32，打乱数据顺序
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 创建验证数据加载器，设置批次大小为32，不打乱数据顺序
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# 创建测试数据加载器，设置批次大小为32，不打乱数据顺序
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
# 创建优化的Transformer模型实例，并移动到指定的设备（CPU或GPU）
model = OptimizedTransformerModel(input_dim, d_model, nhead, num_layers, dropout=dropout).to(device)

# 定义损失函数和优化器
# 使用均方误差损失函数，适用于回归任务
criterion = nn.MSELoss()
# 使用AdamW优化器，设置学习率和权重衰减（L2正则化）
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# 使用学习率调度器，当验证损失不再下降时，降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 训练模型
# 打印训练开始信息
print("Starting training...")
# 调用训练函数，传入模型、数据加载器、损失函数、优化器、学习率调度器等参数
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=15
)

# 测试模型
# 打印测试开始信息
print("\nTesting model...")
# 调用测试函数，传入模型、测试数据加载器、损失函数和测试数据的归一化参数
test_loss, test_acc, predictions, targets = test_model(model, test_loader, criterion, means_test, stds_test)

# 可视化结果
# 打印可视化开始信息
print("\nVisualizing results...")
# 调用可视化函数，绘制训练过程中的损失曲线、准确率曲线以及预测值与真实值的对比图
plot_results(train_losses, val_losses, train_accuracies, val_accuracies, predictions, targets)

# 保存模型
# 使用torch.save函数保存模型的参数（状态字典）
torch.save(model.state_dict(), 'optimized_transformer_sequence_predictor.pth')
# 打印模型保存信息
print("\nModel saved as 'optimized_transformer_sequence_predictor.pth'")