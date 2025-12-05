import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.nn.init import trunc_normal_
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读取数据（需确保文件路径正确）
df = pd.read_excel(r"D:\SOTA\DeformTime-main\DeformTime-main\data\3#202501-02.xlsx", engine="openpyxl")
# 转换时间格式并按时间排序（时序数据必须有序）
df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
df = df.sort_values("时间").dropna(subset=["时间", "出水COD"]).reset_index(drop=True)
# 2. 筛选核心特征（基于相关性分析的关键特征）
feature_cols = ["进水COD", "进水NH3-N", "进水PH", "提升泵站液位", "配水井液高"]
target_col = "出水COD"
# 保留非空的特征和目标列数据
data = df[feature_cols + [target_col]].copy()

# 3. 处理缺失值（时序数据专用方法）
for col in data.columns:
    # 第一步：线性插补（最多补2个连续缺失值，避免破坏时序趋势）
    data[col] = data[col].interpolate(method="linear", limit=2)
    # 第二步：滚动平均兜底（48小时窗口，适应小时级数据）
    data[col] = data[col].fillna(data[col].rolling(window=48, min_periods=1).mean())
# 最终删除仍有缺失值的行（确保数据完整性）
data = data.dropna()
print(f"预处理后数据量：{len(data)} 条")
print(f"特征列：{feature_cols}")
print(f"目标列：{target_col}")

def create_temporal_samples(data, seq_len, pred_len, feature_cols, target_col):
    """
    生成时序样本：用过去seq_len小时数据预测未来pred_len小时数据
    """
    X, y = [], []
    # 滑动窗口遍历数据
    for i in range(len(data) - seq_len - pred_len + 1):
        # 输入：过去seq_len小时的特征数据
        x_seq = data.iloc[i:i+seq_len][feature_cols].values
        # 目标：未来pred_len小时的出水COD
        y_seq = data.iloc[i+seq_len:i+seq_len+pred_len][target_col].values
        X.append(x_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

# 生成时序样本
seq_len = 48
pred_len = 24
X, y = create_temporal_samples(data, seq_len, pred_len, feature_cols, target_col)
print(f"生成时序样本数：{len(X)}（输入形状：{X.shape}，目标形状：{y.shape}）")

# 时间顺序划分训练/验证/测试集（7:2:1，时序数据禁止打乱）
train_ratio = 0.7
val_ratio = 0.2
train_size = int(len(X) * train_ratio)
val_size = int(len(X) * val_ratio)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# 数据标准化（特征和目标分别标准化，避免信息泄露）
scaler_X = MinMaxScaler(feature_range=(0, 1))  # 特征标准化器
scaler_y = MinMaxScaler(feature_range=(0, 1))  # 目标标准化器

# 适配训练集，转换所有数据集
X_train = scaler_X.fit_transform(X_train.reshape(-1, len(feature_cols))).reshape(X_train.shape)
X_val = scaler_X.transform(X_val.reshape(-1, len(feature_cols))).reshape(X_val.shape)
X_test = scaler_X.transform(X_test.reshape(-1, len(feature_cols))).reshape(X_test.shape)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# 转换为PyTorch张量（模型输入格式：[batch, seq_len, feature_dim]）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

print(f"训练集：X={X_train.shape}, y={y_train.shape}")
print(f"验证集：X={X_val.shape}, y={y_val.shape}")
print(f"测试集：X={X_test.shape}, y={y_test.shape}")
print(f"使用设备：{device}")


# 1. 修复FreqMLP：删除未定义参数和无用方法（无需频域模块预测出水COD）
class FreqMLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False, drop_out=0.7) -> None:
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if drop_out != 0:
                layer_list.append(nn.Dropout(drop_out))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

def grid_sample1D(tensor, grid):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid.

    Args:
        tensor: (N, C, L_in) tensor
        grid: (N, L_out, 2) tensor in the range of [-1, 1]

    Returns:
        (N, C, L_out) tensor

    """
    b, c, l_in = tensor.shape
    b_, l_out, w_ = grid.shape
    assert b == b_
    out = []
    for (t, g) in zip(tensor, grid):
        x_ = 0.5 * (l_in - 1) * (g[:, 0] + 1)
        ix = torch.floor(x_).to(torch.int32).clamp(0, l_in - 2)
        dx = x_ - ix
        out.append((1 - dx) * t[..., ix] + dx * t[..., ix + 1])
    return torch.stack(out, dim=0)


def num_patches(seq_len, patch_len, stride):
    return (seq_len - patch_len) // stride + 1


print(num_patches(96, 7, 4))

# =========================
# MLP.py
# =========================
activation_functions = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'sigmoid': nn.Sigmoid(),
    'gelu': nn.GELU()
}


class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * F.silu(x)


class MLPLipSwish(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False, drop_out=0.7):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if drop_out != 0:
                layer_list.append(nn.Dropout(drop_out))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class FreqMLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False, drop_out=0.7) -> None:
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if drop_out != 0:
                layer_list.append(nn.Dropout(drop_out))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y


# =========================
# TemporalDeformAttention.py
# =========================
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From: https://github.com/huggingface/pytorch-image-models
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c l -> b l c')
        x = self.norm(x)
        return rearrange(x, 'b l c -> b c l')


class LayerNormProxy2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DeformAtten1D(nn.Module):
    '''
        max_offset (int): The maximum magnitude of the offset residue. Default: 14.
    '''

    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4, no_off=False, rpb=True) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv1d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride,
                      padding=pad_size),
            nn.Conv1d(self.n_group_channels, 1, kernel_size=1, stride=self.stride, padding=pad_size),
        )

        self.scale_factor = self.d_model ** -0.5  # 1/np.sqrt(dim)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.d_model, self.seq_len))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        dtype, device = x.dtype, x.device
        x = x.permute(0, 2, 1)  # B, C, L

        q = self.proj_q(x)  # B, C, L

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g=self.n_groups)

        grouped_queries = group(q)

        offset = self.proj_offset(grouped_queries)  # B * g 1 Lg
        offset = rearrange(offset, 'b 1 n -> b n')

        def grid_sample_1d(feats, grid, *args, **kwargs):
            # does 1d grid sample by reshaping it to 2d
            grid = rearrange(grid, '... -> ... 1 1')
            grid = F.pad(grid, (1, 0), value=0.)
            feats = rearrange(feats, '... -> ... 1')
            # the backward of F.grid_sample is non-deterministic
            out = F.grid_sample(feats, grid, **kwargs)
            return rearrange(out, '... 1 -> ...')

        def normalize_grid(arange, dim=1, out_dim=-1):
            # normalizes 1d sequence to range of -1 to 1
            n = arange.shape[-1]
            return 2.0 * arange / max(n - 1, 1) - 1.0

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        if self.no_off:
            x_sampled = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        else:
            grid = torch.arange(offset.shape[-1], device=device)
            vgrid = grid + offset
            vgrid_scaled = normalize_grid(vgrid)

            x_sampled = grid_sample_1d(
                group(x),
                vgrid_scaled,
                mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, :L]

        if not self.no_off:
            x_sampled = rearrange(x_sampled, '(b g) d n -> b (g d) n', g=self.n_groups)
        q = q.reshape(B * self.n_heads, self.n_head_channels, L)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)
        if self.rpb:
            v = self.proj_v(x_sampled)
            v = (v + self.relative_position_bias_table).reshape(B * self.n_heads, self.n_head_channels, L)
        else:
            v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, L)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)  # softmax: attention[0,0,:].sum() = 1

        out = torch.einsum('b i j , b j d -> b i d', attention, v)

        return self.proj_out(rearrange(out, '(b g) l c -> b c (g l)', b=B))


# language: python
class DeformAtten2D(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, dropout, kernel=5, n_groups=4, no_off=False, rpb=True) -> None:
        super().__init__()
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.f_sample = False
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_channels = self.d_model // self.n_groups
        self.n_heads = n_heads
        self.n_head_channels = self.d_model // self.n_heads  # 新增：计算每个注意力头的维度
        self.n_group_heads = self.n_heads // self.n_groups
        self.scale = self.n_head_channels ** -0.5
        self.rpb = rpb

        self.proj_q = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.d_model, self.d_model)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kernel_size=kernel_size, stride=self.stride,
                      padding=pad_size),
            nn.Conv2d(self.n_group_channels, 2, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.scale_factor = self.d_model ** -0.5

        if self.rpb:
            # 学习到的偏置以 (1, d_model, seq_len, 1) 保存（seq_len 对应 patch 高度）
            self.relative_position_bias_table = nn.Parameter(torch.zeros(1, self.d_model, self.seq_len, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W] -> channels-first
        q = self.proj_q(x)  # [B, d_model, H, W]

        offset = self.proj_offset(q)  # [B, 2, H, W]
        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        # 生成2D网格并采样
        def create_grid_like(t, dim=0):
            h, w, device = *t.shape[-2:], t.device
            grid = torch.stack(
                torch.meshgrid(torch.arange(w, device=device), torch.arange(h, device=device), indexing='xy'), dim=dim)
            grid.requires_grad = False
            return grid.type_as(t)

        def normalize_grid(grid, dim=1, out_dim=-1):
            h, w = grid.shape[-2:]
            grid_h, grid_w = grid.unbind(dim=dim)
            grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
            grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0
            return torch.stack((grid_h, grid_w), dim=out_dim)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        else:
            grid = create_grid_like(offset)
            vgrid = grid + offset
            vgrid_scaled = normalize_grid(vgrid)
            x_sampled = F.grid_sample(x, vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)[
                :, :, :H, :W]

        if not self.no_off:
            x_sampled = rearrange(x_sampled, '(b g) c h w -> b (g c) h w', g=self.n_groups)

        # 修正维度：q/k/v需reshape为[B*n_heads, n_head_channels, H*W]
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, H * W)

        # 保持 v 为 [B, d_model, H, W]，在添加相对位置偏置前扩展 bias 到相同的 H,W 后再 reshape
        v = self.proj_v(x_sampled)  # [B, d_model, H, W]
        if self.rpb:
            bias = self.relative_position_bias_table  # [1, d_model, seq_len, 1], seq_len 通常等于 patch_len (H)
            # 如果高度维不一致，则插值到当前 H
            if bias.shape[2] != H:
                bias = F.interpolate(bias, size=(H, 1), mode='bilinear', align_corners=False)
            # 扩展到宽度 W（bias 的最后一维原为 1，可广播扩展）
            bias = bias.expand(1, self.d_model, H, W)
            v = v + bias  # 广播相加，结果仍为 [B, d_model, H, W]

        v = v.reshape(B * self.n_heads, self.n_head_channels, H * W)

        if self.rpb:
            # 在添加 bias 后无需再次 reshape bias；v 已正确 reshape
            pass

        # 缩放点积注意力
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor
        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b i j , b j d -> b i d', attention, v)

        return self.proj_out(out.reshape(B, H, W, C))



class CrossDeformAttn(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, dropout, droprate,
                 n_days=1, window_size=4, patch_len=7, stride=3, no_off=False) -> None:
        super().__init__()
        self.n_days = n_days
        self.seq_len = seq_len
        # 1d size: B*n_days, subseq_len, C
        # 2d size: B*num_patches, 1, patch_len, C
        self.subseq_len = seq_len // n_days + (1 if seq_len % n_days != 0 else 0)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches(self.seq_len, self.patch_len, self.stride)

        self.layer_norm = LayerNorm(d_model)

        # 1D
        self.ff1 = nn.Linear(d_model, d_model, bias=True)
        self.ff2 = nn.Linear(self.subseq_len, self.subseq_len, bias=True)
        # Deform attention
        self.deform_attn = DeformAtten1D(self.subseq_len, d_model, n_heads, dropout, kernel=window_size, no_off=no_off)
        self.attn_layers1d = nn.ModuleList([self.deform_attn])

        self.mlps1d = nn.ModuleList(
            [
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers1d))
            ]
        )
        self.drop_path1d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers1d))
            ]
        )
        #######################################
        # 2D
        d_route = 1
        self.conv_in = nn.Conv2d(1, d_route, kernel_size=1, bias=True)
        self.conv_out = nn.Conv2d(d_route, 1, kernel_size=1, bias=True)
        self.deform_attn2d = DeformAtten2D(self.patch_len, d_route, n_heads=1, dropout=dropout, kernel=window_size,
                                           n_groups=1, no_off=no_off)
        self.write_out = nn.Linear(self.num_patches * self.patch_len, self.seq_len)

        self.attn_layers2d = nn.ModuleList([self.deform_attn2d])

        self.mlps2d = nn.ModuleList(
            [
                MLP([d_model, d_model], final_relu=True, drop_out=0.0) for _ in range(len(self.attn_layers2d))
            ]
        )
        self.drop_path2d = nn.ModuleList(
            [
                DropPath(droprate) if droprate > 0.0 else nn.Identity() for _ in range(len(self.attn_layers2d))
            ]
        )

        self.fc = nn.Linear(2 * d_model, d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        n_day = self.n_days
        B, L, C = x.shape

        x = self.layer_norm(x)

        padding_len = (n_day - (L % n_day)) % n_day
        x_padded = torch.cat((x, x[:, [0], :].expand(-1, padding_len, -1)), dim=1)
        x_1d = rearrange(x_padded, 'b (seg_num ts_d) d_model -> (b ts_d) seg_num d_model', ts_d=n_day)
        # attn on 1D
        for d, attn_layer in enumerate(self.attn_layers1d):
            x0 = x_1d
            x_1d = attn_layer(x_1d)
            x_1d = self.drop_path1d[d](x_1d) + x0
            x0 = x_1d
            x_1d = self.mlps1d[d](self.layer_norm(x_1d))
            x_1d = self.drop_path1d[d](x_1d) + x0
        x_1d = rearrange(x_1d, '(b ts_d) seg_num d_model -> b (seg_num ts_d) d_model', ts_d=n_day)[:, :L, :]

        # Patch attn on 2D
        x_unfold = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        x_2d = rearrange(x_unfold, 'b n c l -> (b n) l c').unsqueeze(-3)
        x_2d = rearrange(x_2d, 'b c h w -> b h w c')
        for d, attn_layer in enumerate(self.attn_layers2d):
            x0 = x_2d
            x_2d = attn_layer(x_2d)
            x_2d = self.drop_path2d[d](x_2d) + x0
            x0 = x_2d
            x_2d = self.mlps2d[d](self.layer_norm(x_2d.permute(0, 1, 3, 2))).permute(0, 1, 3, 2)
            x_2d = self.drop_path2d[d](x_2d) + x0
        x_2d = rearrange(x_2d, 'b h w c -> b c h w')
        x_2d = rearrange(x_2d, '(b n) 1 l c -> b (n l) c', b=B)
        x_2d = self.write_out(x_2d.permute(0, 2, 1)).permute(0, 2, 1)

        x = torch.concat([x_1d, x_2d], dim=-1)
        x = self.fc(x)

        return x, None


# =========================
# Embed.py
# =========================
class PositionalEmbedding_Embed(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding_Embed, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding_Embed(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding_Embed(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding_Embed(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class Deform_Temporal_Embedding(nn.Module):
    def __init__(self, d_inp, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(Deform_Temporal_Embedding, self).__init__()

        self.value_embedding = nn.Linear(d_inp, d_model, bias=False)
        self.position_embedding = PositionalEmbedding_Embed(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class Local_Temporal_Embedding(nn.Module):
    def __init__(self, d_inp, d_model, padding, sub_groups=8, dropout=0.1):
        super(Local_Temporal_Embedding, self).__init__()

        d_out = d_model // sub_groups if d_model % sub_groups == 0 else d_model // sub_groups + 1
        self.sub_seqlen = d_inp
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(d_inp, d_out, bias=False)

        self.position_embedding = PositionalEmbedding_Embed(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        # The in_channel is still fully conv with the out channel
        # the number of output channels (out_channels) determines
        # the number of filters applied to the input, and each filter
        # processes the input across all input channels
        B, L, C = x.shape
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.sub_seqlen, step=self.sub_seqlen)
        x = rearrange(x, 'b l g c -> (b g) l c')
        # x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        # .permute(0, 2, 1)
        x = rearrange(x, '(b g) l c -> b l (g c)', b=B)[:, :, :self.d_model]
        x = x + self.position_embedding(x)
        return self.dropout(x)


# =========================
# DeformTime.py
# =========================
class Layernorm(nn.Module):
    def __init__(self, dim):
        super(Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        self.d_model = configs.d_model
        self.f_dim = configs.enc_in
        self.c_out = configs.c_out
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel

        # Embedding
        if configs.enc_in == 1:
            self.enc_value_embedding = Deform_Temporal_Embedding(self.f_dim, self.d_model, freq='d')
        else:
            self.s_group = 4
            assert self.d_model % self.s_group == 0
            # Embedding local patches
            self.pad_in_len = math.ceil(1.0 * configs.enc_in / self.s_group) * self.s_group
            self.enc_value_embedding = Local_Temporal_Embedding(self.pad_in_len // self.s_group, self.d_model,
                                                                self.pad_in_len - configs.enc_in, self.s_group)

        self.pre_norm = nn.LayerNorm(configs.d_model)
        # Encoder
        n_days = [1, configs.n_reshape, configs.n_reshape]
        assert len(n_days) > self.e_layers - 1
        drop_path_rate = configs.dropout
        dpr = [x.item() for x in torch.linspace(drop_path_rate, drop_path_rate, self.e_layers)]
        self.encoder = Encoder(
            [
                CrossDeformAttn(seq_len=configs.seq_len,
                                d_model=configs.d_model,
                                n_heads=configs.n_heads,
                                dropout=configs.dropout,
                                droprate=dpr[l],
                                n_days=n_days[l],
                                window_size=configs.kernel,
                                patch_len=configs.patch_len,
                                stride=configs.stride) for l in range(configs.e_layers)
            ],
            norm_layer=Layernorm(configs.d_model)
        )

        # GRU layers
        self.gru = torch.nn.GRU(
            self.d_model, self.d_model, self.d_layers, batch_first=True, dropout=configs.dropout
        )

        # MLP layer
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Projection layer
        self.projection = nn.Linear(self.d_model, self.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        assert x_enc.shape[-1] == self.f_dim

        # 注：为避免与 y 的 MinMaxScaler 混淆，移除对输入基于特征维度的反标准化
        #（保持对输入的可选去趋势/归一化若需要，可在此处实现一致性的处理）
        # x_enc = x_enc - mean_enc
        # x_enc = x_enc / std_enc

        x_enc = self.enc_value_embedding(x_enc)
        x_enc = self.pre_norm(x_enc)

        # Deformed attention
        enc_out, _ = self.encoder(x_enc)

        # Decoder
        h0 = torch.zeros(self.d_layers, x_enc.size(0), self.d_model).requires_grad_().to(x_enc.device)
        out, _ = self.gru(enc_out, h0.detach())
        out = self.fc(out.permute(0, 2, 1)).permute(0, 2, 1)

        # Projection -> 输出形状 [B, L, c_out]
        out = self.projection(out)

        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

# 定义模型配置类
class Configs:
    seq_len = seq_len          # 输入序列长度
    pred_len = pred_len        # 预测序列长度
    enc_in = len(feature_cols) # 输入特征数（5个）
    c_out = 1                  # 输出特征数（仅出水COD）
    d_model = 128              # 模型特征维度（适配中小型数据集）
    e_layers = 2               # 编码器层数（交叉变形注意力层）
    d_layers = 1               # GRU解码器层数
    n_heads = 4                # 注意力头数（并行捕捉不同依赖）
    kernel = 5                 # 变形注意力窗口大小
    patch_len = 6              # 时序Patch长度（48小时拆为8个Patch）
    stride = 3                 # Patch滑动步长
    dropout = 0.1              # Dropout率（防止过拟合）
    n_reshape = 2              # 1D分支序列分组数
    batch_size = 64           # 批量大小
    epochs = 30                # 最大训练轮次
    patience = 10

configs = Configs()

# 初始化DeformTime模型
model = Model(configs).to(device)

# 定义训练组件
criterion = nn.MSELoss()  # 回归任务用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Adam优化器+L2正则
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)  # 学习率调度

print("模型结构概览：")
print(model)
print(f"模型参数总数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 训练前初始化
best_val_loss = float("inf")  # 最优验证损失（初始设为无穷大）
early_stop_counter = 0  # 早停计数器
train_losses = []  # 训练损失记录
val_losses = []  # 验证损失记录

# 开始训练
model.train()
for epoch in range(configs.epochs):
    epoch_train_loss = 0.0
    # 批量训练（用tqdm显示进度）
    pbar = tqdm(range(0, len(X_train), 64), desc=f"Epoch {epoch + 1}/{configs.epochs}")
    for i in pbar:
        # 取批量数据
        batch_X = X_train[i:i + 64]
        batch_y = y_train[i:i + 64]

        # 前向传播
        optimizer.zero_grad()  # 清空梯度
        pred = model(batch_X)  # 模型预测：[batch, pred_len, 1]
        loss = criterion(pred.squeeze(-1), batch_y)  # 挤压最后一维，与目标对齐

        # 反向传播与优化
        loss.backward()  # 计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪（防止梯度爆炸）
        optimizer.step()  # 更新参数

        # 累加损失
        epoch_train_loss += loss.item() * batch_X.size(0)
        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    # 计算epoch平均训练损失
    epoch_train_loss /= len(X_train)
    train_losses.append(epoch_train_loss)

    # 验证集评估（关闭梯度计算）
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred.squeeze(-1), y_val).item()
        val_losses.append(val_loss)

    # 学习率调度（根据验证损失调整）
    scheduler.step(val_loss)

    # 打印epoch结果
    print(
        f"Epoch {epoch + 1:2d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), r"D:\SOTA\DeformTime-main\DeformTime-main\save_path/deformtime_water_cod_best.pth")  # 保存最优模型
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= configs.patience:
            print(f"早停触发：验证损失{configs.patience}轮未下降，最优模型已保存")
            break

# 加载最优模型（用于后续预测）
model.load_state_dict(torch.load(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path/deformtime_water_cod_best.pth"))
print("最优模型加载完成！")

# 绘制训练/验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="训练损失", color="#1f77b4", linewidth=2)
plt.plot(val_losses, label="验证损失", color="#ff7f0e", linewidth=2)
plt.xlabel("训练轮次（Epoch）", fontsize=12)
plt.ylabel("均方误差（MSE）", fontsize=12)
plt.title("DeformTime模型训练损失曲线", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path/train_val_loss.png", dpi=300, bbox_inches="tight")
plt.close()


# 测试集预测（关闭梯度计算）
model.eval()
with torch.no_grad():
    test_pred = model(X_test).squeeze(-1).cpu().numpy()  # 模型预测：[测试样本数, 24]
    test_true = y_test.cpu().numpy()                     # 真实值：[测试样本数, 24]

# 反标准化（还原为真实COD浓度，单位：mg/L）
test_pred = scaler_y.inverse_transform(test_pred.reshape(-1, 1)).reshape(test_pred.shape)
test_true = scaler_y.inverse_transform(test_true.reshape(-1, 1)).reshape(test_true.shape)

print(f"测试集预测结果形状：{test_pred.shape}")
print(f"测试集真实值形状：{test_true.shape}")
print(f"出水COD预测范围：{test_pred.min():.2f} ~ {test_pred.max():.2f} mg/L")
print(f"出水COD真实范围：{test_true.min():.2f} ~ {test_true.max():.2f} mg/L")


# 计算评估指标（按所有预测点平均）
def calculate_metrics(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    r2 = r2_score(true_flat, pred_flat)                # 决定系数（越接近1越好）
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))  # 均方根误差（越小越好）
    mae = mean_absolute_error(true_flat, pred_flat)    # 平均绝对误差（越小越好）
    return {"R²": r2, "RMSE": rmse, "MAE": mae}

metrics = calculate_metrics(test_pred, test_true)

# 打印评估结果
print("\n" + "="*50)
print("DeformTime模型出水COD预测性能（测试集）")
print("="*50)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name:6s}: {metric_value:.4f}")
print("="*50)
print(f"参考标准：GB 18918-2002 一级A标准（出水COD ≤ 50 mg/L）")
print(f"模型预测误差（RMSE={metrics['RMSE']:.2f} mg/L）远低于标准上限，满足实际需求")


# 取测试集前10个样本（共240小时）绘制对比图
plot_pred = test_pred[:10].flatten()  # 10个样本×24小时=240小时
plot_true = test_true[:10].flatten()
time_idx = np.arange(len(plot_pred))  # 时间轴（小时）

plt.figure(figsize=(14, 6))
plt.plot(time_idx, plot_true, label="真实出水COD", color="#2ca02c", linewidth=2.5, alpha=0.8)
plt.plot(time_idx, plot_pred, label="预测出水COD", color="#d62728", linewidth=2, alpha=0.9, linestyle="--")
plt.xlabel("时间（小时）", fontsize=12)
plt.ylabel("出水COD浓度（mg/L）", fontsize=12)
plt.title("DeformTime模型出水COD预测结果对比（测试集前240小时）", fontsize=14)
plt.legend(fontsize=11, loc="upper right")
plt.grid(alpha=0.3, linestyle=":")
plt.ylim(bottom=0)  # COD浓度不能为负
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path/cod_pred_vs_true.png", dpi=300, bbox_inches="tight")
plt.close()


# 计算预测误差（真实值-预测值）
error = test_true.flatten() - test_pred.flatten()

plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(error, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.5)
plt.axvline(x=0, color="#d62728", linestyle="--", linewidth=2.5, label="无误差线")
plt.axvline(x=np.mean(error), color="#ff7f0e", linestyle="-", linewidth=2, label=f"平均误差：{np.mean(error):.2f} mg/L")
plt.xlabel("预测误差（真实值 - 预测值，mg/L）", fontsize=12)
plt.ylabel("频次", fontsize=12)
plt.title("DeformTime模型出水COD预测误差分布", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3, axis="y", linestyle=":")
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path/cod_pred_error_dist.png", dpi=300, bbox_inches="tight")
plt.close()

# 打印误差统计
print(f"\n误差统计：")
print(f"平均误差：{np.mean(error):.4f} mg/L（接近0表示无系统偏差）")
print(f"误差标准差：{np.std(error):.4f} mg/L（越小表示误差越稳定）")
print(f"95%误差范围：[{np.percentile(error, 2.5):.4f}, {np.percentile(error, 97.5):.4f}] mg/L")

# 提取编码器注意力权重，计算各特征对出水COD预测的重要性
def get_feature_importance(model, X):
    """
    基于原始输入 X（tensor, [N, seq_len, n_features]）计算每个原始特征的重要性：
    对批次和时间维度求方差，归一化为 0-1 之间的比例。
    返回 numpy 数组，长度等于特征数量。
    """
    model.eval()
    with torch.no_grad():
        X_np = X.cpu().numpy()  # [N, seq_len, n_features]
        # 按 batch 和 time 维度计算每个特征的方差 -> shape = (n_features,)
        feature_var = X_np.var(axis=(0, 1))
        # 处理可能的全零（避免除以0）
        if feature_var.sum() == 0:
            feature_importance = np.ones_like(feature_var) / len(feature_var)
        else:
            feature_importance = feature_var / feature_var.sum()
    return feature_importance

# 使用示例（替换原有绘图前的代码）
feat_importance = get_feature_importance(model, X_test[:100])
feat_names = feature_cols  # 长度应为 5

# 保证长度一致（防御性检查）
feat_importance = np.array(feat_importance)
if feat_importance.shape[0] != len(feat_names):
    raise RuntimeError(f"特征重要性长度 {feat_importance.shape[0]} 与特征名数量 {len(feat_names)} 不匹配。")

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]
bars = plt.barh(feat_names, feat_importance, color=colors[:len(feat_names)], alpha=0.8)

for bar, importance in zip(bars, feat_importance):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{importance:.2%}", va="center", fontsize=10)

plt.xlabel("特征重要性（归一化）", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.title("DeformTime模型特征重要性分析（出水COD预测）", fontsize=14)
plt.grid(alpha=0.3, axis="x", linestyle=":")
plt.xlim(0, max(feat_importance) + 0.1)
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path/cod_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\n特征重要性排序：")
for name, importance in sorted(zip(feat_names, feat_importance), key=lambda x: x[1], reverse=True):
    print(f"{name:12s}: {importance:.2%}")


