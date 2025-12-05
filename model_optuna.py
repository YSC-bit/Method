import math
import optuna
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

# ====================== 1. 数据预处理（与原代码一致） ======================
# 读取数据
df = pd.read_excel(r"D:\SOTA\DeformTime-main\DeformTime-main\data\3#202501-02.xlsx", engine="openpyxl")
df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
df = df.sort_values("时间").dropna(subset=["时间", "出水COD"]).reset_index(drop=True)

# 筛选核心特征
feature_cols = ["进水COD", "进水NH3-N", "进水PH", "提升泵站液位", "配水井液高"]
target_col = "出水COD"
data = df[feature_cols + [target_col]].copy()

# 处理缺失值
for col in data.columns:
    data[col] = data[col].interpolate(method="linear", limit=2)
    data[col] = data[col].fillna(data[col].rolling(window=48, min_periods=1).mean())
data = data.dropna()
print(f"预处理后数据量：{len(data)} 条")


# 生成时序样本
def create_temporal_samples(data, seq_len, pred_len, feature_cols, target_col):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x_seq = data.iloc[i:i + seq_len][feature_cols].values
        y_seq = data.iloc[i + seq_len:i + seq_len + pred_len][target_col].values
        X.append(x_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)


seq_len = 48
pred_len = 24
X, y = create_temporal_samples(data, seq_len, pred_len, feature_cols, target_col)
print(f"生成时序样本数：{len(X)}（输入形状：{X.shape}，目标形状：{y.shape}）")

# 划分训练/验证/测试集
train_ratio = 0.7
val_ratio = 0.2
train_size = int(len(X) * train_ratio)
val_size = int(len(X) * val_ratio)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# 数据标准化
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train = scaler_X.fit_transform(X_train.reshape(-1, len(feature_cols))).reshape(X_train.shape)
X_val = scaler_X.transform(X_val.reshape(-1, len(feature_cols))).reshape(X_val.shape)
X_test = scaler_X.transform(X_test.reshape(-1, len(feature_cols))).reshape(X_test.shape)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_val = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# 转换为张量
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


# ====================== 2. 模型定义（与原代码一致） ======================
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
        model = [torch.nn.Linear(in_size, mlp_size), LipSwish()]
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


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, inplace: bool = False, init_values: float = 1e-5):
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
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DeformAtten1D(nn.Module):
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
        self.scale_factor = self.d_model ** -0.5

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.d_model, self.seq_len))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        dtype, device = x.dtype, x.device
        x = x.permute(0, 2, 1)

        q = self.proj_q(x)
        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g=self.n_groups)
        grouped_queries = group(q)
        offset = self.proj_offset(grouped_queries)
        offset = rearrange(offset, 'b 1 n -> b n')

        def grid_sample_1d(feats, grid, *args, **kwargs):
            grid = rearrange(grid, '... -> ... 1 1')
            grid = F.pad(grid, (1, 0), value=0.)
            feats = rearrange(feats, '... -> ... 1')
            out = F.grid_sample(feats, grid, **kwargs)
            return rearrange(out, '... 1 -> ...')

        def normalize_grid(arange, dim=1, out_dim=-1):
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

        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum('b i j , b j d -> b i d', attention, v)
        return self.proj_out(rearrange(out, '(b g) l c -> b c (g l)', b=B))


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
        self.n_head_channels = self.d_model // self.n_heads
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
            self.relative_position_bias_table = nn.Parameter(torch.zeros(1, self.d_model, self.seq_len, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        q = self.proj_q(x)
        offset = self.proj_offset(q)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

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

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, H * W)
        v = self.proj_v(x_sampled)

        if self.rpb:
            bias = self.relative_position_bias_table
            if bias.shape[2] != H:
                bias = F.interpolate(bias, size=(H, 1), mode='bilinear', align_corners=False)
            bias = bias.expand(1, self.d_model, H, W)
            v = v + bias

        v = v.reshape(B * self.n_heads, self.n_head_channels, H * W)
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
        self.subseq_len = seq_len // n_days + (1 if seq_len % n_days != 0 else 0)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches(self.seq_len, self.patch_len, self.stride)

        self.layer_norm = LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model, bias=True)
        self.ff2 = nn.Linear(self.subseq_len, self.subseq_len, bias=True)
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

        for d, attn_layer in enumerate(self.attn_layers1d):
            x0 = x_1d
            x_1d = attn_layer(x_1d)
            x_1d = self.drop_path1d[d](x_1d) + x0
            x0 = x_1d
            x_1d = self.mlps1d[d](self.layer_norm(x_1d))
            x_1d = self.drop_path1d[d](x_1d) + x0
        x_1d = rearrange(x_1d, '(b ts_d) seg_num d_model -> b (seg_num ts_d) d_model', ts_d=n_day)[:, :L, :]

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


class PositionalEmbedding_Embed(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding_Embed, self).__init__()
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
        B, L, C = x.shape
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.sub_seqlen, step=self.sub_seqlen)
        x = rearrange(x, 'b l g c -> (b g) l c')
        x = self.value_embedding(x)
        x = rearrange(x, '(b g) l c -> b l (g c)', b=B)[:, :, :self.d_model]
        x = x + self.position_embedding(x)
        return self.dropout(x)


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

        if configs.enc_in == 1:
            self.enc_value_embedding = Deform_Temporal_Embedding(self.f_dim, self.d_model, freq='d')
        else:
            self.s_group = 4
            assert self.d_model % self.s_group == 0
            self.pad_in_len = math.ceil(1.0 * configs.enc_in / self.s_group) * self.s_group
            self.enc_value_embedding = Local_Temporal_Embedding(self.pad_in_len // self.s_group, self.d_model,
                                                                self.pad_in_len - configs.enc_in, self.s_group)

        self.pre_norm = nn.LayerNorm(configs.d_model)
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

        gru_dropout = self.dropout if self.d_layers > 1 else 0.0
        self.gru = torch.nn.GRU(
            self.d_model, self.d_model, self.d_layers,
            batch_first=True,
            dropout=gru_dropout  # 仅当层数>1时启用dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )

        self.projection = nn.Linear(self.d_model, self.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        assert x_enc.shape[-1] == self.f_dim
        x_enc = self.enc_value_embedding(x_enc)
        x_enc = self.pre_norm(x_enc)
        enc_out, _ = self.encoder(x_enc)

        h0 = torch.zeros(self.d_layers, x_enc.size(0), self.d_model).requires_grad_().to(x_enc.device)
        out, _ = self.gru(enc_out, h0.detach())
        out = self.fc(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.projection(out)

        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


# ====================== 3. Optuna超参数优化 ======================
class OptunaConfigs:
    def __init__(self, trial=None):
        # 固定参数
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = len(feature_cols)
        self.c_out = 1
        self.n_reshape = 2
        self.kernel = 5

        # 待优化参数（由Optuna建议）
        if trial is None:
            # 默认参数（无优化时）
            self.d_model = 128
            self.e_layers = 2
            self.d_layers = 1  # GRU层数
            self.n_heads = 4
            self.patch_len = 6
            self.stride = 3
            self.dropout = 0.0  # d_layers=1时dropout强制为0
            self.batch_size = 64
            self.lr = 1e-4
        else:
            # 超参数搜索空间
            self.d_model = trial.suggest_categorical("d_model", [64, 128, 256])
            self.e_layers = trial.suggest_int("e_layers", 1, 3)
            self.d_layers = trial.suggest_int("d_layers", 1, 2)  # GRU层数最多2层
            self.n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
            self.patch_len = trial.suggest_int("patch_len", 4, 8)
            self.stride = trial.suggest_int("stride", 2, 4)

            # 关键修复：GRU层数=1时，dropout强制为0；否则搜索dropout
            if self.d_layers == 1:
                self.dropout = 0.0
            else:
                self.dropout = trial.suggest_float("dropout", 0.05, 0.2)

            self.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            self.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)


# 定义Optuna目标函数（最小化验证损失）
def objective(trial):
    # 1. 生成当前trial的配置
    configs = OptunaConfigs(trial)

    # 2. 初始化模型
    model = Model(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    # 3. 训练模型（轻量化训练，仅跑少量epoch）
    best_val_loss = float("inf")
    patience = 5  # 早停耐心值
    early_stop_counter = 0

    for epoch in range(15):  # 优化阶段减少训练轮次，加速搜索
        model.train()
        epoch_train_loss = 0.0

        # 批量训练
        for i in range(0, len(X_train), configs.batch_size):
            batch_X = X_train[i:i + configs.batch_size]
            batch_y = y_train[i:i + configs.batch_size]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred.squeeze(-1), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_train_loss += loss.item() * batch_X.size(0)

        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred.squeeze(-1), y_val).item()

        # 学习率调度
        scheduler.step(val_loss)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break

        # 剪枝（Optuna可选功能，提前终止差的trial）
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


# 运行Optuna优化
def run_optuna_optimization(n_trials=20):
    # 创建study，最小化验证损失
    study = optuna.create_study(direction="minimize", study_name="DeformTime_WaterCOD")

    # 运行优化
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 打印优化结果
    print("\n" + "=" * 60)
    print("Optuna超参数优化结果")
    print("=" * 60)
    print(f"最优验证损失: {study.best_value:.4f}")
    print(f"最优参数: {study.best_params}")
    print("=" * 60)

    # 可视化优化结果
    # 3.1 优化历史
    optuna.visualization.plot_optimization_history(study).write_image(
        r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/optuna_history.png")
    # 3.2 参数重要性
    optuna.visualization.plot_param_importances(study).write_image(
        r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/optuna_param_importance.png")
    # 3.3 平行坐标图
    optuna.visualization.plot_parallel_coordinate(study).write_image(
        r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/optuna_parallel_coordinate.png")

    return study.best_params


# 执行超参数优化（建议n_trials=20~50，根据算力调整）
best_params = run_optuna_optimization(n_trials=20)

# ====================== 4. 基于最优参数训练最终模型 ======================
# 构建最优配置
best_configs = OptunaConfigs()
for key, value in best_params.items():
    setattr(best_configs, key, value)

print("\n基于最优参数初始化最终模型...")
print(f"最优配置: {vars(best_configs)}")

# 初始化最终模型
final_model = Model(best_configs).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_configs.lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

# 训练最终模型（完整训练流程）
best_val_loss = float("inf")
early_stop_counter = 0
train_losses = []
val_losses = []

final_model.train()
for epoch in range(30):
    epoch_train_loss = 0.0
    pbar = tqdm(range(0, len(X_train), best_configs.batch_size), desc=f"Final Epoch {epoch + 1}/30")

    for i in pbar:
        batch_X = X_train[i:i + best_configs.batch_size]
        batch_y = y_train[i:i + best_configs.batch_size]

        optimizer.zero_grad()
        pred = final_model(batch_X)
        loss = criterion(pred.squeeze(-1), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_train_loss += loss.item() * batch_X.size(0)
        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    epoch_train_loss /= len(X_train)
    train_losses.append(epoch_train_loss)

    # 验证集评估
    final_model.eval()
    with torch.no_grad():
        val_pred = final_model(X_val)
        val_loss = criterion(val_pred.squeeze(-1), y_val).item()
        val_losses.append(val_loss)

    scheduler.step(val_loss)

    print(
        f"Final Epoch {epoch + 1:2d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 早停与保存最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(final_model.state_dict(),
                   r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/deformtime_water_cod_best_optuna.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= best_configs.patience if hasattr(best_configs, "patience") else 10:
            print(f"早停触发：验证损失{10}轮未下降，最优模型已保存")
            break

# 加载最优模型
final_model.load_state_dict(
    torch.load(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/deformtime_water_cod_best_optuna.pth"))
print("最优模型（Optuna优化后）加载完成！")

# ====================== 5. 模型评估与可视化（与原代码一致） ======================
# 绘制训练/验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="训练损失", color="#1f77b4", linewidth=2)
plt.plot(val_losses, label="验证损失", color="#ff7f0e", linewidth=2)
plt.xlabel("训练轮次（Epoch）", fontsize=12)
plt.ylabel("均方误差（MSE）", fontsize=12)
plt.title("DeformTime模型训练损失曲线（Optuna优化后）", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/train_val_loss_optuna.png", dpi=300,
            bbox_inches="tight")
plt.close()

# 测试集预测
final_model.eval()
with torch.no_grad():
    test_pred = final_model(X_test).squeeze(-1).cpu().numpy()
    test_true = y_test.cpu().numpy()

# 反标准化
test_pred = scaler_y.inverse_transform(test_pred.reshape(-1, 1)).reshape(test_pred.shape)
test_true = scaler_y.inverse_transform(test_true.reshape(-1, 1)).reshape(test_true.shape)


# 计算评估指标
def calculate_metrics(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    r2 = r2_score(true_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
    mae = mean_absolute_error(true_flat, pred_flat)
    return {"R²": r2, "RMSE": rmse, "MAE": mae}


metrics = calculate_metrics(test_pred, test_true)

# 打印评估结果
print("\n" + "=" * 50)
print("DeformTime模型出水COD预测性能（Optuna优化后）")
print("=" * 50)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name:6s}: {metric_value:.4f}")
print("=" * 50)

# 绘制预测对比图
plot_pred = test_pred[:10].flatten()
plot_true = test_true[:10].flatten()
time_idx = np.arange(len(plot_pred))

plt.figure(figsize=(14, 6))
plt.plot(time_idx, plot_true, label="真实出水COD", color="#2ca02c", linewidth=2.5, alpha=0.8)
plt.plot(time_idx, plot_pred, label="预测出水COD", color="#d62728", linewidth=2, alpha=0.9, linestyle="--")
plt.xlabel("时间（小时）", fontsize=12)
plt.ylabel("出水COD浓度（mg/L）", fontsize=12)
plt.title("DeformTime模型出水COD预测结果对比（Optuna优化后）", fontsize=14)
plt.legend(fontsize=11, loc="upper right")
plt.grid(alpha=0.3, linestyle=":")
plt.ylim(bottom=0)
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/cod_pred_vs_true_optuna.png", dpi=300,
            bbox_inches="tight")
plt.close()

# 绘制误差分布
error = test_true.flatten() - test_pred.flatten()
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(error, bins=50, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.5)
plt.axvline(x=0, color="#d62728", linestyle="--", linewidth=2.5, label="无误差线")
plt.axvline(x=np.mean(error), color="#ff7f0e", linestyle="-", linewidth=2, label=f"平均误差：{np.mean(error):.2f} mg/L")
plt.xlabel("预测误差（真实值 - 预测值，mg/L）", fontsize=12)
plt.ylabel("频次", fontsize=12)
plt.title("DeformTime模型出水COD预测误差分布（Optuna优化后）", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3, axis="y", linestyle=":")
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/cod_pred_error_dist_optuna.png", dpi=300,
            bbox_inches="tight")
plt.close()


# 特征重要性分析
def get_feature_importance(model, X):
    model.eval()
    with torch.no_grad():
        X_np = X.cpu().numpy()
        feature_var = X_np.var(axis=(0, 1))
        if feature_var.sum() == 0:
            feature_importance = np.ones_like(feature_var) / len(feature_var)
        else:
            feature_importance = feature_var / feature_var.sum()
    return feature_importance


feat_importance = get_feature_importance(final_model, X_test[:100])
feat_names = feature_cols

plt.figure(figsize=(10, 6))
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]
bars = plt.barh(feat_names, feat_importance, color=colors[:len(feat_names)], alpha=0.8)

for bar, importance in zip(bars, feat_importance):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{importance:.2%}", va="center", fontsize=10)

plt.xlabel("特征重要性（归一化）", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.title("DeformTime模型特征重要性分析（Optuna优化后）", fontsize=14)
plt.grid(alpha=0.3, axis="x", linestyle=":")
plt.xlim(0, max(feat_importance) + 0.1)
plt.savefig(r"D:\SOTA\DeformTime-main\DeformTime-main\save_path\optuna/cod_feature_importance_optuna.png", dpi=300,
            bbox_inches="tight")
plt.close()

# 打印误差统计
print(f"\n误差统计（Optuna优化后）：")
print(f"平均误差：{np.mean(error):.4f} mg/L")
print(f"误差标准差：{np.std(error):.4f} mg/L")
print(f"95%误差范围：[{np.percentile(error, 2.5):.4f}, {np.percentile(error, 97.5):.4f}] mg/L")

# 打印最优参数
print(f"\nOptuna最优参数：")
for key, value in best_params.items():
    print(f"{key:12s}: {value}")