"""AW-GeoNet custom detection modules.

Paper name  ->  code class
--------------------------------------------------------------------------
GAMS        ->  MSBlock          (backbone, P3 stage)
                |- _MSBlockUnit  : heterogeneous-kernel multi-branch block
                |- SIBM          : 1x1 expand -> depthwise conv(k) -> 1x1 project
                |- GQL           : global-query guided branch re-weighting
DPS-Conv    ->  GSConv           (neck, selective replacement)
                |- ChannelShuffle
                |- VoV_GSCSP     (unused in the main model, kept for reference)
GPDA        ->  DeformableC2f    (neck, P3 and P4 outputs)
                |- DeformableAttention : dual attention (channel + spatial)  [DDA]
                |- DeformableConv2d    : offset-field resampling           [ADA]

Only MSBlock, GSConv and DeformableC2f are instantiated by the main
configuration (`models/awgeonet_s.yaml`).  The helper classes are kept
public because they are needed to unpickle the released checkpoints.

Implementation comments are retained in Chinese.  It is registered with Ultralytics through
`models/parse_model.py`; see `models/register.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, Bottleneck, C2f
from ultralytics.nn.modules.conv import autopad

# --------------------------------------------------------------------------
# GAMS: MSBlock (adapted from ms_block.py)
# --------------------------------------------------------------------------
class SIBM(nn.Module):
    """简化版倒残差模块（适配backbone轻量化需求）"""

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        # 1. 通道扩展（1×1卷积升维）
        self.expand_conv = nn.Conv2d(
            in_channels, in_channels * 2,
            kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.act = nn.SiLU()

        # 2. 深度可分离卷积
        self.dw_conv = nn.Conv2d(
            in_channels * 2, in_channels * 2,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            groups=in_channels * 2
        )
        self.bn2 = nn.BatchNorm2d(in_channels * 2)

        # 3. 通道压缩
        self.project_conv = nn.Conv2d(
            in_channels * 2, in_channels,
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        residual = x
        out = self.expand_conv(x)
        out = self.act(self.bn1(out))
        out = self.dw_conv(out)
        out = self.act(self.bn2(out))
        out = self.project_conv(out)
        return out + residual


class GQL(nn.Module):
    """全局查询学习模块（增强多分支特征融合）"""

    def __init__(self, num_branches=3, query_dim=4, in_channels=None):
        super().__init__()
        self.num_branches = num_branches
        self.query_dim = query_dim
        self.in_channels = in_channels

        # 全局查询向量
        self.global_query = nn.Parameter(
            torch.randn(1, num_branches, query_dim **2)
        )

        # 线性层（动态计算分支权重）
        self.linear = nn.Linear(
            num_branches * in_channels,
            query_dim** 2
        )

    def forward(self, branch_features):
        B, C, H, W = branch_features[0].shape

        # 全局平均池化
        gap_features = []
        for feat in branch_features:
            gap = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, C)
            gap_features.append(gap)

        # 拼接特征并计算权重
        gap_concat = torch.cat(gap_features, dim=1)
        K = self.linear(gap_concat).unsqueeze(1)  # [B, 1, query_dim²]
        weights = torch.matmul(self.global_query, K.transpose(1, 2))  # [B, N, 1]
        weights = torch.sigmoid(weights)  # 权重归一化到[0,1]

        # 应用权重到各分支
        weighted_feats = []
        for i in range(self.num_branches):
            w = weights[:, i].view(B, 1, 1, 1)  # 广播到特征图尺寸
            weighted_feats.append(branch_features[i] * w)

        return weighted_feats


class _MSBlockUnit(nn.Module):
    """MSBlock基础单元（实现HKS层级化策略）"""

    def __init__(self, c1, c2, hks_kernels=[3, 5, 7], shortcut=True, kernel_strategy=None):
        super().__init__()
        self.in_channels = c1
        self.out_channels = c2
        self.num_branches = len(hks_kernels)
        self.kernel_strategy = kernel_strategy  # 新增：层级策略标记（shallow/deep等）
        self.hks_kernels = hks_kernels  # 新增：保存hks_kernels为实例属性
        # 通道调整（确保残差连接通道一致）
        if c1 != c2:
            self.channel_adjust = nn.Conv2d(c1, c2, 1, 1, 0)
            c1 = c2  # 内部处理统一使用c2
        else:
            self.channel_adjust = None

        # 1. 输入通道扩展（拆分到多分支）
        self.in_conv = nn.Conv2d(
            c1, c1 * self.num_branches,
            kernel_size=1, stride=1, padding=0
        )

        # 2. 多分支特征提取（每个分支配一个SIBM）
        self.branches = nn.ModuleList()
        for kernel in hks_kernels:
            self.branches.append(SIBM(in_channels=c1, kernel_size=kernel))

        # 3. 分支门控机制（根据层级策略动态调整特征权重）
        self.branch_gate = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)
            for _ in hks_kernels
        ])

        # 4. 初始化分支权重（根据层级策略偏置）
        if self.kernel_strategy == "shallow":
            # 浅层：小kernel权重更高
            self.initial_weights = nn.Parameter(torch.tensor([1.2 if k==3 else 0.8 for k in hks_kernels]))
        elif self.kernel_strategy == "deep":
            # 深层：大kernel权重更高
            self.initial_weights = nn.Parameter(torch.tensor([0.8 if k==3 else 1.2 for k in hks_kernels]))
        else:
            self.initial_weights = nn.Parameter(torch.ones(len(hks_kernels)))

        # 5. 全局查询学习模块（GQL）
        self.gql = GQL(
            num_branches=self.num_branches,
            query_dim=4,
            in_channels=c1
        )

        # 6. 输出融合
        self.out_conv = nn.Conv2d(
            c1 * self.num_branches, c2,
            kernel_size=1, stride=1, padding=0
        )
        self.bn_out = nn.BatchNorm2d(c2)
        self.act_out = nn.SiLU()

        # 7. 残差连接
        self.shortcut = shortcut

    def forward(self, x):
        # 残差预处理
        residual = self.channel_adjust(x) if self.channel_adjust is not None else x

        # 1. 输入扩展并拆分到多分支
        x_expand = self.in_conv(x)
        branches = torch.split(x_expand, self.in_channels, dim=1)  # 按通道拆分

        # 2. 多分支计算（带门控和前序特征累加）
        branch_outputs = []
        for i in range(self.num_branches):
            # 前序分支输出与当前分支输入累加（增强特征流通）
            branch_in = branches[i] if i == 0 else branch_outputs[i-1] + branches[i]
            # 门控机制：突出当前kernel擅长的特征
            gated_in = self.branch_gate[i](branch_in) * self.initial_weights[i]
            branch_out = self.branches[i](gated_in)
            branch_outputs.append(branch_out)

        # 3. GQL动态加权融合
        weighted_branches = self.gql(branch_outputs)

        # 4. 输出融合
        branches_concat = torch.cat(weighted_branches, dim=1)
        out = self.out_conv(branches_concat)
        out = self.act_out(self.bn_out(out))

        # 5. 残差连接
        return out + residual if self.shortcut else out


class MSBlock(nn.Module):
    """适配YOLOv8的MSBlock容器（支持层级化kernel配置）"""

    def __init__(self, c1, c2, n=1, hks_kernels=[3, 5, 7], shortcut=True, kernel_strategy=None):
        super().__init__()
        # 根据n创建n个MSBlock基础单元的串联，统一传递层级策略
        self.blocks = nn.Sequential(*[
            _MSBlockUnit(
                c1 if i == 0 else c2,
                c2,
                hks_kernels=hks_kernels,
                shortcut=shortcut,
                kernel_strategy=kernel_strategy  # 传递层级策略
            ) for i in range(n)
        ])

    def forward(self, x):
        return self.blocks(x)


# --------------------------------------------------------------------------
# DPS-Conv: GSConv (adapted from gsconv_modules.py)
# --------------------------------------------------------------------------
class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        assert c2 % 2 == 0, "c2 must be even for GSConv"

        # 标准卷积分支 (SC)
        self.conv_s = nn.Conv2d(c1, c2 // 2, k, s, autopad(k), groups=1, bias=False)

        # 深度可分离卷积分支 (DSC)
        self.conv_d = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, autopad(k), groups=c1, bias=False),
            nn.Conv2d(c1, c2 // 2, 1, 1, 0, bias=False)
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        self.shuffle = ChannelShuffle(2)

        # 基础残差连接
        self.residual = (c1 == c2) and (s == 1)

    def forward(self, x):
        residual = x if self.residual else None

        # 并行特征提取
        x_s = self.conv_s(x)  # 标准卷积特征
        x_d = self.conv_d(x)  # 深度可分离特征

        # 拼接和通道洗牌
        x = torch.cat([x_s, x_d], dim=1)
        x = self.shuffle(x)

        # 残差连接
        if residual is not None:
            x = x + residual

        return self.act(self.bn(x))


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, -1, h, w)
        return x


class VoV_GSCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, se=False):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        self.osaka = nn.Sequential(
            GSConv(c_, c_, 3, 1),
            GSConv(c_, c_, 3, 1)
        )

        self.m = nn.Sequential(*[
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)
        ])

        # ✅ 修复：只在输入输出通道相同时启用跳跃连接
        self.shortcut = shortcut and (c1 == c2)

        self.se = None
        if se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(2 * c_, (2 * c_) // 16, 1),
                nn.SiLU(),
                nn.Conv2d((2 * c_) // 16, 2 * c_, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        residual = x if self.shortcut else None  # ✅ 条件判断

        x1 = self.osaka(self.cv1(x))
        x2 = self.m(self.cv2(x))

        x_cat = torch.cat((x1, x2), dim=1)

        if self.se is not None:
            se_weights = self.se(x_cat)
            x_cat = x_cat * se_weights

        out = self.cv3(x_cat)

        if self.shortcut:
            out = out + residual

        return out


# --------------------------------------------------------------------------
# GPDA: DeformableC2f / DeformableAttention / DeformableConv2d
# (adapted from Deformable_Attention.py)
# --------------------------------------------------------------------------
class DeformableConv2d(nn.Module):
    """
    [最终无警告版] 纯 PyTorch 实现的可变形卷积
    修复内容：
    1. 移除了 self.weight 别名，彻底解决 "duplicate parameters" 警告。
    2. 保持纯 PyTorch 实现 (Grid Sample)，解决 Windows 0xC0000005 崩溃。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()

        # 稳定性检查
        assert stride == 1, "为了稳定性，此版本 DeformableConv2d 仅支持 stride=1"
        assert padding == kernel_size // 2, "Padding 必须等于 kernel_size // 2"

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # 1. 偏移量预测
        self.offset_conv = nn.Conv2d(
            in_channels,
            2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 2. 标准卷积
        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # Padding 已由 grid_sample 处理
            groups=groups,
            bias=bias
        )

        # 【关键修复】已删除 self.weight = self.regular_conv.weight
        # 这行代码是导致 "duplicate parameters" 警告的根源

        self._init_weights()

    def _init_weights(self):
        # 初始化 offset 为 0
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 计算偏移
        offset = self.offset_conv(x)

        # 2. 生成网格
        dtype = offset.dtype
        device = offset.device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )

        # 3. 归一化并叠加偏移
        x_grid = x_grid + offset[:, 0, :, :]
        y_grid = y_grid + offset[:, 1, :, :]
        x_grid = 2.0 * x_grid / (W - 1) - 1.0
        y_grid = 2.0 * y_grid / (H - 1) - 1.0
        grid = torch.stack([x_grid, y_grid], dim=-1)

        # 4. 重采样
        x_deformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # 5. 填充并卷积
        x_deformed_padded = F.pad(x_deformed, (self.padding, self.padding, self.padding, self.padding),
                                  mode='replicate')

        return self.regular_conv(x_deformed_padded)


class DeformableAttention(nn.Module):
    """双重注意力模块"""

    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力：使用稳定的大核卷积
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x_ca = x * ca

        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)

        sa = self.spatial_attention(spatial_input)
        return x_ca * sa


class DeformableC2f(nn.Module):
    """
    集成可变形卷积的 C2f 模块
    包含智能参数修正逻辑，防止 n=0 或参数错位
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()

        # --- [智能参数修正] ---
        # 自动检测并修正 YAML 参数错位 (例如传入了 [512, 1, False])
        if isinstance(shortcut, int) and isinstance(g, bool):
            shortcut = g  # 修正 shortcut
            g = 1  # 恢复 g 为默认值

        # 强制保障: 防止 m 列表为空
        if n < 1:
            # print(f"Note: DeformableC2f auto-corrected n={n} to 1.")
            n = 1
        # ---------------------

        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.shortcut = shortcut

        self.m = nn.ModuleList()
        for _ in range(n):
            # T_j = SiLU(BN(ADA(DDA(x))))  -- Eq. (3) of the paper
            self.m.append(
                nn.Sequential(
                    DeformableAttention(self.c),
                    DeformableConv2d(self.c, self.c, kernel_size=3, padding=1, groups=1),
                    nn.BatchNorm2d(self.c),
                    nn.SiLU(),
                )
            )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1)) + x if self.shortcut else self.cv2(torch.cat(y, 1))
