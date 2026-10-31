

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, Bottleneck, C2f
from ultralytics.nn.modules.conv import autopad

class SIBM(nn.Module):

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        self.expand_conv = nn.Conv2d(
            in_channels, in_channels * 2,
            kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.act = nn.SiLU()

        self.dw_conv = nn.Conv2d(
            in_channels * 2, in_channels * 2,
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            groups=in_channels * 2
        )
        self.bn2 = nn.BatchNorm2d(in_channels * 2)

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

    def __init__(self, num_branches=3, query_dim=4, in_channels=None):
        super().__init__()
        self.num_branches = num_branches
        self.query_dim = query_dim
        self.in_channels = in_channels

        self.global_query = nn.Parameter(
            torch.randn(1, num_branches, query_dim **2)
        )

        self.linear = nn.Linear(
            num_branches * in_channels,
            query_dim** 2
        )

    def forward(self, branch_features):
        B, C, H, W = branch_features[0].shape

        gap_features = []
        for feat in branch_features:
            gap = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, C)
            gap_features.append(gap)

        gap_concat = torch.cat(gap_features, dim=1)
        K = self.linear(gap_concat).unsqueeze(1)
        weights = torch.matmul(self.global_query, K.transpose(1, 2))
        weights = torch.sigmoid(weights)

        weighted_feats = []
        for i in range(self.num_branches):
            w = weights[:, i].view(B, 1, 1, 1)
            weighted_feats.append(branch_features[i] * w)

        return weighted_feats

class _MSBlockUnit(nn.Module):

    def __init__(self, c1, c2, hks_kernels=[3, 5, 7], shortcut=True, kernel_strategy=None):
        super().__init__()
        self.in_channels = c1
        self.out_channels = c2
        self.num_branches = len(hks_kernels)
        self.kernel_strategy = kernel_strategy
        self.hks_kernels = hks_kernels

        if c1 != c2:
            self.channel_adjust = nn.Conv2d(c1, c2, 1, 1, 0)
            c1 = c2
        else:
            self.channel_adjust = None

        self.in_conv = nn.Conv2d(
            c1, c1 * self.num_branches,
            kernel_size=1, stride=1, padding=0
        )

        self.branches = nn.ModuleList()
        for kernel in hks_kernels:
            self.branches.append(SIBM(in_channels=c1, kernel_size=kernel))

        self.branch_gate = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)
            for _ in hks_kernels
        ])

        if self.kernel_strategy == "shallow":

            self.initial_weights = nn.Parameter(torch.tensor([1.2 if k==3 else 0.8 for k in hks_kernels]))
        elif self.kernel_strategy == "deep":

            self.initial_weights = nn.Parameter(torch.tensor([0.8 if k==3 else 1.2 for k in hks_kernels]))
        else:
            self.initial_weights = nn.Parameter(torch.ones(len(hks_kernels)))

        self.gql = GQL(
            num_branches=self.num_branches,
            query_dim=4,
            in_channels=c1
        )

        self.out_conv = nn.Conv2d(
            c1 * self.num_branches, c2,
            kernel_size=1, stride=1, padding=0
        )
        self.bn_out = nn.BatchNorm2d(c2)
        self.act_out = nn.SiLU()

        self.shortcut = shortcut

    def forward(self, x):

        residual = self.channel_adjust(x) if self.channel_adjust is not None else x

        x_expand = self.in_conv(x)
        branches = torch.split(x_expand, self.in_channels, dim=1)

        branch_outputs = []
        for i in range(self.num_branches):

            branch_in = branches[i] if i == 0 else branch_outputs[i-1] + branches[i]

            gated_in = self.branch_gate[i](branch_in) * self.initial_weights[i]
            branch_out = self.branches[i](gated_in)
            branch_outputs.append(branch_out)

        weighted_branches = self.gql(branch_outputs)

        branches_concat = torch.cat(weighted_branches, dim=1)
        out = self.out_conv(branches_concat)
        out = self.act_out(self.bn_out(out))

        return out + residual if self.shortcut else out

class MSBlock(nn.Module):

    def __init__(self, c1, c2, n=1, hks_kernels=[3, 5, 7], shortcut=True, kernel_strategy=None):
        super().__init__()

        self.blocks = nn.Sequential(*[
            _MSBlockUnit(
                c1 if i == 0 else c2,
                c2,
                hks_kernels=hks_kernels,
                shortcut=shortcut,
                kernel_strategy=kernel_strategy
            ) for i in range(n)
        ])

    def forward(self, x):
        return self.blocks(x)

class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        assert c2 % 2 == 0, "c2 must be even for GSConv"

        self.conv_s = nn.Conv2d(c1, c2 // 2, k, s, autopad(k), groups=1, bias=False)

        self.conv_d = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, autopad(k), groups=c1, bias=False),
            nn.Conv2d(c1, c2 // 2, 1, 1, 0, bias=False)
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        self.shuffle = ChannelShuffle(2)

        self.residual = (c1 == c2) and (s == 1)

    def forward(self, x):
        residual = x if self.residual else None

        x_s = self.conv_s(x)
        x_d = self.conv_d(x)

        x = torch.cat([x_s, x_d], dim=1)
        x = self.shuffle(x)

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
        residual = x if self.shortcut else None

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

class DeformableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()

        assert stride == 1, "For stability, DeformableConv2d supports stride=1 only"
        assert padding == kernel_size // 2, "padding must equal kernel_size // 2"

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.offset_conv = nn.Conv2d(
            in_channels,
            2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias
        )

        self._init_weights()

    def _init_weights(self):

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        B, C, H, W = x.shape

        offset = self.offset_conv(x)

        dtype = offset.dtype
        device = offset.device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )

        x_grid = x_grid + offset[:, 0, :, :]
        y_grid = y_grid + offset[:, 1, :, :]
        x_grid = 2.0 * x_grid / (W - 1) - 1.0
        y_grid = 2.0 * y_grid / (H - 1) - 1.0
        grid = torch.stack([x_grid, y_grid], dim=-1)

        x_deformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        x_deformed_padded = F.pad(x_deformed, (self.padding, self.padding, self.padding, self.padding),
                                  mode='replicate')

        return self.regular_conv(x_deformed_padded)

class DeformableAttention(nn.Module):

    def __init__(self, channels, reduction=16, kernel_size=3):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

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

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()

        if isinstance(shortcut, int) and isinstance(g, bool):
            shortcut = g
            g = 1

        if n < 1:

            n = 1

        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.shortcut = shortcut

        self.m = nn.ModuleList()
        for _ in range(n):

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
