import math
import torch
import torch.nn as nn


class LayerNorm1D(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x_n  = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_n = x_n * self.weight + self.bias
        return x_n


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act  = act_layer() if act_layer else None
        if self.norm:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act:  x = self.act(x)
        return x


class FFN2D(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim,    kernel_size=1, padding=0)
        self.fc2 = ConvLayer2D(dim,    in_dim, kernel_size=1, padding=0,
                               act_layer=None, bn_weight_init=0)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class EfficientScan2D(nn.Module):

    def __init__(self, dim, window_size=7):
        super().__init__()
        self.dim = dim
        self.ws  = window_size


        self.row_gate  = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=window_size,
                      padding=window_size // 2, groups=dim, bias=False),
            nn.Sigmoid()
        )
        self.row_val   = nn.Conv1d(dim, dim, kernel_size=window_size,
                                   padding=window_size // 2, groups=dim, bias=False)

        self.col_gate  = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=window_size,
                      padding=window_size // 2, groups=dim, bias=False),
            nn.Sigmoid()
        )
        self.col_val   = nn.Conv1d(dim, dim, kernel_size=window_size,
                                   padding=window_size // 2, groups=dim, bias=False)


        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid()
        )


        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 3, 1, bias=True),
            nn.Softmax(dim=1)
        )


        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def _row_scan(self, x):
        B, C, H, W = x.shape

        x_row = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        g = self.row_gate(x_row)
        v = self.row_val(x_row)
        out = g * v + (1 - g) * x_row
        out = out.reshape(B, H, C, W).permute(0, 2, 1, 3)
        return out

    def _col_scan(self, x):
        B, C, H, W = x.shape

        x_col = x.permute(0, 3, 1, 2).reshape(B * W, C, H)
        g = self.col_gate(x_col)
        v = self.col_val(x_col)
        out = g * v + (1 - g) * x_col
        out = out.reshape(B, W, C, H).permute(0, 2, 3, 1)
        return out

    def forward(self, x):

        row_out    = self._row_scan(x)
        col_out    = self._col_scan(x)
        global_out = x * self.global_ctx(x)


        weights = self.fusion_gate(x)
        w_r = weights[:, 0:1]
        w_c = weights[:, 1:2]
        w_g = weights[:, 2:3]

        fused = w_r * row_out + w_c * col_out + w_g * global_out

        return self.out_proj(fused)


class HSMSSD2D(nn.Module):
    def __init__(self, d_model, ssd_expand=1.0, state_dim=16):
        super().__init__()
        self.d_model   = d_model
        self.state_dim = state_dim


        ws = max(3, int(math.isqrt(state_dim)))
        if ws % 2 == 0:
            ws += 1


        self.channel_mix = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, 1, bias=False),
            nn.BatchNorm2d(d_model * 2),
            nn.SiLU(),
            nn.Conv2d(d_model * 2, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model)
        )


        self.spatial_scan = EfficientScan2D(dim=d_model, window_size=ws)


        self.gate = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.Sigmoid()
        )


        self.out_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model)
        )

    def forward(self, x):
        identity = x

        channel_out = self.channel_mix(x)
        spatial_out = self.spatial_scan(x)

        gate   = self.gate(x)
        fused  = channel_out * gate + spatial_out * (1 - gate)

        out = self.out_proj(fused)
        y   = out + identity

        h = x.mean(dim=(2, 3))
        return y, h


class DWConv2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = ConvLayer2D(dim, dim, kernel_size=3, padding=1,
                                  groups=dim, bn_weight_init=0, act_layer=None)

    def forward(self, x):
        return self.dwconv(x)


class EfficientViMBlock2D(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, ssd_expand=1.0, state_dim=16):
        super().__init__()
        self.dim = dim

        self.mixer  = HSMSSD2D(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm   = LayerNorm1D(dim)

        self.dwconv1 = DWConv2D(dim)
        self.dwconv2 = DWConv2D(dim)

        self.ffn = FFN2D(in_dim=dim, dim=int(dim * mlp_ratio))


        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        x_prev = x
        x, h   = self.mixer(x)
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)

        return x, h


class PatchMerging2D(nn.Module):
    def __init__(self, in_dim, out_dim, stride=2, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)

        self.conv = nn.Sequential(
            ConvLayer2D(in_dim,     hidden_dim, kernel_size=1, padding=0),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3,
                        stride=stride, padding=1, groups=hidden_dim),
            ConvLayer2D(hidden_dim, out_dim,    kernel_size=1, padding=0, act_layer=None)
        )

        self.dwconv1 = ConvLayer2D(in_dim,  in_dim,  kernel_size=3, padding=1,
                                    groups=in_dim,  act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, kernel_size=3, padding=1,
                                    groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class BasicBlock2D(nn.Module):
    def __init__(self, in_dim, out_dim, depth=1, state_dim=16, stride=2):
        super().__init__()
        self.depth = depth

        self.blocks = nn.ModuleList([
            EfficientViMBlock2D(dim=in_dim, mlp_ratio=2.0,
                                ssd_expand=1.0, state_dim=state_dim)
            for _ in range(depth)
        ])

        self.downsample  = PatchMerging2D(in_dim=in_dim, out_dim=out_dim, stride=stride)
        self.out_channels = out_dim

    def forward(self, x):
        h = None
        for blk in self.blocks:
            x, h = blk(x)
        feats = x
        out   = self.downsample(x)
        return out, feats, h


class MambaFeatureExtractor2D(nn.Module):

    def __init__(self, in_dim=1,
                 embed_dim=[32, 64, 128, 256],
                 depths=[1, 1, 1, 1],
                 state_dim=[49, 25, 9, 9],
                 strides=[2, 2, 2, 1]):
        super().__init__()
        self.num_layers = len(depths)


        self.stem = nn.Sequential(
            ConvLayer2D(in_dim,            embed_dim[0] // 2, kernel_size=3, stride=1, padding=1),
            ConvLayer2D(embed_dim[0] // 2, embed_dim[0],      kernel_size=3, stride=2, padding=1,
                        act_layer=None)
        )

        self.stages = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch  = embed_dim[i]
            out_ch = embed_dim[i + 1] if i < self.num_layers - 1 else embed_dim[i]
            stride = strides[i] if isinstance(strides, list) else strides
            sd     = state_dim[i] if i < len(state_dim) else state_dim[-1]

            self.stages.append(
                BasicBlock2D(in_dim=in_ch, out_dim=out_ch,
                             depth=depths[i], state_dim=sd, stride=stride)
            )

        self.out_channels = embed_dim

    def forward(self, x):
        x = self.stem(x)

        features = []
        for stage in self.stages:
            x, feats, _ = stage(x)
            features.append(feats)

        return features
