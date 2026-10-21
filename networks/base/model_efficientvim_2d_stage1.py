import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base.efficientvim_modules_2d import MambaFeatureExtractor2D


class CBR2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1,
                 dilation=1, stride=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding,
                      dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.act  = act

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.act else x


class ChannelAttn(nn.Module):
    def __init__(self, c, ratio=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(c, c // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // ratio, c, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))
        return x * w


class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        m = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * m


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttn(c)
        self.sa = SpatialAttn()

    def forward(self, x):
        return self.sa(self.ca(x))


class MSCADecoder(nn.Module):

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        merge_c = in_c + skip_c


        self.fuse = CBR2D(merge_c, out_c, kernel_size=1, padding=0)


        self.branch1 = CBR2D(out_c, out_c, dilation=1, padding=1)
        self.branch2 = CBR2D(out_c, out_c, dilation=2, padding=2)
        self.branch4 = CBR2D(out_c, out_c, dilation=4, padding=4)
        self.branch_g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ms_fuse = CBR2D(out_c * 4, out_c, kernel_size=1, padding=0)


        self.refine1 = CBR2D(out_c, out_c, act=False)
        self.refine2 = CBR2D(out_c, out_c, act=False)
        self.relu    = nn.ReLU(inplace=True)


        self.cbam = CBAM(out_c)

    def forward(self, x, skip):

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x    = torch.cat([x, skip], dim=1)
        x    = self.fuse(x)


        b1   = self.branch1(x)
        b2   = self.branch2(x)
        b4   = self.branch4(x)
        bg   = self.branch_g(x).expand_as(x)
        ms   = self.ms_fuse(torch.cat([b1, b2, b4, bg], dim=1))


        s1   = ms
        r1   = self.relu(self.refine1(ms) + s1)
        r2   = self.relu(self.refine2(r1) + r1 + s1)


        out  = self.cbam(r2)
        return out


class AuxHead(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.head = nn.Sequential(
            CBR2D(in_c, in_c // 2, kernel_size=3, padding=1),
            nn.Conv2d(in_c // 2, 1, kernel_size=1)
        )

    def forward(self, x, target_size):
        logits = self.head(x)
        if logits.shape[2:] != target_size:
            logits = F.interpolate(logits, size=target_size,
                                   mode='bilinear', align_corners=True)
        return logits


class ConDSeg2DStage1_EfficientViM(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision


        self.backbone = MambaFeatureExtractor2D(
            in_dim=in_channels,
            embed_dim=[32, 64, 128, 256],
            depths=[1, 1, 1, 1],
            state_dim=[49, 25, 9, 9],
            strides=[2, 2, 2, 1]
        )


        self.decoder2 = MSCADecoder(in_c=256, skip_c=128, out_c=128)
        self.decoder1 = MSCADecoder(in_c=128, skip_c=64,  out_c=64)
        self.decoder0 = MSCADecoder(in_c=64,  skip_c=32,  out_c=32)


        self.main_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CBR2D(32, 64,  kernel_size=3, padding=1),
            CBR2D(64, 64,  kernel_size=3, padding=1),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )


        if deep_supervision:
            self.aux_head2 = AuxHead(in_c=128)
            self.aux_head1 = AuxHead(in_c=64)
            self.aux_head0 = AuxHead(in_c=32)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        H, W = x.shape[2], x.shape[3]


        features = self.backbone(x)
        f0, f1, f2, f3 = features


        d2 = self.decoder2(f3, f2)
        d1 = self.decoder1(d2, f1)
        d0 = self.decoder0(d1, f0)


        main_logits = self.main_head(d0)

        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(d2, target_size=(H, W))
            aux1 = self.aux_head1(d1, target_size=(H, W))
            aux0 = self.aux_head0(d0, target_size=(H, W))
            return {
                'main': main_logits,
                'aux2': aux2,
                'aux1': aux1,
                'aux0': aux0,
            }

        return main_logits


    def forward_with_features(self, x):
        features = self.backbone(x)
        f0, f1, f2, f3 = features

        d2 = self.decoder2(f3, f2)
        d1 = self.decoder1(d2, f1)
        d0 = self.decoder0(d1, f0)

        logits = self.main_head(d0)

        decoder_feats = [d0, d1, d2]

        return features, decoder_feats, logits
