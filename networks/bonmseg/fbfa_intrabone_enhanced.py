import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size,
                            padding=padding, groups=in_channels, bias=bias)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out = self.mlp(F.adaptive_max_pool2d(x, 1).flatten(1))
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1).flatten(1))

        w = self.sigmoid(max_out + avg_out).view(x.size(0), x.size(1), 1, 1)
        return x * w


class EnhancedFrequencyDecomposition(nn.Module):

    def __init__(self, channels):
        super().__init__()


        self.gauss_small = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )

        self.gauss_large = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2,
            groups=channels, bias=False
        )

        self._init_gaussian_kernels()


        self.refine_low = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.refine_high = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def _init_gaussian_kernels(self):

        sigmas = {3: 1.0, 5: 2.0}

        for ks, conv in [(3, self.gauss_small), (5, self.gauss_large)]:

            sigma = sigmas[ks]

            coords = torch.arange(ks).float() - ks // 2
            g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g1d = g1d / g1d.sum()

            g2d = g1d.view(-1, 1) * g1d.view(1, -1)

            with torch.no_grad():
                for i in range(conv.weight.size(0)):
                    conv.weight[i, 0] = g2d

            conv.weight.requires_grad_(False)

    def forward(self, x):

        g_small = self.gauss_small(x)
        g_large = self.gauss_large(x)


        low_freq  = g_large
        high_freq = g_small - g_large


        low_freq  = self.refine_low(low_freq)
        high_freq = self.refine_high(high_freq)

        return low_freq, high_freq


class IntraBoneForegroundPredictor(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            CBAM(channels // 2, reduction=8),
            nn.Conv2d(channels // 2, 1, 1)
        )

    def forward(self, ct_feat, pet_feat):

        joint = torch.cat([ct_feat, pet_feat], dim=1)
        logit = self.predictor(joint)


        mask = torch.sigmoid(logit * 2.0)
        return mask


class BranchGate(nn.Module):

    def __init__(self, channels, num_branches=4):
        super().__init__()
        self.num_branches = num_branches

        self.gate_fc = nn.Sequential(
            nn.Linear(channels * num_branches, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_branches, bias=True)
        )

    def forward(self, branches: list):

        B = branches[0].size(0)

        pooled = torch.cat(
            [f.mean(dim=[2, 3]) for f in branches], dim=1
        )

        weights = torch.softmax(self.gate_fc(pooled), dim=1)


        out = sum(
            weights[:, i].view(B, 1, 1, 1) * branches[i]
            for i in range(self.num_branches)
        )
        return out


class FBFAFusionSingleScaleIntraBone(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels


        self.ct_freq_decomp  = EnhancedFrequencyDecomposition(channels)
        self.pet_freq_decomp = EnhancedFrequencyDecomposition(channels)


        self.fg_predictor = IntraBoneForegroundPredictor(channels)


        self.fusion_low_fg = nn.Sequential(
            DepthwiseSeparableConv(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )
        self.fusion_low_bg = nn.Sequential(
            DepthwiseSeparableConv(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )

        self.fusion_high_fg = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )
        self.fusion_high_bg = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )


        self.branch_gate = BranchGate(channels, num_branches=4)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )


        self.alpha = nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, ct_feat, pet_feat):

        CT_low,  CT_high  = self.ct_freq_decomp(ct_feat)
        PET_low, PET_high = self.pet_freq_decomp(pet_feat)


        M     = self.fg_predictor(ct_feat, pet_feat)
        M_inv = (1.0 - M).detach()


        CT_low_fg,  PET_low_fg  = CT_low  * M,     PET_low  * M
        CT_low_bg,  PET_low_bg  = CT_low  * M_inv, PET_low  * M_inv
        CT_high_fg, PET_high_fg = CT_high * M,     PET_high * M
        CT_high_bg, PET_high_bg = CT_high * M_inv, PET_high * M_inv


        fused_low_fg  = self.fusion_low_fg (torch.cat([CT_low_fg,  PET_low_fg],  dim=1))
        fused_low_bg  = self.fusion_low_bg (torch.cat([CT_low_bg,  PET_low_bg],  dim=1))
        fused_high_fg = self.fusion_high_fg(torch.cat([CT_high_fg, PET_high_fg], dim=1))
        fused_high_bg = self.fusion_high_bg(torch.cat([CT_high_bg, PET_high_bg], dim=1))


        gated  = self.branch_gate([fused_low_fg, fused_low_bg, fused_high_fg, fused_high_bg])
        fused  = self.reconstruction(gated)


        alpha  = torch.sigmoid(self.alpha)
        base = 0.5 * (ct_feat + pet_feat)
        output = alpha * fused + (1 - alpha) * base

        return output


class FBFAFusionMultiScaleIntraBone(nn.Module):

    def __init__(self, channels_list=[32, 64, 128, 256], reduction=16):
        super().__init__()
        self.num_scales = len(channels_list)

        self.fusions = nn.ModuleList([
            FBFAFusionSingleScaleIntraBone(c, reduction)
            for c in channels_list
        ])

    def forward(self, ct_features, pet_features):
        fused_features = []

        for i in range(self.num_scales):
            ct_f = ct_features[i]
            pet_f = pet_features[i]

            if ct_f.shape[2:] != pet_f.shape[2:]:
                pet_f = F.interpolate(pet_f, size=ct_f.shape[2:], mode='bilinear', align_corners=False)

            fused = self.fusions[i](ct_f, pet_f)
            fused_features.append(fused)

        return fused_features


class MultiScaleDecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, scale=2, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear',
                                    align_corners=True) if scale > 1 else nn.Identity()


        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels, reduction=8),

            DepthwiseSeparableConv(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


        if deep_supervision:

            self.ds_head = nn.Conv2d(out_channels, 1, kernel_size=1)
        else:
            self.ds_head = None

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        ds_output = None
        if self.ds_head is not None:
            ds_output = self.ds_head(x)

        return x, ds_output
