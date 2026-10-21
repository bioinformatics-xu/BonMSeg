import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


_NORM_GRID_CACHE: dict = {}

def _make_norm_grid(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (H, W, str(device))
    if key not in _NORM_GRID_CACHE:
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        _NORM_GRID_CACHE[key] = (grid_x, grid_y)
    return _NORM_GRID_CACHE[key]


def generate_rotated_gaussian(
    H: int, W: int,
    mu_x: torch.Tensor,
    mu_y: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-4
) -> torch.Tensor:
    grid_x, grid_y = _make_norm_grid(H, W, mu_x.device)
    gx = grid_x.unsqueeze(0).unsqueeze(0)
    gy = grid_y.unsqueeze(0).unsqueeze(0)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    dx    = gx - mu_x
    dy    = gy - mu_y

    x_rot =  cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy


    sigma_x = sigma_x.clamp_min(0.10)
    sigma_y = sigma_y.clamp_min(0.10)
    exponent = -(x_rot ** 2 / (2.0 * sigma_x ** 2 + eps)
               + y_rot ** 2 / (2.0 * sigma_y ** 2 + eps))

    return torch.exp(exponent.clamp(min=-80.0))


class DGMAHeatmapPredictor(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 2, 16)
        self.net = nn.Sequential(

            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
            nn.Sigmoid()
        )

        nn.init.constant_(self.net[-2].bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LesionRadiusPredictor(nn.Module):

    def __init__(self, channels: int,
                 r_min: float = 0.03,
                 r_max: float = 0.40):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max

        self.net = nn.Sequential(

            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, max(channels // 4, 8), 1, bias=False),
            nn.BatchNorm2d(max(channels // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), 1, 1, bias=True),
            nn.Sigmoid()
        )


        nn.init.constant_(self.net[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig = self.net(x)
        return sig * (self.r_max - self.r_min) + self.r_min


class DGMANMSCenterExtractor(nn.Module):

    def __init__(self, K_max: int = 5, nms_kernel: int = 3, nms_threshold: float = 0.3):
        super().__init__()
        self.K_max         = K_max
        self.nms_threshold = nms_threshold

        pad = nms_kernel // 2
        self.nms_pool = nn.MaxPool2d(nms_kernel, stride=1, padding=pad)

    def forward(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = heatmap.shape
        device = heatmap.device


        hmap_squeeze = heatmap.squeeze(1)
        local_max    = (self.nms_pool(heatmap).squeeze(1) == hmap_squeeze)

        peaks        = hmap_squeeze * local_max.float()


        peaks_flat   = peaks.view(B, -1)
        K_cand       = min(self.K_max, peaks_flat.shape[1])
        top_vals, top_idx = torch.topk(peaks_flat, K_cand, dim=1)


        if K_cand < self.K_max:
            pad_vals = torch.zeros(B, self.K_max - K_cand, device=device)
            pad_idx  = torch.zeros(B, self.K_max - K_cand, dtype=torch.long, device=device)
            top_vals = torch.cat([top_vals, pad_vals], dim=1)
            top_idx  = torch.cat([top_idx,  pad_idx],  dim=1)


        valid_mask = top_vals >= self.nms_threshold


        row_idx = (top_idx // W).float()
        col_idx = (top_idx  % W).float()


        norm_y = (2.0 * row_idx / max(H - 1, 1)) - 1.0
        norm_x = (2.0 * col_idx / max(W - 1, 1)) - 1.0

        centers = torch.stack([norm_x, norm_y], dim=-1)


        centers = centers * valid_mask.unsqueeze(-1).float()

        return centers, valid_mask, top_vals


class DGMAParamPredictor(nn.Module):

    def __init__(self, channels: int,
                 sigma_min: float = 0.01,
                 sigma_max: float = 0.85,
                 prior_aspect_ratio: float = 1.5,
                 sigma_residual_max: float = 0.08):
        super().__init__()
        self.sigma_min         = sigma_min
        self.sigma_max         = sigma_max
        self.prior_beta        = prior_aspect_ratio
        self.sigma_residual_max = sigma_residual_max


        self._log_alpha = nn.Parameter(torch.zeros(1))


        self.param_mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 4)
        )


        nn.init.zeros_(self.param_mlp[-1].bias)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._log_alpha)

    def forward(
        self,
        feature_map: torch.Tensor,
        radius_map:  torch.Tensor,
        centers:     torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = feature_map.shape
        K = centers.shape[1]


        sample_grid = centers.unsqueeze(1)


        feat_sampled = F.grid_sample(
            feature_map, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(2).permute(0, 2, 1)


        r_k = F.grid_sample(
            radius_map, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(1).squeeze(1)


        alpha = self.alpha
        sigma_x_base = alpha * r_k
        sigma_y_base = alpha * r_k * self.prior_beta


        params = self.param_mlp(feat_sampled)


        delta_sigma_x = torch.tanh(params[..., 0]) * self.sigma_residual_max
        delta_sigma_y = torch.tanh(params[..., 1]) * self.sigma_residual_max
        theta         = torch.tanh(params[..., 2]) * math.pi
        weight        = torch.sigmoid(params[..., 3])


        sigma_x = (sigma_x_base + delta_sigma_x).clamp(self.sigma_min, self.sigma_max)
        sigma_y = (sigma_y_base + delta_sigma_y).clamp(self.sigma_min, self.sigma_max)


        def _e(t): return t.unsqueeze(-1).unsqueeze(-1)
        return _e(sigma_x), _e(sigma_y), _e(theta), _e(weight), r_k


def build_gaussian_mixture_attention(
    H: int, W: int,
    centers: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    theta: torch.Tensor,
    weight: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mu_x = centers[..., 0].unsqueeze(-1).unsqueeze(-1)
    mu_y = centers[..., 1].unsqueeze(-1).unsqueeze(-1)


    gaussians = generate_rotated_gaussian(H, W, mu_x, mu_y, sigma_x, sigma_y, theta)


    mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1).float()


    masked_weight = weight * mask_4d
    weight_sum    = masked_weight.sum(dim=1, keepdim=True).clamp(min=1e-3)
    norm_weight   = masked_weight / weight_sum


    mixture = (gaussians * norm_weight * mask_4d).sum(dim=1, keepdim=True)


    return torch.sigmoid(mixture * 4.0 - 2.0)


class DGMA(nn.Module):

    def __init__(
        self,
        channels: int,
        K_max: int             = 5,
        nms_kernel: int        = 3,
        nms_threshold: float   = 0.3,
        r_min: float           = 0.03,
        r_max: float           = 0.40,
        sigma_min: float       = 0.01,
        sigma_max: float       = 0.85,
        prior_aspect_ratio: float = 1.5,
        sigma_residual_max: float = 0.08,
        residual_alpha: float  = 0.1,


    ):
        super().__init__()
        self.K_max          = K_max
        self.nms_threshold  = nms_threshold
        self.residual_alpha = residual_alpha
        self.sigma_min      = sigma_min
        self.sigma_max      = sigma_max


        self.heatmap_predictor = DGMAHeatmapPredictor(channels)

        self.radius_predictor  = LesionRadiusPredictor(channels, r_min, r_max)

        self.nms_extractor     = DGMANMSCenterExtractor(K_max, nms_kernel, nms_threshold)

        self.param_predictor   = DGMAParamPredictor(
            channels, sigma_min, sigma_max, prior_aspect_ratio, sigma_residual_max)


        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.residual_gate = nn.Parameter(torch.tensor(residual_alpha))


        self.fallback_log_scale = nn.Parameter(torch.tensor(1.0))


        self._radius_map: Optional[torch.Tensor] = None

        self.last_state: Optional[Dict] = None

    def forward(
        self, feature_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = feature_map.shape
        device     = feature_map.device


        center_heatmap = self.heatmap_predictor(feature_map)


        radius_map = self.radius_predictor(feature_map)


        centers, valid_mask, peak_scores = self.nms_extractor(center_heatmap)
        k_dynamic     = valid_mask.sum(dim=1)
        used_fallback = (k_dynamic == 0)


        if used_fallback.any():
            valid_mask_aug = valid_mask.clone()
            valid_mask_aug[used_fallback, 0] = True
        else:
            valid_mask_aug = valid_mask


        sigma_x, sigma_y, theta, weight, r_k = self.param_predictor(
            feature_map, radius_map, centers)


        if used_fallback.any():
            fb_idx = used_fallback.nonzero(as_tuple=True)[0]


            center_grid = torch.zeros(len(fb_idx), 1, 1, 2, device=device)
            r_center = F.grid_sample(
                radius_map[fb_idx], center_grid,
                mode='bilinear', align_corners=True, padding_mode='border'
            ).squeeze(-1).squeeze(-1).squeeze(-1)


            fb_scale   = F.softplus(self.fallback_log_scale)
            sigma_fb_x = (fb_scale * r_center).clamp(self.sigma_min, self.sigma_max)
            sigma_fb_y = (fb_scale * r_center * self.param_predictor.prior_beta
                          ).clamp(self.sigma_min, self.sigma_max)


            sigma_x = sigma_x.clone()
            sigma_y = sigma_y.clone()
            sigma_x[fb_idx, 0, 0, 0] = sigma_fb_x
            sigma_y[fb_idx, 0, 0, 0] = sigma_fb_y


        attention_map = build_gaussian_mixture_attention(
            H, W, centers, sigma_x, sigma_y, theta, weight, valid_mask_aug
        )


        channel_w = self.channel_scale(feature_map)
        attended  = feature_map * attention_map * channel_w
        alpha_r   = torch.sigmoid(self.residual_gate)
        enhanced  = attended + alpha_r * feature_map


        self._radius_map = radius_map


        self.last_state = {
            'center_heatmap':  center_heatmap.detach(),
            'radius_map':      radius_map.detach(),
            'attention_map':   attention_map.detach(),
            'centers':         centers.detach(),
            'valid_mask':      valid_mask.detach(),
            'peak_scores':     peak_scores.detach(),
            'k_dynamic':       k_dynamic.detach(),
            'used_fallback':   used_fallback.detach(),
            'r_k':             r_k.detach(),
            'sigma_x':         sigma_x.detach(),
            'sigma_y':         sigma_y.detach(),
            'theta':           theta.detach(),
            'weight':          weight.detach(),
            'alpha':           self.param_predictor.alpha.item(),

            'fallback_scale':  F.softplus(self.fallback_log_scale).item(),
        }

        return enhanced, attention_map


class DGMAWithSOE(nn.Module):

    def __init__(
        self,
        channels: int,
        K_max: int             = 5,
        nms_threshold: float   = 0.3,
        r_min: float           = 0.03,
        r_max: float           = 0.40,
        sigma_min: float       = 0.01,
        sigma_max: float       = 0.85,
        prior_aspect_ratio: float = 1.5,
        sigma_residual_max: float = 0.08,
        min_spatial_size: int  = 65,
    ):
        super().__init__()
        self.min_spatial_size = min_spatial_size


        self.soe_pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.soe_pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.soe_pool7 = nn.MaxPool2d(7, stride=1, padding=3)
        self.soe_attn  = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

        self.dgma = DGMA(
            channels           = channels,
            K_max              = K_max,
            nms_threshold      = nms_threshold,
            r_min              = r_min,
            r_max              = r_max,
            sigma_min          = sigma_min,
            sigma_max          = sigma_max,
            prior_aspect_ratio = prior_aspect_ratio,
            sigma_residual_max = sigma_residual_max,
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, pet_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pet_feat.shape


        r3 = pet_feat - self.soe_pool3(pet_feat)
        r5 = pet_feat - self.soe_pool5(pet_feat)
        r7 = pet_feat - self.soe_pool7(pet_feat)
        soe_enhanced = pet_feat * (1.0 + self.soe_attn(torch.cat([r3, r5, r7], dim=1)))


        if min(H, W) < self.min_spatial_size:


            self.dgma.last_state  = None
            self.dgma._radius_map = None
            return soe_enhanced


        dgma_enhanced, _ = self.dgma(soe_enhanced)
        gate = torch.sigmoid(self.fusion_gate)
        return gate * dgma_enhanced + (1.0 - gate) * soe_enhanced

    @property
    def last_state(self):
        return self.dgma.last_state


class DGMASupervisionLoss(nn.Module):

    def __init__(
        self,
        heatmap_weight:  float = 0.2,
        coverage_weight: float = 0.2,
        radius_weight:   float = 0.0,
    ):
        super().__init__()
        self.heatmap_weight  = heatmap_weight
        self.coverage_weight = coverage_weight
        self.radius_weight   = radius_weight

    def forward(
        self,
        state:       Optional[Dict],
        tumor_mask:  torch.Tensor,
        bone_mask:   torch.Tensor,
    ) -> torch.Tensor:
        if state is None:
            return torch.tensor(0.0, device=tumor_mask.device)

        total     = torch.tensor(0.0, device=tumor_mask.device)
        has_tumor = (tumor_mask.sum(dim=(1, 2, 3)) > 0)


        if has_tumor.any() and self.heatmap_weight > 0:
            hmap = state['center_heatmap']
            if hmap.shape[2:] != tumor_mask.shape[2:]:
                hmap = F.interpolate(hmap, size=tumor_mask.shape[2:],
                                     mode='bilinear', align_corners=True)
            B_t  = has_tumor.nonzero(as_tuple=True)[0]
            hm_t = hmap[B_t];  tm_t = tumor_mask[B_t];  bm_t = bone_mask[B_t]
            non_tumor_bone = bm_t * (1.0 - tm_t)
            attn_on_tumor = (hm_t * tm_t).sum() / (tm_t.sum() + 1e-6)
            attn_on_bg    = (hm_t * non_tumor_bone).sum() / (non_tumor_bone.sum() + 1e-6)
            total = total + self.heatmap_weight * F.relu(attn_on_bg - attn_on_tumor)


        if has_tumor.any() and self.coverage_weight > 0:
            attn = state['attention_map']
            if attn.shape[2:] != tumor_mask.shape[2:]:
                attn = F.interpolate(attn, size=tumor_mask.shape[2:],
                                     mode='bilinear', align_corners=True)
            B_t    = has_tumor.nonzero(as_tuple=True)[0]
            att_t  = attn[B_t];  tm_t = tumor_mask[B_t];  bm_t = bone_mask[B_t]
            attn_tumor = (att_t * tm_t).sum()  / (tm_t.sum()  + 1e-6)
            attn_bone  = (att_t * bm_t).sum()  / (bm_t.sum()  + 1e-6)
            total = total + self.coverage_weight * F.relu(attn_bone - attn_tumor * 2.0)


        if has_tumor.any() and self.radius_weight > 0 and 'r_k' in state:

            r_k        = state['r_k']
            valid_mask = state['valid_mask']
            B_t = has_tumor.nonzero(as_tuple=True)[0]

            r_k_t   = r_k[B_t]
            vm_t    = valid_mask[B_t]

            row_valid = vm_t.any(dim=1)
            if row_valid.any():
                r_k_v  = r_k_t[row_valid]
                vm_v   = vm_t[row_valid]

                tm_v   = tumor_mask[B_t[row_valid]]
                bm_v   = bone_mask[B_t[row_valid]]
                tumor_area = tm_v.sum(dim=(1,2,3)).float()
                bone_area  = bm_v.sum(dim=(1,2,3)).float().clamp(1)
                r_gt   = (tumor_area / bone_area / math.pi).clamp(min=1e-6).sqrt()

                r_sum  = (r_k_v * vm_v.float()).sum(dim=1)
                r_cnt  = vm_v.float().sum(dim=1).clamp(min=1)
                r_pred_mean = r_sum / r_cnt

                total = total + self.radius_weight * F.mse_loss(
                    r_pred_mean, r_gt.detach())

        return total
