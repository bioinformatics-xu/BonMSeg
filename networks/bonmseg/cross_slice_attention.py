import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Tuple,
)


class DWConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class CrossSliceAttention(nn.Module):

    def __init__(
        self,
        feat_ch: int,
        n_heads: int = 4,
        pool_size: int = 16,
        use_self_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feat_ch      = feat_ch
        self.n_heads      = n_heads
        self.pool_size    = pool_size
        self.head_dim     = max(feat_ch // n_heads, 1)
        self.use_self_attn = use_self_attn
        self.scale        = self.head_dim ** -0.5

        d = self.head_dim * n_heads


        self.q_proj = nn.Linear(feat_ch, d, bias=False)
        self.k_proj = nn.Linear(feat_ch, d, bias=False)
        self.v_proj = nn.Linear(feat_ch, d, bias=False)
        self.cross_out = nn.Sequential(
            nn.Linear(d, feat_ch, bias=False),
            nn.LayerNorm(feat_ch),
        )


        if use_self_attn:
            self.sq_proj   = nn.Linear(feat_ch, d, bias=False)
            self.sk_proj   = nn.Linear(feat_ch, d, bias=False)
            self.sv_proj   = nn.Linear(feat_ch, d, bias=False)
            self.self_out  = nn.Sequential(
                nn.Linear(d, feat_ch, bias=False),
                nn.LayerNorm(feat_ch),
            )


        in_gate = feat_ch * (3 if use_self_attn else 2)
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, 3 if use_self_attn else 2),
            nn.Softmax(dim=-1),
        )

        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, 3 if use_self_attn else 2, bias=True),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


        self.refine = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, groups=feat_ch, bias=False),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )

    def _pool_and_flatten(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        P = self.pool_size
        if H != P or W != P:
            feat = F.adaptive_avg_pool2d(feat, (P, P))
        return feat.flatten(2).permute(0, 2, 1)


    def _multihead_attn(self, q, k, v):
        B, Lq, d = q.shape
        Lk = k.shape[1]
        H = self.n_heads
        Dh = self.head_dim

        q = q.view(B, Lq, H, Dh).transpose(1, 2)
        k = k.view(B, Lk, H, Dh).transpose(1, 2)
        v = v.view(B, Lk, H, Dh).transpose(1, 2)


        dropout_p = self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                             dropout_p=dropout_p,
                                             scale=self.scale)

        out = out.transpose(1, 2).contiguous().view(B, Lq, H * Dh)
        return out

    def forward(
        self,
        center_feat: torch.Tensor,
        neighbor_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        B, C, H, W = center_feat.shape
        P = self.pool_size


        q_flat = self._pool_and_flatten(center_feat)


        nb_flats = [self._pool_and_flatten(nf) for nf in neighbor_feats]
        kv_flat  = torch.cat(nb_flats, dim=1)


        Q = self.q_proj(q_flat)
        K = self.k_proj(kv_flat)
        V = self.v_proj(kv_flat)

        cross_raw    = self._multihead_attn(Q, K, V)
        cross_pooled = self.cross_out(cross_raw)

        cross_spatial = (cross_pooled
                         .permute(0, 2, 1)
                         .view(B, C, P, P))
        if H != P or W != P:
            cross_spatial = F.interpolate(
                cross_spatial, size=(H, W), mode='bilinear', align_corners=True)


        if self.use_self_attn:
            SQ = self.sq_proj(q_flat)
            SK = self.sk_proj(q_flat)
            SV = self.sv_proj(q_flat)
            self_raw     = self._multihead_attn(SQ, SK, SV)
            self_pooled  = self.self_out(self_raw)
            self_spatial = (self_pooled
                            .permute(0, 2, 1)
                            .view(B, C, P, P))
            if H != P or W != P:
                self_spatial = F.interpolate(
                    self_spatial, size=(H, W), mode='bilinear', align_corners=True)


        gate_weights = self.fusion_gate(center_feat)
        if self.use_self_attn:
            w_identity = gate_weights[:, 0].view(B, 1, 1, 1)
            w_cross    = gate_weights[:, 1].view(B, 1, 1, 1)
            w_self     = gate_weights[:, 2].view(B, 1, 1, 1)
            fused = (w_identity * center_feat
                     + w_cross  * cross_spatial
                     + w_self   * self_spatial)
        else:
            w_identity = gate_weights[:, 0].view(B, 1, 1, 1)
            w_cross    = gate_weights[:, 1].view(B, 1, 1, 1)
            fused = w_identity * center_feat + w_cross * cross_spatial

        return self.refine(fused)


class CSASliceFusion(nn.Module):

    def __init__(
        self,
        n_slices:         int   = 5,
        feat_ch:          int   = 32,
        n_heads:          int   = 4,
        pool_size:        int   = 16,
        use_self_attn:    bool  = True,
        use_cross_modal:  bool  = True,
        dropout:          float = 0.0,
    ):
        super().__init__()
        assert n_slices % 2 == 1, "n_slices must be odd (3 or 5)"
        self.n_slices      = n_slices
        self.center_idx    = n_slices // 2
        self.n_neighbors   = n_slices - 1
        self.use_cross_modal = use_cross_modal


        self.ct_slice_encoder  = nn.Sequential(
            DWConvBNReLU(1, feat_ch // 2, 3),
            DWConvBNReLU(feat_ch // 2, feat_ch, 3),
        )
        self.pet_slice_encoder = nn.Sequential(
            DWConvBNReLU(1, feat_ch // 2, 3),
            DWConvBNReLU(feat_ch // 2, feat_ch, 3),
        )


        self.ct_csa  = CrossSliceAttention(feat_ch, n_heads, pool_size,
                                           use_self_attn, dropout)
        self.pet_csa = CrossSliceAttention(feat_ch, n_heads, pool_size,
                                           use_self_attn, dropout)


        if use_cross_modal:

            self.pet2ct_proj = nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
                nn.BatchNorm2d(feat_ch),
                nn.ReLU(inplace=True),
            )
            self.cross_modal_csa = CrossSliceAttention(
                feat_ch, n_heads, pool_size, use_self_attn=False, dropout=dropout)


        self.ct_out_proj = nn.Sequential(
            nn.Conv2d(feat_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.pet_out_proj = nn.Sequential(
            nn.Conv2d(feat_ch, 1, 1, bias=True),
            nn.InstanceNorm2d(1, affine=True),
            nn.ReLU(inplace=True),
        )


        nn.init.constant_(self.ct_out_proj[0].bias,  0.1)
        nn.init.constant_(self.pet_out_proj[0].bias, 0.1)


    def _encode_slices(self, x5, encoder):
        B, N, H, W = x5.shape

        x_merged = x5.reshape(B * N, 1, H, W)
        f_merged = encoder(x_merged)
        C = f_merged.shape[1]
        f_split = f_merged.view(B, N, C, H, W)
        return [f_split[:, i] for i in range(N)]

    def forward(
        self,
        ct5:  torch.Tensor,
        pet5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.center_idx


        ct_feats  = self._encode_slices(ct5,  self.ct_slice_encoder)
        pet_feats = self._encode_slices(pet5, self.pet_slice_encoder)


        ct_center  = ct_feats[c]
        pet_center = pet_feats[c]
        ct_neighbors  = [ct_feats[i]  for i in range(self.n_slices) if i != c]
        pet_neighbors = [pet_feats[i] for i in range(self.n_slices) if i != c]


        ct_enriched  = self.ct_csa( ct_center,  ct_neighbors)
        pet_enriched = self.pet_csa(pet_center, pet_neighbors)


        if self.use_cross_modal:
            pet_nb_projected = [self.pet2ct_proj(nf) for nf in pet_neighbors]
            ct_cross_modal   = self.cross_modal_csa(ct_enriched, pet_nb_projected)

            ct_enriched = ct_enriched + 0.5 * ct_cross_modal


        ct_1ch  = self.ct_out_proj(ct_enriched)
        pet_1ch = self.pet_out_proj(pet_enriched)

        return ct_1ch, pet_1ch
