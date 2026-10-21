import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from networks.bonmseg.fbfa_intrabone_enhanced_iddmga import (
    DGMAWithSOE,
)
from networks.bonmseg.cross_slice_attention import (
    CSASliceFusion,
)


from networks.bonmseg.fbfa_intrabone_enhanced import (
    CBAM,
    FBFAFusionMultiScaleIntraBone,
    MultiScaleDecoderBlock,
)


class FBFAIntraBoneTumorSegmentation(nn.Module):

    def __init__(
        self,
        stage1_model_path: str           = None,
        freeze_stage1:     bool          = True,
        bone_dilation:     int           = 7,
        channels_list:     List[int]     = [32, 64, 128, 256],
        enable_deep_supervision: bool    = True,

        n_slices:          int           = 3,
        csa_feat_ch:       int           = 32,
        csa_n_heads:       int           = 4,
        csa_pool_size:     int           = 32,
        csa_use_cross_modal: bool        = True,


        dgma_K_max:           int        = 15,
        dgma_nms_threshold:   float      = 0.15,
        dgma_r_min:           float      = 0.005,
        dgma_r_max:           float      = 0.25,
        dgma_min_spatial_size: int       = 65,
    ):
        super().__init__()

        assert n_slices % 2 == 1, "n_slices must be odd"
        self.n_slices              = n_slices
        self.center_idx            = n_slices // 2
        self.freeze_stage1         = freeze_stage1
        self.bone_dilation         = bone_dilation
        self.enable_deep_supervision = enable_deep_supervision


        self.csa_fusion = CSASliceFusion(
            n_slices         = n_slices,
            feat_ch          = csa_feat_ch,
            n_heads          = csa_n_heads,
            pool_size        = csa_pool_size,
            use_self_attn    = True,
            use_cross_modal  = csa_use_cross_modal,
        )


        from networks.base.model_efficientvim_2d_stage1 import ConDSeg2DStage1_EfficientViM
        self.ct_branch = ConDSeg2DStage1_EfficientViM(in_channels=1, out_channels=1)

        if stage1_model_path:
            checkpoint = torch.load(stage1_model_path, map_location='cpu',
                                    weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model_dict = self.ct_branch.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.ct_branch.load_state_dict(model_dict)
            print(f"[BonMSeg] Loaded {len(pretrained_dict)} CT branch weights")

        if freeze_stage1:
            for param in self.ct_branch.parameters():
                param.requires_grad = False
            self.ct_branch.eval()


        from networks.base.model_stage2_tumor_2d import StableBoneMaskProcessor
        self.bone_processor = StableBoneMaskProcessor(
            dilation_kernel=bone_dilation, min_area_ratio=0.001)


        from networks.base.efficientvim_modules_2d import MambaFeatureExtractor2D
        self.pet_backbone = MambaFeatureExtractor2D(
            in_dim      = 1,
            embed_dim   = channels_list,
            depths      = [2, 2, 2, 2],
            state_dim   = [16, 25, 9, 9],
            strides     = [2, 2, 2, 1],
        )


        self.soe = nn.ModuleList([
            DGMAWithSOE(
                channels           = c,
                K_max              = dgma_K_max,
                nms_threshold      = dgma_nms_threshold,
                r_min              = dgma_r_min,
                r_max              = dgma_r_max,
                min_spatial_size   = dgma_min_spatial_size,
            )
            for c in channels_list
        ])


        self.fbfa_fusion = FBFAFusionMultiScaleIntraBone(
            channels_list=channels_list, reduction=16)


        self.decoder3 = MultiScaleDecoderBlock(
            256, 128, 128, scale=2, deep_supervision=enable_deep_supervision)
        self.decoder2 = MultiScaleDecoderBlock(
            128, 64,  64,  scale=2, deep_supervision=enable_deep_supervision)
        self.decoder1 = MultiScaleDecoderBlock(
            64,  32,  32,  scale=2, deep_supervision=False)


        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32, reduction=8),
            nn.Conv2d(32, 1, 1),
        )


    def forward(
        self,
        ct:          torch.Tensor,
        pet:         torch.Tensor,
        bone_mask:   Optional[torch.Tensor] = None,
        return_intermediate: bool           = False,
    ):
        B, N, H, W = ct.shape
        assert N == self.n_slices, \
            f"Expected {self.n_slices} slices, got {N}"


        ct_1ch, pet_1ch_csa = self.csa_fusion(ct, pet)


        if self.freeze_stage1:
            with torch.no_grad():
                ct_features, decoder_feats, bone_logits = \
                    self.ct_branch.forward_with_features(ct_1ch)
        else:
            ct_features, decoder_feats, bone_logits = \
                self.ct_branch.forward_with_features(ct_1ch)


        if bone_mask is None:
            pet_bone, bone_mask_pred, bone_mask_dilated, mask_valid = \
                self.bone_processor(bone_logits, pet_1ch_csa)
            bone_mask_to_use = bone_mask_dilated
        else:
            bone_mask_to_use = bone_mask
            bone_mask_pred   = bone_mask
            bone_mask_dilated = bone_mask
            pet_bone          = pet_1ch_csa
            mask_valid = (bone_mask.sum(dim=(1, 2, 3)) > 0).float()


        pet_features = self.pet_backbone(pet_bone)


        def align(src, tgt):
            if src.shape[2:] != tgt.shape[2:]:
                return F.interpolate(src, size=tgt.shape[2:],
                                     mode='bilinear', align_corners=False)
            return src

        pet_features_aligned = [
            align(pf, cf) for pf, cf in zip(pet_features, ct_features)
        ]


        pet_features_enhanced = [
            self.soe[i](pet_features_aligned[i])
            for i in range(len(pet_features_aligned))
        ]


        fused_features = self.fbfa_fusion(ct_features, pet_features_enhanced)


        d2, ds3_logits = self.decoder3(fused_features[3], fused_features[2])
        d1, ds2_logits = self.decoder2(d2, fused_features[1])
        d0, _          = self.decoder1(d1, fused_features[0])

        tumor_logits = self.output(d0)


        target_size = tumor_logits.shape[2:]
        if ds3_logits is not None and ds3_logits.shape[2:] != target_size:
            ds3_logits = F.interpolate(ds3_logits, size=target_size,
                                       mode='bilinear', align_corners=False)
        if ds2_logits is not None and ds2_logits.shape[2:] != target_size:
            ds2_logits = F.interpolate(ds2_logits, size=target_size,
                                       mode='bilinear', align_corners=False)

        if return_intermediate:
            dgma_states = [soe_block.last_state for soe_block in self.soe]
            return {
                'tumor_logits':      tumor_logits,
                'ds3_logits':        ds3_logits,
                'ds2_logits':        ds2_logits,
                'bone_mask':         bone_mask_pred,
                'bone_mask_dilated': bone_mask_dilated,
                'pet_bone':          pet_bone,
                'mask_valid':        mask_valid,
                'dgma_states':       dgma_states,
            }
        else:
            return tumor_logits
