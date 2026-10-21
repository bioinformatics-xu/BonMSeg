import torch
import torch.nn as nn
import torch.nn.functional as F


class StableBoneMaskProcessor(nn.Module):

    def __init__(self, dilation_kernel=5, min_area_ratio=0.005):
        super().__init__()
        self.dilation_kernel = dilation_kernel
        self.min_area_ratio = min_area_ratio

    def dilate_mask(self, mask, kernel_size=3):
        if kernel_size <= 1:
            return mask
        padding = kernel_size // 2
        return F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

    def forward(self, bone_logits, pet):

        bone_prob = torch.sigmoid(bone_logits)
        bone_pred = (bone_prob > 0.4).float()


        bone_pred_dilated = self.dilate_mask(bone_pred, self.dilation_kernel)


        B = bone_pred.shape[0]
        total_pixels = bone_pred.shape[2] * bone_pred.shape[3]
        mask_areas = bone_pred_dilated.view(B, -1).sum(dim=1)
        mask_valid = (mask_areas / total_pixels) > self.min_area_ratio


        for b in range(B):
            if not mask_valid[b]:
                bone_pred_dilated[b] = 1.0


        pet_bone = pet * bone_pred_dilated

        return pet_bone, bone_pred, bone_pred_dilated, mask_valid
