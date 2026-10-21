import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred_prob, target):
        B = pred_prob.size(0)
        p = pred_prob.view(B, -1)
        t = target.view(B, -1)
        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma   = gamma

    def forward(self, pred_prob, target):
        tl = self.tversky(pred_prob, target)
        return tl ** self.gamma


class BoundaryLoss(nn.Module):

    def __init__(self, boundary_weight=5.0):
        super().__init__()
        self.bw = max(1.0, min(boundary_weight, 8.0))
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_boundary(self, mask):
        mask_f = mask.float()
        sx = self.sobel_x.to(device=mask.device, dtype=torch.float32)
        sy = self.sobel_y.to(device=mask.device, dtype=torch.float32)
        gx = F.conv2d(mask_f, sx, padding=1)
        gy = F.conv2d(mask_f, sy, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).clamp(0.0, 8.0)
        return (magnitude > 0.5).float()

    def forward(self, pred_logits, target):
        with torch.no_grad():
            boundary = self.get_boundary(target)
            weight   = (1.0 + (self.bw - 1.0) * boundary).clamp(1.0, self.bw)
            weight   = weight.to(dtype=pred_logits.dtype)
        return F.binary_cross_entropy_with_logits(
            pred_logits,
            target.to(dtype=pred_logits.dtype),
            weight=weight,
            reduction='mean')


class SmallTumorLoss(nn.Module):

    def __init__(self, ftl_weight=2.0, bce_weight=0.5, bnd_weight_max=0.03,
                 bnd_rampup_rate=0.002, ds_weight=0.3, alpha=0.3, beta=0.7,
                 gamma=1.5, boundary_weight=5.0, smooth=1.0, phase2_start=25,
                 use_boundary_delay=True, boundary_delay_start=30, ds_cutoff=40,
                 fp_weight_base=0.05):
        super().__init__()
        self.ftl_weight = ftl_weight
        self.bce_weight = bce_weight
        self.bnd_weight_max = bnd_weight_max
        self.bnd_rampup_rate = bnd_rampup_rate
        self.ds_weight = ds_weight
        self.phase2_start = phase2_start
        self.use_boundary_delay = use_boundary_delay
        self.boundary_delay_start = boundary_delay_start
        self.ds_cutoff = ds_cutoff
        self.fp_weight_base = fp_weight_base

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.bnd = BoundaryLoss(boundary_weight=boundary_weight)

        self.last_bnd_w = 0.0
        self.last_bnd_loss = 0.0
        self.last_main_loss = 0.0

    def get_boundary_weight(self, epoch: int) -> float:

        if epoch < self.phase2_start:
            return 0.0
        warmup_start = self.phase2_start
        if self.use_boundary_delay and epoch < self.boundary_delay_start:
            return 0.0
        elif self.use_boundary_delay:
            warmup_start = self.boundary_delay_start
        steps = epoch - warmup_start
        return min(self.bnd_weight_max, self.bnd_rampup_rate * steps)

    def _sync_ftl_params(self):
        self.ftl.tversky.alpha = self.alpha
        self.ftl.tversky.beta = self.beta
        self.ftl.gamma = self.gamma


    def _single_loss(self, logits, target_bone, bone_pred_t, bnd_w: float):
        self._sync_ftl_params()


        pred_prob = torch.sigmoid(logits) * bone_pred_t
        ftl_loss = self.ftl(pred_prob, target_bone)


        bce_full = F.binary_cross_entropy_with_logits(logits, target_bone, reduction='none')
        bone_area = bone_pred_t.sum().clamp(min=1.0)
        bce_loss = (bce_full * bone_pred_t).sum() / bone_area

        main = self.ftl_weight * ftl_loss + self.bce_weight * bce_loss


        bnd_raw = 0.0
        if bnd_w > 0.0:
            bnd_raw = self.bnd(logits, target_bone)
            total = main + bnd_w * bnd_raw
        else:
            total = main

        return total, main.item() if isinstance(main, torch.Tensor) else main, \
            bnd_raw.item() if isinstance(bnd_raw, torch.Tensor) else bnd_raw

    def forward(self, outputs, tumor_mask, bone_pred, is_tumor, current_epoch=0, ds_epochs=None):
        if isinstance(outputs, dict):
            tumor_logits = outputs['tumor_logits']
        else:
            tumor_logits = outputs

        has_tumor = is_tumor.bool()
        self.last_bnd_w = 0.0
        self.last_bnd_loss = 0.0
        self.last_main_loss = 0.0


        fp_loss = torch.tensor(0.0, device=tumor_logits.device)
        if (~has_tumor).any():
            empty_logits = tumor_logits[~has_tumor]
            empty_bone = (bone_pred[~has_tumor] > 0.5).float()
            prob_in_bone = torch.sigmoid(empty_logits) * empty_bone

            bone_area = empty_bone.sum().clamp(min=1.0)
            fp_loss = (prob_in_bone.pow(2)).sum() / bone_area

        if not has_tumor.any():
            return self.fp_weight_base * fp_loss


        idx = has_tumor.nonzero(as_tuple=True)[0]
        logits_t = tumor_logits[idx]
        target_t = tumor_mask[idx]
        bone_pred_t = bone_pred[idx]

        target_bone = target_t * bone_pred_t
        bnd_w = self.get_boundary_weight(current_epoch)


        main_loss, main_val, bnd_val = self._single_loss(
            logits_t, target_bone, bone_pred_t, bnd_w)

        self.last_bnd_w = bnd_w
        self.last_bnd_loss = bnd_val
        self.last_main_loss = main_val

        ds_loss = torch.tensor(0.0, device=tumor_logits.device)
        if isinstance(outputs, dict) and current_epoch <= self.ds_cutoff:
            for key in ['ds3_logits', 'ds2_logits']:
                if outputs.get(key) is not None:
                    ds_logit = outputs[key][idx]
                    if ds_logit.shape[2:] != target_bone.shape[2:]:
                        ds_logit = F.interpolate(ds_logit, size=target_bone.shape[2:],
                                                 mode='bilinear', align_corners=False)
                    ds_total, _, _ = self._single_loss(
                        ds_logit, target_bone, bone_pred_t, bnd_w=0.0)
                    ds_loss = ds_loss + ds_total

        return main_loss + self.ds_weight * ds_loss + self.fp_weight_base * fp_loss


class SingleStageLoss(nn.Module):

    def __init__(self, base_loss_fn,
                 irgda_sup_weight=0.003,
                 fp_suppress_weight=0.08,
                 tumor_fp_weight_ratio=0.4,
                 irgda_start_epoch=60,
                 fp_start_epoch=60,
                 irgda_rampup_epochs=20):
        super().__init__()
        self.tumor_loss          = base_loss_fn
        self.irgda_sup_weight    = irgda_sup_weight
        self.fp_suppress_weight  = fp_suppress_weight
        self.irgda_start_epoch   = irgda_start_epoch
        self.fp_start_epoch      = fp_start_epoch
        self.irgda_rampup_epochs = irgda_rampup_epochs

        self.last_bnd_w      = 0.0
        self.last_bnd_loss   = 0.0
        self.last_total_loss = 0.0

        from networks.bonmseg.fbfa_intrabone_enhanced_iddmga import DGMASupervisionLoss as IRGDASupervisionLoss
        self.irgda_loss_fn = IRGDASupervisionLoss(heatmap_weight=0.2, coverage_weight=0.2, radius_weight=0.1)

    def _get_irgda_weight(self, epoch):
        if epoch < self.irgda_start_epoch:
            return 0.0
        ramp_progress = min(1.0,
            (epoch - self.irgda_start_epoch) / max(self.irgda_rampup_epochs, 1))
        return self.irgda_sup_weight * ramp_progress

    def _get_fp_weight(self, epoch):
        if epoch < self.fp_start_epoch:
            return 0.0
        return self.fp_suppress_weight

    def _fp_suppress_loss(self, logits, bone_pred):
        prob      = torch.sigmoid(logits)
        bone_mask = (bone_pred > 0.5).float()
        bone_area = bone_mask.sum().clamp(min=1.0)
        energy    = (prob * bone_mask).pow(2).sum() / bone_area
        return energy

    def forward(self, outputs, tumor_mask, bone_pred, is_tumor,
                current_epoch=0, ds_epochs=None, model=None):
        if isinstance(outputs, dict):
            logits = outputs['tumor_logits']
        else:
            logits = outputs

        has_tumor = is_tumor.bool()

        self.last_bnd_w      = 0.0
        self.last_bnd_loss   = 0.0
        self.last_total_loss = 0.0

        fp_w = self._get_fp_weight(current_epoch)
        if not has_tumor.any():
            if fp_w > 0:
                fp_loss = self._fp_suppress_loss(logits, bone_pred)
                total   = fp_loss * fp_w
                self.last_total_loss = total.item()
                return total
            return torch.tensor(0.0, device=logits.device)

        tumor_loss = self.tumor_loss(
            outputs, tumor_mask, bone_pred, is_tumor,
            current_epoch=current_epoch)

        self.last_bnd_w    = self.tumor_loss.last_bnd_w
        self.last_bnd_loss = self.tumor_loss.last_bnd_loss

        irgda_w    = self._get_irgda_weight(current_epoch)
        irgda_loss = torch.tensor(0.0, device=logits.device)
        if model is not None and has_tumor.any() and irgda_w > 0:
            tumor_idx = has_tumor.nonzero(as_tuple=True)[0]
            tm_full   = tumor_mask[tumor_idx]
            bm_full   = bone_pred[tumor_idx]
            for soe_module in model.soe:
                state = soe_module.last_state
                if state is None:
                    continue
                attn_h, attn_w = state['attention_map'].shape[2:]
                if tm_full.shape[2] != attn_h or tm_full.shape[3] != attn_w:
                    tm_ds = F.interpolate(tm_full, size=(attn_h, attn_w), mode='nearest')
                    bm_ds = F.interpolate(bm_full, size=(attn_h, attn_w), mode='nearest')
                else:
                    tm_ds, bm_ds = tm_full, bm_full
                irgda_loss = irgda_loss + self.irgda_loss_fn(state, tm_ds, bm_ds)

        fp_loss = torch.tensor(0.0, device=logits.device)
        if fp_w > 0:
            has_empty = ~has_tumor
            if has_empty.any():
                empty_logits = logits[has_empty]
                empty_bone   = bone_pred[has_empty]
                fp_loss      = self._fp_suppress_loss(empty_logits, empty_bone)

        total = tumor_loss + irgda_w * irgda_loss + fp_w * fp_loss
        self.last_total_loss = total.item()
        return total


class BoneOnlyDetailedMetrics:

    def __init__(self, threshold=0.5, smooth=1.0):
        self.threshold = threshold
        self.smooth    = smooth

    def soft_dice_coeff(self, y_true, y_pred):
        inter = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        return (2.0 * inter + self.smooth) / (union + self.smooth)

    def __call__(self, logits, target, bone_pred, is_tumor):
        has_tumor = is_tumor.bool()
        if not has_tumor.any():
            return dict(num_tumor_slices=0,
                        num_empty_slices=has_tumor.numel(),
                        tumor_dice=0.0,
                        tumor_dice_hard=0.0,
                        tumor_dice_at_03=0.0,
                        tumor_precision=0.0,
                        tumor_recall=0.0,
                        tumor_size_ratio=0.0,
                        empty_fp_rate=0.0)

        logits_t    = logits[has_tumor]
        target_t    = target[has_tumor]
        bone_pred_t = bone_pred[has_tumor]

        pred_prob   = torch.sigmoid(logits_t)
        pred_bone   = pred_prob  * bone_pred_t
        target_bone = target_t  * bone_pred_t

        soft_dice = self.soft_dice_coeff(target_bone, pred_bone).item()


        pred_binary      = (pred_prob > self.threshold).float()
        pred_binary_bone = pred_binary * bone_pred_t
        hard_dice        = self.soft_dice_coeff(target_bone, pred_binary_bone).item()

        tp = (pred_binary_bone * target_bone).sum()
        fp = (pred_binary_bone * (1 - target_bone)).sum()
        fn = ((1 - pred_binary_bone) * target_bone).sum()

        precision  = (tp / (tp + fp + 1e-6)).item()
        recall     = (tp / (tp + fn + 1e-6)).item()
        size_ratio = (pred_binary_bone.sum() / (target_bone.sum() + 1e-6)).item()


        pred_03      = (pred_prob > 0.3).float() * bone_pred_t
        dice_at_03   = self.soft_dice_coeff(target_bone, pred_03).item()

        empty_fp_rate = 0.0
        if (~has_tumor).any():
            logits_empty    = logits[~has_tumor]
            bone_pred_empty = bone_pred[~has_tumor]
            pred_empty      = (torch.sigmoid(logits_empty) > self.threshold).float()
            fp_pixels       = (pred_empty * bone_pred_empty).sum()
            total_bone      = bone_pred_empty.sum()
            empty_fp_rate   = (fp_pixels / (total_bone + 1e-6)).item()


        return dict(
            num_tumor_slices=has_tumor.sum().item(),
            num_empty_slices=(~has_tumor).sum().item(),
            tumor_dice=hard_dice,
            tumor_dice_hard=hard_dice,
            tumor_dice_at_03=dice_at_03,
            tumor_precision=precision,
            tumor_recall=recall,
            tumor_size_ratio=size_ratio,
            empty_fp_rate=empty_fp_rate
        )


BoneOnlyFixedLoss = SmallTumorLoss
