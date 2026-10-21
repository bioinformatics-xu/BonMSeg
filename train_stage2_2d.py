import os
import sys
import argparse
import datetime
import time
import logging
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset, Subset, Sampler
import gc
from tqdm import tqdm
import io
import cv2

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


from datasets.intrabone_petct_dataset import (
    get_intrabone_dataloader as get_intrabone_dataloader_512,
    HardNegativeDataset,
    EnhancedCTNormalizer,
    EnhancedPETNormalizer,
    get_augmentation,
    parse_patient_slice,
    build_patient_slice_map,
)
from networks.bonmseg.model import (
    FBFAIntraBoneTumorSegmentation,
)


from bone_only_loss_metrics import (
    SmallTumorLoss,
    SingleStageLoss,
    BoneOnlyDetailedMetrics,
)
from comparison_metrics import compute_extended_patient_metrics, mean_metric


def setup_logger(log_file):
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_float_list(v):
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return tuple(float(x.strip()) for x in str(v).split(',') if x.strip())


def freeze_bn_stats(model):
    for m in model.ct_branch.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001, mode='max'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return False
        improved = (metric - self.best_score > self.min_delta) if self.mode == 'max' \
                   else (self.best_score - metric > self.min_delta)
        if improved:
            self.best_score = metric
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class EmptySliceDataset(torch.utils.data.Dataset):

    def __init__(self, image_list, img_root, mode='train', is_16bit=True,
                 n_slices: int = 3,all_ids: list = None):
        self.img_root  = img_root
        self.mode      = mode
        self.is_16bit  = is_16bit
        self.ct_norm   = EnhancedCTNormalizer()
        self.pet_norm  = EnhancedPETNormalizer()
        self.transform = get_augmentation(is_train=(mode == 'train'))

        assert n_slices % 2 == 1, "n_slices must be odd"
        self.n_slices = n_slices
        self.half     = n_slices // 2

        ref_ids = all_ids if all_ids is not None else image_list
        pmap = build_patient_slice_map(ref_ids)
        self._zmap = {}
        for patient, entries in pmap.items():
            for z, iid in entries:
                self._zmap[(patient, z)] = iid

        self.image_list = self._filter_empty(image_list)
        print(f"  [EmptySlice {n_slices}-slice] {len(self.image_list)} empty slices loaded")
        print(f"  Neighbor map: {len(ref_ids)} total slices")

    def _get_path(self, image_id, suffix):
        p = os.path.join(self.img_root, image_id + suffix)
        if os.path.exists(p):
            return p
        parts = image_id.split('_')
        for n in range(len(parts), 0, -1):
            alt = os.path.join(self.img_root, '_'.join(parts[:n]), image_id + suffix)
            if os.path.exists(alt):
                return alt
        return p

    def _get_bone_path(self, image_id):
        return self._get_path(image_id, '_bone_pred.png')

    def _neighbor_id(self, image_id, delta):
        patient, z = parse_patient_slice(image_id)
        if z is None:
            return image_id
        nb = self._zmap.get((patient, z + delta))
        return nb if nb is not None else image_id

    def _filter_empty(self, raw_list):
        result = []
        for image_id in raw_list:
            tmask = self._get_path(image_id, '_mask.png')
            bone  = self._get_bone_path(image_id)
            if not (os.path.exists(tmask) and os.path.exists(bone)):
                continue
            tm = cv2.imread(tmask, cv2.IMREAD_GRAYSCALE)
            bm = cv2.imread(bone,  cv2.IMREAD_GRAYSCALE)
            if tm is None or bm is None:
                continue
            bone_area = (bm > 127).sum()
            if bone_area < 500:
                continue
            tumor_in_bone = ((tm > 127) & (bm > 127)).sum()
            if tumor_in_bone == 0:
                result.append(image_id)
        return result

    def _load_ct_stack(self, image_id):

        slices = []
        for d in range(-self.half, self.half + 1):
            nid = self._neighbor_id(image_id, d)
            try:
                p = self._get_path(nid, '_CT.png')
                if self.is_16bit:
                    raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    s = self.ct_norm.normalize(raw, method='bone')
                else:
                    raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    s = raw.astype(np.float32) / 255.0
            except Exception:
                p0  = self._get_path(image_id, '_CT.png')
                raw = cv2.imread(p0, cv2.IMREAD_UNCHANGED if self.is_16bit
                                 else cv2.IMREAD_GRAYSCALE)
                s = self.ct_norm.normalize(raw, 'bone') if self.is_16bit \
                    else raw.astype(np.float32) / 255.0
            slices.append(s)
        return np.stack(slices, axis=-1)

    def _load_pet_stack(self, image_id):

        slices = []
        for d in range(-self.half, self.half + 1):
            nid = self._neighbor_id(image_id, d)
            try:
                p   = self._get_path(nid, '_PET.png')
                raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                s   = self.pet_norm.normalize(raw)
            except Exception:
                p0  = self._get_path(image_id, '_PET.png')
                raw = cv2.imread(p0, cv2.IMREAD_UNCHANGED)
                s   = self.pet_norm.normalize(raw)
            slices.append(s)
        return np.stack(slices, axis=-1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id   = self.image_list[idx]
        ct_stack      = self._load_ct_stack(image_id)
        pet_stack     = self._load_pet_stack(image_id)
        bone_path  = self._get_bone_path(image_id)
        bone_raw   = cv2.imread(bone_path, cv2.IMREAD_GRAYSCALE)
        bone_pred  = (bone_raw  > 127).astype(np.float32) \
                     if bone_raw  is not None else np.zeros((512, 512), np.float32)
        tumor_mask = np.zeros_like(bone_pred)

        aug        = self.transform(image=ct_stack, pet_stack=pet_stack,
                                    bone_pred=bone_pred, tumor_mask=tumor_mask)
        ct         = aug['image'].float()
        pet        = aug['pet_stack'].float()
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        def e3(t): return t.unsqueeze(0) if t.dim() == 2 else t
        bone_pred  = e3(bone_pred)
        tumor_mask = e3(tumor_mask)

        ct         = torch.clamp(ct,  0, 1)
        pet        = torch.clamp(pet, 0, 1)
        bone_pred  = (bone_pred  > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float() * bone_pred

        return {
            'ct':          ct,
            'pet':         pet,
            'bone_pred':   bone_pred,
            'tumor_mask':  tumor_mask,
            'name':        image_id,
            'tumor_ratio': torch.tensor(0.0,  dtype=torch.float32),
            'is_tumor':    torch.tensor(False, dtype=torch.bool),
        }


class BalancedBatchSampler(Sampler):

    def __init__(self, dataset, n_tumor, batch_size, tumor_fraction=0.5, seed=42):
        super().__init__()
        self.n_total   = len(dataset)
        self.n_tumor   = n_tumor
        self.n_empty   = self.n_total - n_tumor
        self.bs        = batch_size
        self.n_t       = max(1, int(batch_size * tumor_fraction))
        self.n_e       = batch_size - self.n_t
        self.seed      = seed
        self.rng       = random.Random(seed)

        assert self.n_tumor > 0, "No tumor samples found"
        assert self.n_empty > 0, "No empty samples found"

        self.tumor_idxs = list(range(n_tumor))
        self.empty_idxs = list(range(n_tumor, self.n_total))
        self.n_batches  = max(1, self.n_tumor // self.n_t)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        t_pool = self.tumor_idxs.copy()
        e_pool = self.empty_idxs.copy()
        self.rng.shuffle(t_pool)
        self.rng.shuffle(e_pool)

        t_ptr = 0
        e_ptr = 0

        for _ in range(self.n_batches):
            if t_ptr + self.n_t > len(t_pool):
                self.rng.shuffle(t_pool)
                t_ptr = 0
            if e_ptr + self.n_e > len(e_pool):
                self.rng.shuffle(e_pool)
                e_ptr = 0

            batch = (t_pool[t_ptr:t_ptr + self.n_t]
                     + e_pool[e_ptr:e_ptr + self.n_e])
            self.rng.shuffle(batch)
            yield batch

            t_ptr += self.n_t
            e_ptr += self.n_e


def apply_iddmga_lr_warmup(optimizer, epoch, warmup_end_epoch,
                            iddmga_lr_full, iddmga_group_idxs):


    ratio = min(1.0, epoch / max(warmup_end_epoch, 1))
    lr    = iddmga_lr_full * ratio
    for idx in iddmga_group_idxs:
        optimizer.param_groups[idx]['lr'] = lr


def apply_csa_lr_warmup(optimizer, epoch, warmup_epochs,
                         csa_lr_full, csa_group_idxs):
    ratio = min(1.0, epoch / max(warmup_epochs, 1))
    lr    = csa_lr_full * ratio
    for idx in csa_group_idxs:
        optimizer.param_groups[idx]['lr'] = lr


def forward_with_bone_prior(model, ct, pet, teacher_bone, source: str,
                            return_intermediate: bool = True):
    if source == 'teacher':
        outputs = model(ct, pet, teacher_bone, return_intermediate=return_intermediate)
        return outputs, teacher_bone
    if source == 'student':
        outputs = model(ct, pet, None, return_intermediate=return_intermediate)
        if not return_intermediate:
            raise ValueError("student bone prior requires return_intermediate=True")
        bone_prior = outputs.get('bone_mask_dilated', None)
        if bone_prior is None:
            raise KeyError("model outputs do not contain 'bone_mask_dilated'")
        return outputs, bone_prior.detach().float()
    raise ValueError(f"Unknown bone prior source: {source}")


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler,
                epoch, logger,
                accumulation_steps=4,
                clip_grad_norm=1.0,
                bone_prior_source='student'):
    model.train()
    freeze_bn_stats(model)
    metrics_fn = BoneOnlyDetailedMetrics(threshold=0.5)

    total_loss = 0.0
    all_tdice, all_tprec, all_trecall, all_fp = [], [], [], []
    all_bnd_loss  = []
    all_bnd_total = []

    bnd_w_this_epoch = loss_fn.tumor_loss.get_boundary_weight(epoch)

    _p2 = loss_fn.tumor_loss.phase2_start
    _p3 = loss_fn.irgda_start_epoch
    phase = 1 if epoch < _p2 else (2 if epoch < _p3 else 3)
    phase_desc = {
        1: "Phase1[FTL+BCE+FP_suppress]",
        2: f"Phase2[+BndLoss bnd_w={bnd_w_this_epoch:.4f}]",
        3: "Phase3[+IRGDA+FP]",
    }[phase]
    logger.info(f"  Epoch {epoch} → {phase_desc}  "
                f"DS={'ON' if epoch <= loss_fn.tumor_loss.ds_cutoff else 'OFF'}")

    logger.info(f"  [BonePrior] source={bone_prior_source}")

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f'Epoch {epoch:03d} [Train]')

    for batch_idx, batch in enumerate(pbar):
        ct         = batch['ct'].to(device, non_blocking=True)
        pet        = batch['pet'].to(device, non_blocking=True)
        teacher_bone = batch['bone_pred'].to(device, non_blocking=True)
        tumor_mask = batch['tumor_mask'].to(device, non_blocking=True)
        is_tumor   = batch['is_tumor'].to(device, non_blocking=True)

        if batch_idx == 0:
            n_tumor_in_batch = batch['is_tumor'].sum().item()
            logger.info(f"  [Sanity] batch is_tumor: {n_tumor_in_batch}/{len(batch['is_tumor'])}")
            assert n_tumor_in_batch > 0, "BUG: is_tumor 全是 False，主 loss 无法计算！"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs, bone_pred = forward_with_bone_prior(
                model, ct, pet, teacher_bone, bone_prior_source,
                return_intermediate=True)
            loss    = loss_fn(outputs, tumor_mask, bone_pred, is_tumor,
                              current_epoch=epoch, model=model)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            del ct, pet, teacher_bone, bone_pred, tumor_mask, outputs, loss
            continue

        if loss_fn.last_bnd_loss > 0.0:
            all_bnd_loss.append(loss_fn.last_bnd_loss)
        if loss_fn.last_total_loss > 0.0:
            all_bnd_total.append(loss_fn.last_total_loss)

        scaler.scale(loss / accumulation_steps).backward()

        step = (batch_idx + 1) % accumulation_steps == 0
        last = (batch_idx + 1) == len(dataloader)
        if step or last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0:
            with torch.no_grad():
                m = metrics_fn(outputs['tumor_logits'], tumor_mask, bone_pred, is_tumor)
                if m['num_tumor_slices'] > 0:
                    all_tdice.append(m['tumor_dice'])
                    all_tprec.append(m['tumor_precision'])
                    all_trecall.append(m['tumor_recall'])
                if m['num_empty_slices'] > 0:
                    all_fp.append(m['empty_fp_rate'])

        total_loss += loss.item()
        pbar.set_postfix({
            'loss':   f'{loss.item():.4f}',
            'dice':   f'{np.mean(all_tdice):.3f}'   if all_tdice   else '—',
            'prec':   f'{np.mean(all_tprec):.3f}'   if all_tprec   else '—',
            'recall': f'{np.mean(all_trecall):.3f}' if all_trecall else '—',
            'fp':     f'{np.mean(all_fp):.4f}'      if all_fp      else '—',
        })
        del ct, pet, teacher_bone, bone_pred, tumor_mask, outputs, loss

    n = len(dataloader)
    mean_bnd_loss = np.mean(all_bnd_loss)  if all_bnd_loss  else 0.0
    mean_total    = np.mean(all_bnd_total) if all_bnd_total else 1.0
    bnd_ratio     = (bnd_w_this_epoch * mean_bnd_loss) / (mean_total + 1e-8)

    dice_val   = np.mean(all_tdice)   if all_tdice   else 0.0
    prec_val   = np.mean(all_tprec)   if all_tprec   else 0.0
    recall_val = np.mean(all_trecall) if all_trecall else 0.0
    fp_val     = np.mean(all_fp)      if all_fp      else 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch:03d} [TRAIN]  {phase_desc}")
    logger.info(f"  Loss:      {total_loss/n:.4f}")
    logger.info(f"  Dice@0.5:  {dice_val:.4f}")
    logger.info(f"  Precision: {prec_val:.4f}")
    logger.info(f"  Recall:    {recall_val:.4f}")
    logger.info(f"  FP Rate:   {fp_val:.4f}")
    logger.info(f"  [Bnd] weight={bnd_w_this_epoch:.4f}  "
                f"raw={mean_bnd_loss:.4f}  ratio={bnd_ratio:.4f}")
    if prec_val < 0.40 and dice_val > 0:
        logger.warning(f"  ⚠ Precision={prec_val:.4f} < 0.40：过度预测")
    logger.info(f"{'='*70}\n")

    return {
        'loss':            total_loss / n,
        'tumor_dice':      dice_val,
        'tumor_precision': prec_val,
        'tumor_recall':    recall_val,
        'empty_fp_rate':   fp_val,
        'bnd_weight':      bnd_w_this_epoch,
        'bnd_loss':        mean_bnd_loss,
        'bnd_ratio':       bnd_ratio,
    }


def _cc_filter(prob_np: np.ndarray, bone_np: np.ndarray,
               threshold: float, min_cc_pixels: int) -> np.ndarray:
    binary = ((prob_np > threshold) & (bone_np > 0.5)).astype(np.uint8)
    if min_cc_pixels <= 0 or binary.sum() == 0:
        return binary.astype(np.float32)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cc_pixels:
            out[labels == lbl] = 1
    return out.astype(np.float32)


def validate(model, dataloader, loss_fn, device, epoch, logger,
             eval_thresholds=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
             min_cc_pixels: int = 30,
             worst_slice_topk: int = 30,
             bone_prior_source='student'):
    model.eval()
    metrics_fn = BoneOnlyDetailedMetrics(threshold=0.5)

    thresholds = list(eval_thresholds)


    th_tp    = {t: 0.0 for t in thresholds}
    th_pred  = {t: 0.0 for t in thresholds}
    th_gt    = {t: 0.0 for t in thresholds}
    th_tp_cc = {t: 0.0 for t in thresholds}
    th_pred_cc = {t: 0.0 for t in thresholds}

    all_fp   = []


    patient_tp_50   = defaultdict(float)
    patient_pred_50 = defaultdict(float)
    patient_gt_map  = defaultdict(float)


    slice_cache = {}

    total_loss = 0.0
    logger.info(f"  [BonePrior] validation source={bone_prior_source}")

    pbar = tqdm(dataloader, desc=f'Epoch {epoch:03d} [Val]')
    with torch.no_grad():
        for batch in pbar:
            ct         = batch['ct'].to(device, non_blocking=True)
            pet        = batch['pet'].to(device, non_blocking=True)
            teacher_bone = batch['bone_pred'].to(device, non_blocking=True)
            tumor_mask = batch['tumor_mask'].to(device, non_blocking=True)
            is_tumor   = batch['is_tumor'].to(device, non_blocking=True)
            names      = batch['name']

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs, bone_pred = forward_with_bone_prior(
                    model, ct, pet, teacher_bone, bone_prior_source,
                    return_intermediate=True)
                loss    = loss_fn(outputs, tumor_mask, bone_pred, is_tumor,
                                  current_epoch=epoch, model=None)

            logits = outputs['tumor_logits']


            m = metrics_fn(logits, tumor_mask, bone_pred, is_tumor)
            if m['num_empty_slices'] > 0:
                all_fp.append(m['empty_fp_rate'])


            probs_f32 = torch.sigmoid(logits).float()

            for i in range(len(names)):
                if not is_tumor[i]:
                    continue

                prob_i = probs_f32[i].squeeze(0)
                bp_i   = bone_pred[i].squeeze(0)
                gt_i   = (tumor_mask[i].squeeze(0) * bp_i)

                gt_px  = gt_i.sum().item()
                pid    = parse_patient_slice(names[i])[0]


                prob_np = prob_i.cpu().numpy()
                bone_np = bp_i.cpu().numpy()
                gt_np   = gt_i.cpu().numpy()


                slice_cache[names[i]] = (prob_np, bone_np, gt_np)


                pred_50 = ((prob_np > 0.50) & (bone_np > 0.5)).astype(np.float32)
                patient_tp_50[pid]   += (pred_50 * gt_np).sum()
                patient_pred_50[pid] += pred_50.sum()
                patient_gt_map[pid]  += gt_px


                for t in thresholds:
                    pred_t = ((prob_np > t) & (bone_np > 0.5)).astype(np.float32)
                    tp_t   = (pred_t * gt_np).sum()
                    th_tp[t]   += tp_t
                    th_pred[t] += pred_t.sum()
                    th_gt[t]   += gt_px


                    pred_cc = _cc_filter(prob_np, bone_np, t, min_cc_pixels)
                    tp_cc   = (pred_cc * gt_np).sum()
                    th_tp_cc[t]   += tp_cc
                    th_pred_cc[t] += pred_cc.sum()

            total_loss += loss.item()

            d50 = (2*th_tp[0.50]) / (th_pred[0.50] + th_gt[0.50] + 1e-6) \
                  if 0.50 in thresholds else 0.0
            pbar.set_postfix({'Dice@0.50': f'{d50:.3f}', 'fp': f'{np.mean(all_fp):.6f}' if all_fp else '—'})
            del ct, pet, teacher_bone, bone_pred, tumor_mask, outputs, loss, logits, probs_f32


    n   = len(dataloader)
    fp  = np.mean(all_fp) if all_fp else 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch:03d} [VAL]  Loss={total_loss/n:.4f}  FP={fp:.6f}")
    logger.info(f"  {'─'*60}")
    logger.info(f"  {'Thresh':>8}  {'Dice':>8}  {'Prec':>8}  {'Recall':>8}  "
                f"  {'Dice+CC':>8}  {'Prec+CC':>8}  {'Recall+CC':>10}")
    logger.info(f"  {'─'*60}")

    best_dice  = -1.0
    best_t     = 0.50
    best_stats = {}

    for t in thresholds:
        gt_total = th_gt[t]
        if gt_total < 1:
            continue
        dice_t   = (2*th_tp[t])    / (th_pred[t]    + gt_total + 1e-6)
        prec_t   = th_tp[t]        / (th_pred[t]    + 1e-6)
        rec_t    = th_tp[t]        / (gt_total       + 1e-6)
        dice_cc  = (2*th_tp_cc[t]) / (th_pred_cc[t] + gt_total + 1e-6)
        prec_cc  = th_tp_cc[t]     / (th_pred_cc[t] + 1e-6)
        rec_cc   = th_tp_cc[t]     / (gt_total + 1e-6)

        marker = ' ← best' if dice_cc > best_dice else ''
        if dice_cc > best_dice:
            best_dice = dice_cc
            best_t    = t
            best_stats = dict(dice=dice_cc, prec=prec_cc,
                              recall=rec_cc, fp=fp,
                              dice_raw=dice_t, prec_raw=prec_t)

        logger.info(f"  {t:>8.2f}  {dice_t:>8.4f}  {prec_t:>8.4f}  {rec_t:>8.4f}"
                    f"  {dice_cc:>8.4f}  {prec_cc:>8.4f}  {rec_cc:>10.4f}{marker}")

    logger.info(f"  {'─'*60}")
    logger.info(f"  Best threshold = {best_t:.2f}  "
                f"Dice(CC)={best_stats.get('dice',0):.4f}  "
                f"Prec(CC)={best_stats.get('prec',0):.4f}  "
                f"Recall={best_stats.get('recall',0):.4f}")


    patient_tp_best   = defaultdict(float)
    patient_pred_best = defaultdict(float)
    patient_gt_best   = defaultdict(float)
    for name, (prob_np, bone_np, gt_np) in slice_cache.items():
        pid = parse_patient_slice(name)[0]
        pred_cc = _cc_filter(prob_np, bone_np, best_t, min_cc_pixels)
        patient_tp_best[pid]   += (pred_cc * gt_np).sum()
        patient_pred_best[pid] += pred_cc.sum()
        patient_gt_best[pid]   += gt_np.sum()


    def _cancer_type(pid):
        if pid.startswith('Kidney'):   return 'Kidney'
        if pid.startswith('Prostate'): return 'Prostate'
        if pid.startswith('Urethral'): return 'Urethral'
        return 'Other'

    per_patient_dsc = {}
    type_dsc = defaultdict(list)

    logger.info(f"\n  {'─'*50}  [Patient-level DSC @{best_t:.2f}+CC  ← 论文标准]")
    for pid in sorted(patient_tp_best.keys()):
        tp = patient_tp_best[pid]
        pd = patient_pred_best[pid]
        gt = patient_gt_best[pid]
        d  = (2 * tp) / (pd + gt + 1e-6)
        per_patient_dsc[pid] = d
        type_dsc[_cancer_type(pid)].append(d)
        flag = '  ⚠' if d < 0.30 else ''
        logger.info(f"  [Patient] {pid:40s}  DSC={d:.4f}  "
                    f"GT={gt:.0f}px  Pred={pd:.0f}px{flag}")


    mean_dsc_overall = float(np.mean(list(per_patient_dsc.values()))) \
                       if per_patient_dsc else 0.0
    logger.info(f"\n  {'─'*50}  [Mean Patient DSC — 论文汇报数字]")
    for ctype in ['Kidney', 'Prostate', 'Urethral', 'Other']:
        vals = type_dsc[ctype]
        if vals:
            logger.info(f"  {ctype:12s}: n={len(vals):2d}  "
                        f"Mean DSC = {np.mean(vals):.4f}  "
                        f"(min={min(vals):.4f}  max={max(vals):.4f})")
    logger.info(f"  {'─'*30}")
    logger.info(f"  Overall    : n={len(per_patient_dsc):2d}  "
                f"Mean DSC = {mean_dsc_overall:.4f}  ← 论文 Table 中报这个数")


    logger.info(f"\n  {'─'*50}  [Patient-level @0.50 no-CC  (参考对比)]")
    extended_patient_metrics = compute_extended_patient_metrics(
        slice_cache,
        lambda prob_np, bone_np: _cc_filter(prob_np, bone_np, best_t, min_cc_pixels),
        lambda name: parse_patient_slice(name)[0],
    )
    mean_iou_overall = mean_metric(extended_patient_metrics, "iou")
    mean_hd95_overall = mean_metric(extended_patient_metrics, "hd95")
    mean_assd_overall = mean_metric(extended_patient_metrics, "assd")
    mean_recall_patient = mean_metric(extended_patient_metrics, "recall")
    logger.info(
        f"  Extended metrics @{best_t:.2f}+CC{min_cc_pixels}: "
        f"IoU={mean_iou_overall:.4f}  Recall(patient)={mean_recall_patient:.4f}  "
        f"HD95={mean_hd95_overall:.4f}  ASSD={mean_assd_overall:.4f}"
    )

    for pid in sorted(patient_tp_50.keys()):
        tp, pd, gt = patient_tp_50[pid], patient_pred_50[pid], patient_gt_map[pid]
        d = (2 * tp) / (pd + gt + 1e-6)
        flag = '  ⚠' if d < 0.30 else ''
        logger.info(f"  [Patient] {pid:40s}  DSC={d:.4f}  "
                    f"GT={gt:.0f}px  Pred={pd:.0f}px{flag}")


    logger.info(f"\n  {'─'*50}  [Worst Slices @{best_t:.2f}+CC]")
    logger.info(f"  标签说明: MISS=漏检  MISLOC=空间错位(TP=0)  FP↑=过度预测  FN↑=漏检过多  LOW=一般差")

    slice_records = []
    for name, (prob_np, bone_np, gt_np) in slice_cache.items():
        pred_cc = _cc_filter(prob_np, bone_np, best_t, min_cc_pixels)
        tp_i    = (pred_cc * gt_np).sum()
        pred_i  = pred_cc.sum()
        gt_i    = gt_np.sum()

        sd = (2 * tp_i) / (pred_i + gt_i + 1e-6)
        sp = tp_i / (pred_i + 1e-6)
        sr = tp_i / (gt_i   + 1e-6)

        if   gt_i   == 0:            tag = 'GT=0'
        elif pred_i == 0:            tag = 'MISS'
        elif tp_i   == 0:            tag = 'MISLOC'
        elif sp     <  0.30:         tag = 'FP↑'
        elif sr     <  0.40:         tag = 'FN↑'
        else:                        tag = 'LOW'

        slice_records.append((name, sd, gt_i, pred_i, sp, sr, tag))

    slice_records.sort(key=lambda r: r[1])


    tag_counts = defaultdict(int)
    for _, sd, *_, tag in slice_records:
        if sd < 0.50:
            tag_counts[tag] += 1
    total_bad = sum(tag_counts.values())

    logger.info(f"  共 {len(slice_records)} 张 tumor 切片，Dice<0.50: {total_bad} 张  "
                f"({', '.join(f'{t}:{c}' for t,c in sorted(tag_counts.items()))})")

    shown = 0
    for name, sd, gt_px, pred_px, sp, sr, tag in slice_records:
        if shown >= worst_slice_topk:
            break
        logger.info(f"  [Slice] {name:50s}  Dice={sd:.4f}  "
                    f"Prec={sp:.4f}  Rec={sr:.4f}  "
                    f"GT={gt_px:.0f}px  Pred={pred_px:.0f}px  [{tag}]")
        shown += 1

    logger.info(f"{'='*70}\n")


    return {
        'loss':              total_loss / n,
        'tumor_dice':        mean_dsc_overall,
        'tumor_precision':   best_stats.get('prec',  0.0),
        'tumor_recall':      best_stats.get('recall',0.0),
        'empty_fp_rate':     fp,
        'best_threshold':    best_t,
        'min_cc_pixels':     min_cc_pixels,
        'pixel_dice_global': best_stats.get('dice',  0.0),
        'mean_patient_iou':  mean_iou_overall,
        'mean_patient_hd95': mean_hd95_overall,
        'mean_patient_assd': mean_assd_overall,
        'mean_patient_recall': mean_recall_patient,
        'dice_raw_50':       (2*th_tp.get(0.50,0)) / (th_pred.get(0.50,0) + th_gt.get(0.50,0) + 1e-6),
    }


def main():
    parser = argparse.ArgumentParser(description='BonMSeg: bone-prior-guided PET/CT segmentation')


    parser.add_argument('--data_root',         type=str,      default='data/img1')
    parser.add_argument('--stage1_model_path', type=str,
                        default='checkpoints/stage1/best_model.pth')
    parser.add_argument('--save_dir',          type=str,      default='checkpoints')
    parser.add_argument('--exp_name',          type=str,      default='bonmseg')
    parser.add_argument('--resume_from', type=str, default='None', help='Path to a checkpoint from which training is resumed.')


    parser.add_argument('--batch_size',        type=int,      default=4)
    parser.add_argument('--num_workers',       type=int,      default=8)
    parser.add_argument('--is_16bit',          type=str2bool, default=True)
    parser.add_argument('--train_empty_ratio', type=float, default=0.5, help='Negative-to-positive sample ratio in each training batch.')
    parser.add_argument('--val_empty_ratio',   type=float,    default=1)
    parser.add_argument('--eval_thresholds',   type=parse_float_list,
                        default=(0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
                        help='Comma-separated thresholds for validation, e.g. 0.25,0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--val_min_cc_pixels', type=int,      default=30,
                        help='Connected-component minimum area used during validation threshold selection.')
    parser.add_argument('--best_min_precision', type=float,   default=0.30,
                        help='Minimum validation precision required for saving best checkpoint.')
    parser.add_argument('--best_min_recall',    type=float,   default=0.60,
                        help='Minimum validation recall required for saving best checkpoint.')
    parser.add_argument('--best_recall_weight', type=float,   default=0.05,
                        help='Recall bonus in the best-checkpoint score.')
    parser.add_argument('--best_hd95_weight',   type=float,   default=0.001,
                        help='HD95 penalty in the best-checkpoint score.')

    parser.add_argument('--min_tumor_pixels', type=int, default=100, help='Minimum bone-restricted lesion area for selecting positive slices.')


    parser.add_argument('--freeze_stage1',     type=str2bool, default=False,
                        )
    parser.add_argument('--bone_dilation',     type=int,      default=5)
    parser.add_argument('--stage2_bone_prior_source',
                        type=str,
                        choices=['student', 'teacher'],
                        default='student',
                        help='student: train/evaluate Stage 2 with the internal Stage 1 bone prior; '
                             'teacher: use dataset-provided *_bone_pred.png prior.')
    parser.add_argument('--iddmga_K',          type=int,      default=3)


    parser.add_argument('--n_slices',            type=int,      default=3)
    parser.add_argument('--csa_feat_ch',         type=int,      default=32)
    parser.add_argument('--csa_n_heads',         type=int,      default=4)
    parser.add_argument('--csa_pool_size',       type=int,      default=16)
    parser.add_argument('--csa_use_cross_modal', type=str2bool, default=True)
    parser.add_argument('--csa_warmup_epochs',   type=int,      default=0)


    parser.add_argument('--ct_lr',             type=float,    default=2e-7)
    parser.add_argument('--main_lr',           type=float,    default=4e-6)
    parser.add_argument('--iddmga_lr',         type=float,    default=4e-6)
    parser.add_argument('--lr_min',            type=float,    default=1e-7)
    parser.add_argument('--weight_decay',      type=float,    default=5e-4)
    parser.add_argument('--main_weight_decay', type=float,    default=1e-4)


    parser.add_argument('--iddmga_warmup_epochs', type=int,   default=10)
    parser.add_argument('--iddmga_warmup_start',  type=int,   default=5)


    parser.add_argument('--phase2_start',      type=int,      default=20)
    parser.add_argument('--phase3_start',      type=int,      default=35)
    parser.add_argument('--ds_cutoff',         type=int,      default=40)
    parser.add_argument('--irgda_rampup',      type=int,      default=15)


    parser.add_argument('--boundary_delay_start', type=int,   default=25)
    parser.add_argument('--bnd_weight_max',        type=float, default=0.08)
    parser.add_argument('--bnd_rampup_rate',       type=float, default=0.001)
    parser.add_argument('--use_boundary_delay',    type=str2bool, default=True)


    parser.add_argument('--use_cosine',        type=str2bool, default=False)
    parser.add_argument('--cosine_T0',         type=int,      default=80)
    parser.add_argument('--cosine_Tmult',      type=int,      default=2)
    parser.add_argument('--tumor_fp_weight_ratio', type=float, default=0.4)

    parser.add_argument('--epochs',              type=int,    default=150)
    parser.add_argument('--accumulation_steps',  type=int,    default=4)
    parser.add_argument('--clip_grad_norm',      type=float,  default=1.0)
    parser.add_argument('--seed',                type=int,    default=42)
    parser.add_argument('--early_stop_patience', type=int,    default=30)


    parser.add_argument('--hard_neg_range',          type=int,   default=5)
    parser.add_argument('--hard_neg_max_per_tumor',  type=int,   default=5)


    parser.add_argument('--irgda_sup_weight',   type=float,  default=0.001)
    parser.add_argument('--fp_suppress_weight', type=float,  default=0.005)


    parser.add_argument('--scheduler_patience', type=int,    default=15)
    parser.add_argument('--scheduler_factor',   type=float,  default=0.5)

    args = parser.parse_args()
    set_seed(args.seed)


    ts       = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(os.path.join(save_dir, 'training.log'))
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")

    train_file = os.path.join(args.data_root, 'train_tumor.txt')
    val_file = os.path.join(args.data_root, 'val_tumor.txt')


    with open(train_file) as f:
        all_train_ids = [l.strip() for l in f if l.strip()]
    with open(val_file) as f:
        all_val_ids = [l.strip() for l in f if l.strip()]

    _, train_tumor_ds = get_intrabone_dataloader_512(
        data_root=args.data_root, split_file=train_file, mode='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        min_tumor_pixels=args.min_tumor_pixels, is_16bit=args.is_16bit,
        n_slices=args.n_slices,
        all_slice_ids=all_train_ids)

    _, val_tumor_ds = get_intrabone_dataloader_512(
        data_root=args.data_root, split_file=val_file, mode='val',
        batch_size=args.batch_size, num_workers=args.num_workers,
        min_tumor_pixels=args.min_tumor_pixels, is_16bit=args.is_16bit,
        n_slices=args.n_slices,
        all_slice_ids=all_val_ids)

    train_empty_ds = EmptySliceDataset(all_train_ids, args.data_root,
                                       mode='train', is_16bit=args.is_16bit,
                                       n_slices=args.n_slices,
                                       all_ids=all_train_ids)
    val_empty_ds = EmptySliceDataset(all_val_ids, args.data_root,
                                     mode='val', is_16bit=args.is_16bit,
                                     n_slices=args.n_slices,
                                     all_ids=all_val_ids)


    train_hard_neg = HardNegativeDataset(
        tumor_image_ids = list(train_tumor_ds.image_list),
        all_image_ids   = all_train_ids,
        img_root        = args.data_root,
        neg_range       = args.hard_neg_range,
        mode            = 'train',
        is_16bit        = args.is_16bit,
        max_per_tumor   = args.hard_neg_max_per_tumor,
        n_slices        = args.n_slices,
    )
    n_train_tumor  = len(train_tumor_ds)
    n_hard_neg     = len(train_hard_neg)


    neg_combined = train_hard_neg


    target_empty_ratio = args.train_empty_ratio
    tumor_frac = 1.0 / (1.0 + target_empty_ratio)

    logger.info(
        f"  Hard-negative samples: {len(neg_combined)} slices available. "
        f"Forcing Batch Neg:Tumor ratio = {target_empty_ratio:.2f}:1 "
        f"(tumor_frac={tumor_frac:.2f})\n"
    )

    n_train_neg = len(neg_combined)
    train_combined = ConcatDataset([train_tumor_ds, neg_combined])

    train_sampler = BalancedBatchSampler(
        train_combined,
        n_tumor=n_train_tumor,
        batch_size=args.batch_size,
        tumor_fraction=tumor_frac,
        seed=args.seed)

    train_loader = DataLoader(
        train_combined,
        batch_sampler      = train_sampler,
        num_workers        = args.num_workers,
        pin_memory         = True,
        persistent_workers = (args.num_workers > 0),
        prefetch_factor    = 4 if args.num_workers > 0 else None)

    logger.info(f"Train: {n_train_tumor} tumor + {n_train_neg} neg "
                f"(hard_neg={n_hard_neg}, ratio 1:{args.train_empty_ratio:.1f})  "
                f"batches/epoch={len(train_sampler)}\n")


    val_hard_neg = HardNegativeDataset(
        tumor_image_ids = list(val_tumor_ds.image_list),
        all_image_ids   = all_val_ids,
        img_root        = args.data_root,
        neg_range       = args.hard_neg_range,
        mode            = 'val',
        is_16bit        = args.is_16bit,
        max_per_tumor   = args.hard_neg_max_per_tumor,
        n_slices        = args.n_slices,
    )
    n_val_tumor = len(val_tumor_ds)
    n_val_hard = len(val_hard_neg)
    n_val_target = int(n_val_tumor * args.val_empty_ratio)


    if n_val_hard < n_val_target:

        n_val_sup = min(n_val_target - n_val_hard, len(val_empty_ds))
        val_sup = Subset(val_empty_ds, list(range(n_val_sup)))
        val_neg = ConcatDataset([val_hard_neg, val_sup])
    else:

        val_neg = val_hard_neg
    val_combined = ConcatDataset([val_tumor_ds, val_neg])
    val_loader   = DataLoader(
        val_combined, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None)
    logger.info(f"Val:   {n_val_tumor} tumor + {len(val_neg)} neg "
                f"(hard={n_val_hard}, ratio 1:{len(val_neg)/max(n_val_tumor,1):.2f})\n")


    model = FBFAIntraBoneTumorSegmentation(
        stage1_model_path       = args.stage1_model_path,
        freeze_stage1           = args.freeze_stage1,
        bone_dilation           = args.bone_dilation,
        enable_deep_supervision = True,

        n_slices                = args.n_slices,
        csa_feat_ch             = args.csa_feat_ch,
        csa_n_heads             = args.csa_n_heads,
        csa_pool_size           = args.csa_pool_size,
        csa_use_cross_modal     = args.csa_use_cross_modal,

        dgma_K_max              = args.iddmga_K,
        dgma_nms_threshold      = 0.3,
        dgma_min_spatial_size   = 65,
    ).to(device)


    start_epoch       = 1
    _resume_opt_state = None
    _resume_sch_state = None
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"[Resume] Loading from {args.resume_from}")
        ckpt     = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        sd_saved = ckpt.get('model_state_dict', ckpt)
        sd_model = model.state_dict()
        matched  = {k: v for k, v in sd_saved.items()
                    if k in sd_model and sd_model[k].shape == v.shape}
        sd_model.update(matched)
        model.load_state_dict(sd_model)
        if 'optimizer_state_dict' in ckpt:
            _resume_opt_state = ckpt.get('optimizer_state_dict')
        if 'scheduler_state_dict' in ckpt:
            _resume_sch_state = ckpt.get('scheduler_state_dict')
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        logger.info(f"  Loaded {len(matched)} params | Resuming from epoch {start_epoch}\n")

    freeze_bn_stats(model)


    def _split_wd(named_params_list):
        no_wd, wd = [], []
        for name, param in named_params_list:
            if not param.requires_grad:
                continue
            if (param.ndim <= 1
                    or name.endswith('.bias')
                    or 'bn' in name.lower()
                    or '.norm' in name.lower()
                    or 'batch_norm' in name.lower()):
                no_wd.append(param)
            else:
                wd.append(param)
        return no_wd, wd

    ct_named     = [(n, p) for n, p in model.named_parameters() if 'ct_branch' in n]
    iddmga_named = [(n, p) for n, p in model.named_parameters()
                    if 'soe' in n and 'ct_branch' not in n]

    csa_named    = [(n, p) for n, p in model.named_parameters() if 'csa_fusion' in n]
    main_named   = [(n, p) for n, p in model.named_parameters()
                    if 'ct_branch' not in n
                    and 'soe'       not in n
                    and 'csa_fusion' not in n]

    ct_no_wd,     ct_wd     = _split_wd(ct_named)
    main_no_wd,   main_wd   = _split_wd(main_named)
    iddmga_no_wd, iddmga_wd = _split_wd(iddmga_named)
    csa_no_wd,    csa_wd    = _split_wd(csa_named)

    IDDMGA_GROUP_IDXS = [4, 5]
    CSA_GROUP_IDXS    = [6, 7]

    optimizer = torch.optim.AdamW([

        {'params': ct_no_wd,     'lr': args.ct_lr,     'weight_decay': 0.0,                   'name': 'ct_no_wd'},

        {'params': ct_wd,        'lr': args.ct_lr,     'weight_decay': args.weight_decay,      'name': 'ct_wd'},

        {'params': main_no_wd,   'lr': args.main_lr,   'weight_decay': 0.0,                   'name': 'main_no_wd'},

        {'params': main_wd,      'lr': args.main_lr,   'weight_decay': args.main_weight_decay, 'name': 'main_wd'},

        {'params': iddmga_no_wd, 'lr': args.iddmga_lr, 'weight_decay': 0.0,                   'name': 'iddmga_no_wd'},

        {'params': iddmga_wd,    'lr': args.iddmga_lr, 'weight_decay': args.weight_decay,      'name': 'iddmga_wd'},


        {'params': csa_no_wd,    'lr': args.main_lr,   'weight_decay': 0.0,                   'name': 'csa_no_wd'},

        {'params': csa_wd,       'lr': args.main_lr,   'weight_decay': args.main_weight_decay, 'name': 'csa_wd'},
    ], betas=(0.9, 0.999), eps=1e-8)

    if _resume_opt_state is not None:
        try:
            optimizer.load_state_dict(_resume_opt_state)
            lr_map = {
                'ct_no_wd':     args.ct_lr,
                'ct_wd':        args.ct_lr,
                'main_no_wd':   args.main_lr,
                'main_wd':      args.main_lr,
                'iddmga_no_wd': args.iddmga_lr,
                'iddmga_wd':    args.iddmga_lr,
                'csa_no_wd':    args.main_lr,
                'csa_wd':       args.main_lr,
            }
            for g in optimizer.param_groups:
                name = g.get('name', '')
                if name in lr_map:
                    g['lr'] = lr_map[name]
                    if 'initial_lr' in g:
                        g['initial_lr'] = lr_map[name]
            logger.info("  Optimizer restored; LR overridden")
        except Exception as e:
            logger.warning(f"  Optimizer load failed ({e}), using fresh optimizer")


    base_loss_fn = SmallTumorLoss(
        ftl_weight           = 2.0,
        bce_weight           = 0.5,
        bnd_weight_max       = args.bnd_weight_max,
        bnd_rampup_rate      = args.bnd_rampup_rate,
        alpha                = 0.3,
        beta                 = 0.7,
        gamma                = 1.5,
        boundary_weight      = 5.0,
        phase2_start         = args.phase2_start,
        use_boundary_delay   = args.use_boundary_delay,
        boundary_delay_start = args.boundary_delay_start,
        ds_cutoff            = args.ds_cutoff,
        fp_weight_base       = 0.02)

    loss_fn = SingleStageLoss(
        base_loss_fn,
        irgda_sup_weight    = args.irgda_sup_weight,
        fp_suppress_weight  = args.fp_suppress_weight,
        irgda_start_epoch   = args.phase3_start,
        fp_start_epoch      = args.phase3_start,
        irgda_rampup_epochs = args.irgda_rampup,
    tumor_fp_weight_ratio=args.tumor_fp_weight_ratio)


    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult,
            eta_min=args.lr_min)

        if _resume_sch_state is not None:
            try:
                scheduler.load_state_dict(_resume_sch_state)
                logger.info("  Scheduler state restored")
            except Exception as e:
                logger.warning(f"  Scheduler state load failed ({e}), using fresh scheduler")
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts "
                    f"T0={args.cosine_T0} Tmult={args.cosine_Tmult}\n")
    else:
        scheduler = None

    scaler        = torch.amp.GradScaler('cuda')
    early_stopper = EarlyStopping(patience=args.early_stop_patience, mode='max')

    best_dice  = 0.0
    best_score = -1e9
    best_epoch = 0

    logger.info("=" * 70)
    logger.info("Starting BonMSeg training...")
    logger.info("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()


        warmup_start = args.iddmga_warmup_start
        warmup_end   = warmup_start + args.iddmga_warmup_epochs
        if epoch < warmup_start:
            for idx in IDDMGA_GROUP_IDXS:
                optimizer.param_groups[idx]['lr'] = 0.0
        elif epoch < warmup_end:


            apply_iddmga_lr_warmup(
                optimizer, epoch - warmup_start + 1,
                warmup_end_epoch  = args.iddmga_warmup_epochs,
                iddmga_lr_full    = args.iddmga_lr,
                iddmga_group_idxs = IDDMGA_GROUP_IDXS)


        if epoch <= args.csa_warmup_epochs:
            apply_csa_lr_warmup(
                optimizer, epoch,
                warmup_epochs  = args.csa_warmup_epochs,
                csa_lr_full    = args.main_lr,
                csa_group_idxs = CSA_GROUP_IDXS)


        if epoch == args.phase2_start:
            loss_fn.tumor_loss.alpha = 0.3
            loss_fn.tumor_loss.beta  = 0.7
            loss_fn.tumor_loss.gamma = 1.5
            loss_fn.tumor_loss.ftl.tversky.alpha = 0.3
            loss_fn.tumor_loss.ftl.tversky.beta  = 0.7
            loss_fn.tumor_loss.ftl.gamma         = 1.5
            logger.info(f"[Recall] Phase2 Switch @ epoch {epoch}: "
                        f"alpha=0.3, beta=0.7, gamma=1.5")

        train_m = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            epoch, logger,
            accumulation_steps = args.accumulation_steps,
            clip_grad_norm     = args.clip_grad_norm,
            bone_prior_source  = args.stage2_bone_prior_source)

        val_m = validate(
            model, val_loader, loss_fn, device, epoch, logger,
            eval_thresholds=args.eval_thresholds,
            min_cc_pixels=args.val_min_cc_pixels,
            bone_prior_source=args.stage2_bone_prior_source)

        cur_dice = val_m['tumor_dice']
        if scheduler is not None:
            scheduler.step()


        writer.add_scalar('Train/Loss',       train_m['loss'],          epoch)
        writer.add_scalar('Train/Dice',       train_m['tumor_dice'],    epoch)
        writer.add_scalar('Train/Recall',     train_m['tumor_recall'],  epoch)
        writer.add_scalar('Train/FP',         train_m['empty_fp_rate'], epoch)
        writer.add_scalar('BndLoss/Weight',   train_m['bnd_weight'],    epoch)
        writer.add_scalar('BndLoss/RawVal',   train_m['bnd_loss'],      epoch)
        writer.add_scalar('BndLoss/Ratio',    train_m['bnd_ratio'],     epoch)
        writer.add_scalar('Val/Loss',         val_m['loss'],            epoch)
        writer.add_scalar('Val/Dice',         val_m['tumor_dice'],      epoch)
        writer.add_scalar('Val/Precision',    val_m['tumor_precision'], epoch)
        writer.add_scalar('Val/Recall',       val_m['tumor_recall'],    epoch)
        writer.add_scalar('Val/FP',           val_m['empty_fp_rate'],   epoch)
        writer.add_scalar('Val/IoU',          val_m['mean_patient_iou'], epoch)
        writer.add_scalar('Val/HD95',         val_m['mean_patient_hd95'], epoch)
        writer.add_scalar('Val/ASSD',         val_m['mean_patient_assd'], epoch)
        writer.add_scalar('LR/CT',     optimizer.param_groups[1]['lr'], epoch)
        writer.add_scalar('LR/Main',   optimizer.param_groups[3]['lr'], epoch)
        writer.add_scalar('LR/IDDMGA', optimizer.param_groups[5]['lr'], epoch)
        writer.add_scalar('LR/CSA',    optimizer.param_groups[7]['lr'], epoch)


        cur_prec   = val_m['tumor_precision']
        cur_recall = val_m['tumor_recall']
        cur_hd95   = val_m['mean_patient_hd95']
        cur_score  = (
            cur_dice
            + args.best_recall_weight * cur_recall
            - args.best_hd95_weight * cur_hd95
        )
        prec_ok    = cur_prec   >= args.best_min_precision
        recall_ok  = cur_recall >= args.best_min_recall
        is_best    = (cur_score > best_score) and prec_ok and recall_ok
        if not is_best and cur_score > best_score and epoch <= 20:
            is_best = True

        if is_best:
            best_dice  = cur_dice
            best_score = cur_score
            best_epoch = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'dice':                 best_dice,
                'best_score':           best_score,
                'precision':            val_m['tumor_precision'],
                'recall':               val_m['tumor_recall'],
                'iou':                  val_m['mean_patient_iou'],
                'hd95':                 val_m['mean_patient_hd95'],
                'assd':                 val_m['mean_patient_assd'],
                'best_threshold':        val_m['best_threshold'],
                'min_cc_pixels':         val_m['min_cc_pixels'],
                'fp_rate':              val_m['empty_fp_rate'],
                'args':                 vars(args),
            }, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f"  [Best] Score={best_score:.4f} Dice={best_dice:.4f} "
                        f"Prec={val_m['tumor_precision']:.4f} "
                        f"Recall={val_m['tumor_recall']:.4f} "
                        f"HD95={val_m['mean_patient_hd95']:.4f} "
                        f"FP={val_m['empty_fp_rate']:.6f}")

        logger.info(
            f"  Epoch {epoch} | Time={int(time.time()-t0)}s | "
            f"LR_ct={optimizer.param_groups[1]['lr']:.2e} | "
            f"LR_main={optimizer.param_groups[3]['lr']:.2e} | "
            f"LR_iddmga={optimizer.param_groups[5]['lr']:.2e} | "
            f"LR_csa={optimizer.param_groups[7]['lr']:.2e}\n"
            f"  Best: Score={best_score:.4f} Dice={best_dice:.4f} (epoch {best_epoch})\n")

        if early_stopper(cur_dice):
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(patience={args.early_stop_patience})")
            break

        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Training done. Best Dice={best_dice:.4f} at epoch {best_epoch}")
    writer.close()


if __name__ == '__main__':
    main()
