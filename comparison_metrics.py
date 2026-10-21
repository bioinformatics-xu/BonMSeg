import math
import re
from collections import defaultdict
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def parse_slice_index(image_id: str) -> int:
    match = re.search(r"_(\d+)$", image_id)
    return int(match.group(1)) if match else 0


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def surface_distance_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if pred.any() != gt.any():
        diag = math.sqrt(sum(float(s - 1) ** 2 for s in pred.shape))
        return diag, diag

    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    dt_pred = distance_transform_edt(~pred_surface)
    dt_gt = distance_transform_edt(~gt_surface)
    distances = np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]]).astype(np.float64)
    if distances.size == 0:
        return 0.0, 0.0
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def compute_extended_patient_metrics(
    slice_cache: Dict[str, tuple],
    pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    patient_id_fn: Callable[[str], str],
) -> Dict[str, dict]:
    grouped = defaultdict(list)
    for name, values in slice_cache.items():
        prob_np, bone_np, gt_bone = values[:3]
        pid = patient_id_fn(name)
        pred = pred_fn(prob_np, bone_np).astype(bool)
        gt = gt_bone.astype(bool)
        grouped[pid].append((parse_slice_index(name), name, pred, gt))

    per_patient = {}
    for pid, items in grouped.items():
        items.sort(key=lambda x: (x[0], x[1]))
        pred_vol = np.stack([x[2] for x in items], axis=0)
        gt_vol = np.stack([x[3] for x in items], axis=0)

        tp = float(np.logical_and(pred_vol, gt_vol).sum())
        pred_sum = float(pred_vol.sum())
        gt_sum = float(gt_vol.sum())
        fp = pred_sum - tp
        fn = gt_sum - tp

        dsc = (2.0 * tp) / (pred_sum + gt_sum + 1e-6)
        precision = tp / (pred_sum + 1e-6)
        recall = tp / (gt_sum + 1e-6)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        hd95, assd = surface_distance_metrics(pred_vol, gt_vol)

        per_patient[pid] = {
            "dsc": float(dsc),
            "f1": float(f1),
            "iou": float(iou),
            "prec": float(precision),
            "recall": float(recall),
            "hd95": float(hd95),
            "assd": float(assd),
            "gt_pixels": gt_sum,
            "pred_pixels": pred_sum,
        }
    return per_patient


def mean_metric(per_patient: Dict[str, dict], key: str) -> float:
    values = [m[key] for m in per_patient.values() if key in m and np.isfinite(m[key])]
    return float(np.mean(values)) if values else 0.0
