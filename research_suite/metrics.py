import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    brier_score_loss, classification_report)
from common import REGIONS

def region_metrics(pred, target, spacing=(1, 1, 1), surface=True):
    result = {}
    for name, labels in REGIONS.items():
        p = np.isin(pred, labels); t = np.isin(target, labels)
        pn, tn, tp = int(p.sum()), int(t.sum()), int((p&t).sum()); union = pn+tn-tp
        hd = None
        if surface and pn and tn:
            ps = p ^ binary_erosion(p); ts = t ^ binary_erosion(t)
            ds = np.concatenate([distance_transform_edt(~ts, sampling=spacing)[ps],
                                 distance_transform_edt(~ps, sampling=spacing)[ts]])
            hd = float(np.percentile(ds, 95))
        result[name] = {'dice': 2*tp/(pn+tn) if pn+tn else None,
            'iou': tp/union if union else None, 'hd95_mm': hd,
            'hd95_status': 'defined' if hd is not None else ('not_requested' if not surface else 'empty_prediction_or_reference'),
            'tp': tp, 'fp': pn-tp, 'fn': tn-tp, 'predicted_voxels': pn, 'reference_voxels': tn,
            'reference_volume_ml': tn*float(np.prod(spacing))/1000,
            'predicted_volume_ml': pn*float(np.prod(spacing))/1000}
    return result

def interval(values, seed=42, n=2000):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a): return None
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(a, len(a), replace=True).mean()) for _ in range(n)]
    return np.percentile(means, [2.5, 97.5]).tolist()

def summarize(cases):
    result = {}
    for name in REGIONS:
        dice = [c['regions'][name]['dice'] for c in cases]
        valid = [v for v in dice if v is not None]
        hds = [c['regions'][name]['hd95_mm'] for c in cases if c['regions'][name]['hd95_mm'] is not None]
        result[name] = {'mean_dice': float(np.mean(valid)) if valid else None,
            'defined_dice_cases': len(valid), 'total_cases': len(cases),
            'mean_iou': float(np.mean([c['regions'][name]['iou'] for c in cases if c['regions'][name]['iou'] is not None])) if valid else None,
            'dice_below_0_5_defined_cases': sum(v<.5 for v in valid),
            'small_reference_under_1ml_cases': sum(0<c['regions'][name]['reference_volume_ml']<1 for c in cases),
            'conditional_case_bootstrap_95ci': interval(dice),
            'mean_hd95_mm_defined_only': float(np.mean(hds)) if hds else None,
            'defined_hd95_cases': len(hds),
            'reference_empty_false_positive_cases': sum(c['regions'][name]['reference_voxels']==0 and c['regions'][name]['fp']>0 for c in cases)}
    return result

def classification_metrics(y, scores, threshold=.5):
    y = np.asarray(y, dtype=int); scores = np.asarray(scores, dtype=float); pred = (scores >= threshold).astype(int)
    if not len(y) or not np.isfinite(scores).all(): raise ValueError('Empty/non-finite classification results')
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    bins = []; ece = 0.

    assignments = np.minimum((scores*10).astype(int), 9)
    for i in range(10):
        mask = assignments == i
        if mask.any():
            probability, frequency = float(scores[mask].mean()), float(y[mask].mean())
            ece += mask.mean()*abs(probability-frequency)
            bins.append({'bin':i, 'n':int(mask.sum()), 'probability':probability, 'observed_positive_rate':frequency})
    return {'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred, zero_division=0)),
        'specificity': float(tn/(tn+fp)) if tn+fp else None, 'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, scores)) if len(set(y))==2 else None,
        'pr_auc_average_precision': float(average_precision_score(y, scores)) if len(set(y))==2 else None,
        'brier': float(brier_score_loss(y, scores)), 'positive_class_ece_10bins': float(ece), 'reliability_bins': bins,
        'confusion_matrix': cm.tolist(), 'threshold': threshold,
        'classification_report': classification_report(y, pred, labels=[0,1], target_names=['healthy','tumor'], zero_division=0)}

def risk_coverage(errors, uncertainty):
    e = np.asarray(errors, float); u = np.asarray(uncertainty, float)
    if not len(e) or not np.isfinite(e).all() or not np.isfinite(u).all(): return {'curve': [], 'spearman': None}
    order = np.argsort(u, kind='stable'); curve = []

    for value in np.unique(u[order]):
        keep = u <= value
        curve.append({'coverage': float(keep.mean()), 'risk':float(e[keep].mean()), 'n':int(keep.sum())})
    rho = spearmanr(e, u).statistic if len(np.unique(e))>1 and len(np.unique(u))>1 else None
    return {'curve':curve, 'spearman':float(rho) if rho is not None and np.isfinite(rho) else None,
            'note':'Descriptive ranking, not a validated rejection threshold; ties retained together.'}
