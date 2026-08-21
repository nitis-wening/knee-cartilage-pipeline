# test_mednca_v5_4class_v2.py
"""
Test + Post-processing NCA v5 4-class — FIXED CC
CC only for Patellar + Femoral (not Tibial/Meniscus
bcs after merged into 2 separated component anatomically)
"""

import os, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, label as cc_label
import pandas as pd

import sys
sys.path.insert(0, '/data1/nitis/kneeproject')
from train_mednca_v5_4class import (
    MedNCA3D, sliding_window_inference,
    clip_and_normalize, seg_to_label_4class, compute_dsc,
    compute_vs, compute_hd95_bbox,
    CHANNEL_N, HIDDEN_SIZE, IN_CHANNELS, NUM_CLASSES,
    PATCH_SIZE, OVERLAP, LABEL_NAMES,
    SCALE_FACTOR, FIRE_RATE_COARSE, FIRE_RATE_FINE,
    STEPS_COARSE, STEPS_FINE,
)

# ── Config
NPY_DIR   = '/data1/nitis/kneeproject/data/qdess_npy_1mm'
ANNOT_DIR = '/data1/nitis/kneeproject/data/qdess/v1-release/annotations/v1.0.0'
BEST_PATH = '/data1/nitis/kneeproject/checkpoints/mednca3d_v5_4class_best.pt'
THRESH_PATH = '/data1/nitis/kneeproject/results/v5_4class_thresholds.json'
OUT_DIR   = '/data1/nitis/kneeproject/results'
CORRUPT   = {'MTR_172.h5'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CC only for 1 structured component (Patellar + Femoral)
# Tibial and Meniscus SKIP 
CC_CLASSES = {1, 2}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Post-processing
def keep_largest_component(pred):
    """CC hanya untuk Patellar (1) dan Femoral (2)."""
    result = pred.copy()
    for c in range(1, NUM_CLASSES + 1):
        if c not in CC_CLASSES:
            continue  # skip Tibial dan Meniscus!
        mask = (pred == c)
        if not mask.any(): continue
        labeled, n = cc_label(mask)
        if n == 0: continue
        sizes   = [(labeled == i).sum() for i in range(1, n+1)]
        largest = np.argmax(sizes) + 1
        result[pred == c] = 0
        result[labeled == largest] = c
    return result

def apply_threshold(probs, thresholds):
    """Apply per-class threshold ke softmax probabilities."""
    pred = torch.zeros(probs.shape[1:], dtype=torch.long)
    for c in range(NUM_CLASSES):
        pred[probs[c+1] > thresholds[c]] = c + 1
    return pred.numpy()

def pseudo_ensemble(model, image, n=5):
    """Run inference n× dan average softmax probabilities."""
    img_t = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)
    accum = None
    for _ in range(n):
        with torch.no_grad():
            out  = sliding_window_inference(model, img_t, PATCH_SIZE, NUM_CLASSES, OVERLAP, DEVICE)
            prob = torch.softmax(out, dim=1)
            accum = prob if accum is None else accum + prob
        torch.cuda.empty_cache()
    return (accum / n).squeeze(0).cpu()

# ── Load thresholds
def load_thresholds():
    if os.path.exists(THRESH_PATH):
        import json as js
        with open(THRESH_PATH) as f:
            d = js.load(f)
        thresholds = [d[name] for name in LABEL_NAMES]
        print(f'Loaded thresholds from {THRESH_PATH}:')
        for name, t in zip(LABEL_NAMES, thresholds):
            print(f'  {name:<12}: {t:.2f}')
        return thresholds
    else:
        print('No threshold file found, using defaults')
        return [0.70, 0.60, 0.45, 0.35]

# ── Validation
def run_validation(model, thresholds):
    print('\nRunning validation (baseline + PP)...')
    with open(f'{ANNOT_DIR}/val.json') as f:
        ann = json.load(f)
    val_files = [i['file_name'] for i in ann['images'] if i['file_name'] not in CORRUPT]

    results = {'baseline':[], 'cc':[], 'threshold':[], 'full_pp':[]}

    for fname in tqdm(val_files, desc='  Val inference'):
        stem    = fname.replace('.h5', '')
        e1      = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
        e2      = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
        seg     = np.load(f'{NPY_DIR}/{stem}_seg.npy')
        image   = np.stack([e1, e2], axis=0)
        label   = seg_to_label_4class(seg)
        label_t = torch.from_numpy(label).long()

        # pseudo-ensemble probs
        probs = pseudo_ensemble(model, image, n=5)

        # 1. baseline — argmax
        pred_base = probs.argmax(dim=0).numpy()

        # 2. + CC (Patellar+Femoral only)
        pred_cc = keep_largest_component(pred_base)

        # 3. + threshold
        pred_thr = apply_threshold(probs, thresholds)

        # 4. + full PP = threshold + CC
        pred_full = keep_largest_component(pred_thr)

        for key, pred in zip(['baseline','cc','threshold','full_pp'],
                             [pred_base, pred_cc, pred_thr, pred_full]):
            pt = torch.from_numpy(pred).long()
            po = F.one_hot(pt,      NUM_CLASSES+1).permute(3,0,1,2)
            lo = F.one_hot(label_t, NUM_CLASSES+1).permute(3,0,1,2)
            d  = compute_dsc(po, lo, NUM_CLASSES)
            h  = compute_hd95_bbox(po, lo, NUM_CLASSES)
            v  = compute_vs(po, lo, NUM_CLASSES)
            results[key].append((d, h, v))

    print(f'\n{"="*65}')
    print(f'VAL SET RESULTS — NCA v5 4-class (fixed CC)')
    print(f'{"="*65}')
    labels_map = {
        'baseline' : 'Baseline (no PP)',
        'cc'       : '+ CC (Pat+Fem only)',
        'threshold': '+ Threshold',
        'full_pp'  : '+ Full PP (thr + CC)',
    }
    for key, label in labels_map.items():
        d  = np.mean([r[0] for r in results[key]], axis=0)
        h  = np.mean([r[1] for r in results[key]], axis=0)
        v  = np.mean([r[2] for r in results[key]], axis=0)
        print(f'\n{label}:')
        print(f'  DSC={d.mean():.4f}  HD95={h.mean():.2f}mm  VS={v.mean():.4f}')
        for i, name in enumerate(LABEL_NAMES):
            print(f'    {name:<12}: DSC={d[i]:.4f}  HD95={h[i]:.2f}mm  VS={v[i]:.4f}')

# ── Test set 
def run_test(model, thresholds):
    print('\n\nRunning test set evaluation...')
    with open(f'{ANNOT_DIR}/test.json') as f:
        ann = json.load(f)
    test_files = [i['file_name'] for i in ann['images'] if i['file_name'] not in CORRUPT]

    records = []
    dsc_all, hd95_all, vs_all = [], [], []

    print(f'\n{"Subject":<20} {"DSC":>8} {"HD95":>8} {"VS":>8}')
    print('-'*50)

    for fname in tqdm(test_files, desc='  Test inference'):
        stem    = fname.replace('.h5', '')
        e1      = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
        e2      = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
        seg     = np.load(f'{NPY_DIR}/{stem}_seg.npy')
        image   = np.stack([e1, e2], axis=0)
        label   = seg_to_label_4class(seg)
        label_t = torch.from_numpy(label).long()

        # full PP: ensemble + threshold + CC
        probs      = pseudo_ensemble(model, image, n=5)
        pred_thr   = apply_threshold(probs, thresholds)
        pred_final = keep_largest_component(pred_thr)

        pt = torch.from_numpy(pred_final).long()
        po = F.one_hot(pt,      NUM_CLASSES+1).permute(3,0,1,2)
        lo = F.one_hot(label_t, NUM_CLASSES+1).permute(3,0,1,2)
        d  = compute_dsc(po, lo, NUM_CLASSES)
        h  = compute_hd95_bbox(po, lo, NUM_CLASSES)
        v  = compute_vs(po, lo, NUM_CLASSES)

        dsc_all.append(d); hd95_all.append(h); vs_all.append(v)
        print(f'{stem:<20} {d.mean():>8.4f} {h.mean():>8.2f} {v.mean():>8.4f}')

        for i, name in enumerate(LABEL_NAMES):
            records.append({
                'subject'  : stem,
                'structure': name,
                'dsc'      : float(d[i]),
                'hd95'     : float(h[i]),
                'vs'       : float(v[i]),
            })

    dsc_all  = np.array(dsc_all)
    hd95_all = np.array(hd95_all)
    vs_all   = np.array(vs_all)

    print(f'\n{"="*65}')
    print(f'TEST SET RESULTS — NCA v5 4-class (+ full PP)')
    print(f'{"="*65}')
    print(f'{"Structure":<12} {"DSC mean":>10} {"DSC std":>9} {"HD95":>9} {"VS":>9}')
    print(f'{"-"*55}')
    for i, name in enumerate(LABEL_NAMES):
        print(f'{name:<12} {dsc_all[:,i].mean():>10.4f} {dsc_all[:,i].std():>9.4f} '
              f'{hd95_all[:,i].mean():>9.2f} {vs_all[:,i].mean():>9.4f}')
    print(f'{"-"*55}')
    print(f'{"Average":<12} {dsc_all.mean():>10.4f} {dsc_all.mean(1).std():>9.4f} '
          f'{hd95_all.mean():>9.2f} {vs_all.mean():>9.4f}')

    # save
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, 'mednca3d_v5_4class_test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nCSV saved → {csv_path}')

    import json as js
    summary = {
        'model'    : 'NCA v5 4-class',
        'n_test'   : len(test_files),
        'mean_dsc' : float(dsc_all.mean()),
        'mean_hd95': float(hd95_all.mean()),
        'mean_vs'  : float(vs_all.mean()),
        'per_structure': {
            name: {
                'dsc_mean' : float(dsc_all[:,i].mean()),
                'dsc_std'  : float(dsc_all[:,i].std()),
                'hd95_mean': float(hd95_all[:,i].mean()),
                'vs_mean'  : float(vs_all[:,i].mean()),
            } for i, name in enumerate(LABEL_NAMES)
        }
    }
    json_path = os.path.join(OUT_DIR, 'mednca3d_v5_4class_test_summary.json')
    with open(json_path, 'w') as f: js.dump(summary, f, indent=2)
    print(f'JSON saved → {json_path}')

# ── Main
if __name__ == '__main__':
    print(f'Device : {DEVICE}')
    print(f'Loading NCA v5 4-class from {BEST_PATH}...')

    model = MedNCA3D(
        CHANNEL_N, HIDDEN_SIZE, IN_CHANNELS, NUM_CLASSES,
        STEPS_COARSE, STEPS_FINE, SCALE_FACTOR,
        FIRE_RATE_COARSE, FIRE_RATE_FINE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f'Parameters : {total:,}')
    print(f'Classes    : {LABEL_NAMES}')
    print(f'CC classes : Patellar + Femoral only (Tibial+Meniscus skip)')

    # load thresholds dari file (sudah di-tune sebelumnya)
    thresholds = load_thresholds()

    # validation
    run_validation(model, thresholds)

    # test set
    run_test(model, thresholds)

    print('\nDone!')
