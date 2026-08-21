# test_unet3d_4class.py
"""
Test UNet3D 4-class into  test set SKM-TEA.
Post-processing: CC only Patellar + Femoral
Output: CSV per subject + summary JSON
"""

import os, json, math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, label as cc_label
import pandas as pd

import sys
sys.path.insert(0, '/data1/nitis/kneeproject')
from train_unet3d_4class import (
    UNet3D, SKMTEADataset, sliding_window_inference,
    compute_dsc, compute_vs, compute_hd95_bbox,
    clip_and_normalize, seg_to_label_4class,
    CHANNELS, DROPOUT, IN_CHANNELS, NUM_CLASSES,
    PATCH_SIZE, OVERLAP, LABEL_NAMES,
)

# ── Config 
# Update these paths according to your setup
NPY_DIR   = '/data1/nitis/kneeproject/data/qdess_npy_1mm'
ANNOT_DIR = '/data1/nitis/kneeproject/data/qdess/v1-release/annotations/v1.0.0'
BEST_PATH = '/data1/nitis/kneeproject/checkpoints/unet3d_4class_best.pt'
OUT_DIR   = '/data1/nitis/kneeproject/results'
CORRUPT   = {'MTR_172.h5'}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CC hanya Patellar + Femoral
CC_CLASSES = {1, 2}

os.makedirs(OUT_DIR, exist_ok=True)

def keep_largest_component(pred):
    result = pred.copy()
    for c in range(1, NUM_CLASSES + 1):
        if c not in CC_CLASSES:
            continue
        mask = (pred == c)
        if not mask.any(): continue
        labeled, n = cc_label(mask)
        if n == 0: continue
        sizes   = [(labeled == i).sum() for i in range(1, n+1)]
        largest = np.argmax(sizes) + 1
        result[pred == c] = 0
        result[labeled == largest] = c
    return result

def main():
    print(f'Device : {DEVICE}')
    print(f'Loading UNet3D 4-class from {BEST_PATH}...')

    model = UNet3D(IN_CHANNELS, NUM_CLASSES, CHANNELS, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f'Parameters : {total:,}')
    print(f'Classes    : {LABEL_NAMES}')
    print(f'CC classes : Patellar + Femoral only')

    # ── Validation
    print('\nRunning validation...')
    with open(f'{ANNOT_DIR}/val.json') as f:
        ann = json.load(f)
    val_files = [i['file_name'] for i in ann['images'] if i['file_name'] not in CORRUPT]

    val_dsc, val_hd95, val_vs = [], [], []
    for fname in tqdm(val_files, desc='  Val'):
        stem  = fname.replace('.h5', '')
        e1    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
        e2    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
        seg   = np.load(f'{NPY_DIR}/{stem}_seg.npy')
        image = np.stack([e1, e2], axis=0)
        label = seg_to_label_4class(seg)
        label_t = torch.from_numpy(label).long()

        img_t = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = sliding_window_inference(model, img_t, PATCH_SIZE, NUM_CLASSES, OVERLAP, DEVICE)
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
        pred = keep_largest_component(pred)
        del out, img_t; torch.cuda.empty_cache()

        pt = torch.from_numpy(pred).long()
        po = F.one_hot(pt,      NUM_CLASSES+1).permute(3,0,1,2)
        lo = F.one_hot(label_t, NUM_CLASSES+1).permute(3,0,1,2)
        val_dsc.append(compute_dsc(po, lo, NUM_CLASSES))
        val_hd95.append(compute_hd95_bbox(po, lo, NUM_CLASSES))
        val_vs.append(compute_vs(po, lo, NUM_CLASSES))

    val_dsc  = np.array(val_dsc)
    val_hd95 = np.array(val_hd95)
    val_vs   = np.array(val_vs)
    print(f'\nVAL SET — UNet3D 4-class (+ CC Pat+Fem)')
    print(f'{"Structure":<12} {"DSC":>8} {"HD95":>8} {"VS":>8}')
    print('-'*42)
    for i, name in enumerate(LABEL_NAMES):
        print(f'{name:<12} {val_dsc[:,i].mean():>8.4f} {val_hd95[:,i].mean():>8.2f} {val_vs[:,i].mean():>8.4f}')
    print('-'*42)
    print(f'{"Average":<12} {val_dsc.mean():>8.4f} {val_hd95.mean():>8.2f} {val_vs.mean():>8.4f}')

    # ── Test
    print('\nRunning test set...')
    with open(f'{ANNOT_DIR}/test.json') as f:
        ann = json.load(f)
    test_files = [i['file_name'] for i in ann['images'] if i['file_name'] not in CORRUPT]

    records = []
    dsc_all, hd95_all, vs_all = [], [], []

    print(f'\n{"Subject":<20} {"DSC":>8} {"HD95":>8} {"VS":>8}')
    print('-'*50)

    for fname in tqdm(test_files, desc='  Test'):
        stem  = fname.replace('.h5', '')
        e1    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
        e2    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
        seg   = np.load(f'{NPY_DIR}/{stem}_seg.npy')
        image = np.stack([e1, e2], axis=0)
        label = seg_to_label_4class(seg)
        label_t = torch.from_numpy(label).long()

        img_t = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = sliding_window_inference(model, img_t, PATCH_SIZE, NUM_CLASSES, OVERLAP, DEVICE)
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
        pred = keep_largest_component(pred)
        del out, img_t; torch.cuda.empty_cache()

        pt = torch.from_numpy(pred).long()
        po = F.one_hot(pt,      NUM_CLASSES+1).permute(3,0,1,2)
        lo = F.one_hot(label_t, NUM_CLASSES+1).permute(3,0,1,2)
        d  = compute_dsc(po, lo, NUM_CLASSES)
        h  = compute_hd95_bbox(po, lo, NUM_CLASSES)
        v  = compute_vs(po, lo, NUM_CLASSES)

        dsc_all.append(d); hd95_all.append(h); vs_all.append(v)
        print(f'{stem:<20} {d.mean():>8.4f} {h.mean():>8.2f} {v.mean():>8.4f}')

        for i, name in enumerate(LABEL_NAMES):
            records.append({'subject':stem,'structure':name,
                            'dsc':float(d[i]),'hd95':float(h[i]),'vs':float(v[i])})

    dsc_all  = np.array(dsc_all)
    hd95_all = np.array(hd95_all)
    vs_all   = np.array(vs_all)

    print(f'\n{"="*60}')
    print(f'TEST SET — UNet3D 4-class (+ CC Pat+Fem)')
    print(f'{"="*60}')
    print(f'{"Structure":<12} {"DSC mean":>10} {"DSC std":>9} {"HD95":>9} {"VS":>9}')
    print('-'*52)
    for i, name in enumerate(LABEL_NAMES):
        print(f'{name:<12} {dsc_all[:,i].mean():>10.4f} {dsc_all[:,i].std():>9.4f} '
              f'{hd95_all[:,i].mean():>9.2f} {vs_all[:,i].mean():>9.4f}')
    print('-'*52)
    print(f'{"Average":<12} {dsc_all.mean():>10.4f} {dsc_all.mean(1).std():>9.4f} '
          f'{hd95_all.mean():>9.2f} {vs_all.mean():>9.4f}')

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, 'unet3d_4class_test_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nCSV saved → {csv_path}')

    summary = {
        'model'    : 'UNet3D 4-class',
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
    json_path = os.path.join(OUT_DIR, 'unet3d_4class_test_summary.json')
    with open(json_path, 'w') as f: json.dump(summary, f, indent=2)
    print(f'JSON saved → {json_path}')
    print('\nDone!')


if __name__ == '__main__':
    main()
