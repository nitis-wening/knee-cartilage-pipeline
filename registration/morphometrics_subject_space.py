# morphometrics_subject_space.py
"""
Compute cartilage morphology directly in subject space
(without registration to template) to avoid warping distortion.
Morphological metrics per structure:
  - Volume (mm³, cm³)
  - Mean thickness (mm) via Euclidean Distance Transform
  - Surface area (mm²)

Output:
  results/morphometrics/
    {stem}_morphology.json
    morphology_summary.csv
    morphology_stats.json
"""

import os, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    label as cc_label,
)

import sys
sys.path.insert(0, '/data1/nitis/kneeproject')
from train_mednca_v5_4class import (
    MedNCA3D, sliding_window_inference,
    clip_and_normalize,
    CHANNEL_N, HIDDEN_SIZE, IN_CHANNELS, NUM_CLASSES,
    PATCH_SIZE, OVERLAP, LABEL_NAMES,
    SCALE_FACTOR, FIRE_RATE_COARSE, FIRE_RATE_FINE,
    STEPS_COARSE, STEPS_FINE,
)
import torch

# Update these paths according to your setup
NPY_DIR   = '/data1/nitis/kneeproject/data/qdess_npy_1mm'
ANNOT_DIR = '/data1/nitis/kneeproject/data/qdess/v1-release/annotations/v1.0.0'
BEST_PATH = '/data1/nitis/kneeproject/checkpoints/mednca3d_v5_4class_best.pt'
OUT_DIR   = '/data1/nitis/kneeproject/results/morphometrics'
CORRUPT   = {'MTR_172.h5'}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CC_CLASSES = {1, 2}
SPACING    = (1.0, 1.0, 1.0)  # mm (1mm iso)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Model
def keep_largest_component(pred):
    result = pred.copy()
    for c in range(1, NUM_CLASSES + 1):
        if c not in CC_CLASSES: continue
        mask = (pred == c)
        if not mask.any(): continue
        labeled, n = cc_label(mask)
        if n == 0: continue
        sizes = [(labeled == i).sum() for i in range(1, n+1)]
        largest = np.argmax(sizes) + 1
        result[pred == c] = 0
        result[labeled == largest] = c
    return result

def load_model():
    model = MedNCA3D(
        CHANNEL_N, HIDDEN_SIZE, IN_CHANNELS, NUM_CLASSES,
        STEPS_COARSE, STEPS_FINE, SCALE_FACTOR,
        FIRE_RATE_COARSE, FIRE_RATE_FINE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    return model

def predict(model, image):
    img_t = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = sliding_window_inference(model, img_t, PATCH_SIZE, NUM_CLASSES, OVERLAP, DEVICE)
    pred = out.argmax(1).squeeze(0).cpu().numpy()
    del img_t, out; torch.cuda.empty_cache()
    return keep_largest_component(pred)

# ── Morphometrics 
def compute_volume(mask, spacing):
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return float(mask.sum()) * voxel_vol

def compute_thickness(mask, spacing):
    """
    Mean cartilage thickness via EDT.
    thickness ≈ 2 × mean(EDT di dalam mask)
    """
    if not mask.any():
        return 0.0
    dist = distance_transform_edt(mask, sampling=spacing)
    return float(2.0 * dist[mask].mean())

def compute_surface_area(mask, spacing):
    """
    Surface area estimasi dari boundary voxels.
    """
    if not mask.any():
        return 0.0
    eroded   = binary_erosion(mask)
    boundary = mask & ~eroded
    face_area = spacing[0] * spacing[1]
    return float(boundary.sum() * face_area)

def compute_morphology(pred, spacing=SPACING):
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    results   = {}
    for c, name in enumerate(LABEL_NAMES, start=1):
        mask = (pred == c)
        if not mask.any():
            results[name] = {
                'volume_mm3'       : 0.0,
                'volume_cm3'       : 0.0,
                'mean_thickness_mm': 0.0,
                'surface_area_mm2' : 0.0,
                'n_voxels'         : 0,
            }
            continue
        vol  = compute_volume(mask, spacing)
        thick= compute_thickness(mask, spacing)
        area = compute_surface_area(mask, spacing)
        results[name] = {
            'volume_mm3'       : round(vol, 2),
            'volume_cm3'       : round(vol / 1000, 4),
            'mean_thickness_mm': round(thick, 3),
            'surface_area_mm2' : round(area, 2),
            'n_voxels'         : int(mask.sum()),
        }
    return results

# ── Main 
def main():
    print(f'Device: {DEVICE}')
    print('Loading NCA v5 4-class model...')
    model = load_model()

    # load test subjects
    with open(f'{ANNOT_DIR}/test.json') as f:
        ann = json.load(f)
    test_files = [i['file_name'] for i in ann['images']
                  if i['file_name'] not in CORRUPT]
    print(f'Test subjects: {len(test_files)}')
    print(f'Spacing      : {SPACING} mm (1mm iso)\n')

    records = []

    for fname in tqdm(test_files, desc='Computing morphometrics'):
        stem = fname.replace('.h5', '')
        out_path = os.path.join(OUT_DIR, f'{stem}_morphology.json')

        if os.path.exists(out_path):
            with open(out_path) as f:
                morph = json.load(f)
        else:
            # load MRI + predict
            e1    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
            e2    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
            image = np.stack([e1, e2], axis=0)
            pred  = predict(model, image)

            # hitung morfologi di subject space
            morph = compute_morphology(pred, SPACING)
            with open(out_path, 'w') as f:
                json.dump(morph, f, indent=2)

        # tambah ke records
        for name in LABEL_NAMES:
            rec = morph[name].copy()
            rec['subject']   = stem
            rec['structure'] = name
            records.append(rec)

    # save CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, 'morphology_summary.csv')
    df.to_csv(csv_path, index=False)

    # summary stats
    print(f'\n{"="*65}')
    print('MORPHOLOGY SUMMARY — Subject Space (no registration distortion)')
    print(f'n = {df["subject"].nunique()} test subjects @ 1mm iso')
    print(f'{"="*65}')
    print(f'{"Structure":<12} {"Vol(cm³)":>14} {"Thick(mm)":>14} {"Area(mm²)":>14}')
    print('-'*58)

    stats = {}
    for name in LABEL_NAMES:
        sub = df[df['structure'] == name]
        v = sub['volume_cm3']; t = sub['mean_thickness_mm']; a = sub['surface_area_mm2']
        print(f'{name:<12} '
              f'{v.mean():>6.2f}±{v.std():>5.2f}  '
              f'{t.mean():>6.3f}±{t.std():>5.3f}  '
              f'{a.mean():>7.1f}±{a.std():>6.1f}')
        stats[name] = {
            'volume_cm3_mean'       : round(v.mean(), 3),
            'volume_cm3_std'        : round(v.std(), 3),
            'mean_thickness_mm_mean': round(t.mean(), 3),
            'mean_thickness_mm_std' : round(t.std(), 3),
            'surface_area_mm2_mean' : round(a.mean(), 1),
            'surface_area_mm2_std'  : round(a.std(), 1),
            'CoV_volume'            : round(v.std()/v.mean()*100, 1),
            'CoV_thickness'         : round(t.std()/t.mean()*100, 1),
        }

    print(f'\nCoV (Coefficient of Variation = std/mean × 100%):')
    for name in LABEL_NAMES:
        print(f'  {name:<12}: '
              f'Vol CoV={stats[name]["CoV_volume"]:5.1f}%  '
              f'Thick CoV={stats[name]["CoV_thickness"]:5.1f}%')

    # save stats JSON
    json_path = os.path.join(OUT_DIR, 'morphology_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\nSaved:')
    print(f'  {csv_path}')
    print(f'  {json_path}')
    print(f'\nDone!')

if __name__ == '__main__':
    main()
