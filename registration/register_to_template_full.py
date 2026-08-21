# register_to_template_full.py
"""
Register 36 test subjects ke template (dari template_full - 85 subjects).
Hitung morfologi di subject space.

Output:
  results/registration_full/
    {stem}_registered_mri.npy
    {stem}_registered_seg.npy
    {stem}_morphology.json
    morphology_summary.csv
    morphology_stats.json
"""

import os, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import (
    label as cc_label,
    distance_transform_edt,
    binary_erosion,
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

# ── Config ────────────────────────────────────────────────────────────────────
NPY_DIR   = '/data1/nitis/kneeproject/data/qdess_npy_1mm'
ANNOT_DIR = '/data1/nitis/kneeproject/data/qdess/v1-release/annotations/v1.0.0'
BEST_PATH = '/data1/nitis/kneeproject/checkpoints/mednca3d_v5_4class_best.pt'
TMPL_DIR  = '/data1/nitis/kneeproject/data/template_full'  # 85 subjects
OUT_DIR   = '/data1/nitis/kneeproject/results/registration_full'
CORRUPT   = {'MTR_172.h5'}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CC_CLASSES = {1, 2}
SPACING    = (1.0, 1.0, 1.0)

os.makedirs(OUT_DIR, exist_ok=True)


# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Registration ──────────────────────────────────────────────────────────────
def register_to_template_full(subject_mri, template_mri):
    import itk
    moving_itk = itk.image_from_array(subject_mri.astype(np.float32))
    fixed_itk  = itk.image_from_array(template_mri.astype(np.float32))
    moving_itk.SetSpacing([1.0, 1.0, 1.0])
    fixed_itk.SetSpacing([1.0, 1.0, 1.0])

    parameter_object = itk.ParameterObject.New()
    rigid = parameter_object.GetDefaultParameterMap('rigid', 3)
    rigid['MaximumNumberOfIterations'] = ['256']
    rigid['NumberOfSpatialSamples']    = ['2048']
    parameter_object.AddParameterMap(rigid)
    bspline = parameter_object.GetDefaultParameterMap('bspline', 3)
    bspline['MaximumNumberOfIterations']       = ['256']
    bspline['FinalGridSpacingInPhysicalUnits'] = ['8']
    bspline['NumberOfSpatialSamples']          = ['2048']
    parameter_object.AddParameterMap(bspline)

    result_image, result_transform = itk.elastix_registration_method(
        fixed_itk, moving_itk,
        parameter_object=parameter_object,
        log_to_console=False,
    )
    return itk.array_from_image(result_image).astype(np.float32), result_transform


def apply_transform_seg(seg, result_transform):
    import itk
    seg_itk = itk.image_from_array(seg.astype(np.float32))
    seg_itk.SetSpacing([1.0, 1.0, 1.0])
    for i in range(result_transform.GetNumberOfParameterMaps()):
        result_transform.SetParameter(i, 'ResampleInterpolator',
                                      ['FinalNearestNeighborInterpolator'])
    result = itk.transformix_filter(seg_itk, result_transform)
    return itk.array_from_image(result).round().astype(np.int32)


# ── Morphology (subject space) ────────────────────────────────────────────────
def compute_morphology(pred, spacing=SPACING):
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    results   = {}
    for c, name in enumerate(LABEL_NAMES, start=1):
        mask = (pred == c)
        if not mask.any():
            results[name] = {'volume_mm3':0., 'volume_cm3':0.,
                             'mean_thickness_mm':0., 'surface_area_mm2':0.,
                             'n_voxels':0}
            continue
        vol   = float(mask.sum()) * voxel_vol
        dist  = distance_transform_edt(mask, sampling=spacing)
        thick = float(2.0 * dist[mask].mean())
        eroded = binary_erosion(mask)
        area  = float((mask & ~eroded).sum() * spacing[0] * spacing[1])
        results[name] = {
            'volume_mm3'       : round(vol, 2),
            'volume_cm3'       : round(vol / 1000, 4),
            'mean_thickness_mm': round(thick, 3),
            'surface_area_mm2' : round(area, 2),
            'n_voxels'         : int(mask.sum()),
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Device: {DEVICE}')
    print('Loading model + template (85 subjects)...')

    model        = load_model()
    template_mri = np.load(os.path.join(TMPL_DIR, 'template_mri.npy'))[0]
    template_seg = np.load(os.path.join(TMPL_DIR, 'template_seg.npy'))
    print(f'Template MRI : {template_mri.shape}')
    print(f'Template seg : {np.unique(template_seg)}')

    with open(f'{ANNOT_DIR}/test.json') as f:
        ann = json.load(f)
    test_files = [i['file_name'] for i in ann['images']
                  if i['file_name'] not in CORRUPT]
    print(f'Test subjects: {len(test_files)}')

    records = []

    for fname in tqdm(test_files, desc='Registering'):
        stem = fname.replace('.h5', '')

        out_mri  = os.path.join(OUT_DIR, f'{stem}_registered_mri.npy')
        out_seg  = os.path.join(OUT_DIR, f'{stem}_registered_seg.npy')
        out_morph= os.path.join(OUT_DIR, f'{stem}_morphology.json')

        # skip kalau sudah selesai
        if os.path.exists(out_morph):
            print(f'  SKIP {stem}')
            with open(out_morph) as f: morph = json.load(f)
            for name in LABEL_NAMES:
                rec = morph.get(name, {}).copy()
                rec['subject'] = stem; rec['structure'] = name
                records.append(rec)
            continue

        # load MRI + predict
        e1    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
        e2    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
        image = np.stack([e1, e2], axis=0)
        pred  = predict(model, image)

        # register ke template
        try:
            reg_mri, transform = register_to_template_full(e1, template_mri)
            reg_seg = apply_transform_seg(pred, transform)
            np.save(out_mri, reg_mri)
            np.save(out_seg, reg_seg)
        except Exception as ex:
            print(f'\n  Registration failed for {stem}: {ex}')
            reg_mri = e1; reg_seg = pred

        # morfologi di subject space
        morph = compute_morphology(pred, SPACING)
        with open(out_morph, 'w') as f: json.dump(morph, f, indent=2)

        for name in LABEL_NAMES:
            rec = morph.get(name, {}).copy()
            rec['subject'] = stem; rec['structure'] = name
            records.append(rec)

        print(f'\n  {stem}:')
        for name in LABEL_NAMES:
            m = morph[name]
            print(f'    {name:<12}: vol={m["volume_cm3"]:.3f}cm³  '
                  f'thick={m["mean_thickness_mm"]:.2f}mm  '
                  f'area={m["surface_area_mm2"]:.1f}mm²')

    # summary
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, 'morphology_summary.csv')
    df.to_csv(csv_path, index=False)

    # stats
    print(f'\n{"="*65}')
    print(f'MORPHOLOGY SUMMARY — Template from 85 subjects (n=36 test)')
    print(f'{"="*65}')
    print(f'{"Structure":<12} {"Vol(cm³)":>14} {"Thick(mm)":>14} {"Area(mm²)":>14}')
    print('-'*56)

    stats = {}
    for name in LABEL_NAMES:
        sub = df[df['structure'] == name]
        v=sub['volume_cm3']; t=sub['mean_thickness_mm']; a=sub['surface_area_mm2']
        print(f'{name:<12} {v.mean():>6.2f}±{v.std():>5.2f}  '
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

    print(f'\nCoV:')
    for name in LABEL_NAMES:
        print(f'  {name:<12}: Vol={stats[name]["CoV_volume"]:5.1f}%  '
              f'Thick={stats[name]["CoV_thickness"]:5.1f}%')

    with open(os.path.join(OUT_DIR, 'morphology_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\nSaved to: {OUT_DIR}')
    print('Done!')


if __name__ == '__main__':
    main()