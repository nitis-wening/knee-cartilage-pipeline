# register_to_template.py
"""
Register each test subject into template (CartiMorph-inspired).
Pipeline:
  1. Load template MRI + segmentation
  2. Per test subject:
     a. Load MRI + predict segmentation (NCA v5 4-class)
     b. Register MRI subjct → template (rigid + b-spline)
     c. Apply transform to segmentation subject
     d. Calculate morfology in template space : 
        - Volume per structure
        - Mean thickness per structure
        - Surface area per structure
  3. Save each subject + summary

Output:
  results/registration/
    {stem}_registered_mri.npy    ← MRI after register to template
    {stem}_registered_seg.npy    ← segmentation in template space
    {stem}_morphology.json       ← volume, thickness, area
  results/registration/summary.csv
"""

import os, json, glob
import numpy as np
from tqdm import tqdm
from scipy.ndimage import (
    label as cc_label,
    distance_transform_edt,
    binary_erosion,
)
import pandas as pd

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
NPY_DIR    = '/data1/nitis/kneeproject/data/qdess_npy_1mm'
ANNOT_DIR  = '/data1/nitis/kneeproject/data/qdess/v1-release/annotations/v1.0.0'
BEST_PATH  = '/data1/nitis/kneeproject/checkpoints/mednca3d_v5_4class_best.pt'
TMPL_DIR   = '/data1/nitis/kneeproject/data/template'
OUT_DIR    = '/data1/nitis/kneeproject/results/registration'
CORRUPT    = {'MTR_172.h5'}
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CC_CLASSES = {1, 2}
SPACING    = (1.0, 1.0, 1.0)  # mm

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

# ── Registration
def register_subject_to_template(subject_mri, template_mri):
    """
    Register subject MRI ke template MRI.
    Fixed = template, Moving = subject.
    Return: registered subject MRI + transform parameters.
    """
    import itk

    # subject = moving, template = fixed
    moving_itk = itk.image_from_array(subject_mri.astype(np.float32))
    fixed_itk  = itk.image_from_array(template_mri.astype(np.float32))
    moving_itk.SetSpacing(list(SPACING))
    fixed_itk.SetSpacing(list(SPACING))

    parameter_object = itk.ParameterObject.New()

    # rigid
    rigid = parameter_object.GetDefaultParameterMap('rigid', 3)
    rigid['MaximumNumberOfIterations'] = ['256']
    rigid['NumberOfSpatialSamples']    = ['2048']
    parameter_object.AddParameterMap(rigid)

    # b-spline deformable
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

def apply_transform_to_seg(seg, result_transform):
    """Apply transform to segmentation using nearest neighbor."""
    import itk
    seg_itk = itk.image_from_array(seg.astype(np.float32))
    seg_itk.SetSpacing(list(SPACING))
    for i in range(result_transform.GetNumberOfParameterMaps()):
        result_transform.SetParameter(
            i, 'ResampleInterpolator',
            ['FinalNearestNeighborInterpolator'])
    result = itk.transformix_filter(seg_itk, result_transform)
    return itk.array_from_image(result).round().astype(np.int32)

# ── Morphology
def compute_thickness(mask, spacing=(1.0,1.0,1.0)):
    """
    Calculate mean cartilage thickness using distance transform.
    Method: EDT from outside of mask → mean inside of mask.
    Approx from surface-normal thickness.
    spacing in mm.
    """
    if not mask.any():
        return 0.0
    # gap from boundary
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    # EDT outside of mask
    dist_outside = distance_transform_edt(mask, sampling=spacing)
    # thickness = 2 × mean EDT inside of mask
    # (bcs EDT measure distance to the tepi, thickness ≈ 2×)
    thickness = 2.0 * dist_outside[mask].mean()
    return float(thickness)

def compute_surface_area(mask, spacing=(1.0,1.0,1.0)):
    """
    Estimation surface area from mask 3D.
    Calculate voxel in boundary × voxel face area.
    """
    if not mask.any():
        return 0.0
    eroded   = binary_erosion(mask)
    boundary = mask & ~eroded
    # area per face voxel = spacing[0]*spacing[1] (sagittal)
    # approximasi: pakai mean spacing
    face_area = spacing[0] * spacing[1]
    return float(boundary.sum() * face_area)

def compute_morphology(seg_registered, spacing=(1.0,1.0,1.0)):
    """
    Calculate morfology per structure frm segmentation in template space.
    Return dict with volume, thickness, surface_area per class.
    """
    voxel_vol = spacing[0] * spacing[1] * spacing[2]  # mm³
    results   = {}

    for c, name in enumerate(LABEL_NAMES, start=1):
        mask = (seg_registered == c)
        if not mask.any():
            results[name] = {
                'volume_mm3'      : 0.0,
                'volume_cm3'      : 0.0,
                'mean_thickness_mm': 0.0,
                'surface_area_mm2' : 0.0,
            }
            continue

        volume_mm3 = float(mask.sum()) * voxel_vol
        thickness  = compute_thickness(mask, spacing)
        surf_area  = compute_surface_area(mask, spacing)

        results[name] = {
            'volume_mm3'       : round(volume_mm3, 2),
            'volume_cm3'       : round(volume_mm3 / 1000, 4),
            'mean_thickness_mm': round(thickness, 3),
            'surface_area_mm2' : round(surf_area, 2),
        }

    return results

# ── Main 
def main():
    print(f'Device: {DEVICE}')
    print('Loading model + template...')

    model        = load_model()
    template_mri = np.load(os.path.join(TMPL_DIR, 'template_mri.npy'))[0]  # echo1
    template_seg = np.load(os.path.join(TMPL_DIR, 'template_seg.npy'))
    print(f'Template MRI : {template_mri.shape}')
    print(f'Template seg : {np.unique(template_seg)}')

    # load test subjects
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

        if os.path.exists(out_morph):
            print(f'  SKIP {stem}')
            with open(out_morph) as f:
                morph = json.load(f)
            for name in LABEL_NAMES:
                rec = morph.get(name, {})
                rec['subject'] = stem
                rec['structure'] = name
                records.append(rec)
            continue

        # load MRI
        e1    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
        e2    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
        image = np.stack([e1, e2], axis=0)

        # predict segmentation
        pred = predict(model, image)

        # register subject to template
        try:
            reg_mri, transform = register_subject_to_template(e1, template_mri)
            reg_seg = apply_transform_to_seg(pred, transform)
        except Exception as ex:
            print(f'\n  Registration failed for {stem}: {ex}')
            # fallback: use segmentation without register
            reg_mri = e1
            reg_seg = pred

        # save
        np.save(out_mri, reg_mri)
        np.save(out_seg, reg_seg)

        # calculte morfologi
        morph = compute_morphology(reg_seg, SPACING)
        with open(out_morph, 'w') as f:
            json.dump(morph, f, indent=2)

        # add to records
        for name in LABEL_NAMES:
            rec = morph.get(name, {}).copy()
            rec['subject']   = stem
            rec['structure'] = name
            records.append(rec)

        print(f'\n  {stem}:')
        for name in LABEL_NAMES:
            m = morph[name]
            print(f'    {name:<12}: vol={m["volume_cm3"]:.3f}cm³  '
                  f'thick={m["mean_thickness_mm"]:.2f}mm  '
                  f'area={m["surface_area_mm2"]:.1f}mm²')

    # save summary CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT_DIR, 'morphology_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved → {csv_path}')

    # print overall summary
    print(f'\n{"="*65}')
    print('MORPHOLOGY SUMMARY — SKM-TEA Test Set (Template Space)')
    print(f'{"="*65}')
    print(f'{"Structure":<12} {"Vol(cm³)":>10} {"Thick(mm)":>10} {"Area(mm²)":>10}')
    print('-'*46)
    for name in LABEL_NAMES:
        sub = df[df['structure'] == name]
        print(f'{name:<12} '
              f'{sub["volume_cm3"].mean():>10.3f} '
              f'{sub["mean_thickness_mm"].mean():>10.3f} '
              f'{sub["surface_area_mm2"].mean():>10.1f}')

    print(f'\nDone! Results saved to: {OUT_DIR}')

if __name__ == '__main__':
    main()
