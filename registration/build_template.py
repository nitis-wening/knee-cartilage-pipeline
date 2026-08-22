# build_template.py v3
"""
Build template/atlas from SKM-TEA train subjects.
Fix v3:
  1. Majority vote with dilation+erosion (MALF trick)
  2. Tibial dan Meniscus have 2 separated component →
     vote per-component (med/lat) before merge to the individual label
  3. API itk-elastix  (itk.image_from_array)
"""

import os, json, glob
import numpy as np
from tqdm import tqdm
from scipy.ndimage import (
    binary_dilation, binary_erosion,
    label as cc_label, binary_fill_holes,
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
OUT_DIR   = '/data1/nitis/kneeproject/data/template'
CORRUPT   = {'MTR_172.h5'}
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CC_CLASSES = {1, 2}  # CC hanya Patellar + Femoral

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
def register_to_template(moving_np, fixed_np):
    import itk
    moving_itk = itk.image_from_array(moving_np.astype(np.float32))
    fixed_itk  = itk.image_from_array(fixed_np.astype(np.float32))
    moving_itk.SetSpacing([1.0, 1.0, 1.0])
    fixed_itk.SetSpacing([1.0, 1.0, 1.0])

    parameter_object = itk.ParameterObject.New()
    rigid   = parameter_object.GetDefaultParameterMap('rigid', 3)
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

def apply_transform_seg(seg_np, result_transform):
    import itk
    seg_itk = itk.image_from_array(seg_np.astype(np.float32))
    seg_itk.SetSpacing([1.0, 1.0, 1.0])
    for i in range(result_transform.GetNumberOfParameterMaps()):
        result_transform.SetParameter(
            i, 'ResampleInterpolator',
            ['FinalNearestNeighborInterpolator'])
    result_seg = itk.transformix_filter(seg_itk, result_transform)
    return itk.array_from_image(result_seg).round().astype(np.int32)

# ── Label Fusion: Dilation + Vote + Erosion
def split_two_components(mask):
    """
    Split mask into the 2 biggest component.
    For Tibial (med+lat) dan Meniscus (med+lat)
    that have 2 separated component anatomy. 
    """
    labeled, n = cc_label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool), np.zeros_like(mask, dtype=bool)
    sizes = [(labeled == i).sum() for i in range(1, n+1)]
    sorted_idx = np.argsort(sizes)[::-1]  # dari terbesar
    comp1 = (labeled == sorted_idx[0]+1) if n >= 1 else np.zeros_like(mask, dtype=bool)
    comp2 = (labeled == sorted_idx[1]+1) if n >= 2 else np.zeros_like(mask, dtype=bool)
    return comp1, comp2

def keep_two_largest(mask):
    """Take 2 biggest component only, throw a little noise component."""
    labeled, n = cc_label(mask)
    if n == 0: return np.zeros_like(mask, dtype=bool)
    sizes = [(labeled == i).sum() for i in range(1, n+1)]
    sorted_idx = np.argsort(sizes)[::-1]
    result = np.zeros_like(mask, dtype=bool)
    for k in range(min(2, n)):
        result |= (labeled == sorted_idx[k]+1)
    return result

def majority_vote_with_dilation(seg_list, num_classes=NUM_CLASSES,
                                 dilation_radius=3, threshold=0.25):
    """
    Label fusion with dilation before vote.
    Patellar/Femoral: dilation + normal erotion.
    Tibial/Meniscus:
      →  take 2 largest component per subject (throw noise)
      → increase dilation (5) so it can be overlap
      → decrease threshold (0.15)
      → SKIP erosion (bcs component already small after vote)
      → fill holes only
    """
    n = len(seg_list)
    shape = seg_list[0].shape
    template_seg = np.zeros(shape, dtype=np.int32)

    for c in range(1, num_classes + 1):
        if c in [3, 4]:
            # ── Tibial / Meniscus: 2 component, light erosion 
            dil_r     = dilation_radius   # same with Patellar/Femoral (3)
            thr_c     = 0.20              # a lil bit low from 0.25
            ero_r     = 1                 # lightweight erosion (not full dil_r)
            votes = np.zeros(shape, dtype=np.float32)

            for seg in seg_list:
                mask = (seg == c).astype(bool)
                if not mask.any(): continue
                # Take 2 biggest component only (throw noise)
                mask_clean = keep_two_largest(mask)
                if mask_clean.any():
                    votes += binary_dilation(
                        mask_clean, iterations=dil_r).astype(np.float32)

            won = votes > (n * thr_c)
            if won.any():
                # erosion ringan (1 iteration only, not full dil_r)
                won = binary_erosion(won, iterations=ero_r)
                won = binary_fill_holes(won)
                # minimal size filter: hapus blob < 100 voxel
                labeled_w, nw = cc_label(won)
                for k in range(1, nw+1):
                    if (labeled_w == k).sum() < 100:
                        won[labeled_w == k] = False
                template_seg[won & (template_seg == 0)] = c

        else:
            # ── Patellar / Femoral: 1 component, dilation+erosion normal
            votes = np.zeros(shape, dtype=np.float32)
            for seg in seg_list:
                mask = (seg == c).astype(bool)
                if not mask.any(): continue
                votes += binary_dilation(
                    mask, iterations=dilation_radius).astype(np.float32)
            won = votes > (n * threshold)
            if won.any():
                won = binary_erosion(won, iterations=dilation_radius)
                won = binary_fill_holes(won)
                template_seg[won & (template_seg == 0)] = c

    return template_seg

# ── Main 
def main():
    print(f'Device: {DEVICE}')
    print('Loading NCA model...')
    model = load_model()

    with open(f'{ANNOT_DIR}/train.json') as f:
        ann = json.load(f)
    train_files = [i['file_name'] for i in ann['images']
                   if i['file_name'] not in CORRUPT]
    N = 20
    train_files = train_files[:N]
    print(f'Using {N} train subjects')

    # ── Step 1: Load MRI + predict segmentation
    print('\nStep 1: Load MRI + predict segmentations...')
    all_mri  = []
    all_segs = []
    for fname in tqdm(train_files, desc='  Predict'):
        stem  = fname.replace('.h5', '')
        e1    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo1.npy'))
        e2    = clip_and_normalize(np.load(f'{NPY_DIR}/{stem}_echo2.npy'))
        image = np.stack([e1, e2], axis=0)
        pred  = predict(model, image)
        all_mri.append(e1)
        all_segs.append(pred)

    print(f'  MRI shape: {all_mri[0].shape}')
    # debug: check voxel per class before voting
    for c in range(1, 5):
        counts = [(s==c).sum() for s in all_segs]
        print(f'  Class {c} ({LABEL_NAMES[c-1]}): '
              f'min={min(counts)} max={max(counts)} mean={np.mean(counts):.0f}')

    # ── Step 2: Initial template
    print('\nStep 2: Initial template (mean + dilation vote)...')
    template_mri = np.mean(all_mri, axis=0).astype(np.float32)
    template_seg = majority_vote_with_dilation(
        all_segs, dilation_radius=3, threshold=0.25)

    print('  Template seg voxels:')
    for c in range(5):
        print(f'    class {c} ({LABEL_NAMES[c-1] if c > 0 else "BG"}): '
              f'{(template_seg==c).sum()}')

    np.save(os.path.join(OUT_DIR, 'template_mri_init.npy'), template_mri)
    np.save(os.path.join(OUT_DIR, 'template_seg_init.npy'), template_seg)

    # ── Step 3: Iterative refinement
    print('\nStep 3: Iterative refinement (2 iterations)...')
    for iteration in range(2):
        print(f'\n  Iteration {iteration+1}/2:')
        reg_mris = []; reg_segs = []; n_fail = 0

        for i, (mri, seg) in enumerate(tqdm(
                zip(all_mri, all_segs), total=N,
                desc=f'  Register iter {iteration+1}')):
            try:
                reg_mri, transform = register_to_template(mri, template_mri)
                reg_seg = apply_transform_seg(seg, transform)
                reg_mris.append(reg_mri)
                reg_segs.append(reg_seg)
            except Exception as e:
                print(f'\n  Warning: fail subj {i}: {e}')
                reg_mris.append(mri)
                reg_segs.append(seg)
                n_fail += 1

        template_mri = np.mean(reg_mris, axis=0).astype(np.float32)
        template_seg = majority_vote_with_dilation(
            reg_segs, dilation_radius=3, threshold=0.25)

        print(f'  Fails: {n_fail}/{N}')
        print(f'  Template seg voxels:')
        for c in range(5):
            print(f'    class {c}: {(template_seg==c).sum()}')

    # ── Step 4: Save 
    print('\nStep 4: Saving final template...')
    template_mri_2ch = np.stack([template_mri, template_mri], axis=0)
    np.save(os.path.join(OUT_DIR, 'template_mri.npy'), template_mri_2ch)
    np.save(os.path.join(OUT_DIR, 'template_seg.npy'), template_seg)

    meta = {
        'n_subjects'     : N,
        'files_used'     : train_files,
        'shape'          : list(template_mri.shape),
        'spacing_mm'     : [1.0, 1.0, 1.0],
        'iterations'     : 2,
        'dilation_radius': 3,
        'vote_threshold' : 0.25,
        'seg_voxels'     : {str(c): int((template_seg==c).sum()) for c in range(5)},
        'label_names'    : LABEL_NAMES,
        'note': ('Tibial+Meniscus voted per-component (med/lat split) '
                 'before merging to single label'),
    }
    with open(os.path.join(OUT_DIR, 'template_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nDone!')
    print(f'Template MRI : {template_mri_2ch.shape}')
    print(f'Template seg voxels:')
    for c in range(5):
        print(f'  class {c} ({LABEL_NAMES[c-1] if c > 0 else "BG"}): '
              f'{(template_seg==c).sum()}')
    print(f'Saved to: {OUT_DIR}')

if __name__ == '__main__':
    main()
