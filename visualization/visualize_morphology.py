# visualize_morphology.py
"""
Visualisasi morfologi kartilago dari subject space (tanpa distorsi registrasi).
4 figure:
  1. Boxplot volume + thickness + area
  2. Scatter volume vs thickness per struktur
  3. Heatmap per-subjek
  4. Bar chart mean ± std + referensi literatur
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = '/data1/nitis/kneeproject/results/morphometrics/morphology_summary.csv'
VIZ_DIR  = '/data1/nitis/kneeproject/results/morphometrics/viz'
os.makedirs(VIZ_DIR, exist_ok=True)

LABEL_NAMES = ['Patellar', 'Femoral', 'Tibial', 'Meniscus']
COLORS = {
    'Patellar' : '#E74C3C',
    'Femoral'  : '#3498DB',
    'Tibial'   : '#2ECC71',
    'Meniscus' : '#F39C12',
}
BG     = '#1a1a2e'
FG     = '#eaeaea'
GRID_C = '#2d2d4e'

# Referensi literatur thickness (mm)
LIT_THICK = {
    'Patellar': (2.5, 3.5),
    'Femoral' : (2.0, 2.5),
    'Tibial'  : (2.0, 2.5),
    'Meniscus': (2.0, 3.0),
}


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID_C)
    ax.grid(color=GRID_C, linestyle='--', alpha=0.5)


def main():
    df = pd.read_csv(CSV_PATH)
    n_subj = df['subject'].nunique()
    print(f'Loaded {n_subj} subjects')

    metrics = [
        ('volume_cm3',        'Volume (cm³)',       'Volume per Structure'),
        ('mean_thickness_mm', 'Thickness (mm)',      'Mean Cartilage Thickness'),
        ('surface_area_mm2',  'Surface Area (mm²)',  'Surface Area per Structure'),
    ]

    # ── Figure 1: Boxplot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
    fig.suptitle(f'Cartilage Morphology — SKM-TEA Test Set (Subject Space, n={n_subj})',
                 color=FG, fontsize=13, fontweight='bold')

    for ax, (col, ylabel, title) in zip(axes, metrics):
        data = [df[df['structure']==s][col].values for s in LABEL_NAMES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color=FG),
                        capprops=dict(color=FG),
                        flierprops=dict(marker='o', markersize=5,
                                        markerfacecolor=FG, alpha=0.6))
        for patch, name in zip(bp['boxes'], LABEL_NAMES):
            patch.set_facecolor(COLORS[name]); patch.set_alpha(0.8)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(LABEL_NAMES, rotation=15)
        ax.set_ylabel(ylabel); ax.set_title(title)
        style_ax(ax)

        # tambah referensi thickness
        if col == 'mean_thickness_mm':
            for i, name in enumerate(LABEL_NAMES, 1):
                lo, hi = LIT_THICK[name]
                ax.hlines([lo, hi], i-0.4, i+0.4,
                          colors='yellow', linewidths=1.5,
                          linestyles='dashed', alpha=0.7)

    plt.tight_layout()
    p = os.path.join(VIZ_DIR, '1_boxplot_morphology.png')
    plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(); print(f'Saved: {p}')

    # ── Figure 2: Scatter volume vs thickness ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=BG)
    fig.suptitle('Volume vs Thickness per Structure (Subject Space)',
                 color=FG, fontsize=13, fontweight='bold')

    for ax, name in zip(axes.flat, LABEL_NAMES):
        sub = df[df['structure']==name]
        ax.scatter(sub['volume_cm3'], sub['mean_thickness_mm'],
                   color=COLORS[name], alpha=0.8, s=60,
                   edgecolors='white', linewidth=0.5)
        # trend line
        z = np.polyfit(sub['volume_cm3'], sub['mean_thickness_mm'], 1)
        xr = np.linspace(sub['volume_cm3'].min(), sub['volume_cm3'].max(), 50)
        ax.plot(xr, np.poly1d(z)(xr), '--', color='white', alpha=0.5, linewidth=1)
        # literature range
        lo, hi = LIT_THICK[name]
        ax.axhspan(lo, hi, alpha=0.1, color='yellow', label='Literature range')
        # label outlier (>2std)
        mv = sub['volume_cm3'].mean(); sv = sub['volume_cm3'].std()
        for _, row in sub.iterrows():
            if abs(row['volume_cm3'] - mv) > 2*sv:
                ax.annotate(row['subject'].replace('MTR_',''),
                            (row['volume_cm3'], row['mean_thickness_mm']),
                            textcoords='offset points', xytext=(5,5),
                            color=FG, fontsize=8)
        ax.set_xlabel('Volume (cm³)'); ax.set_ylabel('Thickness (mm)')
        ax.set_title(f'{name} Cartilage', color=COLORS[name])
        ax.legend(fontsize=7, labelcolor=FG,
                  facecolor=GRID_C, edgecolor=GRID_C)
        style_ax(ax)

    plt.tight_layout()
    p = os.path.join(VIZ_DIR, '2_scatter_vol_vs_thick.png')
    plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(); print(f'Saved: {p}')

    # ── Figure 3: Heatmap per subjek ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 11), facecolor=BG)
    fig.suptitle('Per-Subject Morphology Heatmap (Subject Space)',
                 color=FG, fontsize=13, fontweight='bold')

    subjects = sorted(df['subject'].unique())
    cols_hm  = ['volume_cm3', 'mean_thickness_mm', 'surface_area_mm2']
    col_lbls = ['Volume (cm³)', 'Thickness (mm)', 'Surface Area (mm²)']

    for ax, col, col_lbl in zip(axes, cols_hm, col_lbls):
        matrix = np.zeros((len(subjects), len(LABEL_NAMES)))
        for i, subj in enumerate(subjects):
            for j, name in enumerate(LABEL_NAMES):
                val = df[(df['subject']==subj)&(df['structure']==name)][col].values
                matrix[i, j] = val[0] if len(val) > 0 else 0
        matrix_norm = (matrix - matrix.min(0)) / (matrix.max(0) - matrix.min(0) + 1e-8)
        im = ax.imshow(matrix_norm, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(LABEL_NAMES)))
        ax.set_xticklabels([n[:3] for n in LABEL_NAMES], color=FG)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels([s.replace('MTR_','') for s in subjects],
                           color=FG, fontsize=7)
        ax.set_title(col_lbl, color=FG)
        ax.set_facecolor(BG)
        for i in range(len(subjects)):
            for j in range(len(LABEL_NAMES)):
                ax.text(j, i, f'{matrix[i,j]:.1f}',
                        ha='center', va='center', color='black', fontsize=5.5)
        plt.colorbar(im, ax=ax, label='Normalized', shrink=0.8)

    plt.tight_layout()
    p = os.path.join(VIZ_DIR, '3_heatmap_per_subject.png')
    plt.savefig(p, dpi=110, bbox_inches='tight', facecolor=BG)
    plt.close(); print(f'Saved: {p}')

    # ── Figure 4: Bar chart mean ± std + literatur ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=BG)
    fig.suptitle(f'Mean ± Std Morphology — SKM-TEA Test Set (n={n_subj})',
                 color=FG, fontsize=13, fontweight='bold')

    for ax, (col, ylabel, title) in zip(axes, metrics):
        means = [df[df['structure']==s][col].mean() for s in LABEL_NAMES]
        stds  = [df[df['structure']==s][col].std()  for s in LABEL_NAMES]
        x = range(len(LABEL_NAMES))
        bars = ax.bar(x, means, yerr=stds,
                      color=[COLORS[s] for s in LABEL_NAMES],
                      alpha=0.85, capsize=6,
                      error_kw=dict(ecolor=FG, linewidth=1.5))
        # annotate nilai
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.02*max(means),
                    f'{mean:.2f}', ha='center', va='bottom',
                    color=FG, fontsize=9, fontweight='bold')
        # tambah literatur range untuk thickness
        if col == 'mean_thickness_mm':
            for i, name in enumerate(LABEL_NAMES):
                lo, hi = LIT_THICK[name]
                ax.plot([i-0.4, i+0.4], [lo, lo], '--',
                        color='yellow', linewidth=1.5, alpha=0.8)
                ax.plot([i-0.4, i+0.4], [hi, hi], '--',
                        color='yellow', linewidth=1.5, alpha=0.8)
            ax.plot([], [], '--', color='yellow', linewidth=1.5,
                    label='Literature range', alpha=0.8)
            ax.legend(fontsize=8, labelcolor=FG,
                      facecolor=GRID_C, edgecolor=GRID_C)
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_NAMES, rotation=15)
        ax.set_ylabel(ylabel); ax.set_title(title)
        style_ax(ax)

    plt.tight_layout()
    p = os.path.join(VIZ_DIR, '4_bar_mean_std.png')
    plt.savefig(p, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(); print(f'Saved: {p}')

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'MORPHOLOGY SUMMARY — Subject Space (n={n_subj})')
    print(f'{"="*65}')
    print(f'{"Structure":<12} {"Vol(cm³)":>14} {"Thick(mm)":>14} {"Area(mm²)":>14}')
    print('-'*56)
    for name in LABEL_NAMES:
        sub = df[df['structure']==name]
        v=sub['volume_cm3']; t=sub['mean_thickness_mm']; a=sub['surface_area_mm2']
        print(f'{name:<12} {v.mean():>6.2f}±{v.std():>5.2f}  '
              f'{t.mean():>6.3f}±{t.std():>5.3f}  '
              f'{a.mean():>7.1f}±{a.std():>6.1f}')
    print(f'\nAll figures saved to: {VIZ_DIR}')


if __name__ == '__main__':
    main()